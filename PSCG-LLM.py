#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10.14 FINAL-CLEAN PU implementation.

Purpose
-------
Final paper version of the complete PSCG-LLM pipeline for seven-class PU
cross-condition bearing diagnosis. The validated diagnostic path is preserved:
multimodal time-frequency encoder -> progressive class-conditional source-domain
alignment -> lightweight classifier -> representation-risk Router -> selective
LoRA-LLM diagnosis -> reliability gate -> frozen final diagnosis ->
policy-constrained class-level maintenance generation.

Clean-up in this version
------------------------
- Preserve the validated training, Router, LLM diagnosis/switching, K002 healthy
  constraint, P3/P4 minimum-action guardrail, and unsupported-observation guardrail.
- Remove legacy/dead helper modules that are not used by the final pipeline.
- Remove all-sample/hard-sample/feature-export reporting that is not needed for
  the final paper result.
- Do not print the standalone lightweight-model accuracy.
- Final console output contains only complete-model diagnosis metrics and
  maintenance-recommendation evaluation metrics.
- Saved final JSON likewise contains only complete-model and maintenance metrics.

Maintenance metrics are automatic predefined-rule compliance indicators rather
than expert-validated engineering accuracy.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import shutil
import time
import zlib
from contextlib import nullcontext
from itertools import combinations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


FAULT_TYPES = {
    "K002": 0,
    "KA03": 1,
    "KA16": 2,
    "KB23": 3,
    "KB24": 4,
    "KI04": 5,
    "KI16": 6,
}
FAULT_NAMES = {
    0: "K002-healthy",
    1: "KA03-artificial outer-race damage",
    2: "KA16-real repetitive outer-race pitting",
    3: "KB23-real compound IR(+OR) pitting",
    4: "KB24-real distributed compound IR(+OR) pitting",
    5: "KI04-real inner-race pitting",
    6: "KI16-real severe inner-race pitting",
}
# PU operating-condition aliases. S1-S4 are the domain labels used by the
# model, logs, reports, and folder layout. The physical settings remain stored
# explicitly so the aliases do not lose their engineering meaning.
#   S1 = N15_M07_F10: 1500 rpm, 0.7 Nm, 1000 N
#   S2 = N09_M07_F10:  900 rpm, 0.7 Nm, 1000 N
#   S3 = N15_M01_F10: 1500 rpm, 0.1 Nm, 1000 N
#   S4 = N15_M07_F04: 1500 rpm, 0.7 Nm,  400 N
# By default, S1/S2/S3 are source domains and S4 is the unseen target domain.
DOMAIN_TO_ID = {
    "S1": 0,
    "S2": 1,
    "S3": 2,
    "S4": 3,
}
ID_TO_DOMAIN = {v: k for k, v in DOMAIN_TO_ID.items()}
DOMAIN_METADATA = {
    "S1": {
        "setting_name": "N15_M07_F10",
        "speed_rpm": 1500,
        "load_torque_nm": 0.7,
        "radial_force_n": 1000,
    },
    "S2": {
        "setting_name": "N09_M07_F10",
        "speed_rpm": 900,
        "load_torque_nm": 0.7,
        "radial_force_n": 1000,
    },
    "S3": {
        "setting_name": "N15_M01_F10",
        "speed_rpm": 1500,
        "load_torque_nm": 0.1,
        "radial_force_n": 1000,
    },
    "S4": {
        "setting_name": "N15_M07_F04",
        "speed_rpm": 1500,
        "load_torque_nm": 0.7,
        "radial_force_n": 400,
    },
}
# Priority confusion relations for the selected seven-bearing PU task.
CONFUSION_PAIRS = (
    (1, 2),  # artificial vs real outer-race damage
    (3, 4),  # two compound IR(+OR) bearings
    (5, 6),  # two inner-race pitting bearings
    (2, 3),  # outer-race vs compound damage
    (2, 4),
    (3, 5),  # compound vs inner-race damage
    (4, 6),
)
ALL_FAULT_PAIRS = tuple(combinations(sorted(FAULT_NAMES), 2))

FAULT_KNOWLEDGE = {
    0: "Healthy PU bearing K002; current and vibration remain close to the healthy source-domain reference.",
    1: "KA03 is an artificially produced outer-race damage created by electric engraving with damage extent level 2.",
    2: "KA16 is a real repetitive outer-race fatigue-pitting fault with random arrangement, extent level 2, and a single-point characteristic.",
    3: "KB23 is a real compound inner-race plus outer-race fatigue-pitting fault with random arrangement, extent level 2, and a single-point characteristic.",
    4: "KB24 is a real compound inner-race plus outer-race fatigue-pitting fault with no repetition, extent level 3, and distributed damage.",
    5: "KI04 is a real inner-race fatigue-pitting fault with no repetition, extent level 1, and a single-point characteristic.",
    6: "KI16 is a real inner-race fatigue-pitting fault with no repetition, extent level 3, and a single-point characteristic.",
}

# Dataset-derived fault attributes exposed to maintenance generation. These
# fields describe the frozen PU class; they do not contain a maintenance action
# or the rule-reference priority used later for evaluation.
FAULT_MAINTENANCE_ATTRIBUTES: Dict[int, Dict[str, object]] = {
    0: {
        "condition_status": "healthy",
        "affected_raceways": [],
        "damage_origin": "none",
        "damage_type": "none",
        "damage_extent_level": 0,
        "damage_pattern": "none",
    },
    1: {
        "condition_status": "fault",
        "affected_raceways": ["outer_race"],
        "damage_origin": "artificial_electric_engraving",
        "damage_type": "localized_outer_race_damage",
        "damage_extent_level": 2,
        "damage_pattern": "artificial_localized",
    },
    2: {
        "condition_status": "fault",
        "affected_raceways": ["outer_race"],
        "damage_origin": "real_fatigue",
        "damage_type": "fatigue_pitting",
        "damage_extent_level": 2,
        "damage_pattern": "repetitive_single_point_random_arrangement",
    },
    3: {
        "condition_status": "fault",
        "affected_raceways": ["inner_race", "outer_race"],
        "damage_origin": "real_fatigue",
        "damage_type": "compound_fatigue_pitting",
        "damage_extent_level": 2,
        "damage_pattern": "compound_single_point_random_arrangement",
    },
    4: {
        "condition_status": "fault",
        "affected_raceways": ["inner_race", "outer_race"],
        "damage_origin": "real_fatigue",
        "damage_type": "compound_fatigue_pitting",
        "damage_extent_level": 3,
        "damage_pattern": "distributed_non_repetitive",
    },
    5: {
        "condition_status": "fault",
        "affected_raceways": ["inner_race"],
        "damage_origin": "real_fatigue",
        "damage_type": "fatigue_pitting",
        "damage_extent_level": 1,
        "damage_pattern": "single_point_non_repetitive",
    },
    6: {
        "condition_status": "fault",
        "affected_raceways": ["inner_race"],
        "damage_origin": "real_fatigue",
        "damage_type": "fatigue_pitting",
        "damage_extent_level": 3,
        "damage_pattern": "severe_single_point_non_repetitive",
    },
}

# Legacy V10.7 lookup content is retained only for backward-compatible loading of
# old result files. V10.12 never calls it for authoritative maintenance output.


# V10.2-PU-7C4D-CWBPROTO: constrained maintenance knowledge used only after the final
# diagnosis is fixed. Report generation cannot modify model predictions.
FAULT_SERVICE_KNOWLEDGE_ZH: Dict[int, Dict[str, object]] = {
    0: {
        "suspected_cause": "电机电流与轴承振动均未显示稳定的故障特征，整体状态接近K002健康参考。",
        "inspection_items": "检查轴承温升、润滑、安装紧固、电机电流趋势及振动趋势。",
        "recommended_action": "保持当前运行方式，继续执行电流—振动联合趋势监测。",
        "verification_plan": "对比两相电流频谱、振动包络谱与K002健康基线。",
        "base_priority": 0,
    },
    1: {
        "suspected_cause": "KA03为电刻方式形成的人工外圈损伤，损伤程度为2级。",
        "inspection_items": "检查外圈局部损伤、轴承座配合、润滑状态及外圈故障频率成分。",
        "recommended_action": "结合人工损伤试验属性进行复核，并用电流侧带和振动包络谱确认外圈特征。",
        "verification_plan": "联合检查BPFO相关振动成分、电机电流调制侧带及外圈表面。",
        "base_priority": 2,
    },
    2: {
        "suspected_cause": "KA16为真实疲劳点蚀外圈故障，表现为2级重复性单点损伤。",
        "inspection_items": "重点检查外圈重复性点蚀区域、轴承座配合、润滑状态及周期性冲击特征。",
        "recommended_action": "安排及时检修并加强状态监测；若振动冲击、温升或电流调制持续增强，应考虑提前更换轴承。",
        "verification_plan": "采用振动包络谱、BPFO特征及电机电流侧带联合确认外圈重复性故障特征。",
        "base_priority": 2,
    },
    3: {
        "suspected_cause": "KB23为真实内圈并伴外圈的复合疲劳点蚀，随机分布、2级、单点特征。",
        "inspection_items": "同时检查内外圈、润滑、轴系对中、轴承座以及复合故障频率成分。",
        "recommended_action": "尽快降载并安排专业检查，确认复合损伤后更换轴承。",
        "verification_plan": "联合分析BPFI/BPFO、振动包络谱、电流调制侧带及拆检结果。",
        "base_priority": 3,
    },
    4: {
        "suspected_cause": "KB24为真实内圈并伴外圈的复合疲劳点蚀，3级分布式损伤。",
        "inspection_items": "全面检查内外圈分布式点蚀、保持架、润滑污染、轴系对中和轴承座。",
        "recommended_action": "优先停机检查并准备更换轴承，排查导致分布式损伤的载荷与润滑因素。",
        "verification_plan": "结合宽带冲击、BPFI/BPFO、两相电流侧带和拆检结果确认。",
        "base_priority": 4,
    },
    5: {
        "suspected_cause": "KI04为真实内圈疲劳点蚀，1级单点损伤。",
        "inspection_items": "检查内圈局部点蚀、轴配合、对中、润滑和内圈故障频率边带。",
        "recommended_action": "安排计划性检查并持续监测；若内圈冲击增长，应提前更换轴承。",
        "verification_plan": "采用BPFI、阶次/包络分析、电流调制侧带及拆检联合确认。",
        "base_priority": 2,
    },
    6: {
        "suspected_cause": "KI16为真实内圈疲劳点蚀，3级单点损伤，故障程度较高。",
        "inspection_items": "重点检查内圈严重点蚀、轴配合、对中、温升、润滑及强冲击成分。",
        "recommended_action": "尽快降载并安排停机检查，确认后优先更换轴承。",
        "verification_plan": "联合分析强BPFI谐波/边带、振动包络谱、电流侧带与拆检结果。",
        "base_priority": 4,
    },
}

MAINTENANCE_PRIORITY_ZH = {
    0: ("P0-常规监测", "按既定巡检周期持续监测"),
    1: ("P1-计划性维护", "建议在下一维护窗口完成检查"),
    2: ("P2-尽快检查", "建议在24–72小时内复核并安排检查"),
    3: ("P3-紧急维护", "建议尽快降载并在24小时内完成专业检查"),
    4: ("P4-停机优先", "条件允许时立即停机或退出运行并由专业人员确认"),
}


def determine_maintenance_policy_from_attributes(
    label: int,
) -> Tuple[int, str, str]:
    """Return deterministic priority/timeline from the frozen class attributes.

    The safety policy is intentionally deterministic: the LLM still generates the
    maintenance recommendation content, but it cannot freely raise/lower the
    operational priority or invent the response timeline.
    """
    attrs = dict(FAULT_MAINTENANCE_ATTRIBUTES.get(int(label), {}))
    condition_status = str(attrs.get("condition_status", "")).lower()
    affected = {str(v) for v in attrs.get("affected_raceways", []) or []}
    pattern = str(attrs.get("damage_pattern", "")).lower()
    try:
        extent = int(attrs.get("damage_extent_level", 0) or 0)
    except (TypeError, ValueError):
        extent = 0

    if condition_status == "healthy":
        priority = 0
    elif extent == 3 or "severe" in pattern or "distributed" in pattern:
        priority = 4
    elif {"inner_race", "outer_race"}.issubset(affected):
        priority = 3
    elif condition_status == "fault" and extent in (1, 2):
        priority = 2
    else:
        priority = 1

    priority_name, timeline = MAINTENANCE_PRIORITY_ZH[int(priority)]
    return int(priority), str(priority_name), str(timeline)

# Automatic report-evaluation rules. These references are derived from the PU
# dataset fault semantics already encoded above. They are author-defined rules,
# not expert judgments, field validation, or a claim of engineering accuracy.
MAINTENANCE_REQUIRED_FIELDS: Tuple[str, ...] = (
    "maintenance_priority",
    "action_timeline_zh",
    "suspected_cause_zh",
    "inspection_items_zh",
    "recommended_action_zh",
    "verification_plan_zh",
)
RULE_BASED_PRIORITY_REFERENCE: Dict[int, int] = {
    int(label): int(spec["base_priority"])
    for label, spec in FAULT_SERVICE_KNOWLEDGE_ZH.items()
}
OUTER_RACE_LABELS = {1, 2}
COMPOUND_RACE_LABELS = {3, 4}
INNER_RACE_LABELS = {5, 6}
SEVERE_MAINTENANCE_LABELS = {3, 4, 6}
MAINTENANCE_SAFETY_RULE_DEFINITIONS: Tuple[str, ...] = (
    "The draft must not claim to replace human inspection, maintenance manuals, or site safety procedures.",
    "The draft must not assert unknown temperature, defect size, lubricant condition, service history, or an exact failure/RUL deadline.",
    "A healthy diagnosis must not trigger unconditional shutdown or bearing replacement.",
    "A severe or compound diagnosis must include concrete inspection or escalation rather than monitoring alone.",
    "Every generated or rule-reference P3/P4 draft must require professional human confirmation and site safety procedures.",
    "A replacement action must be conditional on inspection or verification.",
)


def _maintenance_field_text(row: Dict[str, object], key: str) -> str:
    aliases = {
        "maintenance_priority": (
            "maintenance_priority", "maintenance_generated_raw_priority"
        ),
        "action_timeline_zh": (
            "action_timeline_zh", "action_timeline",
            "maintenance_generated_raw_action_timeline_zh",
        ),
        "suspected_cause_zh": (
            "suspected_cause_zh", "maintenance_generated_raw_suspected_cause_zh"
        ),
        "inspection_items_zh": (
            "inspection_items_zh", "maintenance_generated_raw_inspection_items_zh"
        ),
        "recommended_action_zh": (
            "recommended_action_zh", "maintenance_generated_raw_recommended_action_zh"
        ),
        "verification_plan_zh": (
            "verification_plan_zh", "maintenance_generated_raw_verification_plan_zh"
        ),
        "human_review_reason_zh": (
            "human_review_reason_zh", "maintenance_generated_raw_human_review_reason_zh"
        ),
        "report_summary_zh": (
            "report_summary_zh", "maintenance_report_summary_zh",
            "maintenance_generated_raw_report_summary_zh",
        ),
    }
    keys = aliases.get(key, (key,))
    for candidate in keys:
        value = " ".join(str(row.get(candidate, "") or "").split())
        if value:
            return value
    return ""


def parse_maintenance_priority(value: object) -> Optional[int]:
    match = re.match(r"^\s*P([0-4])(?=$|[^0-9])", str(value or ""), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


MAINTENANCE_LOW_INFORMATION_VALUES = {
    "无", "未知", "未知原因", "目前未确定原因", "待确认", "待人工确认",
    "待进一步确认", "需要进一步确认", "进行检查", "进行必要的检查",
    "需要进行检查", "现场检查", "人工确认", "进行诊断",
    "不含虚构失效期限的简短时限", "使用可用模态或现场检查的复核方法",
    "人工核验原因", "条件性原因", "具体检查项目", "具体且有条件的措施",
}


def _is_low_information_maintenance_text(value: object) -> bool:
    text = re.sub(r"[\s，。；;、,.!?！？：:]", "", str(value or ""))
    if not text:
        return True
    if text in {re.sub(r"[\s，。；;、,.!?！？：:]", "", item) for item in MAINTENANCE_LOW_INFORMATION_VALUES}:
        return True
    return bool(re.fullmatch(r"(?:待|需要)?(?:人工|专业人员)?(?:进一步)?确认", text))


def assess_class_maintenance_content(
    label: int,
    field_values: Dict[str, str],
    human_review_required: bool,
) -> List[str]:
    """Return policy-compliance issues without changing structural RVR/FCR."""
    issues: List[str] = []
    priority = parse_maintenance_priority(field_values.get("maintenance_priority", ""))
    reference_priority = RULE_BASED_PRIORITY_REFERENCE.get(int(label))
    if priority is not None and reference_priority is not None and priority != reference_priority:
        issues.append("priority_not_predefined_rule_reference")

    for key in (
        "action_timeline_zh", "inspection_items_zh",
        "recommended_action_zh", "verification_plan_zh",
    ):
        if _is_low_information_maintenance_text(field_values.get(key, "")):
            issues.append(f"low_information_{key}")

    cause_text = field_values.get("suspected_cause_zh", "")
    if label == 0:
        if not re.search(r"未见|未发现|未显示|未指示|健康|无[^。；]{0,8}故障", cause_text):
            issues.append("healthy_cause_does_not_state_no_fault_indication")

        # V10.13.9: K002 healthy-only semantic guard. A healthy report must not
        # positively assert a bearing fault, even when wording such as
        # "可能存在点蚀损伤" is syntactically cautious.
        positive_fault_claim = bool(re.search(
            r"(?:可能|疑似|考虑|提示|存在|发现|显示)[^。；]{0,18}(?:点蚀|损伤|裂纹|疲劳|故障|缺陷)",
            cause_text,
        ))
        explicit_no_fault = bool(re.search(
            r"(?:未见|未发现|未显示|未指示|无明确|无稳定)[^。；]{0,18}(?:点蚀|损伤|裂纹|疲劳|故障|异常|指征)",
            cause_text,
        ))
        if positive_fault_claim and not explicit_no_fault:
            issues.append("healthy_positive_fault_claim")

        healthy_monitoring_text = " ".join(
            field_values.get(key, "")
            for key in ("inspection_items_zh", "recommended_action_zh", "verification_plan_zh")
        )
        if not ("电流" in healthy_monitoring_text and "振动" in healthy_monitoring_text):
            issues.append("healthy_report_missing_current_and_vibration_monitoring")

        healthy_action = field_values.get("recommended_action_zh", "")
        # P0 content should stay at routine monitoring / baseline verification.
        # Conditional future escalation is allowed, but immediate repair/replacement
        # language is not appropriate for the frozen K002 healthy diagnosis.
        has_repair_action = bool(re.search(r"更换|修复|停机|降载|拆检|维修", healthy_action))
        future_condition = bool(re.search(
            r"若|如|如果|一旦|后续[^。；]{0,12}(?:发现|出现|确认)|确认[^。；]{0,8}(?:异常|故障)",
            healthy_action,
        ))
        if has_repair_action and not future_condition:
            issues.append("healthy_action_exceeds_routine_monitoring")
    else:
        if not re.search(r"可能|疑似|或与|可能与|考虑|诊断提示", cause_text):
            issues.append("fault_cause_missing_conditional_wording")
        if not re.search(r"损伤|点蚀|凹坑|疲劳|故障|异常", cause_text):
            issues.append("fault_cause_missing_damage_mechanism")

    component_text = " ".join(
        field_values.get(key, "")
        for key in (
            "suspected_cause_zh", "inspection_items_zh",
            "recommended_action_zh", "verification_plan_zh",
        )
    )
    inner_mentioned = bool(re.search(
        r"内圈|内滚道|轴承内环|内外圈|inner[_ -]?race", component_text, flags=re.IGNORECASE
    ))
    outer_mentioned = bool(re.search(
        r"外圈|外滚道|轴承外环|内外圈|outer[_ -]?race", component_text, flags=re.IGNORECASE
    ))
    if label in OUTER_RACE_LABELS and not outer_mentioned:
        issues.append("missing_outer_race_coverage")
    if label in INNER_RACE_LABELS and not inner_mentioned:
        issues.append("missing_inner_race_coverage")
    if label in COMPOUND_RACE_LABELS and not (inner_mentioned and outer_mentioned):
        issues.append("missing_compound_inner_outer_race_coverage")

    action_text = field_values.get("recommended_action_zh", "")
    if label in SEVERE_MAINTENANCE_LABELS and not re.search(
        r"现场检查|专业检查|检修|拆检|停机|降载|更换", action_text
    ):
        issues.append("severe_or_compound_without_concrete_action")
    if label == 0 and re.search(
        r"立即停机|马上停机|必须停机|立即更换|马上更换|直接更换|报废", action_text
    ):
        issues.append("healthy_class_has_emergency_action")
    if "更换" in action_text and not re.search(r"确认|检查|拆检|复核|若|如果|后", action_text):
        issues.append("replacement_not_conditional_on_verification")

    review_text = field_values.get("human_review_reason_zh", "")
    requires_strict_safety = bool(
        label in SEVERE_MAINTENANCE_LABELS
        or (priority is not None and priority >= 3)
        or (reference_priority is not None and reference_priority >= 3)
    )
    if not human_review_required:
        issues.append("human_review_not_required")
    if not re.search(r"人工|专业人员", review_text):
        issues.append("human_review_reason_missing_human_confirmation")
    if requires_strict_safety and not (
        re.search(r"人工|专业人员", review_text)
        and re.search(r"现场安全|安全规程|安全操作", review_text)
    ):
        issues.append("high_priority_without_human_and_site_safety_wording")
    return issues


def _maintenance_report_sentences(row: Dict[str, object]) -> List[str]:
    narrative_fields = (
        "action_timeline_zh",
        "suspected_cause_zh",
        "inspection_items_zh",
        "recommended_action_zh",
        "verification_plan_zh",
    )
    sentences: List[str] = []
    for key in narrative_fields:
        text = _maintenance_field_text(row, key)
        sentences.extend(
            part.strip()
            for part in re.split(r"[。！？!?；;\n]+", text)
            if part.strip()
        )
    return sentences


def _unsupported_claim_reasons(sentence: str) -> List[str]:
    """Flag only explicit unsupported observations, not proposed checks/actions."""
    text = str(sentence).strip()
    if not text:
        return []
    conditional_or_action = bool(re.search(
        r"若|如果|如有|可能|疑似|建议|检查|监测|确认|核实|排查|评估|防止|避免|考虑",
        text,
    ))
    reasons: List[str] = []
    if not conditional_or_action and re.search(
        r"(?:温度|温升).{0,10}(?:已经|已|当前|明显|异常|超标|过高|升高)", text
    ):
        reasons.append("unobserved_temperature")
    if not conditional_or_action and re.search(
        r"(?:润滑脂|润滑油|润滑状态).{0,10}(?:已经|已|存在|明显|严重)?(?:污染|变质|劣化|失效|不足)",
        text,
    ):
        reasons.append("unobserved_lubricant_condition")
    if re.search(r"\d+(?:\.\d+)?\s*(?:mm|毫米|μm|um|微米)", text, flags=re.IGNORECASE):
        reasons.append("unobserved_defect_dimension")
    if re.search(
        r"(?:将在|会在|预计在|剩余寿命|可继续运行).{0,12}\d+\s*(?:小时|天|周|月).{0,12}(?:失效|损坏|故障|寿命)",
        text,
    ):
        reasons.append("unsupported_failure_time_or_rul")
    if re.search(
        r"(?:振动|电流).{0,12}\d+(?:\.\d+)?\s*(?:g|m/s(?:\^?2)?|mm/s|A|安培)\b",
        text,
        flags=re.IGNORECASE,
    ):
        reasons.append("unobserved_signal_amplitude")
    if not conditional_or_action and re.search(
        r"(?:维修历史|维护历史|服役历史|此前维修|已经运行|已运行).{0,12}\d+",
        text,
    ):
        reasons.append("unobserved_service_history")
    return reasons


def evaluate_single_maintenance_report_rules(row: Dict[str, object]) -> Dict[str, object]:
    """Evaluate format and predefined PU-rule compliance for one class report."""
    label = int(row.get("final_label_id", row.get("fault_label_id", -1)))
    field_values = {
        key: _maintenance_field_text(row, key) for key in MAINTENANCE_REQUIRED_FIELDS
    }
    completed_fields = sum(bool(value) for value in field_values.values())
    field_completeness_rate = 100.0 * completed_fields / len(MAINTENANCE_REQUIRED_FIELDS)
    structure_valid = bool(row.get("maintenance_generation_valid", False))

    combined_text = " ".join(field_values.values())
    affected_component_text = " ".join(
        field_values[key]
        for key in ("suspected_cause_zh", "inspection_items_zh", "recommended_action_zh", "verification_plan_zh")
    )
    inner_mentioned = bool(re.search(
        r"内圈|内滚道|轴承内环|内外圈|inner[_ -]?race",
        affected_component_text,
        flags=re.IGNORECASE,
    ))
    outer_mentioned = bool(re.search(
        r"外圈|外滚道|轴承外环|内外圈|outer[_ -]?race",
        affected_component_text,
        flags=re.IGNORECASE,
    ))
    consistency_issues: List[str] = []
    if structure_valid:
        if not bool(row.get("maintenance_diagnosis_consistent", True)):
            consistency_issues.append("class_report_changed_frozen_diagnosis")
        if label in OUTER_RACE_LABELS and not outer_mentioned:
            consistency_issues.append("outer_race_fault_without_outer_race_action")
        if label in OUTER_RACE_LABELS and inner_mentioned and not outer_mentioned:
            consistency_issues.append("outer_race_fault_checks_only_inner_race")
        if label in INNER_RACE_LABELS and not inner_mentioned:
            consistency_issues.append("inner_race_fault_without_inner_race_action")
        if label in INNER_RACE_LABELS and outer_mentioned and not inner_mentioned:
            consistency_issues.append("inner_race_fault_checks_only_outer_race")
        if label in COMPOUND_RACE_LABELS and not (inner_mentioned and outer_mentioned):
            consistency_issues.append("compound_fault_does_not_cover_both_raceways")
        if label == 0 and re.search(r"立即停机|马上停机|必须停机|立即更换|马上更换|直接更换|报废", combined_text):
            consistency_issues.append("healthy_class_has_unconditional_shutdown_or_replacement")
    diagnosis_consistent: Optional[bool] = (
        not consistency_issues if structure_valid else None
    )

    sentences = _maintenance_report_sentences(row)
    unsupported_statements: List[Dict[str, object]] = []
    for sentence in sentences:
        reasons = _unsupported_claim_reasons(sentence)
        if reasons:
            unsupported_statements.append({"sentence": sentence, "reasons": reasons})

    priority = parse_maintenance_priority(field_values["maintenance_priority"])
    reference_priority = RULE_BASED_PRIORITY_REFERENCE.get(label)
    priority_agreement: Optional[bool] = (
        bool(priority == reference_priority)
        if priority is not None and reference_priority is not None
        else None
    )

    safety_issues: List[str] = []
    if re.search(r"无需人工复核|不需要人工复核|可替代人工|替代现场检查|无需遵守.*安全", combined_text):
        safety_issues.append("claims_to_replace_human_or_site_safety_procedure")
    if any(
        "unsupported_failure_time_or_rul" in item["reasons"]
        for item in unsupported_statements
    ):
        safety_issues.append("unsupported_failure_deadline_or_remaining_life")
    if label == 0 and (
        (priority is not None and priority >= 3)
        or bool(re.search(r"立即停机|马上停机|必须停机|立即更换|马上更换|直接更换", combined_text))
    ):
        safety_issues.append("healthy_class_has_emergency_action")
    action_text = field_values["recommended_action_zh"]
    if structure_valid and label in SEVERE_MAINTENANCE_LABELS and not re.search(
        r"检查|检修|拆检|停机|降载|更换|专业人员|人工复核", action_text
    ):
        safety_issues.append("severe_fault_has_monitoring_only_or_no_concrete_action")
    human_review = bool(row.get("human_review_required", True))
    # Check the LLM-generated fields themselves. The code-injected safety note is
    # intentionally excluded so that it cannot hide a missing P3/P4 safeguard.
    review_text = _maintenance_field_text(row, "human_review_reason_zh")
    strict_safety_required = bool(
        (priority is not None and priority >= 3)
        or (reference_priority is not None and reference_priority >= 3)
        or label in SEVERE_MAINTENANCE_LABELS
    )
    if strict_safety_required and (
        not human_review
        or not re.search(r"人工|专业人员", review_text)
        or not re.search(r"现场安全|安全规程|安全操作", review_text)
    ):
        safety_issues.append("p3_p4_without_human_confirmation_or_site_safety_reference")
    replacement_is_negated = bool(re.search(
        r"(?:不|无需|避免)[^。；]{0,12}更换", action_text
    ))
    if (
        "更换" in action_text
        and not replacement_is_negated
        and not re.search(r"确认|检查|拆检|复核|若|如果|后", action_text)
    ):
        safety_issues.append("unconditional_replacement_without_verification")

    content_field_values = dict(field_values)
    content_field_values["human_review_reason_zh"] = review_text
    content_issues = assess_class_maintenance_content(
        label, content_field_values, human_review
    ) if structure_valid else ["structure_not_valid"]

    return {
        "fault_label_id": label,
        "fault_name": FAULT_NAMES.get(label, str(label)),
        "maintenance_structure_valid": structure_valid,
        "maintenance_required_field_count": len(MAINTENANCE_REQUIRED_FIELDS),
        "maintenance_completed_field_count": int(completed_fields),
        "maintenance_field_completeness_rate": round(field_completeness_rate, 6),
        "maintenance_rule_consistency_evaluable": structure_valid,
        "maintenance_rule_consistent": diagnosis_consistent,
        "maintenance_rule_consistency_issues": consistency_issues,
        "maintenance_statement_count": len(sentences),
        "maintenance_unsupported_statement_count": len(unsupported_statements),
        "maintenance_unsupported_claim_rate": round(
            100.0 * len(unsupported_statements) / max(len(sentences), 1), 6
        ),
        "maintenance_unsupported_statement_details": unsupported_statements,
        "maintenance_safety_rule_violation": bool(safety_issues),
        "maintenance_safety_rule_issues": safety_issues,
        "maintenance_priority_parsed": priority,
        "maintenance_rule_priority_reference": reference_priority,
        "maintenance_rule_priority_agreement": priority_agreement,
        "maintenance_human_review_required": human_review,
        "maintenance_content_valid": bool(structure_valid and not content_issues),
        "maintenance_content_validation_issues": content_issues,
        **field_values,
    }


def evaluate_rule_based_maintenance_reports(
    reports: Sequence[Dict[str, object]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Aggregate class-level automatic indicators without claiming expert validity."""
    first_row_by_label: Dict[int, Dict[str, object]] = {}
    for row in reports:
        label = int(row.get("final_label_id", -1))
        generation_status = str(row.get("maintenance_generation_status", "") or "").strip()
        report_attempted = generation_status not in {
            "",
            "disabled_or_llm_unavailable",
            "not_generated_no_template_fallback",
        }
        if report_attempted and label in FAULT_NAMES and label not in first_row_by_label:
            first_row_by_label[label] = row

    details = [
        evaluate_single_maintenance_report_rules(first_row_by_label[label])
        for label in sorted(first_row_by_label)
    ]
    detail_by_label = {int(item["fault_label_id"]): item for item in details}
    for row in reports:
        detail = detail_by_label.get(int(row.get("final_label_id", -1)))
        if detail is not None:
            row.update({
                key: value for key, value in detail.items()
                if key.startswith("maintenance_")
            })

    attempted = len(details)
    valid_details = [item for item in details if bool(item["maintenance_structure_valid"])]
    content_valid_details = [
        item for item in valid_details if bool(item.get("maintenance_content_valid", False))
    ]
    consistency_details = [
        item for item in details if bool(item["maintenance_rule_consistency_evaluable"])
    ]
    priority_details = [
        item for item in valid_details
        if item["maintenance_priority_parsed"] is not None
        and item["maintenance_rule_priority_reference"] is not None
    ]
    total_required_fields = sum(int(item["maintenance_required_field_count"]) for item in details)
    total_completed_fields = sum(int(item["maintenance_completed_field_count"]) for item in details)
    # Semantic rule rates are defined only over structurally valid reports.
    # Invalid/empty reports remain visible through RVR/FCR/failure rate and must
    # not create misleading 0% UCR or safety-violation results.
    total_statements = sum(int(item["maintenance_statement_count"]) for item in valid_details)
    unsupported_statements = sum(
        int(item["maintenance_unsupported_statement_count"]) for item in valid_details
    )
    priority_kappa: Optional[float] = None
    if len(priority_details) >= 2:
        try:
            value = float(cohen_kappa_score(
                [int(item["maintenance_rule_priority_reference"]) for item in priority_details],
                [int(item["maintenance_priority_parsed"]) for item in priority_details],
                labels=[0, 1, 2, 3, 4],
                weights="quadratic",
            ))
            priority_kappa = value if math.isfinite(value) else None
        except Exception:
            priority_kappa = None

    metrics: Dict[str, object] = {
        "maintenance_report_evaluation_scope": "automatic_structure_and_predefined_PU_rules_not_expert_validation",
        "maintenance_rule_definition_version": "PU_v3_predefined_before_generation",
        "maintenance_required_fields": list(MAINTENANCE_REQUIRED_FIELDS),
        "maintenance_priority_reference_source": "author_predefined_mapping_from_PU_fault_attributes_and_damage_levels",
        "maintenance_priority_reference_mapping": dict(RULE_BASED_PRIORITY_REFERENCE),
        "maintenance_safety_rule_definitions": list(MAINTENANCE_SAFETY_RULE_DEFINITIONS),
        "maintenance_report_evaluated": bool(attempted > 0),
        "maintenance_report_count": attempted,
        "maintenance_report_valid_count": len(valid_details),
        "maintenance_report_valid_rate": 100.0 * len(valid_details) / max(attempted, 1),
        "maintenance_report_failure_rate": 100.0 * (attempted - len(valid_details)) / max(attempted, 1),
        "maintenance_content_valid_count": len(content_valid_details),
        "maintenance_content_valid_rate": (
            100.0 * len(content_valid_details) / len(valid_details)
            if valid_details else None
        ),
        "maintenance_completed_field_count": total_completed_fields,
        "maintenance_required_field_count": total_required_fields,
        "maintenance_field_completeness_rate": 100.0 * total_completed_fields / max(total_required_fields, 1),
        "maintenance_rule_consistency_evaluable_count": len(consistency_details),
        "maintenance_rule_consistency_rate": (
            100.0 * sum(bool(item["maintenance_rule_consistent"]) for item in consistency_details)
            / len(consistency_details)
            if consistency_details else None
        ),
        "maintenance_unsupported_claim_rate": (
            100.0 * unsupported_statements / total_statements
            if total_statements > 0 else None
        ),
        "maintenance_statement_count": total_statements,
        "maintenance_unsupported_statement_count": unsupported_statements,
        "maintenance_safety_rule_evaluable_count": len(valid_details),
        "maintenance_safety_rule_violation_count": sum(
            bool(item["maintenance_safety_rule_violation"]) for item in valid_details
        ),
        "maintenance_safety_rule_violation_rate": (
            100.0 * sum(bool(item["maintenance_safety_rule_violation"]) for item in valid_details)
            / len(valid_details)
            if valid_details else None
        ),
        "maintenance_rule_priority_evaluable_count": len(priority_details),
        "maintenance_rule_priority_agreement_count": sum(
            bool(item["maintenance_rule_priority_agreement"]) for item in priority_details
        ),
        "maintenance_rule_priority_agreement_rate": (
            100.0 * sum(bool(item["maintenance_rule_priority_agreement"]) for item in priority_details)
            / len(priority_details)
            if priority_details else None
        ),
        "maintenance_rule_priority_weighted_kappa": priority_kappa,
        "maintenance_human_review_evaluable_count": len(valid_details),
        "maintenance_human_review_required_count": sum(
            bool(item["maintenance_human_review_required"]) for item in valid_details
        ),
        "maintenance_human_review_required_rate": (
            100.0 * sum(bool(item["maintenance_human_review_required"]) for item in valid_details)
            / len(valid_details)
            if valid_details else None
        ),
    }
    return metrics, details


def _report_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(result) if math.isfinite(result) else float(default)


def _safe_percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def build_class_maintenance_summaries(
    reports: Sequence[Dict[str, object]],
    modalities: Sequence[Tuple[str, Sequence[int]]],
    router_threshold: float,
) -> List[Dict[str, object]]:
    """Summarize frozen decisions and preliminary LLM-generated class reports."""
    modality_names = [str(name) for name, _indices in modalities]
    summaries: List[Dict[str, object]] = []
    threshold = float(router_threshold)

    for label in sorted(FAULT_NAMES):
        class_rows = [r for r in reports if int(r.get("final_label_id", -1)) == int(label)]
        count = len(class_rows)
        maintenance_source = next(
            (row for row in class_rows if bool(row.get("maintenance_generation_valid", False))),
            class_rows[0] if class_rows else {},
        )

        confidences = [_report_float(r.get("final_confidence", r.get("small_confidence", 0.0))) for r in class_rows]
        risks = [_report_float(r.get("risk_probability", 0.0)) for r in class_rows]
        entropies = [_report_float(r.get("uncertainty_entropy", 0.0)) for r in class_rows]
        variances = [_report_float(r.get("mc_variance", 0.0)) for r in class_rows]

        correct_count = sum(bool(r.get("final_correct", False)) for r in class_rows)
        llm_routed_count = sum(bool(r.get("llm_used", False)) for r in class_rows)
        llm_final_count = sum(str(r.get("final_source", "")) == "llm_delegate" for r in class_rows)
        changed_count = sum(bool(r.get("final_changed_from_small", False)) for r in class_rows)
        fallback_count = sum(bool(r.get("fallback_used", False)) for r in class_rows)
        small_final_count = sum(str(r.get("final_source", "")) == "small_model" for r in class_rows)
        llm_corrected_count = sum(
            bool(r.get("llm_used", False))
            and (not bool(r.get("small_correct", False)))
            and bool(r.get("final_correct", False))
            for r in class_rows
        )
        llm_harmful_count = sum(
            bool(r.get("llm_used", False))
            and bool(r.get("small_correct", False))
            and (not bool(r.get("final_correct", False)))
            for r in class_rows
        )
        llm_net_correction = int(llm_corrected_count - llm_harmful_count)
        high_uncertainty_count = sum(str(r.get("uncertainty_level", "")).lower() == "high" for r in class_rows)
        low_confidence_count = sum(v < 0.70 for v in confidences)
        high_risk_count = sum(v >= max(0.75, threshold + 0.15) for v in risks)

        mean_modality_weights: List[float] = []
        if modality_names and class_rows:
            collected: List[List[float]] = []
            for row in class_rows:
                weights = row.get("modality_weights", [])
                if isinstance(weights, list) and len(weights) >= len(modality_names):
                    collected.append([_report_float(v, 0.0) for v in weights[:len(modality_names)]])
            if collected:
                mean_modality_weights = np.asarray(collected, dtype=float).mean(axis=0).tolist()
        dominant_modality = "无可用模态权重"
        dominant_weight = 0.0
        if mean_modality_weights:
            best_idx = int(np.argmax(np.asarray(mean_modality_weights, dtype=float)))
            dominant_modality = modality_names[best_idx]
            dominant_weight = float(mean_modality_weights[best_idx])

        risk_p90 = _safe_percentile(risks, 90.0)
        high_risk_ratio = high_risk_count / count if count else 0.0
        high_uncertainty_ratio = high_uncertainty_count / count if count else 0.0
        low_confidence_ratio = low_confidence_count / count if count else 0.0
        llm_routed_ratio = llm_routed_count / count if count else 0.0

        priority_name = str(maintenance_source.get("maintenance_priority", ""))
        action_timeline = str(maintenance_source.get("action_timeline", ""))

        human_review_required = bool(
            maintenance_source.get("human_review_required", True)
            if count else True
        )

        status = (
            "本次测试中存在该类别预测样本；诊断LoRA仅用于诊断复核，类别级结构化维护草案由临时关闭该适配器后的同一指令大模型底座在预定义规则约束下零样本生成，须经人工核验后使用。"
            if count
            else "本次测试未预测到该类别，因此不生成维护文本。"
        )
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        avg_risk = float(np.mean(risks)) if risks else 0.0
        class_precision = correct_count / count if count else 0.0
        report_summary = str(maintenance_source.get("maintenance_report_summary_zh", ""))

        summary: Dict[str, object] = {
            "fault_label_id": int(label),
            "fault_name": FAULT_NAMES[label],
            "report_status_zh": status,
            "predicted_sample_count": int(count),
            "correct_prediction_count": int(correct_count),
            "class_precision_within_predictions": round(float(class_precision), 6),
            "average_confidence": round(avg_conf, 6),
            "median_confidence": round(float(np.median(confidences)) if confidences else 0.0, 6),
            "low_confidence_sample_count": int(low_confidence_count),
            "low_confidence_ratio": round(float(low_confidence_ratio), 6),
            "average_risk_probability": round(avg_risk, 6),
            "risk_probability_p90": round(float(risk_p90), 6),
            "high_risk_sample_count": int(high_risk_count),
            "high_risk_ratio": round(float(high_risk_ratio), 6),
            "average_uncertainty_entropy": round(float(np.mean(entropies)) if entropies else 0.0, 6),
            "average_mc_variance": round(float(np.mean(variances)) if variances else 0.0, 8),
            "high_uncertainty_sample_count": int(high_uncertainty_count),
            "high_uncertainty_ratio": round(float(high_uncertainty_ratio), 6),
            "report_model_scope": "full_model_small_encoder_domain_alignment_router_lora_llm",
            "report_uses_final_full_model_decision": True,
            "small_model_final_decision_count": int(small_final_count),
            "llm_routed_sample_count": int(llm_routed_count),
            "llm_routed_ratio": round(float(llm_routed_ratio), 6),
            "llm_final_decision_count": int(llm_final_count),
            "changed_from_small_count": int(changed_count),
            "llm_corrected_error_count": int(llm_corrected_count),
            "llm_harmful_override_count": int(llm_harmful_count),
            "llm_net_correction_count": int(llm_net_correction),
            "fallback_sample_count": int(fallback_count),
            "dominant_evidence_modality": dominant_modality,
            "dominant_modality_mean_weight": round(float(dominant_weight), 6),
            "maintenance_priority": priority_name,
            "action_timeline": action_timeline,
            "suspected_cause_zh": str(maintenance_source.get("suspected_cause_zh", "")),
            "inspection_items_zh": str(maintenance_source.get("inspection_items_zh", "")),
            "recommended_action_zh": str(maintenance_source.get("recommended_action_zh", "")),
            "verification_plan_zh": str(maintenance_source.get("verification_plan_zh", "")),
            "human_review_required": human_review_required,
            "human_review_reason_zh": str(maintenance_source.get("human_review_reason_zh", "")),
            "maintenance_generation_valid": bool(maintenance_source.get("maintenance_generation_valid", False)),
            "maintenance_content_valid": bool(
                maintenance_source.get("maintenance_content_valid", False)
            ),
            "maintenance_content_validation_issues": maintenance_source.get(
                "maintenance_content_validation_issues", []
            ),
            "maintenance_generation_method": str(maintenance_source.get("maintenance_generation_method", "")),
            "maintenance_required_field_count": int(maintenance_source.get("maintenance_required_field_count", 0)),
            "maintenance_completed_field_count": int(maintenance_source.get("maintenance_completed_field_count", 0)),
            "maintenance_field_completeness_rate": _report_float(
                maintenance_source.get("maintenance_field_completeness_rate", 0.0)
            ),
            "maintenance_rule_consistency_evaluable": bool(
                maintenance_source.get("maintenance_rule_consistency_evaluable", False)
            ),
            "maintenance_rule_consistent": maintenance_source.get("maintenance_rule_consistent"),
            "maintenance_rule_consistency_issues": maintenance_source.get(
                "maintenance_rule_consistency_issues", []
            ),
            "maintenance_statement_count": int(maintenance_source.get("maintenance_statement_count", 0)),
            "maintenance_unsupported_statement_count": int(
                maintenance_source.get("maintenance_unsupported_statement_count", 0)
            ),
            "maintenance_unsupported_claim_rate": _report_float(
                maintenance_source.get("maintenance_unsupported_claim_rate", 0.0)
            ),
            "maintenance_safety_rule_violation": bool(
                maintenance_source.get("maintenance_safety_rule_violation", False)
            ),
            "maintenance_safety_rule_issues": maintenance_source.get(
                "maintenance_safety_rule_issues", []
            ),
            "maintenance_priority_parsed": maintenance_source.get("maintenance_priority_parsed"),
            "maintenance_rule_priority_reference": maintenance_source.get(
                "maintenance_rule_priority_reference"
            ),
            "maintenance_rule_priority_agreement": maintenance_source.get(
                "maintenance_rule_priority_agreement"
            ),
            "maintenance_generated_token_count": int(
                maintenance_source.get("maintenance_generated_token_count", 0)
            ),
            "maintenance_revision_accepted_count": int(sum(
                bool(row.get("maintenance_revision_accepted", False)) for row in class_rows
            )),
            "class_report_summary_zh": report_summary,
            "safety_note_zh": "本报告是诊断后的结构化维护草案和可行性展示，不能替代现场检查、人工复核和安全规程。",
        }
        for idx, name in enumerate(modality_names):
            summary[f"mean_modality_weight_{name}"] = round(
                float(mean_modality_weights[idx]) if idx < len(mean_modality_weights) else 0.0,
                6,
            )
        summaries.append(summary)
    return summaries

def uncertainty_level(
    confidence: float,
    entropy: float,
    variance: float,
    risk_probability: float,
    router_threshold: float,
) -> str:
    """Convert numerical uncertainty/risk indicators into a report-friendly level.

    The thresholds are intentionally conservative. They do not affect model
    decisions; they only make the exported diagnostic report easier to read.
    """
    confidence = float(confidence)
    entropy = float(entropy)
    variance = float(variance)
    risk_probability = float(risk_probability)
    router_threshold = float(router_threshold)
    if (
        risk_probability >= max(0.75, router_threshold + 0.15)
        or confidence < 0.70
        or entropy >= 0.80
        or variance >= 0.020
    ):
        return "high"
    if (
        risk_probability >= router_threshold
        or confidence < 0.90
        or entropy >= 0.30
        or variance >= 0.005
    ):
        return "medium"
    return "low"


def uncertainty_summary(
    confidence: float,
    entropy: float,
    variance: float,
    risk_probability: float,
    prototype_distance: float,
    domain_shift: float,
) -> str:
    return (
        f"confidence={float(confidence):.4f}, entropy={float(entropy):.4f}, "
        f"mc_variance={float(variance):.6f}, risk={float(risk_probability):.4f}, "
        f"prototype_distance={float(prototype_distance):.4f}, domain_shift={float(domain_shift):.4f}"
    )


REVIEWER_SYSTEM_PROMPT = """You are a constrained PU rolling-bearing fault-diagnosis classifier for Router-selected hard samples.
Use only the provided motor-current and vibration evidence. Choose final_label from candidate_labels only.
The most important evidence is candidate_evidence_profiles: higher relative_support is better; rank=1 is best; pairwise/rerank/nearest-prototype matches are independent supports.
Do not overturn the small-model label unless the relative evidence clearly supports another candidate.
Return exactly one SHORT JSON object. Put candidate_scores immediately after answer. No Markdown. No extra text.
Required schema:
{"answer":"A: C<label>","candidate_scores":[{"label_id":<id>,"score":<num>}],"final_label":<id>,"final_fault":"<name>","confidence":<num>,"decision":"classify","reasoning":"<14 words max>"}
The candidate_scores list must contain every candidate label exactly once and scores must sum to 1.
Do not output reasoning_chain or maintenance_suggestion."""

MAINTENANCE_REVISION_SYSTEM_PROMPT = """You generate a preliminary post-diagnosis structured maintenance draft for subsequent human verification.
Use only the supplied diagnosis, candidate set, motor-current/vibration evidence, operating condition, and model statistics.
The task is zero-shot with respect to maintenance: the loaded LoRA adapter was trained only for diagnosis and contains no maintenance targets.
If revision_allowed is true, you may propose a different diagnosis only from candidate_labels. Score every candidate exactly once and make proposed_label the argmax candidate.
If revision_allowed is false, keep the supplied diagnosis and return the single supplied candidate with score 1.
Do not invent unobserved temperatures, defect dimensions, vibration/current amplitudes, lubricant contamination, service history, or inspection findings.
Do not predict an exact remaining useful life or an exact failure deadline from the supplied evidence.
Do not claim that the draft replaces on-site inspection, human review, maintenance manuals, or safety procedures.
For a healthy diagnosis, do not recommend unconditional shutdown or bearing replacement.
For severe or compound faults, include a concrete inspection/escalation action rather than monitoring alone.
For P3/P4, set human_review_required to true and explicitly require professional confirmation and site safety procedures.
Generate all maintenance text yourself in Chinese; do not retrieve or quote a class-bound maintenance template.
maintenance_priority must start with exactly one of P0, P1, P2, P3, or P4.
Return exactly one JSON object, no Markdown and no extra text, using this schema:
{"candidate_scores":[{"label_id":<id>,"score":<num>}],"proposed_label":<id>,"proposed_fault":"<exact candidate name>","revision_reason":"<short reason>","maintenance_priority":"P0/P1/P2/P3/P4 plus a short Chinese label","action_timeline_zh":"...","suspected_cause_zh":"...","inspection_items_zh":"...","recommended_action_zh":"...","verification_plan_zh":"...","human_review_required":true,"human_review_reason_zh":"...","report_summary_zh":"..."}
Candidate scores must sum to 1. All maintenance fields must be non-empty."""

CLASS_MAINTENANCE_SYSTEM_PROMPT = """任务：诊断标签冻结后，生成中文维护建议内容。不得重新分类、修改诊断，也不得自行改变安全策略给出的维护优先级和处置时机。

输入中的 policy_priority_zh 和 policy_action_timeline_zh 已由确定性安全策略根据冻结的 fault_attributes 计算，仅用于约束维护建议强度；不要重新推断优先级。只能依据 fault_semantics、fault_attributes、required_component_terms_zh、available_sensing_modalities 和 target_operating_conditions 生成维护内容。fault_attributes 是公开 PU 数据集类别属性，不是现场拆检结果。

生成要求：
- 故障类 suspected_cause_zh 必须含“可能”或“疑似”，并说明损伤、点蚀、疲劳或故障机理；健康类要明确诊断未显示稳定故障。
- 对于故障类，required_component_terms_zh 中每个中文部件词都必须出现在 suspected_cause_zh、inspection_items_zh 和 recommended_action_zh 中；复合故障必须同时写“内圈”和“外圈”。
- inspection_items_zh 写具体检查对象和方法；recommended_action_zh 写与 policy_priority_zh 强度一致的具体维护措施，但更换必须以检查/复核/拆检确认作为前提。
- verification_plan_zh 写具体复核方案，优先使用电流、振动、包络/频谱或现场检查；不要把“待确认”“无”“进行必要的检查”等占位语当作内容。
- 健康类仍需写电流与振动趋势检查、常规监测和健康基线复核，不得建议无条件停机或更换。
- 【K002-healthy专用硬约束】当 current_diagnosis_label_id=0 或 current_diagnosis_name=K002-healthy 时：suspected_cause_zh 必须明确写“未发现/未显示明确故障指征，当前状态接近健康基线”这一语义；禁止写“可能/疑似存在点蚀、损伤、裂纹、疲劳或故障”等正向故障判断。inspection_items_zh、recommended_action_zh 和 verification_plan_zh 只能围绕电流/振动趋势检查、常规监测和健康基线复核展开；不得直接建议停机、降载、拆检、修复或更换。若需要提到升级措施，只能写成“若后续监测发现异常，再由专业人员进一步检查”的条件式表达。
- 复合、三级、严重或 P3/P4 情况，recommended_action_zh 必须包含具体升级措施；human_review_reason_zh 必须原样包含“专业人员确认”和“现场安全规程”。其他情况 human_review_reason_zh 至少包含“人工”或“专业人员”。
- 不得编造实测温度、缺陷尺寸、信号幅值、润滑状态、维修历史、剩余寿命或确定失效时间，不得声称替代人工和现场规程。
- 当前可用在线感知仅限输入中列出的模态。不得把温度、径向力、润滑状态、维修历史、剩余寿命、缺陷尺寸或未提供的信号幅值写成“测量/监测项目”或已知事实；如确需涉及，只能表述为“由专业人员按现场规程检查确认”。

只输出一个完整 JSON，不要 Markdown、解释或额外字段。仅输出以下七个键：suspected_cause_zh、inspection_items_zh、recommended_action_zh、verification_plan_zh、human_review_required、human_review_reason_zh、report_summary_zh。human_review_required 必须为 true，其余字段必须为结合本次输入生成的非空字符串。maintenance_priority 和 action_timeline_zh 由代码中的安全策略层写入最终报告，不需要也不允许在这里生成。"""

def _rank_map(values: Dict[int, float], *, higher_is_better: bool) -> Dict[int, int]:
    """Return zero-based ranks for candidate labels; 0 is best."""
    ordered = sorted(
        values.items(),
        key=lambda kv: ((-kv[1]) if higher_is_better else kv[1], kv[0]),
    )
    return {int(label): int(rank) for rank, (label, _value) in enumerate(ordered)}


def _rank_score(rank: int, n: int) -> float:
    """Map zero-based rank to [0,1] score, where best rank gets 1."""
    n = max(int(n), 1)
    if n == 1:
        return 1.0
    return float(max(0.0, (n - 1 - int(rank)) / (n - 1)))


def build_llm_candidate_profiles(evidence: Dict[str, object]) -> List[Dict[str, object]]:
    """Build relative, LLM-facing candidate evidence descriptors.

    Earlier versions exposed absolute source-fitted z-score consistencies. On
    the unseen target domain these values can saturate (for example proto/domain z≈99), so the LLM
    saw several candidates with nearly identical tiny support and learned an
    unstable language preference. V9.1 instead exposes relative ranks/gaps among
    the closed candidate set, plus compact pairwise/rerank/prototype agreements.
    """
    detail_by_label: Dict[int, Dict[str, object]] = {}
    for item in evidence.get("advisor_rerank_details", []) or []:
        try:
            detail_by_label[int(float(item.get("label_id")))] = item
        except (TypeError, ValueError, AttributeError):
            continue

    labels: List[int] = []
    for source in (evidence.get("candidate_labels", []), evidence.get("top3", [])):
        for item in source or []:
            try:
                label_id = int(item.get("label_id"))
            except (TypeError, ValueError, AttributeError):
                continue
            if label_id in FAULT_NAMES and label_id not in labels:
                labels.append(label_id)
        if labels:
            break
    labels = sorted(labels)
    if not labels:
        return []

    small_prob: Dict[int, float] = {label_id: 0.0 for label_id in labels}
    for item in evidence.get("top3", []) or []:
        try:
            label_id = int(item.get("label_id"))
            if label_id in small_prob:
                small_prob[label_id] = float(item.get("prob", 0.0))
        except (TypeError, ValueError, AttributeError):
            continue

    def _detail_float(label_id: int, key: str, default: float) -> float:
        try:
            value = float(detail_by_label.get(label_id, {}).get(key, default))
        except (TypeError, ValueError):
            value = default
        if not math.isfinite(value):
            value = default
        return float(value)

    advisor_score = {label_id: _detail_float(label_id, "score", -1e9) for label_id in labels}
    proto_z = {label_id: _detail_float(label_id, "proto_z", 99.0) for label_id in labels}
    domain_z = {label_id: _detail_float(label_id, "domain_z", 99.0) for label_id in labels}
    physics_z = {label_id: _detail_float(label_id, "physics_z", 99.0) for label_id in labels}

    prob_rank = _rank_map(small_prob, higher_is_better=True)
    advisor_rank = _rank_map(advisor_score, higher_is_better=True)
    proto_rank = _rank_map(proto_z, higher_is_better=False)
    domain_rank = _rank_map(domain_z, higher_is_better=False)
    physics_rank = _rank_map(physics_z, higher_is_better=False)
    n = len(labels)

    best_prob = max(small_prob.values()) if small_prob else 0.0
    best_score = max(advisor_score.values()) if advisor_score else 0.0
    best_physics = min(physics_z.values()) if physics_z else 99.0

    try:
        nearest_label = int(evidence.get("nearest_prototype_label_id", -1))
    except (TypeError, ValueError):
        nearest_label = -1
    try:
        rerank_label = int(evidence.get("advisor_rerank_label_id", -1))
    except (TypeError, ValueError):
        rerank_label = -1

    pairwise_final = -1
    pairwise_conf = 0.0
    pairwise_pair: List[int] = []
    pairwise = evidence.get("pairwise_advisor")
    if isinstance(pairwise, dict):
        try:
            pairwise_pair = [int(v) for v in pairwise.get("pair", [])]
            pairwise_final = int(pairwise.get("final_label", -1))
            pairwise_conf = max(
                float(pairwise.get("prob_pair_first", 0.0)),
                float(pairwise.get("prob_pair_second", 0.0)),
            )
        except (TypeError, ValueError):
            pairwise_final = -1
            pairwise_conf = 0.0
            pairwise_pair = []

    profiles: List[Dict[str, object]] = []
    for label_id in labels:
        prob_score = _rank_score(prob_rank[label_id], n)
        advisor_score_rank_score = _rank_score(advisor_rank[label_id], n)
        physics_rank_score = _rank_score(physics_rank[label_id], n)
        proto_rank_score = _rank_score(proto_rank[label_id], n)
        domain_rank_score = _rank_score(domain_rank[label_id], n)
        pairwise_support = 1.0 if (label_id == pairwise_final and pairwise_conf >= 0.90) else 0.0
        nearest_support = 1.0 if label_id == nearest_label else 0.0
        rerank_support = 1.0 if label_id == rerank_label else 0.0
        # Relative support is intentionally rank-based. It resists the saturated
        # z-score problem while still exposing who is strongest within top-k.
        relative_support = (
            0.18 * prob_score
            + 0.24 * advisor_score_rank_score
            + 0.24 * physics_rank_score
            + 0.12 * proto_rank_score
            + 0.08 * domain_rank_score
            + 0.06 * pairwise_support
            + 0.04 * nearest_support
            + 0.04 * rerank_support
        )
        profiles.append({
            "label_id": int(label_id),
            "label": FAULT_NAMES[int(label_id)],
            "relative_support": round(float(relative_support), 4),
            # Kept as an alias so old SFT helper code remains compatible, but
            # the prompt now explains that this is relative rather than absolute.
            "combined_support": round(float(relative_support), 4),
            "small_prob": round(float(small_prob[label_id]), 6),
            "small_prob_rank": int(prob_rank[label_id] + 1),
            "small_prob_gap_to_best": round(float(small_prob[label_id] - best_prob), 6),
            "advisor_score_rank": int(advisor_rank[label_id] + 1),
            "advisor_score_gap_to_best": round(float(advisor_score[label_id] - best_score), 6),
            "physics_rank": int(physics_rank[label_id] + 1),
            "physics_gap_to_best": round(float(physics_z[label_id] - best_physics), 6),
            "prototype_rank": int(proto_rank[label_id] + 1),
            "domain_rank": int(domain_rank[label_id] + 1),
            "pairwise_support": bool(pairwise_support),
            "pairwise_confidence": round(float(pairwise_conf if label_id in pairwise_pair else 0.0), 6),
            "nearest_prototype_match": bool(nearest_support),
            "rerank_match": bool(rerank_support),
        })
    return sorted(profiles, key=lambda row: int(row["label_id"]))



def compact_llm_physical_summary(evidence: Dict[str, object]) -> Dict[str, object]:
    """Return a short physical summary for the text LLM prompt.

    V9.1 exposed the full detailed physical_summary and full aligned token
    matrix, which made the prompt longer and caused Gemma to ignore the
    candidate_scores schema. V9.1 keeps only the diagnostic statistics that
    are most useful for class comparison.
    """
    src = evidence.get("physical_summary", {}) or {}
    out: Dict[str, object] = {
        "norm": round(float(src.get("norm", 0.0)), 5),
        "max_abs": round(float(src.get("max_abs", 0.0)), 5),
        "dominant_rms_modality": src.get("dominant_rms_modality", "unknown"),
        "dominant_kurtosis_modality": src.get("dominant_kurtosis_modality", "unknown"),
        "dominant_high_frequency_modality": src.get("dominant_high_frequency_modality", "unknown"),
    }
    rows = []
    for row in src.get("modality_physics", []) or []:
        if not isinstance(row, dict):
            continue
        rows.append({
            "m": row.get("modality", "unknown"),
            "rms": round(float(row.get("rms", 0.0)), 5),
            "kurtosis": round(float(row.get("kurtosis", 0.0)), 5),
            "crest": round(float(row.get("crest_factor", 0.0)), 5),
            "entropy": round(float(row.get("spectral_entropy", 0.0)), 5),
            "high_freq": round(float(row.get("high_freq_ratio", 0.0)), 5),
        })
    out["modalities"] = rows
    return out


def synthesize_candidate_scores_from_answer(
    evidence: Dict[str, object],
    candidate_ids: Sequence[int],
    answer_label: int,
) -> Tuple[bool, List[Dict[str, float]], int, float, float, str]:
    """Recover usable candidate_scores when the LLM follows answer-first but omits scores.

    This keeps the borrowed Qwen-style answer-first idea useful while remaining
    conservative: synthesized confidence is high only when non-LLM relative
    candidate evidence strongly supports the answered label. Weak answers still
    fail the downstream score/relative-support gates.
    """
    ids = [int(x) for x in candidate_ids if int(x) in FAULT_NAMES]
    answer_label = int(answer_label)
    if answer_label not in ids:
        return False, [], -1, 0.0, 0.0, "answer_label_outside_candidates"
    profiles = {int(p.get("label_id", -1)): p for p in build_llm_candidate_profiles(evidence)}
    supports: Dict[int, float] = {}
    for label_id in ids:
        try:
            supports[label_id] = float(profiles.get(label_id, {}).get("relative_support", 0.0))
        except (TypeError, ValueError):
            supports[label_id] = 0.0
    answer_support = max(0.0, min(1.0, supports.get(answer_label, 0.0)))
    # Confidence stays below the default 0.82 gate unless relative support is strong.
    answer_mass = max(0.55, min(0.93, 0.55 + 0.35 * answer_support))
    others = [x for x in ids if x != answer_label]
    remaining = max(0.0, 1.0 - answer_mass)
    denom = sum(max(0.01, 1.0 - supports.get(x, 0.0)) for x in others) or 1.0
    score_map = {answer_label: answer_mass}
    for x in others:
        score_map[x] = remaining * max(0.01, 1.0 - supports.get(x, 0.0)) / denom
    out = [{"label_id": int(label_id), "score": round(float(score_map[label_id]), 6)} for label_id in ids]
    if out:
        drift = round(1.0 - sum(float(row["score"]) for row in out), 6)
        out[0]["score"] = round(float(out[0]["score"]) + drift, 6)
    ordered = sorted(((int(row["label_id"]), float(row["score"])) for row in out), key=lambda kv: (-kv[1], kv[0]))
    final_label = int(ordered[0][0])
    confidence = float(ordered[0][1])
    second = float(ordered[1][1]) if len(ordered) > 1 else 0.0
    margin = confidence - second
    if final_label != answer_label:
        return False, [], -1, 0.0, 0.0, "synthesized_answer_not_argmax"
    return True, out, final_label, round(confidence, 6), round(margin, 6), "synthesized_from_answer_relative_support"

def build_reviewer_user_message(evidence: Dict[str, object]) -> str:
    # V8.8.2: Router-only LLM delegation with small-model context.
    # Router is still the only hard-sample selector. The LLM receives the
    # small-model distribution as evidence, but it remains the final classifier
    # for routed samples rather than a yes/no reviewer.
    top3_items = evidence.get("top3", []) or []
    candidate_labels = []
    for item in top3_items:
        try:
            label_id = int(item.get("label_id"))
        except (TypeError, ValueError, AttributeError):
            continue
        if label_id in FAULT_NAMES:
            candidate_labels.append({"label_id": label_id, "label": FAULT_NAMES[label_id]})
    if not candidate_labels:
        candidate_labels = evidence.get("candidate_labels", [
            {"label_id": int(k), "label": v} for k, v in FAULT_NAMES.items()
        ])
    # Sort candidates by label id so the prompt does not reveal or encourage copying
    # the small-model top-1/top-k ranking order.
    cleaned: List[Dict[str, object]] = []
    seen: set[int] = set()
    for item in candidate_labels:
        try:
            label_id = int(item.get("label_id"))
        except (TypeError, ValueError, AttributeError):
            continue
        if label_id in FAULT_NAMES and label_id not in seen:
            cleaned.append({"label_id": label_id, "label": FAULT_NAMES[label_id]})
            seen.add(label_id)
    candidate_labels = sorted(cleaned, key=lambda row: int(row["label_id"]))
    candidate_ids = {int(item["label_id"]) for item in candidate_labels if "label_id" in item}
    # Keep the prompt compact. Full fault-knowledge paragraphs made the JSON
    # generation less reliable in V9.1, so candidate labels carry the names.
    small_topk = []
    for item in evidence.get("top3", []) or []:
        try:
            label_id = int(item.get("label_id"))
            if label_id in candidate_ids:
                small_topk.append({
                    "label_id": label_id,
                    "label": FAULT_NAMES[label_id],
                    "prob": round(float(item.get("prob", 0.0)), 6),
                })
        except (TypeError, ValueError, AttributeError):
            continue
    small_model_evidence = {
        "topk": small_topk,
        "confidence": round(float(evidence.get("small_model_confidence", 0.0)), 6),
    }
    uncertainty_evidence = {
        "uncertainty": round(float(evidence.get("uncertainty", 0.0)), 6),
        "mc_variance": round(float(evidence.get("mc_variance", 0.0)), 6),
        "risk_probability": round(float(evidence.get("risk_probability", 0.0)), 6),
        "prototype_distance": round(float(evidence.get("prototype_distance", 0.0)), 6),
        "domain_shift": round(float(evidence.get("domain_shift", 0.0)), 6),
        "nearest_prototype_label_id": int(evidence.get("nearest_prototype_label_id", -1)),
        "nearest_prototype_label": str(evidence.get("nearest_prototype_label", "")),
    }
    rerank_evidence = {
        "label_id": int(evidence.get("advisor_rerank_label_id", -1)),
        "label": str(evidence.get("advisor_rerank_label", "")),
    }
    pairwise_evidence = {}
    if isinstance(evidence.get("pairwise_advisor"), dict):
        pairwise = evidence["pairwise_advisor"]
        pairwise_evidence = {
            "pair": pairwise.get("pair", []),
            "pair_names": pairwise.get("pair_names", []),
            "final_label": pairwise.get("final_label"),
            "final_label_name": pairwise.get("final_label_name", ""),
            "prob_pair_first": round(float(pairwise.get("prob_pair_first", 0.0)), 6),
            "prob_pair_second": round(float(pairwise.get("prob_pair_second", 0.0)), 6),
        }
    llm_evidence = {
        "candidate_labels": candidate_labels,
        "small_model_evidence": small_model_evidence,
        "uncertainty_evidence": uncertainty_evidence,
        "candidate_evidence_profiles": build_llm_candidate_profiles(evidence),
        "rerank_evidence": rerank_evidence,
        "pairwise_evidence": pairwise_evidence,
        # Compact summaries only. Full token matrices and long knowledge text
        # made V9.1 slower and much less JSON-stable.
        "feature_token_summary": evidence.get("feature_token_summary", {}),
        "physical_summary": compact_llm_physical_summary(evidence),
        "modality_weights": [round(float(v), 5) for v in (evidence.get("modality_weights", []) or [])],
        "domain": evidence.get("domain", "unknown"),
    }
    return (
        "Classify this hard sample. Output ONLY the required short JSON. "
        "candidate_scores must be the second field and must include every candidate.\n"
        + json.dumps(llm_evidence, ensure_ascii=False, separators=(",", ":"))
    )


FORBIDDEN_LLM_REASONING_TERMS = (
    "prototype_z", "domain_z", "physics_z", "hidden logits", "score_gain",
)


def has_forbidden_llm_reasoning_terms(text: object) -> bool:
    """Detect stale V7/V8.1 reasoning templates in LLM outputs.

    V8.8 trains the LLM to reason from aligned feature tokens and physical
    summaries only. If the generated explanation still mentions unavailable
    old evidence fields such as probability/prototype_z/domain_z, the result is
    treated as unreliable for overriding the small model.
    """
    lower = str(text or "").lower()
    return any(term in lower for term in FORBIDDEN_LLM_REASONING_TERMS)


def llm_answer_tag(label: int) -> str:
    """Qwen-style answer-first tag used in both SFT targets and inference validation."""
    return f"A: C{int(label)}"


def parse_llm_answer_tag(value: object) -> Optional[int]:
    """Parse tags such as "A: C3" or "C3" into an integer label id."""
    match = re.search(r"(?:^|\b)A?\s*:?\s*C\s*(\d+)(?:\b|$)", str(value or ""), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        label = int(match.group(1))
    except ValueError:
        return None
    return label if label in FAULT_NAMES else None


def build_llm_sft_target(evidence: Dict[str, object], true_label: int, reasoning: str) -> Dict[str, object]:
    """Build the exact assistant JSON used for LoRA-SFT.

    This centralizes the output schema so training targets and inference parser
    stay aligned: answer first, then structured candidate scoring and reasoning.
    """
    candidate_scores = build_teacher_candidate_scores(evidence, true_label)
    target_confidence = next(
        (float(row["score"]) for row in candidate_scores if int(row["label_id"]) == int(true_label)),
        0.0,
    )
    return {
        "answer": llm_answer_tag(true_label),
        "candidate_scores": candidate_scores,
        "final_label": int(true_label),
        "final_fault": FAULT_NAMES[int(true_label)],
        "confidence": round(float(target_confidence), 6),
        "decision": "classify",
        "reasoning": " ".join(str(reasoning).split()[:12]),
    }


def candidate_ids_from_evidence(evidence: Dict[str, object], top3: Optional[List[int]] = None) -> List[int]:
    """Return the closed candidate label IDs used by the LLM classifier."""
    ids: List[int] = []
    sources = []
    if top3:
        sources.append([{"label_id": int(x)} for x in top3])
    sources.append(evidence.get("candidate_labels", []))
    sources.append(evidence.get("top3", []))
    for source in sources:
        for item in source or []:
            try:
                label_id = int(item.get("label_id"))
            except (TypeError, ValueError, AttributeError):
                continue
            if label_id in FAULT_NAMES and label_id not in ids:
                ids.append(label_id)
        if ids:
            break
    return ids or list(sorted(FAULT_NAMES.keys()))


def normalize_candidate_scores_object(
    raw_scores: object,
    candidate_ids: Sequence[int],
) -> Tuple[bool, List[Dict[str, float]], int, float, float, str]:
    """Parse and normalize LLM candidate_scores.

    The LLM is the delegated classifier, but the final label is computed as
    argmax(candidate_scores). This avoids trusting a separate free-form
    final_label/confidence field. The function only checks output form; it does
    not compare against the small-model prediction, pairwise head, or reranker.
    """
    ids = [int(x) for x in candidate_ids if int(x) in FAULT_NAMES]
    if not ids:
        return False, [], -1, 0.0, 0.0, "empty_candidate_set"
    if not isinstance(raw_scores, list):
        return False, [], -1, 0.0, 0.0, "missing_candidate_scores"

    score_map: Dict[int, float] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        label_value = item.get("label_id", item.get("label", item.get("id")))
        score_value = item.get("score", item.get("support", item.get("weight")))
        try:
            label_id = int(label_value)
            score = float(score_value)
        except (TypeError, ValueError):
            continue
        if label_id not in ids:
            continue
        if not math.isfinite(score):
            continue
        score_map[label_id] = max(0.0, score)

    missing = [x for x in ids if x not in score_map]
    if missing:
        return False, [], -1, 0.0, 0.0, "candidate_scores_missing_labels:" + ",".join(map(str, missing))
    total = sum(score_map.values())
    if total <= 0.0:
        return False, [], -1, 0.0, 0.0, "candidate_scores_nonpositive_sum"

    normalized = {label_id: score_map[label_id] / total for label_id in ids}
    ordered = sorted(normalized.items(), key=lambda kv: (-kv[1], kv[0]))
    final_label = int(ordered[0][0])
    confidence = float(ordered[0][1])
    second = float(ordered[1][1]) if len(ordered) > 1 else 0.0
    margin = confidence - second
    out = [
        {"label_id": int(label_id), "score": round(float(normalized[label_id]), 6)}
        for label_id in ids
    ]
    # Correct possible rounding drift while preserving order.
    if out:
        drift = round(1.0 - sum(float(row["score"]) for row in out), 6)
        out[0]["score"] = round(float(out[0]["score"]) + drift, 6)
    return True, out, final_label, round(confidence, 6), round(margin, 6), "ok"


def build_teacher_candidate_scores(evidence: Dict[str, object], true_label: int) -> List[Dict[str, float]]:
    """Create source-only SFT target scores with the true class ranked first."""
    ids = candidate_ids_from_evidence(evidence)
    true_label = int(true_label)
    if true_label not in ids:
        ids = [true_label] + [x for x in ids if x != true_label]
    profiles = {int(p.get("label_id", -1)): p for p in build_llm_candidate_profiles(evidence)}
    raw: Dict[int, float] = {}
    for label_id in ids:
        p = profiles.get(label_id, {})
        support = float(p.get("relative_support", p.get("combined_support", 0.05)))
        raw[label_id] = max(0.03, 0.08 + 0.20 * support)
    # The teacher distribution deliberately teaches ranking, not a free-form
    # self-confidence. True label receives the dominant mass, while other
    # candidates keep small non-zero scores to teach comparison.
    raw[true_label] = max(raw.get(true_label, 0.0), 0.76)
    total = sum(raw.values())
    norm = {k: v / total for k, v in raw.items()}
    out = [{"label_id": int(label_id), "score": round(float(norm[label_id]), 6)} for label_id in ids]
    if out:
        drift = round(1.0 - sum(float(row["score"]) for row in out), 6)
        out[0]["score"] = round(float(out[0]["score"]) + drift, 6)
    return out


def clean_token_physical_reasoning(evidence: Dict[str, object], label: int, action: str) -> str:
    """Source-only SFT target reasoning for direct hard-sample LLM classification."""
    token_summary = evidence.get("feature_token_summary", {}) or {}
    physical_summary = evidence.get("physical_summary", {}) or {}
    modality_weights = evidence.get("modality_weights", []) or []
    try:
        token_std = float(token_summary.get("std", 0.0))
        token_norm = float(token_summary.get("norm", 0.0))
        token_max = float(token_summary.get("max_abs", 0.0))
    except (TypeError, ValueError, AttributeError):
        token_std = token_norm = token_max = 0.0
    try:
        physical_std = float(physical_summary.get("std", 0.0))
        physical_norm = float(physical_summary.get("norm", 0.0))
        physical_max = float(physical_summary.get("max_abs", 0.0))
    except (TypeError, ValueError, AttributeError):
        physical_std = physical_norm = physical_max = 0.0
    if modality_weights:
        names = list(evidence.get("modality_names", ["motor_current", "vibration"]))
        dominant_idx = max(range(len(modality_weights)), key=lambda i: float(modality_weights[i]))
        dominant = names[dominant_idx] if dominant_idx < len(names) else f"modality_{dominant_idx}"
    else:
        dominant = "multimodal"

    profiles = build_llm_candidate_profiles(evidence)
    selected_profile = next((p for p in profiles if int(p.get("label_id", -1)) == int(label)), None)
    # V9.1 fix: V9.1 replaced absolute consistency fields
    # (prototype_consistency/domain_consistency/physical_consistency) with
    # relative rank-based descriptors.  The old support_text block was not used
    # in the returned SFT target, but it still accessed removed keys and crashed
    # during fresh SFT export.  Keep a compact compatibility summary instead.
    if selected_profile is None:
        support_text = "closed-set relative evidence"
    else:
        support_text = (
            f"relative_support={float(selected_profile.get('relative_support', selected_profile.get('combined_support', 0.0))):.4f}, "
            f"physics_rank={int(selected_profile.get('physics_rank', 0))}, "
            f"advisor_rank={int(selected_profile.get('advisor_score_rank', 0))}, "
            f"small_prob_rank={int(selected_profile.get('small_prob_rank', 0))}"
        )

    fault_name = FAULT_NAMES.get(int(label), str(label))
    if int(label) == 0:
        boundary = "The synchronized current and vibration evidence is closest to the K002 healthy reference."
    elif int(label) == 1:
        boundary = "The evidence supports the KA03 artificial outer-race damage pattern."
    elif int(label) == 2:
        boundary = "The evidence supports localized real outer-race fatigue pitting consistent with KA04."
    elif int(label) == 3:
        boundary = "The evidence supports the KB23 compound inner-plus-outer race pitting pattern."
    elif int(label) == 4:
        boundary = "The evidence supports the distributed KB24 compound inner-plus-outer race damage pattern."
    elif int(label) == 5:
        boundary = "The evidence supports localized real inner-race pitting consistent with KI04."
    elif int(label) == 6:
        boundary = "The evidence supports the more extensive real inner-race pitting pattern of KI16."
    else:
        boundary = f"The evidence is consistent with the {fault_name} pattern in the closed label space."
    # V8.8 keeps SFT targets compact so generation can finish exact JSON.
    if selected_profile is not None:
        support_value = float(selected_profile.get("relative_support", selected_profile.get("combined_support", 0.0)))
        support_phrase = (
            f"relative_support={support_value:.4f}, "
            f"physics_rank={int(selected_profile.get('physics_rank', 0))}, "
            f"advisor_rank={int(selected_profile.get('advisor_score_rank', 0))}"
        )
    else:
        support_phrase = "closed-set relative evidence"
    return (
        f"{dominant} evidence, token_norm={token_norm:.4f}, physical_norm={physical_norm:.4f}, "
        f"and {support_phrase} support {FAULT_NAMES.get(int(label), str(label))}."
    )


@dataclass(frozen=True)
class TrialRecord:
    path: str
    fault_name: str
    label: int
    domain_name: str
    domain: int
    trial_name: str


@dataclass
class TrainConfig:
    data_path: str = r"D:\HM-Shen\1\code\LLM\Datasets\PU"
    epochs: int = 20
    batch_size: int = 64
    lr: float = 5e-4
    router_lr: float = 1e-3
    # Training seed: model initialization, DataLoader shuffle, augmentation, Dropout, Router/LoRA randomness.
    seed: int = 42
    # Data split seed: source sample selection, 70/15/15 membership, and target-test sample selection.
    # Keep this fixed when repeating experiments with different training seeds.
    split_seed: int = 42
    source_domains: Tuple[str, ...] = ("S1", "S3", "S4")
    target_domains: Tuple[str, ...] = ("S2",)
    # Balanced source-domain sample split. For every (fault class, source domain),
    # exactly source_samples_per_class_domain windows are selected and divided into
    # 70% source_train, 15% val_part1, and 15% val_part2.
    source_samples_per_class_domain: int = 1000
    target_samples_per_class_domain: int = 1000
    source_train_ratio: float = 0.70
    val_part1_ratio: float = 0.15
    val_part2_ratio: float = 0.15
    # Fixed-point PU preprocessing for this variant: every domain is segmented
    # directly into 1280-point windows with a 512-point stride. Shaft speed is
    # metadata only and is NOT used for window sizing or resampling.
    window_size: int = 1280
    stride: int = 512
    sampling_rate_hz: int = 64000
    # Retained only for result/config compatibility; fixed-revolution normalization
    # is disabled and not used by the dataset loader in this variant.
    window_revolutions: float = 0.0
    rotation_normalize: bool = False
    input_channels: int = 3
    drop_first_column: bool = False
    channel_indices: Optional[List[int]] = None
    modality_spec: str = "motor_current:0,1;vibration:2"
    window_norm: str = "train_scaler"
    max_windows_per_trial: Optional[int] = 600
    # Split sizes are controlled per class-domain by the fields above.
    feature_dim: int = 96
    branch_hidden: int = 48
    # V7.9: only reduce temporal residual depth from 2 to 1; keep width and frequency branch unchanged.
    tcn_layers: int = 1
    dropout: float = 0.25
    label_smoothing: float = 0.03
    proto_weight: float = 0.10
    supcon_weight: float = 0.05
    confusion_pair_weight: float = 0.15
    supcon_temperature: float = 0.20
    # Source-only domain-invariance regularization. V10.6 uses this value as
    # the maximum coefficient after a classification-first warm-up/ramp.
    domain_align_weight: float = 0.05
    domain_align_warmup_epochs: int = 4
    domain_align_ramp_epochs: int = 6
    domain_mixup_weight: float = 0.0
    domain_mixup_alpha: float = 0.40
    # Test-time averaging over small temporal shifts. Set to (0,) to disable.
    tta_shifts: Tuple[int, ...] = (0,)
    augment: bool = True
    time_shift: int = 16
    noise_std: float = 0.015
    channel_dropout_p: float = 0.03
    amplitude_jitter: float = 0.10
    time_mask_ratio: float = 0.03
    robust_selection: bool = True
    robust_val_time_shift: int = 24
    robust_val_noise_std: float = 0.025
    robust_val_channel_dropout_p: float = 0.08
    robust_val_amplitude_jitter: float = 0.15
    robust_val_time_mask_ratio: float = 0.05
    mc_dropout_samples: int = 8
    router_type: str = "mlp"
    router_epochs: int = 20
    router_threshold: float = 0.50
    # Train the Router on actual small-model mistakes. Strong source-only
    # perturbations are added so a nearly perfect source validation split still
    # contains realistic error examples instead of synthetic upper-tail labels.
    router_target_mode: str = "error"
    router_positive_quantile: float = 0.85
    router_hard_augment_copies: int = 2
    router_hard_time_shift: int = 24
    router_hard_noise_std: float = 0.05
    router_hard_channel_dropout_p: float = 0.20
    router_positive_weight_cap: float = 8.0
    # In evaluation, first apply an absolute Router threshold to decide whether
    # a sample is truly hard enough for LLM delegation. By default, every eligible
    # hard sample is delegated to the LLM, so the number of calls is data-driven
    # rather than fixed at 800 calls or 10%. Set --enable-budget-cap only for
    # controlled efficiency studies when a hard upper bound is desired.
    router_budgeted_routing: bool = True
    # Near-certain small-model predictions are blocked before LLM routing. Set
    # to 1.0 to disable this confidence gate.
    router_confidence_gate: float = 1.0
    router_min_error_recall: float = 0.80
    router_calibration_steps: int = 201
    advisor_prob_weight: float = 1.0
    advisor_proto_weight: float = 0.6
    advisor_domain_weight: float = 0.4
    advisor_physics_weight: float = 0.8
    pairwise_enabled: bool = True
    pairwise_all_pairs: bool = False
    evidence_consensus_enabled: bool = False
    evidence_consensus_min_pair_confidence: float = 0.97
    evidence_consensus_min_risk: float = 0.50
    evidence_consensus_max_small_confidence: float = 0.35
    evidence_consensus_min_score_gain: float = 2.00
    pairwise_epochs: int = 5
    pairwise_lr: float = 1e-3
    # Local LLM is loaded lazily after the lightweight model has finished training,
    # so Gemma does not occupy GPU memory during TCN/Router optimization.
    llm_mode: str = "local_hf"
    llm_model_path: str = r"d:\LLM\Gemma-2-2b-it"
    llm_adapter_path: str = ""
    llm_dtype: str = "auto"
    llm_max_input_tokens: int = 2048
    llm_max_new_tokens: int = 384
    # Optional budget cap. These fields are ignored by default because V7.7 uses
    # dynamic threshold-only delegation. They become active only when
    # --enable-budget-cap is passed.
    max_llm_calls_per_eval: int = 640
    llm_max_call_rate: float = 0.08
    llm_batch_size: int = 8
    # Evidence-aware revision gate. The LLM's self-reported confidence is not
    # treated as a calibrated probability; strong independent consensus can
    # approve a revision even when the generated confidence is conservative.
    llm_revision_min_confidence: float = 0.55
    llm_revision_min_evidence_confidence: float = 0.65
    # V8.6: conservative safety gate to prevent harmful LLM overrides on saturated/easy target domains.
    # These fields protect the final classifier; they do not stop the LLM from diagnosing hard samples.
    llm_revision_protect_small_confidence: float = 0.45
    llm_revision_high_risk: float = 0.15
    llm_revision_min_support: int = 2
    llm_revision_min_score_gain: float = 0.00
    llm_override_min_confidence: float = 0.82
    llm_override_min_evidence_confidence: float = 0.50
    llm_override_max_small_confidence: float = 0.60
    llm_override_min_support: int = 1
    llm_override_min_score_gain: float = -0.25
    llm_normal_flip_min_confidence: float = 0.90
    validation_use_llm: bool = False
    eval_log_interval: int = 20
    advisor_mode: str = "llm_delegate"
    # V8.8: hard samples use Router-only candidate-scoring direct LLM delegation.
    direct_llm_delegate: bool = True
    # V8.8.2: LLM directly classifies Router-selected hard samples by scoring each
    # candidate label. It may replace the small-model label only when the score is
    # strong and independent pre-LLM evidence also supports the proposed class.
    llm_delegate_min_confidence: float = 0.70
    llm_delegate_min_score_margin: float = 0.20
    llm_delegate_min_override_support: int = 2
    llm_delegate_max_small_confidence_for_override: float = 0.95
    llm_delegate_min_risk_margin: float = 0.00
    llm_delegate_min_relative_support: float = 0.50
    llm_delegate_min_relative_support_gain: float = 0.05
    llm_delegate_require_exact_json: bool = False
    # V8.8 feature-to-token alignment for independent LLM reasoning.
    llm_feature_token_count: int = 8
    llm_feature_token_dim: int = 16
    token_align_weight: float = 0.05

    # V10.13 post-diagnosis zero-shot maintenance drafting. The diagnosis LoRA is
    # loaded for routed diagnostic review, then temporarily disabled during the
    # separate class-level maintenance generation because its SFT data contain no
    # maintenance targets.
    llm_generate_maintenance: bool = True

    # Full-model reporting configuration: the diagnosis is frozen after the
    # normal Router/LoRA diagnosis gate. The maintenance stage is report-only
    # and cannot request or apply a second diagnosis pass.
    maintenance_revision_enabled: bool = False

    # The maintenance JSON is concise, but 640 tokens leave enough room for all
    # required Chinese fields and the closing brace. Invalid class reports get
    # one metric-accounted single-item retry; diagnosis generation is unchanged.
    maintenance_max_new_tokens: int = 640

    # Seven PU classes fit into one generation batch on a 24-GB-class GPU in the
    # common setup; the code still falls back batch-wise on generation errors.
    maintenance_batch_size: int = 8


    # Optional metric-only repeats for priority stability. A single deterministic
    # generation is the default, so stability is reported as N/A unless the user
    # explicitly requests two or more generations.
    maintenance_stability_repeats: int = 1
    skip_sft_export_when_reusing_lora: bool = False

    # Automatic source-only LoRA-SFT. val_part1 supplies the reviewer examples;
    # target-domain samples are never used for fine-tuning or threshold selection.
    auto_train_lora: bool = True
    lora_output_dir: str = "dars_llm_lora_adapter_v10_13_full_structured_maintenance"
    lora_epochs: int = 2
    lora_batch_size: int = 2
    lora_grad_accum_steps: int = 8
    lora_lr: float = 2e-4
    lora_weight_decay: float = 0.01
    lora_warmup_ratio: float = 0.05
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_max_seq_length: int = 1024
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_gradient_checkpointing: bool = True
    # V8.9.1: always retrain LoRA from freshly exported SFT data. This field
    # remains only for backward-compatible config loading and is ignored by CLI.
    lora_reuse_existing: bool = False

    # Hard source augmentations create realistic low-confidence/revision examples
    # without leaking any target-domain information into the LoRA training set.
    sft_export_path: str = "dars_llm_sft_train_v10_13_full_structured_maintenance.jsonl"
    sft_max_records: Optional[int] = 2400
    sft_hard_augment_copies: int = 2
    # Counterfactual source-only examples deliberately present a plausible
    # wrong top-1 candidate while retaining the true class in the top-3. They
    # teach the delegated classifier to select the evidence-supported label instead of copying the small-model top-1.
    sft_counterfactual_revise_copies: int = 4
    sft_min_revise_fraction: float = 0.40
    sft_min_revision_confidence: float = 0.60
    checkpoint_path: str = "dars_llm_best_70_15_15_v10_13_full_structured_maintenance.pth"
    results_json_path: str = "dars_llm_results_v10_14_final_clean.json"
    class_maintenance_csv_path: str = "dars_llm_pu_class_maintenance_summary_v10_14_final_clean.csv"
    class_maintenance_text_path: str = "dars_llm_pu_class_maintenance_summary_v10_14_final_clean.txt"
    maintenance_rule_evaluation_csv_path: str = "dars_llm_pu_maintenance_rule_evaluation_v10_14_final_clean.csv"
    maintenance_rule_evaluation_json_path: str = "dars_llm_pu_maintenance_rule_evaluation_v10_14_final_clean.json"
    # Optional: reuse only encoder/predictor weights from a prior V6 checkpoint
    # so baseline and LLM-assisted experiments share exactly the same small model.
    small_model_checkpoint_path: str = ""
    eval_only: bool = False
    eval_test_each_epoch: bool = False
    num_workers: int = 0
    cpu_num_threads: int = 1


def set_seed(seed: int, cpu_num_threads: int = 1) -> None:
    if cpu_num_threads and cpu_num_threads > 0:
        try:
            torch.set_num_threads(cpu_num_threads)
            torch.set_num_interop_threads(max(1, min(cpu_num_threads, 4)))
        except RuntimeError:
            pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def natural_key(text: str) -> List[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def parse_csv_tuple(text: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: Sequence[int],
    n_bins: int = 15,
) -> float:
    if probabilities.size == 0:
        return 0.0
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correctness = (prediction == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right >= 1.0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def label_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels_order: Sequence[int],
) -> Dict[str, object]:
    y_t = list(map(int, y_true))
    y_p = list(map(int, y_pred))
    report = classification_report(
        y_t,
        y_p,
        labels=list(labels_order),
        target_names=[FAULT_NAMES[i] for i in labels_order],
        output_dict=True,
        zero_division=0,
    )
    return {
        "acc": 100.0 * accuracy_score(y_t, y_p),
        "balanced_acc": 100.0 * balanced_accuracy_score(y_t, y_p),
        "macro_precision": 100.0 * precision_score(
            y_t, y_p, average="macro", labels=list(labels_order), zero_division=0
        ),
        "macro_recall": 100.0 * recall_score(
            y_t, y_p, average="macro", labels=list(labels_order), zero_division=0
        ),
        "macro_f1": 100.0 * f1_score(
            y_t, y_p, average="macro", labels=list(labels_order), zero_division=0
        ),
        "weighted_f1": 100.0 * f1_score(
            y_t, y_p, average="weighted", labels=list(labels_order), zero_division=0
        ),
        "mcc": float(matthews_corrcoef(y_t, y_p)),
        "kappa": float(cohen_kappa_score(y_t, y_p, labels=list(labels_order))),
        "classification_report": report,
    }


def index_fingerprint(indices: Sequence[int]) -> str:
    """Return a compact deterministic fingerprint for a selected index set.

    The fingerprint is printed in every run so users can verify that changing
    only ``--seed`` does not alter source/validation/target sample membership.
    """
    arr = np.asarray(indices, dtype=np.int64)
    return f"{zlib.crc32(arr.tobytes()) & 0xFFFFFFFF:08x}"


def json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def parse_channel_indices(text: Optional[str]) -> Optional[List[int]]:
    if text is None or text.strip() == "":
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def none_if_negative(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return None if value < 0 else value


def parse_modality_spec(spec: str, input_channels: int) -> List[Tuple[str, List[int]]]:
    result: List[Tuple[str, List[int]]] = []
    for block in spec.split(";"):
        block = block.strip()
        if not block:
            continue
        if ":" not in block:
            raise ValueError(f"Bad modality block {block}; expected name:0,1")
        name, idx_text = block.split(":", 1)
        idxs = [int(x.strip()) for x in idx_text.split(",") if x.strip()]
        for idx in idxs:
            if idx < 0 or idx >= input_channels:
                raise ValueError(f"Modality index {idx} out of range [0, {input_channels - 1}]")
        result.append((name.strip(), idxs))
    return result or [("all", list(range(input_channels)))]


def _resolve_child_dir(parent: str, expected_name: str) -> str:
    """Resolve a child directory case-insensitively while preferring exact names."""
    exact = os.path.join(parent, expected_name)
    if os.path.isdir(exact):
        return exact
    if not os.path.isdir(parent):
        return exact
    expected_lower = expected_name.lower()
    for child in os.listdir(parent):
        candidate = os.path.join(parent, child)
        if os.path.isdir(candidate) and child.lower() == expected_lower:
            return candidate
    return exact


def discover_trials(data_path: str, domains: Sequence[str]) -> List[TrialRecord]:
    """Discover PU MAT files under root/bearing_code/operating_condition/.

    Example:
        PU/K002/S1/N15_M07_F10_K002_1.mat
        PU/KA03/S4/N15_M07_F04_KA03_1.mat

    All MAT files in each class-domain folder are loaded. The official PU data
    normally provide 20 four-second measurements for every bearing/setting.
    """
    records: List[TrialRecord] = []
    for fault_name, label in FAULT_TYPES.items():
        fault_folder = _resolve_child_dir(data_path, fault_name)
        for domain_name in domains:
            folder = _resolve_child_dir(fault_folder, domain_name)
            if not os.path.isdir(folder):
                print(f"Warning: missing folder {folder}")
                continue
            data_files = sorted(
                [f for f in os.listdir(folder) if f.lower().endswith(".mat")],
                key=natural_key,
            )
            if not data_files:
                print(f"Warning: no PU MAT file found in {folder}")
                continue
            print(
                f">>> PU FILES: class={fault_name}; domain={domain_name}; "
                f"mat_files={len(data_files)}"
            )
            for data_file in data_files:
                records.append(TrialRecord(
                    path=os.path.join(folder, data_file),
                    fault_name=fault_name,
                    label=label,
                    domain_name=domain_name,
                    domain=DOMAIN_TO_ID[domain_name],
                    trial_name=os.path.splitext(data_file)[0],
                ))
    return records


def _validate_split_ratios(train_ratio: float, val_part1_ratio: float, val_part2_ratio: float) -> None:
    ratios = [float(train_ratio), float(val_part1_ratio), float(val_part2_ratio)]
    if any(r <= 0.0 for r in ratios):
        raise ValueError(f"All source split ratios must be positive, got {ratios}.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Source split ratios must sum to 1.0, got {ratios} (sum={sum(ratios):.8f}).")


def _split_count(total: int, train_ratio: float, val_part1_ratio: float) -> Tuple[int, int, int]:
    """Return exact integer counts; the remainder is assigned to val_part2."""
    n_train = int(round(total * float(train_ratio)))
    n_val1 = int(round(total * float(val_part1_ratio)))
    n_val2 = int(total - n_train - n_val1)
    if min(n_train, n_val1, n_val2) <= 0:
        raise ValueError(
            f"Invalid split counts for total={total}: train={n_train}, "
            f"val_part1={n_val1}, val_part2={n_val2}."
        )
    return n_train, n_val1, n_val2


def balanced_source_sample_split(
    labels: np.ndarray,
    domains: np.ndarray,
    source_domains: Sequence[str],
    samples_per_class_domain: int,
    train_ratio: float,
    val_part1_ratio: float,
    val_part2_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Select and split exactly N windows for every class-source-domain pair.

    This deliberately implements the simplified experimental protocol requested
    for PU: 1000 windows per (bearing code, source operating condition), divided into
    700/150/150 when the default 70/15/15 ratios are used.
    """
    _validate_split_ratios(train_ratio, val_part1_ratio, val_part2_ratio)
    total_per_group = int(samples_per_class_domain)
    if total_per_group <= 0:
        raise ValueError("source_samples_per_class_domain must be positive.")
    n_train, n_val1, n_val2 = _split_count(total_per_group, train_ratio, val_part1_ratio)

    split_indices: Dict[str, List[int]] = {
        "source_train": [],
        "val_part1": [],
        "val_part2": [],
    }
    for domain_name in source_domains:
        if domain_name not in DOMAIN_TO_ID:
            raise ValueError(f"Unknown source domain: {domain_name}")
        domain_id = DOMAIN_TO_ID[domain_name]
        for label in sorted(FAULT_NAMES):
            idx = np.where((labels == label) & (domains == domain_id))[0]
            if len(idx) < total_per_group:
                raise ValueError(
                    f"Class-domain {FAULT_NAMES[label]} / {domain_name} has {len(idx)} windows, "
                    f"but {total_per_group} are required. Reduce --source-samples-per-class-domain "
                    "or create more windows."
                )
            group_offset = zlib.crc32(f"source|{label}|{domain_name}".encode("utf-8"))
            rng = np.random.default_rng(int(seed) + int(group_offset))
            chosen = rng.choice(idx, size=total_per_group, replace=False)
            rng.shuffle(chosen)
            split_indices["source_train"].extend(chosen[:n_train].tolist())
            split_indices["val_part1"].extend(chosen[n_train:n_train + n_val1].tolist())
            split_indices["val_part2"].extend(chosen[n_train + n_val1:].tolist())

    return {name: np.asarray(sorted(values), dtype=np.int64) for name, values in split_indices.items()}


def balanced_target_sample_selection(
    labels: np.ndarray,
    domains: np.ndarray,
    target_domains: Sequence[str],
    samples_per_class_domain: int,
    seed: int,
) -> np.ndarray:
    """Select exactly N target windows for every class-target-domain pair."""
    total_per_group = int(samples_per_class_domain)
    if total_per_group <= 0:
        raise ValueError("target_samples_per_class_domain must be positive.")
    selected: List[int] = []
    for domain_name in target_domains:
        if domain_name not in DOMAIN_TO_ID:
            raise ValueError(f"Unknown target domain: {domain_name}")
        domain_id = DOMAIN_TO_ID[domain_name]
        for label in sorted(FAULT_NAMES):
            idx = np.where((labels == label) & (domains == domain_id))[0]
            if len(idx) < total_per_group:
                raise ValueError(
                    f"Class-domain {FAULT_NAMES[label]} / {domain_name} has {len(idx)} windows, "
                    f"but {total_per_group} are required. Reduce --target-samples-per-class-domain "
                    "or create more windows."
                )
            group_offset = zlib.crc32(f"target|{label}|{domain_name}".encode("utf-8"))
            rng = np.random.default_rng(int(seed) + int(group_offset))
            chosen = rng.choice(idx, size=total_per_group, replace=False)
            selected.extend(chosen.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


_PU_AUDITED_FOLDERS: set[str] = set()


class PUTrialWindowDataset(Dataset):
    def __init__(
        self,
        records: Sequence[TrialRecord],
        window_size: int,
        stride: int,
        input_channels: int,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        window_norm: str = "sample",
        max_windows_per_trial: Optional[int] = None,
        samples_per_class: Optional[int] = None,
        seed: int = 42,
        drop_first_column: bool = False,
        channel_indices: Optional[Sequence[int]] = None,
        sampling_rate_hz: int = 64000,
        window_revolutions: float = 0.0,
        rotation_normalize: bool = False,
    ):
        self.records = list(records)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.sampling_rate_hz = int(sampling_rate_hz)
        self.window_revolutions = 0.0
        self.rotation_normalize = False
        self.input_channels = int(input_channels)
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.window_norm = window_norm
        self.max_windows_per_trial = max_windows_per_trial
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.drop_first_column = drop_first_column
        self.channel_indices = list(channel_indices) if channel_indices is not None else None
        self.data, self.labels, self.domains, self.trials = self._load_windows()
        if len(self.data) == 0:
            raise ValueError("No PU windows created. Check MAT folders, Name/Data fields, selected signals, window, and stride settings.")
        self._select_samples_per_class()
        self._normalize(fit_scaler)
        self.data = self.data.reshape(len(self.data), -1).astype(np.float32)
        self.labels = self.labels.astype(np.int64)
        self.domains = self.domains.astype(np.int64)

    @staticmethod
    def _mat_text(value: object) -> str:
        """Convert MATLAB char/cell/string representations to plain text."""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return PUTrialWindowDataset._mat_text(value.reshape(-1)[0])
            if value.dtype.kind in {"U", "S"}:
                return "".join(str(x) for x in value.reshape(-1))
        return str(value)

    @staticmethod
    def _numeric_vector(value: object) -> Optional[np.ndarray]:
        """Extract a one-dimensional numeric signal from a MATLAB object."""
        try:
            arr = np.asarray(value)
        except Exception:
            return None
        while arr.dtype == object and arr.size == 1:
            arr = np.asarray(arr.reshape(-1)[0])
        if arr.dtype == object or arr.dtype.names is not None:
            return None
        if not np.issubdtype(arr.dtype, np.number):
            return None
        arr = np.asarray(arr, dtype=np.float64).squeeze()
        if arr.ndim == 0:
            return None
        arr = arr.reshape(-1)
        return arr if arr.size > 1 else None

    @classmethod
    def _collect_named_signals(
        cls,
        obj: object,
        output: Dict[str, np.ndarray],
        depth: int = 0,
    ) -> None:
        """Recursively collect PU ``Name``/``Data`` signal pairs from loadmat output."""
        if obj is None or depth > 20:
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                if not str(key).startswith("__"):
                    cls._collect_named_signals(value, output, depth + 1)
            return

        field_names = getattr(obj, "_fieldnames", None)
        if field_names:
            lower = {str(name).lower(): str(name) for name in field_names}
            if "name" in lower and "data" in lower:
                signal_name = cls._mat_text(getattr(obj, lower["name"])).strip()
                signal_data = cls._numeric_vector(getattr(obj, lower["data"]))
                if signal_name and signal_data is not None:
                    output[signal_name] = signal_data
            for field in field_names:
                cls._collect_named_signals(getattr(obj, field), output, depth + 1)
            return

        if isinstance(obj, np.void) and obj.dtype.names:
            lower = {str(name).lower(): str(name) for name in obj.dtype.names}
            if "name" in lower and "data" in lower:
                signal_name = cls._mat_text(obj[lower["name"]]).strip()
                signal_data = cls._numeric_vector(obj[lower["data"]])
                if signal_name and signal_data is not None:
                    output[signal_name] = signal_data
            for field in obj.dtype.names:
                cls._collect_named_signals(obj[field], output, depth + 1)
            return

        if isinstance(obj, np.ndarray):
            if obj.dtype.names:
                for item in obj.reshape(-1):
                    cls._collect_named_signals(item, output, depth + 1)
            elif obj.dtype == object:
                for item in obj.reshape(-1):
                    cls._collect_named_signals(item, output, depth + 1)

    @staticmethod
    def _normalize_signal_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

    @classmethod
    def _select_named_signal(
        cls,
        signals: Dict[str, np.ndarray],
        aliases: Sequence[str],
    ) -> Tuple[Optional[np.ndarray], str]:
        normalized = {cls._normalize_signal_name(name): (name, values) for name, values in signals.items()}
        normalized_aliases = [cls._normalize_signal_name(alias) for alias in aliases]
        for alias in normalized_aliases:
            if alias in normalized:
                original, values = normalized[alias]
                return values, original
        for alias in normalized_aliases:
            for key, (original, values) in normalized.items():
                if alias and (alias in key or key in alias):
                    return values, original
        return None, ""

    @classmethod
    def _read_pu_mat(cls, path: str) -> Tuple[np.ndarray, Dict[str, str]]:
        """Read synchronized two-phase current and vibration signals from a PU MAT file."""
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        named: Dict[str, np.ndarray] = {}
        cls._collect_named_signals(mat, named)
        if not named:
            raise ValueError("No MATLAB Name/Data signal pairs were found")

        current_1, current_1_name = cls._select_named_signal(
            named, ("phase_current_1", "phase current 1", "current_1", "current1")
        )
        current_2, current_2_name = cls._select_named_signal(
            named, ("phase_current_2", "phase current 2", "current_2", "current2")
        )
        vibration, vibration_name = cls._select_named_signal(
            named, ("vibration_1", "vibration 1", "vibration", "acceleration")
        )
        missing = []
        if current_1 is None:
            missing.append("phase_current_1")
        if current_2 is None:
            missing.append("phase_current_2")
        if vibration is None:
            missing.append("vibration_1")
        if missing:
            available = ", ".join(sorted(named.keys()))
            raise ValueError(f"Missing {missing}; available PU signals: {available}")

        min_length = min(len(current_1), len(current_2), len(vibration))
        if min_length <= 0:
            raise ValueError("Selected PU signals are empty")
        values = np.column_stack([
            np.asarray(current_1[:min_length], dtype=np.float32),
            np.asarray(current_2[:min_length], dtype=np.float32),
            np.asarray(vibration[:min_length], dtype=np.float32),
        ])
        finite_rows = np.isfinite(values).all(axis=1)
        values = values[finite_rows]
        selected_names = {
            "current_1": current_1_name,
            "current_2": current_2_name,
            "vibration": vibration_name,
        }
        return values, selected_names

    def _read_signals(self, path: str) -> Optional[np.ndarray]:
        try:
            values, selected_names = self._read_pu_mat(path)
        except Exception as exc:
            print(f"Warning: failed to load PU MAT file {path}: {exc}")
            return None

        if values.ndim != 2 or values.shape[0] <= 1:
            print(f"Warning: skip {path}, invalid PU signal shape {getattr(values, 'shape', None)}")
            return None

        # Channel order before optional selection:
        #   0 = phase_current_1, 1 = phase_current_2, 2 = vibration_1.
        # Speed, torque, radial force, and temperature are deliberately excluded.
        if self.channel_indices is not None:
            if not self.channel_indices or max(self.channel_indices) >= values.shape[1]:
                print(f"Warning: skip {path}, PU channel index exceeds {values.shape[1]}")
                return None
            values = values[:, self.channel_indices]
        elif values.shape[1] >= self.input_channels:
            values = values[:, :self.input_channels]
        else:
            print(f"Warning: skip {path}, got only {values.shape[1]} PU channels")
            return None

        if values.shape[1] != self.input_channels:
            print(
                f"Warning: skip {path}, selected {values.shape[1]} channels but "
                f"input_channels={self.input_channels}"
            )
            return None

        audit_key = str(Path(path).parent.resolve())
        if audit_key not in _PU_AUDITED_FOLDERS:
            _PU_AUDITED_FOLDERS.add(audit_key)
            raw_indices = self.channel_indices if self.channel_indices is not None else list(range(self.input_channels))
            raw_labels = {0: "I1", 1: "I2", 2: "Vib"}
            active_labels = [raw_labels.get(i, f"raw_{i}") for i in raw_indices]
            print(
                f"Loaded PU MAT: {os.path.basename(path)} | rows={values.shape[0]} | "
                f"active_channels={active_labels} | MAT names={selected_names} | "
                f"output_window={self.window_size}, output_stride={self.stride}, "
                f"rotation_normalize={self.rotation_normalize}"
            )
        return np.ascontiguousarray(values, dtype=np.float32)

    def _load_windows(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        windows, labels, domains, trials = [], [], [], []
        audited_domains: set[str] = set()
        for record in self.records:
            signals = self._read_signals(record.path)
            if signals is None:
                continue

            # Direct point-based sliding window. No fixed-revolution slicing and no
            # speed-dependent resampling are used in this PU-1280 variant.
            raw_window_size, raw_stride = self.window_size, self.stride

            if record.domain_name not in audited_domains:
                audited_domains.add(record.domain_name)
                rpm = DOMAIN_METADATA.get(record.domain_name, {}).get("speed_rpm", "unknown")
                print(
                    f">>> FIXED WINDOW: domain={record.domain_name}; speed={rpm} rpm (metadata only); "
                    f"rotation_normalize=False; window={raw_window_size}; stride={raw_stride}; "
                    f"network_window={self.window_size}"
                )

            max_start = len(signals) - raw_window_size
            if max_start < 0:
                print(
                    f"Warning: skip {record.path}, only {len(signals)} rows but "
                    f"raw_window={raw_window_size} is required."
                )
                continue

            for count, start in enumerate(range(0, max_start + 1, raw_stride)):
                if self.max_windows_per_trial is not None and count >= self.max_windows_per_trial:
                    break
                segment = signals[start : start + raw_window_size]
                windows.append(np.asarray(segment, dtype=np.float32))
                labels.append(record.label)
                domains.append(record.domain)
                trials.append(record.trial_name)
        return np.asarray(windows), np.asarray(labels), np.asarray(domains), trials

    def _select_samples_per_class(self) -> None:
        if self.samples_per_class is None:
            return
        rng = np.random.default_rng(self.seed)
        selected: List[int] = []
        for label in sorted(FAULT_NAMES):
            idx = np.where(self.labels == label)[0]
            if len(idx) < self.samples_per_class:
                raise ValueError(
                    f"Class {label} has {len(idx)} windows, need {self.samples_per_class}. "
                    "Use --train-samples-per-class -1 etc. to keep all windows."
                )
            selected.extend(rng.choice(idx, size=self.samples_per_class, replace=False).tolist())
        selected = sorted(selected)
        self.data = self.data[selected]
        self.labels = self.labels[selected]
        self.domains = self.domains[selected]
        self.trials = [self.trials[i] for i in selected]

    def _normalize(self, fit_scaler: bool) -> None:
        if self.window_norm == "sample":
            mean = self.data.mean(axis=1, keepdims=True)
            std = self.data.std(axis=1, keepdims=True)
            self.data = (self.data - mean) / (std + 1e-6)
        elif self.window_norm == "train_scaler":
            flat = self.data.reshape(-1, self.data.shape[-1])
            if fit_scaler:
                self.scaler.fit(flat)
            self.data = self.scaler.transform(flat).reshape(self.data.shape)
        elif self.window_norm == "none":
            pass
        else:
            raise ValueError("window_norm must be sample, train_scaler or none")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.domains[idx], dtype=torch.long),
        )



class PUArrayWindowDataset(Dataset):
    """Dataset backed by selected window arrays from a shared raw source pool."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        domains: np.ndarray,
        trials: Sequence[str],
        window_size: int,
        input_channels: int,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        window_norm: str = "sample",
    ):
        self.window_size = int(window_size)
        self.input_channels = int(input_channels)
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.data = np.asarray(data, dtype=np.float32).reshape(-1, self.window_size, self.input_channels).copy()
        self.labels = np.asarray(labels, dtype=np.int64).copy()
        self.domains = np.asarray(domains, dtype=np.int64).copy()
        self.trials = list(trials)
        self._normalize(window_norm, fit_scaler)
        self.data = self.data.reshape(len(self.data), -1).astype(np.float32)

    def _normalize(self, window_norm: str, fit_scaler: bool) -> None:
        if window_norm == "sample":
            mean = self.data.mean(axis=1, keepdims=True)
            std = self.data.std(axis=1, keepdims=True)
            self.data = (self.data - mean) / (std + 1e-6)
        elif window_norm == "train_scaler":
            flat = self.data.reshape(-1, self.data.shape[-1])
            if fit_scaler:
                self.scaler.fit(flat)
            if not hasattr(self.scaler, "mean_"):
                raise RuntimeError("Training scaler has not been fitted.")
            self.data = self.scaler.transform(flat).reshape(self.data.shape)
        elif window_norm == "none":
            pass
        else:
            raise ValueError("window_norm must be sample, train_scaler or none")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.domains[idx], dtype=torch.long),
        )


def subset_array_dataset(
    pool: PUTrialWindowDataset,
    indices: np.ndarray,
    window_size: int,
    input_channels: int,
    scaler: StandardScaler,
    fit_scaler: bool,
    window_norm: str,
) -> PUArrayWindowDataset:
    idx = np.asarray(indices, dtype=np.int64)
    return PUArrayWindowDataset(
        data=pool.data[idx],
        labels=pool.labels[idx],
        domains=pool.domains[idx],
        trials=[pool.trials[int(i)] for i in idx],
        window_size=window_size,
        input_channels=input_channels,
        scaler=scaler,
        fit_scaler=fit_scaler,
        window_norm=window_norm,
    )


def print_sample_split_info(name: str, dataset: Dataset) -> None:
    labels = np.asarray(dataset.labels)
    domains = np.asarray(dataset.domains)
    print(f"\n[{name}]")
    print(f"  windows: {len(dataset)}")
    print(f"  windows per class: {dict((FAULT_NAMES[i], int((labels == i).sum())) for i in sorted(FAULT_NAMES))}")
    print(f"  windows per domain: {dict((ID_TO_DOMAIN[d], int((domains == d).sum())) for d in sorted(set(domains.tolist())))}")
    class_domain = {}
    for d in sorted(set(domains.tolist())):
        for c in sorted(FAULT_NAMES):
            count = int(((domains == d) & (labels == c)).sum())
            if count:
                class_domain[f"{ID_TO_DOMAIN[d]}-{FAULT_TYPES and list(FAULT_TYPES.keys())[list(FAULT_TYPES.values()).index(c)]}"] = count
    print(f"  windows per class-domain: {class_domain}")

def augment_batch(
    x_flat: torch.Tensor,
    window_size: int,
    input_channels: int,
    time_shift: int,
    noise_std: float,
    channel_dropout_p: float,
    amplitude_jitter: float = 0.0,
    time_mask_ratio: float = 0.0,
) -> torch.Tensor:
    """Source-only signal augmentation for cross-condition robustness.

    In addition to the original shift/noise/channel dropout, V7.3 uses mild
    per-channel gain jitter and a short contiguous time mask. These operations
    imitate sensor gain and local acquisition disturbances without using target data.
    """
    x = x_flat.view(x_flat.size(0), window_size, input_channels)
    if time_shift > 0:
        shifts = torch.randint(-time_shift, time_shift + 1, (x.size(0),), device=x.device)
        x = torch.stack([torch.roll(x[i], int(shifts[i].item()), dims=0) for i in range(x.size(0))], dim=0)
    if amplitude_jitter > 0:
        gain = 1.0 + (2.0 * torch.rand(x.size(0), 1, input_channels, device=x.device) - 1.0) * amplitude_jitter
        x = x * gain
    if noise_std > 0:
        # Scale noise by each channel's current standard deviation so one modality
        # does not receive disproportionately large perturbations.
        scale = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-3)
        x = x + torch.randn_like(x) * noise_std * scale
    if time_mask_ratio > 0:
        mask_len = max(1, int(round(window_size * float(time_mask_ratio))))
        if mask_len < window_size:
            starts = torch.randint(0, window_size - mask_len + 1, (x.size(0),), device=x.device)
            for i, s in enumerate(starts.tolist()):
                x[i, s:s + mask_len, :] = 0.0
    if channel_dropout_p > 0:
        mask = (torch.rand(x.size(0), 1, input_channels, device=x.device) > channel_dropout_p).float()
        # Keep at least one channel for every sample.
        empty = mask.sum(dim=2, keepdim=True) == 0
        if empty.any():
            keep_idx = torch.randint(0, input_channels, (x.size(0),), device=x.device)
            for i in torch.where(empty.view(-1))[0].tolist():
                mask[i, 0, keep_idx[i]] = 1.0
        x = x * mask
    return x.reshape(x_flat.size(0), -1)


class ModalityTCNBranch(nn.Module):
    """V8.6 ultra-light time-frequency branch.

    Compared with V8.0, this branch removes the second temporal convolution and
    uses a single compact spectral convolution. It keeps the same constructor,
    output dimension and multimodal fusion interface, so the full experimental
    pipeline remains unchanged while the small encoder is intentionally less
    saturated. This makes the selective LLM branch easier to evaluate fairly.
    """
    def __init__(self, in_channels: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.in_channels = int(in_channels)
        self.time_net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=9, stride=4, padding=4, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.freq_net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _avg_max_pool(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)

    def forward(self, x_global: torch.Tensor) -> torch.Tensor:
        mean = x_global.mean(dim=-1, keepdim=True)
        std = x_global.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        x_instance = (x_global - mean) / std

        # Time morphology: one shallow convolutional view.
        time_h = self.time_net(x_instance)
        time_vec = self._avg_max_pool(time_h)

        # Frequency morphology: compact log-FFT magnitude view.
        freq = torch.log1p(torch.fft.rfft(x_instance, dim=-1).abs())
        f_mean = freq.mean(dim=-1, keepdim=True)
        f_std = freq.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        freq = (freq - f_mean) / f_std
        freq_h = self.freq_net(freq)
        freq_vec = self._avg_max_pool(freq_h)
        return self.out(torch.cat([time_vec, freq_vec], dim=1))

class GatedFusion(nn.Module):
    def __init__(self, hidden: int, feature_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, max(16, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden // 2), 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, hs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.stack(hs, dim=1)
        weights = torch.softmax(self.gate(h).squeeze(-1), dim=1)
        fused = (h * weights.unsqueeze(-1)).sum(dim=1)
        return self.proj(fused), weights


class MultimodalTCNEncoder(nn.Module):
    def __init__(self, modalities: List[Tuple[str, List[int]]], window_size: int, input_channels: int, hidden: int, feature_dim: int, tcn_layers: int, dropout: float):
        super().__init__()
        self.modalities = modalities
        self.window_size = window_size
        self.input_channels = input_channels
        self.branches = nn.ModuleDict({
            name: ModalityTCNBranch(len(indices), hidden, tcn_layers, dropout)
            for name, indices in modalities
        })
        self.fusion = GatedFusion(hidden, feature_dim, dropout)

    def forward(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x_flat.view(x_flat.size(0), self.window_size, self.input_channels).transpose(1, 2)
        hs = []
        for name, indices in self.modalities:
            idx = torch.as_tensor(indices, device=x.device, dtype=torch.long)
            hs.append(self.branches[name](x.index_select(1, idx)))
        z, weights = self.fusion(hs)
        return F.normalize(z, dim=1), weights


class SmallPredictor(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, dropout: float):
        super().__init__()
        hidden = max(128, feature_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PrototypeMemory:
    def __init__(self, num_classes: int, num_domains: int, feature_dim: int):
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.feature_dim = feature_dim
        self.physical_dim = 0
        self.class_prototypes = torch.zeros(num_classes, feature_dim)
        self.class_counts = torch.zeros(num_classes)
        self.domain_class_prototypes = torch.zeros(num_domains, num_classes, feature_dim)
        self.domain_class_counts = torch.zeros(num_domains, num_classes)
        self.class_proto_dist_mean = torch.zeros(num_classes)
        self.class_proto_dist_std = torch.ones(num_classes)
        self.class_domain_dist_mean = torch.zeros(num_classes)
        self.class_domain_dist_std = torch.ones(num_classes)
        self.physical_mean = torch.zeros(num_classes, 1)
        self.physical_std = torch.ones(num_classes, 1)
        self.class_physical_dist_mean = torch.zeros(num_classes)
        self.class_physical_dist_std = torch.ones(num_classes)

    def fit(self, features: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor, physical: Optional[torch.Tensor] = None) -> None:
        features = F.normalize(features.detach().cpu(), dim=1)
        labels = labels.detach().cpu()
        domains = domains.detach().cpu()
        physical = physical.detach().cpu() if physical is not None else None
        if physical is not None and physical.size(1) != self.physical_dim:
            self.physical_dim = int(physical.size(1))
            self.physical_mean = torch.zeros(self.num_classes, self.physical_dim)
            self.physical_std = torch.ones(self.num_classes, self.physical_dim)
        self.class_prototypes.zero_()
        self.class_counts.zero_()
        self.domain_class_prototypes.zero_()
        self.domain_class_counts.zero_()
        self.class_proto_dist_mean.zero_()
        self.class_proto_dist_std.fill_(1.0)
        self.class_domain_dist_mean.zero_()
        self.class_domain_dist_std.fill_(1.0)
        self.class_physical_dist_mean.zero_()
        self.class_physical_dist_std.fill_(1.0)
        for c in range(self.num_classes):
            cmask = labels == c
            if cmask.any():
                proto = features[cmask].mean(dim=0)
                self.class_prototypes[c] = F.normalize(proto, dim=0)
                self.class_counts[c] = int(cmask.sum().item())
                own_dist = 1.0 - (features[cmask] @ self.class_prototypes[c])
                self.class_proto_dist_mean[c] = own_dist.mean()
                self.class_proto_dist_std[c] = own_dist.std(unbiased=False).clamp_min(1e-4)
                if physical is not None:
                    pc = physical[cmask]
                    self.physical_mean[c] = pc.mean(dim=0)
                    self.physical_std[c] = pc.std(dim=0, unbiased=False).clamp_min(1e-4)
                    pd = (((pc - self.physical_mean[c]) / self.physical_std[c]) ** 2).mean(dim=1).sqrt()
                    self.class_physical_dist_mean[c] = pd.mean()
                    self.class_physical_dist_std[c] = pd.std(unbiased=False).clamp_min(1e-4)
            for d in range(self.num_domains):
                dmask = cmask & (domains == d)
                if dmask.any():
                    proto = features[dmask].mean(dim=0)
                    self.domain_class_prototypes[d, c] = F.normalize(proto, dim=0)
                    self.domain_class_counts[d, c] = int(dmask.sum().item())
            if cmask.any():
                domain_own = []
                for i in torch.where(cmask)[0]:
                    valid_domains = self.domain_class_counts[:, c] > 0
                    if valid_domains.any():
                        sims = self.domain_class_prototypes[valid_domains, c] @ features[i]
                        domain_own.append(1.0 - sims.max())
                if domain_own:
                    dd = torch.stack(domain_own)
                    self.class_domain_dist_mean[c] = dd.mean()
                    self.class_domain_dist_std[c] = dd.std(unbiased=False).clamp_min(1e-4)

    def to(self, device: torch.device):
        return (
            self.class_prototypes.to(device),
            self.class_counts.to(device),
            self.domain_class_prototypes.to(device),
            self.domain_class_counts.to(device),
        )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "class_prototypes": self.class_prototypes.clone(),
            "class_counts": self.class_counts.clone(),
            "domain_class_prototypes": self.domain_class_prototypes.clone(),
            "domain_class_counts": self.domain_class_counts.clone(),
            "class_proto_dist_mean": self.class_proto_dist_mean.clone(),
            "class_proto_dist_std": self.class_proto_dist_std.clone(),
            "class_domain_dist_mean": self.class_domain_dist_mean.clone(),
            "class_domain_dist_std": self.class_domain_dist_std.clone(),
            "physical_mean": self.physical_mean.clone(),
            "physical_std": self.physical_std.clone(),
            "class_physical_dist_mean": self.class_physical_dist_mean.clone(),
            "class_physical_dist_std": self.class_physical_dist_std.clone(),
        }


def prototype_scores(z: torch.Tensor, pred: torch.Tensor, memory: PrototypeMemory) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    class_proto, class_counts, domain_proto, domain_counts = memory.to(z.device)
    valid_class = class_counts > 0
    sim_all = z @ F.normalize(class_proto, dim=1).t()
    sim_all[:, ~valid_class] = -1e6
    nearest_sim, nearest_class = sim_all.max(dim=1)
    pred_sim = sim_all.gather(1, pred.view(-1, 1)).squeeze(1)
    pred_sim = torch.where(valid_class[pred], pred_sim, nearest_sim)
    proto_dist = 1.0 - pred_sim
    domain_dists = []
    for i in range(z.size(0)):
        c = int(pred[i].item())
        valid_domains = domain_counts[:, c] > 0
        if valid_domains.any():
            sims = domain_proto[valid_domains, c] @ z[i]
            domain_dists.append(1.0 - sims.max())
        else:
            domain_dists.append(proto_dist[i])
    domain_shift = torch.stack(domain_dists)
    nearest_dist = 1.0 - nearest_sim
    return proto_dist, domain_shift, nearest_class, nearest_dist


def extract_physical_features(
    x_flat: torch.Tensor,
    window_size: int,
    input_channels: int,
    modalities: List[Tuple[str, List[int]]],
) -> torch.Tensor:
    x = x_flat.view(x_flat.size(0), window_size, input_channels)
    feats = []
    eps = 1e-6
    for _name, indices in modalities:
        idx = torch.tensor(indices, device=x.device, dtype=torch.long)
        xm = x.index_select(2, idx)
        flat = xm.reshape(xm.size(0), -1)
        mean = flat.mean(dim=1)
        std = flat.std(dim=1, unbiased=False).clamp_min(eps)
        centered = flat - mean.unsqueeze(1)
        rms = torch.sqrt((flat ** 2).mean(dim=1).clamp_min(eps))
        mean_abs = flat.abs().mean(dim=1)
        ptp = flat.max(dim=1).values - flat.min(dim=1).values
        kurt = (centered ** 4).mean(dim=1) / (std ** 4 + eps)
        skew = (centered ** 3).mean(dim=1) / (std ** 3 + eps)
        crest = flat.abs().max(dim=1).values / (rms + eps)
        spec = torch.fft.rfft(flat, dim=1).abs()
        power = spec ** 2
        prob = power / power.sum(dim=1, keepdim=True).clamp_min(eps)
        spectral_entropy = -(prob * torch.log(prob.clamp_min(eps))).sum(dim=1) / math.log(prob.size(1))
        split = max(1, prob.size(1) // 3)
        low_ratio = prob[:, :split].sum(dim=1)
        high_ratio = prob[:, -split:].sum(dim=1)
        feats.extend([mean_abs, rms, std, ptp, kurt, skew, crest, spectral_entropy, low_ratio, high_ratio])
    return torch.stack(feats, dim=1)


def candidate_evidence_scores(
    z_i: torch.Tensor,
    phys_i: torch.Tensor,
    candidate_labels: List[int],
    candidate_probs: List[float],
    memory: PrototypeMemory,
    config: TrainConfig,
) -> Tuple[int, List[Dict[str, float]]]:
    class_proto, class_counts, domain_proto, domain_counts = memory.to(z_i.device)
    details: List[Dict[str, float]] = []
    scores = []
    for c, p in zip(candidate_labels, candidate_probs):
        if class_counts[c] <= 0:
            proto_z = domain_z = phys_z = 10.0
        else:
            proto_dist = 1.0 - torch.dot(z_i, F.normalize(class_proto[c], dim=0))
            proto_z = (proto_dist - memory.class_proto_dist_mean[c].to(z_i.device)) / memory.class_proto_dist_std[c].to(z_i.device)
            valid_domains = domain_counts[:, c] > 0
            if valid_domains.any():
                sims = domain_proto[valid_domains, c] @ z_i
                domain_dist = 1.0 - sims.max()
            else:
                domain_dist = proto_dist
            domain_z = (domain_dist - memory.class_domain_dist_mean[c].to(z_i.device)) / memory.class_domain_dist_std[c].to(z_i.device)
            if memory.physical_dim > 0 and phys_i.numel() == memory.physical_dim:
                pm = memory.physical_mean[c].to(z_i.device)
                ps = memory.physical_std[c].to(z_i.device)
                phys_dist = (((phys_i - pm) / ps) ** 2).mean().sqrt()
                phys_z = (phys_dist - memory.class_physical_dist_mean[c].to(z_i.device)) / memory.class_physical_dist_std[c].to(z_i.device)
            else:
                phys_z = torch.tensor(0.0, device=z_i.device)
        score = (
            config.advisor_prob_weight * math.log(max(float(p), 1e-8))
            - config.advisor_proto_weight * float(proto_z.clamp(-5, 5).item())
            - config.advisor_domain_weight * float(domain_z.clamp(-5, 5).item())
            - config.advisor_physics_weight * float(phys_z.clamp(-5, 5).item())
        )
        scores.append(score)
        details.append({
            "label_id": float(c),
            "prob": float(p),
            "score": float(score),
            "proto_z": float(proto_z.clamp(-99, 99).item()),
            "domain_z": float(domain_z.clamp(-99, 99).item()),
            "physics_z": float(phys_z.clamp(-99, 99).item()),
        })
    best = int(candidate_labels[int(np.argmax(scores))])
    return best, details


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if features.size(0) < 2:
        return torch.tensor(0.0, device=features.device)
    z = F.normalize(features, dim=1)
    logits = z @ z.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    label_mask = labels.view(-1, 1).eq(labels.view(1, -1)).float().to(features.device)
    logits_mask = torch.ones_like(label_mask) - torch.eye(label_mask.size(0), device=features.device)
    label_mask = label_mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positives = label_mask.sum(dim=1)
    valid = positives > 0
    if not valid.any():
        return torch.tensor(0.0, device=features.device)
    return -((label_mask * log_prob).sum(dim=1)[valid] / positives[valid]).mean()


def prototype_loss(features: torch.Tensor, labels: torch.Tensor, memory: PrototypeMemory) -> torch.Tensor:
    class_proto, class_counts, _, _ = memory.to(features.device)
    valid = class_counts[labels] > 0
    if not valid.any():
        return torch.tensor(0.0, device=features.device)
    target_proto = F.normalize(class_proto[labels[valid]], dim=1)
    z = F.normalize(features[valid], dim=1)
    return (1.0 - (z * target_proto).sum(dim=1)).mean()


def progressive_domain_align_weight(config: TrainConfig, epoch_index: int) -> float:
    """Classification-first source-only alignment schedule retained in V10.7.

    For the default 20 epochs: epochs 1-4 use 0; epochs 5-10 ramp to the
    configured maximum (default 0.05); epochs 11-20 keep that maximum.
    Setting domain_align_weight=0 disables the alignment term exactly.
    """
    maximum = max(0.0, float(getattr(config, "domain_align_weight", 0.0)))
    if maximum <= 0.0:
        return 0.0
    warmup = max(0, int(getattr(config, "domain_align_warmup_epochs", 4)))
    ramp = max(0, int(getattr(config, "domain_align_ramp_epochs", 6)))
    e = int(epoch_index)
    if e < warmup:
        return 0.0
    if ramp <= 0:
        return maximum
    progress = min(1.0, max(0.0, float(e - warmup + 1) / float(ramp)))
    return maximum * progress


def class_conditional_domain_alignment_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
) -> torch.Tensor:
    """Align same-class feature centroids across source domains only."""
    losses: List[torch.Tensor] = []
    for c in torch.unique(labels):
        class_mask = labels == c
        present_domains = torch.unique(domains[class_mask])
        if present_domains.numel() < 2:
            continue
        centroids: List[torch.Tensor] = []
        for d in present_domains:
            mask = class_mask & (domains == d)
            if mask.any():
                centroids.append(F.normalize(features[mask].mean(dim=0), dim=0))
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                losses.append(1.0 - torch.dot(centroids[i], centroids[j]))
    if not losses:
        return torch.tensor(0.0, device=features.device)
    return torch.stack(losses).mean()


def confusion_pair_focus_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Extra source-only discrimination loss for repeatedly confused fault pairs.

    The global 6-class CE remains the main objective. This term gives extra
    source-only discrimination to predefined bearing-fault confusion pairs
    without using any target labels.
    """
    losses: List[torch.Tensor] = []
    for a, b in CONFUSION_PAIRS:
        mask = (labels == a) | (labels == b)
        if int(mask.sum().item()) < 2:
            continue
        pair_logits = logits[mask][:, [a, b]]
        pair_targets = (labels[mask] == b).long()
        losses.append(F.cross_entropy(pair_logits, pair_targets))
    if not losses:
        return torch.zeros((), device=logits.device)
    return torch.stack(losses).mean()


class RiskRouter(nn.Module):
    """V8.6 DG-safe leakage-free router.

    The router uses only representation-level statistics produced by the encoder.
    It does not receive softmax confidence, entropy, prototype distance, predicted
    class, top-k probabilities, or any label-dependent evidence.
    """
    def __init__(self, router_type: str = "logistic"):
        super().__init__()
        if router_type == "logistic":
            self.net = nn.Linear(3, 2)
        elif router_type == "mlp":
            self.net = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 2))
        else:
            raise ValueError("router_type must be logistic or mlp")

    def forward(self, risk_features: torch.Tensor) -> torch.Tensor:
        if risk_features.dim() == 1:
            risk_features = risk_features.view(1, -1)
        return self.net(risk_features)


class PairwiseAdvisor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StructuredReviewer:
    def __init__(
        self,
        mode: str,
        model_path: str,
        adapter_path: str,
        dtype_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        device: torch.device,
        load_immediately: bool = False,
    ):
        self.mode = mode
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.dtype_name = dtype_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.loaded = False
        self.using_adapter = False
        self.tokenizer = None
        self.model = None
        if mode == "local_hf" and load_immediately:
            self.load()

    def load(self, adapter_path: Optional[str] = None) -> None:
        if adapter_path is not None:
            self.adapter_path = adapter_path
        if self.loaded:
            return
        if self.mode == "local_hf":
            self._load_local_hf()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.using_adapter = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_dtype(self) -> torch.dtype:
        name = self.dtype_name.lower()
        if self.device.type != "cuda":
            return torch.float32
        if name == "bf16":
            return torch.bfloat16
        if name == "fp16":
            return torch.float16
        if name == "fp32":
            return torch.float32
        # auto: prefer BF16 on modern GPUs such as RTX 4090, otherwise FP16.
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _load_local_hf(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype = self._resolve_dtype()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                use_fast=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Decoder-only batched generation should use left padding so that
            # each prompt ends immediately before its generated continuation.
            self.tokenizer.padding_side = "left"
            load_kwargs = dict(
                pretrained_model_name_or_path=self.model_path,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            try:
                base_model = AutoModelForCausalLM.from_pretrained(dtype=dtype, **load_kwargs)
            except TypeError:
                # Compatibility with older Transformers releases.
                base_model = AutoModelForCausalLM.from_pretrained(torch_dtype=dtype, **load_kwargs)
            adapter = self.adapter_path.strip()
            if adapter:
                adapter_dir = resolve_lora_adapter_directory(Path(adapter))
                adapter = str(adapter_dir)
                self.adapter_path = adapter
                from peft import PeftModel
                base_model = PeftModel.from_pretrained(
                    base_model,
                    adapter,
                    is_trainable=False,
                )
                self.using_adapter = True
            self.model = base_model.to(self.device)
            self.model.eval()
            self.loaded = True
            adapter_msg = f" + LoRA adapter {adapter}" if self.using_adapter else " (base model, no adapter)"
            print(f"Loaded local HF delegated classifier: {self.model_path}{adapter_msg}; dtype={dtype}")
        except Exception as exc:
            print(f"LLM delegated classifier not loaded; evidence fallback will be used: {exc}")
            self.loaded = False

    def build_prompt(self, evidence: Dict[str, object]) -> str:
        # Gemma-2 chat templates commonly support only user/assistant roles.
        # Merge the reviewer instruction into one user turn instead of sending
        # a separate system message. This keeps base-model inference and LoRA
        # training in exactly the same prompt format.
        user_content = (
            REVIEWER_SYSTEM_PROMPT
            + "\n\nDiagnostic evidence and task:\n"
            + build_reviewer_user_message(evidence)
        )
        messages = [{"role": "user", "content": user_content}]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_content + "\n\nAssistant:\n"

    def build_maintenance_prompt(self, context: Dict[str, object]) -> str:
        """Build the independent zero-shot maintenance/re-diagnosis prompt."""
        is_class_report = bool(
            str(context.get("task", "")) == "post_diagnosis_zero_shot_class_maintenance_generation"
            and not bool(context.get("revision_allowed", False))
        )
        instruction = (
            CLASS_MAINTENANCE_SYSTEM_PROMPT
            if is_class_report else MAINTENANCE_REVISION_SYSTEM_PROMPT
        )
        user_content = (
            instruction
            + "\n\nMaintenance context:\n"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        )
        messages = [{"role": "user", "content": user_content}]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_content + "\n\nAssistant:\n"

    @staticmethod
    def _maintenance_retry_fields(failure_issues: Sequence[str]) -> List[str]:
        """Map validation failures to only the LLM fields that actually need repair."""
        all_fields = [
            "suspected_cause_zh",
            "inspection_items_zh",
            "recommended_action_zh",
            "verification_plan_zh",
            "human_review_reason_zh",
            "report_summary_zh",
        ]
        mapping = {
            "missing_suspected_cause_zh": ["suspected_cause_zh"],
            "healthy_cause_does_not_state_no_fault_indication": ["suspected_cause_zh"],
            "healthy_positive_fault_claim": ["suspected_cause_zh"],
            "healthy_action_exceeds_routine_monitoring": ["recommended_action_zh", "report_summary_zh"],
            "fault_cause_missing_conditional_wording": ["suspected_cause_zh"],
            "fault_cause_missing_damage_mechanism": ["suspected_cause_zh"],
            "missing_inspection_items_zh": ["inspection_items_zh"],
            "low_information_inspection_items_zh": ["inspection_items_zh"],
            "missing_recommended_action_zh": ["recommended_action_zh"],
            "low_information_recommended_action_zh": ["recommended_action_zh"],
            "severe_or_compound_without_concrete_action": ["recommended_action_zh"],
            "healthy_class_has_emergency_action": ["recommended_action_zh"],
            "replacement_not_conditional_on_verification": ["recommended_action_zh"],
            "missing_verification_plan_zh": ["verification_plan_zh"],
            "low_information_verification_plan_zh": ["verification_plan_zh"],
            "healthy_report_missing_current_and_vibration_monitoring": [
                "inspection_items_zh", "recommended_action_zh", "verification_plan_zh"
            ],
            "missing_human_review_reason_zh": ["human_review_reason_zh"],
            "human_review_not_required": ["human_review_reason_zh"],
            "human_review_reason_missing_human_confirmation": ["human_review_reason_zh"],
            "high_priority_without_human_and_site_safety_wording": ["human_review_reason_zh"],
            "missing_report_summary_zh": ["report_summary_zh"],
            "missing_outer_race_coverage": [
                "suspected_cause_zh", "inspection_items_zh", "recommended_action_zh"
            ],
            "missing_inner_race_coverage": [
                "suspected_cause_zh", "inspection_items_zh", "recommended_action_zh"
            ],
            "missing_compound_inner_outer_race_coverage": [
                "suspected_cause_zh", "inspection_items_zh", "recommended_action_zh"
            ],
            "internal_model_statistics_in_maintenance_text": all_fields,
            "invalid_json": all_fields,
            "missing_fields": all_fields,
        }
        selected: List[str] = []
        for issue in failure_issues:
            for field in mapping.get(str(issue), []):
                if field not in selected:
                    selected.append(field)
        return selected or all_fields

    def build_maintenance_retry_prompt(
        self,
        context: Dict[str, object],
        failure_status: str,
        failure_issues: Sequence[str],
        base_result: Dict[str, object],
        retry_fields: Sequence[str],
    ) -> str:
        """Repair only failed LLM-generated fields; preserve valid fields verbatim."""
        issue_guidance = {
            "missing_suspected_cause_zh": "原因字段不能为空",
            "healthy_cause_does_not_state_no_fault_indication": "健康类必须明确诊断未显示稳定故障",
            "healthy_positive_fault_claim": "K002健康类不得写可能或疑似存在点蚀、损伤、裂纹、疲劳或故障；必须明确未发现明确故障指征且状态接近健康基线",
            "healthy_action_exceeds_routine_monitoring": "K002健康类维护措施仅限常规监测、电流/振动趋势检查和健康基线复核；不得直接建议停机、降载、拆检、修复或更换",
            "fault_cause_missing_conditional_wording": "故障原因必须加入‘可能’或‘疑似’",
            "fault_cause_missing_damage_mechanism": "故障原因必须写明疲劳、点蚀、损伤或故障机理",
            "missing_inspection_items_zh": "写出具体部件和检查项目",
            "low_information_inspection_items_zh": "检查项不能是占位语",
            "missing_recommended_action_zh": "写出具体且有条件的维护措施",
            "low_information_recommended_action_zh": "维护措施不能是占位语",
            "severe_or_compound_without_concrete_action": "严重或复合故障必须有检查、降载、停机检查、拆检或确认后更换等具体措施",
            "replacement_not_conditional_on_verification": "更换必须以检查、复核或拆检确认作为前提",
            "missing_verification_plan_zh": "写出具体复核方法",
            "low_information_verification_plan_zh": "复核方案不能是占位语",
            "healthy_report_missing_current_and_vibration_monitoring": "健康类必须同时包含电流和振动趋势监测并与健康基线复核",
            "human_review_reason_missing_human_confirmation": "人工核验原因必须包含人工或专业人员",
            "high_priority_without_human_and_site_safety_wording": "必须原样包含‘专业人员确认’和‘现场安全规程’",
            "missing_report_summary_zh": "补充简短维护摘要",
            "missing_outer_race_coverage": "相关字段必须明确写‘外圈’",
            "missing_inner_race_coverage": "相关字段必须明确写‘内圈’",
            "missing_compound_inner_outer_race_coverage": "相关字段必须同时明确写‘内圈’和‘外圈’",
        }
        guidance = []
        for issue in failure_issues:
            item = issue_guidance.get(str(issue))
            if item and item not in guidance:
                guidance.append(item)
        field_list = [str(v) for v in retry_fields]
        preserved = {
            key: base_result.get(key, "")
            for key in (
                "suspected_cause_zh", "inspection_items_zh", "recommended_action_zh",
                "verification_plan_zh", "human_review_reason_zh", "report_summary_zh"
            )
            if key not in field_list
        }
        healthy_retry_guard = (
            "\n【K002-healthy重试硬约束】当前冻结诊断为健康。所有待修复字段必须保持健康语义："
            "明确未发现/未显示故障指征、当前状态接近健康基线；维护仅限电流/振动趋势监测与健康基线复核；"
            "禁止正向声称可能/疑似存在点蚀、损伤、裂纹、疲劳或故障，禁止直接建议停机、降载、拆检、修复或更换。"
            if int(context.get("current_diagnosis_label_id", -1)) == 0 else ""
        )
        retry_note = (
            "\n\n上一次维护内容未完全通过自动校验。本次不是重写整份报告，只修复指定字段。"
            f"失败状态={failure_status}；失败项={','.join(str(x) for x in failure_issues) or 'unknown'}。"
            f"只允许输出这些键={json.dumps(field_list, ensure_ascii=False)}。"
            f"定向要求={'；'.join(guidance) or '严格遵守维护内容规则'}。"
            f"已通过并锁定、禁止改写的字段={json.dumps(preserved, ensure_ascii=False, separators=(',', ':'))}。"
            f"安全策略已锁定：priority={context.get('policy_priority_zh', '')}；"
            f"timeline={context.get('policy_action_timeline_zh', '')}。"
            + healthy_retry_guard
            + "输出一个仅包含待修复键的完整JSON对象，不要输出其他键、Markdown或解释。"
        )
        user_content = (
            CLASS_MAINTENANCE_SYSTEM_PROMPT
            + retry_note
            + "\n\nMaintenance context:\n"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        )
        messages = [{"role": "user", "content": user_content}]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_content + "\n\nAssistant:\n"

    @staticmethod
    def _merge_maintenance_field_retry(
        base_result: Dict[str, object],
        retry_raw: str,
        retry_fields: Sequence[str],
        context: Dict[str, object],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Merge a partial retry JSON into preserved primary fields for re-validation."""
        parsed_retry = StructuredReviewer._extract_json_object(str(retry_raw))
        if not isinstance(parsed_retry, dict):
            return "", None
        allowed = {str(field) for field in retry_fields}
        merged: Dict[str, Any] = {
            "suspected_cause_zh": str(base_result.get("suspected_cause_zh", "") or ""),
            "inspection_items_zh": str(base_result.get("inspection_items_zh", "") or ""),
            "recommended_action_zh": str(base_result.get("recommended_action_zh", "") or ""),
            "verification_plan_zh": str(base_result.get("verification_plan_zh", "") or ""),
            "human_review_required": True,
            "human_review_reason_zh": str(base_result.get("human_review_reason_zh", "") or ""),
            "report_summary_zh": str(base_result.get("report_summary_zh", "") or ""),
        }
        for key in allowed:
            if key == "human_review_required":
                continue
            if key in parsed_retry:
                value = " ".join(str(parsed_retry.get(key, "") or "").split())
                if value:
                    merged[key] = value
        return json.dumps(merged, ensure_ascii=False), merged

    @staticmethod
    def _apply_final_class_maintenance_guardrails(
        result: Dict[str, object],
        context: Dict[str, object],
    ) -> Dict[str, object]:
        """Apply two transparent post-generation safety guardrails only.

        1) Remove clauses that treat unavailable quantities as sensed/measured
           evidence and fall back to current/vibration + on-site verification.
        2) If a P3/P4 report still lacks a concrete escalation after the normal
           field-level retry, inject the minimum action required by the already
           locked deterministic safety policy.

        This function never changes the diagnosis, Router decision, LLM diagnosis,
        priority, or action timeline.
        """
        out = dict(result)
        is_class_report = bool(
            str(context.get("task", ""))
            == "post_diagnosis_zero_shot_class_maintenance_generation"
            and not bool(context.get("revision_allowed", False))
        )
        if not is_class_report or not bool(out.get("maintenance_generation_valid", False)):
            return out

        label = int(context.get("current_diagnosis_label_id", -1))
        try:
            priority = int(context.get("policy_priority", -1))
        except (TypeError, ValueError):
            priority = -1
        if priority < 0:
            priority, _priority_name, _timeline = determine_maintenance_policy_from_attributes(label)

        component_terms = [
            str(value).strip()
            for value in (context.get("required_component_terms_zh", []) or [])
            if str(value).strip()
        ]
        component_phrase = "和".join(component_terms) if component_terms else "相关轴承部位"
        changes: List[str] = []

        # PU online inputs in this experiment are current + vibration. The terms
        # below require an independent sensor, service record, dimensional
        # measurement, or site inspection and therefore must not be presented as
        # available online evidence or as an online measurement item.
        unavailable_term_re = re.compile(
            r"温度|温升|径向力|润滑状态|润滑油|润滑脂|维修历史|维护历史|"
            r"服役历史|剩余寿命|\bRUL\b|缺陷尺寸|裂纹尺寸|点蚀尺寸",
            flags=re.IGNORECASE,
        )
        numeric_unavailable_re = re.compile(
            r"\d+(?:\.\d+)?\s*(?:mm|毫米|μm|um|微米|g|m/s(?:\^?2)?|mm/s|A|安培)\b",
            flags=re.IGNORECASE,
        )

        def _fallback_text(field: str) -> str:
            if field == "suspected_cause_zh":
                if label == 0:
                    return "未发现明确故障指征，当前状态接近健康基线"
                return (
                    f"{component_phrase}可能存在疲劳或点蚀损伤，具体机理需结合"
                    "电流、振动和现场检查进一步确认"
                )
            if field == "inspection_items_zh":
                return f"检查{component_phrase}的电流与振动趋势，并由专业人员按现场规程检查确认"
            if field == "verification_plan_zh":
                return f"结合电流、振动及现场检查对{component_phrase}进行复核"
            if field == "recommended_action_zh":
                return "若发现异常，由专业人员现场检查确认后再采取相应维护措施"
            return str(out.get(field, "") or "")

        # Remove only the clauses containing unavailable quantities. This keeps
        # valid LLM-generated content untouched whenever possible.
        for field in (
            "suspected_cause_zh",
            "inspection_items_zh",
            "recommended_action_zh",
            "verification_plan_zh",
        ):
            original = " ".join(str(out.get(field, "") or "").split())
            if not original:
                continue
            clauses = [part.strip() for part in re.split(r"[；;。]+", original) if part.strip()]
            kept: List[str] = []
            removed = False
            for clause in clauses:
                unavailable = bool(
                    unavailable_term_re.search(clause)
                    or numeric_unavailable_re.search(clause)
                    or _unsupported_claim_reasons(clause)
                )
                if unavailable:
                    removed = True
                    continue
                kept.append(clause)
            if removed:
                cleaned = "；".join(kept).strip("； ")
                if not cleaned:
                    cleaned = _fallback_text(field)
                out[field] = cleaned
                changes.append(f"remove_unavailable_measurement:{field}")

        # Final minimum-action guardrail. It is reached only after the normal
        # LLM generation + field-level retry path. The policy priority itself is
        # already deterministic and frozen before generation.
        action = " ".join(str(out.get("recommended_action_zh", "") or "").split())
        concrete_action_re = re.compile(
            r"降载|停机(?:检查)?|退出运行|专业检查|现场检查|检修|拆检|"
            r"确认后更换|检查确认后更换|拆检确认后更换"
        )
        if priority >= 3 and not concrete_action_re.search(action):
            if priority >= 4:
                action = (
                    f"在现场安全条件允许时立即停机或退出运行，并由专业人员对{component_phrase}"
                    "进行拆检确认；若确认存在严重损伤，再更换受损部件"
                )
            else:
                action = (
                    f"建议尽快降载，并由专业人员在24小时内对{component_phrase}进行专业检查；"
                    "若检查确认存在严重损伤，再安排拆检或确认后更换受损部件"
                )
            out["recommended_action_zh"] = action
            changes.append("policy_minimum_concrete_action")

        if changes:
            # Recompute only maintenance-content validity from the final guarded
            # fields. Structural validity, frozen diagnosis, priority and timeline
            # remain unchanged.
            final_values = {
                "maintenance_priority": str(context.get("policy_priority_zh", out.get("maintenance_priority", ""))),
                "action_timeline_zh": str(context.get("policy_action_timeline_zh", out.get("action_timeline_zh", ""))),
                "suspected_cause_zh": str(out.get("suspected_cause_zh", "") or ""),
                "inspection_items_zh": str(out.get("inspection_items_zh", "") or ""),
                "recommended_action_zh": str(out.get("recommended_action_zh", "") or ""),
                "verification_plan_zh": str(out.get("verification_plan_zh", "") or ""),
                "human_review_reason_zh": str(out.get("human_review_reason_zh", "") or ""),
            }
            content_issues = assess_class_maintenance_content(
                label,
                final_values,
                bool(out.get("human_review_required", True)),
            )
            out["maintenance_content_validation_issues"] = list(content_issues)
            out["maintenance_content_valid"] = bool(
                out.get("maintenance_generation_valid", False) and not content_issues
            )
            out["maintenance_guardrail_applied"] = True
            out["maintenance_guardrail_changes"] = list(changes)
        else:
            out["maintenance_guardrail_applied"] = False
            out["maintenance_guardrail_changes"] = []
        return out

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:idx + 1])
                        return value if isinstance(value, dict) else None
                    except json.JSONDecodeError:
                        return None
        return None



    @staticmethod
    def _recover_core_fields_from_truncated_json(text: str) -> Dict[str, Any]:
        """Recover the core classification fields when generation is cut off.

        Gemma sometimes produces a correct answer/candidate_scores and then runs
        out of tokens while writing non-essential fields.  For diagnosis, the
        closed-set candidate_scores are the core decision object, so we recover
        them with conservative regex parsing instead of discarding the whole LLM
        decision as invalid.
        """
        recovered: Dict[str, Any] = {}

        def _string_field(name: str) -> Optional[str]:
            m = re.search(r'"' + re.escape(name) + r'"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text, flags=re.DOTALL)
            if not m:
                return None
            try:
                return json.loads('"' + m.group(1) + '"')
            except Exception:
                return m.group(1)

        def _int_field(name: str) -> Optional[int]:
            m = re.search(r'"' + re.escape(name) + r'"\s*:\s*(-?\d+)', text)
            if not m:
                return None
            try:
                return int(m.group(1))
            except ValueError:
                return None

        def _float_field(name: str) -> Optional[float]:
            m = re.search(r'"' + re.escape(name) + r'"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text)
            if not m:
                return None
            try:
                return float(m.group(1))
            except ValueError:
                return None

        for name in ("answer", "decision", "final_fault", "reasoning_chain", "reasoning"):
            value = _string_field(name)
            if value is not None:
                recovered[name] = value
        final_label = _int_field("final_label")
        if final_label is not None:
            recovered["final_label"] = final_label
        confidence = _float_field("confidence")
        if confidence is not None:
            recovered["confidence"] = confidence

        # Prefer parsing the complete candidate_scores array if it is balanced.
        key = '"candidate_scores"'
        key_pos = text.find(key)
        scores: List[Dict[str, float]] = []
        if key_pos >= 0:
            lb = text.find('[', key_pos)
            if lb >= 0:
                depth = 0
                in_string = False
                escaped = False
                end = -1
                for idx in range(lb, len(text)):
                    ch = text[idx]
                    if in_string:
                        if escaped:
                            escaped = False
                        elif ch == '\\':
                            escaped = True
                        elif ch == '"':
                            in_string = False
                        continue
                    if ch == '"':
                        in_string = True
                    elif ch == '[':
                        depth += 1
                    elif ch == ']':
                        depth -= 1
                        if depth == 0:
                            end = idx
                            break
                if end > lb:
                    try:
                        value = json.loads(text[lb:end + 1])
                        if isinstance(value, list):
                            scores = value
                    except Exception:
                        scores = []
                if not scores:
                    segment = text[lb:]
                    for m in re.finditer(r'\{\s*"label_id"\s*:\s*(-?\d+)\s*,\s*"score"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\}', segment):
                        try:
                            scores.append({"label_id": int(m.group(1)), "score": float(m.group(2))})
                        except ValueError:
                            continue
        if scores:
            recovered["candidate_scores"] = scores
        return recovered

    @staticmethod
    def _parse_label_fallback(text: str, top3: List[int]) -> int:
        lower = text.lower()
        for c in top3:
            if FAULT_NAMES[c].lower() in lower or f"label {c}" in lower or f'"{c}"' in text:
                return c
        return top3[0]

    def _normalize_response(self, text: str, evidence: Dict[str, object], top3: List[int]) -> Dict[str, object]:
        """Normalize V8.8 candidate-scoring LLM output.

        The delegated LLM no longer provides a single trusted final_label with a
        self-reported confidence. It must score every closed-set candidate. The
        final label used downstream is argmax(candidate_scores). This preserves
        Router-only direct delegation while avoiding the uncalibrated confidence
        problem observed in V8.6/V8.7.
        """
        raw = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
        parsed = self._extract_json_object(cleaned)
        exact_json = parsed is not None
        if parsed is None:
            parsed = self._recover_core_fields_from_truncated_json(cleaned)
        json_repaired = bool((not exact_json) and parsed)
        parsed = parsed or {}

        candidate_ids = candidate_ids_from_evidence(evidence, top3)
        answer_label = parse_llm_answer_tag(parsed.get("answer", ""))
        scores_valid, candidate_scores, score_label, score_conf, score_margin, score_reason = normalize_candidate_scores_object(
            parsed.get("candidate_scores"), candidate_ids
        )
        if not scores_valid and answer_label is not None:
            scores_valid, candidate_scores, score_label, score_conf, score_margin, score_reason = synthesize_candidate_scores_from_answer(
                evidence, candidate_ids, int(answer_label)
            )

        # Use candidate_scores as the classification result. final_label in the
        # raw JSON is retained only for diagnostics and safe repair reporting.
        raw_final_label = parsed.get("final_label")
        try:
            raw_final_label_int = int(raw_final_label)
        except (TypeError, ValueError):
            raw_final_label_int = None

        if scores_valid:
            final_label = int(score_label)
            confidence = float(score_conf)
            label_repaired_from_scores = raw_final_label_int != final_label
        else:
            # No candidate_scores means the V8.8 classifier output is unusable.
            # Keep a closed-set fallback label only for report completeness; the
            # gate will reject this LLM decision.
            final_label = self._parse_label_fallback(cleaned, top3)
            confidence = 0.0
            label_repaired_from_scores = False

        decision = str(parsed.get("decision", "classify")).strip().lower()
        if decision in {"accept", "revise"}:
            decision = "classify"
        if decision != "classify":
            decision = "classify"

        answer_consistent = bool(answer_label is not None and int(answer_label) == int(final_label))

        reasoning_chain = str(parsed.get("reasoning_chain", ""))[:900].strip()
        if reasoning_chain and not (reasoning_chain.startswith("<think>") and reasoning_chain.endswith("</think>")):
            reasoning_chain = "<think>" + reasoning_chain.strip("<think>/ ") + "</think>"

        reasoning = str(parsed.get("reasoning", ""))[:800].strip()
        if not reasoning and reasoning_chain:
            reasoning = re.sub(r"</?think>", "", reasoning_chain).strip()[:300]
        if not reasoning:
            reasoning = cleaned[:300]
        forbidden_reasoning_terms = has_forbidden_llm_reasoning_terms(reasoning) or has_forbidden_llm_reasoning_terms(reasoning_chain)

        # Diagnostic generation does not produce maintenance. V10.13 runs a
        # separate LLM maintenance-draft pass after diagnosis.
        maintenance = ""

        raw_fault_label: Optional[int] = None
        try:
            raw_fault_label = infer_label_from_fault_text(parsed.get("final_fault", ""))
        except Exception:
            raw_fault_label = None
        label_fault_consistent = bool(raw_fault_label is None or raw_fault_label == final_label)

        # V8.9.2: accept a recovered core decision if answer/final label and
        # candidate_scores are complete. Exact full JSON is no longer required
        # because non-essential trailing fields can be truncated by generation.
        structured_valid = bool(scores_valid and final_label in candidate_ids and answer_consistent)
        return {
            "answer": llm_answer_tag(final_label),
            "answer_consistent": bool(answer_consistent),
            "reasoning_chain": reasoning_chain,
            "decision": decision,
            "final_label": final_label,
            "final_fault": FAULT_NAMES.get(final_label, str(final_label)),
            "candidate_scores": candidate_scores,
            "confidence": confidence,
            "score_margin": float(score_margin),
            "candidate_scores_valid": bool(scores_valid),
            "candidate_score_reason": score_reason,
            "label_repaired_from_scores": bool(label_repaired_from_scores),
            "label_fault_consistent": bool(label_fault_consistent),
            "reasoning": reasoning,
            "maintenance_suggestion": maintenance,
            "valid_json": structured_valid,
            "exact_json": bool(exact_json),
            "json_repaired": bool(json_repaired),
            "forbidden_evidence_terms": bool(forbidden_reasoning_terms),
            "raw_text": raw[:1200],
        }

    def review_batch(
        self,
        evidences: Sequence[Dict[str, object]],
        top3_lists: Sequence[List[int]],
        batch_size: int = 4,
    ) -> List[Dict[str, object]]:
        """Review multiple routed samples in small generation batches.

        Batched decoding keeps the GPU busy and is much faster than calling
        ``generate`` once per sample. Any failed batch falls back to the
        deterministic prototype/evidence reviewer, so evaluation can continue.
        """
        if len(evidences) != len(top3_lists):
            raise ValueError("evidences and top3_lists must have the same length")
        if not evidences:
            return []
        if not self.loaded:
            return [
                self.prototype_review(evidence, top3)
                for evidence, top3 in zip(evidences, top3_lists)
            ]

        results: List[Dict[str, object]] = []
        batch_size = max(1, int(batch_size))
        for start in range(0, len(evidences), batch_size):
            batch_evidence = list(evidences[start:start + batch_size])
            batch_top3 = list(top3_lists[start:start + batch_size])
            prompts = [self.build_prompt(evidence) for evidence in batch_evidence]
            try:
                # Tokenize each prompt without blind right truncation. The old path
                # could cut the structured evidence in the middle of a JSON field;
                # Gemma then continued that input fragment instead of producing the
                # requested review JSON. Preserve both the instruction head and the
                # final evidence/chat-generation tail.
                encoded_prompts: List[List[int]] = []
                max_tokens = max(256, int(self.max_input_tokens))
                for prompt in prompts:
                    ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    if len(ids) > max_tokens:
                        head = min(384, max_tokens // 3)
                        tail = max_tokens - head
                        ids = ids[:head] + ids[-tail:]
                    encoded_prompts.append(ids)
                padded_len = max(len(ids) for ids in encoded_prompts)
                input_ids = torch.full(
                    (len(encoded_prompts), padded_len),
                    int(self.tokenizer.pad_token_id),
                    dtype=torch.long,
                    device=self.device,
                )
                attention_mask = torch.zeros(
                    (len(encoded_prompts), padded_len),
                    dtype=torch.long,
                    device=self.device,
                )
                for row, ids in enumerate(encoded_prompts):
                    token_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
                    input_ids[row, -len(ids):] = token_tensor
                    attention_mask[row, -len(ids):] = 1
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                input_length = int(input_ids.shape[1])
                with torch.inference_mode():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                generated = out[:, input_length:]
                texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                results.extend(
                    self._normalize_response(text, evidence, top3)
                    for text, evidence, top3 in zip(texts, batch_evidence, batch_top3)
                )
            except Exception as exc:
                for evidence, top3 in zip(batch_evidence, batch_top3):
                    result = self.prototype_review(evidence, top3)
                    result["reasoning"] = (
                        f"LLM batch query failed; evidence fallback used. Error: {exc}"
                    )
                    result["valid_json"] = False
                    results.append(result)
        return results

    def review(self, evidence: Dict[str, object], top3: List[int]) -> Dict[str, object]:
        return self.review_batch([evidence], [top3], batch_size=1)[0]

    @staticmethod
    def _empty_maintenance_result(
        context: Dict[str, object],
        status: str,
        reason: str,
        raw_text: str = "",
    ) -> Dict[str, object]:
        current_label = int(context.get("current_diagnosis_label_id", 0))
        candidate_ids: List[int] = []
        for item in context.get("candidate_labels", []) or []:
            try:
                label_id = int(item.get("label_id"))
            except (TypeError, ValueError, AttributeError):
                continue
            if label_id in FAULT_NAMES and label_id not in candidate_ids:
                candidate_ids.append(label_id)
        if current_label not in candidate_ids:
            candidate_ids = [current_label] + candidate_ids
        return {
            "maintenance_generation_status": status,
            "maintenance_generation_valid": False,
            "maintenance_content_valid": False,
            "maintenance_content_validation_issues": [status],
            "candidate_scores": [],
            "candidate_scores_valid": False,
            "candidate_score_reason": status,
            "final_label": current_label,
            "proposed_label": current_label,
            "proposed_fault": FAULT_NAMES.get(current_label, str(current_label)),
            "confidence": 0.0,
            "score_margin": 0.0,
            "label_fault_consistent": True,
            "forbidden_evidence_terms": False,
            "valid_json": False,
            "exact_json": False,
            "diagnosis_consistent": True,
            "revision_reason": reason,
            "maintenance_priority": "",
            "action_timeline_zh": "",
            "suspected_cause_zh": "",
            "inspection_items_zh": "",
            "recommended_action_zh": "",
            "verification_plan_zh": "",
            "human_review_required": True,
            "human_review_reason_zh": reason,
            "report_summary_zh": "",
            "maintenance_generated_token_count": 0,
            "raw_text": str(raw_text)[:4000],
        }

    @staticmethod
    def _normalize_maintenance_response(
        raw: str,
        context: Dict[str, object],
    ) -> Dict[str, object]:
        """Normalize a structured maintenance draft for reporting and rule checks."""
        parsed = StructuredReviewer._extract_json_object(str(raw))
        if not isinstance(parsed, dict):
            return StructuredReviewer._empty_maintenance_result(
                context, "invalid_json", "维护大模型未返回可解析的JSON，需人工复核。", raw
            )

        current_label = int(context.get("current_diagnosis_label_id", 0))
        is_class_report = bool(
            str(context.get("task", "")) == "post_diagnosis_zero_shot_class_maintenance_generation"
            and not bool(context.get("revision_allowed", False))
        )
        candidate_ids: List[int] = []
        for item in context.get("candidate_labels", []) or []:
            try:
                label_id = int(item.get("label_id"))
            except (TypeError, ValueError, AttributeError):
                continue
            if label_id in FAULT_NAMES and label_id not in candidate_ids:
                candidate_ids.append(label_id)
        if not candidate_ids:
            candidate_ids = [current_label]
        if current_label not in candidate_ids:
            candidate_ids.insert(0, current_label)

        def _text(key: str, limit: int = 1000, aliases: Sequence[str] = ()) -> str:
            for candidate_key in (key, *aliases):
                value = " ".join(str(parsed.get(candidate_key, "")).split())[:limit]
                if value:
                    return value
            return ""

        if is_class_report:
            # Diagnosis is already frozen. Candidate scoring and label fields are
            # deliberately injected by code so the maintenance-only response can
            # stay short and cannot revise diagnosis.
            scores_valid = True
            candidate_scores = [{"label_id": int(current_label), "score": 1.0}]
            proposed_label = int(current_label)
            confidence = 1.0
            score_margin = 1.0
            score_reason = "frozen_class_diagnosis"
            proposed_fault = FAULT_NAMES.get(int(current_label), str(current_label))
            label_fault_consistent = True
            proposed_field_consistent = True
        else:
            scores_valid, candidate_scores, proposed_label, confidence, score_margin, score_reason = normalize_candidate_scores_object(
                parsed.get("candidate_scores"), candidate_ids
            )
            if not scores_valid:
                return StructuredReviewer._empty_maintenance_result(
                    context,
                    "invalid_candidate_scores",
                    f"维护大模型候选分数无效（{score_reason}），需人工复核。",
                    raw,
                )
            proposed_fault = _text("proposed_fault", 300)
            inferred_fault_label = infer_label_from_fault_text(proposed_fault)
            label_fault_consistent = bool(
                inferred_fault_label is None or int(inferred_fault_label) == int(proposed_label)
            )
            raw_proposed = parsed.get("proposed_label")
            try:
                raw_proposed_int = int(raw_proposed)
            except (TypeError, ValueError):
                raw_proposed_int = None
            proposed_field_consistent = bool(raw_proposed_int == int(proposed_label))

        maintenance_text_fields = (
            "maintenance_priority",
            "action_timeline_zh",
            "suspected_cause_zh",
            "inspection_items_zh",
            "recommended_action_zh",
            "verification_plan_zh",
            "human_review_reason_zh",
            "report_summary_zh",
        )
        llm_class_text_fields = (
            "suspected_cause_zh",
            "inspection_items_zh",
            "recommended_action_zh",
            "verification_plan_zh",
            "human_review_reason_zh",
            "report_summary_zh",
        )
        text_aliases: Dict[str, Sequence[str]] = {
            "revision_reason": ("revision_reasoning", "reasoning"),
            "human_review_reason_zh": ("human_review_reasoning", "human_review_reason"),
            "report_summary_zh": ("maintenance_report_summary_zh", "summary_zh"),
        }
        required_text = llm_class_text_fields if is_class_report else (
            "revision_reason", *maintenance_text_fields
        )
        values = {
            key: _text(key, aliases=text_aliases.get(key, ()))
            for key in required_text
        }
        if is_class_report:
            # Priority and timeline are safety-policy outputs, not free-form LLM fields.
            values["maintenance_priority"] = str(context.get("policy_priority_zh", ""))
            values["action_timeline_zh"] = str(context.get("policy_action_timeline_zh", ""))
            values["revision_reason"] = "诊断标签已冻结；安全策略锁定优先级和时限，大模型仅生成维护建议内容。"
        structure_valid = all(bool(values[key]) for key in required_text)
        priority_valid = bool(
            re.match(r"^\s*P[0-4](?=$|[^0-9])", values["maintenance_priority"], flags=re.IGNORECASE)
        )
        revision_allowed = bool(context.get("revision_allowed", False))
        revision_scope_valid = bool(revision_allowed or int(proposed_label) == current_label)
        narrative_text = " ".join(values.get(key, "") for key in maintenance_text_fields)
        internal_statistics_present = bool(re.search(
            r"\b(?:token_id|token_count|sample_id|confidence|risk_probability|"
            r"uncertainty_entropy|entropy|mc_variance|prototype_distance|domain_shift|"
            r"modality_weight|candidate_score)\b",
            narrative_text,
            flags=re.IGNORECASE,
        ))
        valid = bool(
            structure_valid
            and priority_valid
            and proposed_field_consistent
            and label_fault_consistent
            and revision_scope_valid
            and not internal_statistics_present
        )

        raw_review = parsed.get("human_review_required", True)
        human_review = raw_review if isinstance(raw_review, bool) else str(raw_review).strip().lower() in {
            "true", "1", "yes", "y", "是", "需要"
        }
        if is_class_report:
            # Every generated maintenance report is a draft for mandatory human
            # verification, irrespective of the priority proposed by the LLM.
            human_review = True
        issues: List[str] = []
        if not structure_valid:
            issues.append("missing_fields")
            for missing_key in required_text:
                if not bool(values.get(missing_key, "")):
                    issues.append(f"missing_{missing_key}")
        if not priority_valid:
            issues.append("invalid_priority")
        if not proposed_field_consistent:
            issues.append("proposed_label_not_candidate_argmax")
        if not label_fault_consistent:
            issues.append("proposed_fault_mismatch")
        if not revision_scope_valid:
            issues.append("revision_not_allowed_in_class_report")
        if internal_statistics_present:
            issues.append("internal_model_statistics_in_maintenance_text")
        if issues:
            human_review = True
        content_issues = (
            assess_class_maintenance_content(current_label, values, bool(human_review))
            if is_class_report and valid else []
        )
        content_valid = bool(valid and (not is_class_report or not content_issues))

        return {
            "maintenance_generation_status": "ok" if valid else "rule_validation_failed",
            "maintenance_generation_valid": bool(valid),
            "maintenance_validation_issues": issues,
            "maintenance_content_valid": content_valid,
            "maintenance_content_validation_issues": content_issues,
            "candidate_scores": candidate_scores,
            "candidate_scores_valid": True,
            "candidate_score_reason": score_reason,
            "final_label": int(proposed_label),
            "proposed_label": int(proposed_label),
            "proposed_fault": FAULT_NAMES.get(int(proposed_label), proposed_fault),
            "confidence": float(confidence),
            "score_margin": float(score_margin),
            "label_fault_consistent": bool(label_fault_consistent),
            "forbidden_evidence_terms": bool(internal_statistics_present),
            "valid_json": bool(valid),
            "exact_json": True,
            "diagnosis_consistent": bool(int(proposed_label) == current_label),
            **values,
            "human_review_required": bool(human_review),
            "raw_text": str(raw)[:4000],
        }

    def _generate_maintenance_text_batch(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        seed: int,
        disable_diagnosis_adapter: bool = False,
    ) -> List[Tuple[str, int]]:
        """Generate maintenance text, optionally bypassing the diagnosis LoRA.

        The adapter is disabled only inside the PEFT context manager. It is
        restored automatically before routed diagnostic review can run again.
        """
        encoded_prompts: List[List[int]] = []
        max_tokens = max(256, int(self.max_input_tokens))
        for prompt in prompts:
            ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > max_tokens:
                head = min(384, max_tokens // 3)
                tail = max_tokens - head
                ids = ids[:head] + ids[-tail:]
            encoded_prompts.append(ids)
        padded_len = max(len(ids) for ids in encoded_prompts)
        input_ids = torch.full(
            (len(encoded_prompts), padded_len),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (len(encoded_prompts), padded_len), dtype=torch.long, device=self.device
        )
        for row_idx, ids in enumerate(encoded_prompts):
            token_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
            input_ids[row_idx, -len(ids):] = token_tensor
            attention_mask[row_idx, -len(ids):] = 1
        input_length = int(input_ids.shape[1])
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        adapter_context = nullcontext()
        if disable_diagnosis_adapter:
            disable_adapter = getattr(self.model, "disable_adapter", None)
            if not callable(disable_adapter):
                raise RuntimeError(
                    "The loaded PEFT model cannot temporarily disable its diagnosis adapter."
                )
            adapter_context = disable_adapter()
        with adapter_context:
            with torch.random.fork_rng(devices=devices, enabled=True):
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
                with torch.inference_mode():
                    output = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max(128, int(max_new_tokens)),
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
        generated_ids = output[:, input_length:]
        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [
            (
                str(text_value),
                int((token_ids != int(self.tokenizer.pad_token_id)).sum().item()),
            )
            for text_value, token_ids in zip(texts, generated_ids)
        ]

    def maintenance_review_batch(
        self,
        contexts: Sequence[Dict[str, object]],
        batch_size: int = 4,
        max_new_tokens: int = 640,
        seed: int = 42,
    ) -> List[Dict[str, object]]:
        """Generate drafts and retry each invalid class report once without templates."""
        if not contexts:
            return []
        if not self.loaded or self.model is None or self.tokenizer is None:
            return [
                self._empty_maintenance_result(
                    context, "llm_not_loaded", "LoRA大模型未加载，维护建议未生成。"
                )
                for context in contexts
            ]
        if not self.using_adapter:
            return [
                self._empty_maintenance_result(
                    context,
                    "lora_adapter_not_loaded",
                    "诊断LoRA适配器未加载，禁止退化为基础大模型生成维护报告。",
                )
                for context in contexts
            ]

        results: List[Dict[str, object]] = []
        batch_size = max(1, int(batch_size))
        for start in range(0, len(contexts), batch_size):
            batch_contexts = list(contexts[start:start + batch_size])
            batch_class_flags = [
                bool(
                    str(context.get("task", ""))
                    == "post_diagnosis_zero_shot_class_maintenance_generation"
                    and not bool(context.get("revision_allowed", False))
                )
                for context in batch_contexts
            ]
            if any(batch_class_flags) and not all(batch_class_flags):
                raise ValueError(
                    "Class maintenance and maintenance revision contexts must not share one batch."
                )
            disable_diagnosis_adapter = bool(all(batch_class_flags))
            prompts = [self.build_maintenance_prompt(context) for context in batch_contexts]
            try:
                generated = self._generate_maintenance_text_batch(
                    prompts,
                    max_new_tokens=max_new_tokens,
                    seed=int(seed) + int(start),
                    disable_diagnosis_adapter=disable_diagnosis_adapter,
                )
            except Exception as exc:
                results.extend(
                    self._empty_maintenance_result(
                        context,
                        "generation_error",
                        f"大模型维护生成失败，需人工复核：{exc}",
                    )
                    for context in batch_contexts
                )
                continue

            for local_idx, ((text_value, primary_tokens), context) in enumerate(
                zip(generated, batch_contexts)
            ):
                normalized = self._normalize_maintenance_response(text_value, context)
                normalized["maintenance_generated_token_count"] = int(primary_tokens)
                normalized["maintenance_final_generated_token_count"] = int(primary_tokens)
                normalized["maintenance_retry_used"] = False
                is_class_report = bool(
                    str(context.get("task", ""))
                    == "post_diagnosis_zero_shot_class_maintenance_generation"
                    and not bool(context.get("revision_allowed", False))
                )

                needs_retry = bool(
                    is_class_report
                    and (
                        not bool(normalized.get("maintenance_generation_valid", False))
                        or not bool(normalized.get("maintenance_content_valid", False))
                    )
                )
                if needs_retry:
                    primary_status = (
                        str(normalized.get("maintenance_generation_status", "unknown"))
                        if not bool(normalized.get("maintenance_generation_valid", False))
                        else "content_validation_failed"
                    )
                    primary_issues = [
                        str(value) for value in (
                            list(normalized.get("maintenance_validation_issues", []) or [])
                            + list(normalized.get("maintenance_content_validation_issues", []) or [])
                        )
                    ]
                    retry_fields = self._maintenance_retry_fields(primary_issues)
                    # V10.13.11 inherits V10.13.9 K002 logic: only for K002-healthy, repair the coherent maintenance
                    # content as one unit so a corrected cause cannot coexist with an
                    # old fault-oriented action from the primary generation.
                    if int(context.get("current_diagnosis_label_id", -1)) == 0:
                        retry_fields = [
                            "suspected_cause_zh",
                            "inspection_items_zh",
                            "recommended_action_zh",
                            "verification_plan_zh",
                            "report_summary_zh",
                        ]
                    retry_prompt = self.build_maintenance_retry_prompt(
                        context,
                        primary_status,
                        primary_issues,
                        base_result=normalized,
                        retry_fields=retry_fields,
                    )
                    try:
                        retry_generated = self._generate_maintenance_text_batch(
                            [retry_prompt],
                            max_new_tokens=max_new_tokens,
                            seed=int(seed) + 100000 + int(start) + int(local_idx),
                            disable_diagnosis_adapter=True,
                        )
                        retry_text, retry_tokens = retry_generated[0]
                        merged_retry_text, merged_retry = self._merge_maintenance_field_retry(
                            normalized, retry_text, retry_fields, context
                        )
                        retry_normalized = (
                            self._normalize_maintenance_response(merged_retry_text, context)
                            if merged_retry is not None
                            else self._empty_maintenance_result(
                                context,
                                "invalid_retry_json",
                                "局部维护字段重试未返回可解析JSON，保留首次生成结果。",
                                retry_text,
                            )
                        )
                        def _report_quality(result: Dict[str, object]) -> Tuple[int, int, int, int, int, int]:
                            structural_issues = list(
                                result.get("maintenance_validation_issues", []) or []
                            )
                            content_issues = list(
                                result.get("maintenance_content_validation_issues", []) or []
                            )
                            safety_issue_count = sum(
                                str(issue) in {
                                    "healthy_class_has_emergency_action",
                                    "high_priority_without_human_and_site_safety_wording",
                                    "replacement_not_conditional_on_verification",
                                    "human_review_not_required",
                                }
                                for issue in content_issues
                            )
                            return (
                                int(bool(result.get("maintenance_content_valid", False))),
                                int(bool(result.get("maintenance_generation_valid", False))),
                                -int(safety_issue_count),
                                -len(content_issues),
                                -len(structural_issues),
                                int(bool(result.get("maintenance_exact_json", False))),
                            )

                        retry_selected = bool(
                            _report_quality(retry_normalized) > _report_quality(normalized)
                        )
                        selected_result = retry_normalized if retry_selected else normalized
                        selected_result["maintenance_generated_token_count"] = int(
                            primary_tokens + retry_tokens
                        )
                        selected_result["maintenance_final_generated_token_count"] = int(
                            retry_tokens if retry_selected else primary_tokens
                        )
                        selected_result["maintenance_retry_used"] = True
                        selected_result["maintenance_retry_selected"] = retry_selected
                        selected_result["maintenance_primary_status"] = primary_status
                        selected_result["maintenance_primary_validation_issues"] = primary_issues
                        selected_result["maintenance_primary_raw_text"] = str(text_value)[:4000]
                        selected_result["maintenance_retry_raw_text"] = str(retry_text)[:4000]
                        selected_result["maintenance_retry_validation_issues"] = list(
                            retry_normalized.get("maintenance_validation_issues", []) or []
                        ) + list(
                            retry_normalized.get("maintenance_content_validation_issues", []) or []
                        )
                        normalized = selected_result
                    except Exception as exc:
                        normalized["maintenance_retry_used"] = True
                        normalized["maintenance_retry_error"] = str(exc)

                # V10.13.11: two final transparent safety guardrails only.
                # They do not modify diagnosis, Router, LLM diagnosis, priority, or timeline.
                if is_class_report:
                    normalized = self._apply_final_class_maintenance_guardrails(
                        normalized, context
                    )

                normalized["maintenance_model_using_lora"] = bool(
                    self.using_adapter and not is_class_report
                )
                normalized["maintenance_diagnosis_lora_loaded"] = bool(self.using_adapter)
                normalized["maintenance_adapter_disabled_for_generation"] = bool(
                    is_class_report
                )
                normalized["maintenance_model_path"] = str(self.model_path)
                normalized["maintenance_adapter_path"] = str(self.adapter_path)
                results.append(normalized)
        return results

    @staticmethod
    def prototype_review(evidence: Dict[str, object], top3: List[int]) -> Dict[str, object]:
        pairwise = evidence.get("pairwise_advisor")
        if isinstance(pairwise, dict) and int(pairwise.get("final_label", -1)) in top3:
            final_label = int(pairwise["final_label"])
            reason = "Fallback classifier selected a label from a known confusion pair."
        elif "advisor_rerank_label_id" in evidence:
            final_label = int(evidence["advisor_rerank_label_id"])
            reason = "Fallback classifier used deterministic evidence scoring."
        else:
            final_label = int(evidence.get("small_model_label_id", top3[0]))
            reason = "Fallback retained the small-model top-1 prediction."
        ids = candidate_ids_from_evidence(evidence, top3)
        if final_label not in ids:
            final_label = ids[0]
        scores = [
            {"label_id": int(label_id), "score": round(1.0 if int(label_id) == final_label else 0.0, 6)}
            for label_id in ids
        ]
        return {
            "answer": llm_answer_tag(final_label),
            "answer_consistent": True,
            "reasoning_chain": "<think>Fallback evidence scoring selected the closed-set candidate with the strongest deterministic support.</think>",
            "decision": "classify",
            "final_label": final_label,
            "final_fault": FAULT_NAMES[final_label],
            "candidate_scores": scores,
            "confidence": 1.0,
            "score_margin": 1.0,
            "candidate_scores_valid": True,
            "candidate_score_reason": "fallback",
            "label_repaired_from_scores": False,
            "label_fault_consistent": True,
            "reasoning": reason,
            "maintenance_suggestion": "",
            "valid_json": False,
            "exact_json": False,
            "json_repaired": False,
            "forbidden_evidence_terms": False,
            "raw_text": "",
        }


class FeatureTokenAligner(nn.Module):
    """Trainable feature-to-token alignment bridge.

    The encoder feature is projected into a compact set of numeric pseudo-tokens.
    These tokens are used as the LLM-facing evidence for routed hard samples. An
    auxiliary token classifier is trained with source labels so the token space is
    discriminative before LoRA-SFT examples are exported.
    """
    def __init__(self, feature_dim: int, token_count: int = 8, token_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.token_count = int(token_count)
        self.token_dim = int(token_dim)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, token_count * token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(z).view(z.size(0), self.token_count, self.token_dim)
        return torch.tanh(self.norm(tokens))


def summarize_feature_tokens(tokens_i: torch.Tensor, max_tokens: int = 8, max_dims: int = 8) -> Tuple[List[List[float]], Dict[str, float]]:
    """Convert aligned tokens into compact JSON-safe prompt evidence."""
    t = tokens_i.detach().float().cpu()
    t = t[:max_tokens, :max_dims]
    token_rows = [[round(float(v), 4) for v in row.tolist()] for row in t]
    summary = {
        "mean": round(float(t.mean().item()), 5) if t.numel() else 0.0,
        "std": round(float(t.std(unbiased=False).item()), 5) if t.numel() else 0.0,
        "norm": round(float(t.norm().item()), 5) if t.numel() else 0.0,
        "max_abs": round(float(t.abs().max().item()), 5) if t.numel() else 0.0,
    }
    return token_rows, summary


PHYSICAL_FEATURE_NAMES = (
    "mean_abs", "rms", "std", "peak_to_peak", "kurtosis", "skewness",
    "crest_factor", "spectral_entropy", "low_freq_ratio", "high_freq_ratio",
)


def summarize_physical_features(physical_i: Optional[torch.Tensor], modality_names: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Summarize modality-wise physics/frequency evidence for the LLM.

    This keeps the Gemma branch text-only, but gives it information closer to
    the STFT/statistical evidence used by the reference Qwen-Omni code.
    """
    if physical_i is None or physical_i.numel() == 0:
        return {}
    p = physical_i.detach().float().cpu()
    out: Dict[str, object] = {
        "mean": round(float(p.mean().item()), 5),
        "std": round(float(p.std(unbiased=False).item()), 5),
        "norm": round(float(p.norm().item()), 5),
        "max_abs": round(float(p.abs().max().item()), 5),
    }
    chunk = len(PHYSICAL_FEATURE_NAMES)
    modality_names = list(modality_names or ["motor_current", "vibration"])
    modality_rows: List[Dict[str, object]] = []
    for mi in range(max(1, int(math.ceil(float(p.numel()) / float(chunk))))):
        start = mi * chunk
        end = min(start + chunk, int(p.numel()))
        if start >= end:
            continue
        vals = p[start:end]
        row: Dict[str, object] = {"modality": modality_names[mi] if mi < len(modality_names) else f"modality_{mi}"}
        for name, value in zip(PHYSICAL_FEATURE_NAMES, vals.tolist()):
            row[name] = round(float(value), 5)
        modality_rows.append(row)
    out["modality_physics"] = modality_rows

    def _best_by(key: str) -> str:
        best_name = "unknown"
        best_value = -float("inf")
        for row in modality_rows:
            try:
                value = float(row.get(key, -float("inf")))
            except (TypeError, ValueError):
                value = -float("inf")
            if value > best_value:
                best_value = value
                best_name = str(row.get("modality", "unknown"))
        return best_name

    out["dominant_rms_modality"] = _best_by("rms")
    out["dominant_kurtosis_modality"] = _best_by("kurtosis")
    out["dominant_high_frequency_modality"] = _best_by("high_freq_ratio")
    return out


class DARSModel(nn.Module):
    def __init__(self, config: TrainConfig, modalities: List[Tuple[str, List[int]]], device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.modalities = modalities
        self.num_classes = len(FAULT_NAMES)
        self.physical_dim = len(modalities) * 10
        self.encoder = MultimodalTCNEncoder(
            modalities=modalities,
            window_size=config.window_size,
            input_channels=config.input_channels,
            hidden=config.branch_hidden,
            feature_dim=config.feature_dim,
            tcn_layers=config.tcn_layers,
            dropout=config.dropout,
        ).to(device)
        self.predictor = SmallPredictor(config.feature_dim, self.num_classes, config.dropout).to(device)
        self.feature_tokenizer = FeatureTokenAligner(
            feature_dim=config.feature_dim,
            token_count=int(getattr(config, "llm_feature_token_count", 8)),
            token_dim=int(getattr(config, "llm_feature_token_dim", 16)),
            dropout=config.dropout,
        ).to(device)
        self.token_predictor = nn.Linear(
            int(getattr(config, "llm_feature_token_dim", 16)), self.num_classes
        ).to(device)
        # IMPORTANT: Stage-1 small-model training must be independent of the Router
        # architecture. V6 initialized a logistic Router before the pairwise modules.
        # Initializing the larger V7 MLP here consumed extra global RNG values and
        # unintentionally changed DataLoader shuffling and signal augmentation, which
        # changed the small-model checkpoint. Keep the V6-compatible placeholder for
        # Stage 1, then replace it with the requested Router only after the small model
        # has been selected.
        self.requested_router_type = config.router_type
        self.active_router_type = "logistic"
        self.router = RiskRouter(self.active_router_type).to(device)
        self.pair_input_dim = config.feature_dim + self.physical_dim + len(modalities)
        # Keep the V7.1 four-pair placeholder during Stage 1 so the small-model
        # random trajectory remains unchanged. Expand to all pairs only after the
        # small-model checkpoint has been selected.
        self.requested_pairwise_all_pairs = bool(config.pairwise_all_pairs)
        self.pairwise_pairs = CONFUSION_PAIRS
        self.pairwise_advisors = nn.ModuleDict({
            self.pair_key(pair): PairwiseAdvisor(self.pair_input_dim)
            for pair in self.pairwise_pairs
        }).to(device)
        self.memory = PrototypeMemory(self.num_classes, len(DOMAIN_TO_ID), config.feature_dim)
        self.reviewer = StructuredReviewer(
            mode=config.llm_mode,
            model_path=config.llm_model_path,
            adapter_path=config.llm_adapter_path,
            dtype_name=config.llm_dtype,
            max_input_tokens=config.llm_max_input_tokens,
            max_new_tokens=config.llm_max_new_tokens,
            device=device,
            load_immediately=False,
        )


    def initialize_requested_router(self) -> None:
        """Replace the Stage-1 compatibility Router with the requested Router."""
        if self.active_router_type == self.requested_router_type:
            return
        self.router = RiskRouter(self.requested_router_type).to(self.device)
        self.active_router_type = self.requested_router_type

    def initialize_requested_pairwise_advisors(self) -> None:
        """Expand the pairwise bank after Stage 1 without perturbing its RNG."""
        requested_pairs = ALL_FAULT_PAIRS if self.requested_pairwise_all_pairs else CONFUSION_PAIRS
        if tuple(self.pairwise_pairs) == tuple(requested_pairs):
            return
        old_state = self.pairwise_advisors.state_dict()
        new_bank = nn.ModuleDict({
            self.pair_key(pair): PairwiseAdvisor(self.pair_input_dim)
            for pair in requested_pairs
        }).to(self.device)
        # Preserve compatible placeholder weights; all pairs are trained afterward.
        new_bank.load_state_dict(old_state, strict=False)
        self.pairwise_pairs = requested_pairs
        self.pairwise_advisors = new_bank

    @staticmethod
    def pair_key(pair: Tuple[int, int]) -> str:
        return f"{pair[0]}_{pair[1]}"

    def pairwise_input(self, z: torch.Tensor, physical: torch.Tensor, modality_weights: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.detach(), physical.detach(), modality_weights.detach()], dim=1)

    def risk_features(self, z: torch.Tensor) -> torch.Tensor:
        """Build DG-safe router features from encoder representations only."""
        zn = F.normalize(z.detach(), dim=1)
        feature_std = zn.std(dim=1, unbiased=False)
        feature_norm = z.detach().norm(dim=1)
        feature_abs_mean = z.detach().abs().mean(dim=1)
        return torch.stack([feature_std, feature_norm, feature_abs_mean], dim=1).detach()


def make_router_targets(
    pred: torch.Tensor,
    labels: torch.Tensor,
    risk_features: torch.Tensor,
    mode: str,
    positive_quantile: float,
) -> torch.Tensor:
    """Build router labels.

    error: only small-model mistakes are positives.
    hybrid: mistakes are positives, plus upper-tail risk-evidence samples.

    The hybrid mode avoids a degenerate router when held-out source validation is
    perfectly classified, which is common on small source-domain PU splits.
    """
    error_targets = (pred != labels).long()
    if mode == "error":
        return error_targets
    if mode != "hybrid":
        raise ValueError("router_target_mode must be error or hybrid")
    raw_risk = risk_features.mean(dim=1)
    q = min(max(float(positive_quantile), 0.50), 0.99)
    threshold = torch.quantile(raw_risk.detach(), q)
    tail_targets = (raw_risk >= threshold).long()
    return torch.maximum(error_targets, tail_targets)


def enable_dropout(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_predict(model: DARSModel, x: torch.Tensor, samples: int):
    model.encoder.eval()
    model.predictor.eval()
    z, modality_weights = model.encoder(x)
    probs = []
    samples = max(1, samples)
    if samples > 1:
        enable_dropout(model.predictor)
    for _ in range(samples):
        logits = model.predictor(z)
        probs.append(F.softmax(logits, dim=1))
    probs_t = torch.stack(probs, dim=0)
    mean_prob = probs_t.mean(dim=0)
    logits_mean = torch.log(mean_prob.clamp_min(1e-8))
    entropy = -(mean_prob * torch.log(mean_prob.clamp_min(1e-8))).sum(dim=1) / math.log(mean_prob.size(1))
    variance = probs_t.var(dim=0).mean(dim=1) if samples > 1 else torch.zeros_like(entropy)
    return z, logits_mean, mean_prob, entropy, variance, modality_weights


@torch.no_grad()
@torch.no_grad()
def collect_features(model: DARSModel, loader: DataLoader):
    model.eval()
    zs, ys, ds, ps = [], [], [], []
    for x, labels, domains in loader:
        x = x.to(model.device)
        z, _ = model.encoder(x)
        zs.append(z.cpu())
        ys.append(labels.cpu())
        ds.append(domains.cpu())
        phys = extract_physical_features(x, model.config.window_size, model.config.input_channels, model.modalities)
        ps.append(phys.cpu())
    return torch.cat(zs), torch.cat(ys), torch.cat(ds), torch.cat(ps)



@torch.no_grad()
def refresh_prototypes(model: DARSModel, loader: DataLoader) -> None:
    z, labels, domains, physical = collect_features(model, loader)
    model.memory.fit(z, labels, domains, physical)


def build_evidence(
    idx: int,
    mean_prob: torch.Tensor,
    entropy: torch.Tensor,
    variance: torch.Tensor,
    proto_dist: torch.Tensor,
    domain_shift: torch.Tensor,
    nearest_class: torch.Tensor,
    modality_weights: torch.Tensor,
    domains: torch.Tensor,
    feature_tokens: Optional[torch.Tensor] = None,
    physical_features: Optional[torch.Tensor] = None,
    modality_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    probs = mean_prob[idx]
    top_prob, top_idx = torch.topk(probs, k=min(3, probs.numel()))
    pred = int(top_idx[0].item())
    token_rows, token_summary = summarize_feature_tokens(
        feature_tokens[idx] if feature_tokens is not None else torch.empty(0)
    )
    physical_summary = summarize_physical_features(
        physical_features[idx] if physical_features is not None else None,
        modality_names=modality_names,
    )
    candidate_labels = [
        {"label_id": int(c.item()), "label": FAULT_NAMES[int(c.item())]}
        for c in top_idx
    ]
    candidate_knowledge = {str(int(c.item())): FAULT_KNOWLEDGE[int(c.item())] for c in top_idx}
    resolved_modality_names = list(modality_names or ["motor_current", "vibration"])
    return {
        "modality_names": resolved_modality_names,
        # Small-model fields are retained for reporting and gate accounting only.
        # They are not included in the LLM prompt by build_reviewer_user_message().
        "small_model_label_id": pred,
        "small_model_label": FAULT_NAMES[pred],
        "top3": [{"label_id": int(c.item()), "label": FAULT_NAMES[int(c.item())], "prob": float(p.item())} for p, c in zip(top_prob, top_idx)],
        "small_model_confidence": float(top_prob[0].item()),
        "uncertainty": float(entropy[idx].item()),
        "mc_variance": float(variance[idx].item()),
        "prototype_distance": float(proto_dist[idx].item()),
        "domain_shift": float(domain_shift[idx].item()),
        "nearest_prototype_label_id": int(nearest_class[idx].item()),
        "nearest_prototype_label": FAULT_NAMES[int(nearest_class[idx].item())],
        "domain": ID_TO_DOMAIN.get(int(domains[idx].item()), str(int(domains[idx].item()))),
        "modality_weights": [float(v) for v in modality_weights[idx].detach().cpu().tolist()],
        # V8.8 independent LLM evidence.
        "candidate_labels": candidate_labels,
        "candidate_knowledge": candidate_knowledge,
        "aligned_feature_tokens": token_rows,
        "feature_token_summary": token_summary,
        "physical_summary": physical_summary,
    }


def build_sample_maintenance_revision_context(
    evidence: Dict[str, object],
    report_row: Dict[str, object],
) -> Dict[str, object]:
    """Build a target-label-free context for routed-sample maintenance review."""
    current_label = int(report_row.get("final_label_id", report_row.get("small_label_id", 0)))
    candidate_labels = list(evidence.get("candidate_labels", []))
    candidate_ids: List[int] = []
    for item in candidate_labels:
        try:
            label_id = int(item.get("label_id"))
        except (TypeError, ValueError, AttributeError):
            continue
        if label_id in FAULT_NAMES and label_id not in candidate_ids:
            candidate_ids.append(label_id)
    if current_label not in candidate_ids:
        candidate_labels.append({
            "label_id": current_label,
            "label": FAULT_NAMES.get(current_label, str(current_label)),
        })
        candidate_ids.append(current_label)
    return {
        "task": "post_diagnosis_sample_maintenance_with_consensus_revision",
        "scope": "router_selected_sample",
        "revision_allowed": True,
        "current_diagnosis_label_id": current_label,
        "current_diagnosis_name": FAULT_NAMES.get(current_label, str(current_label)),
        "current_diagnosis_confidence": _report_float(report_row.get("final_confidence", 0.0)),
        "candidate_labels": candidate_labels,
        "candidate_fault_semantics": {
            str(label_id): FAULT_KNOWLEDGE.get(label_id, "") for label_id in candidate_ids
        },
        "candidate_evidence_profiles": build_llm_candidate_profiles(evidence),
        "feature_token_summary": evidence.get("feature_token_summary", {}),
        "physical_summary": compact_llm_physical_summary(evidence),
        "modality_names": evidence.get("modality_names", []),
        "modality_weights": evidence.get("modality_weights", []),
        "risk_probability": _report_float(evidence.get("risk_probability", 0.0)),
        "uncertainty_entropy": _report_float(evidence.get("uncertainty", 0.0)),
        "mc_variance": _report_float(evidence.get("mc_variance", 0.0)),
        "prototype_distance": _report_float(evidence.get("prototype_distance", 0.0)),
        "domain_shift": _report_float(evidence.get("domain_shift", 0.0)),
        "operating_domain": str(evidence.get("domain", "unknown")),
        "constraints": {
            "no_target_ground_truth": True,
            "proposed_label_must_be_candidate_argmax": True,
            "label_change_requires_external_consensus_gate": True,
            "maintenance_cannot_edit_metrics": True,
        },
    }


def build_class_maintenance_context(
    label: int,
    class_rows: Sequence[Dict[str, object]],
    modalities: Sequence[Tuple[str, Sequence[int]]],
    config: "TrainConfig",
) -> Dict[str, object]:
    """Build the final class-level maintenance context after labels are frozen."""
    rows = list(class_rows)
    modality_names = [str(name) for name, _indices in modalities]
    target_conditions = []
    for domain_name in config.target_domains:
        meta = DOMAIN_METADATA.get(str(domain_name), {})
        target_conditions.append({
            "domain": str(domain_name),
            "speed_rpm": meta.get("speed_rpm"),
            "load_torque_nm": meta.get("load_torque_nm"),
            "radial_force_n": meta.get("radial_force_n"),
        })
    attrs = dict(FAULT_MAINTENANCE_ATTRIBUTES.get(int(label), {}))
    raceway_map_zh = {"inner_race": "内圈", "outer_race": "外圈"}
    required_component_terms_zh = [
        raceway_map_zh[raceway]
        for raceway in attrs.get("affected_raceways", [])
        if raceway in raceway_map_zh
    ]
    policy_priority, policy_priority_zh, policy_action_timeline_zh = (
        determine_maintenance_policy_from_attributes(int(label))
    )
    return {
        "task": "post_diagnosis_zero_shot_class_maintenance_generation",
        "scope": "class_level_dataset_semantics_and_available_sensing_context",
        "revision_allowed": False,
        "current_diagnosis_label_id": int(label),
        "current_diagnosis_name": FAULT_NAMES.get(int(label), str(label)),
        "candidate_labels": [{
            "label_id": int(label),
            "label": FAULT_NAMES.get(int(label), str(label)),
        }],
        "fault_semantics": FAULT_KNOWLEDGE.get(int(label), ""),
        "fault_attributes": attrs,
        "required_component_terms_zh": required_component_terms_zh,
        "policy_priority": int(policy_priority),
        "policy_priority_zh": str(policy_priority_zh),
        "policy_action_timeline_zh": str(policy_action_timeline_zh),
        "healthy_class_constraints": (
            {
                "must_state_no_fault_indication": True,
                "must_reference_healthy_baseline": True,
                "required_monitoring_modalities": ["motor_current", "vibration"],
                "forbid_positive_fault_claim": True,
                "forbid_immediate_repair_replacement_shutdown": True,
            }
            if int(label) == 0 else {}
        ),
        "predicted_sample_count": len(rows),
        "available_sensing_modalities": modality_names,
        "target_operating_conditions": target_conditions,
        "constraints": {
            "diagnosis_is_frozen": True,
            "no_target_ground_truth": True,
            "do_not_propose_label_change": True,
            "maintenance_cannot_edit_metrics": True,
            "priority_and_timeline_locked_by_deterministic_safety_policy": True,
            "llm_generates_maintenance_content_not_priority": True,
            "fixed_maintenance_template_not_supplied": True,
        },
    }


def apply_pairwise_advisor(
    model: DARSModel,
    z_i: torch.Tensor,
    physical_i: torch.Tensor,
    weights_i: torch.Tensor,
    top3: List[int],
) -> Optional[Dict[str, object]]:
    if not model.config.pairwise_enabled:
        return None
    top3_set = set(top3)
    for pair in CONFUSION_PAIRS:
        if pair[0] in top3_set and pair[1] in top3_set:
            pair_input = model.pairwise_input(
                z_i.view(1, -1),
                physical_i.view(1, -1),
                weights_i.view(1, -1),
            )
            logits = model.pairwise_advisors[model.pair_key(pair)](pair_input)
            prob = F.softmax(logits, dim=1).squeeze(0)
            choice = int(prob.argmax().item())
            label = pair[choice]
            return {
                "pair": [int(pair[0]), int(pair[1])],
                "pair_names": [FAULT_NAMES[pair[0]], FAULT_NAMES[pair[1]]],
                "final_label": int(label),
                "final_label_name": FAULT_NAMES[int(label)],
                "prob_pair_first": float(prob[0].item()),
                "prob_pair_second": float(prob[1].item()),
            }
    return None



@torch.no_grad()
def evaluate_small_model(model: DARSModel, loader: DataLoader) -> Dict[str, float]:
    """Evaluate only the frozen encoder/predictor; Router is not involved."""
    model.encoder.eval()
    model.predictor.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    prob_rows: List[np.ndarray] = []
    uncertainty_sum = confidence_sum = 0.0
    for x, labels, _domains in loader:
        x = x.to(model.device, non_blocking=True)
        _z, logits, mean_prob, entropy, _variance, _weights = mc_predict(
            model, x, model.config.mc_dropout_samples
        )
        pred = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        prob_rows.append(mean_prob.detach().cpu().numpy())
        uncertainty_sum += float(entropy.sum().item())
        confidence_sum += float(mean_prob.max(dim=1).values.sum().item())
    n = max(len(y_true), 1)
    labels_order = sorted(FAULT_NAMES)
    basic = label_metrics(y_true, y_pred, labels_order)
    probs = np.concatenate(prob_rows, axis=0) if prob_rows else np.empty((0, len(labels_order)))
    return {
        "small_acc": float(basic["acc"]),
        "small_balanced_acc": float(basic["balanced_acc"]),
        "small_macro_precision": float(basic["macro_precision"]),
        "small_macro_recall": float(basic["macro_recall"]),
        "small_macro_f1": float(basic["macro_f1"]),
        "small_weighted_f1": float(basic["weighted_f1"]),
        "small_mcc": float(basic["mcc"]),
        "small_kappa": float(basic["kappa"]),
        "small_top3_acc": 100.0 * top_k_accuracy_score(
            y_true, probs, k=min(3, len(labels_order)), labels=labels_order
        ) if len(y_true) else 0.0,
        "small_nll": float(log_loss(y_true, probs, labels=labels_order)) if len(y_true) else 0.0,
        "small_ece": expected_calibration_error(probs, y_true),
        "avg_uncertainty": uncertainty_sum / n,
        "avg_confidence": confidence_sum / n,
    }


@torch.no_grad()
@torch.no_grad()
def evaluate_small_model_robust(model: DARSModel, loader: DataLoader, seed: int) -> Dict[str, float]:
    """Fixed source-only stress validation used only for checkpoint tie-breaking.

    The RNG is forked and restored, so this evaluation cannot alter the next
    epoch's training shuffle/augmentation sequence. No target-domain sample or
    label is used.
    """
    devices = [model.device.index or 0] if model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        model.encoder.eval()
        model.predictor.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        for x, labels, _domains in loader:
            x = x.to(model.device, non_blocking=True)
            x = augment_batch(
                x,
                model.config.window_size,
                model.config.input_channels,
                model.config.robust_val_time_shift,
                model.config.robust_val_noise_std,
                model.config.robust_val_channel_dropout_p,
                model.config.robust_val_amplitude_jitter,
                model.config.robust_val_time_mask_ratio,
            )
            _z, logits, _mean_prob, _entropy, _variance, _weights = mc_predict(
                model, x, max(4, min(model.config.mc_dropout_samples, 8))
            )
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        labels_order = sorted(FAULT_NAMES)
        basic = label_metrics(y_true, y_pred, labels_order)
        return {
            "robust_acc": float(basic["acc"]),
            "robust_macro_f1": float(basic["macro_f1"]),
            "robust_mcc": float(basic["mcc"]),
        }


@torch.no_grad()
def _router_features_and_errors(
    model: DARSModel,
    x: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z, logits, mean_prob, entropy, variance, _weights = mc_predict(
        model, x, model.config.mc_dropout_samples
    )
    pred = logits.argmax(dim=1)
    rf = model.risk_features(z)
    risk_prob = F.softmax(model.router(rf), dim=1)[:, 1]
    raw_risk = rf.mean(dim=1)
    errors = (pred != labels).long()
    return rf, errors, risk_prob, raw_risk


@torch.no_grad()
def collect_router_calibration_data(
    model: DARSModel,
    loader: DataLoader,
    hard_augment_copies: int = 0,
) -> Dict[str, np.ndarray]:
    """Collect Router scores against actual small-model mistakes.

    When the clean source calibration split is perfectly classified, optional
    source-only hard perturbations provide real prediction errors for threshold
    calibration. No target-domain sample is used.
    """
    model.eval()
    scores: List[float] = []
    error_targets: List[int] = []
    raw_risks: List[float] = []
    small_correct: List[int] = []
    is_clean: List[int] = []

    for x, labels, _domains in loader:
        x = x.to(model.device, non_blocking=True)
        labels = labels.to(model.device, non_blocking=True)
        variants: List[Tuple[torch.Tensor, bool]] = [(x, True)]
        for _ in range(max(0, int(hard_augment_copies))):
            hard_x = augment_batch(
                x,
                model.config.window_size,
                model.config.input_channels,
                time_shift=max(int(model.config.router_hard_time_shift), model.config.time_shift),
                noise_std=max(float(model.config.router_hard_noise_std), model.config.noise_std),
                channel_dropout_p=max(
                    float(model.config.router_hard_channel_dropout_p),
                    model.config.channel_dropout_p,
                ),
            )
            variants.append((hard_x, False))

        for variant_x, clean_flag in variants:
            _rf, errors, risk_prob, raw_risk = _router_features_and_errors(
                model, variant_x, labels
            )
            scores.extend(risk_prob.cpu().tolist())
            error_targets.extend(errors.cpu().tolist())
            raw_risks.extend(raw_risk.cpu().tolist())
            small_correct.extend((1 - errors).cpu().tolist())
            is_clean.extend([int(clean_flag)] * int(errors.numel()))

    return {
        "scores": np.asarray(scores, dtype=np.float64),
        "error_targets": np.asarray(error_targets, dtype=np.int64),
        "raw_risks": np.asarray(raw_risks, dtype=np.float64),
        "small_correct": np.asarray(small_correct, dtype=np.int64),
        "is_clean": np.asarray(is_clean, dtype=np.int64),
    }


def calibrate_router_threshold(
    model: DARSModel,
    loader: DataLoader,
    min_error_recall: float,
    steps: int,
) -> Dict[str, Any]:
    """Calibrate a descriptive source-domain Router threshold.

    The final evaluator uses global budget-aware ranking by default, so this
    threshold is mainly retained for diagnostics and controlled budget studies. Calibration is
    nevertheless grounded in actual mistakes rather than arbitrary upper-tail
    labels whenever possible.
    """
    clean_data = collect_router_calibration_data(model, loader, hard_augment_copies=0)
    data = clean_data
    target_source = "actual_clean_small_model_errors"

    if int(clean_data["error_targets"].sum()) == 0:
        data = collect_router_calibration_data(
            model,
            loader,
            hard_augment_copies=max(1, int(model.config.router_hard_augment_copies)),
        )
        target_source = "actual_errors_from_source_hard_augmentation"

    scores = data["scores"]
    targets = data["error_targets"].copy()
    small_correct = data["small_correct"]
    if scores.size == 0:
        raise ValueError("Router calibration loader produced no samples.")

    if int(targets.sum()) == 0:
        raw_risks = data["raw_risks"]
        q = min(max(float(model.config.router_positive_quantile), 0.50), 0.99)
        tail_threshold = float(np.quantile(raw_risks, q))
        targets = (raw_risks >= tail_threshold).astype(np.int64)
        target_source = "upper_tail_last_resort"

    steps = max(int(steps), 3)
    candidates = np.unique(
        np.concatenate([
            np.linspace(0.0, 1.0, steps, dtype=np.float64),
            scores,
            np.asarray([0.0, 1.0], dtype=np.float64),
        ])
    )

    rows: List[Dict[str, float]] = []
    for threshold in candidates:
        routed = scores >= threshold
        tp = int(np.logical_and(routed, targets == 1).sum())
        fp = int(np.logical_and(routed, targets == 0).sum())
        fn = int(np.logical_and(~routed, targets == 1).sum())
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        trigger_rate = float(routed.mean())
        accepted = ~routed
        selective_acc = float(small_correct[accepted].mean()) if accepted.any() else 0.0
        rows.append({
            "threshold": float(threshold),
            "error_recall": float(recall),
            "error_precision": float(precision),
            "router_f1": float(f1),
            "trigger_rate": trigger_rate,
            "coverage": 1.0 - trigger_rate,
            "selective_acc": selective_acc,
        })

    requested_recall = min(max(float(min_error_recall), 0.0), 1.0)
    feasible = [row for row in rows if row["error_recall"] + 1e-12 >= requested_recall]
    if feasible:
        chosen = min(
            feasible,
            key=lambda row: (
                row["trigger_rate"],
                -row["selective_acc"],
                -row["error_precision"],
                -row["threshold"],
            ),
        )
        selection_rule = "minimum_trigger_rate_at_required_error_recall"
    else:
        chosen = max(
            rows,
            key=lambda row: (
                row["router_f1"],
                row["error_recall"],
                -row["trigger_rate"],
            ),
        )
        selection_rule = "maximum_router_f1_fallback"

    model.config.router_threshold = float(chosen["threshold"])
    auc = 0.0
    try:
        if len(np.unique(targets)) > 1:
            auc = float(roc_auc_score(targets, scores))
    except Exception:
        auc = 0.0

    clean_scores = clean_data["scores"]
    clean_trigger_rate = (
        float((clean_scores >= model.config.router_threshold).mean())
        if clean_scores.size else 0.0
    )
    result = {
        **chosen,
        "risk_auc": auc,
        "positive_targets": float(int(targets.sum())),
        "num_samples": float(scores.size),
        "clean_trigger_rate": clean_trigger_rate,
        "requested_error_recall": requested_recall,
        "target_source": target_source,
        "selection_rule": selection_rule,
    }
    return result


def train_router(
    model: DARSModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> Dict[str, float]:
    """Train Router on clean and hard-augmented source validation samples.

    Actual small-model errors are positive targets. Class weighting prevents the
    abundant correct samples from collapsing the Router into an always-safe or
    always-risky classifier.
    """
    if epochs <= 0:
        return {"router_loss": 0.0, "risk_target_rate": 0.0, "risk_auc": 0.0}
    for p in list(model.encoder.parameters()) + list(model.predictor.parameters()) + list(model.feature_tokenizer.parameters()) + list(model.token_predictor.parameters()):
        p.requires_grad_(False)
    for p in model.router.parameters():
        p.requires_grad_(True)

    final_loss_sum = final_total = final_target_sum = final_nb = 0
    final_targets: List[int] = []
    final_scores: List[float] = []

    for epoch in range(max(1, int(epochs))):
        loss_sum = total = target_sum = nb = 0
        epoch_targets: List[int] = []
        epoch_scores: List[float] = []
        model.router.train()

        for x, labels, _domains in loader:
            x = x.to(model.device, non_blocking=True)
            labels = labels.to(model.device, non_blocking=True)
            variants = [x]
            for _ in range(max(0, int(model.config.router_hard_augment_copies))):
                variants.append(
                    augment_batch(
                        x,
                        model.config.window_size,
                        model.config.input_channels,
                        time_shift=max(
                            int(model.config.router_hard_time_shift),
                            model.config.time_shift,
                        ),
                        noise_std=max(
                            float(model.config.router_hard_noise_std),
                            model.config.noise_std,
                        ),
                        channel_dropout_p=max(
                            float(model.config.router_hard_channel_dropout_p),
                            model.config.channel_dropout_p,
                        ),
                    )
                )

            rf_parts: List[torch.Tensor] = []
            target_parts: List[torch.Tensor] = []
            with torch.no_grad():
                for variant_x in variants:
                    rf, errors, _risk_prob, _raw_risk = _router_features_and_errors(
                        model, variant_x, labels
                    )
                    if model.config.router_target_mode == "hybrid":
                        z, logits, mean_prob, entropy, variance, _weights = mc_predict(
                            model, variant_x, model.config.mc_dropout_samples
                        )
                        pred = logits.argmax(dim=1)
                        proto_dist, domain_shift, _nearest, _nearest_dist = prototype_scores(
                            z, pred, model.memory
                        )
                        rf = model.risk_features(z)
                        targets = make_router_targets(
                            pred,
                            labels,
                            rf,
                            "hybrid",
                            model.config.router_positive_quantile,
                        )
                    else:
                        targets = errors
                    rf_parts.append(rf)
                    target_parts.append(targets)

            rf_all = torch.cat(rf_parts, dim=0)
            targets_all = torch.cat(target_parts, dim=0)
            router_logits = model.router(rf_all)

            positives = int(targets_all.sum().item())
            negatives = int(targets_all.numel() - positives)
            class_weight = None
            if positives > 0 and negatives > 0:
                positive_weight = min(
                    float(model.config.router_positive_weight_cap),
                    max(1.0, negatives / max(positives, 1)),
                )
                class_weight = torch.tensor(
                    [1.0, positive_weight],
                    device=model.device,
                    dtype=router_logits.dtype,
                )
            loss = F.cross_entropy(router_logits, targets_all, weight=class_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.router.parameters(), 5.0)
            optimizer.step()

            probs = F.softmax(router_logits.detach(), dim=1)[:, 1]
            epoch_targets.extend(targets_all.cpu().tolist())
            epoch_scores.extend(probs.cpu().tolist())
            loss_sum += float(loss.item())
            target_sum += positives
            total += int(targets_all.numel())
            nb += 1

        final_loss_sum, final_total, final_target_sum, final_nb = (
            loss_sum, total, target_sum, nb
        )
        final_targets, final_scores = epoch_targets, epoch_scores

    for p in list(model.encoder.parameters()) + list(model.predictor.parameters()) + list(model.feature_tokenizer.parameters()) + list(model.token_predictor.parameters()):
        p.requires_grad_(True)

    auc = 0.0
    try:
        if len(set(final_targets)) > 1:
            auc = float(roc_auc_score(final_targets, final_scores))
    except Exception:
        auc = 0.0
    return {
        "router_loss": final_loss_sum / max(final_nb, 1),
        "risk_target_rate": 100.0 * final_target_sum / max(final_total, 1),
        "risk_auc": auc,
    }

def train_pairwise_advisors(model: DARSModel, loader: DataLoader, optimizer: torch.optim.Optimizer, epochs: int) -> Dict[str, float]:
    if (not model.config.pairwise_enabled) or epochs <= 0:
        return {"pairwise_loss": 0.0, "pairwise_acc": 0.0, "pairwise_samples": 0.0}
    for p in list(model.encoder.parameters()) + list(model.predictor.parameters()) + list(model.router.parameters()):
        p.requires_grad_(False)
    for p in model.pairwise_advisors.parameters():
        p.requires_grad_(True)
    loss_sum = correct = total = nb = 0
    for _ in range(epochs):
        for x, labels, _domains in loader:
            x, labels = x.to(model.device), labels.to(model.device)
            with torch.no_grad():
                z, weights = model.encoder(x)
                physical = extract_physical_features(x, model.config.window_size, model.config.input_channels, model.modalities)
                pair_input = model.pairwise_input(z, physical, weights)
            batch_losses = []
            for pair in model.pairwise_pairs:
                a, b = pair
                mask = (labels == a) | (labels == b)
                if not mask.any():
                    continue
                targets = (labels[mask] == b).long()
                logits = model.pairwise_advisors[model.pair_key(pair)](pair_input[mask])
                loss = F.cross_entropy(logits, targets)
                batch_losses.append(loss)
                correct += int((logits.argmax(dim=1) == targets).sum().item())
                total += int(targets.numel())
            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            nb += 1
    for p in list(model.encoder.parameters()) + list(model.predictor.parameters()) + list(model.router.parameters()):
        p.requires_grad_(True)
    return {
        "pairwise_loss": loss_sum / max(nb, 1),
        "pairwise_acc": 100.0 * correct / max(total, 1),
        "pairwise_samples": float(total),
    }


def infer_label_from_fault_text(text: object) -> Optional[int]:
    lower = str(text or "").lower()
    for label_id, name in FAULT_NAMES.items():
        if name.lower() in lower:
            return int(label_id)
    return None

def _independent_llm_override_support(
    evidence: Dict[str, object],
    proposed: int,
    fallback_label: int,
) -> Tuple[int, List[str], Optional[str]]:
    """Count independent pre-LLM sources supporting a label change.

    V10.7 deliberately counts *sources*, not weighted votes. The three sources
    are: advisor reranking, nearest source prototype, and the pairwise advisor.
    A default override needs at least two of these three to agree with the LLM.
    """
    reasons: List[str] = []

    try:
        rerank_label = int(evidence.get("advisor_rerank_label_id", -1))
    except (TypeError, ValueError):
        rerank_label = -1
    if rerank_label == int(proposed):
        reasons.append("advisor_rerank")

    try:
        nearest_label = int(evidence.get("nearest_prototype_label_id", -1))
    except (TypeError, ValueError):
        nearest_label = -1
    if nearest_label == int(proposed):
        reasons.append("nearest_prototype")

    pairwise = evidence.get("pairwise_advisor")
    if isinstance(pairwise, dict):
        try:
            pair = [int(x) for x in pairwise.get("pair", [])]
            pair_final = int(pairwise.get("final_label", -1))
            p_first = float(pairwise.get("prob_pair_first", 0.0))
            p_second = float(pairwise.get("prob_pair_second", 0.0))
            pair_conf = max(p_first, p_second)
        except (TypeError, ValueError):
            pair = []
            pair_final = -1
            pair_conf = 0.0
        # Only use pairwise evidence when it explicitly compares the old/new
        # labels. A confident contradiction vetoes the change.
        if int(proposed) in pair and int(fallback_label) in pair and pair_conf >= 0.80:
            if pair_final != int(proposed):
                return len(reasons), reasons, (
                    f"pairwise_contradiction:pair_final={pair_final},conf={pair_conf:.3f}"
                )
            reasons.append("pairwise")

    # De-duplicate defensively, though each source is appended at most once.
    unique_reasons = list(dict.fromkeys(reasons))
    return len(unique_reasons), unique_reasons, None


def _candidate_profile_map(evidence: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    profiles = build_llm_candidate_profiles(evidence)
    out: Dict[int, Dict[str, object]] = {}
    for p in profiles:
        try:
            out[int(p.get("label_id"))] = p
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def _relative_profile_support_decision(
    evidence: Dict[str, object],
    proposed: int,
    fallback_label: int,
    config: TrainConfig,
) -> Tuple[bool, float, float, bool, str]:
    """Check whether non-LLM relative evidence favors the proposed override."""
    profiles = _candidate_profile_map(evidence)
    proposed_profile = profiles.get(int(proposed))
    fallback_profile = profiles.get(int(fallback_label))
    if proposed_profile is None or fallback_profile is None:
        return False, 0.0, 0.0, False, "missing_relative_candidate_profile"
    try:
        proposed_support = float(proposed_profile.get("relative_support", proposed_profile.get("combined_support", 0.0)))
        fallback_support = float(fallback_profile.get("relative_support", fallback_profile.get("combined_support", 0.0)))
    except (TypeError, ValueError):
        return False, 0.0, 0.0, False, "invalid_relative_support"
    min_support = float(getattr(config, "llm_delegate_min_relative_support", 0.0))
    min_gain = float(getattr(config, "llm_delegate_min_relative_support_gain", 0.0))
    if proposed_support < min_support:
        return False, proposed_support, fallback_support, False, (
            f"relative_support_too_low:label={proposed},support={proposed_support:.3f},min={min_support:.3f}"
        )
    gain = proposed_support - fallback_support
    if gain < min_gain:
        return False, proposed_support, fallback_support, False, (
            f"relative_support_gain_too_low:label={proposed},gain={gain:.3f},min={min_gain:.3f},"
            f"small_support={fallback_support:.3f}"
        )
    is_best = bool(
        int(proposed_profile.get("advisor_score_rank", 99)) == 1
        or int(proposed_profile.get("physics_rank", 99)) == 1
        or bool(proposed_profile.get("rerank_match", False))
    )
    return True, proposed_support, fallback_support, is_best, "relative_profile_supports_override"

def apply_llm_revision_gate(
    evidence: Dict[str, object],
    review: Dict[str, object],
    config: TrainConfig,
) -> Tuple[int, bool, str]:
    """V10.7 evidence-consensus delegated-classifier gate.

    A routed LLM may always *keep* the small-model label when its structured
    output is valid. A label *change* is accepted only when the LLM is confident
    enough and at least two independent pre-LLM sources support the same label.
    This is designed to retain useful hard-sample corrections while preventing
    the negative net corrections observed in the V10.5.2 full method.
    """
    fallback_label = int(evidence.get("small_model_label_id", 0))
    candidate_ids = candidate_ids_from_evidence(evidence)
    allowed = set(int(x) for x in candidate_ids)

    if not bool(review.get("valid_json", False)):
        return fallback_label, False, f"llm_consensus_fallback:invalid_json:{review.get('candidate_score_reason','unknown')}"
    if not bool(review.get("candidate_scores_valid", False)):
        return fallback_label, False, f"llm_consensus_fallback:invalid_candidate_scores:{review.get('candidate_score_reason','unknown')}"

    try:
        proposed = int(review.get("final_label", fallback_label))
    except (TypeError, ValueError):
        return fallback_label, False, "llm_consensus_fallback:invalid_final_label"
    if proposed not in allowed:
        return fallback_label, False, f"llm_consensus_fallback:label_outside_candidates:{proposed}"
    if bool(review.get("forbidden_evidence_terms", False)):
        return fallback_label, False, "llm_consensus_fallback:stale_or_unavailable_evidence_terms"
    if not bool(review.get("label_fault_consistent", True)):
        return fallback_label, False, "llm_consensus_fallback:label_fault_mismatch"

    score = float(review.get("confidence", 0.0))
    margin = float(review.get("score_margin", 0.0))
    profiles = _candidate_profile_map(evidence)
    proposed_profile = profiles.get(int(proposed), {})
    fallback_profile = profiles.get(int(fallback_label), {})
    try:
        proposed_rel = float(proposed_profile.get("relative_support", proposed_profile.get("combined_support", 0.0)))
    except (TypeError, ValueError):
        proposed_rel = 0.0
    try:
        fallback_rel = float(fallback_profile.get("relative_support", fallback_profile.get("combined_support", 0.0)))
    except (TypeError, ValueError):
        fallback_rel = 0.0

    # A valid LLM keep decision does not alter the small model and therefore
    # cannot create a harmful override.
    if proposed == fallback_label:
        return proposed, True, (
            f"llm_consensus_keep:label={proposed},score={score:.3f},margin={margin:.3f},"
            f"relative={proposed_rel:.3f}"
        )

    min_conf = float(getattr(config, "llm_delegate_min_confidence", 0.70))
    min_margin = float(getattr(config, "llm_delegate_min_score_margin", 0.20))
    if score < min_conf:
        return fallback_label, False, (
            f"llm_consensus_reject:confidence={score:.3f}<min={min_conf:.3f}"
        )
    if margin < min_margin:
        return fallback_label, False, (
            f"llm_consensus_reject:margin={margin:.3f}<min={min_margin:.3f}"
        )

    # Near-certain small-model outputs are protected unless explicitly relaxed.
    small_conf = float(evidence.get("small_model_confidence", 0.0))
    max_small = float(getattr(config, "llm_delegate_max_small_confidence_for_override", 0.95))
    if small_conf > max_small:
        return fallback_label, False, (
            f"llm_consensus_reject:small_confidence={small_conf:.3f}>max={max_small:.3f}"
        )

    # Healthy<->fault flips receive a stricter LLM confidence requirement.
    if (proposed == 0 or fallback_label == 0):
        normal_min = float(getattr(config, "llm_normal_flip_min_confidence", 0.90))
        if score < normal_min:
            return fallback_label, False, (
                f"llm_consensus_reject:normal_boundary_confidence={score:.3f}<min={normal_min:.3f}"
            )

    support_count, support_sources, contradiction = _independent_llm_override_support(
        evidence, proposed, fallback_label
    )
    if contradiction is not None:
        return fallback_label, False, f"llm_consensus_reject:{contradiction}"
    min_supporters = int(getattr(config, "llm_delegate_min_override_support", 2))
    if support_count < min_supporters:
        return fallback_label, False, (
            f"llm_consensus_reject:support={support_count}<min={min_supporters};"
            f"sources={','.join(support_sources) or 'none'}"
        )

    rel_ok, proposed_rel2, fallback_rel2, _is_best, rel_reason = _relative_profile_support_decision(
        evidence, proposed, fallback_label, config
    )
    if not rel_ok:
        return fallback_label, False, f"llm_consensus_reject:{rel_reason}"
    proposed_rel = proposed_rel2
    fallback_rel = fallback_rel2

    # The sample has already passed Router selection. Optionally require only a
    # small extra risk margin; the V10.7 default is zero to avoid blocking strong
    # consensus corrections solely because Router calibration is conservative.
    risk = float(evidence.get("risk_probability", 0.0))
    router_threshold = float(getattr(config, "router_threshold", 0.0))
    risk_margin = float(getattr(config, "llm_delegate_min_risk_margin", 0.0))
    if risk + 1e-12 < router_threshold + risk_margin:
        return fallback_label, False, (
            f"llm_consensus_reject:risk={risk:.3f}<required={router_threshold + risk_margin:.3f}"
        )

    return proposed, True, (
        f"llm_consensus_change:small={fallback_label},label={proposed},score={score:.3f},"
        f"margin={margin:.3f},support={support_count},sources={'+'.join(support_sources)},"
        f"relative={proposed_rel:.3f},small_relative={fallback_rel:.3f}"
    )


def apply_maintenance_revision_gate(
    evidence: Dict[str, object],
    maintenance_review: Dict[str, object],
    current_label: int,
    current_confidence: float,
    config: TrainConfig,
) -> Tuple[int, bool, str]:
    """Legacy compatibility helper; inactive in the V10.13 report-only build."""
    if not bool(getattr(config, "maintenance_revision_enabled", True)):
        return int(current_label), False, "maintenance_revision_disabled"
    if not bool(maintenance_review.get("maintenance_generation_valid", False)):
        return int(current_label), False, (
            "maintenance_revision_reject:invalid_maintenance_output:"
            + str(maintenance_review.get("maintenance_generation_status", "unknown"))
        )

    gate_evidence = dict(evidence)
    # The existing consensus gate protects its fallback label. During this second
    # pass, the fallback is the already-completed diagnostic decision rather than
    # necessarily the original small-model top-1.
    gate_evidence["small_model_label_id"] = int(current_label)
    gate_evidence["small_model_label"] = FAULT_NAMES.get(int(current_label), str(current_label))
    gate_evidence["small_model_confidence"] = float(current_confidence)
    final_label, accepted, reason = apply_llm_revision_gate(
        gate_evidence, maintenance_review, config
    )
    proposed = int(maintenance_review.get("proposed_label", current_label))
    if proposed == int(current_label):
        return int(current_label), False, "maintenance_consistency_keep:" + reason
    if not accepted or int(final_label) == int(current_label):
        return int(current_label), False, "maintenance_revision_reject:" + reason
    return int(final_label), True, "maintenance_revision_accept:" + reason


@torch.no_grad()
def evaluate(
    model: DARSModel,
    loader: DataLoader,
    use_reviewer: bool = True,
    allow_llm: bool = True,
    split_name: str = "evaluation",
) -> Dict[str, object]:
    """Evaluate the complete selective small/large-model system."""
    model.eval()
    sync_device(model.device)
    evaluation_start = time.perf_counter()
    feature_start = evaluation_start

    y_true: List[int] = []
    y_small: List[int] = []
    y_final: List[int] = []
    small_probability_rows: List[np.ndarray] = []
    risk_targets_all: List[int] = []
    risk_scores_all: List[float] = []
    unc_sum = var_sum = proto_sum = shift_sum = conf_sum = 0.0
    threshold_candidate_count = 0
    confidence_blocked_count = 0
    eligible_candidate_count = 0
    candidate_pool: List[Dict[str, object]] = []
    evidence_samples: List[Dict[str, object]] = []
    prediction_reports: List[Dict[str, object]] = []
    total_batches = len(loader)

    for batch_idx, (x, labels, domains) in enumerate(loader, start=1):
        base_index = len(y_true)
        x = x.to(model.device, non_blocking=True)
        labels = labels.to(model.device, non_blocking=True)
        domains = domains.to(model.device, non_blocking=True)
        z, logits, mean_prob, entropy, variance, modality_weights = mc_predict(
            model, x, model.config.mc_dropout_samples
        )
        pred = logits.argmax(dim=1)
        feature_tokens = model.feature_tokenizer(z)
        physical_batch = extract_physical_features(
            x, model.config.window_size, model.config.input_channels, model.modalities
        )
        proto_dist, domain_shift, nearest_class, _nearest_dist = prototype_scores(
            z, pred, model.memory
        )
        rf = model.risk_features(z)
        risk_prob = F.softmax(model.router(rf), dim=1)[:, 1]
        small_confidence = mean_prob.max(dim=1).values
        threshold_mask = risk_prob >= model.config.router_threshold
        confidence_gate = float(getattr(model.config, "router_confidence_gate", 1.0))
        if confidence_gate < 1.0:
            confidence_mask = small_confidence <= confidence_gate
        else:
            confidence_mask = torch.ones_like(threshold_mask, dtype=torch.bool)
        route_mask = threshold_mask & confidence_mask
        threshold_candidate_count += int(threshold_mask.sum().item())
        confidence_blocked_count += int((threshold_mask & (~confidence_mask)).sum().item())
        eligible_candidate_count += int(route_mask.sum().item())

        if use_reviewer:
            # V7.4: threshold-first routing. The LLM budget is only a cap inside
            # genuinely hard samples, rather than forcing a fixed top-risk ratio
            # from every target domain.
            candidate_indices_t = torch.nonzero(route_mask, as_tuple=False).flatten()
            candidate_indices = candidate_indices_t.detach().cpu().tolist()

            if candidate_indices:
                routed_x = x.index_select(0, candidate_indices_t)
                routed_physical = extract_physical_features(
                    routed_x,
                    model.config.window_size,
                    model.config.input_channels,
                    model.modalities,
                )
                for local_idx, sample_idx in enumerate(candidate_indices):
                    evidence = build_evidence(
                        sample_idx,
                        mean_prob,
                        entropy,
                        variance,
                        proto_dist,
                        domain_shift,
                        nearest_class,
                        modality_weights,
                        domains,
                        feature_tokens,
                        physical_batch,
                        [name for name, _indices in model.modalities],
                    )
                    top3 = [int(item["label_id"]) for item in evidence["top3"]]
                    top3_probs = [float(item["prob"]) for item in evidence["top3"]]
                    physical_i = routed_physical[local_idx]
                    rerank_label, rerank_details = candidate_evidence_scores(
                        z[sample_idx],
                        physical_i,
                        top3,
                        top3_probs,
                        model.memory,
                        model.config,
                    )
                    evidence["risk_probability"] = float(risk_prob[sample_idx].item())
                    evidence["advisor_rerank_label_id"] = int(rerank_label)
                    evidence["advisor_rerank_label"] = FAULT_NAMES[int(rerank_label)]
                    evidence["advisor_rerank_details"] = rerank_details
                    pairwise_review = apply_pairwise_advisor(
                        model,
                        z[sample_idx],
                        physical_i,
                        modality_weights[sample_idx],
                        top3,
                    )
                    if pairwise_review is not None:
                        evidence["pairwise_advisor"] = pairwise_review

                    small_label = int(evidence["small_model_label_id"])
                    disagreement_bonus = 0.0
                    if int(rerank_label) != small_label:
                        disagreement_bonus += 0.20
                    if int(nearest_class[sample_idx].item()) != small_label:
                        disagreement_bonus += 0.10
                    if (
                        pairwise_review is not None
                        and int(pairwise_review.get("final_label", small_label)) != small_label
                    ):
                        disagreement_bonus += 0.15
                    priority = (
                        float(risk_prob[sample_idx].item())
                        + 0.20 * float(entropy[sample_idx].item())
                        + 0.10 * float(domain_shift[sample_idx].item())
                        + 0.10 * (1.0 - float(mean_prob[sample_idx].max().item()))
                        + disagreement_bonus
                    )
                    candidate_pool.append({
                        "global_index": base_index + sample_idx,
                        "priority": priority,
                        "evidence": evidence,
                        # V8.8.1 FIX: pass the actual per-sample top-3 candidate set
                        # to the LLM output parser. V8.8 accidentally passed all 8
                        # labels here, so valid top-3 candidate_scores were rejected as
                        # missing non-candidate labels such as 0/1/3/4/6.
                        "top3": top3,
                    })

        risk_target = (pred != labels).long()

        labels_cpu = labels.detach().cpu().tolist()
        pred_cpu = pred.detach().cpu().tolist()
        domain_cpu = domains.detach().cpu().tolist()
        confidence_cpu = small_confidence.detach().cpu().tolist()
        entropy_cpu = entropy.detach().cpu().tolist()
        variance_cpu = variance.detach().cpu().tolist()
        proto_cpu = proto_dist.detach().cpu().tolist()
        shift_cpu = domain_shift.detach().cpu().tolist()
        risk_cpu = risk_prob.detach().cpu().tolist()
        nearest_cpu = nearest_class.detach().cpu().tolist()
        top3_values_cpu, top3_indices_cpu = torch.topk(
            mean_prob.detach().cpu(), k=min(3, mean_prob.size(1)), dim=1
        )
        weights_cpu = modality_weights.detach().cpu().tolist()
        for local_i, (true_label, small_label) in enumerate(zip(labels_cpu, pred_cpu)):
            conf = float(confidence_cpu[local_i])
            ent = float(entropy_cpu[local_i])
            var = float(variance_cpu[local_i])
            risk = float(risk_cpu[local_i])
            proto_value = float(proto_cpu[local_i])
            shift_value = float(shift_cpu[local_i])
            level = uncertainty_level(
                conf, ent, var, risk, float(model.config.router_threshold)
            )
            prediction_reports.append({
                "sample_index": int(base_index + local_i),
                "split": split_name,
                "true_label_id": int(true_label),
                "true_fault": FAULT_NAMES[int(true_label)],
                "small_label_id": int(small_label),
                "small_fault": FAULT_NAMES[int(small_label)],
                "final_label_id": int(small_label),
                "final_fault": FAULT_NAMES[int(small_label)],
                "final_source": "small_model",
                "small_accepted": True,
                "deferred_to_llm": False,
                "llm_used": False,
                "fallback_used": False,
                "final_confidence": conf,
                "small_confidence": conf,
                "llm_confidence": None,
                "risk_probability": risk,
                "uncertainty_entropy": ent,
                "mc_variance": var,
                "uncertainty_level": level,
                "uncertainty_summary": uncertainty_summary(
                    conf, ent, var, risk, proto_value, shift_value
                ),
                "prototype_distance": proto_value,
                "domain_shift": shift_value,
                "nearest_prototype_label_id": int(nearest_cpu[local_i]),
                "nearest_prototype_fault": FAULT_NAMES[int(nearest_cpu[local_i])],
                "domain": ID_TO_DOMAIN.get(int(domain_cpu[local_i]), str(int(domain_cpu[local_i]))),
                "top3": [
                    {
                        "label_id": int(lbl),
                        "fault": FAULT_NAMES[int(lbl)],
                        "prob": float(prob),
                    }
                    for prob, lbl in zip(top3_values_cpu[local_i].tolist(), top3_indices_cpu[local_i].tolist())
                ],
                "modality_weights": [float(v) for v in weights_cpu[local_i]],
                "maintenance_suggestion": "",
                "maintenance_generation_pending": True,
                "diagnostic_reasoning": "Small-model result accepted because the sample did not pass the risk threshold or confidence gate for delegated LLM diagnosis.",
                "gate_reason": "not_routed_or_not_eligible",
                "structured_json_valid": None,
                "exact_json": None,
            })

        y_true.extend(labels_cpu)
        y_small.extend(pred_cpu)
        y_final.extend(pred_cpu)
        small_probability_rows.append(mean_prob.detach().cpu().numpy())
        risk_targets_all.extend(risk_target.detach().cpu().tolist())
        risk_scores_all.extend(risk_prob.detach().cpu().tolist())
        unc_sum += float(entropy.sum().item())
        var_sum += float(variance.sum().item())
        proto_sum += float(proto_dist.sum().item())
        shift_sum += float(domain_shift.sum().item())
        conf_sum += float(mean_prob.max(dim=1).values.sum().item())

        interval = max(0, int(model.config.eval_log_interval))
        if interval and (batch_idx % interval == 0 or batch_idx == total_batches):
            print(
                f"[{split_name}] feature pass {batch_idx}/{total_batches} | "
                f"samples {len(y_true)} | threshold-candidates {threshold_candidate_count} | "
                f"eligible {eligible_candidate_count} | conf-blocked {confidence_blocked_count} | "
                f"ranked {len(candidate_pool)}"
            )

    sync_device(model.device)
    feature_seconds = time.perf_counter() - feature_start
    n = max(len(y_true), 1)
    candidate_pool.sort(key=lambda item: float(item["priority"]), reverse=True)
    if bool(model.config.router_budgeted_routing):
        # Optional capped mode: apply a hard budget after the absolute threshold.
        # This is useful for controlled efficiency studies, but it is not the default
        # because it can make the LLM call count look fixed.
        rate = min(max(float(model.config.llm_max_call_rate), 0.0), 1.0)
        rate_budget = int(math.ceil(n * rate))
        hard_cap = int(model.config.max_llm_calls_per_eval)
        if hard_cap < 0:
            hard_cap = len(candidate_pool)
        budget = min(len(candidate_pool), hard_cap, rate_budget)
    else:
        # Dynamic threshold-only delegation: every sample that passes the
        # calibrated risk threshold and confidence gate is delegated to the LLM.
        # Therefore calls = eligible hard samples, not a fixed 800/10% budget.
        budget = len(candidate_pool)
    selected = candidate_pool[:budget]
    selected_indices = {int(item["global_index"]) for item in selected}
    override_source = ["none"] * n

    triggers = len(selected_indices)
    false_triggers = sum(int(risk_targets_all[i] == 0) for i in selected_indices)
    routed_errors = sum(int(risk_targets_all[i] == 1) for i in selected_indices)
    total_errors = int(sum(risk_targets_all))
    miss_triggers = max(total_errors - routed_errors, 0)

    llm_calls = llm_valid_json = llm_exact_json = 0
    llm_requested_overrides = 0
    llm_accepted_overrides = llm_rejected_overrides = 0
    llm_raw_kept_small = 0
    llm_raw_corrected_errors = 0
    llm_raw_harmful_overrides = 0
    llm_raw_wrong_to_wrong = 0
    consensus_accepted_overrides = 0
    llm_load_seconds = 0.0
    llm_inference_seconds = 0.0
    maintenance_revision_calls = 0
    maintenance_revision_valid = 0
    maintenance_revision_requested = 0
    maintenance_revision_accepted = 0
    maintenance_revision_rejected = 0
    class_maintenance_calls = 0
    class_maintenance_valid = 0
    class_maintenance_retry_calls = 0
    class_maintenance_stability_calls = 0
    class_maintenance_stability_matches = 0
    class_maintenance_stability_comparisons = 0
    class_maintenance_generated_tokens = 0
    maintenance_inference_seconds = 0.0
    class_maintenance_inference_seconds = 0.0

    if selected:
        routing_name = (
            "threshold-gated budget-capped routing"
            if bool(model.config.router_budgeted_routing)
            else "dynamic threshold-only routing"
        )
        print(
            f"[{split_name}] {routing_name} | ranked candidates={len(candidate_pool)} | "
            f"selected={len(selected)} ({100.0 * len(selected) / n:.2f}%) | "
            f"threshold candidates={threshold_candidate_count} "
            f"({100.0 * threshold_candidate_count / n:.2f}%) | "
            f"eligible={eligible_candidate_count} "
            f"({100.0 * eligible_candidate_count / n:.2f}%) | "
            f"confidence-blocked={confidence_blocked_count} "
            f"({100.0 * confidence_blocked_count / n:.2f}%)"
        )

    if model.config.advisor_mode == "prototype":
        for item in selected:
            evidence = item["evidence"]
            top3 = item["top3"]
            fallback = model.reviewer.prototype_review(evidence, top3)
            global_index = int(item["global_index"])
            y_final[global_index] = int(fallback["final_label"])
            if int(fallback["final_label"]) != int(evidence["small_model_label_id"]):
                override_source[global_index] = "prototype"
            evidence["review_output"] = fallback
            evidence["revision_gate"] = {
                "accepted": int(fallback["final_label"]) != int(evidence["small_model_label_id"]),
                "source": "prototype",
                "reason": "prototype_fallback",
                "final_label_after_gate": int(fallback["final_label"]),
            }
            prediction_reports[global_index].update({
                "final_label_id": int(fallback["final_label"]),
                "final_fault": FAULT_NAMES[int(fallback["final_label"])],
                "final_source": "prototype_fallback",
                "small_accepted": False,
                "deferred_to_llm": True,
                "llm_used": False,
                "fallback_used": True,
                "final_confidence": float(fallback.get("confidence", prediction_reports[global_index]["small_confidence"])),
                "llm_confidence": None,
                "maintenance_suggestion": "",
                "maintenance_generation_pending": True,
                "diagnostic_reasoning": str(fallback.get("reasoning", "Prototype/evidence fallback classification was used.")),
                "gate_reason": "prototype_fallback",
                "structured_json_valid": bool(fallback.get("valid_json", False)),
                "exact_json": bool(fallback.get("exact_json", False)),
            })
            if len(evidence_samples) < 5:
                evidence_samples.append(evidence)

    wants_llm = (
        use_reviewer
        and allow_llm
        and model.config.advisor_mode in {"llm_delegate", "llm_revise", "llm_explain"}
        and len(selected) > 0
    )
    if wants_llm and not model.reviewer.loaded:
        sync_device(model.device)
        load_start = time.perf_counter()
        model.reviewer.load()
        sync_device(model.device)
        llm_load_seconds = time.perf_counter() - load_start
    wants_llm = wants_llm and model.reviewer.loaded

    if (
        selected
        and (not wants_llm)
        and use_reviewer
        and allow_llm
        and model.config.advisor_mode in {"llm_delegate", "llm_revise"}
    ):
        print(
            f"[{split_name}] delegated LLM unavailable; applying deterministic evidence fallback "
            f"to {len(selected)} routed samples."
        )
        for item in selected:
            evidence = item["evidence"]
            top3 = item["top3"]
            fallback = model.reviewer.prototype_review(evidence, top3)
            global_index = int(item["global_index"])
            final_label = int(fallback["final_label"])
            y_final[global_index] = final_label
            if final_label != int(evidence["small_model_label_id"]):
                override_source[global_index] = "consensus"
                consensus_accepted_overrides += 1
            evidence["review_output"] = fallback
            evidence["revision_gate"] = {
                "accepted": final_label != int(evidence["small_model_label_id"]),
                "source": "evidence_fallback",
                "reason": "llm_unavailable_deterministic_fallback",
                "final_label_after_gate": final_label,
            }
            prediction_reports[global_index].update({
                "final_label_id": final_label,
                "final_fault": FAULT_NAMES[final_label],
                "final_source": "evidence_fallback",
                "small_accepted": False,
                "deferred_to_llm": True,
                "llm_used": False,
                "fallback_used": True,
                "final_confidence": float(fallback.get("confidence", prediction_reports[global_index]["small_confidence"])),
                "llm_confidence": None,
                "maintenance_suggestion": "",
                "maintenance_generation_pending": True,
                "diagnostic_reasoning": str(fallback.get("reasoning", "LLM unavailable; deterministic evidence fallback was used.")),
                "gate_reason": "llm_unavailable_deterministic_fallback",
                "structured_json_valid": bool(fallback.get("valid_json", False)),
                "exact_json": bool(fallback.get("exact_json", False)),
            })
            if len(evidence_samples) < 5:
                evidence_samples.append(evidence)

    if wants_llm:
        print(
            f"[{split_name}] delegated LLM classifier | calls={len(selected)} "
            f"({100.0 * len(selected) / n:.2f}%) | "
            f"batch_size={model.config.llm_batch_size}"
        )
        sync_device(model.device)
        llm_start = time.perf_counter()
        reviews = model.reviewer.review_batch(
            [item["evidence"] for item in selected],
            [item["top3"] for item in selected],
            batch_size=model.config.llm_batch_size,
        )
        sync_device(model.device)
        llm_inference_seconds = time.perf_counter() - llm_start
        llm_calls = len(reviews)
        for item, review in zip(selected, reviews):
            evidence = item["evidence"]
            global_index = int(item["global_index"])
            small_label = int(evidence["small_model_label_id"])
            llm_valid_json += int(bool(review.get("valid_json", False)))
            llm_exact_json += int(bool(review.get("exact_json", False)))
            try:
                requested_label = int(review.get("final_label", small_label))
            except (TypeError, ValueError):
                requested_label = small_label
            requested = requested_label != small_label
            llm_requested_overrides += int(requested)
            true_label = int(y_true[global_index])
            if requested_label == small_label:
                llm_raw_kept_small += 1
            elif small_label != true_label and requested_label == true_label:
                llm_raw_corrected_errors += 1
            elif small_label == true_label and requested_label != true_label:
                llm_raw_harmful_overrides += 1
            elif small_label != true_label and requested_label != true_label:
                llm_raw_wrong_to_wrong += 1

            if model.config.advisor_mode in {"llm_delegate", "llm_revise"}:
                final_label, accepted, gate_reason = apply_llm_revision_gate(
                    evidence, review, model.config
                )
                if accepted:
                    # Delegated path: for routed hard samples, the LLM branch is
                    # the final classifier, not a post-hoc reviewer. It may output
                    # the same label as the small model, but the decision is still
                    # counted as an LLM-delegated classification.
                    gate_source = "llm_delegate"
                    y_final[global_index] = int(final_label)
                    override_source[global_index] = "llm"
                    llm_accepted_overrides += 1
                else:
                    if requested:
                        llm_rejected_overrides += 1
                    # V8.8 confident delegate: if the LLM output cannot be parsed or
                    # repaired into a legal candidate label, fall back only to the
                    # small-model label. Evidence consensus is disabled here so the
                    # hard-sample classifier is truly the LLM branch.
                    final_label = small_label
                    gate_source = "llm_parse_fallback"
                    gate_reason = f"{gate_reason}; direct_delegate_parse_fallback_to_small"
                    y_final[global_index] = int(final_label)
            else:
                final_label, accepted, gate_reason, gate_source = (
                    small_label,
                    False,
                    "explanation_only",
                    "none",
                )

            evidence["review_output"] = review
            evidence["revision_gate"] = {
                "accepted": bool(accepted),
                "source": gate_source,
                "reason": gate_reason,
                "final_label_after_gate": int(final_label),
            }
            llm_conf_value = None
            try:
                llm_conf_value = float(review.get("confidence", 0.0))
            except (TypeError, ValueError):
                llm_conf_value = None
            if accepted and gate_source == "llm_delegate":
                final_source = "llm_delegate"
                fallback_used = False
                report_conf = llm_conf_value if llm_conf_value is not None else prediction_reports[global_index]["small_confidence"]
                suggestion = ""
                reasoning = str(review.get("reasoning", "")).strip() or "Delegated LLM classifier produced the final diagnosis from structured evidence."
            elif gate_source == "evidence_fallback":
                final_source = "evidence_fallback"
                fallback_used = True
                report_conf = prediction_reports[global_index]["small_confidence"]
                suggestion = ""
                reasoning = f"LLM output was rejected or unavailable; evidence fallback was used. {gate_reason}"
            else:
                final_source = gate_source
                fallback_used = gate_source not in {"none", "llm_delegate"}
                report_conf = prediction_reports[global_index]["small_confidence"]
                suggestion = ""
                reasoning = str(review.get("reasoning", "")).strip() or gate_reason
            prediction_reports[global_index].update({
                "final_label_id": int(final_label),
                "final_fault": FAULT_NAMES[int(final_label)],
                "final_source": final_source,
                "small_accepted": False if gate_source != "none" else True,
                "deferred_to_llm": True,
                "llm_used": True,
                "fallback_used": bool(fallback_used),
                "final_confidence": float(report_conf),
                "llm_confidence": llm_conf_value,
                "maintenance_suggestion": suggestion,
                "diagnostic_reasoning": reasoning,
                "gate_reason": gate_reason,
                "structured_json_valid": bool(review.get("valid_json", False)),
                "exact_json": bool(review.get("exact_json", False)),
                "raw_llm_text": str(review.get("raw_text", ""))[:500],
            })
            if len(evidence_samples) < 5:
                evidence_samples.append(evidence)

    # V10.12 controlled second pass. The maintenance branch may propose a
    # re-diagnosis only for the already Router-selected hard samples. It cannot
    # read target truth/correctness and cannot write metric fields. Every proposed
    # change passes through apply_maintenance_revision_gate(), which reuses the
    # unchanged V10.7 evidence-consensus conditions.
    maintenance_enabled = bool(
        getattr(model.config, "llm_generate_maintenance", True)
        and use_reviewer
        and allow_llm
        and model.config.llm_mode == "local_hf"
    )
    if maintenance_enabled and not model.reviewer.loaded:
        sync_device(model.device)
        maintenance_load_start = time.perf_counter()
        model.reviewer.load()
        sync_device(model.device)
        llm_load_seconds += time.perf_counter() - maintenance_load_start

    # V10.13 report-only contract: the maintenance stage cannot re-diagnose.
    maintenance_revision_enabled = False

    if (
        maintenance_enabled
        and model.reviewer.loaded
        and selected
        and maintenance_revision_enabled
    ):
        maintenance_contexts = [
            build_sample_maintenance_revision_context(
                item["evidence"], prediction_reports[int(item["global_index"])]
            )
            for item in selected
        ]
        sync_device(model.device)
        maintenance_start = time.perf_counter()
        maintenance_reviews = model.reviewer.maintenance_review_batch(
            maintenance_contexts,
            batch_size=int(getattr(model.config, "maintenance_batch_size", 4)),
            max_new_tokens=int(getattr(model.config, "maintenance_max_new_tokens", 512)),
            seed=int(model.config.seed) + 7000,
        )
        sync_device(model.device)
        maintenance_inference_seconds += time.perf_counter() - maintenance_start
        maintenance_revision_calls = len(maintenance_reviews)

        for item, review in zip(selected, maintenance_reviews):
            global_index = int(item["global_index"])
            evidence = item["evidence"]
            current_label = int(y_final[global_index])
            current_confidence = _report_float(
                prediction_reports[global_index].get("final_confidence", 0.0)
            )
            proposed_label = int(review.get("proposed_label", current_label))
            valid_review = bool(review.get("maintenance_generation_valid", False))
            requested = bool(valid_review and proposed_label != current_label)
            maintenance_revision_valid += int(valid_review)
            maintenance_revision_requested += int(requested)

            revised_label, revision_accepted, revision_reason = apply_maintenance_revision_gate(
                evidence,
                review,
                current_label,
                current_confidence,
                model.config,
            )
            maintenance_revision_accepted += int(revision_accepted)
            maintenance_revision_rejected += int(requested and not revision_accepted)
            row = prediction_reports[global_index]
            row.update({
                "maintenance_revision_generation_valid": valid_review,
                "maintenance_revision_requested": requested,
                "maintenance_revision_proposed_label_id": proposed_label,
                "maintenance_revision_proposed_fault": FAULT_NAMES.get(proposed_label, str(proposed_label)),
                "maintenance_revision_accepted": bool(revision_accepted),
                "maintenance_revision_gate_reason": revision_reason,
                "maintenance_revision_candidate_scores": review.get("candidate_scores", []),
                "maintenance_revision_raw_text": str(review.get("raw_text", ""))[:1200],
            })
            if revision_accepted:
                y_final[global_index] = int(revised_label)
                override_source[global_index] = "maintenance_llm"
                row.update({
                    "final_label_id": int(revised_label),
                    "final_fault": FAULT_NAMES[int(revised_label)],
                    "final_source": "maintenance_consensus_revision",
                    "final_confidence": float(review.get("confidence", current_confidence)),
                    "fallback_used": False,
                    "gate_reason": revision_reason,
                    "diagnostic_reasoning": (
                        str(row.get("diagnostic_reasoning", ""))
                        + " Maintenance-stage re-diagnosis passed the existing evidence-consensus gate."
                    ).strip(),
                })

    elif maintenance_enabled and model.reviewer.loaded and selected and not maintenance_revision_enabled:
        # FAST mode: no second LLM pass over routed samples. Diagnosis is already
        # fixed by the normal Router -> diagnosis-LoRA -> consensus gate.
        maintenance_revision_calls = 0

    # Once diagnostic consensus decisions have been applied, labels are frozen.
    # Generate one preliminary class-level maintenance draft per predicted
    # class and attach it to every sample of that final class. This proof-of-concept
    # reporting call cannot change labels and is not expert or field validation.
    if maintenance_enabled and model.reviewer.loaded:
        present_labels = sorted({int(value) for value in y_final})
        class_contexts: List[Dict[str, object]] = []
        class_rows_list: List[List[Dict[str, object]]] = []
        for label in present_labels:
            class_rows = [
                row for row in prediction_reports
                if int(row.get("final_label_id", -1)) == int(label)
            ]
            class_rows_list.append(class_rows)
            class_contexts.append(
                build_class_maintenance_context(
                    label, class_rows, model.modalities, model.config
                )
            )
        sync_device(model.device)
        class_maintenance_start = time.perf_counter()
        class_maintenance_reviews = model.reviewer.maintenance_review_batch(
            class_contexts,
            batch_size=int(getattr(model.config, "maintenance_batch_size", 4)),
            max_new_tokens=int(getattr(model.config, "maintenance_max_new_tokens", 512)),
            seed=int(model.config.seed) + 8000,
        )
        sync_device(model.device)
        primary_class_seconds = time.perf_counter() - class_maintenance_start
        maintenance_inference_seconds += primary_class_seconds
        class_maintenance_inference_seconds += primary_class_seconds
        class_maintenance_calls = len(class_maintenance_reviews)
        class_maintenance_retry_calls += sum(
            bool(review.get("maintenance_retry_used", False))
            for review in class_maintenance_reviews
        )
        class_maintenance_generated_tokens = sum(
            int(review.get("maintenance_generated_token_count", 0))
            for review in class_maintenance_reviews
        )

        # Optional repeat generation for deterministic priority-stability reporting.
        # The primary reports remain authoritative; repeats are metric-only and are
        # never attached to samples or allowed to modify diagnosis.
        stability_repeats = max(1, int(getattr(model.config, "maintenance_stability_repeats", 1)))
        for repeat_idx in range(1, stability_repeats):
            sync_device(model.device)
            repeat_start = time.perf_counter()
            repeated_reviews = model.reviewer.maintenance_review_batch(
                class_contexts,
                batch_size=int(getattr(model.config, "maintenance_batch_size", 4)),
                max_new_tokens=int(getattr(model.config, "maintenance_max_new_tokens", 512)),
                seed=int(model.config.seed) + 8000 + repeat_idx,
            )
            sync_device(model.device)
            repeat_seconds = time.perf_counter() - repeat_start
            maintenance_inference_seconds += repeat_seconds
            class_maintenance_inference_seconds += repeat_seconds
            class_maintenance_stability_calls += len(repeated_reviews)
            class_maintenance_retry_calls += sum(
                bool(review.get("maintenance_retry_used", False))
                for review in repeated_reviews
            )
            for primary_review, repeated_review in zip(class_maintenance_reviews, repeated_reviews):
                primary_priority = parse_maintenance_priority(
                    primary_review.get("maintenance_priority", "")
                )
                repeated_priority = parse_maintenance_priority(
                    repeated_review.get("maintenance_priority", "")
                )
                priorities_evaluable = bool(
                    primary_review.get("maintenance_generation_valid", False)
                    and repeated_review.get("maintenance_generation_valid", False)
                    and primary_priority is not None
                    and repeated_priority is not None
                )
                if priorities_evaluable:
                    class_maintenance_stability_comparisons += 1
                    class_maintenance_stability_matches += int(
                        primary_priority == repeated_priority
                    )

        for label, class_rows, review in zip(present_labels, class_rows_list, class_maintenance_reviews):
            valid_report = bool(review.get("maintenance_generation_valid", False))
            class_maintenance_valid += int(valid_report)
            for row in class_rows:
                row.update({
                    "maintenance_suggestion": str(review.get("recommended_action_zh", "")) if valid_report else "",
                    "maintenance_suggestion_zh": str(review.get("recommended_action_zh", "")) if valid_report else "",
                    "maintenance_priority": str(review.get("maintenance_priority", "")) if valid_report else "",
                    "action_timeline": str(review.get("action_timeline_zh", "")) if valid_report else "",
                    "suspected_cause_zh": str(review.get("suspected_cause_zh", "")) if valid_report else "",
                    "inspection_items_zh": str(review.get("inspection_items_zh", "")) if valid_report else "",
                    "recommended_action_zh": str(review.get("recommended_action_zh", "")) if valid_report else "",
                    "verification_plan_zh": str(review.get("verification_plan_zh", "")) if valid_report else "",
                    "human_review_required": bool(review.get("human_review_required", True)) if valid_report else True,
                    "human_review_reason_zh": str(review.get("human_review_reason_zh", "")) if valid_report else "维护报告生成无效，需人工复核。",
                    "maintenance_report_summary_zh": str(review.get("report_summary_zh", "")) if valid_report else "",
                    "maintenance_diagnosis_consistent": bool(review.get("diagnosis_consistent", False)) if valid_report else False,
                    "maintenance_generated_token_count": int(review.get("maintenance_generated_token_count", 0)),
                    "maintenance_safety_note_zh": "本报告是诊断后的结构化维护草案和可行性展示，不能替代现场检查、人工复核、维修手册或现场安全规程。",
                    "maintenance_generation_status": str(review.get("maintenance_generation_status", "unknown")),
                    "maintenance_generation_valid": valid_report,
                    "maintenance_content_valid": bool(review.get("maintenance_content_valid", False)),
                    "maintenance_content_validation_issues": review.get(
                        "maintenance_content_validation_issues", []
                    ),
                    "maintenance_generation_method": "policy_constrained_llm_maintenance_generation_after_frozen_diagnosis",
                    "maintenance_model_using_lora": bool(review.get("maintenance_model_using_lora", False)),
                    "maintenance_diagnosis_lora_loaded": bool(review.get("maintenance_diagnosis_lora_loaded", False)),
                    "maintenance_adapter_disabled_for_generation": bool(review.get("maintenance_adapter_disabled_for_generation", False)),
                    "maintenance_validation_issues": review.get("maintenance_validation_issues", []),
                    # Preserve generated fields for auditing and FCR calculation
                    # even when strict validation prevents them from becoming a
                    # usable maintenance suggestion in the sample-level export.
                    "maintenance_generated_raw_priority": str(review.get("maintenance_priority", "")),
                    "maintenance_generated_raw_action_timeline_zh": str(review.get("action_timeline_zh", "")),
                    "maintenance_generated_raw_suspected_cause_zh": str(review.get("suspected_cause_zh", "")),
                    "maintenance_generated_raw_inspection_items_zh": str(review.get("inspection_items_zh", "")),
                    "maintenance_generated_raw_recommended_action_zh": str(review.get("recommended_action_zh", "")),
                    "maintenance_generated_raw_verification_plan_zh": str(review.get("verification_plan_zh", "")),
                    "maintenance_generated_raw_human_review_reason_zh": str(review.get("human_review_reason_zh", "")),
                    "maintenance_generated_raw_report_summary_zh": str(review.get("report_summary_zh", "")),
                    "maintenance_retry_used": bool(review.get("maintenance_retry_used", False)),
                    "maintenance_retry_selected": bool(review.get("maintenance_retry_selected", False)),
                    "maintenance_retry_validation_issues": review.get(
                        "maintenance_retry_validation_issues", []
                    ),
                    "maintenance_retry_raw_text": str(
                        review.get("maintenance_retry_raw_text", "")
                    )[:4000],
                    "maintenance_primary_status": str(review.get("maintenance_primary_status", "")),
                    "maintenance_primary_validation_issues": review.get(
                        "maintenance_primary_validation_issues", []
                    ),
                    "maintenance_primary_raw_text": str(
                        review.get("maintenance_primary_raw_text", "")
                    )[:4000],
                    "maintenance_recommendation_granularity": "class_level",
                    "maintenance_shared_by_final_class": True,
                    "class_maintenance_raw_text": str(review.get("raw_text", ""))[:4000],
                })
    else:
        for row in prediction_reports:
            row.update({
                "maintenance_suggestion": "",
                "maintenance_suggestion_zh": "",
                "maintenance_generation_status": "disabled_or_llm_unavailable",
                "maintenance_generation_valid": False,
                "maintenance_generation_method": "not_generated_no_template_fallback",
                "human_review_required": True,
                "human_review_reason_zh": "维护大模型未启用或不可用，未使用固定模板替代。",
                "maintenance_diagnosis_consistent": False,
                "maintenance_generated_token_count": 0,
                "maintenance_safety_note_zh": "未生成维护草案；必须由人员按维修手册和现场安全规程处置。",
            })

    maintenance_rule_metrics, _maintenance_rule_details = evaluate_rule_based_maintenance_reports(
        prediction_reports
    )

    # Only now are final labels considered authoritative. All metrics below are
    # recomputed from y_final; no maintenance output can directly edit a metric.
    for i, row in enumerate(prediction_reports):
        row["small_correct"] = bool(int(y_small[i]) == int(y_true[i]))
        row["final_correct"] = bool(int(y_final[i]) == int(y_true[i]))
        row["final_changed_from_small"] = bool(int(y_final[i]) != int(y_small[i]))

    risk_auc = 0.0
    try:
        if len(set(risk_targets_all)) > 1:
            risk_auc = float(roc_auc_score(risk_targets_all, risk_scores_all))
    except Exception:
        risk_auc = 0.0

    non_trigger_idx = [i for i in range(n) if i not in selected_indices]
    trigger_idx = sorted(selected_indices)
    selective_acc = (
        accuracy_score([y_true[i] for i in non_trigger_idx], [y_small[i] for i in non_trigger_idx]) * 100
        if non_trigger_idx else 0.0
    )
    triggered_small_acc = (
        accuracy_score([y_true[i] for i in trigger_idx], [y_small[i] for i in trigger_idx]) * 100
        if trigger_idx else 0.0
    )
    triggered_final_acc = (
        accuracy_score([y_true[i] for i in trigger_idx], [y_final[i] for i in trigger_idx]) * 100
        if trigger_idx else 0.0
    )

    system_corrected_errors = sum(
        int(y_small[i] != y_true[i] and y_final[i] == y_true[i]) for i in range(n)
    )
    system_harmful_overrides = sum(
        int(y_small[i] == y_true[i] and y_final[i] != y_true[i]) for i in range(n)
    )
    system_wrong_to_wrong = sum(
        int(y_small[i] != y_true[i] and y_final[i] != y_true[i] and y_final[i] != y_small[i])
        for i in range(n)
    )

    def source_outcomes(source: str) -> Tuple[int, int, int]:
        idxs = [i for i, value in enumerate(override_source) if value == source]
        corrected = sum(int(y_small[i] != y_true[i] and y_final[i] == y_true[i]) for i in idxs)
        harmful = sum(int(y_small[i] == y_true[i] and y_final[i] != y_true[i]) for i in idxs)
        return corrected, harmful, corrected - harmful

    llm_corrected, llm_harmful, llm_net = source_outcomes("llm")
    consensus_corrected, consensus_harmful, consensus_net = source_outcomes("consensus")
    maintenance_corrected, maintenance_harmful, maintenance_net = source_outcomes("maintenance_llm")

    labels_order = sorted(FAULT_NAMES)
    small_metrics = label_metrics(y_true, y_small, labels_order)
    final_metrics = label_metrics(y_true, y_final, labels_order)
    small_probs = (
        np.concatenate(small_probability_rows, axis=0)
        if small_probability_rows
        else np.empty((0, len(labels_order)), dtype=np.float64)
    )
    small_top3_acc = (
        100.0 * top_k_accuracy_score(y_true, small_probs, k=min(3, len(labels_order)), labels=labels_order)
        if len(y_true) else 0.0
    )
    small_nll = float(log_loss(y_true, small_probs, labels=labels_order)) if len(y_true) else 0.0
    small_ece = expected_calibration_error(small_probs, y_true)

    sync_device(model.device)
    evaluation_seconds = time.perf_counter() - evaluation_start
    throughput = n / max(evaluation_seconds, 1e-9)
    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(model.device)) / (1024.0 ** 2)
        if model.device.type == "cuda"
        else 0.0
    )

    return {
        "small_acc": float(small_metrics["acc"]),
        "small_balanced_acc": float(small_metrics["balanced_acc"]),
        "small_macro_precision": float(small_metrics["macro_precision"]),
        "small_macro_recall": float(small_metrics["macro_recall"]),
        "small_macro_f1": float(small_metrics["macro_f1"]),
        "small_weighted_f1": float(small_metrics["weighted_f1"]),
        "small_mcc": float(small_metrics["mcc"]),
        "small_kappa": float(small_metrics["kappa"]),
        "small_top3_acc": small_top3_acc,
        "small_nll": small_nll,
        "small_ece": small_ece,
        "final_acc": float(final_metrics["acc"]),
        "final_balanced_acc": float(final_metrics["balanced_acc"]),
        "final_macro_precision": float(final_metrics["macro_precision"]),
        "final_macro_recall": float(final_metrics["macro_recall"]),
        "final_macro_f1": float(final_metrics["macro_f1"]),
        "final_weighted_f1": float(final_metrics["weighted_f1"]),
        "final_mcc": float(final_metrics["mcc"]),
        "final_kappa": float(final_metrics["kappa"]),
        "small_classification_report": small_metrics["classification_report"],
        "final_classification_report": final_metrics["classification_report"],
        "trigger_rate": 100.0 * triggers / n,
        "coverage": 100.0 * (n - triggers) / n,
        "threshold_candidate_rate": 100.0 * threshold_candidate_count / n,
        "eligible_candidate_rate": 100.0 * eligible_candidate_count / n,
        "confidence_blocked_rate": 100.0 * confidence_blocked_count / n,
        "eligible_candidate_count": int(eligible_candidate_count),
        "confidence_blocked_count": int(confidence_blocked_count),
        "false_trigger_rate": 100.0 * false_triggers / n,
        "miss_trigger_rate": 100.0 * miss_triggers / n,
        "router_error_recall": 100.0 * routed_errors / max(total_errors, 1),
        "router_precision": 100.0 * routed_errors / max(triggers, 1),
        "selective_acc": selective_acc,
        "triggered_small_acc": triggered_small_acc,
        "triggered_final_acc": triggered_final_acc,
        "risk_auc": risk_auc,
        "llm_rate": 100.0 * llm_calls / n,
        "llm_calls": llm_calls,
        "total_llm_calls_including_maintenance": int(
            llm_calls + maintenance_revision_calls + class_maintenance_calls
            + class_maintenance_stability_calls + class_maintenance_retry_calls
        ),
        "llm_valid_json_rate": 100.0 * llm_valid_json / max(llm_calls, 1),
        "llm_exact_json_rate": 100.0 * llm_exact_json / max(llm_calls, 1),
        "llm_override_rate": 100.0 * llm_accepted_overrides / max(llm_calls, 1),
        "llm_requested_overrides": llm_requested_overrides,
        "llm_accepted_overrides": llm_accepted_overrides,
        "llm_rejected_overrides": llm_rejected_overrides,
        "llm_raw_kept_small": llm_raw_kept_small,
        "llm_raw_corrected_errors": llm_raw_corrected_errors,
        "llm_raw_harmful_overrides": llm_raw_harmful_overrides,
        "llm_raw_wrong_to_wrong": llm_raw_wrong_to_wrong,
        "llm_raw_net_corrections": llm_raw_corrected_errors - llm_raw_harmful_overrides,
        "consensus_accepted_overrides": consensus_accepted_overrides,
        "llm_corrected_errors": llm_corrected,
        "llm_harmful_overrides": llm_harmful,
        "llm_net_corrections": llm_net,
        "consensus_corrected_errors": consensus_corrected,
        "consensus_harmful_overrides": consensus_harmful,
        "consensus_net_corrections": consensus_net,
        "maintenance_revision_calls": int(maintenance_revision_calls),
        "maintenance_revision_valid_count": int(maintenance_revision_valid),
        "maintenance_revision_requested": int(maintenance_revision_requested),
        "maintenance_revision_accepted": int(maintenance_revision_accepted),
        "maintenance_revision_rejected": int(maintenance_revision_rejected),
        "maintenance_corrected_errors": int(maintenance_corrected),
        "maintenance_harmful_overrides": int(maintenance_harmful),
        "maintenance_net_corrections": int(maintenance_net),
        "class_maintenance_calls": int(class_maintenance_calls),
        "class_maintenance_valid_count": int(class_maintenance_valid),
        "class_maintenance_retry_calls": int(class_maintenance_retry_calls),
        "class_maintenance_stability_calls": int(class_maintenance_stability_calls),
        "class_maintenance_priority_stability_rate": (
            100.0 * class_maintenance_stability_matches / class_maintenance_stability_comparisons
            if class_maintenance_stability_comparisons > 0 else None
        ),
        "class_maintenance_average_generated_tokens": (
            float(class_maintenance_generated_tokens) / max(class_maintenance_calls, 1)
        ),
        **maintenance_rule_metrics,
        "system_corrected_errors": system_corrected_errors,
        "system_harmful_overrides": system_harmful_overrides,
        "system_wrong_to_wrong": system_wrong_to_wrong,
        "system_net_corrections": system_corrected_errors - system_harmful_overrides,
        "avg_uncertainty": unc_sum / n,
        "avg_variance": var_sum / n,
        "avg_proto_dist": proto_sum / n,
        "avg_domain_shift": shift_sum / n,
        "avg_confidence": conf_sum / n,
        "feature_inference_seconds": feature_seconds,
        "llm_load_seconds": llm_load_seconds,
        "llm_inference_seconds": llm_inference_seconds,
        "maintenance_inference_seconds": maintenance_inference_seconds,
        "class_maintenance_inference_seconds": class_maintenance_inference_seconds,
        "average_ms_per_class_maintenance_call": 1000.0 * class_maintenance_inference_seconds / max(
            class_maintenance_calls + class_maintenance_stability_calls
            + class_maintenance_retry_calls, 1
        ),
        "total_llm_inference_seconds": llm_inference_seconds + maintenance_inference_seconds,
        "evaluation_seconds": evaluation_seconds,
        "throughput_samples_per_second": throughput,
        "average_ms_per_sample": 1000.0 * evaluation_seconds / n,
        "average_ms_per_llm_call": 1000.0 * llm_inference_seconds / max(llm_calls, 1),
        "average_ms_per_total_llm_call": 1000.0 * (
            llm_inference_seconds + maintenance_inference_seconds
        ) / max(
            llm_calls + maintenance_revision_calls + class_maintenance_calls
            + class_maintenance_stability_calls + class_maintenance_retry_calls,
            1,
        ),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "confusion_matrix": confusion_matrix(y_true, y_final, labels=labels_order),
        "evidence_samples": evidence_samples,
        "prediction_reports": prediction_reports,
    }


def save_maintenance_rule_evaluation(
    reports: Sequence[Dict[str, object]],
    csv_path_text: str,
    json_path_text: str,
) -> Dict[str, object]:
    """Save automatic format/rule indicators without labeling them expert scores."""
    metrics, details = evaluate_rule_based_maintenance_reports(reports)
    if csv_path_text:
        csv_path = Path(csv_path_text)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_rows: List[Dict[str, object]] = []
        for detail in details:
            csv_rows.append({
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict)) else value
                )
                for key, value in detail.items()
            })
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved rule-based maintenance evaluation CSV: {csv_path} | reports={len(details)}")
    if json_path_text:
        json_path = Path(json_path_text)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                json_ready({"metrics": metrics, "reports": details}),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved rule-based maintenance evaluation JSON: {json_path}")
    return metrics


def _metric_percent_text(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric:.2f}%" if math.isfinite(numeric) else "N/A"


def print_final_complete_model_results(metrics: Dict[str, object]) -> None:
    """Print only the complete PSCG-LLM target-domain diagnosis metrics."""
    print("\n================ Final complete-model diagnosis ================")
    print(f"ACC       : {float(metrics['final_acc']):.2f}%")
    print(f"Precision : {float(metrics['final_macro_precision']):.2f}%")
    print(f"Recall    : {float(metrics['final_macro_recall']):.2f}%")
    print(f"Macro-F1  : {float(metrics['final_macro_f1']):.2f}%")
    print(f"MCC       : {float(metrics['final_mcc']):.3f}")
    print(f"Kappa     : {float(metrics['final_kappa']):.3f}")


def print_final_maintenance_results(metrics: Dict[str, object]) -> None:
    """Print only the final policy-constrained maintenance evaluation indicators."""
    kappa = metrics.get("maintenance_rule_priority_weighted_kappa")
    kappa_text = f"{float(kappa):.3f}" if kappa is not None else "N/A"
    print("\n================ Final maintenance evaluation ==================")
    print(f"Content validity     : {_metric_percent_text(metrics.get('maintenance_content_valid_rate'))}")
    print(f"RVR                  : {_metric_percent_text(metrics.get('maintenance_report_valid_rate'))}")
    print(f"FCR                  : {_metric_percent_text(metrics.get('maintenance_field_completeness_rate'))}")
    print(f"DRC                  : {_metric_percent_text(metrics.get('maintenance_rule_consistency_rate'))}")
    print(f"UCR                  : {_metric_percent_text(metrics.get('maintenance_unsupported_claim_rate'))}")
    print(f"Safety violation     : {_metric_percent_text(metrics.get('maintenance_safety_rule_violation_rate'))}")
    print(f"Priority agreement   : {_metric_percent_text(metrics.get('maintenance_rule_priority_agreement_rate'))}")
    print(f"Priority kappa       : {kappa_text}")


def save_final_maintenance_outputs(
    reports: Sequence[Dict[str, object]],
    modalities: Sequence[Tuple[str, Sequence[int]]],
    router_threshold: float,
    config: TrainConfig,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Save only class-level maintenance suggestions and their automatic evaluation."""
    maintenance_metrics = save_maintenance_rule_evaluation(
        reports,
        csv_path_text=config.maintenance_rule_evaluation_csv_path,
        json_path_text=config.maintenance_rule_evaluation_json_path,
    )
    summaries = build_class_maintenance_summaries(
        reports, modalities, float(router_threshold)
    )

    # Keep only final maintenance content/validation fields; omit small-model and
    # routed-sample diagnostic audit fields from the publication output.
    keep = (
        "fault_label_id", "fault_name", "maintenance_priority", "action_timeline",
        "suspected_cause_zh", "inspection_items_zh", "recommended_action_zh",
        "verification_plan_zh", "human_review_required", "human_review_reason_zh",
        "maintenance_generation_valid", "maintenance_content_valid",
        "maintenance_field_completeness_rate", "maintenance_rule_consistent",
        "maintenance_unsupported_claim_rate", "maintenance_safety_rule_violation",
        "maintenance_rule_priority_agreement", "maintenance_rule_priority_reference",
    )
    concise = [{key: row.get(key) for key in keep} for row in summaries]

    if config.class_maintenance_csv_path and concise:
        path = Path(config.class_maintenance_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(concise).to_csv(path, index=False, encoding="utf-8-sig")

    if config.class_maintenance_text_path and concise:
        path = Path(config.class_maintenance_text_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("PSCG-LLM 类别级维护建议（最终完整模型）\n")
            f.write("=" * 72 + "\n")
            for row in concise:
                f.write(f"故障类别：{row.get('fault_name')}\n")
                f.write(f"维护优先级：{row.get('maintenance_priority')}\n")
                f.write(f"处置时限：{row.get('action_timeline')}\n")
                f.write(f"可能原因：{row.get('suspected_cause_zh')}\n")
                f.write(f"检查项目：{row.get('inspection_items_zh')}\n")
                f.write(f"维护措施：{row.get('recommended_action_zh')}\n")
                f.write(f"复核方案：{row.get('verification_plan_zh')}\n")
                f.write("-" * 72 + "\n")
    return maintenance_metrics, concise


def final_results_payload(
    config: TrainConfig,
    diagnosis_metrics: Dict[str, object],
    maintenance_metrics: Dict[str, object],
) -> Dict[str, object]:
    """Minimal paper-result JSON: full-model diagnosis + maintenance metrics only."""
    return {
        "seed": int(config.seed),
        "split_seed": int(config.split_seed),
        "source_domains": list(config.source_domains),
        "target_domains": list(config.target_domains),
        "full_model_diagnosis": {
            "acc_percent": float(diagnosis_metrics["final_acc"]),
            "macro_precision_percent": float(diagnosis_metrics["final_macro_precision"]),
            "macro_recall_percent": float(diagnosis_metrics["final_macro_recall"]),
            "macro_f1_percent": float(diagnosis_metrics["final_macro_f1"]),
            "mcc": float(diagnosis_metrics["final_mcc"]),
            "kappa": float(diagnosis_metrics["final_kappa"]),
        },
        "maintenance_recommendation": {
            "content_valid_rate_percent": maintenance_metrics.get("maintenance_content_valid_rate"),
            "rvr_percent": maintenance_metrics.get("maintenance_report_valid_rate"),
            "fcr_percent": maintenance_metrics.get("maintenance_field_completeness_rate"),
            "drc_percent": maintenance_metrics.get("maintenance_rule_consistency_rate"),
            "ucr_percent": maintenance_metrics.get("maintenance_unsupported_claim_rate"),
            "safety_violation_rate_percent": maintenance_metrics.get("maintenance_safety_rule_violation_rate"),
            "priority_agreement_rate_percent": maintenance_metrics.get("maintenance_rule_priority_agreement_rate"),
            "priority_weighted_kappa": maintenance_metrics.get("maintenance_rule_priority_weighted_kappa"),
        },
    }

def create_datasets(config: TrainConfig):
    print(f">>> CURRENT SPLIT: source_domains={config.source_domains}, target_domains={config.target_domains}")
    print(">>> PU RAW CHANNEL MAP: raw channel 0=phase_current_1; raw channel 1=phase_current_2; raw channel 2=vibration_1")
    selected_raw = config.channel_indices if config.channel_indices is not None else list(range(config.input_channels))
    selected_names = {0: "phase_current_1", 1: "phase_current_2", 2: "vibration_1"}
    print(
        ">>> ACTIVE INPUT CHANNELS: "
        + "; ".join(
            f"model channel {model_idx}={selected_names.get(raw_idx, f'raw_{raw_idx}')}"
            for model_idx, raw_idx in enumerate(selected_raw)
        )
    )
    print(
        f">>> ACTIVE MODALITIES: {parse_modality_spec(config.modality_spec, config.input_channels)}; "
        "mechanical parameters and temperature are excluded"
    )
    for domain_name in tuple(config.source_domains) + tuple(config.target_domains):
        meta = DOMAIN_METADATA.get(domain_name, {})
        if meta:
            role = "source" if domain_name in config.source_domains else "target"
            print(
                f">>> PU CONDITION: {domain_name}={meta['setting_name']} ({role}) | "
                f"speed={meta['speed_rpm']} rpm; torque={meta['load_torque_nm']} Nm; "
                f"radial_force={meta['radial_force_n']} N"
            )
    print(
        f">>> SEEDS: training_seed={config.seed}; split_seed={config.split_seed}; "
        f"target_selection_seed={config.split_seed + 1000}"
    )
    print(
        f">>> SOURCE PROTOCOL: {config.source_samples_per_class_domain} windows per class-domain, "
        f"ratios={config.source_train_ratio:.2f}/{config.val_part1_ratio:.2f}/{config.val_part2_ratio:.2f}"
    )
    print(
        f">>> V10.12 DG CONTROLS: fusion=original_learned_gated_fusion; "
        f"domain_align_max={config.domain_align_weight:.3f}; "
        f"warmup/ramp={config.domain_align_warmup_epochs}/{config.domain_align_ramp_epochs}; "
        f"LLM consensus supporters>={config.llm_delegate_min_override_support}; "
        f"LLM score/margin>={config.llm_delegate_min_confidence:.2f}/{config.llm_delegate_min_score_margin:.2f}"
    )
    print(
        f">>> WINDOW PROTOCOL: pool all MAT files within each class-domain; "
        f"fixed_point_window={config.window_size}; stride={config.stride}; "
        f"rotation_normalize=False; fixed_revolutions=disabled; resample=False; "
        f"sampling_rate={config.sampling_rate_hz} Hz; "
        f"max_windows_per_trial={config.max_windows_per_trial}; split_unit=window"
    )

    source_records = discover_trials(config.data_path, config.source_domains)
    target_records = discover_trials(config.data_path, config.target_domains)
    if not source_records or not target_records:
        raise ValueError("Empty source or target records. Check data path and domain names.")

    # Load all candidate windows once without normalization. Exact balanced
    # source/target subsets are then selected by class and domain.
    raw_common = dict(
        window_size=config.window_size,
        stride=config.stride,
        input_channels=config.input_channels,
        scaler=StandardScaler(),
        fit_scaler=False,
        window_norm="none",
        max_windows_per_trial=config.max_windows_per_trial,
        samples_per_class=None,
        drop_first_column=config.drop_first_column,
        channel_indices=config.channel_indices,
        sampling_rate_hz=config.sampling_rate_hz,
        window_revolutions=config.window_revolutions,
        rotation_normalize=config.rotation_normalize,
    )
    source_pool = PUTrialWindowDataset(source_records, seed=config.split_seed, **raw_common)
    target_pool = PUTrialWindowDataset(target_records, seed=config.split_seed + 100, **raw_common)

    split_indices = balanced_source_sample_split(
        labels=source_pool.labels,
        domains=source_pool.domains,
        source_domains=config.source_domains,
        samples_per_class_domain=config.source_samples_per_class_domain,
        train_ratio=config.source_train_ratio,
        val_part1_ratio=config.val_part1_ratio,
        val_part2_ratio=config.val_part2_ratio,
        seed=config.split_seed,
    )
    target_indices = balanced_target_sample_selection(
        labels=target_pool.labels,
        domains=target_pool.domains,
        target_domains=config.target_domains,
        samples_per_class_domain=config.target_samples_per_class_domain,
        seed=config.split_seed + 1000,
    )

    split_fingerprints = {
        "source_train": index_fingerprint(split_indices["source_train"]),
        "val_part1": index_fingerprint(split_indices["val_part1"]),
        "val_part2": index_fingerprint(split_indices["val_part2"]),
        "target_test": index_fingerprint(target_indices),
    }
    print(
        ">>> SPLIT FINGERPRINTS: "
        + ", ".join(f"{name}={value}" for name, value in split_fingerprints.items())
    )

    # Defensive sample-index overlap check inside the source pool.
    source_sets = {name: set(values.tolist()) for name, values in split_indices.items()}
    source_names = list(source_sets)
    for i, left in enumerate(source_names):
        for right in source_names[i + 1:]:
            overlap = source_sets[left] & source_sets[right]
            if overlap:
                raise RuntimeError(f"Source sample leakage between {left} and {right}.")

    scaler = StandardScaler()
    source_train_ds = subset_array_dataset(
        source_pool, split_indices["source_train"], config.window_size, config.input_channels,
        scaler, True, config.window_norm,
    )
    val_part1_ds = subset_array_dataset(
        source_pool, split_indices["val_part1"], config.window_size, config.input_channels,
        scaler, False, config.window_norm,
    )
    val_part2_ds = subset_array_dataset(
        source_pool, split_indices["val_part2"], config.window_size, config.input_channels,
        scaler, False, config.window_norm,
    )
    target_test_ds = subset_array_dataset(
        target_pool, target_indices, config.window_size, config.input_channels,
        scaler, False, config.window_norm,
    )

    expected_source = (
        len(config.source_domains) * len(FAULT_NAMES) * config.source_samples_per_class_domain
    )
    expected_target = (
        len(config.target_domains) * len(FAULT_NAMES) * config.target_samples_per_class_domain
    )
    actual_source = len(source_train_ds) + len(val_part1_ds) + len(val_part2_ds)
    if actual_source != expected_source:
        raise RuntimeError(f"Expected {expected_source} source windows, got {actual_source}.")
    if len(target_test_ds) != expected_target:
        raise RuntimeError(f"Expected {expected_target} target windows, got {len(target_test_ds)}.")

    n_train, n_val1, n_val2 = _split_count(
        config.source_samples_per_class_domain,
        config.source_train_ratio,
        config.val_part1_ratio,
    )
    print("\n================ Fixed sample split ================")
    print(f"Source total: {actual_source}")
    print(
        f"  source_train: {len(source_train_ds)} "
        f"({100.0 * len(source_train_ds) / actual_source:.2f}%; {n_train} per class-domain)"
    )
    print(
        f"  val_part1:    {len(val_part1_ds)} "
        f"({100.0 * len(val_part1_ds) / actual_source:.2f}%; {n_val1} per class-domain)"
    )
    print(
        f"  val_part2:    {len(val_part2_ds)} "
        f"({100.0 * len(val_part2_ds) / actual_source:.2f}%; {n_val2} per class-domain)"
    )
    print(f"Target test total: {len(target_test_ds)}")

    return (
        source_records,
        target_records,
        source_train_ds,
        val_part1_ds,
        val_part2_ds,
        target_test_ds,
        split_indices,
        target_indices,
    )


@torch.no_grad()
def export_sft_dataset(
    model: DARSModel,
    loader: DataLoader,
    output_path: str,
    max_records: Optional[int] = None,
) -> Dict[str, int]:
    """Export balanced source-only delegated-classifier examples for LoRA-SFT.

    Besides naturally mistaken hard augmentations, this function creates
    counterfactual revision examples from correctly labelled source samples:
    a plausible wrong candidate is placed at top-1 while the true class stays
    in the top-3 and remains supported by independent evidence. This fixes the
    previous accept-only bias without using any target-domain label or sample.
    """
    model.eval()
    all_records: List[Dict[str, object]] = []
    skipped_not_top3 = 0
    skipped_counterfactual_no_support = 0
    counterfactual_records = 0

    def evidence_support_count(evidence: Dict[str, object], label: int) -> int:
        supporters: List[int] = []
        if "advisor_rerank_label_id" in evidence:
            supporters.append(int(evidence["advisor_rerank_label_id"]))
        if "nearest_prototype_label_id" in evidence:
            supporters.append(int(evidence["nearest_prototype_label_id"]))
        pairwise = evidence.get("pairwise_advisor")
        if isinstance(pairwise, dict) and "final_label" in pairwise:
            supporters.append(int(pairwise["final_label"]))
        return sum(v == label for v in supporters)

    def teacher_confidence(
        evidence: Dict[str, object],
        target_label: int,
        selected_prob: float,
        action: str,
    ) -> float:
        detail_map = {
            int(item["label_id"]): float(item["score"])
            for item in evidence.get("advisor_rerank_details", [])
            if "label_id" in item and "score" in item
        }
        target_score = detail_map.get(target_label, 0.0)
        other_scores = [v for k, v in detail_map.items() if k != target_label]
        margin = target_score - max(other_scores) if other_scores else 0.0
        margin_component = 1.0 / (1.0 + math.exp(-2.0 * max(min(margin, 4.0), -4.0)))
        support = evidence_support_count(evidence, target_label)
        risk = float(evidence.get("risk_probability", 0.0))
        base = (
            0.55
            + 0.08 * min(support, 2)
            + 0.10 * margin_component
            + 0.06 * min(max(risk, 0.0), 1.0)
        )
        if action == "revise":
            base = max(base, float(model.config.sft_min_revision_confidence))
        else:
            base = max(base, min(max(selected_prob, 0.0), 1.0))
        return round(min(max(base, 0.55), 0.97), 6)

    def append_record(
        evidence: Dict[str, object],
        true_label: int,
        small_label: int,
        action: str,
        source_variant: str,
    ) -> None:
        nonlocal skipped_not_top3
        allowed_labels = {
            int(item.get("label_id"))
            for item in (evidence.get("candidate_labels", []) or evidence.get("top3", []) or [])
            if isinstance(item, dict) and int(item.get("label_id", -999)) in FAULT_NAMES
        }
        if allowed_labels and int(true_label) not in allowed_labels:
            skipped_not_top3 += 1
            return
        selected_detail = next(
            (
                d for d in evidence.get("advisor_rerank_details", [])
                if int(d["label_id"]) == true_label
            ),
            None,
        )
        selected_prob = next(
            (
                float(item["prob"])
                for item in evidence.get("top3", [])
                if int(item["label_id"]) == true_label
            ),
            0.0,
        )
        # V8.8 clean-SFT: the assistant target must not mention small-model
        # probability, prototype_z, domain_z, physics_z, advisor or pairwise terms.
        # This prevents the LoRA adapter from learning stale V7-style reasoning
        # templates that are not present in the independent LLM prompt.
        reasoning = clean_token_physical_reasoning(evidence, true_label, action)

        # V8.9.1 answer-first always-retrain LoRA-SFT target. This borrows the useful Qwen-style
        # training signal from the reference code: teach the model to emit a
        # machine-checkable answer tag first, then a compact evidence reasoning
        # chain, while still keeping the robust candidate_scores JSON contract.
        target = build_llm_sft_target(evidence, true_label, reasoning)
        all_records.append({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        REVIEWER_SYSTEM_PROMPT
                        + "\n\nDiagnostic evidence and task:\n"
                        + build_reviewer_user_message(evidence)
                    ),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        target, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
            "metadata": {
                "true_label": true_label,
                "small_model_label": small_label,
                "domain": evidence.get("domain"),
                "action": action,
                "source_variant": source_variant,
                "independent_support": evidence_support_count(evidence, true_label),
            },
        })

    def add_counterfactual_revisions(
        evidence: Dict[str, object],
        true_label: int,
        source_variant: str,
    ) -> None:
        nonlocal skipped_counterfactual_no_support, counterfactual_records
        copies = max(0, int(model.config.sft_counterfactual_revise_copies))
        if copies <= 0:
            return
        support_true = evidence_support_count(evidence, true_label)
        if support_true <= 0:
            skipped_counterfactual_no_support += 1
            return

        original_top3 = [dict(item) for item in evidence.get("top3", [])]
        wrong_candidates = [
            int(item["label_id"])
            for item in original_top3
            if int(item["label_id"]) != true_label
        ]
        if not wrong_candidates:
            return

        preferred: List[int] = []
        for a, b in CONFUSION_PAIRS:
            if true_label == a and b in wrong_candidates:
                preferred.append(b)
            elif true_label == b and a in wrong_candidates:
                preferred.append(a)
        ordered_wrong = preferred + [c for c in wrong_candidates if c not in preferred]

        detail_map = {
            int(item["label_id"]): float(item["score"])
            for item in evidence.get("advisor_rerank_details", [])
            if "label_id" in item and "score" in item
        }
        pairwise = evidence.get("pairwise_advisor")
        pairwise_true = (
            isinstance(pairwise, dict)
            and int(pairwise.get("final_label", -1)) == true_label
        )

        for copy_idx in range(copies):
            wrong_label = ordered_wrong[copy_idx % len(ordered_wrong)]
            score_gain = detail_map.get(true_label, -1e9) - detail_map.get(wrong_label, -1e9)
            if not (support_true >= 2 or pairwise_true or score_gain >= 0.02):
                skipped_counterfactual_no_support += 1
                continue
            cf = json.loads(json.dumps(evidence, ensure_ascii=False))
            item_map = {int(item["label_id"]): dict(item) for item in original_top3}
            remaining = [
                int(item["label_id"])
                for item in original_top3
                if int(item["label_id"]) not in {wrong_label, true_label}
            ]
            ordered_labels = [wrong_label, true_label] + remaining
            if len(ordered_labels) >= 3:
                synthetic_probs = [0.50, 0.35, 0.15]
            else:
                synthetic_probs = [0.56, 0.44]
            new_top3 = []
            for rank, label in enumerate(ordered_labels):
                item = item_map[label]
                item["prob"] = synthetic_probs[rank]
                new_top3.append(item)
            cf["top3"] = new_top3
            new_prob_map = {int(item["label_id"]): float(item["prob"]) for item in new_top3}
            for detail in cf.get("advisor_rerank_details", []):
                label_id = int(detail.get("label_id", -1))
                if label_id in new_prob_map:
                    detail["prob"] = new_prob_map[label_id]
            cf["small_model_label_id"] = wrong_label
            cf["small_model_label"] = FAULT_NAMES[wrong_label]
            cf["small_model_confidence"] = float(new_top3[0]["prob"])
            cf["uncertainty"] = max(float(cf.get("uncertainty", 0.0)), 0.65)
            cf["risk_probability"] = max(float(cf.get("risk_probability", 0.0)), 0.80)
            cf["counterfactual_training_example"] = True
            append_record(
                cf,
                true_label=true_label,
                small_label=wrong_label,
                action="revise",
                source_variant=f"{source_variant}_counterfactual_{copy_idx + 1}",
            )
            counterfactual_records += 1

    def collect_batch(
        batch_x: torch.Tensor,
        labels: torch.Tensor,
        domains: torch.Tensor,
        source_variant: str,
    ) -> None:
        nonlocal skipped_not_top3
        z, logits, mean_prob, entropy, variance, modality_weights = mc_predict(
            model, batch_x, model.config.mc_dropout_samples
        )
        physical = extract_physical_features(
            batch_x, model.config.window_size, model.config.input_channels, model.modalities
        )
        feature_tokens = model.feature_tokenizer(z)
        pred = logits.argmax(dim=1)
        proto_dist, domain_shift, nearest_class, _ = prototype_scores(z, pred, model.memory)
        rf = model.risk_features(z)
        risk_prob = F.softmax(model.router(rf), dim=1)[:, 1]

        for i in range(batch_x.size(0)):
            true_label = int(labels[i].item())
            evidence = build_evidence(
                i,
                mean_prob,
                entropy,
                variance,
                proto_dist,
                domain_shift,
                nearest_class,
                modality_weights,
                domains,
                feature_tokens,
                physical,
                [name for name, _indices in model.modalities],
            )
            top3 = [int(item["label_id"]) for item in evidence["top3"]]
            top3_probs = [float(item["prob"]) for item in evidence["top3"]]
            rerank_label, rerank_details = candidate_evidence_scores(
                z[i], physical[i], top3, top3_probs, model.memory, model.config
            )
            evidence["risk_probability"] = float(risk_prob[i].item())
            evidence["advisor_rerank_label_id"] = int(rerank_label)
            evidence["advisor_rerank_label"] = FAULT_NAMES[int(rerank_label)]
            evidence["advisor_rerank_details"] = rerank_details
            pairwise_review = apply_pairwise_advisor(
                model, z[i], physical[i], modality_weights[i], top3
            )
            if pairwise_review is not None:
                evidence["pairwise_advisor"] = pairwise_review

            small_label = int(pred[i].item())
            action = "accept" if small_label == true_label else "revise"
            append_record(
                evidence,
                true_label=true_label,
                small_label=small_label,
                action=action,
                source_variant=source_variant,
            )
            if action == "accept":
                add_counterfactual_revisions(evidence, true_label, source_variant)

    for x, labels, domains in loader:
        x = x.to(model.device, non_blocking=True)
        labels = labels.to(model.device, non_blocking=True)
        domains = domains.to(model.device, non_blocking=True)
        collect_batch(x, labels, domains, "clean")

        for copy_idx in range(max(0, int(model.config.sft_hard_augment_copies))):
            hard_x = augment_batch(
                x,
                model.config.window_size,
                model.config.input_channels,
                time_shift=max(16, model.config.time_shift * 2),
                noise_std=max(0.03, model.config.noise_std * 3.0),
                channel_dropout_p=max(0.15, model.config.channel_dropout_p * 3.0),
            )
            collect_batch(hard_x, labels, domains, f"hard_aug_{copy_idx + 1}")

    rng = random.Random(model.config.seed)
    accepts = [r for r in all_records if r["metadata"]["action"] == "accept"]
    revises = [r for r in all_records if r["metadata"]["action"] == "revise"]
    rng.shuffle(accepts)
    rng.shuffle(revises)

    def balanced_take(pool: List[Dict[str, object]], n: int) -> List[Dict[str, object]]:
        if n <= 0 or not pool:
            return []
        buckets: Dict[int, List[Dict[str, object]]] = {c: [] for c in sorted(FAULT_NAMES)}
        for rec in pool:
            try:
                buckets[int(rec["metadata"]["true_label"])].append(rec)
            except Exception:
                buckets[0].append(rec)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        result: List[Dict[str, object]] = []
        labels = [c for c in sorted(FAULT_NAMES) if buckets[c]]
        if not labels:
            return []
        ptr = {c: 0 for c in labels}
        while len(result) < n:
            progressed = False
            for c in labels:
                bucket = buckets[c]
                if not bucket:
                    continue
                result.append(bucket[ptr[c] % len(bucket)])
                ptr[c] += 1
                progressed = True
                if len(result) >= n:
                    break
            if not progressed:
                break
        return result

    if max_records is None:
        records = accepts + revises
    else:
        limit = max(1, int(max_records))
        desired_revise = int(
            round(
                limit
                * min(max(model.config.sft_min_revise_fraction, 0.0), 0.85)
            )
        )
        desired_accept = limit - desired_revise

        # V8.8 trains the LoRA branch as a hard-sample classifier, not a copy-the-small-model reviewer.
        # Therefore revision/counterfactual examples are class-balanced and intentionally dominate the SFT set.
        selected_revises = balanced_take(revises, desired_revise)
        selected_accepts = balanced_take(accepts, desired_accept)
        records = selected_revises + selected_accepts
        if len(records) < limit:
            pool = revises + accepts
            rng.shuffle(pool)
            if pool:
                records.extend(pool[i % len(pool)] for i in range(limit - len(records)))

    rng.shuffle(records)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "exported": len(records),
        "accept": sum(r["metadata"]["action"] == "accept" for r in records),
        "revise": sum(r["metadata"]["action"] == "revise" for r in records),
        "candidate_records": len(all_records),
        "counterfactual_records": counterfactual_records,
        "skipped_not_top3": skipped_not_top3,
        "skipped_counterfactual_no_support": skipped_counterfactual_no_support,
    }
    print(f"Exported LoRA-SFT data to {output}: {stats}")
    return stats


class ReviewerSFTDataset(Dataset):
    """Tokenized conversational SFT data with loss only on assistant tokens."""

    def __init__(self, jsonl_path: str, tokenizer: Any, max_length: int):
        self.samples: List[Dict[str, torch.Tensor]] = []
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"LoRA-SFT file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            if len(messages) < 2 or messages[-1].get("role") != "assistant":
                continue
            prompt_messages = messages[:-1]
            answer = str(messages[-1].get("content", "")).strip()
            if not answer:
                continue

            if getattr(tokenizer, "chat_template", None):
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = str(prompt_messages[-1].get("content", "")) + "\n\nAssistant:\n"

            prompt_ids = tokenizer(
                prompt_text, add_special_tokens=False
            )["input_ids"]
            eos = tokenizer.eos_token or ""
            answer_ids = tokenizer(
                answer + eos, add_special_tokens=False
            )["input_ids"]
            if len(answer_ids) >= max_length:
                answer_ids = answer_ids[: max_length - 1]
            max_prompt = max(1, max_length - len(answer_ids))
            if len(prompt_ids) > max_prompt:
                # Preserve both the instruction header and the final evidence fields.
                head = min(128, max_prompt // 3)
                tail = max_prompt - head
                prompt_ids = prompt_ids[:head] + prompt_ids[-tail:]

            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })
        if not self.samples:
            raise ValueError("No valid LoRA-SFT samples were created.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def make_sft_collator(pad_token_id: int):
    def collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(int(item["input_ids"].numel()) for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, item in enumerate(batch):
            length = int(item["input_ids"].numel())
            input_ids[i, :length] = item["input_ids"]
            attention_mask[i, :length] = 1
            labels[i, :length] = item["labels"]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return collate


def resolve_llm_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    name = dtype_name.lower()
    if device.type != "cuda":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def force_fresh_sft_export(
    model: DARSModel,
    loader: DataLoader,
    config: TrainConfig,
) -> Dict[str, int]:
    """Always overwrite the LoRA-SFT jsonl file for the current run."""
    if not config.sft_export_path:
        raise ValueError("auto_train_lora requires a non-empty sft_export_path")
    output = Path(config.sft_export_path)
    if output.exists():
        print(f"Removing old LoRA-SFT file before fresh export: {output}")
        output.unlink()
    return export_sft_dataset(
        model, loader, config.sft_export_path, config.sft_max_records
    )


def validate_lora_adapter_directory(adapter_dir: Path) -> Tuple[Path, Path]:
    """Return the required LoRA artifacts or raise for an incomplete directory."""
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    if not adapter_dir.is_dir():
        raise RuntimeError(f"LoRA adapter directory does not exist: {adapter_dir}")
    if not config_path.is_file() or config_path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty LoRA adapter config: {config_path}")
    try:
        parsed_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid LoRA adapter config: {config_path}: {exc}") from exc
    if not isinstance(parsed_config, dict) or not parsed_config:
        raise RuntimeError(f"LoRA adapter config is not a non-empty JSON object: {config_path}")

    weight_candidates = (
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    )
    weight_path = next(
        (path for path in weight_candidates if path.is_file() and path.stat().st_size > 0),
        None,
    )
    if weight_path is None:
        expected = " or ".join(str(path) for path in weight_candidates)
        raise RuntimeError(f"Missing or empty LoRA adapter weights: expected {expected}")
    return config_path, weight_path


def resolve_lora_adapter_directory(adapter_dir: Path) -> Path:
    """Use the requested adapter, or its retained backup after an interrupted install."""
    adapter_dir = Path(adapter_dir)
    try:
        validate_lora_adapter_directory(adapter_dir)
        return adapter_dir
    except RuntimeError as primary_error:
        backup_dir = adapter_dir.with_name(f"{adapter_dir.name}.backup")
        try:
            validate_lora_adapter_directory(backup_dir)
        except RuntimeError as backup_error:
            raise RuntimeError(
                f"Invalid local LoRA adapter: {primary_error}; "
                f"no valid retained backup is available: {backup_error}"
            ) from primary_error
        print(
            "Primary LoRA adapter is incomplete; loading retained backup instead: "
            f"{backup_dir}"
        )
        return backup_dir


def install_validated_lora_adapter(staging_dir: Path, output_dir: Path) -> Tuple[Path, Optional[Path]]:
    """Install a validated adapter while keeping the last valid output as backup."""
    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)
    validate_lora_adapter_directory(staging_dir)

    backup_dir = output_dir.with_name(f"{output_dir.name}.backup")
    moved_previous: Optional[Path] = None
    previous_was_valid = False
    if output_dir.exists():
        try:
            validate_lora_adapter_directory(output_dir)
            previous_was_valid = True
        except RuntimeError as exc:
            print(f"Existing LoRA output is incomplete and will be preserved separately: {exc}")

        if previous_was_valid:
            # The current valid output is safer than an older backup, so rotate it
            # only after the newly written staging directory has passed validation.
            if backup_dir.exists():
                if backup_dir.is_dir():
                    shutil.rmtree(backup_dir)
                else:
                    backup_dir.unlink()
            moved_previous = backup_dir
        else:
            moved_previous = output_dir.with_name(
                f"{output_dir.name}.incomplete-{time.time_ns()}"
            )
        output_dir.rename(moved_previous)

    try:
        staging_dir.rename(output_dir)
        validate_lora_adapter_directory(output_dir)
    except Exception:
        failed_dir: Optional[Path] = None
        if output_dir.exists():
            failed_dir = output_dir.with_name(
                f"{output_dir.name}.failed-{time.time_ns()}"
            )
            output_dir.rename(failed_dir)
        if moved_previous is not None and moved_previous.exists() and not output_dir.exists():
            moved_previous.rename(output_dir)
        if failed_dir is not None:
            print(f"Preserved failed LoRA installation for inspection: {failed_dir}")
        raise

    retained_backup = backup_dir if previous_was_valid and backup_dir.exists() else None
    return output_dir, retained_backup


def train_lora_adapter(config: TrainConfig, sft_path: str, device: torch.device) -> str:
    """Fine-tune Gemma with PEFT LoRA using a compact custom PyTorch loop.

    V8.9.1 intentionally retrains LoRA from scratch on every run. The existing
    adapter remains untouched during training. The new adapter is first saved
    and validated in a staging directory, then installed while retaining the
    last valid adapter as ``<lora_output_dir>.backup``.
    """
    output_dir = Path(config.lora_output_dir)
    adapter_config = output_dir / "adapter_config.json"
    if config.lora_reuse_existing and adapter_config.exists():
        print(
            "Warning: --reuse-existing-lora was provided, but V8.9.1 ignores it; "
            "LoRA will be retrained from the newly exported SFT data."
        )
    if device.type != "cuda":
        raise RuntimeError("Automatic LoRA training requires CUDA for this 2B local model.")

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise ImportError(
            "Automatic LoRA training requires transformers and peft. "
            "Install them in the current environment before running."
        ) from exc

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        try:
            validate_lora_adapter_directory(output_dir)
            print(f"Existing LoRA adapter will be preserved until replacement: {output_dir}")
        except RuntimeError as exc:
            print(f"Existing LoRA output is incomplete but will not be deleted before training: {exc}")

    dtype = resolve_llm_dtype(config.llm_dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_model_path, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs = dict(
        pretrained_model_name_or_path=config.llm_model_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    try:
        base_model = AutoModelForCausalLM.from_pretrained(dtype=dtype, **load_kwargs)
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(torch_dtype=dtype, **load_kwargs)

    target_modules = [x.strip() for x in config.lora_target_modules.split(",") if x.strip()]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(config.lora_rank),
        lora_alpha=int(config.lora_alpha),
        lora_dropout=float(config.lora_dropout),
        target_modules=target_modules,
        bias="none",
    )
    lora_model = get_peft_model(base_model, peft_config).to(device)
    if hasattr(lora_model, "print_trainable_parameters"):
        lora_model.print_trainable_parameters()
    if config.lora_gradient_checkpointing:
        try:
            lora_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            lora_model.gradient_checkpointing_enable()
        if hasattr(lora_model, "enable_input_require_grads"):
            lora_model.enable_input_require_grads()
    lora_model.config.use_cache = False

    dataset = ReviewerSFTDataset(sft_path, tokenizer, config.lora_max_seq_length)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(config.lora_batch_size)),
        shuffle=True,
        num_workers=0,
        collate_fn=make_sft_collator(int(tokenizer.pad_token_id)),
    )
    trainable = [p for p in lora_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.lora_lr),
        weight_decay=float(config.lora_weight_decay),
    )
    grad_accum = max(1, int(config.lora_grad_accum_steps))
    updates_per_epoch = max(1, math.ceil(len(loader) / grad_accum))
    total_updates = max(1, updates_per_epoch * max(1, int(config.lora_epochs)))
    warmup_steps = int(round(total_updates * max(0.0, float(config.lora_warmup_ratio))))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    use_fp16_scaler = dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler)

    print(
        f"LoRA-SFT start | samples={len(dataset)} | epochs={config.lora_epochs} | "
        f"micro_batch={config.lora_batch_size} | grad_accum={grad_accum} | "
        f"effective_batch={config.lora_batch_size * grad_accum} | dtype={dtype}"
    )
    lora_model.train()
    optimizer.zero_grad(set_to_none=True)
    epoch_losses: List[float] = []
    global_update = 0
    for epoch in range(max(1, int(config.lora_epochs))):
        running_loss = 0.0
        seen_batches = 0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(
                device_type="cuda",
                dtype=dtype if dtype in {torch.float16, torch.bfloat16} else torch.float32,
                enabled=dtype in {torch.float16, torch.bfloat16},
            ):
                outputs = lora_model(**batch)
                raw_loss = outputs.loss
                loss = raw_loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(raw_loss.detach().item())
            seen_batches += 1

            should_step = ((step + 1) % grad_accum == 0) or (step + 1 == len(loader))
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

        avg_loss = running_loss / max(seen_batches, 1)
        epoch_losses.append(avg_loss)
        print(
            f"LoRA epoch {epoch + 1}/{config.lora_epochs} | "
            f"loss {avg_loss:.4f} | updates {global_update}/{total_updates}"
        )

    lora_model.config.use_cache = True
    staging_dir = output_dir.with_name(
        f".{output_dir.name}.staging-{os.getpid()}-{time.time_ns()}"
    )
    staging_dir.mkdir(parents=False, exist_ok=False)
    metadata = {
        "base_model": config.llm_model_path,
        "sft_path": str(sft_path),
        "samples": len(dataset),
        "epochs": int(config.lora_epochs),
        "rank": int(config.lora_rank),
        "alpha": int(config.lora_alpha),
        "target_modules": target_modules,
        "epoch_losses": epoch_losses,
    }
    try:
        lora_model.save_pretrained(staging_dir)
        tokenizer.save_pretrained(staging_dir)
        (staging_dir / "dars_lora_training.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staged_config, staged_weights = validate_lora_adapter_directory(staging_dir)
        print(
            "Validated staged LoRA adapter: "
            f"config={staged_config.name}, weights={staged_weights.name}"
        )
        installed_dir, retained_backup = install_validated_lora_adapter(
            staging_dir, output_dir
        )
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(f"Saved and validated trained LoRA adapter: {installed_dir}")
    if retained_backup is not None:
        print(f"Retained previous valid LoRA adapter backup: {retained_backup}")

    del lora_model, base_model, optimizer, scheduler, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return str(installed_dir)


EVAL_RUNTIME_CONFIG_FIELDS = {
    # Runtime locations and output destinations may legitimately change after
    # training and must not be replaced by stale paths stored in the checkpoint.
    "data_path",
    "checkpoint_path",
    "small_model_checkpoint_path",
    "llm_model_path",
    "llm_adapter_path",
    "lora_output_dir",
    "results_json_path",
    "class_maintenance_csv_path",
    "class_maintenance_text_path",
    "maintenance_rule_evaluation_csv_path",
    "maintenance_rule_evaluation_json_path",
    # Keep the corrected report-generation policy rather than restoring the old
    # 320-token/incomplete-schema settings from the training checkpoint.
    "llm_generate_maintenance",
    "maintenance_revision_enabled",
    "maintenance_max_new_tokens",
    "maintenance_batch_size",
    "maintenance_stability_repeats",
    # Evaluation execution controls are runtime choices, not learned settings.
    "eval_only",
    "validation_use_llm",
    "eval_log_interval",
    "batch_size",
    "num_workers",
    "cpu_num_threads",
    "auto_train_lora",
    "skip_sft_export_when_reusing_lora",
}


def restore_eval_config_from_checkpoint(
    config: TrainConfig,
    saved_config: Dict[str, object],
) -> List[str]:
    """Restore training-time data/model settings before building eval datasets."""
    restored: List[str] = []
    for key, saved_value in saved_config.items():
        if key in EVAL_RUNTIME_CONFIG_FIELDS or not hasattr(config, key):
            continue
        current_value = getattr(config, key)
        if isinstance(current_value, tuple) and isinstance(saved_value, list):
            saved_value = tuple(saved_value)
        if current_value != saved_value:
            restored.append(f"{key}: {current_value!r} -> {saved_value!r}")
            setattr(config, key, saved_value)
    config.eval_only = True
    config.auto_train_lora = False
    config.maintenance_revision_enabled = False
    return restored


def verify_eval_split_fingerprints(
    checkpoint: Dict[str, object],
    split_indices: Dict[str, np.ndarray],
    target_indices: np.ndarray,
) -> None:
    """Fail fast when eval-only does not reproduce the checkpoint data split."""
    split_info = checkpoint.get("split", {})
    saved = split_info.get("fingerprints", {}) if isinstance(split_info, dict) else {}
    if not isinstance(saved, dict) or not saved:
        print("Warning: checkpoint has no split fingerprints; exact eval split cannot be verified.")
        return
    current = {
        "source_train": index_fingerprint(split_indices["source_train"]),
        "val_part1": index_fingerprint(split_indices["val_part1"]),
        "val_part2": index_fingerprint(split_indices["val_part2"]),
        "target_test": index_fingerprint(target_indices),
    }
    mismatches = {
        key: (str(saved.get(key)), value)
        for key, value in current.items()
        if str(saved.get(key)) != value
    }
    if mismatches:
        detail = "; ".join(
            f"{key}: checkpoint={old}, current={new}"
            for key, (old, new) in mismatches.items()
        )
        raise RuntimeError(
            "Eval-only data split does not match the training checkpoint. " + detail
        )
    print(f"Eval-only split fingerprints verified: {current}")


def train_dars_llm(config: TrainConfig) -> DARSModel:
    total_run_start = time.perf_counter()
    stage_times: Dict[str, float] = {}
    checkpoint_file = Path(config.checkpoint_path)
    preloaded_checkpoint: Optional[Dict[str, object]] = None
    saved_config: Dict[str, object] = {}
    if config.eval_only:
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found for --eval-only: {checkpoint_file}")
        preloaded_checkpoint = torch.load(
            checkpoint_file, map_location="cpu", weights_only=False
        )
        raw_saved_config = preloaded_checkpoint.get("config", {})
        saved_config = raw_saved_config if isinstance(raw_saved_config, dict) else {}
        restored = restore_eval_config_from_checkpoint(config, saved_config)
        if restored:
            print("Eval-only restored training-time configuration from checkpoint:")
            for item in restored:
                print(f"  {item}")
        else:
            print("Eval-only configuration already matches the training checkpoint.")

    set_seed(config.seed, config.cpu_num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    modalities = parse_modality_spec(config.modality_spec, config.input_channels)
    print(f"Training seed: {config.seed}; fixed data split seed: {config.split_seed}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Modalities: {modalities}")
    print("Full-model mode: LoRA-assisted diagnosis and base-backbone post-diagnosis maintenance drafting are enabled for the final target test.")
    print("LLM-SFT schema: V9.2 direct delegate: compact answer-first SFT + physics-relative evidence; valid LLM candidate_scores are accepted as the routed final classifier; prompt tokens masked with -100.")
    if config.router_budgeted_routing:
        print(
            f"Selective LLM budget cap (V8.8.1 candidate-scoring direct delegate mode): min({config.max_llm_calls_per_eval} calls, "
            f"{100.0 * config.llm_max_call_rate:.1f}% of evaluation split)"
        )
    else:
        print("Selective LLM calls: dynamic threshold-only delegation (no fixed 800/10% cap)")
    print(
        "Routing policy: "
        + (
            "calibrated threshold + optional budget cap"
            if config.router_budgeted_routing
            else "calibrated threshold-only dynamic delegation"
        )
    )

    data_stage_start = time.perf_counter()
    (
        source_records,
        target_records,
        source_train_ds,
        val_part1_ds,
        val_part2_ds,
        target_test_ds,
        split_indices,
        target_indices,
    ) = create_datasets(config)
    stage_times["data_preparation"] = time.perf_counter() - data_stage_start
    if preloaded_checkpoint is not None:
        verify_eval_split_fingerprints(
            preloaded_checkpoint, split_indices, target_indices
        )

    print_sample_split_info("source_train: small model / prototypes / pairwise", source_train_ds)
    print_sample_split_info("val_part1: Router training + LoRA-SFT export", val_part1_ds)
    print_sample_split_info("val_part2: small-model selection + Router threshold calibration", val_part2_ds)
    print_sample_split_info("unseen target-domain test", target_test_ds)

    use_cuda = device.type == "cuda"
    loader_common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=use_cuda,
        persistent_workers=config.num_workers > 0,
    )
    source_train_loader = DataLoader(source_train_ds, shuffle=True, **loader_common)
    val_part1_loader = DataLoader(val_part1_ds, shuffle=True, **loader_common)
    val_part1_eval_loader = DataLoader(val_part1_ds, shuffle=False, **loader_common)
    val_part2_loader = DataLoader(val_part2_ds, shuffle=False, **loader_common)
    target_test_loader = DataLoader(target_test_ds, shuffle=False, **loader_common)


    model = DARSModel(config, modalities, device)
    if not config.eval_only:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    if config.eval_only:
        if preloaded_checkpoint is None:
            raise RuntimeError("Eval-only checkpoint was not preloaded before dataset creation.")
        checkpoint = preloaded_checkpoint
        model.initialize_requested_router()
        model.load_state_dict(checkpoint["model"], strict=False)
        if "router_threshold" in saved_config:
            calibrated_threshold = float(saved_config["router_threshold"])
            model.config.router_threshold = calibrated_threshold
            config.router_threshold = calibrated_threshold
            print(f"Loaded calibrated Router threshold: {calibrated_threshold:.6f}")
        refresh_prototypes(model, source_train_loader)

        saved_adapter = str(saved_config.get("llm_adapter_path", "") or "")
        adapter_to_load = config.llm_adapter_path or saved_adapter
        if config.llm_mode == "local_hf":
            # Evaluation-only must never repeat the expensive LoRA training stage.
            # Reuse the adapter path stored in the checkpoint, an explicit CLI
            # path, or the default output directory, in that order.
            if not adapter_to_load:
                default_adapter = Path(config.lora_output_dir)
                if (default_adapter / "adapter_config.json").exists():
                    adapter_to_load = str(default_adapter)
            if not adapter_to_load:
                raise FileNotFoundError(
                    "No saved LoRA adapter was found for --eval-only. Pass "
                    "--llm-adapter-path or keep the adapter directory beside the checkpoint."
                )
            print(f"Eval-only mode: reusing saved LoRA adapter without retraining: {adapter_to_load}")
            model.config.llm_adapter_path = adapter_to_load
            config.llm_adapter_path = adapter_to_load
            model.reviewer.load(adapter_to_load)
            if not model.reviewer.loaded or not model.reviewer.using_adapter:
                raise RuntimeError(
                    "The saved LoRA reviewer could not be loaded in --eval-only mode; "
                    "stop instead of exporting empty maintenance metrics."
                )

        target_test = evaluate(
            model,
            target_test_loader,
            use_reviewer=True,
            allow_llm=True,
            split_name="target_test",
        )
        stage_times["evaluation_only"] = time.perf_counter() - total_run_start
        stage_times["total_run"] = stage_times["evaluation_only"]

        maintenance_metrics, _maintenance_summaries = save_final_maintenance_outputs(
            target_test.get("prediction_reports", []),
            model.modalities,
            float(model.config.router_threshold),
            config,
        )
        print_final_complete_model_results(target_test)
        print_final_maintenance_results(maintenance_metrics)

        if config.results_json_path:
            result_file = Path(config.results_json_path)
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(
                json.dumps(
                    json_ready(final_results_payload(config, target_test, maintenance_metrics)),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\nSaved final result JSON: {result_file}")
        return model

    small_stage_start = time.perf_counter()
    # ------------------------------------------------------------------
    # Stage 1: train only the TCN encoder and small predictor.
    # val_part2 selects the best epoch. Router and
    # reviewer statistics are deliberately excluded from checkpoint selection.
    # ------------------------------------------------------------------
    main_params = (
        list(model.encoder.parameters())
        + list(model.predictor.parameters())
        + list(model.feature_tokenizer.parameters())
        + list(model.token_predictor.parameters())
    )
    optimizer = torch.optim.AdamW(main_params, lr=config.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs), eta_min=config.lr * 0.05
    )
    ce = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    best_score = -1.0
    best_robust_score = -1.0
    best_selection_key = (-1.0, -1.0, -1.0)
    best_state = None
    best_epoch = 0

    reuse_small_path = str(config.small_model_checkpoint_path or "").strip()
    if reuse_small_path:
        reuse_file = Path(reuse_small_path)
        if not reuse_file.exists():
            raise FileNotFoundError(f"Small-model checkpoint not found: {reuse_file}")
        reuse_obj = torch.load(reuse_file, map_location=device, weights_only=False)
        reuse_state = reuse_obj.get("model", reuse_obj) if isinstance(reuse_obj, dict) else reuse_obj
        current_state = model.state_dict()
        small_only = {
            k: v for k, v in reuse_state.items()
            if (k.startswith("encoder.") or k.startswith("predictor."))
            and k in current_state and tuple(current_state[k].shape) == tuple(v.shape)
        }
        if not small_only:
            raise RuntimeError(
                f"No compatible encoder/predictor weights found in {reuse_file}."
            )
        model.load_state_dict(small_only, strict=False)
        refresh_prototypes(model, source_train_loader)
        reused_metrics = evaluate_small_model(model, val_part2_loader)
        best_score = float(reused_metrics["small_macro_f1"])
        reused_robust = evaluate_small_model_robust(model, val_part2_loader, config.seed + 93001)
        best_robust_score = float(reused_robust["robust_macro_f1"])
        best_selection_key = (round(best_score, 6), round(best_robust_score, 6), -float(reused_metrics["small_nll"]))
        best_epoch = int(reuse_obj.get("best_epoch", 0)) if isinstance(reuse_obj, dict) else 0
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"\nReused Stage-1 checkpoint from: {reuse_file}")
    else:
        for epoch in range(config.epochs):
            current_domain_align_weight = progressive_domain_align_weight(config, epoch)
            model.encoder.train()
            model.predictor.train()
            model.feature_tokenizer.train()
            model.token_predictor.train()
            total_loss = total_cls = total_proto = total_sup = total_pair_focus = total_align = total_token = 0.0
            correct = total = nb = 0
            for x, labels, domains in source_train_loader:
                x = x.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                domains = domains.to(device, non_blocking=True)
                if config.augment:
                    x = augment_batch(
                        x, config.window_size, config.input_channels,
                        config.time_shift, config.noise_std, config.channel_dropout_p,
                        config.amplitude_jitter, config.time_mask_ratio,
                    )
                optimizer.zero_grad(set_to_none=True)
                z, _weights = model.encoder(x)
                logits = model.predictor(z)
                cls = ce(logits, labels)
                proto = prototype_loss(z, labels, model.memory)
                sup = supervised_contrastive_loss(z, labels, config.supcon_temperature)
                pair_focus = confusion_pair_focus_loss(logits, labels)
                align = class_conditional_domain_alignment_loss(z, labels, domains)
                tokens = model.feature_tokenizer(z)
                token_logits = model.token_predictor(tokens.mean(dim=1))
                token_loss = ce(token_logits, labels)
                mixup = torch.zeros((), device=device)
                loss = (
                    cls
                    + config.proto_weight * proto
                    + config.supcon_weight * sup
                    + config.confusion_pair_weight * pair_focus
                    + current_domain_align_weight * align
                    + float(getattr(config, "token_align_weight", 0.05)) * token_loss
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(main_params, 5.0)
                optimizer.step()

                total_loss += float(loss.item())
                total_cls += float(cls.item())
                total_proto += float(proto.item())
                total_sup += float(sup.item())
                total_pair_focus += float(pair_focus.item())
                total_align += float(align.item())
                total_token += float(token_loss.item())
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += int(labels.numel())
                nb += 1
            refresh_prototypes(model, source_train_loader)
            small_val_metrics = evaluate_small_model(model, val_part2_loader)
            robust_val_metrics = evaluate_small_model_robust(
                model, val_part2_loader, config.seed + 93001
            ) if config.robust_selection else {
                "robust_acc": small_val_metrics["small_acc"],
                "robust_macro_f1": small_val_metrics["small_macro_f1"],
                "robust_mcc": small_val_metrics["small_mcc"],
            }
            selection_score = float(small_val_metrics["small_macro_f1"])
            robust_score = float(robust_val_metrics["robust_macro_f1"])
            # Clean source validation is the primary criterion. Once it saturates,
            # fixed stress-validation F1 and then lower NLL select the more robust
            # checkpoint without looking at target-domain labels.
            selection_key = (
                round(selection_score, 6),
                round(robust_score, 6),
                -float(small_val_metrics["small_nll"]),
            )
            if selection_key > best_selection_key:
                best_selection_key = selection_key
                best_score = selection_score
                best_robust_score = robust_score
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            print(f"\nEpoch {epoch + 1}/{config.epochs} [small model]")
            print(
                f"  train loss {total_loss / max(nb, 1):.4f} | "
                f"cls {total_cls / max(nb, 1):.4f} | "
                f"proto {total_proto / max(nb, 1):.4f} | "
                f"supcon {total_sup / max(nb, 1):.4f} | "
                f"pair {total_pair_focus / max(nb, 1):.4f}"
            )
            print(
                f"  V10.7 domain-align schedule: lambda={current_domain_align_weight:.5f} "
                f"(max={config.domain_align_weight:.5f}, warmup={config.domain_align_warmup_epochs}, "
                f"ramp={config.domain_align_ramp_epochs})"
            )
            scheduler.step()

    if best_state is None:
        raise RuntimeError("No best small-model checkpoint was selected.")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=False)
    refresh_prototypes(model, source_train_loader)
    print(f"\nStage-1 checkpoint selected at epoch {best_epoch}.")
    model.initialize_requested_router()
    print(
        f"Router initialized after small-model selection: {model.active_router_type} "
        "(Stage-1 RNG isolated from Router architecture)"
    )
    print(
        f"Pairwise bank initialized after small-model selection: {len(model.pairwise_pairs)} pairs "
        "(Stage-1 RNG isolated from pairwise-bank size)"
    )
    stage_times["small_model_training"] = time.perf_counter() - small_stage_start

    pairwise_stage_start = time.perf_counter()
    # ------------------------------------------------------------------
    # Stage 2: freeze the selected small model and train lightweight
    # confusion-pair advisors on source_train only.
    # ------------------------------------------------------------------
    pairwise_optimizer = torch.optim.AdamW(
        model.pairwise_advisors.parameters(), lr=config.pairwise_lr, weight_decay=1e-4
    )
    pstats = train_pairwise_advisors(
        model, source_train_loader, pairwise_optimizer, config.pairwise_epochs
    )
    print(
        f"Pairwise advisor | loss {pstats['pairwise_loss']:.4f} | "
        f"acc {pstats['pairwise_acc']:.2f}% | samples {pstats['pairwise_samples']:.0f} | "
        f"pairs {len(model.pairwise_pairs)}"
    )
    stage_times["pairwise_advisor_training"] = time.perf_counter() - pairwise_stage_start

    router_stage_start = time.perf_counter()
    # ------------------------------------------------------------------
    # Stage 3: train Router on val_part1. The encoder, predictor
    # and prototypes are fixed, so Router learns only whether the small-model
    # prediction is risky enough to justify an LLM call.
    # ------------------------------------------------------------------
    router_optimizer = torch.optim.AdamW(
        model.router.parameters(), lr=config.router_lr, weight_decay=1e-4
    )
    rstats = train_router(
        model, val_part1_loader, router_optimizer, config.router_epochs
    )
    print(
        f"Router train | loss {rstats['router_loss']:.4f} | "
        f"risk-target rate {rstats['risk_target_rate']:.2f}% | "
        f"training AUC {rstats['risk_auc']:.3f}"
    )
    stage_times["router_training"] = time.perf_counter() - router_stage_start

    calibration_stage_start = time.perf_counter()
    # ------------------------------------------------------------------
    # Stage 4: use val_part2 to calibrate the threshold.
    # This set never updates Router weights. It only determines the cost-
    # reliability operating point: catch the requested fraction of small-model
    # errors with the fewest possible LLM calls.
    # ------------------------------------------------------------------
    calibration = calibrate_router_threshold(
        model,
        val_part2_loader,
        config.router_min_error_recall,
        config.router_calibration_steps,
    )
    config.router_threshold = float(model.config.router_threshold)
    print("\nRouter threshold calibration")
    print(
        f"  threshold {calibration['threshold']:.6f} | "
        f"target source {calibration['target_source']} | "
        f"error recall {100.0 * calibration['error_recall']:.2f}% | "
        f"precision {100.0 * calibration['error_precision']:.2f}% | "
        f"augmented trigger rate {100.0 * calibration['trigger_rate']:.2f}% | "
        f"clean threshold trigger {100.0 * calibration['clean_trigger_rate']:.2f}% | "
        f"coverage {100.0 * calibration['coverage']:.2f}% | "
        f"selective acc {100.0 * calibration['selective_acc']:.2f}% | "
        f"risk AUC {calibration['risk_auc']:.3f}"
    )
    stage_times["router_calibration"] = time.perf_counter() - calibration_stage_start

    sft_stage_start = time.perf_counter()
    # ------------------------------------------------------------------
    # Stage 5: build source-only SFT data, fine-tune LoRA, then lazily load the
    # trained adapter. No target-domain sample enters LoRA training.
    # ------------------------------------------------------------------
    if config.auto_train_lora:
        force_fresh_sft_export(model, val_part1_eval_loader, config)
    elif config.sft_export_path and not bool(
        getattr(config, "skip_sft_export_when_reusing_lora", False)
    ):
        # Manual mode can still export a fresh jsonl for inspection.
        force_fresh_sft_export(model, val_part1_eval_loader, config)
    elif bool(getattr(config, "skip_sft_export_when_reusing_lora", False)):
        print("Reuse existing LoRA adapter; skip redundant SFT export.")
    stage_times["sft_export"] = time.perf_counter() - sft_stage_start

    lora_stage_start = time.perf_counter()
    if config.llm_mode == "local_hf":
        if config.auto_train_lora:
            trained_adapter = train_lora_adapter(config, config.sft_export_path, device)
            config.llm_adapter_path = trained_adapter
            model.config.llm_adapter_path = trained_adapter
            model.reviewer.load(trained_adapter)
        else:
            model.reviewer.load(config.llm_adapter_path)
        if not model.reviewer.loaded or not model.reviewer.using_adapter:
            raise RuntimeError(
                "The LoRA reviewer could not be loaded; stop instead of running a "
                "small-model-only final evaluation with empty maintenance metrics."
            )
    stage_times["lora_training_and_loading"] = time.perf_counter() - lora_stage_start

    # Save the complete selected small model, trained Router/advisors, calibrated
    # threshold and the path of the source-trained LoRA adapter.
    # calibrated threshold. This checkpoint is the one used for target testing.
    final_checkpoint = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "prototype_memory": model.memory.state_dict(),
        "config": asdict(config),
        "fault_names": FAULT_NAMES,
        "modalities": modalities,
        "best_epoch": best_epoch,
        "router_calibration": calibration,
        "split": {
            "protocol": "balanced_window_70_15_15",
            "training_seed": int(config.seed),
            "split_seed": int(config.split_seed),
            "target_selection_seed": int(config.split_seed + 1000),
            "source_domains": list(config.source_domains),
            "target_domains": list(config.target_domains),
            "source_samples_per_class_domain": config.source_samples_per_class_domain,
            "target_samples_per_class_domain": config.target_samples_per_class_domain,
            "source_train_indices": split_indices["source_train"].tolist(),
            "val_part1_indices": split_indices["val_part1"].tolist(),
            "val_part2_indices": split_indices["val_part2"].tolist(),
            "target_test_indices": target_indices.tolist(),
            "fingerprints": {
                "source_train": index_fingerprint(split_indices["source_train"]),
                "val_part1": index_fingerprint(split_indices["val_part1"]),
                "val_part2": index_fingerprint(split_indices["val_part2"]),
                "target_test": index_fingerprint(target_indices),
            },
        },
    }
    torch.save(final_checkpoint, checkpoint_file)
    print(f"Saved final checkpoint: {checkpoint_file}")

    # Final target-domain evaluation of the complete PSCG-LLM system.
    final_eval_stage_start = time.perf_counter()
    final_target_test = evaluate(
        model,
        target_test_loader,
        use_reviewer=True,
        allow_llm=True,
        split_name="final_target_test",
    )
    stage_times["final_evaluation"] = time.perf_counter() - final_eval_stage_start
    stage_times["total_run"] = time.perf_counter() - total_run_start

    maintenance_metrics, _maintenance_summaries = save_final_maintenance_outputs(
        final_target_test.get("prediction_reports", []),
        model.modalities,
        float(model.config.router_threshold),
        config,
    )
    print_final_complete_model_results(final_target_test)
    print_final_maintenance_results(maintenance_metrics)

    if config.results_json_path:
        result_file = Path(config.results_json_path)
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(
            json.dumps(
                json_ready(final_results_payload(config, final_target_test, maintenance_metrics)),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved final result JSON: {result_file}")
    return model


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="Train/evaluate the final clean PSCG-LLM model on PU bearings and report only complete-model diagnosis plus maintenance metrics."
    )
    p.add_argument("--data-path", default=r"D:\HM-Shen\1\code\LLM\Datasets\PUBD")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--router-lr", type=float, default=1e-3)
    p.add_argument(
        "--seed",
        type=int,
        default=422,
        help="Training seed only: model initialization, shuffle, augmentation, Dropout, Router and LoRA.",
    )
    p.add_argument(
        "--split-seed",
        dest="split_seed",
        type=int,
        default=42,
        help="Fixed data-split seed: source 70/15/15 membership and target-test sample selection.",
    )
    p.add_argument("--source-domains", default="S1,S3")
    p.add_argument("--target-domains", default="S4")
    p.add_argument("--source-samples-per-class-domain", type=int, default=1000)
    p.add_argument("--target-samples-per-class-domain", type=int, default=1000)
    p.add_argument("--source-train-ratio", type=float, default=0.70)
    p.add_argument("--val-part1-ratio", type=float, default=0.15)
    p.add_argument("--val-part2-ratio", type=float, default=0.15)

    p.add_argument("--window-size", type=int, default=1280, help="Direct fixed-point PU window length. Default: 1280.")
    p.add_argument("--stride", type=int, default=512, help="Direct fixed-point sliding-window stride. Default: 512.")
    p.add_argument("--sampling-rate-hz", type=int, default=64000)
    p.add_argument("--window-revolutions", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument(
        "--disable-rotation-normalization",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--input-channels", type=int, default=3)
    p.add_argument("--no-drop-first-column", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--channel-indices", default=None)
    p.add_argument(
        "--modality-spec",
        default="motor_current:0,1;vibration:2",
    )
    p.add_argument("--window-norm", default="train_scaler", choices=["sample", "train_scaler", "none"])
    p.add_argument("--max-windows-per-trial", type=int, default=600)

    p.add_argument("--feature-dim", type=int, default=96)
    p.add_argument("--branch-hidden", type=int, default=48)
    p.add_argument("--tcn-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--label-smoothing", type=float, default=0.03)
    p.add_argument("--proto-weight", type=float, default=0.10)
    p.add_argument("--supcon-weight", type=float, default=0.05)
    p.add_argument("--confusion-pair-weight", type=float, default=0.15)
    p.add_argument("--supcon-temperature", type=float, default=0.20)
    p.add_argument(
        "--domain-align-weight", type=float, default=0.05,
        help="Maximum source-only domain-alignment coefficient after the V10.7 warm-up/ramp.",
    )
    p.add_argument("--domain-align-warmup-epochs", type=int, default=4)
    p.add_argument("--domain-align-ramp-epochs", type=int, default=6)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--time-shift", type=int, default=16)
    p.add_argument("--noise-std", type=float, default=0.015)
    p.add_argument("--channel-dropout-p", type=float, default=0.03)
    p.add_argument("--amplitude-jitter", type=float, default=0.10)
    p.add_argument("--time-mask-ratio", type=float, default=0.03)
    p.add_argument("--disable-robust-selection", action="store_true")
    p.add_argument("--robust-val-time-shift", type=int, default=24)
    p.add_argument("--robust-val-noise-std", type=float, default=0.025)
    p.add_argument("--robust-val-channel-dropout-p", type=float, default=0.08)
    p.add_argument("--robust-val-amplitude-jitter", type=float, default=0.15)
    p.add_argument("--robust-val-time-mask-ratio", type=float, default=0.05)
    p.add_argument("--mc-dropout-samples", type=int, default=8)

    p.add_argument("--router-type", default="mlp", choices=["logistic", "mlp"])
    p.add_argument("--router-epochs", type=int, default=20)
    p.add_argument("--router-threshold", type=float, default=0.50)
    p.add_argument("--router-target-mode", default="error", choices=["error", "hybrid"])
    p.add_argument("--router-positive-quantile", type=float, default=0.85)
    p.add_argument("--router-hard-augment-copies", type=int, default=2)
    p.add_argument("--router-hard-time-shift", type=int, default=24)
    p.add_argument("--router-hard-noise-std", type=float, default=0.05)
    p.add_argument("--router-hard-channel-dropout-p", type=float, default=0.20)
    p.add_argument("--router-positive-weight-cap", type=float, default=8.0)
    p.add_argument(
        "--enable-budget-cap",
        action="store_true",
        help="Keep for compatibility; V8.8 uses Router-only hard-sample selection with a post-threshold top-risk cap by default.",
    )
    p.add_argument(
        "--disable-budgeted-routing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--router-confidence-gate",
        type=float,
        default=1.0,
        help="Do not route samples whose small-model confidence exceeds this value; set 1.0 to disable.",
    )
    p.add_argument("--router-min-error-recall", type=float, default=0.80)
    p.add_argument("--router-calibration-steps", type=int, default=201)

    p.add_argument("--advisor-prob-weight", type=float, default=1.0)
    p.add_argument("--advisor-proto-weight", type=float, default=0.6)
    p.add_argument("--advisor-domain-weight", type=float, default=0.4)
    p.add_argument("--advisor-physics-weight", type=float, default=0.8)
    p.add_argument("--disable-pairwise-advisor", action="store_true")
    p.add_argument(
        "--enable-evidence-consensus",
        action="store_true",
        help="Opt-in deterministic consensus fallback after LLM rejection. Disabled by default to avoid non-LLM harmful overrides.",
    )
    p.add_argument("--pairwise-epochs", type=int, default=10)
    p.add_argument("--pairwise-lr", type=float, default=1e-3)

    p.add_argument("--llm-mode", default="local_hf", choices=["none", "local_hf"])
    p.add_argument("--llm-model-path", default=r"D:\HM-Shen\1\code\LLM\Gemma-2-2b-it")
    p.add_argument("--llm-adapter-path", default="")
    p.add_argument("--llm-dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--llm-max-input-tokens", type=int, default=2048)
    p.add_argument("--llm-max-new-tokens", type=int, default=384)
    p.add_argument("--max-llm-calls-per-eval", type=int, default=640)
    p.add_argument("--llm-max-call-rate", type=float, default=0.08)
    p.add_argument("--llm-batch-size", type=int, default=8)
    p.add_argument("--llm-revision-min-confidence", type=float, default=0.55)
    p.add_argument("--llm-revision-min-evidence-confidence", type=float, default=0.65)
    p.add_argument("--llm-revision-protect-small-confidence", type=float, default=0.45)
    p.add_argument("--llm-revision-high-risk", type=float, default=0.15)
    p.add_argument("--llm-revision-min-support", type=int, default=2)
    p.add_argument("--llm-revision-min-score-gain", type=float, default=0.00)
    p.add_argument("--llm-override-min-confidence", type=float, default=0.82)
    p.add_argument("--llm-override-min-evidence-confidence", type=float, default=0.50)
    p.add_argument("--llm-override-max-small-confidence", type=float, default=0.60)
    p.add_argument("--llm-override-min-support", type=int, default=1)
    p.add_argument("--llm-override-min-score-gain", type=float, default=-0.25)
    p.add_argument("--llm-normal-flip-min-confidence", type=float, default=0.90)
    p.add_argument(
        "--llm-delegate-min-confidence",
        type=float,
        default=0.70,
        help="Minimum candidate_score required for a V10.7 evidence-consensus LLM label change.",
    )
    p.add_argument(
        "--llm-delegate-min-score-margin",
        type=float,
        default=0.20,
        help="Minimum top-score margin required for a V10.7 evidence-consensus LLM label change.",
    )
    p.add_argument(
        "--llm-delegate-min-override-support",
        type=int,
        default=2,
        help="Minimum number of independent pre-LLM supporters required when the delegated LLM changes the small-model label.",
    )
    p.add_argument(
        "--llm-delegate-max-small-confidence-for-override",
        type=float,
        default=0.95,
        help="Protect only near-certain small-model predictions; strong 2-of-3 consensus may correct other hard samples.",
    )
    p.add_argument(
        "--llm-delegate-min-risk-margin",
        type=float,
        default=0.00,
        help="Extra Router-risk margin for LLM changes; 0 keeps Router selection itself as the risk criterion.",
    )
    p.add_argument(
        "--llm-delegate-min-relative-support",
        type=float,
        default=0.50,
        help="Require the LLM label to have enough non-LLM relative candidate support.",
    )
    p.add_argument(
        "--llm-delegate-min-relative-support-gain",
        type=float,
        default=0.05,
        help="Require relative_support(label) to exceed small-label support by this margin.",
    )
    p.add_argument(
        "--allow-repaired-llm-json",
        action="store_true",
        help="V9.1: kept for compatibility; exact JSON is required by default.",
    )
    p.add_argument(
        "--enable-validation-llm",
        action="store_true",
        help="Also run real LLM generation on val_part2; disabled by default for speed.",
    )
    p.add_argument(
        "--eval-log-interval",
        type=int,
        default=20,
        help="Print evaluation progress every N batches; 0 disables progress logs.",
    )
    p.add_argument(
        "--advisor-mode", default="llm_delegate",
        choices=["prototype", "llm_explain", "llm_revise", "llm_delegate"],
    )
    p.add_argument("--llm-feature-token-count", type=int, default=8)
    p.add_argument("--llm-feature-token-dim", type=int, default=16)
    p.add_argument("--token-align-weight", type=float, default=0.05)
    p.add_argument(
        "--disable-llm-maintenance",
        action="store_true",
        help="Disable the post-diagnosis zero-shot structured maintenance-draft generation.",
    )
    p.add_argument("--maintenance-max-new-tokens", type=int, default=640)
    p.add_argument("--maintenance-batch-size", type=int, default=8)
    p.add_argument(
        "--maintenance-stability-repeats",
        type=int,
        default=1,
        help="Optional repeat count for priority stability; 1 performs no repeat and reports stability as N/A.",
    )

    p.add_argument("--disable-auto-lora", action="store_true")
    p.add_argument("--lora-output-dir", default="dars_llm_lora_adapter_v10_13_full_structured_maintenance")
    p.add_argument("--lora-epochs", type=int, default=2)
    p.add_argument("--lora-batch-size", type=int, default=2)
    p.add_argument("--lora-grad-accum-steps", type=int, default=8)
    p.add_argument("--lora-lr", type=float, default=2e-4)
    p.add_argument("--lora-weight-decay", type=float, default=0.01)
    p.add_argument("--lora-warmup-ratio", type=float, default=0.05)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-max-seq-length", type=int, default=1024)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--disable-lora-gradient-checkpointing", action="store_true")
    p.add_argument("--reuse-existing-lora", action="store_true", help="Accepted for compatibility but ignored in V8.9.1; LoRA is always retrained.")
    p.add_argument("--sft-export-path", default="dars_llm_sft_train_v10_13_full_structured_maintenance.jsonl")
    p.add_argument("--sft-max-records", type=int, default=2400)
    p.add_argument("--sft-hard-augment-copies", type=int, default=2)
    p.add_argument("--sft-counterfactual-revise-copies", type=int, default=2)
    p.add_argument("--sft-min-revise-fraction", type=float, default=0.55)
    p.add_argument("--sft-min-revision-confidence", type=float, default=0.78)
    p.add_argument("--checkpoint-path", default="dars_llm_best_70_15_15_v10_13_full_structured_maintenance.pth")
    p.add_argument("--results-json-path", default="dars_llm_results_v10_14_final_clean.json")
    p.add_argument("--class-maintenance-csv-path", default="dars_llm_pu_class_maintenance_summary_v10_14_final_clean.csv")
    p.add_argument("--class-maintenance-text-path", default="dars_llm_pu_class_maintenance_summary_v10_14_final_clean.txt")
    p.add_argument("--maintenance-rule-evaluation-csv-path", default="dars_llm_pu_maintenance_rule_evaluation_v10_14_final_clean.csv")
    p.add_argument("--maintenance-rule-evaluation-json-path", default="dars_llm_pu_maintenance_rule_evaluation_v10_14_final_clean.json")
    p.add_argument(
        "--small-model-checkpoint",
        default="",
        help="Reuse only encoder/predictor weights from a prior checkpoint for a fair baseline/LLM comparison.",
    )
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-test-each-epoch", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu-num-threads", type=int, default=1)
    a = p.parse_args()

    return TrainConfig(
        data_path=a.data_path,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        router_lr=a.router_lr,
        seed=a.seed,
        split_seed=getattr(a, "split_seed", 42),
        source_domains=parse_csv_tuple(a.source_domains),
        target_domains=parse_csv_tuple(a.target_domains),
        source_samples_per_class_domain=a.source_samples_per_class_domain,
        target_samples_per_class_domain=a.target_samples_per_class_domain,
        source_train_ratio=a.source_train_ratio,
        val_part1_ratio=a.val_part1_ratio,
        val_part2_ratio=a.val_part2_ratio,
        window_size=a.window_size,
        stride=a.stride,
        sampling_rate_hz=a.sampling_rate_hz,
        window_revolutions=0.0,
        rotation_normalize=False,
        input_channels=a.input_channels,
        drop_first_column=False,
        channel_indices=parse_channel_indices(a.channel_indices),
        modality_spec=a.modality_spec,
        window_norm=a.window_norm,
        max_windows_per_trial=none_if_negative(a.max_windows_per_trial),
        feature_dim=a.feature_dim,
        branch_hidden=a.branch_hidden,
        tcn_layers=a.tcn_layers,
        dropout=a.dropout,
        label_smoothing=a.label_smoothing,
        proto_weight=a.proto_weight,
        supcon_weight=a.supcon_weight,
        confusion_pair_weight=a.confusion_pair_weight,
        supcon_temperature=a.supcon_temperature,
        domain_align_weight=a.domain_align_weight,
        domain_align_warmup_epochs=a.domain_align_warmup_epochs,
        domain_align_ramp_epochs=a.domain_align_ramp_epochs,
        domain_mixup_weight=0.0,
        domain_mixup_alpha=0.40,
        tta_shifts=(0,),
        augment=not a.no_augment,
        time_shift=a.time_shift,
        noise_std=a.noise_std,
        channel_dropout_p=a.channel_dropout_p,
        amplitude_jitter=a.amplitude_jitter,
        time_mask_ratio=a.time_mask_ratio,
        robust_selection=not a.disable_robust_selection,
        robust_val_time_shift=a.robust_val_time_shift,
        robust_val_noise_std=a.robust_val_noise_std,
        robust_val_channel_dropout_p=a.robust_val_channel_dropout_p,
        robust_val_amplitude_jitter=a.robust_val_amplitude_jitter,
        robust_val_time_mask_ratio=a.robust_val_time_mask_ratio,
        mc_dropout_samples=a.mc_dropout_samples,
        router_type=a.router_type,
        router_epochs=a.router_epochs,
        router_threshold=a.router_threshold,
        router_target_mode=a.router_target_mode,
        router_positive_quantile=a.router_positive_quantile,
        router_hard_augment_copies=a.router_hard_augment_copies,
        router_hard_time_shift=a.router_hard_time_shift,
        router_hard_noise_std=a.router_hard_noise_std,
        router_hard_channel_dropout_p=a.router_hard_channel_dropout_p,
        router_positive_weight_cap=a.router_positive_weight_cap,
        router_budgeted_routing=bool(not a.disable_budgeted_routing),
        router_confidence_gate=a.router_confidence_gate,
        router_min_error_recall=a.router_min_error_recall,
        router_calibration_steps=a.router_calibration_steps,
        advisor_prob_weight=a.advisor_prob_weight,
        advisor_proto_weight=a.advisor_proto_weight,
        advisor_domain_weight=a.advisor_domain_weight,
        advisor_physics_weight=a.advisor_physics_weight,
        pairwise_enabled=not a.disable_pairwise_advisor,
        pairwise_all_pairs=False,
        evidence_consensus_enabled=bool(a.enable_evidence_consensus),
        evidence_consensus_min_pair_confidence=0.97,
        evidence_consensus_min_risk=0.50,
        evidence_consensus_max_small_confidence=0.35,
        evidence_consensus_min_score_gain=2.00,
        pairwise_epochs=a.pairwise_epochs,
        pairwise_lr=a.pairwise_lr,
        # Full-model build: preserve the requested LLM mode (local_hf by default).
        llm_mode=a.llm_mode,
        llm_model_path=a.llm_model_path,
        llm_adapter_path=a.llm_adapter_path,
        llm_dtype=a.llm_dtype,
        llm_max_input_tokens=a.llm_max_input_tokens,
        llm_max_new_tokens=a.llm_max_new_tokens,
        max_llm_calls_per_eval=a.max_llm_calls_per_eval,
        llm_max_call_rate=a.llm_max_call_rate,
        llm_batch_size=a.llm_batch_size,
        llm_revision_min_confidence=a.llm_revision_min_confidence,
        llm_revision_min_evidence_confidence=a.llm_revision_min_evidence_confidence,
        llm_revision_protect_small_confidence=a.llm_revision_protect_small_confidence,
        llm_revision_high_risk=a.llm_revision_high_risk,
        llm_revision_min_support=a.llm_revision_min_support,
        llm_revision_min_score_gain=a.llm_revision_min_score_gain,
        llm_override_min_confidence=a.llm_override_min_confidence,
        llm_override_min_evidence_confidence=a.llm_override_min_evidence_confidence,
        llm_override_max_small_confidence=a.llm_override_max_small_confidence,
        llm_override_min_support=a.llm_override_min_support,
        llm_override_min_score_gain=a.llm_override_min_score_gain,
        llm_normal_flip_min_confidence=a.llm_normal_flip_min_confidence,
        llm_delegate_min_confidence=a.llm_delegate_min_confidence,
        llm_delegate_min_score_margin=a.llm_delegate_min_score_margin,
        llm_delegate_min_override_support=a.llm_delegate_min_override_support,
        llm_delegate_max_small_confidence_for_override=a.llm_delegate_max_small_confidence_for_override,
        llm_delegate_min_risk_margin=a.llm_delegate_min_risk_margin,
        llm_delegate_min_relative_support=a.llm_delegate_min_relative_support,
        llm_delegate_min_relative_support_gain=a.llm_delegate_min_relative_support_gain,
        llm_delegate_require_exact_json=False,  # V9.1: allow answer-first fallback scores, then use evidence gate
        validation_use_llm=a.enable_validation_llm,
        eval_log_interval=a.eval_log_interval,
        advisor_mode=a.advisor_mode,
        llm_feature_token_count=a.llm_feature_token_count,
        llm_feature_token_dim=a.llm_feature_token_dim,
        token_align_weight=a.token_align_weight,
        llm_generate_maintenance=not a.disable_llm_maintenance,
        maintenance_revision_enabled=False,
        maintenance_max_new_tokens=a.maintenance_max_new_tokens,
        maintenance_batch_size=a.maintenance_batch_size,
        maintenance_stability_repeats=max(1, int(a.maintenance_stability_repeats)),
        skip_sft_export_when_reusing_lora=False,
        auto_train_lora=not a.disable_auto_lora,
        lora_output_dir=a.lora_output_dir,
        lora_epochs=a.lora_epochs,
        lora_batch_size=a.lora_batch_size,
        lora_grad_accum_steps=a.lora_grad_accum_steps,
        lora_lr=a.lora_lr,
        lora_weight_decay=a.lora_weight_decay,
        lora_warmup_ratio=a.lora_warmup_ratio,
        lora_rank=a.lora_rank,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        lora_max_seq_length=a.lora_max_seq_length,
        lora_target_modules=a.lora_target_modules,
        lora_gradient_checkpointing=not a.disable_lora_gradient_checkpointing,
        # V8.9.1 always exports fresh SFT data and retrains LoRA; the compatibility flag is ignored.
        lora_reuse_existing=False,
        sft_export_path=a.sft_export_path,
        sft_max_records=none_if_negative(a.sft_max_records),
        sft_hard_augment_copies=a.sft_hard_augment_copies,
        sft_counterfactual_revise_copies=a.sft_counterfactual_revise_copies,
        sft_min_revise_fraction=a.sft_min_revise_fraction,
        sft_min_revision_confidence=a.sft_min_revision_confidence,
        checkpoint_path=a.checkpoint_path,
        results_json_path=a.results_json_path,
        class_maintenance_csv_path=a.class_maintenance_csv_path,
        class_maintenance_text_path=a.class_maintenance_text_path,
        maintenance_rule_evaluation_csv_path=a.maintenance_rule_evaluation_csv_path,
        maintenance_rule_evaluation_json_path=a.maintenance_rule_evaluation_json_path,
        small_model_checkpoint_path=a.small_model_checkpoint,
        eval_only=a.eval_only,
        eval_test_each_epoch=a.eval_test_each_epoch,
        num_workers=a.num_workers,
        cpu_num_threads=a.cpu_num_threads,
    )


if __name__ == "__main__":
    # V10.13.1 runs the complete PSCG-LLM pipeline exactly once.
    train_dars_llm(parse_args())
