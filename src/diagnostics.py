import math
import os
from typing import Any

import numpy as np


RUL_CYCLE_MINUTES = float(os.getenv("RUL_CYCLE_MINUTES", "2.0"))


FAULT_PLAYBOOK = {
    "stator": {
        "fault_type": "stator_insulation_degradation",
        "probable_causes": [
            "Over-temperature stress in windings",
            "Cooling airflow restriction",
            "Voltage stress and harmonics",
        ],
        "recommended_actions": [
            "Run insulation resistance and surge comparison tests",
            "Inspect fan ducts and heat exchanger path",
            "Reduce load peaks until winding temperature stabilizes",
        ],
    },
    "rotor": {
        "fault_type": "rotor_bar_or_eccentricity_issue",
        "probable_causes": [
            "Rotor bar crack initiation",
            "Rotor-stator eccentricity",
            "Mechanical imbalance at high speed",
        ],
        "recommended_actions": [
            "Perform rotor current signature analysis",
            "Check shaft alignment and runout",
            "Inspect coupling and balance condition",
        ],
    },
    "bearing": {
        "fault_type": "bearing_fatigue_progression",
        "probable_causes": [
            "Insufficient lubrication film",
            "Contamination in bearing races",
            "Sustained vibration and shock loading",
        ],
        "recommended_actions": [
            "Schedule vibration spectrum inspection",
            "Relubricate and sample lubricant",
            "Plan bearing replacement window",
        ],
    },
    "cooling": {
        "fault_type": "cooling_path_inefficiency",
        "probable_causes": [
            "Blocked cooling passages",
            "Reduced coolant or air flow",
            "Heat exchanger fouling",
        ],
        "recommended_actions": [
            "Clean cooling channels and filters",
            "Verify fan or pump operating point",
            "Check ambient thermal load and enclosure ventilation",
        ],
    },
    "power_supply": {
        "fault_type": "electrical_supply_instability",
        "probable_causes": [
            "Voltage sag or imbalance",
            "Excessive current draw",
            "Power quality distortion",
        ],
        "recommended_actions": [
            "Measure line-to-line voltage balance",
            "Inspect drive and protection settings",
            "Analyze harmonic distortion and grounding",
        ],
    },
    "lubrication": {
        "fault_type": "lubrication_system_degradation",
        "probable_causes": [
            "Oil starvation",
            "Viscosity breakdown",
            "Lubricant contamination",
        ],
        "recommended_actions": [
            "Check lubrication pump and lines",
            "Replace oil or grease with spec-compliant grade",
            "Add particle and moisture monitoring",
        ],
    },
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp((value - lo) / (hi - lo), 0.0, 1.0)


def _severity(score: float, status: str) -> str:
    if status == "CRITICAL" or score >= 0.8:
        return "CRITICAL"
    if status == "WARNING" or score >= 0.55:
        return "WARNING"
    return "LOW"


def _float_at(raw_features: np.ndarray, idx: int, fallback: float = 0.0) -> float:
    try:
        return float(raw_features[idx])
    except Exception:
        return float(fallback)


def build_realtime_diagnosis(
    machine_id: str,
    step: int,
    prediction: float,
    rul_std: float,
    status: str,
    stage_probs: list[float],
    ui_sensors: dict[str, float],
    raw_features: np.ndarray,
) -> dict[str, Any]:
    process_temp_k = float(ui_sensors.get("temperature", 0.0))
    ambient_temp_k = float(ui_sensors.get("air_temperature", 0.0))
    torque_nm = float(ui_sensors.get("torque", 0.0))
    tool_wear_min = float(ui_sensors.get("tool_wear", 0.0))
    speed_rpm = float(ui_sensors.get("speed", 0.0))
    delta_temp_k = max(0.0, process_temp_k - ambient_temp_k)

    s13 = _float_at(raw_features, 8, 2390.0)
    s15 = _float_at(raw_features, 10, 8.4)
    voltage_v = 360.0 + _norm(s13, 2385.0, 2395.0) * 80.0
    vibration_rms = 1.5 + _norm(s15, 8.2, 8.7) * 10.5

    mech_power_w = max(0.0, torque_nm * (2.0 * math.pi * speed_rpm / 60.0))
    thermal_penalty = 0.25 * _norm(delta_temp_k, 10.0, 75.0)
    wear_penalty = 0.20 * _norm(tool_wear_min, 0.0, 300.0)
    vib_penalty = 0.15 * _norm(vibration_rms, 1.0, 12.0)
    efficiency = _clamp(0.95 - thermal_penalty - wear_penalty - vib_penalty, 0.55, 0.98)
    electrical_power_w = mech_power_w / max(efficiency * 0.92, 0.3)
    power_kw = electrical_power_w / 1000.0
    current_a = electrical_power_w / max(math.sqrt(3.0) * voltage_v * 0.88, 1e-6)

    failure_base = 1.0 - _clamp(prediction / 220.0, 0.0, 1.0)
    uncertainty_boost = 0.30 * _norm(rul_std, 0.0, 25.0)
    status_boost = 0.25 if status == "CRITICAL" else 0.12 if status == "WARNING" else 0.0
    failure_probability = _clamp(failure_base + uncertainty_boost + status_boost, 0.0, 1.0)
    health_index = 100.0 * (1.0 - failure_probability)
    time_to_failure_hours = max(0.0, float(prediction) * RUL_CYCLE_MINUTES / 60.0)

    component_scores = {
        "stator": _clamp(
            0.38 * _norm(process_temp_k, 300.0, 385.0)
            + 0.25 * _norm(current_a, 80.0, 260.0)
            + 0.17 * _norm(vibration_rms, 2.0, 12.0)
            + 0.20 * failure_probability,
            0.0,
            1.0,
        ),
        "rotor": _clamp(
            0.32 * _norm(speed_rpm, 1700.0, 3200.0)
            + 0.28 * _norm(vibration_rms, 2.0, 12.0)
            + 0.20 * _norm(tool_wear_min, 50.0, 300.0)
            + 0.20 * failure_probability,
            0.0,
            1.0,
        ),
        "bearing": _clamp(
            0.42 * _norm(vibration_rms, 2.0, 14.0)
            + 0.26 * _norm(tool_wear_min, 50.0, 300.0)
            + 0.14 * _norm(delta_temp_k, 10.0, 80.0)
            + 0.18 * failure_probability,
            0.0,
            1.0,
        ),
        "cooling": _clamp(
            0.52 * _norm(delta_temp_k, 10.0, 80.0)
            + 0.23 * _norm(process_temp_k, 300.0, 390.0)
            + 0.25 * failure_probability,
            0.0,
            1.0,
        ),
        "power_supply": _clamp(
            0.45 * _norm(abs(voltage_v - 400.0), 0.0, 70.0)
            + 0.35 * _norm(current_a, 80.0, 280.0)
            + 0.20 * failure_probability,
            0.0,
            1.0,
        ),
        "lubrication": _clamp(
            0.40 * _norm(tool_wear_min, 60.0, 300.0)
            + 0.30 * _norm(vibration_rms, 2.0, 14.0)
            + 0.30 * failure_probability,
            0.0,
            1.0,
        ),
    }

    dominant_component = max(component_scores, key=component_scores.get)
    dominant_score = float(component_scores[dominant_component])
    playbook = FAULT_PLAYBOOK[dominant_component]

    probs = np.array(stage_probs if isinstance(stage_probs, list) else [], dtype=np.float32)
    probs = probs[np.isfinite(probs)] if probs.size else np.array([], dtype=np.float32)
    if probs.size >= 2:
        probs = np.sort(probs)
        margin = float(probs[-1] - probs[-2])
    else:
        margin = 0.15

    fault_confidence = _clamp(
        0.48 + 0.38 * dominant_score + 0.18 * margin - 0.20 * _norm(rul_std, 0.0, 30.0),
        0.10,
        0.99,
    )

    component_health = {
        comp: round(100.0 * (1.0 - score), 1)
        for comp, score in component_scores.items()
    }

    return {
        "voltage": round(voltage_v, 2),
        "current": round(current_a, 2),
        "power_kw": round(power_kw, 2),
        "vibration": round(vibration_rms, 3),
        "efficiency": round(efficiency * 100.0, 2),
        "health_index": round(health_index, 2),
        "failure_probability": round(failure_probability, 4),
        "time_to_failure_hours": round(time_to_failure_hours, 2),
        "fault_component": dominant_component,
        "fault_type": playbook["fault_type"],
        "fault_severity": _severity(dominant_score, status),
        "fault_confidence": round(fault_confidence, 4),
        "probable_causes": playbook["probable_causes"],
        "recommended_actions": playbook["recommended_actions"],
        "component_scores": {
            k: round(v, 4) for k, v in component_scores.items()
        },
        "component_health": component_health,
        "diagnosis_version": "v1.0-rule-fusion",
        "diagnosis_generated": {
            "machine_id": machine_id,
            "step": int(step),
        },
    }