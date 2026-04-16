import os
from typing import Any


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


FAILURE_WARN = _env_float("ALARM_FAILURE_WARN", 0.55)
FAILURE_CRIT = _env_float("ALARM_FAILURE_CRIT", 0.80)
TTF_WARN_HOURS = _env_float("ALARM_TTF_WARN_HOURS", 12.0)
TTF_CRIT_HOURS = _env_float("ALARM_TTF_CRIT_HOURS", 6.0)
STRICT_TTF_EMERGENCY = _env_bool("ALARM_STRICT_TTF_EMERGENCY", False)


def evaluate_alarm(diagnosis: dict[str, Any]) -> dict[str, Any]:
    failure_probability = float(diagnosis.get("failure_probability", 0.0) or 0.0)
    ttf_hours = float(diagnosis.get("time_to_failure_hours", 0.0) or 0.0)
    severity = str(diagnosis.get("fault_severity", "LOW") or "LOW").upper()
    status = str(diagnosis.get("status", "HEALTHY") or "HEALTHY").upper()
    confidence = float(diagnosis.get("fault_confidence", 0.0) or 0.0)
    component = str(diagnosis.get("fault_component", "unknown") or "unknown")

    reasons = []
    if failure_probability >= FAILURE_CRIT:
        reasons.append("Failure probability crossed critical threshold")
    elif failure_probability >= FAILURE_WARN:
        reasons.append("Failure probability crossed warning threshold")

    if ttf_hours <= TTF_CRIT_HOURS:
        reasons.append("Time-to-failure is in critical window")
    elif ttf_hours <= TTF_WARN_HOURS:
        reasons.append("Time-to-failure is in warning window")

    if severity == "CRITICAL":
        reasons.append("Fault severity is critical")
    elif severity == "WARNING":
        reasons.append("Fault severity is warning")

    if confidence >= 0.8:
        reasons.append("Fault localization confidence is high")

    level = "INFO"
    priority = "P4"
    recommended_window_hours = 72.0

    critical_ttf = ttf_hours <= TTF_CRIT_HOURS
    warning_ttf = ttf_hours <= TTF_WARN_HOURS
    critical_signal = failure_probability >= FAILURE_CRIT or severity == "CRITICAL"
    warning_signal = failure_probability >= FAILURE_WARN or severity in {"WARNING", "CRITICAL"}

    if (
        critical_signal
        or (
            critical_ttf
            and (
                STRICT_TTF_EMERGENCY
                or warning_signal
                or status == "CRITICAL"
            )
        )
    ):
        level = "EMERGENCY"
        priority = "P1"
        recommended_window_hours = 2.0
    elif (
        warning_signal
        or warning_ttf
    ):
        level = "ALERT"
        priority = "P2"
        recommended_window_hours = 8.0
    elif failure_probability >= 0.35 or confidence >= 0.65:
        level = "ADVISORY"
        priority = "P3"
        recommended_window_hours = 24.0

    return {
        "alarm_level": level,
        "maintenance_priority": priority,
        "alarm_reasons": reasons,
        "recommended_window_hours": recommended_window_hours,
        "alarm_component": component,
    }
