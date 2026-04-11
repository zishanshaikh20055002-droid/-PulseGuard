import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from src.fault_localization import FAULT_FEATURE_NAMES, build_fault_feature_vector


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_META_PATH = os.path.join(BASE_DIR, "models", "fault_localizer_meta.json")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if not np.isfinite(parsed):
            return float(default)
        return parsed
    except (TypeError, ValueError):
        return float(default)


def _load_baseline(meta_path: str) -> tuple[list[str], dict[str, dict[str, float]], str]:
    if not os.path.exists(meta_path):
        return list(FAULT_FEATURE_NAMES), {}, "meta_not_found"

    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        return list(FAULT_FEATURE_NAMES), {}, "meta_unreadable"

    feature_names = [str(name) for name in meta.get("feature_names", FAULT_FEATURE_NAMES)]
    feature_stats = meta.get("feature_stats", {})
    if not isinstance(feature_stats, dict) or not feature_stats:
        return feature_names, {}, "feature_stats_missing"

    baseline: dict[str, dict[str, float]] = {}
    for name in feature_names:
        blob = feature_stats.get(name, {})
        if not isinstance(blob, dict):
            continue
        baseline[name] = {
            "mean": _safe_float(blob.get("mean"), 0.0),
            "std": max(_safe_float(blob.get("std"), 1e-3), 1e-3),
        }

    if not baseline:
        return feature_names, {}, "feature_stats_invalid"

    return feature_names, baseline, "ok"


def evaluate_feature_drift(
    recent_rows: list[dict[str, Any]],
    threshold: float = 1.5,
    meta_path: str = DEFAULT_META_PATH,
) -> dict[str, Any]:
    checked_at = datetime.utcnow().isoformat() + "Z"
    feature_names, baseline, baseline_state = _load_baseline(meta_path)

    if not baseline:
        return {
            "enabled": False,
            "detected": False,
            "score": 0.0,
            "threshold": float(threshold),
            "rows_evaluated": 0,
            "feature_scores": {},
            "top_shifted_features": [],
            "baseline_state": baseline_state,
            "checked_at": checked_at,
            "reason": "No baseline profile available for drift detection",
        }

    if not recent_rows:
        return {
            "enabled": True,
            "detected": False,
            "score": 0.0,
            "threshold": float(threshold),
            "rows_evaluated": 0,
            "feature_scores": {},
            "top_shifted_features": [],
            "baseline_state": baseline_state,
            "checked_at": checked_at,
            "reason": "No recent samples available",
        }

    matrix = np.vstack([build_fault_feature_vector(row) for row in recent_rows]).astype(np.float32)

    feature_scores: dict[str, float] = {}
    details: list[dict[str, Any]] = []
    for idx, feature_name in enumerate(feature_names):
        if idx >= matrix.shape[1] or feature_name not in baseline:
            continue

        base_mean = baseline[feature_name]["mean"]
        base_std = max(baseline[feature_name]["std"], 1e-3)
        recent_mean = float(np.mean(matrix[:, idx]))
        z_shift = abs(recent_mean - base_mean) / base_std

        feature_scores[feature_name] = round(float(z_shift), 4)
        details.append(
            {
                "feature": feature_name,
                "score": round(float(z_shift), 4),
                "recent_mean": round(recent_mean, 6),
                "baseline_mean": round(base_mean, 6),
            }
        )

    if not feature_scores:
        return {
            "enabled": True,
            "detected": False,
            "score": 0.0,
            "threshold": float(threshold),
            "rows_evaluated": int(len(recent_rows)),
            "feature_scores": {},
            "top_shifted_features": [],
            "baseline_state": baseline_state,
            "checked_at": checked_at,
            "reason": "No overlapping features between baseline and recent rows",
        }

    overall_score = float(np.mean(list(feature_scores.values())))
    top_shifted = sorted(details, key=lambda item: item["score"], reverse=True)[:5]
    max_score = top_shifted[0]["score"] if top_shifted else 0.0
    drift_detected = bool(overall_score >= float(threshold) or max_score >= float(threshold) * 1.35)

    return {
        "enabled": True,
        "detected": drift_detected,
        "score": round(overall_score, 4),
        "max_feature_score": round(float(max_score), 4),
        "threshold": float(threshold),
        "rows_evaluated": int(len(recent_rows)),
        "feature_scores": feature_scores,
        "top_shifted_features": top_shifted,
        "baseline_state": baseline_state,
        "checked_at": checked_at,
        "reason": "drift_detected" if drift_detected else "within_threshold",
    }
