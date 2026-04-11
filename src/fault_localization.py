import json
import logging
import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np


logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "fault_localizer.pkl")
DEFAULT_META_PATH = os.path.join(BASE_DIR, "models", "fault_localizer_meta.json")


FAULT_FEATURE_NAMES = [
    "temperature",
    "air_temperature",
    "delta_temperature",
    "torque",
    "tool_wear",
    "speed",
    "voltage",
    "current",
    "power_kw",
    "vibration",
    "efficiency",
    "health_index",
    "failure_probability",
    "time_to_failure_hours",
    "RUL",
    "RUL_std",
    "stage_prob_healthy",
    "stage_prob_warning",
    "stage_prob_critical",
    "power_per_current",
]


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if not np.isfinite(parsed):
            return float(default)
        return parsed
    except (TypeError, ValueError):
        return float(default)


def build_fault_feature_vector(payload: dict[str, Any]) -> np.ndarray:
    stage_probs = payload.get("stage_probs", [0.0, 0.0, 0.0])
    if not isinstance(stage_probs, (list, tuple)):
        stage_probs = [0.0, 0.0, 0.0]

    probs = [
        _float_or_default(stage_probs[i], 0.0) if i < len(stage_probs) else 0.0
        for i in range(3)
    ]
    total = sum(max(0.0, p) for p in probs)
    if total > 0:
        probs = [max(0.0, p) / total for p in probs]

    temperature = _float_or_default(payload.get("temperature"), 0.0)
    air_temperature = _float_or_default(payload.get("air_temperature"), 0.0)
    current = _float_or_default(payload.get("current"), 0.0)
    power_kw = _float_or_default(payload.get("power_kw"), 0.0)

    feature_values = [
        temperature,
        air_temperature,
        max(0.0, temperature - air_temperature),
        _float_or_default(payload.get("torque"), 0.0),
        _float_or_default(payload.get("tool_wear"), 0.0),
        _float_or_default(payload.get("speed"), 0.0),
        _float_or_default(payload.get("voltage"), 0.0),
        current,
        power_kw,
        _float_or_default(payload.get("vibration"), 0.0),
        _float_or_default(payload.get("efficiency"), 0.0),
        _float_or_default(payload.get("health_index"), 0.0),
        _float_or_default(payload.get("failure_probability"), 0.0),
        _float_or_default(payload.get("time_to_failure_hours"), 0.0),
        _float_or_default(payload.get("RUL"), 0.0),
        _float_or_default(payload.get("RUL_std"), 0.0),
        probs[0],
        probs[1],
        probs[2],
        power_kw / max(current, 1e-6),
    ]
    return np.nan_to_num(np.asarray(feature_values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


class FaultLocalizer:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, meta_path: str = DEFAULT_META_PATH):
        self.model_path = model_path
        self.meta_path = meta_path
        self.model = None
        self.classes: list[str] = []
        self.version = "rules-only"
        self.trained_at = ""
        self.feature_names = list(FAULT_FEATURE_NAMES)
        self.ready = False
        self.reload()

    def reload(self):
        self.model = None
        self.classes = []
        self.version = "rules-only"
        self.trained_at = ""
        self.feature_names = list(FAULT_FEATURE_NAMES)
        self.ready = False

        if not os.path.exists(self.model_path):
            logger.info("[FAULT-ML] Model file not found, using rule-based localization")
            return

        try:
            artifact = joblib.load(self.model_path)
            if isinstance(artifact, dict):
                self.model = artifact.get("model")
                self.classes = [str(c) for c in artifact.get("classes", [])]
                if artifact.get("feature_names"):
                    self.feature_names = [str(x) for x in artifact["feature_names"]]
                self.version = str(artifact.get("version", "unknown"))
                self.trained_at = str(artifact.get("trained_at", ""))
            else:
                self.model = artifact
                if hasattr(self.model, "classes_"):
                    self.classes = [str(c) for c in getattr(self.model, "classes_", [])]
                self.version = "unknown"

            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                    self.version = str(meta.get("version", self.version))
                    self.trained_at = str(meta.get("trained_at", self.trained_at))

            self.ready = self.model is not None and hasattr(self.model, "predict_proba")
            if self.ready:
                logger.info(
                    "[FAULT-ML] Loaded model version=%s classes=%s",
                    self.version,
                    ",".join(self.classes) if self.classes else "unknown",
                )
        except Exception as exc:
            logger.error("[FAULT-ML] Failed to load model: %s", exc)
            self.ready = False

    def info(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.ready),
            "model_path": self.model_path,
            "meta_path": self.meta_path,
            "version": self.version,
            "trained_at": self.trained_at,
            "feature_names": self.feature_names,
            "classes": self.classes,
            "fallback": "rule-based localization",
            "checked_at": datetime.utcnow().isoformat() + "Z",
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.ready or self.model is None:
            return None

        try:
            x = build_fault_feature_vector(payload).reshape(1, -1)
            probs = self.model.predict_proba(x)[0]
            classes = self.classes or [str(c) for c in getattr(self.model, "classes_", [])]
            if not classes:
                return None

            idx = int(np.argmax(probs))
            component = str(classes[idx])
            confidence = float(probs[idx])
            component_probs = {
                str(classes[i]): float(probs[i]) for i in range(min(len(classes), len(probs)))
            }

            return {
                "fault_component": component,
                "fault_confidence": round(confidence, 4),
                "fault_model_source": "ml",
                "fault_model_version": self.version,
                "fault_component_probabilities": {
                    k: round(v, 4) for k, v in component_probs.items()
                },
            }
        except Exception as exc:
            logger.error("[FAULT-ML] Inference failure: %s", exc)
            return None
