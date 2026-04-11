import logging
import os
import threading
import time
from datetime import datetime
from typing import Any

from src.database import (
    DB_PATH,
    count_feedback_labels,
    count_training_candidates,
    create_retrain_run,
    fetch_latest_retrain_run,
    fetch_recent_feature_rows,
    mark_resolved_feedback_applied,
    update_retrain_run,
)
from src.drift_detection import evaluate_feature_drift
from src.export_fault_training_data import export_training_data
from src.train_fault_localization import train_fault_localizer


logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "data", "fault_localization_labeled.csv")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "fault_localizer.pkl")
DEFAULT_META_PATH = os.path.join(BASE_DIR, "models", "fault_localizer_meta.json")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


class AutoRetrainCoordinator:
    def __init__(self, fault_localizer, metrics: dict[str, Any] | None = None):
        self.fault_localizer = fault_localizer
        self.metrics = metrics or {}

        self.enabled = _env_bool("RETRAIN_ENABLED", True)
        self.require_drift = _env_bool("RETRAIN_REQUIRE_DRIFT", True)
        self.check_interval_seconds = max(60, int(_env_float("RETRAIN_CHECK_MINUTES", 30.0) * 60))
        self.cooldown_seconds = max(60, int(_env_float("RETRAIN_MIN_COOLDOWN_MINUTES", 120.0) * 60))
        self.min_rows = max(50, _env_int("RETRAIN_MIN_ROWS", 300))
        self.min_feedback = max(0, _env_int("RETRAIN_MIN_FEEDBACK", 20))
        self.min_confidence = max(0.0, min(1.0, _env_float("RETRAIN_MIN_CONFIDENCE", 0.25)))

        self.drift_threshold = max(0.1, _env_float("DRIFT_ZSCORE_THRESHOLD", 1.5))
        self.drift_window_rows = max(50, _env_int("DRIFT_WINDOW_ROWS", 500))

        self.dataset_path = os.getenv("RETRAIN_DATASET_PATH", DEFAULT_DATASET_PATH)
        self.model_path = os.getenv("RETRAIN_MODEL_OUT", DEFAULT_MODEL_PATH)
        self.meta_path = os.getenv("RETRAIN_META_OUT", DEFAULT_META_PATH)

        self._stop_event = threading.Event()
        self._manual_event = threading.Event()
        self._run_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None

        self._manual_reason = "manual request"
        self._manual_user = "system"
        self._last_success_epoch = 0.0
        self._status: dict[str, Any] = {
            "enabled": bool(self.enabled),
            "running": False,
            "check_interval_seconds": self.check_interval_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "min_rows": self.min_rows,
            "min_feedback": self.min_feedback,
            "min_confidence": self.min_confidence,
            "require_drift": bool(self.require_drift),
            "drift_threshold": self.drift_threshold,
            "drift_window_rows": self.drift_window_rows,
            "last_check_at": None,
            "last_decision": "not_started",
            "last_reason": "scheduler not started",
            "last_error": "",
            "last_drift": {
                "detected": False,
                "score": 0.0,
                "rows_evaluated": 0,
                "threshold": self.drift_threshold,
            },
            "pending_feedback": 0,
            "resolved_feedback_ready": 0,
            "candidate_rows": 0,
            "manual_trigger_queued": False,
            "manual_trigger_reason": "",
            "manual_trigger_requested_by": "",
            "latest_run": fetch_latest_retrain_run(),
        }

    def _update_status(self, **kwargs):
        with self._state_lock:
            self._status.update(kwargs)

    def _now_iso(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="auto-retraining-coordinator",
        )
        self._thread.start()
        logger.info("[RETRAIN] Scheduler started (enabled=%s)", self.enabled)

    def stop(self):
        self._stop_event.set()
        self._manual_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            snapshot = dict(self._status)
        snapshot["latest_run"] = fetch_latest_retrain_run()
        return snapshot

    def trigger_manual(self, requested_by: str, reason: str = "manual retraining") -> dict[str, Any]:
        self._manual_user = requested_by
        self._manual_reason = reason.strip()[:180] if reason else "manual retraining"
        self._manual_event.set()
        self._update_status(
            manual_trigger_queued=True,
            manual_trigger_reason=self._manual_reason,
            manual_trigger_requested_by=requested_by,
            last_decision="queued_manual",
            last_reason=f"Queued manual retraining by {requested_by}",
        )
        return self.status()

    def _run_loop(self):
        self._run_cycle(trigger_type="scheduled", trigger_reason="startup_check", requested_by="system")

        while not self._stop_event.is_set():
            manual_triggered = self._manual_event.wait(timeout=self.check_interval_seconds)
            if self._stop_event.is_set():
                break

            if manual_triggered:
                self._manual_event.clear()
                self._run_cycle(
                    trigger_type="manual",
                    trigger_reason=self._manual_reason,
                    requested_by=self._manual_user,
                )
                self._update_status(
                    manual_trigger_queued=False,
                    manual_trigger_reason="",
                    manual_trigger_requested_by="",
                )
            else:
                self._run_cycle(
                    trigger_type="scheduled",
                    trigger_reason="periodic_check",
                    requested_by="system",
                )

    def _record_drift_metrics(self, drift_report: dict[str, Any]):
        score = float(drift_report.get("score", 0.0))
        detected = 1.0 if drift_report.get("detected") else 0.0
        rows = float(drift_report.get("rows_evaluated", 0))

        if "drift_score_gauge" in self.metrics:
            self.metrics["drift_score_gauge"].set(score)
        if "drift_detected_flag" in self.metrics:
            self.metrics["drift_detected_flag"].set(detected)
        if "drift_rows_gauge" in self.metrics:
            self.metrics["drift_rows_gauge"].set(rows)

    def _record_feedback_metrics(self, pending_feedback: int, resolved_feedback: int):
        if "feedback_pending_gauge" in self.metrics:
            self.metrics["feedback_pending_gauge"].set(float(pending_feedback))
        if "feedback_ready_gauge" in self.metrics:
            self.metrics["feedback_ready_gauge"].set(float(resolved_feedback))

    def _record_retrain_counter(self, status: str, trigger_type: str):
        if "auto_retrain_runs" in self.metrics:
            self.metrics["auto_retrain_runs"].labels(status=status, trigger=trigger_type).inc()

    def _run_cycle(self, trigger_type: str, trigger_reason: str, requested_by: str):
        if not self._run_lock.acquire(blocking=False):
            self._update_status(
                last_check_at=self._now_iso(),
                last_decision="skipped",
                last_reason="A retraining cycle is already running",
            )
            return

        run_id: int | None = None
        start_epoch = time.time()
        self._update_status(running=True, last_check_at=self._now_iso(), last_error="")

        try:
            recent_rows = fetch_recent_feature_rows(limit=self.drift_window_rows)
            drift_report = evaluate_feature_drift(
                recent_rows=recent_rows,
                threshold=self.drift_threshold,
                meta_path=self.meta_path,
            )
            self._record_drift_metrics(drift_report)

            pending_feedback = count_feedback_labels(resolved=False)
            resolved_feedback_ready = count_feedback_labels(only_unapplied_resolved=True)
            candidate_rows = count_training_candidates(min_confidence=self.min_confidence)
            self._record_feedback_metrics(pending_feedback, resolved_feedback_ready)

            cooldown_remaining = 0
            if self._last_success_epoch > 0:
                cooldown_remaining = max(0, int(self.cooldown_seconds - (time.time() - self._last_success_epoch)))

            manual = trigger_type == "manual"
            feedback_gate = self.min_feedback > 0 and resolved_feedback_ready >= self.min_feedback
            drift_gate = bool(drift_report.get("detected", False))

            reasons: list[str] = []
            should_train = False

            if manual:
                should_train = True
                reasons.append(f"manual trigger by {requested_by}")
                if trigger_reason:
                    reasons.append(trigger_reason)
            else:
                if not self.enabled:
                    reasons.append("scheduler disabled")
                if candidate_rows < self.min_rows:
                    reasons.append(f"candidate_rows {candidate_rows} < min_rows {self.min_rows}")
                if cooldown_remaining > 0:
                    reasons.append(f"cooldown_active {cooldown_remaining}s")

                if self.require_drift:
                    if drift_gate:
                        reasons.append("drift threshold exceeded")
                    elif feedback_gate:
                        reasons.append("feedback threshold reached (drift bypass)")
                    else:
                        reasons.append("drift not detected")
                elif drift_gate:
                    reasons.append("drift threshold exceeded")

                if self.enabled and candidate_rows >= self.min_rows and cooldown_remaining <= 0:
                    if self.require_drift:
                        should_train = drift_gate or feedback_gate
                    else:
                        should_train = drift_gate or feedback_gate

            self._update_status(
                pending_feedback=pending_feedback,
                resolved_feedback_ready=resolved_feedback_ready,
                candidate_rows=candidate_rows,
                last_drift={
                    "detected": bool(drift_report.get("detected", False)),
                    "score": float(drift_report.get("score", 0.0)),
                    "rows_evaluated": int(drift_report.get("rows_evaluated", 0)),
                    "threshold": float(drift_report.get("threshold", self.drift_threshold)),
                    "top_shifted_features": drift_report.get("top_shifted_features", []),
                    "reason": drift_report.get("reason", ""),
                },
            )

            if not should_train:
                self._update_status(
                    last_decision="skipped",
                    last_reason="; ".join(reasons[:4]) if reasons else "gates_not_met",
                )
                return

            run_id = create_retrain_run(
                trigger_type=trigger_type,
                drift_score=float(drift_report.get("score", 0.0)),
                drift_detected=bool(drift_report.get("detected", False)),
                feedback_samples=int(resolved_feedback_ready),
                dataset_rows=int(candidate_rows),
                message="; ".join(reasons[:4]) if reasons else "training started",
            )

            exported_rows = export_training_data(
                db_path=DB_PATH,
                output_csv=self.dataset_path,
                min_confidence=self.min_confidence,
                limit=0,
                include_resolved_feedback=True,
            )

            train_metrics = train_fault_localizer(
                input_csv=self.dataset_path,
                model_out=self.model_path,
                meta_out=self.meta_path,
                version=None,
                test_size=0.2,
                random_state=42,
            )

            self.fault_localizer.reload()
            feedback_applied = mark_resolved_feedback_applied(run_id)

            run_metrics = {
                "training": train_metrics,
                "drift": drift_report,
                "feedback_applied": int(feedback_applied),
                "candidate_rows_at_start": int(candidate_rows),
                "exported_rows": int(exported_rows),
                "duration_seconds": round(time.time() - start_epoch, 3),
            }

            update_retrain_run(
                run_id=run_id,
                status="succeeded",
                model_version=str(train_metrics.get("version", "unknown")),
                message=f"retraining complete ({trigger_type})",
                metrics=run_metrics,
                dataset_rows=int(train_metrics.get("rows", exported_rows)),
                feedback_samples=int(resolved_feedback_ready),
                drift_score=float(drift_report.get("score", 0.0)),
                drift_detected=bool(drift_report.get("detected", False)),
            )

            self._last_success_epoch = time.time()
            self._record_retrain_counter(status="succeeded", trigger_type=trigger_type)

            self._update_status(
                last_decision="trained",
                last_reason=f"completed in {run_metrics['duration_seconds']}s",
                latest_run=fetch_latest_retrain_run(),
            )
        except Exception as exc:
            error_message = str(exc)
            if run_id is not None:
                update_retrain_run(
                    run_id=run_id,
                    status="failed",
                    model_version="",
                    message=error_message[:300],
                    metrics={"error": error_message},
                )
            self._record_retrain_counter(status="failed", trigger_type=trigger_type)
            self._update_status(
                last_decision="failed",
                last_reason=f"{trigger_type} cycle failed",
                last_error=error_message,
                latest_run=fetch_latest_retrain_run(),
            )
            logger.error("[RETRAIN] cycle failed: %s", exc)
        finally:
            self._update_status(running=False)
            self._run_lock.release()
