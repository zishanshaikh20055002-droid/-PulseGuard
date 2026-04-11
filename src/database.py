import sqlite3
import os
import json
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "machine_data.db")
DB_TIMEOUT_SECONDS = 10

BASE_COLUMNS = {
    "id",
    "timestamp",
    "machine_id",
    "RUL",
    "RUL_std",
    "status",
    "temperature",
    "air_temperature",
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
    "fault_component",
    "fault_type",
    "fault_severity",
    "fault_confidence",
    "diagnostics_json",
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT_SECONDS)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column_name: str, column_type: str):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    if column_name not in existing:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_type}")


def _ensure_feedback_table(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            machine_id TEXT NOT NULL,
            prediction_id INTEGER,
            predicted_component TEXT,
            corrected_component TEXT NOT NULL,
            reviewer TEXT,
            notes TEXT,
            resolved INTEGER DEFAULT 0,
            resolved_by TEXT,
            resolved_at DATETIME,
            used_in_training_run_id INTEGER,
            metadata_json TEXT,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id)
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_machine_ts ON feedback_labels(machine_id, timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_prediction ON feedback_labels(prediction_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_resolved ON feedback_labels(resolved, used_in_training_run_id)"
    )


def _ensure_retrain_runs_table(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS retrain_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            ended_at DATETIME,
            status TEXT NOT NULL,
            trigger_type TEXT,
            drift_score REAL,
            drift_detected INTEGER,
            feedback_samples INTEGER,
            dataset_rows INTEGER,
            model_version TEXT,
            message TEXT,
            metrics_json TEXT
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_retrain_runs_started ON retrain_runs(started_at DESC)"
    )


def _json_dumps(data: Any) -> str:
    try:
        return json.dumps(data if data is not None else {}, ensure_ascii=True)
    except Exception:
        return "{}"


def _json_loads(data: str | None, fallback: Any):
    if not data:
        return fallback
    try:
        parsed = json.loads(data)
        return parsed
    except Exception:
        return fallback

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _connect() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            machine_id TEXT,
            RUL REAL,
            RUL_std REAL,
            status TEXT,
            temperature REAL,
            air_temperature REAL,
            torque REAL,
            tool_wear REAL,
            speed REAL,
            voltage REAL,
            current REAL,
            power_kw REAL,
            vibration REAL,
            efficiency REAL,
            health_index REAL,
            failure_probability REAL,
            time_to_failure_hours REAL,
            fault_component TEXT,
            fault_type TEXT,
            fault_severity TEXT,
            fault_confidence REAL,
            diagnostics_json TEXT
        )
        """)

        _ensure_column(conn, "predictions", "voltage", "REAL")
        _ensure_column(conn, "predictions", "RUL_std", "REAL")
        _ensure_column(conn, "predictions", "current", "REAL")
        _ensure_column(conn, "predictions", "power_kw", "REAL")
        _ensure_column(conn, "predictions", "vibration", "REAL")
        _ensure_column(conn, "predictions", "efficiency", "REAL")
        _ensure_column(conn, "predictions", "health_index", "REAL")
        _ensure_column(conn, "predictions", "failure_probability", "REAL")
        _ensure_column(conn, "predictions", "time_to_failure_hours", "REAL")
        _ensure_column(conn, "predictions", "fault_component", "TEXT")
        _ensure_column(conn, "predictions", "fault_type", "TEXT")
        _ensure_column(conn, "predictions", "fault_severity", "TEXT")
        _ensure_column(conn, "predictions", "fault_confidence", "REAL")
        _ensure_column(conn, "predictions", "diagnostics_json", "TEXT")

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_machine_ts ON predictions(machine_id, timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(timestamp)"
        )

        _ensure_feedback_table(conn)
        _ensure_retrain_runs_table(conn)

        conn.commit()


def _to_record(columns, row):
    record = dict(zip(columns, row))
    blob = record.pop("diagnostics_json", None)
    if blob:
        try:
            payload = json.loads(blob)
            if isinstance(payload, dict):
                record.update(payload)
        except Exception:
            pass
    return record


def _to_feedback_record(columns, row):
    record = dict(zip(columns, row))
    record["resolved"] = bool(record.get("resolved", 0))
    record["metadata"] = _json_loads(record.pop("metadata_json", None), {})
    return record


def _to_retrain_record(columns, row):
    record = dict(zip(columns, row))
    record["drift_detected"] = bool(record.get("drift_detected", 0))
    record["metrics"] = _json_loads(record.pop("metrics_json", None), {})
    return record

def insert_data(data):
    diagnostics_blob = _json_dumps({
        "probable_causes": data.get("probable_causes", []),
        "recommended_actions": data.get("recommended_actions", []),
        "component_scores": data.get("component_scores", {}),
        "component_health": data.get("component_health", {}),
        "diagnosis_version": data.get("diagnosis_version", "v1"),
        "fault_model_source": data.get("fault_model_source", "rules"),
        "fault_model_version": data.get("fault_model_version", "rules-only"),
        "fault_component_probabilities": data.get("fault_component_probabilities", {}),
        "alarm_level": data.get("alarm_level", "INFO"),
        "maintenance_priority": data.get("maintenance_priority", "P4"),
        "alarm_reasons": data.get("alarm_reasons", []),
        "recommended_window_hours": data.get("recommended_window_hours", 72.0),
    })

    with _connect() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO predictions (
            machine_id, RUL, RUL_std, status,
            temperature, air_temperature,
            torque, tool_wear, speed,
            voltage, current, power_kw, vibration, efficiency,
            health_index, failure_probability, time_to_failure_hours,
            fault_component, fault_type, fault_severity, fault_confidence,
            diagnostics_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("machine_id", "M1"),
            data.get("RUL", 0.0),
            data.get("RUL_std", 0.0),
            data.get("status", "UNKNOWN"),
            data.get("temperature", 0.0),
            data.get("air_temperature", 0.0),
            data.get("torque", 0.0),
            data.get("tool_wear", 0.0),
            data.get("speed", 0.0),
            data.get("voltage", 0.0),
            data.get("current", 0.0),
            data.get("power_kw", 0.0),
            data.get("vibration", 0.0),
            data.get("efficiency", 0.0),
            data.get("health_index", 0.0),
            data.get("failure_probability", 0.0),
            data.get("time_to_failure_hours", 0.0),
            data.get("fault_component", "unknown"),
            data.get("fault_type", "unknown"),
            data.get("fault_severity", "LOW"),
            data.get("fault_confidence", 0.0),
            diagnostics_blob,
        ))

        conn.commit()


def fetch_latest_diagnosis(machine_id: str):
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM predictions
            WHERE machine_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (machine_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cursor.description]
        return _to_record(columns, row)


def fetch_recent_diagnosis(machine_id: str, limit: int = 100):
    cap = max(1, min(int(limit), 500))
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM predictions
            WHERE machine_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (machine_id, cap),
        )
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return [_to_record(columns, row) for row in rows]


def fetch_fleet_overview(limit: int = 100):
    cap = max(1, min(int(limit), 500))
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.*
            FROM predictions p
            INNER JOIN (
                SELECT machine_id, MAX(id) AS max_id
                FROM predictions
                GROUP BY machine_id
            ) latest
              ON latest.machine_id = p.machine_id
             AND latest.max_id = p.id
            ORDER BY p.id DESC
            LIMIT ?
            """,
            (cap,),
        )
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return [_to_record(columns, row) for row in rows]


def fetch_machine_ids(limit: int = 500):
    cap = max(1, min(int(limit), 1000))
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT machine_id
            FROM predictions
            WHERE machine_id IS NOT NULL AND machine_id != ''
            ORDER BY machine_id ASC
            LIMIT ?
            """,
            (cap,),
        )
        rows = cursor.fetchall()
        return [str(row[0]) for row in rows]


def fetch_prediction_by_id(prediction_id: int):
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM predictions
            WHERE id = ?
            LIMIT 1
            """,
            (int(prediction_id),),
        )
        row = cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cursor.description]
        return _to_record(columns, row)


def fetch_latest_prediction(machine_id: str):
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM predictions
            WHERE machine_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (machine_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cursor.description]
        return _to_record(columns, row)


def insert_feedback_label(
    machine_id: str,
    corrected_component: str,
    reviewer: str,
    notes: str = "",
    prediction_id: int | None = None,
    predicted_component: str | None = None,
    metadata: dict[str, Any] | None = None,
    resolved: bool = False,
):
    with _connect() as conn:
        cursor = conn.cursor()

        resolved_prediction_id = prediction_id
        resolved_predicted_component = predicted_component

        if resolved_prediction_id is None:
            cursor.execute(
                """
                SELECT id, fault_component
                FROM predictions
                WHERE machine_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (machine_id,),
            )
            latest = cursor.fetchone()
            if latest:
                resolved_prediction_id = int(latest[0])
                if not resolved_predicted_component:
                    resolved_predicted_component = str(latest[1] or "unknown")

        if resolved_prediction_id is not None and not resolved_predicted_component:
            cursor.execute(
                "SELECT fault_component FROM predictions WHERE id = ?",
                (int(resolved_prediction_id),),
            )
            ref = cursor.fetchone()
            if ref:
                resolved_predicted_component = str(ref[0] or "unknown")

        cursor.execute(
            """
            INSERT INTO feedback_labels (
                machine_id,
                prediction_id,
                predicted_component,
                corrected_component,
                reviewer,
                notes,
                resolved,
                resolved_by,
                resolved_at,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                machine_id,
                resolved_prediction_id,
                resolved_predicted_component or "unknown",
                corrected_component,
                reviewer,
                notes,
                1 if resolved else 0,
                reviewer if resolved else None,
                None,
                _json_dumps(metadata or {}),
            ),
        )
        feedback_id = int(cursor.lastrowid)

        if resolved:
            cursor.execute(
                """
                UPDATE feedback_labels
                SET resolved_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (feedback_id,),
            )

        conn.commit()

    return fetch_feedback_label_by_id(feedback_id)


def fetch_feedback_label_by_id(feedback_id: int):
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM feedback_labels
            WHERE id = ?
            LIMIT 1
            """,
            (int(feedback_id),),
        )
        row = cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cursor.description]
        return _to_feedback_record(columns, row)


def fetch_feedback_labels(limit: int = 100, machine_id: str | None = None, resolved: bool | None = None):
    cap = max(1, min(int(limit), 500))
    query = """
        SELECT *
        FROM feedback_labels
        WHERE 1 = 1
    """
    params: list[Any] = []

    if machine_id:
        query += " AND machine_id = ?"
        params.append(machine_id)
    if resolved is not None:
        query += " AND resolved = ?"
        params.append(1 if resolved else 0)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(cap)

    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return [_to_feedback_record(columns, row) for row in rows]


def resolve_feedback_label(feedback_id: int, resolved_by: str):
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE feedback_labels
            SET resolved = 1,
                resolved_by = ?,
                resolved_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (resolved_by, int(feedback_id)),
        )
        changed = cursor.rowcount
        conn.commit()

    if changed <= 0:
        return None
    return fetch_feedback_label_by_id(feedback_id)


def count_feedback_labels(
    resolved: bool | None = None,
    only_unapplied_resolved: bool = False,
) -> int:
    query = "SELECT COUNT(*) FROM feedback_labels WHERE 1 = 1"
    params: list[Any] = []

    if resolved is not None:
        query += " AND resolved = ?"
        params.append(1 if resolved else 0)
    if only_unapplied_resolved:
        query += " AND resolved = 1 AND used_in_training_run_id IS NULL"

    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        row = cursor.fetchone()
        return int(row[0]) if row else 0


def mark_resolved_feedback_applied(run_id: int) -> int:
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE feedback_labels
            SET used_in_training_run_id = ?
            WHERE resolved = 1
              AND used_in_training_run_id IS NULL
            """,
            (int(run_id),),
        )
        changed = int(cursor.rowcount)
        conn.commit()
        return changed


def create_retrain_run(
    trigger_type: str,
    drift_score: float,
    drift_detected: bool,
    feedback_samples: int,
    dataset_rows: int,
    message: str,
):
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO retrain_runs (
                status,
                trigger_type,
                drift_score,
                drift_detected,
                feedback_samples,
                dataset_rows,
                message
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "running",
                trigger_type,
                float(drift_score),
                1 if drift_detected else 0,
                int(feedback_samples),
                int(dataset_rows),
                message,
            ),
        )
        run_id = int(cursor.lastrowid)
        conn.commit()
        return run_id


def update_retrain_run(
    run_id: int,
    status: str,
    model_version: str = "",
    message: str = "",
    metrics: dict[str, Any] | None = None,
    dataset_rows: int | None = None,
    feedback_samples: int | None = None,
    drift_score: float | None = None,
    drift_detected: bool | None = None,
):
    updates = [
        "status = ?",
        "model_version = ?",
        "message = ?",
        "metrics_json = ?",
        "ended_at = CURRENT_TIMESTAMP",
    ]
    params: list[Any] = [
        status,
        model_version,
        message,
        _json_dumps(metrics or {}),
    ]

    if dataset_rows is not None:
        updates.append("dataset_rows = ?")
        params.append(int(dataset_rows))
    if feedback_samples is not None:
        updates.append("feedback_samples = ?")
        params.append(int(feedback_samples))
    if drift_score is not None:
        updates.append("drift_score = ?")
        params.append(float(drift_score))
    if drift_detected is not None:
        updates.append("drift_detected = ?")
        params.append(1 if drift_detected else 0)

    params.append(int(run_id))

    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE retrain_runs SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )
        conn.commit()

    return fetch_retrain_run_by_id(run_id)


def fetch_retrain_run_by_id(run_id: int):
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM retrain_runs WHERE id = ? LIMIT 1",
            (int(run_id),),
        )
        row = cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cursor.description]
        return _to_retrain_record(columns, row)


def fetch_latest_retrain_run():
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cursor.description]
        return _to_retrain_record(columns, row)


def fetch_recent_feature_rows(limit: int = 500, machine_id: str | None = None):
    cap = max(20, min(int(limit), 5000))
    query = """
        SELECT
            machine_id,
            RUL,
            RUL_std,
            status,
            temperature,
            air_temperature,
            torque,
            tool_wear,
            speed,
            voltage,
            current,
            power_kw,
            vibration,
            efficiency,
            health_index,
            failure_probability,
            time_to_failure_hours,
            diagnostics_json
        FROM predictions
        WHERE 1 = 1
    """
    params: list[Any] = []
    if machine_id:
        query += " AND machine_id = ?"
        params.append(machine_id)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(cap)

    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        parsed = [_to_record(columns, row) for row in rows]
        parsed.reverse()
        return parsed


def count_training_candidates(min_confidence: float = 0.0) -> int:
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE fault_component IS NOT NULL
              AND fault_component != ''
              AND fault_confidence >= ?
            """,
            (float(min_confidence),),
        )
        row = cursor.fetchone()
        return int(row[0]) if row else 0