import argparse
import os
import sqlite3

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(BASE_DIR, "data", "machine_data.db")
DEFAULT_OUT = os.path.join(BASE_DIR, "data", "fault_localization_labeled.csv")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    )
    return cursor.fetchone() is not None


def export_training_data(
    db_path: str,
    output_csv: str,
    min_confidence: float = 0.0,
    limit: int = 0,
    include_resolved_feedback: bool = True,
):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        use_feedback = include_resolved_feedback and _table_exists(conn, "feedback_labels")

        if use_feedback:
            query = """
            WITH latest_feedback AS (
                SELECT fb.prediction_id, fb.corrected_component
                FROM feedback_labels fb
                INNER JOIN (
                    SELECT prediction_id, MAX(id) AS max_id
                    FROM feedback_labels
                    WHERE resolved = 1
                    GROUP BY prediction_id
                ) pick
                  ON pick.max_id = fb.id
            )
            SELECT
                p.machine_id,
                p.RUL,
                p.RUL_std,
                p.status,
                p.temperature,
                p.air_temperature,
                p.torque,
                p.tool_wear,
                p.speed,
                p.voltage,
                p.current,
                p.power_kw,
                p.vibration,
                p.efficiency,
                p.health_index,
                p.failure_probability,
                p.time_to_failure_hours,
                COALESCE(lf.corrected_component, p.fault_component) AS fault_component,
                p.fault_confidence,
                p.fault_severity,
                CASE WHEN lf.corrected_component IS NULL THEN 0 ELSE 1 END AS feedback_override
            FROM predictions p
            LEFT JOIN latest_feedback lf ON lf.prediction_id = p.id
            WHERE COALESCE(lf.corrected_component, p.fault_component) IS NOT NULL
              AND COALESCE(lf.corrected_component, p.fault_component) != ''
              AND p.fault_confidence >= ?
            ORDER BY p.id DESC
            """
        else:
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
                fault_component,
                fault_confidence,
                fault_severity,
                0 AS feedback_override
            FROM predictions
            WHERE fault_component IS NOT NULL
              AND fault_component != ''
              AND fault_confidence >= ?
            ORDER BY id DESC
            """

        if limit and limit > 0:
            query += f" LIMIT {int(limit)}"

        df = pd.read_sql_query(query, conn, params=(float(min_confidence),))

    if df.empty:
        raise ValueError("No records matched export criteria")

    # Keep only supervised target and model features expected by trainer.
    out = df.copy()
    out["fault_component"] = out["fault_component"].astype(str)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Exported {len(out)} rows to {output_csv}")
    return int(len(out))


def main():
    parser = argparse.ArgumentParser(description="Export diagnosis DB records into supervised fault-localization dataset")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum fault_confidence filter")
    parser.add_argument("--limit", type=int, default=0, help="Optional max row count")
    parser.add_argument("--no-feedback", action="store_true", help="Ignore resolved human relabel feedback")
    args = parser.parse_args()

    export_training_data(
        db_path=args.db,
        output_csv=args.out,
        min_confidence=args.min_confidence,
        limit=args.limit,
        include_resolved_feedback=not args.no_feedback,
    )


if __name__ == "__main__":
    main()
