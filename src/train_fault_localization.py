import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.fault_localization import FAULT_FEATURE_NAMES, build_fault_feature_vector


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(BASE_DIR, "data", "fault_localization_labeled.csv")
DEFAULT_MODEL_OUT = os.path.join(BASE_DIR, "models", "fault_localizer.pkl")
DEFAULT_META_OUT = os.path.join(BASE_DIR, "models", "fault_localizer_meta.json")


def _build_matrix(df: pd.DataFrame):
    records = df.to_dict("records")
    x = np.vstack([build_fault_feature_vector(rec) for rec in records]).astype(np.float32)
    y = df["fault_component"].astype(str).values
    return x, y


def train_fault_localizer(
    input_csv: str,
    model_out: str,
    meta_out: str,
    version: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Training dataset not found: {input_csv}. Create it first from labeled diagnostics."
        )

    df = pd.read_csv(input_csv)
    if "fault_component" not in df.columns:
        raise ValueError("Dataset must include target column: fault_component")

    df = df.dropna(subset=["fault_component"]).copy()

    label_counts = df["fault_component"].astype(str).value_counts()
    dropped_classes = [str(label) for label, count in label_counts.items() if int(count) < 2]
    if dropped_classes:
        df = df[~df["fault_component"].astype(str).isin(dropped_classes)].copy()

    if len(df) < 50:
        raise ValueError("Need at least 50 labeled rows for training a stable classifier")

    surviving_classes = df["fault_component"].astype(str).nunique()
    if surviving_classes < 2:
        raise ValueError("Need at least 2 classes with >=2 samples each for training")

    x, y = _build_matrix(df)

    feature_stats = {}
    for idx, feature_name in enumerate(FAULT_FEATURE_NAMES):
        column = x[:, idx]
        feature_stats[feature_name] = {
            "mean": float(np.mean(column)),
            "std": float(np.std(column)),
            "p10": float(np.percentile(column, 10)),
            "p90": float(np.percentile(column, 90)),
        }

    class_counts = (
        df["fault_component"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    counts_after_filter = pd.Series(y).value_counts()
    min_count = int(counts_after_filter.min()) if not counts_after_filter.empty else 0

    effective_test_size = float(test_size)
    if len(df) < 80 or min_count < 2:
        effective_test_size = 0.0

    if effective_test_size > 0:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=effective_test_size,
            random_state=random_state,
            stratify=y,
        )
    else:
        x_train, x_test, y_train, y_test = x, x, y, y

    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    train_min_count = int(pd.Series(y_train).value_counts().min())
    calibrate = train_min_count >= 3 and len(pd.Series(y_train).unique()) >= 2
    if calibrate:
        model = CalibratedClassifierCV(base_model, method="sigmoid", cv=min(3, train_min_count))
    else:
        model = base_model
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    trained_at = datetime.utcnow().isoformat() + "Z"
    model_version = version or f"fault-localizer-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    classes = [str(c) for c in model.classes_]
    artifact = {
        "model": model,
        "classes": classes,
        "feature_names": list(FAULT_FEATURE_NAMES),
        "version": model_version,
        "trained_at": trained_at,
    }

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(artifact, model_out)

    metrics = {
        "version": model_version,
        "trained_at": trained_at,
        "dataset": os.path.abspath(input_csv),
        "rows": int(len(df)),
        "classes": classes,
        "dropped_classes": dropped_classes,
        "calibrated": bool(calibrate),
        "class_counts": class_counts,
        "feature_names": list(FAULT_FEATURE_NAMES),
        "feature_stats": feature_stats,
        "weighted_f1": report.get("weighted avg", {}).get("f1-score", 0.0),
        "accuracy": report.get("accuracy", 0.0),
    }
    with open(meta_out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("Training completed")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train supervised fault localization model")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to labeled CSV")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_OUT, help="Path to saved model artifact")
    parser.add_argument("--meta-out", default=DEFAULT_META_OUT, help="Path to metadata JSON")
    parser.add_argument("--version", default=None, help="Explicit model version")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    train_fault_localizer(
        input_csv=args.input,
        model_out=args.model_out,
        meta_out=args.meta_out,
        version=args.version,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
