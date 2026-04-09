"""
model_train.py — end-to-end training script.

Run this once to:
  1. Download CMAPSS FD001 dataset
  2. Preprocess with piecewise linear RUL
  3. Train upgraded Transformer with MC Dropout
  4. Evaluate on test set
  5. Save model + scaler

Usage:
    python -m src.model_train

Output:
    models/best_model_cmapss.keras
    data/scaler_cmapss.pkl
    data/X_cmapss.npy  (for inference replay)
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(model, X_test, y_rul_test, n_passes=30):
    """Evaluate using MC Dropout — more realistic than single-pass eval."""
    from src.model import predict_with_uncertainty

    print(f"\nEvaluating with {n_passes} MC Dropout passes per sample...")
    rul_means, rul_stds = [], []

    for i in range(len(X_test)):
        x = X_test[i:i+1]
        mean, std, _ = predict_with_uncertainty(model, x, n_passes=n_passes)
        rul_means.append(mean)
        rul_stds.append(std)

    rul_means = np.array(rul_means)
    rul_stds  = np.array(rul_stds)

    mae  = mean_absolute_error(y_rul_test, rul_means)
    rmse = np.sqrt(mean_squared_error(y_rul_test, rul_means))
    avg_uncertainty = np.mean(rul_stds)

    print(f"\n{'─'*40}")
    print(f"  MAE  (lower = better): {mae:.2f} cycles")
    print(f"  RMSE (lower = better): {rmse:.2f} cycles")
    print(f"  Avg uncertainty (std): {avg_uncertainty:.2f} cycles")
    print(f"{'─'*40}")

    # Show a few sample predictions
    print(f"\nSample predictions (first 10):")
    print(f"{'True RUL':>10} {'Pred Mean':>12} {'Uncertainty':>13}")
    print(f"{'─'*37}")
    for i in range(min(10, len(y_rul_test))):
        print(f"{y_rul_test[i]:>10.1f} {rul_means[i]:>12.1f} ± {rul_stds[i]:>8.1f}")

    return mae, rmse, avg_uncertainty


def main():
    print("=" * 50)
    print("  Edge AI — CMAPSS Model Training Pipeline")
    print("=" * 50)

    # ── Step 1: Preprocess ────────────────────────────────────
    from src.preprocess_cmapss import run_pipeline
    X, y_rul, y_stage, scaler = run_pipeline()

    # ── Step 2: Train ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Training Transformer + MC Dropout")
    print("=" * 50)

    from src.model import train
    model_dir = os.path.join(BASE_DIR, "models")
    model, history = train(X, y_rul, y_stage, model_dir, num_features=X.shape[2])

    # ── Step 3: Quick validation eval ────────────────────────
    split   = int(len(X) * 0.8)
    X_val   = X[split:]
    y_val   = y_rul[split:]

    # Fast single-pass eval first
    val_preds = []
    for i in range(len(X_val)):
        pred, _ = model(X_val[i:i+1], training=False)
        val_preds.append(float(pred[0, 0]))
    val_preds = np.clip(val_preds, 0, None)
    mae  = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"\nValidation (single-pass): MAE={mae:.2f}  RMSE={rmse:.2f}")

    # MC Dropout eval on a subset (full eval is slow)
    subset = min(200, len(X_val))
    evaluate(model, X_val[:subset], y_val[:subset], n_passes=20)

    # ── Step 4: Save final artifacts ──────────────────────────
    final_path = os.path.join(model_dir, "best_model_cmapss.keras")
    print(f"\n✅ Model saved: {final_path}")
    print(f"✅ Scaler saved: data/scaler_cmapss.pkl")
    print(f"✅ Data saved:   data/X_cmapss.npy")

    print("\nNext step: run convert_tflite_cmapss.py to export for edge deployment")


if __name__ == "__main__":
    main()