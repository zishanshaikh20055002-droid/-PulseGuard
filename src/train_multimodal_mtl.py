"""
train_multimodal_mtl.py

Training scaffold for the multimodal multi-task model.

Dataset format (.npz expected keys):
- X_process:    (N, process_window, process_features)
- X_vibration:  (N, vibration_window, 3)
- X_acoustic:   (N, acoustic_window, 1)
- X_electrical: (N, electrical_window, electrical_features)
- X_thermal:    (N, thermal_embedding_dim)
- y_rul:        (N,)
- y_faults:     (N, num_fault_classes)
- y_anomaly:    (N,) binary label or continuous anomaly target
"""

import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.model_multimodal_mtl import (
    build_multimodal_mtl_model,
    compile_multimodal_mtl_model,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(BASE_DIR, "data", "multimodal_train.npz")


def _load_dataset(npz_path):
    data = np.load(npz_path)
    return {
        "X_process": data["X_process"],
        "X_vibration": data["X_vibration"],
        "X_acoustic": data["X_acoustic"],
        "X_electrical": data["X_electrical"],
        "X_thermal": data["X_thermal"],
        "y_rul": data["y_rul"],
        "y_faults": data["y_faults"],
        "y_anomaly": data["y_anomaly"],
    }


def _make_synthetic_dataset(n=1024):
    rng = np.random.default_rng(42)
    X_process = rng.normal(size=(n, 30, 14)).astype(np.float32)
    X_vibration = rng.normal(size=(n, 256, 3)).astype(np.float32)
    X_acoustic = rng.normal(size=(n, 2048, 1)).astype(np.float32)
    X_electrical = rng.normal(size=(n, 64, 4)).astype(np.float32)
    X_thermal = rng.normal(size=(n, 128)).astype(np.float32)

    y_rul = rng.uniform(0, 200, size=(n,)).astype(np.float32)
    y_faults = rng.integers(0, 2, size=(n, 6)).astype(np.float32)
    y_anomaly = rng.integers(0, 2, size=(n,)).astype(np.float32)

    return {
        "X_process": X_process,
        "X_vibration": X_vibration,
        "X_acoustic": X_acoustic,
        "X_electrical": X_electrical,
        "X_thermal": X_thermal,
        "y_rul": y_rul,
        "y_faults": y_faults,
        "y_anomaly": y_anomaly,
    }


def train(dataset_path=DEFAULT_DATASET, epochs=30, batch_size=32):
    if os.path.exists(dataset_path):
        print(f"Loading dataset: {dataset_path}")
        ds = _load_dataset(dataset_path)
    else:
        print("Dataset not found; using synthetic data for architecture smoke training")
        ds = _make_synthetic_dataset(n=1024)

    inputs = [
        ds["X_process"],
        ds["X_vibration"],
        ds["X_acoustic"],
        ds["X_electrical"],
        ds["X_thermal"],
    ]

    targets = {
        "head_rul": ds["y_rul"],
        "head_faults": ds["y_faults"],
        "head_anomaly_score": ds["y_anomaly"],
    }

    (
        Xp_train,
        Xp_val,
        Xv_train,
        Xv_val,
        Xa_train,
        Xa_val,
        Xe_train,
        Xe_val,
        Xt_train,
        Xt_val,
        yr_train,
        yr_val,
        yf_train,
        yf_val,
        ya_train,
        ya_val,
    ) = train_test_split(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        targets["head_rul"],
        targets["head_faults"],
        targets["head_anomaly_score"],
        test_size=0.2,
        random_state=42,
    )

    model = build_multimodal_mtl_model(
        process_window=Xp_train.shape[1],
        process_features=Xp_train.shape[2],
        vibration_window=Xv_train.shape[1],
        acoustic_window=Xa_train.shape[1],
        electrical_window=Xe_train.shape[1],
        electrical_features=Xe_train.shape[2],
        thermal_embedding_dim=Xt_train.shape[1],
        num_fault_classes=yf_train.shape[1],
    )
    compile_multimodal_mtl_model(model)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(BASE_DIR, "models", "best_multimodal_mtl.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        [Xp_train, Xv_train, Xa_train, Xe_train, Xt_train],
        {
            "head_rul": yr_train,
            "head_faults": yf_train,
            "head_anomaly_score": ya_train,
        },
        validation_data=(
            [Xp_val, Xv_val, Xa_val, Xe_val, Xt_val],
            {
                "head_rul": yr_val,
                "head_faults": yf_val,
                "head_anomaly_score": ya_val,
            },
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("Training complete. Saved best model to models/best_multimodal_mtl.keras")


if __name__ == "__main__":
    train()
