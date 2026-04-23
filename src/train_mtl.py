import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.model_mtl import build_mtl_transformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train():
    print("Loading prepared datasets...")
    X = np.load(os.path.join(BASE_DIR, "data", "X_mtl.npy"))
    y_rul = np.load(os.path.join(BASE_DIR, "data", "y_rul_mtl.npy"))
    y_faults = np.load(os.path.join(BASE_DIR, "data", "y_faults_mtl.npy"))
    y_recon = np.load(os.path.join(BASE_DIR, "data", "y_recon_mtl.npy"))

    X_train, X_val, y_train_rul, y_val_rul, y_train_faults, y_val_faults, y_train_recon, y_val_recon = train_test_split(
        X, y_rul, y_faults, y_recon, test_size=0.2, random_state=42
    )

    print("Building Multi-Task Transformer...")
    model = build_mtl_transformer(window_size=30, num_features=5)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "head_rul": "mse",
            "head_faults": "binary_crossentropy", 
            "head_anomaly": "mse"
        },
        loss_weights={
            "head_rul": 1.0,      # High priority
            "head_faults": 2.0,   # Highest priority (hardest task)
            "head_anomaly": 0.5   # Background reconstruction task
        },
        metrics={
            "head_rul": ["mae"],
            "head_faults": ["accuracy", tf.keras.metrics.AUC(name="auc")]
        }
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(BASE_DIR, "models", "best_mtl_model.keras"),
            save_best_only=True,
            monitor="val_loss"
        )
    ]

    print("\nðŸš€ Starting Multi-Task Training on RTX 3050A...")
    model.fit(
        X_train,
        {
            "head_rul": y_train_rul,
            "head_faults": y_train_faults,
            "head_anomaly": y_train_recon
        },
        validation_data=(
            X_val,
            {
                "head_rul": y_val_rul,
                "head_faults": y_val_faults,
                "head_anomaly": y_val_recon
            }
        ),
        epochs=50,
        batch_size=64,
        callbacks=callbacks
    )
    print("âœ… Training complete. Best model saved to models/best_mtl_model.keras")

if __name__ == "__main__":
    train()