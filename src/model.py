"""
model.py — Transformer with Monte Carlo Dropout for RUL prediction
           with uncertainty quantification.

Key upgrade over v1:
  - MC Dropout: dropout stays active at inference time
  - Run N forward passes → distribution of predictions
  - Output: RUL mean ± std (epistemic uncertainty)
  - 14 CMAPSS sensors instead of 5 ai4i2020 sensors
  - Piecewise linear RUL target (more realistic)

MC Dropout intuition:
  Each forward pass with dropout active gives a slightly different
  prediction. The spread of those predictions IS the model's uncertainty.
  High spread = "I'm not sure". Low spread = "I'm confident".
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


# ── Transformer block ─────────────────────────────────────────
def transformer_block(x, embed_dim, num_heads, ff_dim, dropout_rate=0.2):
    """
    Standard transformer encoder block.
    Dropout is applied ALWAYS (not just during training) — this is the
    key difference that enables MC Dropout uncertainty estimation.
    """
    # Multi-head self-attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim
    )(x, x)
    # training=None means dropout follows the global keras learning_phase
    # We control this at inference time by calling model(x, training=True)
    attn = layers.Dropout(dropout_rate)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # Feed-forward network
    ffn = keras.Sequential([
        layers.Dense(ff_dim, activation="gelu"),  # GELU > ReLU for transformers
        layers.Dropout(dropout_rate),
        layers.Dense(embed_dim),
    ])
    ffn_out = ffn(x)
    ffn_out = layers.Dropout(dropout_rate)(ffn_out)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_out)
    return x


# ── Positional encoding ───────────────────────────────────────
def positional_encoding(window_size, embed_dim):
    positions = np.arange(window_size)[:, np.newaxis]
    dims      = np.arange(embed_dim)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / embed_dim)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)


# ── Build model ───────────────────────────────────────────────
def build_model(
    window_size  = 30,
    num_features = 14,    # 14 CMAPSS sensors
    embed_dim    = 64,    # larger embedding for 14 features
    num_heads    = 4,
    ff_dim       = 128,
    num_blocks   = 3,     # deeper for harder task
    dropout_rate = 0.2,
):
    inputs = keras.Input(shape=(window_size, num_features))

    # Project input features to embed_dim
    x = layers.Dense(embed_dim)(inputs)

    # Add positional encoding
    x = x + positional_encoding(window_size, embed_dim)

    # Stack transformer blocks (each has MC Dropout inside)
    for _ in range(num_blocks):
        x = transformer_block(x, embed_dim, num_heads, ff_dim, dropout_rate)

    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Shared dense layers
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # ── Output heads ──
    # RUL regression head
    rul_output = layers.Dense(1, name="rul")(x)

    # Health stage classification head (3 classes: healthy/warning/critical)
    stage_output = layers.Dense(3, activation="softmax", name="stage")(x)

    model = keras.Model(inputs=inputs, outputs=[rul_output, stage_output])
    return model


# ── MC Dropout inference ──────────────────────────────────────
def predict_with_uncertainty(model, x, n_passes=30):
    """
    Run N stochastic forward passes with dropout active.
    Returns mean prediction and standard deviation (uncertainty).

    Args:
        model:    trained Keras model
        x:        input array, shape (1, window_size, num_features)
        n_passes: number of MC samples (30 is standard in literature)

    Returns:
        rul_mean  (float): expected RUL
        rul_std   (float): uncertainty — higher = less confident
        stage_probs (array): mean class probabilities across passes
    """
    rul_preds   = []
    stage_preds = []

    for _ in range(n_passes):
        # training=True keeps dropout active — this is the MC Dropout trick
        rul, stage = model(x, training=True)
        rul_preds.append(float(rul[0, 0]))
        stage_preds.append(stage[0].numpy())

    rul_mean    = float(np.mean(rul_preds))
    rul_std     = float(np.std(rul_preds))
    stage_probs = np.mean(stage_preds, axis=0)

    return rul_mean, rul_std, stage_probs


# ── Train ─────────────────────────────────────────────────────
def train(X, y_rul, y_stage, model_dir, num_features=14):
    split    = int(len(X) * 0.8)
    X_train  = X[:split];       X_val  = X[split:]
    yr_train = y_rul[:split];   yr_val = y_rul[split:]
    ys_train = y_stage[:split]; ys_val = y_stage[split:]

    # Class weights for imbalanced stages
    total   = len(ys_train)
    classes = np.unique(ys_train)
    weights = {int(c): total / (len(classes) * np.sum(ys_train == c))
               for c in classes}
    print("Class weights:", weights)

    # Generate an array mapping the correct weight to every single training sample
    stage_sample_weights = np.array([weights[int(y)] for y in ys_train])

    model = build_model(num_features=num_features)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss={
            "rul":   "mse",
            "stage": "sparse_categorical_crossentropy",
        },
        loss_weights={"rul": 1.0, "stage": 2.0},
        metrics={
            "rul":   ["mae"],
            "stage": ["accuracy"],
        },
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model_cmapss.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        X_train,
        {"rul": yr_train, "stage": ys_train},
        validation_data=(X_val, {"rul": yr_val, "stage": ys_val}),
        epochs=100,
        batch_size=128,
        sample_weight={"stage": stage_sample_weights}, # Replaced class_weight with sample_weight
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("Loading CMAPSS processed data...")
    X       = np.load(os.path.join(base, "data", "X_cmapss.npy"))
    y_rul   = np.load(os.path.join(base, "data", "y_rul_cmapss.npy"))
    y_stage = np.load(os.path.join(base, "data", "y_stage_cmapss.npy"))

    print(f"X: {X.shape} | y_rul: {y_rul.shape} | y_stage: {y_stage.shape}")

    model_dir = os.path.join(base, "models")
    os.makedirs(model_dir, exist_ok=True)

    model, history = train(X, y_rul, y_stage, model_dir, num_features=X.shape[2])

    print("\n✅ Training complete")
    print(f"   Saved to {model_dir}/best_model_cmapss.keras")