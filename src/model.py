import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ── transformer block ─────────────────────────────────────────
def transformer_block(x, embed_dim, num_heads, ff_dim, dropout=0.1):
    # multi-head self attention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim
    )(x, x)
    attn_output = layers.Dropout(dropout)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    # feed forward network
    ffn = keras.Sequential([
        layers.Dense(ff_dim, activation='relu'),
        layers.Dense(embed_dim)
    ])
    ffn_output = ffn(x)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    return x

# ── positional encoding ───────────────────────────────────────
def positional_encoding(window_size, embed_dim):
    positions = np.arange(window_size)[:, np.newaxis]
    dims      = np.arange(embed_dim)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / embed_dim)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)

# ── build model ───────────────────────────────────────────────
def build_model(window_size=30, num_features=5,
                embed_dim=32, num_heads=4,
                ff_dim=64, num_blocks=2, dropout=0.1):

    inputs = keras.Input(shape=(window_size, num_features))

    # project input features to embed_dim
    x = layers.Dense(embed_dim)(inputs)

    # add positional encoding
    pos_enc = positional_encoding(window_size, embed_dim)
    x = x + pos_enc

    # stack transformer blocks
    for _ in range(num_blocks):
        x = transformer_block(x, embed_dim, num_heads, ff_dim, dropout)

    # global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)

    # two output heads
    rul_output   = layers.Dense(1, name='rul')(x)
    stage_output = layers.Dense(3, activation='softmax', name='stage')(x)

    model = keras.Model(inputs=inputs,
                        outputs=[rul_output, stage_output])
    return model

# ── train ─────────────────────────────────────────────────────
def train(X, y_rul, y_stage, model_dir):
    # train/val split
    split    = int(len(X) * 0.8)
    X_train  = X[:split];      X_val  = X[split:]
    yr_train = y_rul[:split];  yr_val = y_rul[split:]
    ys_train = y_stage[:split]; ys_val = y_stage[split:]

    # class weights for imbalanced health stages
    total   = len(ys_train)
    classes = np.unique(ys_train)
    weights = {c: total / (len(classes) * np.sum(ys_train == c))
               for c in classes}
    print("Class weights:", weights)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            'rul'  : 'mse',
            'stage': 'sparse_categorical_crossentropy'
        },
        loss_weights={'rul': 1.0, 'stage': 2.0},
        metrics={
            'rul'  : 'mae',
            'stage': 'accuracy'
        }
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.keras'),
            monitor='val_loss', save_best_only=True
        )
    ]

    history = model.fit(
        X_train,
        {'rul': yr_train, 'stage': ys_train},
        validation_data=(X_val, {'rul': yr_val, 'stage': ys_val}),
        epochs=50,
        batch_size=64,
        
        callbacks=callbacks,
        verbose=1
    )
    return model, history

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("Loading processed data...")
    X       = np.load(os.path.join(base, 'data', 'X.npy'))
    y_rul   = np.load(os.path.join(base, 'data', 'y_rul.npy'))
    y_stage = np.load(os.path.join(base, 'data', 'y_stage.npy'))

    model_dir = os.path.join(base, 'models')
    model, history = train(X, y_rul, y_stage, model_dir)

    print("\nTraining complete")
    print(f"Model saved to {model_dir}")