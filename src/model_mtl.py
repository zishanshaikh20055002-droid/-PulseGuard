import tensorflow as tf
from tensorflow.keras import layers, Model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Self-Attention
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + x)

    # Feed-Forward Network
    res = x
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(res + x)
    return x

def build_mtl_transformer(window_size=30, num_features=5):
    inputs = layers.Input(shape=(window_size, num_features), name="sensor_input")
    
    # ── Shared Representation Backbone ──
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
    
    # For RUL and Classification, we pool the sequence into a single feature vector
    shared_features = layers.GlobalAveragePooling1D()(x)
    shared_features = layers.Dropout(0.2)(shared_features)

    # ── HEAD 1: System RUL (Regression) ──
    # Predicts the remaining useful life in cycles/minutes
    rul_dense = layers.Dense(64, activation="relu")(shared_features)
    rul_dense = layers.Dropout(0.2)(rul_dense)
    out_rul = layers.Dense(1, activation="relu", name="head_rul")(rul_dense)

    # ── HEAD 2: Fault Diagnostics (Multi-Label Classification) ──
    # Predicts the probability of TWF, HDF, PWF, OSF, RNF independently
    fault_dense = layers.Dense(64, activation="relu")(shared_features)
    fault_dense = layers.Dropout(0.2)(fault_dense)
    # Sigmoid is used because a machine could theoretically have multiple simultaneous faults
    out_faults = layers.Dense(5, activation="sigmoid", name="head_faults")(fault_dense)

    # ── HEAD 3: Anomaly Detection (Reconstruction Autoencoder) ──
    # Attempts to rebuild the input. If the reconstruction error is high, 
    # it means the machine is behaving in a zero-day "unknown" state.
    decoder = layers.Conv1DTranspose(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    out_reconstruction = layers.Conv1DTranspose(filters=num_features, kernel_size=3, padding="same", activation="linear", name="head_anomaly")(decoder)

    # Compile the Multi-Task Model
    model = Model(inputs=inputs, outputs=[out_rul, out_faults, out_reconstruction], name="MTL_Machine_Health")
    
    return model

if __name__ == "__main__":
    # Test the architecture compilation
    model = build_mtl_transformer(window_size=30, num_features=5)
    model.summary()