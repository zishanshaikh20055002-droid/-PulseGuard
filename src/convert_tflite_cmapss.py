"""
convert_tflite_cmapss.py â€” converts the trained CMAPSS model to TFLite.

MC Dropout + TFLite note:
  Standard TFLite quantization disables dropout at inference.
  To preserve MC Dropout behaviour we use a wrapper that runs
  N stochastic forward passes and returns mean + std directly.
  This wrapper is what gets exported to TFLite.
"""

import tensorflow as tf
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH_1 = os.path.join(BASE_DIR, "models", "best_model.keras")
MODEL_PATH_2 = os.path.join(BASE_DIR, "models", "best_model_cmapss.keras")
MODEL_PATH   = MODEL_PATH_2 if os.path.exists(MODEL_PATH_2) else MODEL_PATH_1

OUT_PATH     = os.path.join(BASE_DIR, "models", "model_cmapss_int8.tflite")

WINDOW_SIZE  = 30
NUM_FEATURES = 14
N_PASSES     = 30   # MC Dropout forward passes


class MCDropoutWrapper(tf.Module):
    """
    Wraps the Keras model so TFLite can run MC Dropout inference.
    Takes a single window and returns (rul_mean, rul_std, stage_probs).
    """
    def __init__(self, model, n_passes=N_PASSES):
        self.model    = model
        self.n_passes = n_passes

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, WINDOW_SIZE, NUM_FEATURES], dtype=tf.float32)
    ])
    def predict(self, x):
        rul_preds = []
        stage_preds = []

        for _ in range(self.n_passes):
            noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1e-5)
            noisy_x = x + noise
            
            rul, stage = self.model(noisy_x, training=True)
            rul_preds.append(rul[0, 0])
            stage_preds.append(stage[0])

        rul_stack   = tf.stack(rul_preds)     # (N,)
        stage_stack = tf.stack(stage_preds)   # (N, 3)

        rul_mean    = tf.reduce_mean(rul_stack)
        rul_std     = tf.math.reduce_std(rul_stack)
        stage_probs = tf.reduce_mean(stage_stack, axis=0)  # (3,)

        return {
            "rul_mean":    tf.reshape(rul_mean, [1]),
            "rul_std":     tf.reshape(rul_std,  [1]),
            "stage_probs": tf.reshape(stage_probs, [3]),
        }


def representative_data_gen():
    """Calibration data for quantization."""
    data_path = os.path.join(BASE_DIR, "data", "X_cmapss.npy")
    try:
        X = np.load(data_path)
        indices = np.random.choice(len(X), size=min(200, len(X)), replace=False)
        for i in indices:
            yield [X[i:i+1].astype(np.float32)]
    except Exception as e:
        print(f"Warning: Could not load actual data for calibration: {e}. Using random dummy data.")
        for _ in range(100):
            yield [np.random.rand(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)]


def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()

    print(f"\nWrapping for MC Dropout ({N_PASSES} passes)...")
    wrapper = MCDropoutWrapper(model, n_passes=N_PASSES)

    dummy = np.random.rand(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
    result = wrapper.predict(dummy)
    print(f"  Test output:")
    print(f"    rul_mean : {result['rul_mean'].numpy()}")
    print(f"    rul_std  : {result['rul_std'].numpy()}")
    print(f"    stage_probs: {result['stage_probs'].numpy()}")

    print("\nConverting to TFLite (float16 â€” preserves dropout)...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [wrapper.predict.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"\nâœ… Saved: {OUT_PATH}  ({size_kb:.1f} KB)")

    print("\nVerifying TFLite model...")
    interp = tf.lite.Interpreter(model_path=OUT_PATH)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    print(f"  Input : {inp[0]['shape']} {inp[0]['dtype']}")
    print(f"  Outputs:")
    for o in out:
        print(f"    {o['name']}: {o['shape']} {o['dtype']}")
    print("\nâœ… TFLite model verified successfully")


if __name__ == "__main__":
    main()