import tensorflow as tf
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_mtl_model.keras")
OUT_PATH = os.path.join(BASE_DIR, "models", "model_mtl_int8.tflite")

WINDOW_SIZE = 30
NUM_FEATURES = 5

def representative_data_gen():
    """Provides calibration data for INT8 quantization."""
    data_path = os.path.join(BASE_DIR, "data", "X_mtl.npy")
    try:
        X = np.load(data_path)
        # Select 200 random samples for calibration
        indices = np.random.choice(len(X), size=min(200, len(X)), replace=False)
        for i in indices:
            yield [X[i:i+1].astype(np.float32)]
    except Exception as e:
        print(f"Warning: Could not load calibration data. Using dummy data. {e}")
        for _ in range(100):
            yield [np.random.rand(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)]

def convert_model():
    print(f"Loading MTL model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Force Dropout layers to remain active for manual MC Dropout inference
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.training = True

    print("Converting to INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"✅ Conversion complete. Saved at {OUT_PATH} ({size_kb:.1f} KB)")
    
    # Verify outputs
    interp = tf.lite.Interpreter(model_path=OUT_PATH)
    interp.allocate_tensors()
    print("\nOutputs:")
    for o in interp.get_output_details():
        print(f"  {o['name']}: {o['shape']}")

if __name__ == "__main__":
    convert_model()