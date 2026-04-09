import tensorflow as tf
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH_1 = os.path.join(BASE_DIR, "models", "best_model.keras")
MODEL_PATH_2 = os.path.join(BASE_DIR, "models", "best_model_cmapss.keras")
MODEL_PATH = MODEL_PATH_2 if os.path.exists(MODEL_PATH_2) else MODEL_PATH_1
OUT_PATH = os.path.join(BASE_DIR, "models", "model_cmapss_int8.tflite")

WINDOW_SIZE = 30
NUM_FEATURES = 14

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# =====================================================================
# THE FIX: Force all Dropout layers to remain active during inference
# =====================================================================
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dropout):
        # This tells TF to never disable this layer, even when predicting
        layer.training = True 

def representative_data_gen():
    for _ in range(100):
        yield [np.random.rand(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)]

print("Converting to INT8 TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ Conversion complete. Saved at {OUT_PATH}")