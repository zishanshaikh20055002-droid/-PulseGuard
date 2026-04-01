from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os
import asyncio
import joblib

app = FastAPI()

# -------------------- PATHS --------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model_int8.tflite")
SCALER_PATH = os.path.join(BASE_DIR, "data", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "X.npy")

# -------------------- LOAD MODEL --------------------

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------- LOAD SCALER --------------------

scaler = joblib.load(SCALER_PATH)

# -------------------- HOME --------------------

@app.get("/")
def home():
    return {"message": "Edge AI Predictive Maintenance API running 🚀"}

# -------------------- PREDICT API --------------------

class InputData(BaseModel):
    data: list

@app.post("/predict")
def predict(input_data: InputData):
    sample = np.array(input_data.data).astype(np.float32)
    sample = sample.reshape(1, 30, 5)

    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    if prediction < 5:
        status = "CRITICAL"
    elif prediction < 20:
        status = "WARNING"
    else:
        status = "HEALTHY"

    return {
        "RUL": float(prediction),
        "status": status
    }

# -------------------- WEBSOCKET --------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    X = np.load(DATA_PATH)

    start = 200

    for i in range(start, start + 50):
        sample = X[i:i+1].astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        if prediction < 5:
            status = "CRITICAL"
        elif prediction < 20:
            status = "WARNING"
        else:
            status = "HEALTHY"

        sensor_values = sample[0][-1]
        original_values = scaler.inverse_transform([sensor_values])[0]

        data = {
            "step": i,
            "RUL": float(prediction),
            "status": status,
            "temperature": float(original_values[0]),
            "air_temperature": float(original_values[1]),
            "torque": float(original_values[2]),
            "tool_wear": float(original_values[3]),
            "speed": float(original_values[4])
        }

        await websocket.send_json(data)
        await asyncio.sleep(1)