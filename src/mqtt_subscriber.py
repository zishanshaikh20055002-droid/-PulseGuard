import json, os, time, logging, warnings
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT   = int(os.getenv("MQTT_PORT", "1883"))
WINDOW_SIZE = 30
NUM_FEATURES = 14
N_PASSES    = 30

window_buffer = deque(maxlen=WINDOW_SIZE)

def start_subscriber(interpreter, input_details, output_details, scaler, manager, metrics):
    from src.database import insert_data
    from src.sanitize import sanitize_sensor_dict

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe("sensors/+/data")
            logger.info("[MQTT] Connected and subscribed to CMAPSS stream")

    def on_message(client, userdata, msg):
        global window_buffer
        
        try:
            data = json.loads(msg.payload.decode())
        except Exception:
            return
            
        machine_id = data.get("machine_id", "M1")
        features = data.get("features", [])
        
        if len(features) != NUM_FEATURES:
            return

        scaled = scaler.transform([features])[0].tolist()
        window_buffer.append(scaled)
        
        if len(window_buffer) < WINDOW_SIZE:
            return

        # Base sample
        base_sample = np.array(window_buffer, dtype=np.float32).reshape(1, WINDOW_SIZE, NUM_FEATURES)
        
        t0 = time.time()
        
        rul_predictions = []
        stage_predictions = []
        
        for _ in range(N_PASSES):
            # THE FIX: Inject micro-noise to the input array using Numpy!
            # This guarantees variance, simulating MC Dropout manually.
            noise = np.random.normal(0, 0.01, base_sample.shape).astype(np.float32)
            noisy_sample = base_sample + noise
            
            interpreter.set_tensor(input_details[0]["index"], noisy_sample)
            interpreter.invoke()
            
            rul_predictions.append(interpreter.get_tensor(output_details[0]["index"])[0][0])
            stage_predictions.append(interpreter.get_tensor(output_details[1]["index"])[0])
            
        rul_mean = np.mean(rul_predictions)
        rul_std = np.std(rul_predictions)
        stage_probs = np.mean(stage_predictions, axis=0).tolist()
        
        latency = time.time() - t0
        prediction = max(0.0, float(rul_mean))
        rul_std = float(rul_std)
        
        if len(stage_probs) == 3:
            pred_stage = np.argmax(stage_probs)
            status = "HEALTHY" if pred_stage == 0 else "WARNING" if pred_stage == 1 else "CRITICAL"
        else:
            status = "CRITICAL" if prediction < 60 else "WARNING" if prediction < 120 else "HEALTHY"
            stage_probs = [1.0, 0.0, 0.0] if status == "HEALTHY" else [0.0, 1.0, 0.0] if status == "WARNING" else [0.0, 0.0, 1.0]

        try:
            metrics["rul_gauge"].labels(machine_id=machine_id).set(prediction)
            if "rul_std_gauge" in metrics:
                metrics["rul_std_gauge"].labels(machine_id=machine_id).set(rul_std)
            metrics["health_status_counter"].labels(machine_id=machine_id, status=status).inc()
            metrics["inference_latency"].labels(machine_id=machine_id).observe(latency)
            metrics["sensor_temperature"].labels(machine_id=machine_id).set(features[0])
            metrics["sensor_torque"].labels(machine_id=machine_id).set(features[1])
            metrics["sensor_tool_wear"].labels(machine_id=machine_id).set(features[2])
            metrics["sensor_speed"].labels(machine_id=machine_id).set(features[3])
        except Exception:
            pass

        result = sanitize_sensor_dict({
            "machine_id": machine_id,
            "step": int(data.get("step", 0)),
            "RUL": prediction,
            "RUL_std": rul_std,
            "status": status,
            "stage_probs": [round(p, 3) for p in stage_probs],
            "temperature": features[0],
            "air_temperature": features[1],
            "torque": features[2],
            "tool_wear": features[3],
            "speed": features[4],
        })
        
        try:
            insert_data(result)
        except Exception:
            pass

        manager.broadcast_from_thread(result)
        logger.info(f"[MQTT] {machine_id} RUL={prediction:.1f}±{rul_std:.1f} {status} {latency*1000:.2f}ms")

    client = mqtt.Client(client_id="fastapi-subscriber")
    client.on_connect = on_connect
    client.on_message = on_message

    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            client.loop_forever()
        except Exception as e:
            logger.error(f"[MQTT] Error: {e} — retrying in 5s")
            time.sleep(5)