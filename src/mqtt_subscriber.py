import json, os, time, logging, warnings
import numpy as np
import paho.mqtt.client as mqtt
from src.ingestion import HardwareAgnosticBuffer

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT   = int(os.getenv("MQTT_PORT", "1883"))
WINDOW_SIZE = 30
NUM_FEATURES = 14  # CMAPSS uses 14
N_PASSES    = 30

ingestion_buffer = HardwareAgnosticBuffer(window_size=WINDOW_SIZE, num_features=NUM_FEATURES)

def start_subscriber(interpreter, input_details, output_details, scaler, manager, metrics):
    from src.database import insert_data
    from src.sanitize import sanitize_sensor_dict

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe("sensors/+/data")
            logger.info("[MQTT] Connected and subscribed to CMAPSS sensor stream")

    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
        except Exception:
            return
            
        machine_id = data.get("machine_id", "M1")
        step = int(data.get("step", 0))
        raw_features = data.get("features", [])
        
        ingestion_buffer.process_payload(machine_id, step, raw_features)
        raw_window = ingestion_buffer.get_valid_window(machine_id)
        
        if raw_window is None:
            return

        scaled_window = scaler.transform(raw_window[0])
        base_sample = scaled_window.reshape(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
        clean_features = raw_window[0][-1]
        
        t0 = time.time()
        rul_predictions = []
        stage_predictions = []
        
        for _ in range(N_PASSES):
            noise = np.random.normal(0, 0.01, base_sample.shape).astype(np.float32)
            interpreter.set_tensor(input_details[0]["index"], base_sample + noise)
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
        except Exception:
            pass

        result = sanitize_sensor_dict({
            "machine_id": machine_id,
            "step": step,
            "RUL": prediction,
            "RUL_std": rul_std,
            "status": status,
            "stage_probs": [round(p, 3) for p in stage_probs],
            # Map first 5 CMAPSS features to UI so telemetry bars animate
            "temperature": float(clean_features[0]),
            "air_temperature": float(clean_features[1]),
            "torque": float(clean_features[2]),
            "tool_wear": float(clean_features[3]),
            "speed": float(clean_features[4]),
        })
        
        try:
            insert_data(result)
        except Exception:
            pass

        manager.broadcast_from_thread(result)
        logger.info(f"[MQTT] {machine_id} Step={step} RUL={prediction:.1f}±{rul_std:.1f} {status} {latency*1000:.2f}ms")

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