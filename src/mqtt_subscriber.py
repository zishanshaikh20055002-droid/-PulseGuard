import json, os, time, logging
import numpy as np
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT   = int(os.getenv("MQTT_PORT", "1883"))
WINDOW_SIZE = 30

def start_subscriber(interpreter, input_details, output_details, scaler, manager, metrics):
    from src.database import insert_data
    from src.sanitize import sanitize_sensor_dict
    window_buffer = []

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe("sensors/+/data")
            logger.info("[MQTT] Connected and subscribed")

    def on_message(client, userdata, msg):
        nonlocal window_buffer
        try:
            data = json.loads(msg.payload.decode())
        except Exception:
            return
        machine_id = data.get("machine_id", "M1")
        try:
            features = [
                float(data["air_temperature"]),
                float(data["temperature"]),
                float(data["rotational_speed"]),
                float(data["torque"]),
                float(data["tool_wear"]),
            ]
        except (KeyError, ValueError) as e:
            logger.error(f"[MQTT] Missing field: {e}")
            return

        scaled = scaler.transform([features])[0].tolist()
        window_buffer.append(scaled)
        if len(window_buffer) > WINDOW_SIZE:
            window_buffer.pop(0)
        if len(window_buffer) < WINDOW_SIZE:
            return

        sample = np.array(window_buffer, dtype=np.float32).reshape(1, WINDOW_SIZE, 5)
        t0 = time.time()
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        prediction = float(interpreter.get_tensor(output_details[0]["index"])[0][0])
        latency = time.time() - t0
        prediction = max(0.0, prediction)
        status = "CRITICAL" if prediction < 60 else "WARNING" if prediction < 120 else "HEALTHY"

        try:
            metrics["rul_gauge"].labels(machine_id=machine_id).set(prediction)
            metrics["health_status_counter"].labels(machine_id=machine_id, status=status).inc()
            metrics["inference_latency"].labels(machine_id=machine_id).observe(latency)
            metrics["sensor_temperature"].labels(machine_id=machine_id).set(features[1])
            metrics["sensor_torque"].labels(machine_id=machine_id).set(features[3])
            metrics["sensor_tool_wear"].labels(machine_id=machine_id).set(features[4])
            metrics["sensor_speed"].labels(machine_id=machine_id).set(features[2])
        except Exception as e:
            logger.error(f"[MQTT] Metrics: {e}")

        result = sanitize_sensor_dict({
            "machine_id": machine_id,
            "step": int(data.get("step", 0)),
            "RUL": prediction,
            "status": status,
            "temperature": features[1],
            "air_temperature": features[0],
            "torque": features[3],
            "tool_wear": features[4],
            "speed": features[2],
        })
        try:
            insert_data(result)
        except Exception as e:
            logger.error(f"[MQTT] DB: {e}")

        result["RUL"] = prediction
        manager.broadcast_from_thread(result)
        logger.info(f"[MQTT] {machine_id} RUL={prediction:.1f} {status} {latency*1000:.2f}ms")

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