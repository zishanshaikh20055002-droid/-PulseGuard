import json
import os
import time
import logging
import random
import numpy as np
import joblib
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PUBLISHER] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MQTT_BROKER   = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT     = int(os.getenv("MQTT_PORT", "1883"))
MACHINE_ID    = os.getenv("MACHINE_ID", "M1")
PUBLISH_TOPIC = f"sensors/{MACHINE_ID}/data"
CONTROL_TOPIC = "sensors/control/mode"
PUBLISH_RATE  = float(os.getenv("PUBLISH_RATE", "1.0"))

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Point to the new CMAPSS data and scaler
DATA_PATH   = os.path.join(BASE_DIR, "data", "X_cmapss.npy")
SCALER_PATH = os.path.join(BASE_DIR, "data", "scaler_cmapss.pkl")

current_mode = os.getenv("PUBLISHER_MODE", "replay")

def generate_random(step: int) -> dict:
    # Generate dummy random data for all 14 features if replay fails
    return {
        "machine_id": MACHINE_ID,
        "step": step,
        "features": [round(random.uniform(0, 100), 2) for _ in range(14)]
    }

def main():
    global current_mode

    logger.info("Loading replay data...")
    try:
        X      = np.load(DATA_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"X.npy: {X.shape}")
    except Exception as e:
        logger.warning(f"Could not load replay data: {e} — using random mode")
        X = None
        scaler = None
        current_mode = "random"

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Connected to {MQTT_BROKER}:{MQTT_PORT}")
            client.subscribe(CONTROL_TOPIC)
        else:
            logger.error(f"Connection failed rc={rc}")

    def on_message(client, userdata, msg):
        global current_mode
        if msg.topic == CONTROL_TOPIC:
            new_mode = msg.payload.decode().strip().lower()
            if new_mode in ("replay", "random"):
                current_mode = new_mode
                logger.info(f"Mode → {current_mode}")

    client = mqtt.Client(client_id=f"publisher-{MACHINE_ID}")
    client.on_connect = on_connect
    client.on_message = on_message

    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            break
        except Exception as e:
            logger.error(f"Broker not ready: {e} — retrying in 3s")
            time.sleep(3)

    client.loop_start()
    logger.info(f"Publishing to {PUBLISH_TOPIC} | mode={current_mode}")

    step = 200
    replay_idx = step

    while True:
        try:
            if current_mode == "replay" and X is not None:
                sample  = X[replay_idx % len(X)]
                last    = sample[-1]
                orig    = scaler.inverse_transform([last])[0]
                
                # Bundle the 14 features into an array
                reading = {
                    "machine_id": MACHINE_ID,
                    "step": step,
                    "features": [round(float(x), 4) for x in orig]
                }
                replay_idx += 1
            else:
                reading = generate_random(step)

            result = client.publish(PUBLISH_TOPIC, json.dumps(reading), qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"[{current_mode.upper()}] step={step} sent 14 sensor features")
            step += 1
            time.sleep(PUBLISH_RATE)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Publish error: {e}")
            time.sleep(1)

    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    main()