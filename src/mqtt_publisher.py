"""
mqtt_publisher.py — simulates an edge device publishing sensor data.

Two modes (set via PUBLISHER_MODE env var or MQTT control topic):
  replay  — reads from X.npy, inverse-transforms to real sensor values
  random  — generates random values within realistic physical ranges

Run standalone:  python -m src.mqtt_publisher
Docker service:  see docker-compose.yml
"""

import json
import os
import time
import logging
import random
import numpy as np
import joblib
import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PUBLISHER] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MQTT_BROKER   = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT     = int(os.getenv("MQTT_PORT", "1883"))
MACHINE_ID    = os.getenv("MACHINE_ID", "M1")
PUBLISH_TOPIC = f"sensors/{MACHINE_ID}/data"
CONTROL_TOPIC = "sensors/control/mode"
PUBLISH_RATE  = float(os.getenv("PUBLISH_RATE", "1.0"))

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "X.npy")
SCALER_PATH = os.path.join(BASE_DIR, "data", "scaler.pkl")

# Realistic ranges from ai4i2020 dataset
SENSOR_RANGES = {
    "air_temperature":  (295.0, 304.0),
    "temperature":      (305.0, 313.0),
    "rotational_speed": (1168.0, 2886.0),
    "torque":           (3.8, 76.6),
    "tool_wear":        (0.0, 253.0),
}

current_mode = os.getenv("PUBLISHER_MODE", "replay")


def generate_random(step: int) -> dict:
    air = random.uniform(*SENSOR_RANGES["air_temperature"])
    return {
        "machine_id":      MACHINE_ID,
        "step":            step,
        "air_temperature": round(air, 3),
        "temperature":     round(air + random.uniform(8.0, 12.0), 3),
        "rotational_speed": round(random.uniform(*SENSOR_RANGES["rotational_speed"]), 1),
        "torque":          round(random.uniform(*SENSOR_RANGES["torque"]), 2),
        "tool_wear":       round(random.uniform(*SENSOR_RANGES["tool_wear"]), 1),
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
                reading = {
                    "machine_id":      MACHINE_ID,
                    "step":            step,
                    "air_temperature": round(float(orig[0]), 3),
                    "temperature":     round(float(orig[1]), 3),
                    "rotational_speed": round(float(orig[2]), 1),
                    "torque":          round(float(orig[3]), 2),
                    "tool_wear":       round(float(orig[4]), 1),
                }
                replay_idx += 1
            else:
                reading = generate_random(step)

            result = client.publish(PUBLISH_TOPIC, json.dumps(reading), qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(
                    f"[{current_mode.upper()}] step={step} "
                    f"temp={reading['temperature']:.1f}K "
                    f"torque={reading['torque']:.1f}Nm "
                    f"wear={reading['tool_wear']:.1f}min"
                )
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