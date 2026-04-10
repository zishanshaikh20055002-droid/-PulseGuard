import time
import json
import os
import logging
import pandas as pd
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PUBLISHER] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

# Using the complete run-to-failure training file so we have plenty of data
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train_FD001.txt")

def start_publishing():
    client = mqtt.Client(client_id="publisher-M1")
    
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            logger.info(f"Connected to mosquitto:{MQTT_PORT}")
            break
        except Exception as e:
            logger.error(f"Connection failed. Retrying in 3s...")
            time.sleep(3)

    client.loop_start()
    logger.info("Loading NASA CMAPSS dataset...")
    
    columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_measurement_{i}' for i in range(1, 22)]
    df = pd.read_csv(DATA_PATH, sep=r'\s+', names=columns)
    
    # Filter to only Machine 1
    df_m1 = df[df['unit_number'] == 1].copy()
    
    # 14 features used in the CMAPSS model
    features = [f'sensor_measurement_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
    
    # Fast forward to cycle 100 out of 192 to observe active degradation
    start_index = 100
    logger.info(f"Fast-forwarding to step {start_index} to observe active degradation...")
    
    # Convert to a Python list of dictionaries first to completely bypass Pandas index issues
    records = df_m1[features].to_dict('records')
    
    # Slice the clean Python list
    sliced_records = records[start_index:]
    
    step = start_index
    for row in sliced_records:
        # Extract the values from the dictionary in the correct order
        feature_values = [row[feat] for feat in features]
        
        payload = {
            "machine_id": "M1",
            "step": step,
            "features": feature_values
        }
        
        client.publish("sensors/M1/data", json.dumps(payload))
        logger.info(f"[REPLAY] step={step} sent 14 sensor features (CMAPSS)")
        
        step += 1
        time.sleep(1.0)

if __name__ == "__main__":
    start_publishing()