"""
example_plc_integration.py

Example: End-to-end integration of domain adaptation + PLC hardware bridge.

Usage:
    # Start PLC streaming with MQTT publishing
    python src/example_plc_integration.py --protocol modbus --host 192.168.1.100 --port 502
    
    # Or with file polling (for demos/testing)
    python src/example_plc_integration.py --protocol file --file-path /tmp/sensor_data.json
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.domain_adaptation import (
    build_domain_adapted_mtl_model,
    DomainAdaptationCallback,
    mmd_loss,
)
from src.plc_bridge import (
    PLCConfig,
    PLCStreamingBuffer,
    create_plc_connection,
)
from src.sensor_contract import RealSensorPacket, canonicalize_feature_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mqtt_publish_fn(mqtt_client):
    """
    Factory function to create MQTT publish callback.
    
    Example usage in PLCStreamingBuffer:
        buffer = PLCStreamingBuffer(plc_connection, mqtt_publisher_fn=mqtt_publish_fn(client))
    """
    def publish(update: dict) -> None:
        try:
            machine_id = update.get("machine_id", "M1")
            feature = update.get("feature", "unknown")
            value = update.get("value", 0.0)
            
            topic = f"sensors/{machine_id}/feature/{feature}"
            payload = json.dumps({"value": value, "timestamp": time.time()})
            
            mqtt_client.publish(topic, payload, qos=1)
        except Exception as e:
            logger.error(f"MQTT publish error: {e}")
    
    return publish


def on_sensor_update_fn(model: tf.keras.Model):
    """
    Factory function to create sensor callback for real-time inference.
    
    Called whenever a new sensor batch is received from PLC.
    """
    def on_reading(packet: RealSensorPacket, updates: list) -> None:
        try:
            # Accumulate sensor values into inference window
            logger.info(f"Received update from {packet.machine_id}: {len(updates)} features")
            
            # Example: In production, buffer these into Windows and run periodic inference
            # model_output = model.predict([X_window])
            # Publish diagnosis to MQTT
        except Exception as e:
            logger.error(f"Inference callback error: {e}")
    
    return on_reading


def example_domain_adaptation_training():
    """
    Example: Train multimodal model with domain adaptation on multisource data.
    """
    print("\n" + "="*60)
    print("DOMAIN ADAPTATION TRAINING EXAMPLE")
    print("="*60)
    
    # Load pre-trained base model
    model_path = "models/best_multimodal_mtl.keras"
    if not os.path.exists(model_path):
        print(f"❌ Base model not found at {model_path}")
        print("   Run: python -m src.train_multimodal_mtl --epochs 30")
        return
    
    print(f"✅ Loading base model from {model_path}")
    base_model = tf.keras.models.load_model(model_path)
    
    # Wrap with domain adaptation layers
    try:
        adapted_model, feature_extractor = build_domain_adapted_mtl_model(
            base_model,
            num_domains=5,  # AI4I, CWRU, MIMII, MetroPT-3, Edge-IIoT
            use_mmd=True,
            use_dann=True,
        )
        print("✅ Added domain adaptation components (MMD + DANN + Gradient Reversal)")
    except Exception as e:
        print(f"❌ Domain adaptation build error: {e}")
        return
    
    # Compile with domain losses
    adapted_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=[
            "mse",  # RUL task
            "binary_crossentropy",  # Faults task
            "binary_crossentropy",  # Anomaly task
            "categorical_crossentropy",  # Domain task
        ],
        loss_weights=[1.0, 2.0, 1.0, 0.5],  # Domain loss gets lower weight
        metrics={
            0: ["mae"],
            1: ["accuracy"],
            2: ["auc"],
        }
    )
    
    # Add progressive unfreezing callback
    callback = DomainAdaptationCallback(
        frozen_layers=[
            "process_conv1", "process_conv2", "process_bilstm",
            "vibration_conv1", "vibration_conv2", "vibration_bilstm",
            # ... other encoder layers
        ],
        total_epochs=20,
    )
    
    print("\n✅ Domain-adapted model ready for training:")
    print("   - Shared encoders for all modalities")
    print("   - Domain discriminator with gradient reversal")
    print("   - MMD loss for invariant representations")
    print("   - Progressive unfreezing to reduce negative transfer")
    print("\n   To train: adapted_model.fit(..., callbacks=[callback])")


def example_plc_streaming():
    """
    Example: Stream sensor data from PLC and run real-time diagnostics.
    """
    print("\n" + "="*60)
    print("PLC STREAMING EXAMPLE")
    print("="*60)
    
    # Create demo sensor data file for testing
    demo_data = {
        "temperature": 310.5,
        "torque": 45.2,
        "speed": 2500,
        "vibration": 0.145,
        "tool_wear": 12,
    }
    
    demo_file = Path("/tmp/edge_ai_sensor_demo.json")
    demo_file.parent.mkdir(exist_ok=True)
    with open(demo_file, "w") as f:
        json.dump(demo_data, f)
    
    print(f"✅ Created demo sensor file: {demo_file}")
    
    # Configure PLC connection (file-based for demo)
    config = PLCConfig(
        protocol="file",
        host="localhost",
        port=502,
        machine_id="M1",
        poll_interval_ms=500,
        file_path=str(demo_file),
    )
    
    print(f"✅ PLC Config: {config.protocol} from {config.file_path}")
    
    # Create PLC connection
    try:
        plc_conn = create_plc_connection(config)
        print(f"✅ Created {plc_conn.__class__.__name__}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    # Create streaming buffer (without real MQTT for demo)
    def demo_callback(packet: RealSensorPacket, updates: list):
        logger.info(f"📊 Received: {packet.machine_id} @ {packet.timestamp}")
        for upd in updates:
            logger.info(f"  - {upd['feature']}: {upd['value']}")
    
    buffer = PLCStreamingBuffer(
        plc_conn,
        mqtt_publisher_fn=None,  # Skip MQTT in demo
        on_reading=demo_callback,
    )
    
    # Start streaming for demo
    print("\n✅ Starting PLC stream (will collect 5 readings)...")
    if buffer.start():
        readings_collected = 0
        while readings_collected < 5 and buffer.running:
            time.sleep(1.0)
            buffered = buffer.plc.get_buffered_readings()
            readings_collected += len(buffered)
            logger.info(f"Buffered readings so far: {readings_collected}")
        
        buffer.stop()
        print(f"\n✅ Collected {readings_collected} sensor readings")
    else:
        print("❌ Failed to start PLC stream")


def main():
    parser = argparse.ArgumentParser(description="PLC + Domain Adaptation Integration Examples")
    parser.add_argument("--example", choices=["domain_adaptation", "plc_streaming", "both"], default="both")
    parser.add_argument("--protocol", default="file", help="PLC protocol: modbus, opcua, mqtt, file")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=502)
    parser.add_argument("--file-path", default="/tmp/edge_ai_sensor_demo.json")
    
    args = parser.parse_args()
    
    if args.example in ["domain_adaptation", "both"]:
        example_domain_adaptation_training()
    
    if args.example in ["plc_streaming", "both"]:
        example_plc_streaming()
    
    print("\n" + "="*60)
    print("✅ EXAMPLES COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
