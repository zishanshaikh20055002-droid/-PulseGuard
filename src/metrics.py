from prometheus_client import Gauge, Counter, Histogram, Info

# Core RUL Metrics
rul_gauge = Gauge("machine_rul", "Remaining Useful Life (cycles)", ["machine_id"])

# ADDED: Standard Deviation metric for Grafana confidence bands
rul_std_gauge = Gauge("machine_rul_std", "Standard Deviation of RUL Prediction", ["machine_id"])

health_status_counter = Counter("machine_health_status_total", "Count of health status updates", ["machine_id", "status"])
inference_latency = Histogram("ml_inference_latency_seconds", "Inference latency in seconds", ["machine_id"])

# System and Mode Info
simulation_mode_info = Info("simulation_mode", "Current simulation mode")
simulation_damage_level = Gauge("simulation_damage_level", "Current simulated damage level (0-100)")
ws_active_connections = Gauge("websocket_active_connections", "Number of active WebSocket clients")

# Sensor Metrics
sensor_temperature = Gauge("sensor_process_temperature_kelvin", "Process Temperature", ["machine_id"])
sensor_torque = Gauge("sensor_torque_nm", "Torque", ["machine_id"])
sensor_tool_wear = Gauge("sensor_tool_wear_minutes", "Tool Wear", ["machine_id"])
sensor_speed = Gauge("sensor_rotational_speed_rpm", "Rotational Speed", ["machine_id"])