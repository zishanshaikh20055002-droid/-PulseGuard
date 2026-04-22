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
sensor_voltage = Gauge("sensor_voltage_v", "Estimated line voltage", ["machine_id"])
sensor_current = Gauge("sensor_current_a", "Estimated line current", ["machine_id"])
sensor_power_kw = Gauge("sensor_power_kw", "Estimated electrical power in kW", ["machine_id"])
sensor_vibration = Gauge("sensor_vibration_rms", "Estimated vibration RMS", ["machine_id"])

# Diagnosis-level KPIs
machine_health_index = Gauge("machine_health_index", "Machine health index 0-100", ["machine_id"])
failure_probability = Gauge("machine_failure_probability", "Failure probability from diagnosis engine", ["machine_id"])
time_to_failure_hours = Gauge("machine_time_to_failure_hours", "Estimated time to failure in hours", ["machine_id"])
fault_component_counter = Counter(
	"machine_fault_component_total",
	"Fault localization events by component and severity",
	["machine_id", "component", "severity"],
)
alarm_events = Counter(
    "machine_alarm_events_total",
    "Alarm policy events by machine, level and priority",
    ["machine_id", "level", "priority"],
)

# Industrial hardening metrics
drift_score_gauge = Gauge("fault_drift_score", "Average feature drift z-score")
drift_detected_flag = Gauge("fault_drift_detected", "Drift detection flag (1=drift, 0=normal)")
drift_rows_gauge = Gauge("fault_drift_rows_evaluated", "Rows used by the latest drift evaluation")
auto_retrain_runs = Counter(
	"fault_auto_retrain_runs_total",
	"Auto-retraining run outcomes by status and trigger",
	["status", "trigger"],
)
feedback_relabels_total = Counter(
	"fault_feedback_relabels_total",
	"Human relabel feedback submissions by corrected component and resolution state",
	["component", "resolved"],
)
feedback_pending_gauge = Gauge("fault_feedback_pending", "Pending feedback labels awaiting resolution")
feedback_ready_gauge = Gauge(
	"fault_feedback_ready_for_training",
	"Resolved feedback labels not yet consumed by retraining",
)
telemetry_drop_events = Counter(
    "telemetry_drop_events_total",
    "Dropped telemetry messages by source and reason",
    ["source", "reason"],
)
telemetry_persistence_errors = Counter(
    "telemetry_persistence_errors_total",
    "Telemetry records that failed database persistence by source",
    ["source"],
)
