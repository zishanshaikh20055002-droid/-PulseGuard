"""
schemas.py â€” request/response models with strict validation.

Every field has:
  - Type enforcement (Pydantic rejects wrong types automatically)
  - Range constraints where physically meaningful
  - Custom validators for business logic
  - Sanitization before the value reaches any other code
"""

import re
from typing import Literal, Annotated
from pydantic import BaseModel, Field, field_validator, model_validator



Temperature  = Annotated[float, Field(ge=250.0,  le=400.0,  description="Temperature in Kelvin (250â€“400 K)")]
Torque       = Annotated[float, Field(ge=0.0,    le=100.0,  description="Torque in Nm (0â€“100)")]
ToolWear     = Annotated[float, Field(ge=0.0,    le=300.0,  description="Tool wear in minutes (0â€“300)")]
RotSpeed     = Annotated[float, Field(ge=0.0,    le=3000.0, description="Rotational speed in rpm (0â€“3000)")]



class SensorReading(BaseModel):
    """
    A single row of sensor data sent to the /predict endpoint.
    All physical ranges are based on the ai4i2020 dataset bounds.
    """

    model_config = {"str_strip_whitespace": True}   # auto-strip all strings

    machine_id:      str
    air_temperature: Temperature
    temperature:     Temperature
    rotational_speed: RotSpeed
    torque:          Torque
    tool_wear:       ToolWear

    @field_validator("machine_id")
    @classmethod
    def validate_machine_id(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.fullmatch(r"M\d{1,3}", v):
            raise ValueError("machine_id must match pattern M1â€“M999 (e.g. 'M1', 'M42')")
        return v

    @model_validator(mode="after")
    def process_temp_must_exceed_air(self) -> "SensorReading":
        """Process temperature must be higher than air temperature."""
        if self.temperature <= self.air_temperature:
            raise ValueError(
                f"process temperature ({self.temperature}) must be greater "
                f"than air temperature ({self.air_temperature})"
            )
        return self



class BatchSensorReading(BaseModel):
    readings: list[SensorReading] = Field(min_length=1, max_length=100)



class ModeRequest(BaseModel):
    """
    Used when mode is passed in a request body instead of a path param.
    Accepts only the three valid mode strings â€” nothing else.
    """
    mode: Literal["normal", "degrade", "failure"]



class LoginRequest(BaseModel):
    """
    Validates login credentials before they hit the auth logic.
    Enforces length limits to block oversized payloads.
    """
    model_config = {"str_strip_whitespace": True}

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("username")
    @classmethod
    def username_safe(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.fullmatch(r"[a-z0-9_\-]+", v):
            raise ValueError("username may only contain letters, numbers, _ and -")
        return v



class PredictionResponse(BaseModel):
    machine_id: str
    RUL:        float
    status:     Literal["HEALTHY", "WARNING", "CRITICAL"]
    step:       int
    temperature:     float
    air_temperature: float
    torque:          float
    tool_wear:       float
    speed:           float
    voltage:         float
    current:         float
    power_kw:        float
    vibration:       float
    efficiency:      float
    health_index:    float
    failure_probability: float
    time_to_failure_hours: float
    fault_component: str
    fault_type:      str
    fault_severity:  str
    fault_confidence: float
    probable_causes: list[str]
    recommended_actions: list[str]
    fault_model_source: str
    fault_model_version: str
    fault_component_probabilities: dict[str, float]
    alarm_level: str
    maintenance_priority: str
    alarm_reasons: list[str]
    recommended_window_hours: float