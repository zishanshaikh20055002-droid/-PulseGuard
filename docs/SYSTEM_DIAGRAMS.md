# System Diagrams

## 1) High-Level Block Diagram

```mermaid
graph LR
    A[Sensor Data / Simulation] --> B[MQTT Broker]
    B --> C[FastAPI Ingestion Layer]
    C --> D[Preprocessing and Windowing]
    D --> E[ML Inference Engine]
    E --> F[Diagnosis and Alarm Policy]
    F --> G[SQLite Persistence]
    F --> H[WebSocket Live Updates]
    H --> I[Dashboard UI]
    C --> J[Prometheus Metrics]
    J --> K[Grafana Dashboard]
```

## 2) System Architecture Diagram

```mermaid
graph TB
    subgraph Data_Layer[Data Layer]
        S1[Simulation Publisher]
        S2[PLC / Hardware Sensors]
        S3[MQTT Topics]
    end

    subgraph Backend_Layer[Backend and ML Layer]
        B1[FastAPI App]
        B2[MQTT Subscriber]
        B3[Preprocessing Pipeline]
        B4[TFLite / ML Models]
        B5[Diagnostics Engine]
        B6[Alarm Policy]
        B7[SQLite Database]
    end

    subgraph Experience_Layer[User Experience Layer]
        U1[WebSocket Stream]
        U2[Dashboard HTML]
        U3[REST Diagnosis APIs]
        U4[Fleet Overview APIs]
    end

    subgraph Ops_Layer[Observability and Ops]
        O1[Prometheus]
        O2[Grafana]
        O3[Logs and Metrics]
    end

    S1 --> S3
    S2 --> S3
    S3 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> B5
    B5 --> B6
    B5 --> B7
    B5 --> U1
    B5 --> U3
    B5 --> U4
    U1 --> U2
    B1 --> B2
    B1 --> U3
    B1 --> U4
    B1 --> O1
    O1 --> O2
    B1 --> O3
    B2 --> O3
    B5 --> O3
```

## 3) How To Read the Diagrams

- The block diagram shows the main data flow from telemetry to dashboard.
- The architecture diagram shows how the system is split into data, backend, user experience, and observability layers.
- Both diagrams reflect the current implemented system and the next-step hardware path.

## 4) Teacher-Friendly Summary

If asked to explain the architecture in one sentence, say:

"Sensor data enters through MQTT or simulation, gets preprocessed and scored by the ML backend, is turned into diagnosis and alarm outputs, then is exposed through APIs, a live dashboard, and observability tools."
```