# Slide-Ready System Diagrams

This file contains presentation-friendly Mermaid diagrams with short labels and clean flow for PPT/defense usage.

## Slide 1: End-to-End Block Diagram

```mermaid
graph LR
    A[Machine Data<br/>Simulation or PLC] --> B[MQTT Broker]
    B --> C[Ingestion and Preprocessing]
    C --> D[ML Inference]
    D --> E[Diagnosis and Alarm Policy]
    E --> F[APIs and WebSocket]
    F --> G[Live Dashboard]
    E --> H[SQLite Storage]
    E --> I[Prometheus Metrics]
    I --> J[Grafana Monitoring]
```

Speaker cue:
- "Data comes in, model predicts, diagnosis decides urgency, and results are shown live with full monitoring."

## Slide 2: Layered System Architecture

```mermaid
graph TB
    subgraph L1[Data Sources]
        S1[Simulation Publisher]
        S2[PLC and Sensors]
        S3[External Datasets]
    end

    subgraph L2[Streaming and Backend]
        B1[MQTT Broker]
        B2[FastAPI App]
        B3[Subscriber Service]
        B4[Preprocessing and Windows]
    end

    subgraph L3[Intelligence Layer]
        M1[RUL and Health Model]
        M2[Fault Localization]
        M3[Alarm Policy Engine]
    end

    subgraph L4[Delivery Layer]
        D1[REST APIs]
        D2[WebSocket Stream]
        D3[Dashboard UI]
        D4[Fleet and Diagnosis APIs]
    end

    subgraph L5[Ops and Governance]
        O1[JWT and RBAC]
        O2[Rate Limiting]
        O3[SQLite Persistence]
        O4[Prometheus]
        O5[Grafana]
    end

    S1 --> B1
    S2 --> B1
    B1 --> B3
    B3 --> B4
    B4 --> M1
    M1 --> M2
    M2 --> M3
    M3 --> D1
    M3 --> D2
    D2 --> D3
    D1 --> D4
    M3 --> O3
    B2 --> O1
    B2 --> O2
    B2 --> O4
    O4 --> O5
    S3 -. training pipeline .-> M1
```

Speaker cue:
- "The system is layered from data to intelligence to delivery, with security and observability as first-class concerns."

## Slide 3: Model and Decision Flow

```mermaid
flowchart TD
    X[Sensor Window] --> Y[Model Outputs]
    Y --> Y1[RUL]
    Y --> Y2[Failure Probability]
    Y --> Y3[Fault Component Scores]

    Y1 --> Z[Health Stage]
    Y2 --> P[Alarm Threshold Logic]
    Y3 --> Q[Top Component and Fault Type]

    Z --> R[Maintenance Priority]
    P --> R
    Q --> R

    R --> U[API Response and Dashboard]
```

Speaker cue:
- "We combine predictive outputs with policy logic to produce operational decisions, not just model scores."

## Slide 4: Fleet Monitoring Flow

```mermaid
graph LR
    A1[M1] --> F[Fleet Aggregator]
    A2[M2] --> F
    A3[M3] --> F
    A4[Mn] --> F

    F --> B1[Rank by Alarm Pressure]
    F --> B2[Show Latest Diagnosis]
    F --> B3[Recommend Priority Queue]

    B1 --> C[Operator Dashboard]
    B2 --> C
    B3 --> C
```

Speaker cue:
- "Fleet mode helps teams decide which machine to inspect first based on risk and urgency."

## Slide 5: Continuous Improvement Loop

```mermaid
graph LR
    A[Live Predictions] --> B[Operator Feedback and Relabel]
    B --> C[Feedback Store]
    C --> D[Retraining Trigger]
    D --> E[Model Update]
    E --> F[Runtime Reload]
    F --> A
```

Speaker cue:
- "The system learns from real operator feedback, enabling continuous improvement over time."

## Fast PPT Usage Tips

- Use one diagram per slide.
- Keep animation simple: reveal left to right.
- Add one sentence under each diagram explaining value.
- End with Slide 5 to show this is a living system, not a static model.
