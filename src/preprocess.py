import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

FEATURE_COLS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]
TARGET_COL   = 'Machine failure'
WINDOW_SIZE  = 30   # how many past rows the model sees at once

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    return df

def add_rul(df):
    max_wear = df['Tool wear [min]'].max()
    df['RUL'] = max_wear - df['Tool wear [min]']
    print(f"RUL range: {df['RUL'].min()} to {df['RUL'].max()}")
    return df

def add_health_stage(df):
    conditions = [
        df['RUL'] > 100,
        (df['RUL'] > 20) & (df['RUL'] <= 100),
        df['RUL'] <= 20
    ]
    stages = [0, 1, 2]   # 0=healthy  1=warning  2=critical
    df['health_stage'] = np.select(conditions, stages)
    print("Health stage distribution:")
    print(df['health_stage'].value_counts().sort_index())
    return df

def normalize(df):
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    print("Features normalized")
    return df, scaler

def create_windows(df, window_size=WINDOW_SIZE):
    X, y_rul, y_stage = [], [], []

    values  = df[FEATURE_COLS].values
    rul     = df['RUL'].values
    stage   = df['health_stage'].values

    for i in range(len(df) - window_size):
        X.append(values[i : i + window_size])
        y_rul.append(rul[i + window_size])
        y_stage.append(stage[i + window_size])

    X       = np.array(X, dtype=np.float32)
    y_rul   = np.array(y_rul, dtype=np.float32)
    y_stage = np.array(y_stage, dtype=np.int32)

    print(f"Windows created â€” X: {X.shape}, y_rul: {y_rul.shape}")
    return X, y_rul, y_stage

def run_pipeline(data_path):
    df               = load_data(data_path)
    df               = add_rul(df)
    df               = add_health_stage(df)
    df, scaler       = normalize(df)
    X, y_rul, y_stage = create_windows(df)
    return X, y_rul, y_stage, scaler

if __name__ == "__main__":
    import joblib

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    X, y_rul, y_stage, scaler = run_pipeline(
        os.path.join(base, 'data', 'ai4i2020.csv')
    )

    np.save(os.path.join(base, 'data', 'X.npy'), X)
    np.save(os.path.join(base, 'data', 'y_rul.npy'), y_rul)
    np.save(os.path.join(base, 'data', 'y_stage.npy'), y_stage)
    joblib.dump(scaler, os.path.join(base, 'data', 'scaler.pkl'))

    print("\nFinal shapes:")
    print(f"  X       : {X.shape}")
    print(f"  y_rul   : {y_rul.shape}")
    print(f"  y_stage : {y_stage.shape}")
    print("\nSaved to data/ folder")