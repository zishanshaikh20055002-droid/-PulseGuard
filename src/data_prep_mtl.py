import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
WINDOW_SIZE = 30

def calculate_rul(df):
    """Reverse-engineers RUL by counting backwards from failure points."""
    rul = []
    current_rul = 150 # Max cap for RUL
    
    # Iterate backwards through the dataset
    for failure in reversed(df['Machine failure'].values):
        if failure == 1:
            current_rul = 0
        else:
            current_rul = min(current_rul + 1, 150)
        rul.append(current_rul)
        
    df['RUL'] = rul[::-1] # Reverse it back to normal chronological order
    return df

def prepare_mtl_data():
    print("Loading AI4I dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Calculate the RUL target
    df = calculate_rul(df)
    
    # 2. Extract the 5 core features
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    X_raw = df[features].values
    
    # 3. Extract the 5 specific fault targets
    fault_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    y_faults_raw = df[fault_cols].values
    y_rul_raw = df['RUL'].values
    
    # 4. Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Save the scaler for the FastApi subscriber later
    joblib.dump(scaler, os.path.join(BASE_DIR, "models", "scaler_mtl.pkl"))
    
    print(f"Building {WINDOW_SIZE}-step sliding windows...")
    X_windows, y_rul_windows, y_faults_windows = [], [], []
    
    for i in range(len(X_scaled) - WINDOW_SIZE):
        X_windows.append(X_scaled[i : i + WINDOW_SIZE])
        y_rul_windows.append(y_rul_raw[i + WINDOW_SIZE])
        y_faults_windows.append(y_faults_raw[i + WINDOW_SIZE])
        
    X = np.array(X_windows, dtype=np.float32)
    y_rul = np.array(y_rul_windows, dtype=np.float32)
    y_faults = np.array(y_faults_windows, dtype=np.float32)
    
    # The anomaly detector target is just the input window itself!
    y_recon = X.copy()
    
    print(f"Data shapes: X: {X.shape}, RUL: {y_rul.shape}, Faults: {y_faults.shape}")
    
    # Save the prepared arrays
    np.save(os.path.join(BASE_DIR, "data", "X_mtl.npy"), X)
    np.save(os.path.join(BASE_DIR, "data", "y_rul_mtl.npy"), y_rul)
    np.save(os.path.join(BASE_DIR, "data", "y_faults_mtl.npy"), y_faults)
    np.save(os.path.join(BASE_DIR, "data", "y_recon_mtl.npy"), y_recon)
    print("✅ Multi-Task Data preparation complete!")

if __name__ == "__main__":
    prepare_mtl_data()