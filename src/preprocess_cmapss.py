"""
preprocess_cmapss.py â€” downloads and preprocesses the NASA CMAPSS FD001 dataset.

Key differences from ai4i2020:
  - Real turbofan engine run-to-failure data (100 engines)
  - Piecewise linear RUL: flat at RUL_MAX until degradation starts,
    then linear countdown â€” industry standard approach
  - 14 informative sensors selected (7 constant sensors removed)
  - Normalisation per-sensor across all engines

Run standalone:
    python -m src.preprocess_cmapss
"""

import os
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

CMAPSS_URL = "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip"
RAW_FILE   = os.path.join(DATA_DIR, "CMAPSSData.zip")

RUL_MAX     = 125
WINDOW_SIZE = 30

COLS = (
    ["unit", "cycle"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

CONSTANT_SENSORS = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}

FEATURE_COLS = [c for c in COLS[5:] if c not in CONSTANT_SENSORS]

NUM_FEATURES = len(FEATURE_COLS)   # 14


def download_cmapss():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(os.path.join(DATA_DIR, "train_FD001.txt")):
        print("âœ… CMAPSS data already downloaded")
        return

    print(f"Downloading CMAPSS dataset from NASA...")
    print(f"  URL: {CMAPSS_URL}")

    try:
        req = urllib.request.Request(
            CMAPSS_URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()

        print(f"  Downloaded {len(data) / 1024:.1f} KB")

        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if name.endswith(".txt"):
                    z.extract(name, DATA_DIR)
                    print(f"  Extracted: {name}")

        print("âœ… CMAPSS dataset ready")

    except Exception as e:
        print(f"\nâŒ Auto-download failed: {e}")
        print("\nManual download steps:")
        print("  1. Go to: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        print("  2. Download the ZIP file")
        print(f"  3. Extract train_FD001.txt and test_FD001.txt to: {DATA_DIR}")
        raise


def load_fd001(split: str = "train") -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{split}_FD001.txt")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute piecewise linear RUL per engine:
      - Find max cycle for each engine (failure point)
      - RUL = max_cycle - current_cycle  (raw remaining cycles)
      - Cap at RUL_MAX: engines are "new" until within RUL_MAX of failure

    This avoids the artificial assumption that degradation starts at t=0.
    """
    max_cycles = df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]
    df = df.merge(max_cycles, on="unit")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=RUL_MAX)  # piecewise cap
    df.drop("max_cycle", axis=1, inplace=True)
    return df


def add_health_stage(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        df["RUL"] > 80,
        (df["RUL"] > 30) & (df["RUL"] <= 80),
        df["RUL"] <= 30
    ]
    df["health_stage"] = np.select(conditions, [0, 1, 2])
    return df


def create_windows(df: pd.DataFrame, scaler=None):
    """
    Create sliding windows per engine.
    Each engine's sequence is windowed independently
    so windows don't span across engines.
    """
    X, y_rul, y_stage = [], [], []

    for unit_id, group in df.groupby("unit"):
        group = group.sort_values("cycle")
        values = group[FEATURE_COLS].values
        rul    = group["RUL"].values
        stage  = group["health_stage"].values

        if len(group) < WINDOW_SIZE:
            continue

        for i in range(len(group) - WINDOW_SIZE):
            X.append(values[i : i + WINDOW_SIZE])
            y_rul.append(rul[i + WINDOW_SIZE])
            y_stage.append(stage[i + WINDOW_SIZE])

    X       = np.array(X, dtype=np.float32)
    y_rul   = np.array(y_rul, dtype=np.float32)
    y_stage = np.array(y_stage, dtype=np.int32)

    return X, y_rul, y_stage


def run_pipeline():
    
    # 2. Safety check: Ensure manual download was completed
    train_file = os.path.join(DATA_DIR, "train_FD001.txt")
    if not os.path.exists(train_file):
        print("\nâŒ Dataset not found locally!")
        print("Please manually download the NASA CMAPSS dataset from Kaggle:")
        print("URL: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        print(f"Extract 'train_FD001.txt' directly into this folder: {DATA_DIR}")
        return None, None, None, None

    print("\nLoading FD001 training data...")
    df = load_fd001("train")
    print(f"  Engines: {df['unit'].nunique()} | Rows: {len(df)}")

    df = add_rul(df)
    df = add_health_stage(df)

    print(f"\nRUL distribution:")
    print(f"  min={df['RUL'].min():.0f}  max={df['RUL'].max():.0f}  "
          f"mean={df['RUL'].mean():.1f}")
    print(f"\nHealth stage distribution:")
    print(df["health_stage"].value_counts().sort_index().to_string())

    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    print(f"\nCreating sliding windows (size={WINDOW_SIZE})...")
    X, y_rul, y_stage = create_windows(df, scaler)

    print(f"\nFinal shapes:")
    print(f"  X       : {X.shape}")
    print(f"  y_rul   : {y_rul.shape}")
    print(f"  y_stage : {y_stage.shape}")

    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "X_cmapss.npy"),       X)
    np.save(os.path.join(DATA_DIR, "y_rul_cmapss.npy"),   y_rul)
    np.save(os.path.join(DATA_DIR, "y_stage_cmapss.npy"), y_stage)
    joblib.dump(scaler, os.path.join(DATA_DIR, "scaler_cmapss.pkl"))

    print(f"\nâœ… Saved to {DATA_DIR}/")
    print(f"   X_cmapss.npy, y_rul_cmapss.npy, y_stage_cmapss.npy, scaler_cmapss.pkl")

    return X, y_rul, y_stage, scaler


if __name__ == "__main__":
    run_pipeline()