import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "machine_data.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        machine_id TEXT,
        RUL REAL,
        status TEXT,
        temperature REAL,
        air_temperature REAL,
        torque REAL,
        tool_wear REAL,
        speed REAL
    )
    """)

    conn.commit()
    conn.close()

def insert_data(data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (
        machine_id, RUL, status,
        temperature, air_temperature,
        torque, tool_wear, speed
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["machine_id"],
        data["RUL"],
        data["status"],
        data["temperature"],
        data["air_temperature"],
        data["torque"],
        data["tool_wear"],
        data["speed"]
    ))

    conn.commit()
    conn.close()