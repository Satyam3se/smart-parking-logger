import sqlite3
import os
import math
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(__file__), "parking.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            duration_minutes REAL DEFAULT 0,
            fee_charged REAL DEFAULT 0.0,
            status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'EXITED')),
            cropped_plate_path TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_entry(plate_number: str, cropped_plate_path: Optional[str] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Logs vehicle entry.
    Returns (success, message, record_dict).
    """
    clean_plate = plate_number.strip().upper()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if vehicle is already actively parked
    cursor.execute("""
        SELECT * FROM parking_logs
        WHERE plate_number = ? AND status = 'ACTIVE'
        ORDER BY id DESC LIMIT 1
    """, (clean_plate,))
    existing = cursor.fetchone()

    if existing:
        conn.close()
        return False, f"Vehicle {clean_plate} is already inside (Checked in at {existing['entry_time']}).", dict(existing)

    entry_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO parking_logs (plate_number, entry_time, status, cropped_plate_path)
        VALUES (?, ?, 'ACTIVE', ?)
    """, (clean_plate, entry_time_str, cropped_plate_path))
    conn.commit()
    inserted_id = cursor.lastrowid

    cursor.execute("SELECT * FROM parking_logs WHERE id = ?", (inserted_id,))
    record = cursor.fetchone()
    conn.close()

    return True, f"Vehicle {clean_plate} successfully logged in at {entry_time_str}.", dict(record)

def calculate_fee(duration_minutes: float, rate_per_hour: float = 40.0, min_fee: float = 20.0) -> float:
    """
    Standard parking fee calculation for India (INR / ₹):
    e.g. ₹20.00 / 30 mins -> ₹40.00/hr, with a minimum flat fee of ₹20.00.
    Billed in 30-minute intervals.
    """
    if duration_minutes <= 0:
        return min_fee
    
    # Half-hour chunks (30 min blocks)
    half_hours = math.ceil(duration_minutes / 30.0)
    fee = half_hours * (rate_per_hour / 2.0)
    return round(max(min_fee, fee), 2)

def log_exit(plate_number: str, rate_per_hour: float = 40.0, min_fee: float = 20.0) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Logs vehicle exit, calculates duration and fees, and marks status as EXITED.
    Returns (success, message, record_dict).
    """
    clean_plate = plate_number.strip().upper()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Find active entry record
    cursor.execute("""
        SELECT * FROM parking_logs
        WHERE plate_number = ? AND status = 'ACTIVE'
        ORDER BY id DESC LIMIT 1
    """, (clean_plate,))
    existing = cursor.fetchone()

    if not existing:
        conn.close()
        return False, f"No active entry record found for vehicle {clean_plate}.", None

    entry_dt = datetime.strptime(existing["entry_time"], "%Y-%m-%d %H:%M:%S")
    exit_dt = datetime.now()
    exit_time_str = exit_dt.strftime("%Y-%m-%d %H:%M:%S")

    duration_minutes = max(1.0, round((exit_dt - entry_dt).total_seconds() / 60.0, 1))
    fee = calculate_fee(duration_minutes, rate_per_hour=rate_per_hour, min_fee=min_fee)

    cursor.execute("""
        UPDATE parking_logs
        SET exit_time = ?, duration_minutes = ?, fee_charged = ?, status = 'EXITED'
        WHERE id = ?
    """, (exit_time_str, duration_minutes, fee, existing["id"]))
    conn.commit()

    cursor.execute("SELECT * FROM parking_logs WHERE id = ?", (existing["id"],))
    updated_record = cursor.fetchone()
    conn.close()

    msg = f"Vehicle {clean_plate} exited. Duration: {duration_minutes:.1f} mins. Total Fee: ₹{fee:.2f}"
    return True, msg, dict(updated_record)

def get_lot_summary(total_capacity: int = 50) -> Dict[str, Any]:
    """
    Retrieves live metrics: occupied, available, today's revenue, active vehicles count.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS active_count FROM parking_logs WHERE status = 'ACTIVE'")
    active_count = cursor.fetchone()["active_count"]

    today_str = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT COALESCE(SUM(fee_charged), 0.0) AS today_revenue
        FROM parking_logs
        WHERE status = 'EXITED' AND exit_time LIKE ?
    """, (f"{today_str}%",))
    today_revenue = cursor.fetchone()["today_revenue"]

    cursor.execute("SELECT COUNT(*) AS total_exits_today FROM parking_logs WHERE status = 'EXITED' AND exit_time LIKE ?", (f"{today_str}%",))
    exits_today = cursor.fetchone()["total_exits_today"]

    conn.close()

    available = max(0, total_capacity - active_count)
    return {
        "total_capacity": total_capacity,
        "occupied": active_count,
        "available": available,
        "today_revenue": round(today_revenue, 2),
        "exits_today": exits_today
    }

def get_all_records(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetches records with optional status filter ('ACTIVE', 'EXITED', or None).
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    if status_filter and status_filter in ('ACTIVE', 'EXITED'):
        cursor.execute("SELECT * FROM parking_logs WHERE status = ? ORDER BY id DESC", (status_filter,))
    else:
        cursor.execute("SELECT * FROM parking_logs ORDER BY id DESC")

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
