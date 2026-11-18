import sqlite3
import json
from pathlib import Path
from datetime import datetime, date

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "smartface.db"

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con

def initialize():
    with connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            attendance_date TEXT NOT NULL,
            check_in TEXT NOT NULL,
            method TEXT NOT NULL DEFAULT 'face',
            FOREIGN KEY(employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE,
            UNIQUE(employee_id, attendance_date)
        );
        """)

def add_employee(employee_id, name, embedding):
    with connect() as con:
        con.execute(
            "INSERT INTO employees(employee_id,name,embedding,created_at) VALUES(?,?,?,?)",
            (employee_id.strip(), name.strip(), json.dumps(embedding.tolist()), datetime.now().isoformat(timespec="seconds"))
        )

def get_employees():
    with connect() as con:
        rows = con.execute("SELECT employee_id,name,embedding,created_at FROM employees ORDER BY name").fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["embedding"] = json.loads(item["embedding"])
        result.append(item)
    return result

def delete_employee(employee_id):
    with connect() as con:
        con.execute("DELETE FROM employees WHERE employee_id=?", (employee_id,))

def mark_attendance(employee_id):
    now = datetime.now()
    with connect() as con:
        cur = con.execute(
            """INSERT OR IGNORE INTO attendance(employee_id,attendance_date,check_in,method)
               VALUES(?,?,?,'face')""",
            (employee_id, now.date().isoformat(), now.isoformat(timespec="seconds"))
        )
        return cur.rowcount == 1

def get_attendance(day=None):
    day = day or date.today().isoformat()
    with connect() as con:
        rows = con.execute("""
            SELECT a.employee_id, e.name, a.attendance_date, a.check_in, a.method
            FROM attendance a JOIN employees e ON e.employee_id=a.employee_id
            WHERE a.attendance_date=? ORDER BY a.check_in DESC
        """, (day,)).fetchall()
        return [dict(r) for r in rows]

def dashboard_counts():
    with connect() as con:
        employees = con.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
        today = con.execute("SELECT COUNT(*) FROM attendance WHERE attendance_date=?", (date.today().isoformat(),)).fetchone()[0]
        return employees, today
