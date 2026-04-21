import sqlite3
import csv
import os
import numpy as np
from datetime import datetime

DB_PATH = 'attendance.db'
CSV_PATH = 'attendance.csv'

# Minimum seconds between duplicate CSV rows for the same student
CSV_DEDUP_WINDOW_SEC = 300  # 5 minutes — covers a single class period


def init_db():
    """Initializes the SQLite database with Users and Logs tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            embedding BLOB
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS Logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            camera_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES Users(user_id)
        )
    ''')
    conn.commit()
    conn.close()


def has_embedding(name):
    """
    Check whether a student already has an embedding stored in the DB.
    Returns True if the student exists AND has a non-NULL embedding.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT embedding FROM Users WHERE name = ?', (name,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return False
    return row[0] is not None and len(row[0]) > 0


def save_embedding(name, embedding):
    """Saves a normalized numpy array embedding into the database.
    
    IMPORTANT: This is INSERT-only for new users. If the user already
    exists, this will update their embedding. Use has_embedding() first
    to implement idempotent enrollment.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    embedding_bytes = embedding.astype(np.float32).tobytes()
    try:
        c.execute('INSERT INTO Users (name, embedding) VALUES (?, ?)', (name, embedding_bytes))
    except sqlite3.IntegrityError:
        print(f"  [WARN] User '{name}' already exists. Updating embedding.")
        c.execute('UPDATE Users SET embedding = ? WHERE name = ?', (embedding_bytes, name))
    conn.commit()
    conn.close()


def load_all_embeddings():
    """
    Loads all embeddings into memory for real-time comparison.
    
    Returns a list of dicts:
        [{'user_id': int, 'name': str, 'embedding': np.ndarray}, ...]
    
    The 'name' field is the student's folder name from image_db 
    (e.g. 'sharad', 'aditya') which is used for display and matching.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT user_id, name, embedding FROM Users WHERE embedding IS NOT NULL')
    rows = c.fetchall()
    conn.close()

    users = []
    for row in rows:
        user_id, name, embedding_blob = row
        if embedding_blob is None or len(embedding_blob) == 0:
            continue
        # FP32 embeddings from MobileFaceNet / w600k_mbf
        embedding = np.frombuffer(embedding_blob, dtype=np.float32).copy()
        users.append({'user_id': user_id, 'name': name, 'embedding': embedding})
    return users


def _init_csv():
    """Create the CSV file with headers if it doesn't exist."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Date', 'Time'])


def _is_duplicate_csv_entry(name, now):
    """
    Check the tail of the CSV to see if this student was logged
    within the last CSV_DEDUP_WINDOW_SEC seconds.
    
    This prevents the CSV from bloating with repeated entries
    for the same student during a single class session.
    """
    if not os.path.exists(CSV_PATH):
        return False

    try:
        with open(CSV_PATH, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception:
        return False

    # Walk backwards through the CSV (skip header)
    for row in reversed(rows[1:]):
        if len(row) < 3:
            continue
        csv_name, csv_date, csv_time = row[0], row[1], row[2]
        try:
            csv_dt = datetime.strptime(f"{csv_date} {csv_time}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        # Only check entries for the same student
        if csv_name != name:
            continue

        delta = (now - csv_dt).total_seconds()
        if delta < CSV_DEDUP_WINDOW_SEC:
            return True  # duplicate — too recent
        else:
            break  # older entry found, no need to keep searching

    return False


def log_attendance(user_id, camera_id, student_name=None):
    """
    Logs attendance to BOTH the SQLite Logs table AND attendance.csv.
    
    Args:
        user_id: The SQLite user_id (integer).
        camera_id: The camera/source that recognised the student.
        student_name: The display name for CSV logging. If None,
                      we look it up from the DB.
    """
    # --- SQLite log (unchanged) ---
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO Logs (user_id, camera_id) VALUES (?, ?)', (user_id, camera_id))
    conn.commit()

    # --- Resolve name if not provided ---
    if student_name is None:
        c.execute('SELECT name FROM Users WHERE user_id = ?', (user_id,))
        row = c.fetchone()
        student_name = row[0] if row else f"user_{user_id}"
    conn.close()

    # --- CSV log with dedup throttle ---
    _init_csv()
    now = datetime.now()

    if _is_duplicate_csv_entry(student_name, now):
        return  # silently skip — already logged recently

    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            student_name,
            now.strftime('%Y-%m-%d'),
            now.strftime('%H:%M:%S')
        ])


if __name__ == "__main__":
    init_db()
    _init_csv()
    print("Database and CSV initialized successfully.")
