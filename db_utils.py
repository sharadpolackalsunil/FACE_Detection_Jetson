import sqlite3
import numpy as np

DB_PATH = 'attendance.db'

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

def save_embedding(name, embedding):
    """Saves a normalized numpy array embedding into the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    embedding_bytes = embedding.tobytes()
    try:
        c.execute('INSERT INTO Users (name, embedding) VALUES (?, ?)', (name, embedding_bytes))
    except sqlite3.IntegrityError:
        print(f"User {name} already exists. Updating embedding.")
        c.execute('UPDATE Users SET embedding = ? WHERE name = ?', (embedding_bytes, name))
    conn.commit()
    conn.close()

def load_all_embeddings():
    """Loads all embeddings into memory for real-time comparison."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT user_id, name, embedding FROM Users')
    rows = c.fetchall()
    conn.close()
    
    users = []
    for row in rows:
        user_id, name, embedding_blob = row
        # Assuming FP32 bindings from the models
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        users.append({'user_id': user_id, 'name': name, 'embedding': embedding})
    return users

def log_attendance(user_id, camera_id):
    """Logs the attendance record for a recognized user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO Logs (user_id, camera_id) VALUES (?, ?)', (user_id, camera_id))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
