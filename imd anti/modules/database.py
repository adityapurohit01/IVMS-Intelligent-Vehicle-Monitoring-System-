import sqlite3
import os

DB_PATH = "traffic_v.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Check if table exists to handle migration or recreation
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detections'")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video TEXT,
                frame_no INTEGER,
                timestamp TEXT,
                track_id INTEGER,
                vehicle_type TEXT,
                vehicle_color TEXT,
                plate TEXT,
                confidence REAL,
                x1 INTEGER,
                y1 INTEGER,
                x2 INTEGER,
                y2 INTEGER,
                thumb BLOB
            )
        ''')
    else:
        try:
            cursor.execute("SELECT vehicle_type FROM detections LIMIT 1")
        except sqlite3.OperationalError:
            print("Migrating database schema...")
            cursor.execute("DROP TABLE detections")
            cursor.execute('''
                CREATE TABLE detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video TEXT,
                    frame_no INTEGER,
                    timestamp TEXT,
                    track_id INTEGER,
                    vehicle_type TEXT,
                    vehicle_color TEXT,
                    plate TEXT,
                    confidence REAL,
                    x1 INTEGER,
                    y1 INTEGER,
                    x2 INTEGER,
                    y2 INTEGER,
                    thumb BLOB
                )
            ''')
            
    conn.commit()
    conn.close()

def insert_detection(video, frame_no, timestamp, track_id, v_type, v_color, plate, confidence, x1, y1, x2, y2, thumb):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO detections (video, frame_no, timestamp, track_id, vehicle_type, vehicle_color, plate, confidence, x1, y1, x2, y2, thumb)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (video, frame_no, timestamp, track_id, v_type, v_color, plate, confidence, x1, y1, x2, y2, thumb))
    conn.commit()
    conn.close()

def search_detections(plate_query=None, color_query=None, type_query=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = "SELECT * FROM detections WHERE 1=1"
    params = []
    
    if plate_query:
        query += " AND plate LIKE ?"
        params.append(f"%{plate_query}%")
    
    if color_query and color_query != "Any":
        query += " AND vehicle_color = ?"
        params.append(color_query)
        
    if type_query and type_query != "Any":
        query += " AND vehicle_type = ?"
        params.append(type_query)
        
    query += " ORDER BY frame_no"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_unique_vehicles():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Count unique combinations of video and track_id
    cursor.execute("SELECT COUNT(*) FROM (SELECT DISTINCT video, track_id FROM detections)")
    count = cursor.fetchone()[0]
    conn.close()
    return count

from collections import Counter

def get_full_log():
    """
    Returns a summary of all unique vehicles detected.
    Uses a Voting Mechanism to select the 'best' plate for each vehicle:
    1. Frequency (Mode): The plate string that appears most often.
    2. Length: If ties, prefer the longer string (assuming it's more complete).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get ALL detections
    cursor.execute('SELECT track_id, vehicle_type, vehicle_color, plate, timestamp, confidence, video FROM detections')
    rows = cursor.fetchall()
    conn.close()
    
    # Group by (video, track_id) to ensure uniqueness across different videos
    vehicles = {}
    for r in rows:
        tid = r[0]
        video = r[6]
        key = (video, tid)
        
        if key not in vehicles:
            vehicles[key] = []
        vehicles[key].append(r)
        
    final_log = []
    
    for (video, tid), detections in vehicles.items():
        # 1. Find Best Plate
        plates = [d[3] for d in detections if len(d[3]) > 3] # Filter noise
        if not plates:
            best_plate = detections[0][3] # Fallback
        else:
            # Count frequency
            counts = Counter(plates)
            # Sort by Length (desc), then by Frequency (desc)
            sorted_plates = sorted(counts.items(), key=lambda x: (len(x[0]), x[1]), reverse=True)
            best_plate = sorted_plates[0][0]
            
        # 2. Find Best Color (Mode)
        colors = [d[2] for d in detections]
        best_color = Counter(colors).most_common(1)[0][0]
        
        # 3. Find Best Type (Mode)
        types = [d[1] for d in detections]
        best_type = Counter(types).most_common(1)[0][0]
        
        # 4. Get timestamp of the FIRST detection
        first_detection = min(detections, key=lambda x: x[4])
        
        # 5. Max Confidence
        max_conf = max(d[5] for d in detections)
        
        final_log.append((tid, best_type, best_color, best_plate, first_detection[4], max_conf, video))
        
    return sorted(final_log, key=lambda x: (x[6], x[0])) # Sort by Video then ID


