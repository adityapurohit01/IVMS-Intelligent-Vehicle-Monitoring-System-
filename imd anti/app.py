import asyncio
import sys

# Fix for Windows asyncio loop policy
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import time
import datetime

# Set page config FIRST
st.set_page_config(page_title="IVMS - Intelligent Vehicle Monitoring System", layout="wide")

# Import modules
from modules.detector import EASTDetectorHighLevel
from modules.ocr import TrOCREngine
from modules.vehicle_detector import VehicleDetector, detect_color
from modules.deep_color import DeepColorDetector
from modules.database import init_db, insert_detection, search_detections, get_unique_vehicles
from modules.utils import create_test_video, crop_rotated_rect
from modules.tracker import CentroidTracker

st.title("IVMS: Intelligent Vehicle Monitoring System")

# Sidebar
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5)
use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=False)
use_deep_color = st.sidebar.checkbox("Use Deep Learning Color (Slower)", value=False)

# Initialize DB
init_db()

if st.sidebar.button("Clear Database History"):
    import sqlite3
    conn = sqlite3.connect('traffic_v.db')
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS detections")
    conn.commit()
    conn.close()
    init_db()
    st.sidebar.success("Database Cleared!")
    st.rerun()


# Load Models with Caching to prevent reload
@st.cache_resource
def load_models(use_deep=False):
    # Load Vehicle Detector
    v_detector = VehicleDetector()
    # Load Text Detector
    t_detector = EASTDetectorHighLevel(conf_threshold=0.5, nms_threshold=0.4)
    # Load OCR
    ocr_engine = TrOCREngine()
    
    # Load Deep Color if requested
    deep_color_model = None
    if use_deep:
        deep_color_model = DeepColorDetector()
        
    return v_detector, t_detector, ocr_engine, deep_color_model

try:
    v_detector, t_detector, ocr_engine, deep_color_model = load_models(use_deep_color)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["Live Processing", "Search & Analytics"])

with tab1:
    st.header("Video Processing")
    input_source = st.radio("Select Input Source", ["Upload Video", "Use Test Video"])

    video_path = None
    if "video_name" not in st.session_state:
        st.session_state.video_name = None

    if input_source == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            st.session_state.video_name = uploaded_file.name
    elif input_source == "Use Test Video":
        if st.button("Generate & Use Test Video"):
            video_path = create_test_video()
            st.session_state.video_name = "test_video.mp4"
            st.success(f"Test video generated: {video_path}")
            
    # Use session state for DB name
    video_name_for_db = st.session_state.video_name


    if video_path:
        if st.button("Start Processing"):
            cap = cv2.VideoCapture(video_path)

            st_frame = st.empty()
            st_stats = st.empty()
            st_progress = st.progress(0)

            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 30
            
            frame_no = 0
            
            # Initialize Tracker
            tracker = CentroidTracker(maxDisappeared=10, maxDistance=100)
            
            # Keep track of which IDs we have already OCR'd successfully to avoid re-OCR
            processed_ids = set()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_no += 1
                timestamp = str(datetime.timedelta(seconds=int(frame_no/fps)))
                
                # 1. Detect Vehicles
                vehicles = v_detector.detect(frame)
                
                # Prepare rects for tracker
                rects = []
                vehicle_data = [] # Store (box, label, color)
                
                for (vx1, vy1, vx2, vy2), label, v_conf in vehicles:
                    rects.append((vx1, vy1, vx2, vy2))
                    
                    # Crop & Color
                    vehicle_crop = frame[vy1:vy2, vx1:vx2]
                    
                    if use_deep_color and deep_color_model:
                        color = deep_color_model.predict(vehicle_crop)
                    else:
                        color = detect_color(vehicle_crop)
                        
                    vehicle_data.append({'box': (vx1, vy1, vx2, vy2), 'label': label, 'color': color, 'crop': vehicle_crop})
                
                # Update Tracker
                objects = tracker.update(rects)
                
                # Match objects to vehicle data
                # This is a simple heuristic matching since tracker returns IDs and centroids
                # We need to map back to the boxes to draw and OCR
                
                current_vehicle_count = len(objects)
                
                for (objectID, centroid) in objects.items():
                    # Find the closest vehicle box to this centroid
                    best_match = None
                    min_dist = 99999
                    
                    for v in vehicle_data:
                        (vx1, vy1, vx2, vy2) = v['box']
                        cx = (vx1 + vx2) // 2
                        cy = (vy1 + vy2) // 2
                        dist = np.linalg.norm(np.array([cx, cy]) - centroid)
                        if dist < min_dist:
                            min_dist = dist
                            best_match = v
                    
                    if best_match and min_dist < 50:
                        # Draw ID
                        text_label = f"ID {objectID} | {best_match['label']}"
                        cv2.putText(frame, text_label, (best_match['box'][0], best_match['box'][1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.rectangle(frame, (best_match['box'][0], best_match['box'][1]), 
                                      (best_match['box'][2], best_match['box'][3]), (0, 255, 0), 2)
                        
                        # OCR Logic: Run frequently (every 2 frames) to build a consensus on the best plate
                        if objectID not in processed_ids or frame_no % 2 == 0:

                            # Run Plate Detection on Vehicle Crop
                            plate_detections = t_detector.detect(best_match['crop'])
                            
                            if plate_detections:
                                (rect, p_conf) = plate_detections[0]
                                plate_img = crop_rotated_rect(best_match['crop'], rect, scale=1.3)
                                plate_text = ocr_engine.recognize(plate_img)

                                
                                if plate_text and len(plate_text) > 3:
                                    # Store Result
                                    insert_detection(
                                        video=video_name_for_db,
                                        frame_no=frame_no,
                                        timestamp=timestamp,
                                        track_id=objectID,
                                        v_type=best_match['label'],
                                        v_color=best_match['color'],
                                        plate=plate_text,
                                        confidence=float(p_conf),
                                        x1=best_match['box'][0], y1=best_match['box'][1],
                                        x2=best_match['box'][2], y2=best_match['box'][3],
                                        thumb=None
                                    )
                                    
                                    processed_ids.add(objectID)
                                    
                                    # Draw Plate
                                    cv2.putText(frame, plate_text, (best_match['box'][0], best_match['box'][1] - 30), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Update Stats
                st_stats.markdown(f"**Frame:** {frame_no}/{total_frames} | **Time:** {timestamp} | **Active Vehicles:** {len(objects)}")
                
                # Display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB")
                
                if total_frames > 0:
                    st_progress.progress(min(frame_no / total_frames, 1.0))
            
            cap.release()
            st.success("Processing Complete!")

with tab2:
    st.header("Search & Analytics")
    
    # Summary Metrics
    total_unique = get_unique_vehicles()
    st.metric("Total Unique Vehicles Detected", total_unique)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        search_plate = st.text_input("Search Number Plate")
    with col2:
        search_color = st.selectbox("Filter by Color", ["Any", "Red", "Blue", "Green", "White", "Black", "Grey", "Yellow"])
    with col3:
        search_type = st.selectbox("Filter by Type", ["Any", "car", "bus", "motorbike", "truck"])
        
    # Logic: If search is clicked OR if no search is active, show something
    # But user wants to see the list.
    
    # Get all unique data first
    from modules.database import get_full_log
    all_data = get_full_log()
    
    # Filter in Python (easier for this list view) or use SQL
    filtered_data = []
    
    # Get current video name from session state
    # We rely on session state because video_path might be None if using Test Video and switching tabs
    
    current_video_name = st.session_state.get("video_name", None)

        
    for r in all_data:
        # r = (track_id, type, color, plate, timestamp, conf, video)
        
        # FILTER: Only show results for the CURRENT video if one is selected/processed
        if current_video_name and r[6] != current_video_name:
            continue
            
        # Check filters
        if search_plate and search_plate.upper() not in r[3].upper(): continue
        if search_color != "Any" and search_color != r[2]: continue
        if search_type != "Any" and search_type != r[1]: continue
        
        filtered_data.append({
            "Vehicle ID": r[0],
            "Type": r[1],
            "Color": r[2],
            "Plate Number": r[3],
            "Timestamp": r[4],
            "Confidence": f"{r[5]:.2f}",
            # "Video Source": r[6] # Hide source if we are only showing current
        })
    
    if filtered_data:
        st.subheader(f"Vehicle Log ({len(filtered_data)})")
        st.dataframe(filtered_data, use_container_width=True)
    else:
        if current_video_name:
            st.info(f"No vehicles found in {current_video_name} matching criteria.")
        else:
            st.info("No video processed yet or no matches.")



