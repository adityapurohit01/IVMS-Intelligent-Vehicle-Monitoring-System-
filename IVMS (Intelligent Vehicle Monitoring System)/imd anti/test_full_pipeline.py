import cv2
import numpy as np
import os
from modules.vehicle_detector import VehicleDetector, detect_color
from modules.detector import EASTDetectorHighLevel
from modules.ocr import TrOCREngine
from modules.utils import create_test_video, crop_rotated_rect
from modules.database import init_db, insert_detection, search_detections

def test_full_pipeline():
    print("1. Generating Test Video...")
    video_path = create_test_video()
    
    print("2. Loading Models...")
    v_detector = VehicleDetector()
    t_detector = EASTDetectorHighLevel()
    ocr_engine = TrOCREngine()
    print("   Models loaded.")
    
    print("3. Processing Video...")
    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    
    init_db()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_no += 1
        if frame_no % 10 == 0: print(f"   Frame {frame_no}")
        
        # Detect Vehicles
        vehicles = v_detector.detect(frame)
        for (vx1, vy1, vx2, vy2), label, v_conf in vehicles:
            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            color = detect_color(vehicle_crop)
            
            # Detect Plate
            plate_detections = t_detector.detect(vehicle_crop)
            if plate_detections:
                (rect, p_conf) = plate_detections[0]
                plate_img = crop_rotated_rect(vehicle_crop, rect)
                text = ocr_engine.recognize(plate_img)
                
                if text:
                    print(f"   FOUND: {label} ({color}) - Plate: {text}")
                    insert_detection("test.mp4", frame_no, "00:00:01", 0, label, color, text, float(p_conf), 0, 0, 0, 0, None)

    cap.release()
    print("4. Done.")
    
    print("5. Testing Search...")
    results = search_detections(color_query="White") # Test video has white background/plate
    print(f"   Search found {len(results)} records.")

if __name__ == "__main__":
    test_full_pipeline()
