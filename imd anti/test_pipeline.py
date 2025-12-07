import cv2
import os
from modules.detector import EASTDetectorHighLevel
from modules.ocr import TrOCREngine
from modules.utils import create_test_video, crop_rotated_rect
from modules.database import init_db, insert_detection, get_detections

def test_pipeline():
    print("1. Generating Test Video...")
    video_path = create_test_video()
    print(f"   Video created at {video_path}")

    print("2. Loading Models...")
    detector = EASTDetectorHighLevel()
    ocr = TrOCREngine()
    print("   Models loaded.")

    print("3. Processing Video...")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detections_count = 0
    
    init_db()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"   Processing frame {frame_count}...")

        detections = detector.detect(frame)
        
        for (rect, conf) in detections:
            plate_crop = crop_rotated_rect(frame, rect)
            text = ocr.recognize(plate_crop)
            
            if text:
                print(f"   [Frame {frame_count}] Detected: {text} (Conf: {conf:.2f})")
                detections_count += 1
                insert_detection("test_video.mp4", frame_count, 0, text, float(conf), 0, 0, 0, 0, None)

    cap.release()
    print(f"4. Processing Complete. Total Frames: {frame_count}, Detections: {detections_count}")

    print("5. Verifying Database...")
    rows = get_detections("test_video.mp4")
    print(f"   Database contains {len(rows)} records for this video.")
    
    if detections_count > 0:
        print("SUCCESS: Pipeline is working!")
    else:
        print("WARNING: No detections. Check model or video.")

if __name__ == "__main__":
    test_pipeline()
