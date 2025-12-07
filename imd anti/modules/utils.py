import cv2
import numpy as np
import os

def create_test_video(filename="test_video.mp4", duration=5, fps=30):
    if os.path.exists(filename):
        return filename
        
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    # Create a simple animation
    # A white plate moving from left to right
    plate_w, plate_h = 120, 40
    text = "ABC1234"
    
    for i in range(duration * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate position
        x = int((width - plate_w) * (i / (duration * fps)))
        y = height // 2
        
        # Draw plate
        cv2.rectangle(frame, (x, y), (x + plate_w, y + plate_h), (255, 255, 255), -1)
        
        # Draw text (simulating a plate)
        cv2.putText(frame, text, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        out.write(frame)
        
    out.release()
    return filename

def crop_rotated_rect(image, rect, scale=1.2):
    # rect is ((cx, cy), (w, h), angle)
    center, size, angle = rect
    width, height = size
    
    # Expand the box by scale factor to ensure we don't cut off text
    width *= scale
    height *= scale
    
    # Rotate the image around the center of the rect
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rot = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    # Extract the rectangle
    x, y = int(center[0] - width / 2), int(center[1] - height / 2)
    w, h = int(width), int(height)
    
    if x < 0: x = 0
    if y < 0: y = 0
    
    crop = img_rot[y:y+h, x:x+w]
    return crop


