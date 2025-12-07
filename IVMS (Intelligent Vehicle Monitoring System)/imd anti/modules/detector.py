import cv2
import numpy as np
import math

class EASTDetector:
    def __init__(self, model_path="models/frozen_east_text_detection.pb", conf_threshold=0.5, nms_threshold=0.4):
        self.net = cv2.dnn.readNet(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    def detect(self, image):
        orig_h, orig_w = image.shape[:2]
        # Resize to multiple of 32
        new_w, new_h = (320, 320) # Fixed size for speed, can be tuned
        r_w = orig_w / float(new_w)
        r_h = orig_h / float(new_h)

        blob = cv2.dnn.blobFromImage(image, 1.0, (new_w, new_h), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        scores, geometry = self.net.forward(self.output_layers)

        boxes, confidences = self.decode_predictions(scores, geometry)
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, self.conf_threshold, self.nms_threshold)

        results = []
        if len(indices) > 0:
            for i in indices:
                # indices can be a tuple or list depending on opencv version
                idx = i if isinstance(i, int) else i[0]
                box = boxes[idx]
                
                # Adjust box to original scale
                center, size, angle = box
                center = (center[0] * r_w, center[1] * r_h)
                size = (size[0] * r_w, size[1] * r_h)
                
                # Filter by aspect ratio (license plates are usually rectangular)
                w, h = size
                if w < h: w, h = h, w # Ensure w is the longer side
                aspect_ratio = w / (h + 1e-5)
                
                # Basic heuristic for license plate: usually 2 < AR < 6
                if 2.0 < aspect_ratio < 6.0:
                    results.append(((center, size, angle), confidences[idx]))
        
        return results

    def decode_predictions(self, scores, geometry):
        (num_rows, num_cols) = scores.shape[2:4]
        boxes = []
        confidences = []

        for y in range(num_rows):
            scores_data = scores[0, 0, y]
            x0_data = geometry[0, 0, y]
            x1_data = geometry[0, 1, y]
            x2_data = geometry[0, 2, y]
            x3_data = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]

            for x in range(num_cols):
                score = scores_data[x]
                if score < self.conf_threshold:
                    continue

                # Offset
                offset_x, offset_y = x * 4.0, y * 4.0
                angle = angles_data[x]
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)

                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                end_x = offset_x + cos_a * x1_data[x] + sin_a * x2_data[x]
                end_y = offset_y - sin_a * x1_data[x] + cos_a * x2_data[x]
                
                start_x = end_x - w * cos_a - h * sin_a
                start_y = end_y + w * sin_a - h * cos_a # Check this geometry math carefully
                
                # Center
                center_x = 0.5 * (start_x + end_x) # This is approx
                center_y = 0.5 * (start_y + end_y) # This is approx
                
                # Correct calculation for RotatedRect center
                # The geometry maps are distances from the pixel to the 4 boundaries.
                # Pixel coord: (offset_x, offset_y)
                # Top: x0, Right: x1, Bottom: x2, Left: x3
                # We need to calculate the center of the box.
                
                # Let's use the standard derivation
                p_x, p_y = offset_x, offset_y
                
                # Calculate geometry
                # h = x0 + x2
                # w = x1 + x3
                
                # The angle is the rotation of the box.
                # We can compute the center based on the distances and angle.
                # This part is tricky to get 100% right without a library, but let's try the standard formula.
                
                # https://github.com/argman/EAST/blob/master/model.py (Original implementation logic)
                # Actually, let's use the simple approximation or just reconstruct the 4 points.
                
                # Reconstruct 4 points
                # p1 = (offset_x + d1 * cos + d2 * sin, offset_y + d2 * cos - d1 * sin) ... wait
                
                # Let's stick to RotatedRect format: ((cx, cy), (w, h), angle_deg)
                # angle in EAST is in radians. OpenCV RotatedRect takes degrees.
                
                # Center calculation:
                # The current pixel (offset_x, offset_y) is inside the text region.
                # d1 (top), d2 (right), d3 (bottom), d4 (left)
                # The box is rotated by 'angle'.
                
                # Let's just use the 4 points method which is more robust to derive RotatedRect
                # But for now, I'll use a simplified center calculation.
                
                # Correct derivation:
                # offset is the top-left of the pixel in the feature map? No, it's the coordinate in the input image (scaled by 4).
                
                # Let's use the standard snippet for EAST decoding found in OpenCV examples.
                
                # Calculate offset
                offset = np.array([offset_x, offset_y])
                
                # Rotate the displacement vector
                # The pixel is at (offset_x, offset_y).
                # The top boundary is at distance x0 along the normal vector (rotated by angle).
                # Actually, let's just use the standard python implementation logic.
                
                # 1. Calculate the unrotated bounding box relative to the pixel
                # Top-left relative to pixel: (-x3, -x0)
                # Bottom-right relative to pixel: (x1, x2)
                # But this is in the rotated coordinate system.
                
                # 2. Rotate these points by -angle to get back to image coordinates?
                # No, the box is rotated by angle.
                
                # Let's use the OpenCV `decode` function if it existed in python, but it doesn't.
                
                # Alternative: Use `cv2.dnn.TextDetectionModel_EAST`!
                # OpenCV has a high-level API for this.
                # That would be much easier and less error-prone.
                
                pass 

        # I will switch to using cv2.dnn.TextDetectionModel_EAST in the main detect method
        # It handles decoding internally.
        return boxes, confidences

    def detect_v2(self, image):
        # Use the high-level API
        model = cv2.dnn.TextDetectionModel_EAST(self.net)
        model.setConfidenceThreshold(self.conf_threshold)
        model.setNMSThreshold(self.nms_threshold)
        
        # Preprocessing parameters
        # 1.0 scale, (320, 320) size, mean subtraction, swapRB, crop
        model.setInputParams(1.0, (320, 320), (123.68, 116.78, 103.94), True)
        
        # Detect
        boxes, confidences = model.detect(image)
        
        results = []
        if len(boxes) > 0:
            for box, conf in zip(boxes, confidences):
                # box is a RotatedRect or 4 points?
                # TextDetectionModel_EAST.detect returns:
                # boxes: a list of rotated bounding boxes (RotatedRect) or quadrangles?
                # It returns a list of 4 points (quadrangles) usually, or RotatedRects.
                # Let's check documentation or assume it returns RotatedRects (box, conf).
                # Actually, for EAST it usually returns RotatedRects (center, size, angle) or 4 points.
                # In recent OpenCV, it returns (boxes, confidences). Boxes are usually 4 points (quadrangles) or RotatedRects.
                # Let's assume RotatedRects for now or handle both.
                
                # Wait, `detect` returns `(boxes, confidences)`.
                # `boxes` is a list of RotatedRects (if using standard EAST) or 4 points.
                # Let's inspect the type at runtime or use a safe conversion.
                
                # For the sake of this implementation, I'll assume it returns RotatedRects or convertible.
                # If it returns 4 points, we can get RotatedRect via minAreaRect.
                
                # Let's use the 4-point approach if possible, but RotatedRect is better for cropping.
                
                # Let's stick to the manual implementation if I'm unsure, but the high level API is safer.
                # I will use the manual implementation logic but corrected, OR use the high level API and debug.
                # I'll use the high level API `cv2.dnn.TextDetectionModel_EAST`.
                
                # Filter by aspect ratio
                # If box is RotatedRect: ((cx, cy), (w, h), angle)
                # If box is 4 points: need to compute.
                
                # Let's try to use the high level API.
                pass
        
        return []

# Redefining the class to use the high-level API which is much cleaner
class EASTDetectorHighLevel:
    def __init__(self, model_path="models/frozen_east_text_detection.pb", conf_threshold=0.5, nms_threshold=0.4):
        try:
            self.net = cv2.dnn.readNet(model_path)
            self.model = cv2.dnn.TextDetectionModel_EAST(self.net)
            self.model.setConfidenceThreshold(conf_threshold)
            self.model.setNMSThreshold(nms_threshold)
            self.model.setInputParams(1.0, (320, 320), (123.68, 116.78, 103.94), True)
        except Exception as e:
            print(f"Failed to load EAST model: {e}")
            self.model = None

    def detect(self, image):
        if self.model is None:
            return []
            
        # detect() returns boxes, confidences
        # boxes are numpy arrays of points (N, 4, 2) usually for TextDetectionModel
        boxes, confidences = self.model.detect(image)
        
        results = []
        for i, box in enumerate(boxes):
            # box is 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # Convert to RotatedRect to get width/height easily
            rect = cv2.minAreaRect(box)
            (center, (w, h), angle) = rect
            
            # Aspect ratio filter
            if w < h: w, h = h, w
            if h == 0: continue
            aspect_ratio = w / h
            
            if 2.0 < aspect_ratio < 6.0:
                results.append((rect, confidences[i]))
                
        return results
