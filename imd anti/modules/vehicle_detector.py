import cv2
import numpy as np

class VehicleDetector:
    def __init__(self, proto_path="models/deploy.prototxt", model_path="models/mobilenet_iter_73000.caffemodel", conf_threshold=0.5):
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        self.conf_threshold = conf_threshold
        # MobileNet SSD classes
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.VEHICLE_CLASSES = ["bus", "car", "motorbike", "train"]

    def detect(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                idx = int(detections[0, 0, i, 1])
                label = self.CLASSES[idx]
                
                if label in self.VEHICLE_CLASSES:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure within bounds
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    results.append(((startX, startY, endX, endY), label, confidence))
        
        return results

def detect_color(image_crop):
    """
    Robust color detection using K-Means Clustering and Nearest Neighbor Classification.
    """
    if image_crop is None or image_crop.size == 0:
        return "Unknown"

    # 1. Resize for speed (K-Means is slow on large images)
    img = cv2.resize(image_crop, (50, 50))
    
    # 2. Center Crop (Keep 50% center) to remove background
    h, w = img.shape[:2]
    margin_h, margin_w = int(h * 0.25), int(w * 0.25)
    center_crop = img[margin_h:h-margin_h, margin_w:w-margin_w]
    
    # 3. Convert to RGB and flatten
    img_rgb = cv2.cvtColor(center_crop, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # 4. K-Means Clustering
    # We use K=4 to capture: Main Color, Shadow, Highlight, Background/Windshield
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    try:
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    except Exception as e:
        print(f"KMeans Error: {e}")
        return "Unknown"
    
    # 5. Analyze Clusters
    # Count pixels in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    
    # Sort clusters by size (largest first)
    sorted_indices = np.argsort(counts)[::-1]
    
    best_color_name = "Unknown"
    best_dist = float('inf')
    
    # Reference Colors (R, G, B)
    # Tuned for common vehicle shades
    colors_db = {
        "Black": (30, 30, 30),
        "White": (230, 230, 230),
        "Silver/Grey": (128, 128, 128),
        "Red": (200, 20, 20),
        "Blue": (20, 40, 180), # Darker blue
        "Green": (20, 150, 20),
        "Yellow": (220, 220, 20),
        "Orange": (220, 100, 20),
        "Brown": (100, 60, 30),
        "Purple": (100, 20, 100)
    }
    
    # Helper to get saturation
    def get_saturation(rgb):
        mx = max(rgb)
        mn = min(rgb)
        if mx == 0: return 0
        return (mx - mn) / mx

    # Iterate through clusters to find the "best" one
    # We want to avoid "Black" (tires/shadows) or "White" (glare) if there is a prominent color
    
    candidate_colors = []
    
    for i in sorted_indices:
        center = centers[i]
        count = counts[i]
        ratio = count / len(pixels)
        
        # Skip very small clusters
        if ratio < 0.05: continue
        
        r, g, b = center
        
        # Calculate Saturation and Brightness
        sat = get_saturation(center)
        brightness = np.mean(center)
        
        # Heuristics to identify "non-body" parts
        is_very_dark = brightness < 40 # Tires, deep shadow
        is_very_bright = brightness > 240 # Glare
        
        # Classify this cluster
        min_d = float('inf')
        c_name = "Unknown"
        
        for name, ref_rgb in colors_db.items():
            # Euclidean distance
            d = np.linalg.norm(center - np.array(ref_rgb))
            if d < min_d:
                min_d = d
                c_name = name
        
        candidate_colors.append({
            'name': c_name,
            'rgb': center,
            'ratio': ratio,
            'sat': sat,
            'bright': brightness,
            'is_extreme': is_very_dark or is_very_bright
        })
        
    # Decision Logic
    # 1. Prefer Chromatic colors if they are significant
    chromatic = [c for c in candidate_colors if c['sat'] > 0.2 and not c['is_extreme']]
    
    if chromatic:
        # Pick the largest chromatic cluster
        chosen = chromatic[0]
        return chosen['name']
    
    # 2. If no chromatic, check for Silver/Grey (medium brightness, low saturation)
    # We want to distinguish Silver from White/Black
    achromatic = [c for c in candidate_colors if not c['is_extreme']]
    
    if achromatic:
        return achromatic[0]['name']
        
    # 3. Fallback to the largest cluster even if extreme (e.g. a black car)
    if candidate_colors:
        return candidate_colors[0]['name']
        
    return "Unknown"









