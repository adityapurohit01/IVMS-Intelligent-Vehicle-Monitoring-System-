import cv2
import numpy as np
import os
import urllib.request

# We will use a ResNet-based classifier trained on vehicle colors.
# Since training one from scratch requires a dataset, we will use a pre-trained ONNX model if available,
# or simulate a robust deep learning feature extractor using a small MobileNet classifier.

# For this implementation, we will use a lightweight pre-trained model if possible.
# However, finding a direct public link for a "Vehicle Color ResNet ONNX" is rare.
# Instead, we will implement a "Deep Feature + KNN" approach or a robust Histogram-based approach 
# which is often used when a specific deep model isn't available.

# BUT, to be "State of the Art" without training, we can use a trick:
# We can use the MobileNet we already have (for detection) to extract features? No, that's for detection.

# Let's build a robust HSV-Histogram Classifier which is very fast and accurate for this specific task
# if we don't have a 1GB PyTorch model.
# OR, we can use a small custom CNN model structure and load weights if we had them.

# Since I cannot download a 100MB ResNet file easily without a direct link, 
# I will implement the "Spatial Center-Weighted HSV Histogram" method which is 
# significantly better than K-Means for vehicles because it encodes texture/finish.

# Actually, let's stick to the K-Means but with a "Smart Mask".
# The previous K-Means was good, but we can improve it by using a "Saliency Map".

# IMPROVED APPROACH: Saliency-based Color Extraction
# We use OpenCV's Saliency API to find the "most interesting" part of the car (usually the body),
# ignoring the road and background.

def get_vehicle_color_saliency(image):
    if image is None or image.size == 0: return "Unknown"
    
    # 1. Resize
    img = cv2.resize(image, (128, 128))
    
    # 2. Saliency Map (Fine Grained)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    
    if success:
        # Binarize saliency map
        _, mask = cv2.threshold(saliencyMap * 255, 50, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        
        # 3. Apply mask to image
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        
        # 4. Calculate mean color of salient region
        # Convert to HSV
        hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
        
        # Only consider non-zero pixels
        pixels = hsv[mask > 0]
        
        if len(pixels) > 0:
            mean_hsv = np.mean(pixels, axis=0)
            return classify_hsv(mean_hsv)
            
    # Fallback to center crop if saliency fails
    return classify_hsv(np.mean(cv2.cvtColor(img[32:96, 32:96], cv2.COLOR_BGR2HSV), axis=(0,1)))

def classify_hsv(hsv):
    H, S, V = hsv
    
    if S < 40:
        if V < 50: return "Black"
        if V > 200: return "White"
        return "Silver"
        
    if V < 30: return "Black"
    
    if H < 10: return "Red"
    if H < 25: return "Orange"
    if H < 35: return "Yellow"
    if H < 85: return "Green"
    if H < 130: return "Blue"
    if H < 145: return "Purple"
    if H < 170: return "Pink"
    return "Red"
