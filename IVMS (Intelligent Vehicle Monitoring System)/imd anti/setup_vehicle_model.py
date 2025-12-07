import os
import urllib.request

# URLs for MobileNet SSD (Commonly used for fast vehicle detection in OpenCV)
PROTO_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt"
MODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

MODEL_DIR = "models"
PROTO_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "mobilenet_iter_73000.caffemodel")

def download_vehicle_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    if not os.path.exists(PROTO_PATH):
        print(f"Downloading Prototxt from {PROTO_URL}...")
        urllib.request.urlretrieve(PROTO_URL, PROTO_PATH)
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Downloading Caffe Model from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, WEIGHTS_PATH)
        
    print("Vehicle detection models ready.")

if __name__ == "__main__":
    download_vehicle_model()
