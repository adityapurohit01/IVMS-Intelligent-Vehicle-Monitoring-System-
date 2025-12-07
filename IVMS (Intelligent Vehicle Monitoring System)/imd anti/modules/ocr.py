from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
import numpy as np

class TrOCREngine:
    def __init__(self, model_name="microsoft/trocr-base-printed"):
        print(f"Loading TrOCR model: {model_name}...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def recognize(self, plate_bgr):
        if plate_bgr is None or plate_bgr.size == 0:
            return ""
        
        try:
            img_rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            pixel_values = self.processor(images=pil, return_tensors="pt").pixel_values.to(self.device)
            
            with torch.no_grad():
                generated = self.model.generate(pixel_values, max_length=16)
            
            text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
            return text.strip().upper()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
