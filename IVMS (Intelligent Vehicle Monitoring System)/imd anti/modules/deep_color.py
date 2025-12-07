from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import cv2
import numpy as np

class DeepColorDetector:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        print(f"Loading Deep Color Model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            print("Deep Color Model Loaded Successfully!")
        except Exception as e:
            print(f"Failed to load Deep Color Model: {e}")
            self.model = None

        # Define labels for Zero-Shot Classification
        # We include "car" in the prompt to give context
        self.color_labels = [
            "black car", 
            "white car", 
            "silver grey car", 
            "red car", 
            "blue car", 
            "green car", 
            "yellow car", 
            "orange car",
            "purple car",
            "brown car"
        ]
        
        # Map full labels back to simple color names
        self.label_map = {
            "black car": "Black",
            "white car": "White",
            "silver grey car": "Silver/Grey",
            "red car": "Red",
            "blue car": "Blue",
            "green car": "Green",
            "yellow car": "Yellow",
            "orange car": "Orange",
            "purple car": "Purple",
            "brown car": "Brown"
        }

    def predict(self, image_bgr):
        if self.model is None or image_bgr is None or image_bgr.size == 0:
            return "Unknown"

        try:
            # Convert BGR to RGB and PIL
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)

            # Prepare inputs
            inputs = self.processor(
                text=self.color_labels, 
                images=pil_image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

            # Get the highest probability label
            predicted_idx = probs.argmax().item()
            predicted_label = self.color_labels[predicted_idx]
            confidence = probs[0][predicted_idx].item()
            
            # Optional: Threshold? CLIP is usually good, but if confidence is low, maybe fallback?
            # For now, just return the best match.
            
            return self.label_map[predicted_label]

        except Exception as e:
            print(f"Deep Color Error: {e}")
            return "Unknown"
