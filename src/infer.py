"""
Standalone Inference Module for EfficientNet-B0 Deepfake / AI-Generated Image Detection.
Loads trained weights, performs preprocessing, and outputs class probabilities.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model(weights_path: str = "efficientnet_b0_best.pth", device: str = "cpu"):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # Fallback dummy initialization for testing
        pass
        
    model.to(device)
    model.eval()
    return model

def predict_image(image: Image.Image, model, device: str = "cpu"):
    transform = get_transforms()
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    # Index 0: Real, Index 1: AI-Generated
    return {
        "Real": float(probs[0]),
        "AI-Generated": float(probs[1]),
        "Prediction": "AI-Generated" if probs[1] > 0.5 else "Real",
        "Confidence": float(max(probs))
    }
