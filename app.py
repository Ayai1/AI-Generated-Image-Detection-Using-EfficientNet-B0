"""
Streamlit Web Application for AI-Generated Image Forensics & Grad-CAM Visualization.
"""

import streamlit as st
from PIL import Image
import numpy as np
import torch
from src.infer import load_model, predict_image, get_transforms
from src.gradcam import GradCAM, overlay_cam

st.set_page_config(page_title="AI-Generated Image Detector", layout="wide", page_icon="ðŸ”")

st.title("ðŸ” AI-Generated Image Forensics (EfficientNet-B0 + Grad-CAM)")
st.markdown("""
Upload any digital image to evaluate whether it is an authentic capture or AI-synthesized (Midjourney, DALL-E, Stable Diffusion).  
*Trained on 120,000 balanced images with 98.5% holdout accuracy.*
""")

uploaded_file = st.file_uploader("Upload Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

@st.cache_resource
def get_cached_model():
    return load_model("efficientnet_b0_best.pth", device="cpu")

model = get_cached_model()

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    res = predict_image(img, model, device="cpu")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Input Image", use_column_width=True)
        st.metric(
            label="Prediction Result",
            value=res["Prediction"],
            delta=f"{res['Confidence']*100:.1f}% Confidence"
        )
        st.write(f"**Real Probability:** `{res['Real']*100:.2f}%`")
        st.write(f"**AI-Generated Probability:** `{res['AI-Generated']*100:.2f}%`")

    with col2:
        st.subheader("Grad-CAM Explainability Heatmap")
        try:
            target_layer = model.features[-1]
            gradcam = GradCAM(model, target_layer)
            transform = get_transforms()
            input_tensor = transform(img).unsqueeze(0)
            target_class = 1 if res["Prediction"] == "AI-Generated" else 0
            cam = gradcam.generate_heatmap(input_tensor, target_class=target_class)
            overlay = overlay_cam(img, cam)
            st.image(overlay, caption="Attention Heatmap (Regions Influencing Model Decision)", use_column_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM visualization: {e}")
