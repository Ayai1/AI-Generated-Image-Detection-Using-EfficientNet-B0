# AI-Generated-Image-Detection-Using-EfficientNet-B0
AI Generated Image detection using a lightweight, highly accurate deep learning model. Trained on dataset with over 120k images from various sources like DALL-E,Midjourney,SDXL, etc.

🌟

A lightweight, high-accuracy deep learning model for detecting AI-generated vs real images.
Trained on 120,000 images from diverse sources (MidJourney/SDXL/DALLE/camera photos) using EfficientNet-B0, achieving:

- 98.5% accuracy with 120,000-image test set
- Exaplainability using gradcam heatmaps
- GPU for fast inference
- Clean training pipeline optimized for Windows + CUDA

🚀 Features:

-Detects AI-generated vs real images
-Lightweight and efficient EfficientNet-B0 backbone
-Trained on 120,000 images (balanced)
-CUDA-optimized PyTorch training pipeline
-High accuracy & robust performance
-Grad-CAM explainability for visualizing model decisions
-Confusion matrix + classification report included

📊 Results
Accuracy	98.5%
Precision (Real) :	0.99
Precision (Fake) : 0.98
F1 (Real)	: 0.99
F1 (Fake)	: 0.99
Test Images	20,000

Confusion matrix:

<img width="421" height="331" alt="image" src="https://github.com/user-attachments/assets/1e0575b6-289e-46b8-a05b-76657700f661" />

GradCAM Examples:

<img width="466" height="442" alt="image" src="https://github.com/user-attachments/assets/a1de0ce8-d452-4415-8a85-2c62a540157f" />

