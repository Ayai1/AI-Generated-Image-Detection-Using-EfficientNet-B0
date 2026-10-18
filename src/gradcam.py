"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for EfficientNet-B0.
Extracts gradient activations from the final convolutional block to visualize spatial evidence.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=1):
        self.model.eval()
        output = self.model(input_tensor)
        
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze(0)

        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        if np.max(cam) != np.min(cam):
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        else:
            cam = np.zeros_like(cam)
            
        return cam

def overlay_cam(img: Image.Image, cam: np.ndarray, alpha=0.5):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cam_resized = cv2.resize(cam, (img_cv.shape[1], img_cv.shape[0]))
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 1 - alpha, cam_heatmap, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
