"""EigenCAM heatmap generation for the RGB stream backbone."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import torch

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_heatmap(fusion_model: torch.nn.Module, input_tensor: torch.Tensor, face_crop_bgr: np.ndarray) -> str | None:
    """Return a base64-encoded JPEG of an EigenCAM heatmap overlaid on face_crop_bgr.

    Args:
        fusion_model: FusionModel instance with a .rgb_model attribute.
        input_tensor: (1, 3, 224, 224) ImageNet-normalised float32 tensor.
        face_crop_bgr: (H, W, 3) uint8 BGR numpy array (the raw face crop).

    Returns:
        base64 string of JPEG, or None if generation fails.
    """
    try:
        rgb_model = fusion_model.rgb_model
        # EfficientNet-B4 structure inside RGBStreamResNet:
        #   backbone = Sequential(features, avgpool)
        #   backbone[0] = features Sequential (9 blocks, [-1] = final 1792-ch conv)
        target_layers = [rgb_model.backbone[0][-1]]

        cam = EigenCAM(model=rgb_model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor)[0]  # (7, 7) → upsampled to input size

        rgb_image = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if rgb_image.shape[:2] != grayscale_cam.shape:
            rgb_image = cv2.resize(rgb_image, (grayscale_cam.shape[1], grayscale_cam.shape[0]))

        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")
    except Exception:
        return None
