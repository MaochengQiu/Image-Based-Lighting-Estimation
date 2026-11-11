"""Preserve camera-observed directions; never infer HDR highlights from clipped LDR."""
import cv2
import numpy as np
from .data import erp_directions, perspective


def linearize_srgb(image):
    image = np.clip(np.asarray(image, dtype=np.float32), 0, 1)
    return np.where(image <= 0.04045, image / 12.92,
                    ((image + 0.055) / 1.055) ** 2.4)


def constrain_visible(predicted_hdr, input_rgb, hfov_degrees=90, exposure=None):
    prediction = np.asarray(predicted_hdr, dtype=np.float32)
    image = np.asarray(input_rgb, dtype=np.float32)
    if prediction.ndim != 3 or image.ndim != 3 or prediction.shape[2] != 3 or image.shape[2] != 3:
        raise ValueError("Expected H x W x 3 RGB arrays")
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Prediction radiance must be finite and nonnegative")
    if not 0 < hfov_degrees < 180 or (exposure is not None and exposure <= 0):
        raise ValueError("Invalid camera field of view or exposure")
    height, width = image.shape[:2]
    linear = linearize_srgb(image)
    if exposure is None:
        crop = perspective(prediction, (height, width), hfov_degrees, 0)
        valid = (image.max(-1) < 0.97) & (linear.mean(-1) > 0.02)
        ratio = crop.mean(-1)[valid] / linear.mean(-1)[valid]
        gain = float(np.clip(np.median(ratio), 0.001, 1000)) if ratio.size else 1.0
        scale_method = "robust relative scale estimated from raw prediction; not absolute calibration"
    else:
        gain = 1.0 / exposure
        scale_method = "provided linear exposure multiplier"
    rays = erp_directions(*prediction.shape[:2])
    z = rays[..., 2]
    focal = width / (2 * np.tan(np.deg2rad(hfov_degrees) / 2))
    u = rays[..., 0] / np.maximum(z, 1e-6) * focal + (width - 1) / 2
    v = -rays[..., 1] / np.maximum(z, 1e-6) * focal + (height - 1) / 2
    mask = (z > 0) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    edge = np.minimum.reduce([u, width - 1 - u, v, height - 1 - v])
    alpha = (np.clip(edge / 3, 0, 1) * mask).astype(np.float32)[..., None]
    map_u = np.clip(u, -10000, 10000).astype(np.float32)
    map_v = np.clip(v, -10000, 10000).astype(np.float32)
    observed = cv2.remap(linear, map_u, map_v, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE) * gain
    saturated = cv2.remap((image >= 0.98).astype(np.float32), map_u, map_v,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) > 0
    # A clipped observation only supplies a lower bound; do not overwrite a larger HDR prediction.
    observed = np.where(saturated, np.maximum(observed, prediction), observed)
    result = prediction * (1 - alpha) + observed * alpha
    return result.astype(np.float32), {"gain": gain, "scale_method": scale_method,
                                     "observed_pixel_fraction": float(mask.mean()),
                                     "unobserved_directions": "unchanged autoregressive output"}
