from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False


def cv2_available() -> bool:
    return _CV2_AVAILABLE


def remove_background_grabcut(
    image: np.ndarray,
    iterations: int = 5,
    margin_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GrabCut ile arka planı kaldırır.
    cv2 yoksa orijinal görüntü + tam ön plan maskesi döndürür (fallback).

    Returns:
        masked_rgb: Arka plan siyah yapılmış RGB (veya orijinal, cv2 yoksa)
        fg_mask:    0/1 ikili maske (1=ön plan)
    """
    if not _CV2_AVAILABLE:
        return image.copy(), np.ones(image.shape[:2], dtype=np.uint8)

    h, w = image.shape[:2]
    mx = max(1, int(w * margin_ratio))
    my = max(1, int(h * margin_ratio))
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)

    masked_rgb = image.copy()
    masked_rgb[fg_mask == 0] = 0
    return masked_rgb, fg_mask


def remove_background(
    path: str | Path,
    image_size: int = 224,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dosyadan yükle → yeniden boyutlandır → GrabCut uygula (cv2 varsa).

    Returns:
        masked_rgb: uint8 RGB, shape (image_size, image_size, 3)
        fg_mask:    uint8 0/1, shape (image_size, image_size)
    """
    image = np.asarray(
        Image.open(path).convert("RGB").resize((image_size, image_size), Image.LANCZOS),
        dtype=np.uint8,
    )
    return remove_background_grabcut(image, iterations=iterations)
