"""
CLIP tabanlı zero-shot çiçek/değil ön filtresi.
Resim yüklendiğinde CNN'e girmeden önce "bu çiçek mi?" kontrolü yapar.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image


_MODEL = None
_PREPROCESS = None
_TOKENIZER = None
_TEXT_FEATURES = None
_DEVICE = None

# Pozitif/negatif promptlar — average alınır
_POSITIVE_PROMPTS = [
    "a photo of a flower",
    "a close-up photo of a flower",
    "a flower bloom",
    "a flowering plant",
    "a bouquet of flowers",
]
_NEGATIVE_PROMPTS = [
    "a photo of a person",
    "a photo of a car",
    "a photo of a building",
    "a photo of an animal",
    "a photo of food",
    "a photo of furniture",
    "a screenshot of text",
    "a photo of an empty landscape",
]


def _ensure_loaded() -> None:
    global _MODEL, _PREPROCESS, _TOKENIZER, _TEXT_FEATURES, _DEVICE
    if _MODEL is not None:
        return
    import open_clip
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _MODEL, _, _PREPROCESS = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    _MODEL = _MODEL.to(_DEVICE).eval()
    _TOKENIZER = open_clip.get_tokenizer("ViT-B-32")

    with torch.no_grad():
        pos = _TOKENIZER(_POSITIVE_PROMPTS).to(_DEVICE)
        neg = _TOKENIZER(_NEGATIVE_PROMPTS).to(_DEVICE)
        pos_features = _MODEL.encode_text(pos)
        neg_features = _MODEL.encode_text(neg)
        pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
        neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
        _TEXT_FEATURES = {
            "positive": pos_features.mean(dim=0),
            "negative": neg_features.mean(dim=0),
        }


def is_flower(image: Image.Image | Path | str, threshold: float = 0.15) -> tuple[bool, float]:
    """
    Resmin çiçek olup olmadığını döndürür.

    Args:
        image: PIL Image, Path veya string path
        threshold: Pozitif − negatif skor farkı eşiği. > threshold → çiçek

    Returns:
        (is_flower: bool, score_diff: float)
    """
    _ensure_loaded()

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    with torch.no_grad():
        x = _PREPROCESS(image).unsqueeze(0).to(_DEVICE)
        img_features = _MODEL.encode_image(x)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        img_features = img_features.squeeze(0)

        pos_sim = float(img_features @ _TEXT_FEATURES["positive"])
        neg_sim = float(img_features @ _TEXT_FEATURES["negative"])

    diff = pos_sim - neg_sim
    return diff > threshold, diff
