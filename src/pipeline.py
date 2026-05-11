from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.background_removal import cv2_available, remove_background
from src.feature_extraction import load_rgb_image
from src.inference import predict_any
from src.screening import ClassProfiles

# Karar eşikleri (percentile — 102 sınıf arasındaki sıralama)
COLOR_REJECT_THRESHOLD = 80.0    # renk çok uzak → kesinlikle bu olamaz
TEXTURE_WARN_THRESHOLD = 75.0    # doku da kötü → muhtemelen değil


@dataclass
class ScreeningResult:
    label: str
    model_score: float         # model olasılığı (0–1)
    color_distance: float      # Bhattacharyya mesafesi
    color_percentile: float    # 0=en iyi, 100=en kötü (102 sınıf içinde)
    texture_distance: float    # Chi-kare mesafesi
    texture_percentile: float  # 0=en iyi, 100=en kötü
    verdict: str               # "olabilir" | "muhtemelen değil" | "kesinlikle bu olamaz"
    bg_removed: bool = True    # GrabCut uygulandı mı

    @property
    def verdict_emoji(self) -> str:
        return {
            "olabilir": "🟢",
            "muhtemelen değil": "🟡",
            "kesinlikle bu olamaz": "🔴",
        }.get(self.verdict, "⚪")

    @property
    def color_match_pct(self) -> float:
        """Renk uyum yüzdesi: 100 = mükemmel, 0 = hiç uymuyor."""
        return max(0.0, 100.0 - self.color_percentile)

    @property
    def texture_match_pct(self) -> float:
        """Doku uyum yüzdesi: 100 = mükemmel, 0 = hiç uymuyor."""
        return max(0.0, 100.0 - self.texture_percentile)


def _determine_verdict(color_pct: float, texture_pct: float) -> str:
    if color_pct > COLOR_REJECT_THRESHOLD:
        return "kesinlikle bu olamaz"
    if texture_pct > TEXTURE_WARN_THRESHOLD:
        return "muhtemelen değil"
    return "olabilir"


def predict_pipeline(
    image_path: str | Path,
    model_path: str | Path,
    profiles_path: str | Path = "models/class_profiles.npz",
    top_k: int = 3,
    image_size: int = 224,
) -> tuple[list[ScreeningResult], np.ndarray, np.ndarray]:
    """
    Tam pipeline: arka plan kaldırma → renk/doku ön eleme → model → karar füzyonu.

    Returns:
        results:      top_k ScreeningResult listesi
        original_rgb: orijinal yeniden boyutlandırılmış RGB görüntü
        masked_rgb:   arka planı kaldırılmış RGB görüntü
    """
    image_path = Path(image_path)
    profiles_path = Path(profiles_path)

    # 1. Orijinal görüntüyü yükle
    original_rgb = load_rgb_image(image_path, image_size=image_size)

    # 2. Arka planı kaldır (cv2 yoksa orijinal görüntü kullanılır)
    masked_rgb, _fg_mask = remove_background(image_path, image_size=image_size)
    bg_removed = cv2_available()

    # 3. Sınıf profillerini yükle
    profiles = ClassProfiles.load(profiles_path)

    # 4. Model tahmini (orijinal görüntüden — model bu şekilde eğitildi)
    model_preds = predict_any(image_path, model_path, top_k=top_k)

    # 5. Arka planı kaldırılmış görüntü üzerinden renk + doku mesafelerini hesapla
    results: list[ScreeningResult] = []
    for pred in model_preds:
        label = pred["label"]
        color_dist, color_pct = profiles.get_color_percentile(masked_rgb, label)
        texture_dist, texture_pct = profiles.get_texture_percentile(masked_rgb, label)
        verdict = _determine_verdict(color_pct, texture_pct)
        results.append(
            ScreeningResult(
                label=label,
                model_score=pred["score"],
                color_distance=color_dist,
                color_percentile=color_pct,
                texture_distance=texture_dist,
                texture_percentile=texture_pct,
                verdict=verdict,
                bg_removed=bg_removed,
            )
        )

    return results, original_rgb, masked_rgb
