"""
Sınıf başına renk ve doku profilleri oluşturur.

Çalıştırma:
    python scripts/build_class_profiles.py

Çıktı:
    models/class_profiles.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_extraction import hsv_histogram, lbp_features, load_rgb_image
from src.screening import ClassProfiles

TRAIN_DIR = ROOT / "data" / "processed" / "oxford102_70_15_15" / "train"
OUT_PATH = ROOT / "models" / "class_profiles.npz"
IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect_images(class_dir: Path) -> list[Path]:
    return [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]


def build_profiles(train_dir: Path, image_size: int = IMAGE_SIZE) -> ClassProfiles:
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"Sınıf klasörü bulunamadı: {train_dir}")

    print(f"Toplam {len(class_dirs)} sınıf bulundu.")

    classes: list[str] = []
    color_means: list[np.ndarray] = []
    texture_means: list[np.ndarray] = []

    for i, class_dir in enumerate(class_dirs):
        images = _collect_images(class_dir)
        if not images:
            print(f"  [UYARI] Boş sınıf atlandı: {class_dir.name}")
            continue

        color_hists: list[np.ndarray] = []
        lbp_hists: list[np.ndarray] = []

        for img_path in images:
            try:
                img = load_rgb_image(img_path, image_size=image_size)
                color_hists.append(hsv_histogram(img))
                lbp_hists.append(lbp_features(img))
            except Exception as exc:
                print(f"  [HATA] {img_path.name}: {exc}")

        if not color_hists:
            continue

        classes.append(class_dir.name)
        color_means.append(np.mean(color_hists, axis=0))
        texture_means.append(np.mean(lbp_hists, axis=0))

        if (i + 1) % 10 == 0 or (i + 1) == len(class_dirs):
            print(f"  {i + 1}/{len(class_dirs)} sınıf işlendi...")

    profiles = ClassProfiles(
        classes=classes,
        color_means=np.stack(color_means),
        texture_means=np.stack(texture_means),
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles.save(OUT_PATH)
    print(f"\nProfiller kaydedildi: {OUT_PATH}")
    print(f"Toplam sınıf: {len(classes)}")
    return profiles


if __name__ == "__main__":
    build_profiles(TRAIN_DIR)
