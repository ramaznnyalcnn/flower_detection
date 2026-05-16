"""
Sınıf başına renk ve doku profilleri oluşturur.

Profiller, inference sırasında kullanılan GrabCut maskelenmiş görüntüler üzerinden
hesaplanır — böylece profil/inference uyumsuzluğu (HSV histogramını siyah arka
plan piksellerinin bozması) ortadan kalkar.

Ayrıca her sınıf için renk ve doku mesafelerinin ortalama/std değerleri
hesaplanır → inference'ta adaptive threshold (mean + 2σ) için kullanılır.

Çalıştırma:
    python scripts/build_class_profiles.py
    python scripts/build_class_profiles.py --limit 5    # hızlı duman testi

Çıktı:
    models/class_profiles.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.background_removal import remove_background_grabcut
from src.feature_extraction import hsv_histogram, lbp_features, load_rgb_image
from src.screening import ClassProfiles, _bhattacharyya, _chi2

TRAIN_DIR = ROOT / "data" / "processed" / "oxford102_70_15_15" / "train"
OUT_PATH = ROOT / "models" / "class_profiles.npz"
IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect_images(class_dir: Path) -> list[Path]:
    return [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]


def build_profiles(
    train_dir: Path,
    image_size: int = IMAGE_SIZE,
    limit_per_class: int | None = None,
    out_path: Path | None = None,
) -> ClassProfiles:
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"Sınıf klasörü bulunamadı: {train_dir}")

    cleaned_flag = ROOT / "data" / "processed" / ".cleaned"
    if not cleaned_flag.exists():
        print(
            "[UYARI] data/processed/.cleaned bulunamadı — clean_data.py çalıştırılmamış "
            "olabilir. Bulanık/duplicate görseller profil kalitesini düşürür."
        )

    print(f"Toplam {len(class_dirs)} sınıf bulundu.")
    if limit_per_class:
        print(f"Sınıf başına ilk {limit_per_class} görsel kullanılacak (hızlı test modu).")

    classes: list[str] = []
    color_means: list[np.ndarray] = []
    texture_means: list[np.ndarray] = []
    color_dist_means: list[float] = []
    color_dist_stds: list[float] = []
    texture_dist_means: list[float] = []
    texture_dist_stds: list[float] = []

    for i, class_dir in enumerate(class_dirs):
        images = _collect_images(class_dir)
        if limit_per_class:
            images = images[:limit_per_class]
        if not images:
            print(f"  [UYARI] Boş sınıf atlandı: {class_dir.name}")
            continue

        color_hists: list[np.ndarray] = []
        lbp_hists: list[np.ndarray] = []

        for img_path in images:
            try:
                img = load_rgb_image(img_path, image_size=image_size)
                masked_img, _ = remove_background_grabcut(img)
                color_hists.append(hsv_histogram(masked_img))
                lbp_hists.append(lbp_features(masked_img))
            except Exception as exc:
                print(f"  [HATA] {img_path.name}: {exc}")

        if not color_hists:
            continue

        color_mean = np.mean(color_hists, axis=0)
        texture_mean = np.mean(lbp_hists, axis=0)

        c_dists = np.array([_bhattacharyya(h, color_mean) for h in color_hists])
        t_dists = np.array([_chi2(h, texture_mean) for h in lbp_hists])

        classes.append(class_dir.name)
        color_means.append(color_mean)
        texture_means.append(texture_mean)
        color_dist_means.append(float(c_dists.mean()))
        color_dist_stds.append(float(max(c_dists.std(), 1e-6)))
        texture_dist_means.append(float(t_dists.mean()))
        texture_dist_stds.append(float(max(t_dists.std(), 1e-6)))

        if (i + 1) % 10 == 0 or (i + 1) == len(class_dirs):
            print(f"  {i + 1}/{len(class_dirs)} sınıf işlendi...")

    profiles = ClassProfiles(
        classes=classes,
        color_means=np.stack(color_means),
        texture_means=np.stack(texture_means),
        color_dist_means=np.array(color_dist_means, dtype=np.float32),
        color_dist_stds=np.array(color_dist_stds, dtype=np.float32),
        texture_dist_means=np.array(texture_dist_means, dtype=np.float32),
        texture_dist_stds=np.array(texture_dist_stds, dtype=np.float32),
    )
    save_path = out_path if out_path else OUT_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    profiles.save(save_path)
    print(f"\nProfiller kaydedildi: {save_path}")
    print(f"Toplam sınıf: {len(classes)}")
    return profiles


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Çiçek sınıf profilleri oluşturucu")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Sınıf başına maksimum görsel sayısı (test için).",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=TRAIN_DIR,
        help="Eğitim verisi klasörü (varsayılan: oxford102 70/15/15 train).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Çıktı .npz dosyası (varsayılan: models/class_profiles.npz).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_profiles(args.train_dir, limit_per_class=args.limit, out_path=args.output)
