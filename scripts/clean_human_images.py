"""
İnsan içeren resimleri tespit edip kaldırır.
Yüz tespiti (Haar Cascade) + minimum boyut kontrolü.

Kullanım:
    python scripts/clean_human_images.py data/raw/web_flowers
    python scripts/clean_human_images.py data/raw/web_flowers --dry-run
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("⚠️  OpenCV yok, sadece boyut kontrolü yapılacak")


def has_face(img_path: Path) -> bool:
    if not CV2_OK:
        return False
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0


def is_valid_image(img_path: Path, min_size: int = 80) -> bool:
    try:
        img = Image.open(img_path)
        return img.width >= min_size and img.height >= min_size
    except Exception:
        return False


def clean_directory(directory: Path, dry_run: bool = False) -> tuple[int, int]:
    images = list(directory.rglob("*.jpg")) + list(directory.rglob("*.png"))
    removed = checked = 0
    for img_path in images:
        checked += 1
        reason = None
        if not is_valid_image(img_path):
            reason = "çok küçük/bozuk"
        elif has_face(img_path):
            reason = "insan yüzü"

        if reason:
            if dry_run:
                print(f"  [DRY] Silinecek ({reason}): {img_path.name}")
            else:
                img_path.unlink()
            removed += 1

    return checked, removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Silmeden sadece listele")
    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Klasör bulunamadı: {args.directory}")
        return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Temizleniyor: {args.directory}\n")
    total_checked = total_removed = 0

    cls_dirs = [d for d in args.directory.iterdir() if d.is_dir()]
    if not cls_dirs:
        cls_dirs = [args.directory]

    for cls_dir in sorted(cls_dirs):
        checked, removed = clean_directory(cls_dir, dry_run=args.dry_run)
        total_checked += checked
        total_removed += removed
        if removed:
            print(f"  {cls_dir.name:<15} {checked} resim → {removed} silindi")
        else:
            print(f"  {cls_dir.name:<15} {checked} resim → temiz ✅")

    print(f"\nToplam: {total_checked} kontrol, {total_removed} silindi")


if __name__ == "__main__":
    main()
