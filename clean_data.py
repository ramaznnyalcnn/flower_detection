"""
Veri Temizleme ve Normalizasyon Pipeline'ı
==========================================
Her iki kaynaktan (Oxford 102 + Google) gelen görselleri:
1. Boyut normalizasyonu (resize + padding)
2. Bulanıklık tespiti (Laplacian variance)
3. Duplicate tespiti (perceptual hash)
4. Renk uzayı kontrolü (grayscale → RGB dönüşümü)
5. Format standardizasyonu (hepsi JPG)
6. Sınıf dengesizliği raporu

Kullanım:
    python clean_data.py --source oxford     # Sadece Oxford verisini temizle
    python clean_data.py --source google     # Sadece Google verisini temizle
    python clean_data.py --source all        # Hepsini temizle
"""

import os
import argparse
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
from PIL import Image, ImageFilter, ImageStat

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠ OpenCV bulunamadı. Bulanıklık tespiti basit modda çalışacak.")
    print("  Kurulum: pip install opencv-python")

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("⚠ imagehash bulunamadı. Duplicate tespiti MD5 ile yapılacak.")
    print("  Kurulum: pip install imagehash")


# ── Ayarlar ──────────────────────────────────────────────
TARGET_SIZE = 224           # Hedef boyut (224x224 — standart CNN input)
MIN_ORIGINAL_SIZE = 64      # Bu boyutun altındaki görseller silinir
BLUR_THRESHOLD = 50         # Laplacian variance eşiği (altı = bulanık)
MIN_FILE_SIZE = 3000        # 3KB altı dosyalar silinir
MAX_FILE_SIZE = 20_000_000  # 20MB üstü dosyalar silinir

OXFORD_DIR = Path("data/raw/oxford102")
GOOGLE_DIR = Path("data/raw/google_images")
PROCESSED_DIR = Path("data/processed")


class ImageCleaner:
    """Görsel temizleme ve normalizasyon sınıfı."""

    def __init__(self, target_size=TARGET_SIZE, filter_images=True):
        self.target_size = target_size
        self.filter_images = filter_images
        self.stats = {
            "total_scanned": 0,
            "valid": 0,
            "removed_corrupt": 0,
            "removed_small": 0,
            "removed_blur": 0,
            "removed_duplicate": 0,
            "removed_ratio": 0,
            "converted_grayscale": 0,
            "resized": 0,
        }
        self.seen_hashes = set()

    def check_corrupt(self, path):
        """Dosyanın bozuk olup olmadığını kontrol et."""
        try:
            img = Image.open(path)
            img.verify()
            return True
        except Exception:
            return False

    def check_size(self, path):
        """Dosya ve görsel boyutunu kontrol et."""
        file_size = os.path.getsize(path)
        if file_size < MIN_FILE_SIZE or file_size > MAX_FILE_SIZE:
            return False

        try:
            img = Image.open(path)
            w, h = img.size
            if w < MIN_ORIGINAL_SIZE or h < MIN_ORIGINAL_SIZE:
                return False
            return True
        except Exception:
            return False

    def check_ratio(self, path):
        """En-boy oranını kontrol et."""
        try:
            img = Image.open(path)
            w, h = img.size
            ratio = max(w, h) / max(min(w, h), 1)
            return ratio <= 3.5
        except Exception:
            return False

    def check_blur(self, path):
        """Bulanıklık tespiti (Laplacian variance)."""
        if HAS_CV2:
            try:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return False
                variance = cv2.Laplacian(img, cv2.CV_64F).var()
                return variance > BLUR_THRESHOLD
            except Exception:
                return True  # Hata durumunda geçir
        else:
            # Basit bulanıklık tespiti (PIL ile)
            try:
                img = Image.open(path).convert("L")
                edges = img.filter(ImageFilter.FIND_EDGES)
                stat = ImageStat.Stat(edges)
                return stat.stddev[0] > 10
            except Exception:
                return True

    def check_duplicate(self, path):
        """Perceptual hash ile duplicate tespiti."""
        if HAS_IMAGEHASH:
            try:
                img = Image.open(path)
                phash = str(imagehash.phash(img, hash_size=12))
                if phash in self.seen_hashes:
                    return False
                self.seen_hashes.add(phash)
                return True
            except Exception:
                return True
        else:
            # MD5 hash fallback
            hasher = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            h = hasher.hexdigest()
            if h in self.seen_hashes:
                return False
            self.seen_hashes.add(h)
            return True

    def normalize_image(self, src_path, dst_path):
        """
        Görseli normalize et:
        - Grayscale → RGB dönüşümü
        - Resize (en-boy oranını koruyarak)
        - Center crop veya padding
        - JPG olarak kaydet
        """
        try:
            img = Image.open(src_path)

            # RGBA → RGB
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode == "L":
                img = img.convert("RGB")
                self.stats["converted_grayscale"] += 1
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Resize — en-boy oranını koruyarak küçült, sonra center crop
            w, h = img.size
            scale = self.target_size / min(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

            # Center crop
            left = (new_w - self.target_size) // 2
            top = (new_h - self.target_size) // 2
            img = img.crop((left, top, left + self.target_size, top + self.target_size))

            self.stats["resized"] += 1

            # JPG olarak kaydet
            dst_path = dst_path.with_suffix(".jpg")
            img.save(dst_path, "JPEG", quality=95)
            return True

        except Exception as e:
            print(f"    ⚠ Normalize hatası: {src_path.name} — {e}")
            return False

    def clean_directory(self, src_dir, dst_dir, split_name="all"):
        """
        Bir dizindeki tüm görselleri temizle ve normalize et.

        Args:
            src_dir: Kaynak dizin (sınıf alt klasörleri içermeli)
            dst_dir: Hedef dizin
            split_name: "train", "val", "test" veya "all"
        """
        src_dir = Path(src_dir)
        dst_dir = Path(dst_dir)

        if not src_dir.exists():
            print(f"  ❌ Kaynak dizin bulunamadı: {src_dir}")
            return

        # Sınıf klasörlerini bul
        class_dirs = [d for d in src_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]

        if not class_dirs:
            print(f"  ❌ Sınıf klasörleri bulunamadı: {src_dir}")
            return

        print(f"\n  📂 {src_dir.name}/{split_name} — {len(class_dirs)} sınıf bulundu")

        class_counts = Counter()

        for class_dir in sorted(class_dirs):
            class_name = class_dir.name
            out_dir = dst_dir / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            images = list(class_dir.glob("*.*"))
            valid = 0

            for img_path in images:
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]:
                    continue

                self.stats["total_scanned"] += 1

                # Kontrol zinciri
                if not self.check_corrupt(img_path):
                    self.stats["removed_corrupt"] += 1
                    continue

                if self.filter_images:
                    if not self.check_size(img_path):
                        self.stats["removed_small"] += 1
                        continue

                    if not self.check_ratio(img_path):
                        self.stats["removed_ratio"] += 1
                        continue

                    if not self.check_blur(img_path):
                        self.stats["removed_blur"] += 1
                        continue

                    if not self.check_duplicate(img_path):
                        self.stats["removed_duplicate"] += 1
                        continue

                # Normalize et ve kaydet
                dst_path = out_dir / f"{class_name}_{valid:04d}.jpg"
                if self.normalize_image(img_path, dst_path):
                    valid += 1
                    self.stats["valid"] += 1

            class_counts[class_name] = valid

        return class_counts

    def print_report(self, class_counts=None):
        """Temizleme raporunu yazdır."""
        print("\n" + "=" * 55)
        print("📊 Temizleme Raporu")
        print("=" * 55)
        print(f"  Taranan     : {self.stats['total_scanned']}")
        print(f"  Geçerli     : {self.stats['valid']}")
        print(f"  ─────────────────────────────────")
        print(f"  Bozuk       : {self.stats['removed_corrupt']}")
        print(f"  Küçük       : {self.stats['removed_small']}")
        print(f"  Orantısız   : {self.stats['removed_ratio']}")
        print(f"  Bulanık     : {self.stats['removed_blur']}")
        print(f"  Duplicate   : {self.stats['removed_duplicate']}")
        print(f"  ─────────────────────────────────")
        print(f"  Grayscale→RGB: {self.stats['converted_grayscale']}")
        print(f"  Resize       : {self.stats['resized']}")

        if class_counts:
            values = list(class_counts.values())
            print(f"\n  Sınıf istatistikleri:")
            print(f"    Sınıf sayısı: {len(class_counts)}")
            print(f"    Min         : {min(values)}")
            print(f"    Max         : {max(values)}")
            print(f"    Ortalama    : {np.mean(values):.1f}")
            print(f"    Medyan      : {np.median(values):.1f}")

            # Dengesizlik uyarısı
            if max(values) > 3 * min(values):
                print(f"\n  ⚠ UYARI: Sınıf dengesizliği tespit edildi!")
                print(f"    En az: {min(class_counts, key=class_counts.get)} ({min(values)})")
                print(f"    En çok: {max(class_counts, key=class_counts.get)} ({max(values)})")

        print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Görsel veri temizleme pipeline'ı")
    parser.add_argument("--source", choices=["oxford", "google", "all"], default="all")
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Bozuk dosyalar dışında görsel eleme yapma; sadece RGB/resize/JPG normalizasyonu uygula.",
    )
    args = parser.parse_args()

    cleaner = ImageCleaner(target_size=args.target_size, filter_images=not args.keep_all)
    all_counts = Counter()

    if args.source in ["oxford", "all"]:
        print("\n🌿 Oxford Flowers 102 temizleniyor...")
        for split in ["train", "val", "test"]:
            src = OXFORD_DIR / split
            dst = PROCESSED_DIR / "oxford102" / split
            counts = cleaner.clean_directory(src, dst, split)
            if counts:
                all_counts.update(counts)

    if args.source in ["google", "all"]:
        print("\n🌐 Google görselleri temizleniyor...")
        dst = PROCESSED_DIR / "google_images"
        counts = cleaner.clean_directory(GOOGLE_DIR, dst, "all")
        if counts:
            all_counts.update(counts)

    cleaner.print_report(all_counts if all_counts else None)

    # İstatistikleri kaydet
    report = {
        "timestamp": datetime.now().isoformat(),
        "stats": cleaner.stats,
        "class_counts": dict(all_counts) if all_counts else {},
        "settings": {
            "target_size": args.target_size,
            "blur_threshold": BLUR_THRESHOLD,
            "min_file_size": MIN_FILE_SIZE,
            "filter_images": cleaner.filter_images,
        }
    }
    report_path = PROCESSED_DIR / "cleaning_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Rapor kaydedildi: {report_path}")


if __name__ == "__main__":
    main()
