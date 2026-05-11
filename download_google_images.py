"""
Google/Bing'den Çiçek Görseli Otomatik İndirme
================================================
Oxford 102'deki türler için web'den ek görsel toplar.
İndirilen görseller farklı boyut, kalite ve arka plana sahip olacak.
Bu, cross-dataset generalization testinin temelidir.

Kullanım:
    python download_google_images.py                    # Tüm türler
    python download_google_images.py --classes rose sunflower daisy  # Belirli türler
    python download_google_images.py --limit 50         # Tür başına 50 görsel
"""

import os
import argparse
import hashlib
import json
import time
from pathlib import Path
from datetime import datetime

try:
    from bing_image_downloader import downloader
except ImportError:
    print("❌ bing-image-downloader kurulu değil!")
    print("   Çalıştır: pip install bing-image-downloader")
    exit(1)

try:
    from PIL import Image
except ImportError:
    print("❌ Pillow kurulu değil!")
    print("   Çalıştır: pip install Pillow")
    exit(1)


# ── Ayarlar ──────────────────────────────────────────────
OUTPUT_DIR = Path("data/raw/google_images")

# İndirilecek çiçek türleri ve arama terimleri
# Arama terimini "flower" ekleyerek spesifikleştiriyoruz
FLOWER_SEARCH_TERMS = {
    "rose": ["rose flower", "red rose flower", "rose garden close up"],
    "sunflower": ["sunflower", "sunflower field close up", "helianthus flower"],
    "daisy": ["daisy flower", "oxeye daisy", "white daisy close up"],
    "tulip": ["tulip flower", "red tulip", "tulip garden close up"],
    "dandelion": ["dandelion flower", "yellow dandelion", "taraxacum flower"],
    "lily": ["lily flower", "tiger lily flower", "lilium close up"],
    "orchid": ["orchid flower", "phalaenopsis orchid", "orchid close up"],
    "lavender": ["lavender flower", "lavender field close up", "lavandula"],
    "iris": ["iris flower", "purple iris", "iris germanica"],
    "marigold": ["marigold flower", "tagetes flower", "orange marigold"],
    "hibiscus": ["hibiscus flower", "red hibiscus", "hibiscus close up"],
    "magnolia": ["magnolia flower", "magnolia tree bloom", "magnolia close up"],
    "poppy": ["poppy flower", "red poppy field", "papaver flower"],
    "daffodil": ["daffodil flower", "yellow daffodil", "narcissus flower"],
    "camellia": ["camellia flower", "camellia japonica", "pink camellia"],
    "lotus": ["lotus flower", "pink lotus", "nelumbo flower"],
    "carnation": ["carnation flower", "dianthus caryophyllus", "pink carnation"],
    "peony": ["peony flower", "pink peony", "paeonia close up"],
    "chrysanthemum": ["chrysanthemum flower", "mum flower close up"],
    "jasmine": ["jasmine flower", "white jasmine", "jasminum close up"],
}


def compute_hash(filepath):
    """Dosyanın MD5 hash'ini hesapla (duplicate tespiti için)."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_image(filepath):
    """
    Görselin geçerli olup olmadığını kontrol et.
    Returns: (is_valid, reason)
    """
    try:
        img = Image.open(filepath)
        img.verify()  # Bozuk dosya kontrolü

        # Tekrar aç (verify sonrası kullanılamaz)
        img = Image.open(filepath)
        w, h = img.size

        # Çok küçük görselleri filtrele (ikon, thumbnail)
        if w < 100 or h < 100:
            return False, f"Çok küçük: {w}x{h}"

        # Çok uzun/geniş görselleri filtrele (banner, strip)
        ratio = max(w, h) / min(w, h)
        if ratio > 4:
            return False, f"Orantısız: {w}x{h} (ratio={ratio:.1f})"

        # Çok düşük dosya boyutu (muhtemelen bozuk veya placeholder)
        file_size = os.path.getsize(filepath)
        if file_size < 5000:  # 5KB altı
            return False, f"Çok küçük dosya: {file_size} bytes"

        return True, "OK"

    except Exception as e:
        return False, f"Bozuk dosya: {str(e)}"


def download_flowers(classes=None, limit=30, skip_existing=True):
    """
    Belirtilen çiçek türlerinin görsellerini indir.

    Args:
        classes: İndirilecek çiçek türleri listesi (None = hepsi)
        limit: Her arama terimi için indirilecek görsel sayısı
        skip_existing: Zaten indirilen türleri atla
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = classes if classes else list(FLOWER_SEARCH_TERMS.keys())

    stats = {
        "downloaded": 0,
        "valid": 0,
        "invalid": 0,
        "duplicate": 0,
        "classes": {}
    }

    print(f"\n🌸 Çiçek Görseli İndirme Başlıyor")
    print(f"   Tür sayısı : {len(targets)}")
    print(f"   Tür başına  : ~{limit * 2} görsel hedef (birden fazla arama terimi)")
    print(f"   Hedef klasör: {OUTPUT_DIR.resolve()}\n")

    for flower_name in targets:
        class_dir = OUTPUT_DIR / flower_name
        search_terms = FLOWER_SEARCH_TERMS.get(flower_name, [f"{flower_name} flower"])

        # Zaten yeterli görsel varsa atla
        if skip_existing and class_dir.exists():
            existing = len(list(class_dir.glob("*.*")))
            if existing >= limit:
                print(f"  ⏭ {flower_name}: zaten {existing} görsel mevcut, atlanıyor")
                continue

        print(f"\n  🔍 {flower_name} indiriliyor...")
        class_dir.mkdir(parents=True, exist_ok=True)

        seen_hashes = set()
        valid_count = 0

        for term in search_terms:
            try:
                # Geçici klasöre indir
                temp_dir = OUTPUT_DIR / "_temp"
                downloader.download(
                    term,
                    limit=limit,
                    output_dir=str(temp_dir),
                    adult_filter_off=False,
                    force_replace=False,
                    timeout=10,
                    verbose=False
                )

                # İndirilen görselleri kontrol et ve taşı
                term_dir = temp_dir / term
                if term_dir.exists():
                    for img_path in term_dir.iterdir():
                        if not img_path.is_file():
                            continue

                        stats["downloaded"] += 1

                        # Geçerlilik kontrolü
                        is_valid, reason = validate_image(img_path)
                        if not is_valid:
                            stats["invalid"] += 1
                            img_path.unlink()
                            continue

                        # Duplicate kontrolü
                        img_hash = compute_hash(img_path)
                        if img_hash in seen_hashes:
                            stats["duplicate"] += 1
                            img_path.unlink()
                            continue

                        seen_hashes.add(img_hash)

                        # Temiz isimle taşı
                        ext = img_path.suffix.lower()
                        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                            ext = ".jpg"
                        new_name = f"{flower_name}_{valid_count:04d}{ext}"
                        dest = class_dir / new_name
                        img_path.rename(dest)

                        valid_count += 1
                        stats["valid"] += 1

                # Temp klasörü temizle
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)

                time.sleep(1)  # Rate limiting

            except Exception as e:
                print(f"    ⚠ Hata ({term}): {e}")
                continue

        stats["classes"][flower_name] = valid_count
        print(f"    ✓ {flower_name}: {valid_count} geçerli görsel")

    # ── Özet Rapor ──
    print("\n" + "=" * 50)
    print("📊 İndirme Özeti:")
    print(f"   Toplam indirilen : {stats['downloaded']}")
    print(f"   Geçerli          : {stats['valid']}")
    print(f"   Geçersiz         : {stats['invalid']}")
    print(f"   Duplicate        : {stats['duplicate']}")
    print(f"\n   Sınıf dağılımı:")
    for cls, count in sorted(stats["classes"].items()):
        print(f"     {cls:20s}: {count} görsel")
    print("=" * 50)

    # İstatistikleri kaydet
    stats["timestamp"] = datetime.now().isoformat()
    with open(OUTPUT_DIR / "download_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Çiçek görsellerini web'den indir")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="İndirilecek çiçek türleri (varsayılan: hepsi)")
    parser.add_argument("--limit", type=int, default=30,
                        help="Her arama terimi için görsel sayısı (varsayılan: 30)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Mevcut görselleri de tekrar indir")

    args = parser.parse_args()
    download_flowers(
        classes=args.classes,
        limit=args.limit,
        skip_existing=not args.no_skip
    )
