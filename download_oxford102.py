"""
Oxford Flowers 102 Veri Setini İndir ve Hazırla
================================================
102 çiçek türü, ~8K görüntü
Kaynak: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
"""

import os
import tarfile
import shutil
import scipy.io
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
from collections import Counter

# ── Ayarlar ──────────────────────────────────────────────
DATA_DIR = Path("data/raw/oxford102")
IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
SPLITS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

# 102 çiçek türünün İngilizce isimleri
FLOWER_NAMES = {
    1: "pink primrose", 2: "hard-leaved pocket orchid", 3: "canterbury bells",
    4: "sweet pea", 5: "english marigold", 6: "tiger lily", 7: "moon orchid",
    8: "bird of paradise", 9: "monkshood", 10: "globe thistle",
    11: "snapdragon", 12: "colt's foot", 13: "king protea", 14: "spear thistle",
    15: "yellow iris", 16: "globe-flower", 17: "purple coneflower",
    18: "peruvian lily", 19: "balloon flower", 20: "giant white arum lily",
    21: "fire lily", 22: "pincushion flower", 23: "fritillary",
    24: "red ginger", 25: "grape hyacinth", 26: "corn poppy",
    27: "prince of wales feathers", 28: "stemless gentian", 29: "artichoke",
    30: "sweet william", 31: "carnation", 32: "garden phlox",
    33: "love in the mist", 34: "mexican aster", 35: "alpine sea holly",
    36: "ruby-lipped cattleya", 37: "cape flower", 38: "great masterwort",
    39: "siam tulip", 40: "lenten rose", 41: "barbeton daisy",
    42: "daffodil", 43: "sword lily", 44: "poinsettia", 45: "bolero deep blue",
    46: "wallflower", 47: "marigold", 48: "buttercup", 49: "oxeye daisy",
    50: "common dandelion", 51: "petunia", 52: "wild pansy",
    53: "primula", 54: "sunflower", 55: "pelargonium", 56: "bishop of llandaff",
    57: "gaura", 58: "geranium", 59: "orange dahlia", 60: "pink-yellow dahlia",
    61: "cautleya spicata", 62: "japanese anemone", 63: "black-eyed susan",
    64: "silverbush", 65: "californian poppy", 66: "osteospermum",
    67: "spring crocus", 68: "bearded iris", 69: "windflower",
    70: "tree poppy", 71: "gazania", 72: "azalea", 73: "water lily",
    74: "rose", 75: "thorn apple", 76: "morning glory",
    77: "passion flower", 78: "lotus", 79: "toad lily",
    80: "anthurium", 81: "frangipani", 82: "clematis",
    83: "hibiscus", 84: "columbine", 85: "desert-rose",
    86: "tree mallow", 87: "magnolia", 88: "cyclamen",
    89: "watercress", 90: "canna lily", 91: "hippeastrum",
    92: "bee balm", 93: "ball moss", 94: "foxglove",
    95: "bougainvillea", 96: "camellia", 97: "mallow",
    98: "mexican petunia", 99: "bromelia", 100: "blanket flower",
    101: "trumpet creeper", 102: "blackberry lily"
}


def download_file(url, dest):
    """Dosyayı indir, varsa atla."""
    if dest.exists():
        print(f"  ✓ Zaten mevcut: {dest.name}")
        return
    print(f"  ↓ İndiriliyor: {dest.name} ...")
    urlretrieve(url, dest)
    print(f"  ✓ Tamamlandı: {dest.name}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Dosyaları indir ──
    print("\n📥 Dosyalar indiriliyor...")
    download_file(IMAGES_URL, DATA_DIR / "102flowers.tgz")
    download_file(LABELS_URL, DATA_DIR / "imagelabels.mat")
    download_file(SPLITS_URL, DATA_DIR / "setid.mat")

    # ── 2. Görselleri çıkart ──
    images_dir = DATA_DIR / "jpg"
    if not images_dir.exists():
        print("\n📦 Arşiv açılıyor...")
        with tarfile.open(DATA_DIR / "102flowers.tgz", "r:gz") as tar:
            tar.extractall(DATA_DIR)
        print(f"  ✓ {len(list(images_dir.glob('*.jpg')))} görsel çıkartıldı")
    else:
        print(f"\n✓ Görseller zaten mevcut: {len(list(images_dir.glob('*.jpg')))} adet")

    # ── 3. Etiketleri yükle ──
    labels = scipy.io.loadmat(DATA_DIR / "imagelabels.mat")["labels"][0]
    splits = scipy.io.loadmat(DATA_DIR / "setid.mat")

    train_ids = splits["trnid"][0]  # Eğitim
    val_ids = splits["valid"][0]     # Doğrulama
    test_ids = splits["tstid"][0]    # Test

    print(f"\n📊 Veri Seti İstatistikleri:")
    print(f"  Toplam görsel : {len(labels)}")
    print(f"  Toplam tür    : {len(set(labels))}")
    print(f"  Eğitim        : {len(train_ids)}")
    print(f"  Doğrulama     : {len(val_ids)}")
    print(f"  Test           : {len(test_ids)}")

    # ── 4. Klasörlere ayır ──
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_dir = DATA_DIR / split_name
        if split_dir.exists() and len(list(split_dir.rglob("*.jpg"))) > 0:
            print(f"\n✓ {split_name}/ zaten düzenlenmiş")
            continue

        print(f"\n📂 {split_name}/ klasörü oluşturuluyor...")
        for img_id in ids:
            label = labels[img_id - 1]
            class_name = FLOWER_NAMES.get(label, f"class_{label}")
            # Klasör adını temizle
            safe_name = class_name.replace(" ", "_").replace("'", "").replace("-", "_")
            class_dir = split_dir / safe_name
            class_dir.mkdir(parents=True, exist_ok=True)

            src = images_dir / f"image_{img_id:05d}.jpg"
            dst = class_dir / f"image_{img_id:05d}.jpg"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

        # İstatistik
        class_counts = Counter()
        for img_id in ids:
            label = labels[img_id - 1]
            class_counts[FLOWER_NAMES.get(label, f"class_{label}")] += 1

        print(f"  ✓ {len(ids)} görsel, {len(class_counts)} sınıf")
        print(f"  Min: {min(class_counts.values())} | Max: {max(class_counts.values())} | "
              f"Ort: {np.mean(list(class_counts.values())):.1f}")

    # ── 5. Özet ──
    print("\n" + "=" * 50)
    print("✅ Oxford Flowers 102 hazır!")
    print(f"   Konum: {DATA_DIR.resolve()}")
    print(f"   Yapı:  {DATA_DIR}/train|val|test/çiçek_adı/image_XXXXX.jpg")
    print("=" * 50)


if __name__ == "__main__":
    main()
