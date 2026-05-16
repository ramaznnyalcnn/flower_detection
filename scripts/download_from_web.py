"""
İnternetten çiçek görseli indirme — iNaturalist + Wikimedia Commons
Kullanım:
    python scripts/download_from_web.py                    # tüm sınıflar
    python scripts/download_from_web.py --classes gul lale # belirli sınıflar
    python scripts/download_from_web.py --limit 100        # sınıf başına limit
"""
import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

from PIL import Image
import io

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "web_flowers"

# Her sınıf için iNaturalist taxon_id ve Wikimedia kategorisi
FLOWER_SOURCES = {
    "gul": {
        "inaturalist_taxon": 47792,    # Rosa
        "wikimedia_cat": "Rosa_(plant)",
        "tr_name": "Gül",
    },
    "lale": {
        "inaturalist_taxon": 48486,    # Tulipa
        "wikimedia_cat": "Tulipa",
        "tr_name": "Lale",
    },
    "papatya": {
        "inaturalist_taxon": 55733,    # Bellis
        "wikimedia_cat": "Bellis_perennis",
        "tr_name": "Papatya",
    },
    "aysicegi": {
        "inaturalist_taxon": 50197,    # Helianthus
        "wikimedia_cat": "Helianthus_annuus",
        "tr_name": "Ayçiçeği",
    },
    "karahindiba": {
        "inaturalist_taxon": 53359,    # Taraxacum
        "wikimedia_cat": "Taraxacum_officinale",
        "tr_name": "Karahindiba",
    },
    "lilyum": {
        "inaturalist_taxon": 48461,    # Lilium
        "wikimedia_cat": "Lilium",
        "tr_name": "Lilyum",
    },
    "orkide": {
        "inaturalist_taxon": 47217,    # Orchidaceae
        "wikimedia_cat": "Orchidaceae",
        "tr_name": "Orkide",
    },
    "kasimpati": {
        "inaturalist_taxon": 56060,    # Tagetes
        "wikimedia_cat": "Tagetes",
        "tr_name": "Kasımpatı",
    },
    "iris": {
        "inaturalist_taxon": 50015,    # Iris
        "wikimedia_cat": "Iris_(plant)",
        "tr_name": "İris",
    },
    "menekse": {
        "inaturalist_taxon": 48737,    # Viola
        "wikimedia_cat": "Viola_(plant)",
        "tr_name": "Menekşe",
    },
    "sardunya": {
        "inaturalist_taxon": 55886,    # Pelargonium
        "wikimedia_cat": "Pelargonium",
        "tr_name": "Sardunya",
    },
    "bougainvillea": {
        "inaturalist_taxon": 56428,    # Bougainvillea
        "wikimedia_cat": "Bougainvillea",
        "tr_name": "Bougainvillea",
    },
}


def _download_image(url: str, out_path: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FlowerBot/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.width < 100 or img.height < 100:
            return False
        img.save(out_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def download_inaturalist(cls: str, taxon_id: int, out_dir: Path, limit: int = 150) -> int:
    import json, urllib.parse
    saved = 0
    page = 1
    while saved < limit:
        per_page = min(200, limit - saved)
        url = (
            f"https://api.inaturalist.org/v1/observations"
            f"?taxon_id={taxon_id}&quality_grade=research"
            f"&photos=true&per_page={per_page}&page={page}"
            f"&order_by=votes"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FlowerBot/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f"    iNaturalist API hatası: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for obs in results:
            if saved >= limit:
                break
            for photo in obs.get("photos", [])[:1]:
                img_url = photo.get("url", "").replace("square", "medium")
                if not img_url:
                    continue
                h = hashlib.md5(img_url.encode()).hexdigest()[:8]
                out_path = out_dir / f"inat_{h}.jpg"
                if out_path.exists():
                    saved += 1
                    continue
                if _download_image(img_url, out_path):
                    saved += 1
            time.sleep(0.1)

        page += 1
        time.sleep(1)

    return saved


def download_wikimedia(cls: str, category: str, out_dir: Path, limit: int = 50) -> int:
    import json
    saved = 0
    cmcontinue = ""
    while saved < limit:
        url = (
            f"https://commons.wikimedia.org/w/api.php"
            f"?action=query&generator=categorymembers"
            f"&gcmtitle=Category:{category}&gcmtype=file"
            f"&gcmlimit=50&prop=imageinfo&iiprop=url"
            f"&iiurlwidth=600&format=json"
        )
        if cmcontinue:
            url += f"&gcmcontinue={urllib.request.quote(cmcontinue)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FlowerBot/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f"    Wikimedia API hatası: {e}")
            break

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if saved >= limit:
                break
            info = page.get("imageinfo", [{}])[0]
            img_url = info.get("thumburl") or info.get("url", "")
            if not img_url or not any(img_url.lower().endswith(e) for e in [".jpg", ".jpeg", ".png"]):
                continue
            h = hashlib.md5(img_url.encode()).hexdigest()[:8]
            out_path = out_dir / f"wiki_{h}.jpg"
            if out_path.exists():
                saved += 1
                continue
            if _download_image(img_url, out_path):
                saved += 1
            time.sleep(0.05)

        cont = data.get("continue", {})
        cmcontinue = cont.get("gcmcontinue", "")
        if not cmcontinue:
            break

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=list(FLOWER_SOURCES.keys()))
    parser.add_argument("--limit", type=int, default=150, help="Sınıf başına toplam hedef resim")
    parser.add_argument("--source", choices=["inaturalist", "wikimedia", "both"], default="both")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Hedef: {args.limit} resim/sınıf | Kaynak: {args.source}\n")

    for cls in args.classes:
        if cls not in FLOWER_SOURCES:
            print(f"⚠️  Bilinmeyen sınıf: {cls}")
            continue
        info = FLOWER_SOURCES[cls]
        out_dir = OUT_DIR / cls
        out_dir.mkdir(exist_ok=True)
        mevcut = len(list(out_dir.glob("*.jpg")))
        kalan = args.limit - mevcut
        print(f"🌸 {info['tr_name']} ({cls}) — mevcut: {mevcut}, hedef: {args.limit}")

        if kalan <= 0:
            print(f"   ✅ Zaten yeterli resim var")
            continue

        inat_limit = int(kalan * 0.7) if args.source in ("both", "inaturalist") else 0
        wiki_limit = kalan - inat_limit if args.source in ("both", "wikimedia") else 0

        if inat_limit > 0:
            n = download_inaturalist(cls, info["inaturalist_taxon"], out_dir, inat_limit)
            print(f"   iNaturalist: +{n} resim")
        if wiki_limit > 0:
            n = download_wikimedia(cls, info["wikimedia_cat"], out_dir, wiki_limit)
            print(f"   Wikimedia:   +{n} resim")

        toplam = len(list(out_dir.glob("*.jpg")))
        print(f"   Toplam: {toplam} resim\n")

    print("Tamamlandı!")


if __name__ == "__main__":
    main()
