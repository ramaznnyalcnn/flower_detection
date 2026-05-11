"""Build a unified dataset from Oxford 102, TF Flowers, and (optionally) Kaggle Flowers.

Usage:
    python scripts/build_unified_dataset.py \
        --oxford-dir data/processed/oxford102_70_15_15 \
        --output-dir data/raw/unified \
        --aliases    configs/class_aliases.yaml \
        [--kaggle-dir data/raw/kaggle_flowers] \
        [--skip-tf]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Allow running as `python scripts/build_unified_dataset.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.data_utils import write_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Alias handling
# ---------------------------------------------------------------------------

def load_alias_map(yaml_path: str | Path) -> dict[str, str]:
    """Return alias→canonical dict built from the YAML file."""
    with open(yaml_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    mapping: dict[str, str] = {}
    for canonical, aliases in data.items():
        for alias in aliases:
            mapping[alias] = canonical
    return mapping


def canonical_name(raw: str, alias_map: dict[str, str]) -> str:
    return alias_map.get(raw, raw)


# ---------------------------------------------------------------------------
# Image discovery helpers
# ---------------------------------------------------------------------------

def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_from_flat_or_split(source_dir: Path) -> dict[str, list[Path]]:
    """Collect {class_name: [image_paths]} from a dir that is either:
    - flat:  source_dir/<class>/*.jpg
    - split: source_dir/{train,val,test}/<class>/*.jpg
    """
    grouped: dict[str, list[Path]] = defaultdict(list)
    split_dirs = [source_dir / s for s in SPLITS]
    roots = split_dirs if all(p.exists() for p in split_dirs) else [source_dir]
    for root in roots:
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for img in sorted(class_dir.rglob("*")):
                if is_image(img):
                    grouped[class_dir.name].append(img)
    return grouped


# ---------------------------------------------------------------------------
# Copy helpers
# ---------------------------------------------------------------------------

def copy_images(
    grouped: dict[str, list[Path]],
    output_dir: Path,
    alias_map: dict[str, str],
    source_tag: str,
    counters: dict[str, int],
    source_log: dict[str, dict[str, int]],
) -> None:
    """Copy images into output_dir/<canonical_class>/ with collision-safe names."""
    for raw_class, paths in grouped.items():
        cls = canonical_name(raw_class, alias_map)
        class_out = output_dir / cls
        class_out.mkdir(parents=True, exist_ok=True)
        for img_path in paths:
            idx = counters.get(cls, 0)
            dest = class_out / f"{source_tag}_{idx:06d}{img_path.suffix.lower()}"
            shutil.copy2(img_path, dest)
            counters[cls] = idx + 1
            source_log.setdefault(cls, {}).setdefault(source_tag, 0)
            source_log[cls][source_tag] += 1


# ---------------------------------------------------------------------------
# TF Flowers ingestion (reads from cached tarball — no TensorFlow required)
# ---------------------------------------------------------------------------

TF_FLOWERS_CACHE = Path.home() / "tensorflow_datasets" / "downloads" / "tf_flowers"
TF_FLOWERS_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


def _find_tf_archive() -> Path | None:
    """Return path to the cached TF Flowers tgz, or None if not found."""
    if TF_FLOWERS_CACHE.exists():
        for p in TF_FLOWERS_CACHE.glob("*.tgz"):
            return p
    return None


def _download_tf_archive() -> Path:
    """Download TF Flowers tgz to the cache directory."""
    import urllib.request

    TF_FLOWERS_CACHE.mkdir(parents=True, exist_ok=True)
    dest = TF_FLOWERS_CACHE / "flower_photos.tgz"
    print(f"  Downloading {TF_FLOWERS_URL} …")
    urllib.request.urlretrieve(TF_FLOWERS_URL, dest)
    return dest


def ingest_tf_flowers(output_dir: Path, alias_map: dict[str, str],
                      counters: dict[str, int], source_log: dict[str, dict[str, int]]) -> None:
    import io
    import tarfile

    try:
        from PIL import Image as PILImage
    except ImportError:
        print("[SKIP] Pillow not installed — skipping TF Flowers.")
        return

    archive = _find_tf_archive()
    if archive is None:
        try:
            archive = _download_tf_archive()
        except Exception as exc:
            print(f"[SKIP] Could not download TF Flowers: {exc}")
            return

    print(f"Extracting TF Flowers from {archive.name} …")
    with tarfile.open(archive, "r:gz") as tf_tar:
        members = [m for m in tf_tar.getmembers() if m.isfile()]
        for member in members:
            parts = member.name.split("/")
            # expected structure: flower_photos/<class>/<filename>
            if len(parts) < 3:
                continue
            raw_class = parts[1]
            if raw_class.startswith("."):
                continue

            fobj = tf_tar.extractfile(member)
            if fobj is None:
                continue

            cls = canonical_name(raw_class, alias_map)
            class_out = output_dir / cls
            class_out.mkdir(parents=True, exist_ok=True)

            idx = counters.get(cls, 0)
            dest = class_out / f"tf_{idx:06d}.jpg"
            try:
                img = PILImage.open(io.BytesIO(fobj.read())).convert("RGB")
                img.save(dest, "JPEG", quality=95)
            except Exception:
                continue
            counters[cls] = idx + 1
            source_log.setdefault(cls, {}).setdefault("tf", 0)
            source_log[cls]["tf"] += 1

    total_tf = sum(v.get("tf", 0) for v in source_log.values())
    print(f"  TF Flowers done: {total_tf} images")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def write_stats(
    output_dir: Path,
    source_log: dict[str, dict[str, int]],
    stats_path: Path,
) -> None:
    classes = {}
    total = 0
    for cls in sorted(source_log):
        count = sum(source_log[cls].values())
        total += count
        classes[cls] = {"total": count, "by_source": source_log[cls]}

    payload = {
        "total_images": total,
        "num_classes": len(classes),
        "classes": classes,
    }
    write_json(stats_path, payload)
    print(f"Stats → {stats_path}  ({len(classes)} classes, {total} images)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified flower dataset.")
    parser.add_argument("--oxford-dir", default="data/processed/oxford102_70_15_15",
                        help="Processed Oxford 102 directory (split or flat)")
    parser.add_argument("--output-dir", default="data/raw/unified",
                        help="Raw unified output directory")
    parser.add_argument("--aliases", default="configs/class_aliases.yaml",
                        help="Path to class_aliases.yaml")
    parser.add_argument("--kaggle-dir", default=None,
                        help="Optional Kaggle Flowers Recognition directory")
    parser.add_argument("--skip-tf", action="store_true",
                        help="Skip TF Flowers download")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete and recreate output-dir if it exists")
    parser.add_argument("--stats-path", default="results/unified_class_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    alias_map = load_alias_map(args.aliases)
    counters: dict[str, int] = {}
    source_log: dict[str, dict[str, int]] = {}

    # Oxford 102
    oxford_dir = Path(args.oxford_dir)
    if oxford_dir.exists():
        print(f"Reading Oxford 102 from {oxford_dir} …")
        grouped = collect_from_flat_or_split(oxford_dir)
        copy_images(grouped, output_dir, alias_map, "oxford", counters, source_log)
        print(f"  Oxford done: {len(grouped)} classes, {sum(len(v) for v in grouped.values())} images")
    else:
        print(f"[WARN] Oxford dir not found: {oxford_dir}")

    # TF Flowers
    if not args.skip_tf:
        ingest_tf_flowers(output_dir, alias_map, counters, source_log)

    # Kaggle (optional)
    if args.kaggle_dir:
        kaggle_dir = Path(args.kaggle_dir)
        if kaggle_dir.exists():
            print(f"Reading Kaggle Flowers from {kaggle_dir} …")
            grouped = collect_from_flat_or_split(kaggle_dir)
            copy_images(grouped, output_dir, alias_map, "kaggle", counters, source_log)
            print(f"  Kaggle done: {len(grouped)} classes, {sum(len(v) for v in grouped.values())} images")
        else:
            print(f"[WARN] Kaggle dir not found: {kaggle_dir}")

    write_stats(output_dir, source_log, Path(args.stats_path))
    print(f"\nUnified dataset written to: {output_dir}")
    print("Next step:")
    print(
        f"  python scripts/prepare_stratified_split.py "
        f"--input-dir {output_dir} "
        f"--output-dir data/processed/unified "
        f"--train-ratio 0.70 --val-ratio 0.15 --seed 42 --overwrite"
    )


if __name__ == "__main__":
    main()
