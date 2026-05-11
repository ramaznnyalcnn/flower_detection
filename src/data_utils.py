from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_records(root: str | Path) -> list[ImageRecord]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    records: list[ImageRecord] = []
    class_dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    for class_dir in sorted(class_dirs):
        for image_path in sorted(class_dir.rglob("*")):
            if is_image_file(image_path):
                records.append(ImageRecord(image_path, class_dir.name))
    return records


def limit_records_per_class(
    records: Iterable[ImageRecord],
    limit: int | None,
    random_state: int = 42,
) -> list[ImageRecord]:
    records = list(records)
    if not limit or limit <= 0:
        return records

    rng = random.Random(random_state)
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    limited: list[ImageRecord] = []
    for label in sorted(grouped):
        samples = grouped[label]
        rng.shuffle(samples)
        limited.extend(samples[:limit])
    return sorted(limited, key=lambda item: (item.label, str(item.path)))


def _stratified_split(
    records: list[ImageRecord],
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> dict[str, list[ImageRecord]]:
    rng = random.Random(random_state)
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    splits = {"train": [], "val": [], "test": []}
    for label in sorted(grouped):
        samples = grouped[label][:]
        rng.shuffle(samples)
        n = len(samples)

        if n < 3:
            splits["train"].extend(samples)
            continue

        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1 if n > 3 else 0

        splits["test"].extend(samples[:n_test])
        splits["val"].extend(samples[n_test:n_test + n_val])
        splits["train"].extend(samples[n_test + n_val:])

    for split in splits:
        splits[split] = sorted(splits[split], key=lambda item: (item.label, str(item.path)))
    return splits


def load_dataset_splits(
    dataset_dir: str | Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> dict[str, list[ImageRecord]]:
    dataset_dir = Path(dataset_dir)
    split_dirs = {split: dataset_dir / split for split in DEFAULT_SPLITS}

    if all(path.exists() for path in split_dirs.values()):
        return {split: collect_image_records(path) for split, path in split_dirs.items()}

    records = collect_image_records(dataset_dir)
    return _stratified_split(records, val_ratio, test_ratio, random_state)


def ensure_records(records: list[ImageRecord], context: str) -> None:
    if not records:
        raise ValueError(f"No images found for {context}. Run the download and clean scripts first.")


def labels_from_records(records: Iterable[ImageRecord]) -> list[str]:
    return [record.label for record in records]


def paths_from_records(records: Iterable[ImageRecord]) -> list[Path]:
    return [record.path for record in records]


def class_names_from_records(records: Iterable[ImageRecord]) -> list[str]:
    return sorted({record.label for record in records})


def class_distribution(records: Iterable[ImageRecord]) -> dict[str, int]:
    return dict(sorted(Counter(record.label for record in records).items()))


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

