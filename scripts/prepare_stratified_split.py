from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_by_class(input_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)

    split_dirs = [input_dir / split for split in SPLITS]
    if all(path.exists() for path in split_dirs):
        for split_dir in split_dirs:
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                for image_path in sorted(class_dir.rglob("*")):
                    if is_image(image_path):
                        grouped[class_dir.name].append(image_path)
        return grouped

    for class_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if is_image(image_path):
                grouped[class_dir.name].append(image_path)
    return grouped


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if total < 3:
        return total, 0, 0

    train_count = max(1, int(round(total * train_ratio)))
    val_count = max(1, int(round(total * val_ratio)))
    if train_count + val_count >= total:
        val_count = 1
        train_count = total - 2
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def copy_split(
    grouped: dict[str, list[Path]],
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, dict[str, int]]:
    rng = random.Random(seed)
    summary: dict[str, dict[str, int]] = {}

    for class_name in sorted(grouped):
        images = grouped[class_name][:]
        rng.shuffle(images)
        train_count, val_count, test_count = split_counts(len(images), train_ratio, val_ratio)
        split_items = {
            "train": images[:train_count],
            "val": images[train_count:train_count + val_count],
            "test": images[train_count + val_count:],
        }
        summary[class_name] = {split: len(paths) for split, paths in split_items.items()}

        for split, paths in split_items.items():
            class_output = output_dir / split / class_name
            class_output.mkdir(parents=True, exist_ok=True)
            for index, source_path in enumerate(paths):
                dest_name = f"{class_name}_{index:05d}{source_path.suffix.lower()}"
                shutil.copy2(source_path, class_output / dest_name)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic stratified image split.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Ratios must satisfy train > 0, val > 0, train + val < 1.")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    grouped = collect_by_class(input_dir)
    if not grouped:
        raise ValueError(f"No images found under {input_dir}")

    summary = copy_split(
        grouped=grouped,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    totals = {split: sum(counts[split] for counts in summary.values()) for split in SPLITS}
    print(f"Classes: {len(summary)}")
    print(f"Total images: {sum(totals.values())}")
    for split in SPLITS:
        print(f"{split}: {totals[split]}")
    print(f"Wrote: {output_dir}")


if __name__ == "__main__":
    main()

