"""Walk the test split, run predict_pipeline on each image, log everything.

Usage:
    python scripts/run_eval.py \
        --model models/oxford102_70_15_15_cnn/resnet50_v2.pt \
        --data data/processed/oxford102_70_15_15/test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import predict_pipeline
from src.screening import ClassProfiles
from src.embeddings import EmbeddingIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/oxford102_70_15_15_cnn/resnet50_v2.pt",
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--data",
        default="data/processed/oxford102_70_15_15/test",
        help="ImageFolder-style directory (one subdir per class)",
    )
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="Max images per class (for quick checks)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data)
    model_path = Path(args.model)
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    # Pre-load profiles / embeddings once so we don't reload per image
    profiles_path = model_path.parent / "class_profiles.npz"
    if not profiles_path.exists():
        profiles_path = Path("models/class_profiles.npz")
    profiles = ClassProfiles.load(profiles_path) if profiles_path.exists() else None

    embeddings_path = model_path.parent / "train_embeddings.npz"
    if not embeddings_path.exists():
        embeddings_path = Path("models/train_embeddings.npz")
    embeddings_index = None
    if embeddings_path.exists():
        try:
            embeddings_index = EmbeddingIndex.load(embeddings_path)
        except Exception as exc:
            print(f"[uyarı] embeddings yüklenemedi: {exc}")

    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    total_imgs = 0
    correct_top1 = 0
    abstained = 0
    start = time.perf_counter()

    for cls_dir in class_dirs:
        ground_truth = cls_dir.name
        imgs = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        if args.limit_per_class:
            imgs = imgs[: args.limit_per_class]
        for img_path in imgs:
            total_imgs += 1
            results, _, _ = predict_pipeline(
                img_path, model_path,
                profiles=profiles,
                embeddings_index=embeddings_index,
                top_k=args.top_k,
                use_tta=not args.no_tta,
                log_source="eval",
                ground_truth=ground_truth,
            )
            if results:
                if results[0].label == ground_truth:
                    correct_top1 += 1
                if results[0].abstain:
                    abstained += 1
            if total_imgs % 50 == 0:
                elapsed = time.perf_counter() - start
                rate = total_imgs / elapsed
                print(f"  {total_imgs} images | top1 acc={correct_top1/total_imgs:.3f} | "
                      f"abstain={abstained/total_imgs:.3f} | {rate:.1f} img/s")

    elapsed = time.perf_counter() - start
    print()
    print(f"Total: {total_imgs} images in {elapsed:.1f}s ({total_imgs/elapsed:.1f} img/s)")
    if total_imgs:
        print(f"Top-1 accuracy: {correct_top1/total_imgs:.4f}")
        print(f"Abstain rate: {abstained/total_imgs:.4f}")
    print()
    print("All predictions logged to logs/predictions.db (source='eval').")
    print("Run: python scripts/analyze_predictions.py")


if __name__ == "__main__":
    main()
