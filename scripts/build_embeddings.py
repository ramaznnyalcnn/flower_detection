"""
Eğitim setindeki tüm görseller için penultimate embedding'leri hesaplar ve
diske kaydeder. Inference'ta KNN-tabanlı yeniden sıralama için kullanılır.

Kullanım:
    python scripts/build_embeddings.py --model models/resnet50.pt
    python scripts/build_embeddings.py --model models/resnet50.pt --limit 5

Çıktı:
    models/train_embeddings.npz   (embeddings, labels, architecture)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import EmbeddingIndex, extract_embedding

TRAIN_DIR = ROOT / "data" / "processed" / "oxford102_70_15_15" / "train"
OUT_PATH = ROOT / "models" / "train_embeddings.npz"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect_images(class_dir: Path) -> list[Path]:
    return [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]


def build_embeddings(
    train_dir: Path,
    model_path: Path,
    out_path: Path = OUT_PATH,
    limit_per_class: int | None = None,
    prefer_cuda: bool = True,
) -> EmbeddingIndex:
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"Sınıf klasörü bulunamadı: {train_dir}")

    print(f"Toplam {len(class_dirs)} sınıf bulundu.")
    if limit_per_class:
        print(f"Sınıf başına ilk {limit_per_class} görsel kullanılacak.")

    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    architecture: str | None = None

    for i, class_dir in enumerate(class_dirs):
        images = _collect_images(class_dir)
        if limit_per_class:
            images = images[:limit_per_class]
        if not images:
            continue

        for img_path in images:
            try:
                emb, arch = extract_embedding(img_path, model_path, prefer_cuda=prefer_cuda)
                if architecture is None:
                    architecture = arch
                embeddings.append(emb)
                labels.append(class_dir.name)
            except Exception as exc:
                print(f"  [HATA] {img_path.name}: {exc}")

        if (i + 1) % 10 == 0 or (i + 1) == len(class_dirs):
            print(f"  {i + 1}/{len(class_dirs)} sınıf işlendi ({len(embeddings)} embedding)...")

    if not embeddings:
        raise RuntimeError("Hiç embedding çıkarılamadı.")

    index = EmbeddingIndex(
        embeddings=np.stack(embeddings),
        labels=labels,
        architecture=architecture or "resnet50",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(out_path)
    print(f"\nEmbedding indexi kaydedildi: {out_path}")
    print(f"Toplam: {len(embeddings)} örnek, {len(set(labels))} sınıf, dim={index.embeddings.shape[1]}")
    return index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eğitim seti embedding builder")
    parser.add_argument("--model", type=Path, required=True, help="CNN checkpoint (.pt)")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Sınıf başına maksimum görsel")
    parser.add_argument("--cpu", action="store_true", help="GPU yerine CPU kullan")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_embeddings(
        args.train_dir,
        args.model,
        out_path=args.out,
        limit_per_class=args.limit,
        prefer_cuda=not args.cpu,
    )
