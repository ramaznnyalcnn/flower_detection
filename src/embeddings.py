"""
Penultimate (son sınıflama katmanından önceki) görüntü embedding'leri.

Kullanım:
    # Eğitim seti embedding'lerini oluştur (bir kere):
    python scripts/build_embeddings.py

    # Inference'ta KNN reranking:
    index = EmbeddingIndex.load("models/train_embeddings.npz")
    knn_probs = index.knn_class_softmax("ornek.jpg", "models/resnet50.pt", k=20)

Tasarım notu:
    - ResNet50 için 2048-d (`model.avgpool` çıkışı, FC öncesi)
    - EfficientNet-B0 için 1280-d (`classifier[0]` öncesi)
    - ConvNeXt-Tiny için 768-d (avgpool çıkışı)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.cnn_model import build_transforms, load_cnn_checkpoint


def _penultimate_module(model, architecture: str):
    """
    Modelin son sınıflama katmanından önceki feature'ları yakalayan modülü döndür.
    `register_forward_hook` ile bağlanır.
    """
    if architecture == "resnet50":
        return model.avgpool
    if architecture == "efficientnet_b0":
        return model.avgpool
    if architecture == "convnext_tiny":
        return model.avgpool
    raise ValueError(f"Bilinmeyen architecture: {architecture}")


def extract_embedding(
    image_path: str | Path,
    checkpoint_path: str | Path,
    prefer_cuda: bool = True,
) -> tuple[np.ndarray, str]:
    """
    Tek bir görselin penultimate embedding'ini çıkarır.

    Returns:
        (embedding, architecture)
            embedding: 1-D L2-normalized float32 dizi
            architecture: cosine similarity için
    """
    import torch

    model, checkpoint, device = load_cnn_checkpoint(checkpoint_path, prefer_cuda=prefer_cuda)
    architecture = checkpoint.get("architecture", "resnet50")
    transform = build_transforms(checkpoint.get("image_size", 224), train=False)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    feats: list[torch.Tensor] = []

    def _hook(_module, _inputs, output):
        feats.append(output.detach().flatten(start_dim=1))

    handle = _penultimate_module(model, architecture).register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(tensor)
    finally:
        handle.remove()

    emb = feats[0].cpu().numpy().astype(np.float32).ravel()
    norm = np.linalg.norm(emb) + 1e-10
    return emb / norm, architecture


class EmbeddingIndex:
    """
    Eğitim seti embedding'leri için basit cosine-similarity KNN dizini.
    """

    def __init__(self, embeddings: np.ndarray, labels: list[str], architecture: str) -> None:
        # Embeddings (n_samples, dim) — L2-normalized
        self.embeddings = embeddings.astype(np.float32)
        self.labels = np.asarray(labels)
        self.architecture = architecture
        self.classes = sorted(set(labels))

    # ------------------------------------------------------------------
    # Disk
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            embeddings=self.embeddings,
            labels=self.labels,
            architecture=np.array(self.architecture),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingIndex":
        data = np.load(path, allow_pickle=True)
        return cls(
            embeddings=data["embeddings"],
            labels=list(data["labels"]),
            architecture=str(data["architecture"]),
        )

    # ------------------------------------------------------------------
    # Sorgu
    # ------------------------------------------------------------------
    def knn_class_softmax(
        self,
        image_path: str | Path,
        checkpoint_path: str | Path,
        k: int = 20,
        temperature: float = 10.0,
        prefer_cuda: bool = True,
    ) -> dict[str, float]:
        """
        Sorgu görselin en yakın k komşusunu bulur, komşuların sınıf dağılımını
        cosine-similarity ağırlıklı softmax ile döner.

        temperature: yüksek → daha keskin (model softmax'ına yakın)
        """
        query, arch = extract_embedding(image_path, checkpoint_path, prefer_cuda=prefer_cuda)
        if arch != self.architecture:
            raise ValueError(
                f"Embedding mimari uyumsuzluğu: index={self.architecture}, query={arch}"
            )

        sims = self.embeddings @ query  # cosine (her ikisi L2-normalized)
        top_k_idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
        top_sims = sims[top_k_idx]
        top_labels = self.labels[top_k_idx]

        # Sınıf başına sim-ağırlıklı softmax
        weights = np.exp(temperature * top_sims)
        weights /= weights.sum() + 1e-10

        probs: dict[str, float] = {cls: 0.0 for cls in self.classes}
        for lbl, w in zip(top_labels, weights):
            probs[str(lbl)] += float(w)
        return probs
