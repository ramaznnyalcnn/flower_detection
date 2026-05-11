from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE


def compute_embedding(x: np.ndarray, y=None, method: str = "pca", random_state: int = 42) -> np.ndarray:
    method = method.lower()
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(x)
    if method == "lda":
        if y is None:
            raise ValueError("LDA requires labels.")
        return LinearDiscriminantAnalysis(n_components=2).fit_transform(x, y)
    if method == "tsne":
        return TSNE(n_components=2, init="pca", learning_rate="auto", random_state=random_state).fit_transform(x)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:  # pragma: no cover - optional dependency branch
            raise ImportError("umap-learn is required for UMAP visualizations.") from exc
        return umap.UMAP(n_components=2, random_state=random_state).fit_transform(x)
    raise ValueError(f"Unknown embedding method: {method}")


def save_embedding_plot(embedding: np.ndarray, labels, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(labels)
    classes = sorted(set(labels))
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    colors = np.asarray([class_to_index[label] for label in labels])

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=10, cmap="tab20", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if len(classes) <= 20:
        handles, _ = scatter.legend_elements()
        ax.legend(handles, classes, loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

