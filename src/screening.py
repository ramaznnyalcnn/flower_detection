from __future__ import annotations

from pathlib import Path

import numpy as np

from src.feature_extraction import hsv_histogram, lbp_features


def _bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    """
    Bhattacharyya mesafesi — düşük değer = benzer dağılım.
    Her iki histogram normalize edilmiş (toplam=1) kabul edilir.
    """
    bc = np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None)))
    return float(-np.log(bc + 1e-10))


def _chi2(p: np.ndarray, q: np.ndarray) -> float:
    """Chi-kare mesafesi — düşük değer = benzer dağılım."""
    return float(np.sum((p - q) ** 2 / (p + q + 1e-10)))


class ClassProfiles:
    """
    Her sınıf için ortalama renk (HSV histogram) ve doku (LBP histogram)
    profilleri saklar ve sorgu görüntüsünü tüm sınıflarla karşılaştırır.

    Kullanım:
        profiles = ClassProfiles.load("models/class_profiles.npz")
        color_rank  = profiles.rank_color(masked_image)
        texture_rank = profiles.rank_texture(masked_image)
    """

    def __init__(
        self,
        classes: list[str],
        color_means: np.ndarray,   # (n_classes, color_dim)
        texture_means: np.ndarray, # (n_classes, texture_dim)
    ) -> None:
        self.classes = classes
        self.color_means = color_means.astype(np.float32)
        self.texture_means = texture_means.astype(np.float32)
        self._class_index = {c: i for i, c in enumerate(classes)}

    # ------------------------------------------------------------------
    # Kaydetme / yükleme
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            classes=np.array(self.classes),
            color_means=self.color_means,
            texture_means=self.texture_means,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ClassProfiles":
        data = np.load(path, allow_pickle=True)
        return cls(
            classes=list(data["classes"]),
            color_means=data["color_means"],
            texture_means=data["texture_means"],
        )

    # ------------------------------------------------------------------
    # Skorlama
    # ------------------------------------------------------------------
    def _color_distances(self, image: np.ndarray) -> np.ndarray:
        """Sorgu görüntüsü ile her sınıf profili arasındaki Bhattacharyya mesafesi."""
        hist = hsv_histogram(image)
        return np.array([_bhattacharyya(hist, self.color_means[i]) for i in range(len(self.classes))])

    def _texture_distances(self, image: np.ndarray) -> np.ndarray:
        """Sorgu görüntüsü ile her sınıf profili arasındaki Chi-kare mesafesi."""
        lbp = lbp_features(image)
        return np.array([_chi2(lbp, self.texture_means[i]) for i in range(len(self.classes))])

    def rank_color(self, image: np.ndarray) -> list[tuple[str, float, float]]:
        """
        Returns:
            [(class_name, distance, percentile), ...]  azalan percentile sırasında
            percentile: 0.0 = en iyi eşleşme, 100.0 = en kötü
        """
        dists = self._color_distances(image)
        return self._to_ranked(dists)

    def rank_texture(self, image: np.ndarray) -> list[tuple[str, float, float]]:
        """
        Returns:
            [(class_name, distance, percentile), ...]  azalan percentile sırasında
        """
        dists = self._texture_distances(image)
        return self._to_ranked(dists)

    def _to_ranked(self, dists: np.ndarray) -> list[tuple[str, float, float]]:
        n = len(dists)
        ranks = np.argsort(np.argsort(dists))  # 0=en küçük mesafe
        percentiles = ranks / (n - 1) * 100.0
        order = np.argsort(dists)
        return [
            (self.classes[i], float(dists[i]), float(percentiles[i]))
            for i in order
        ]

    def get_color_percentile(self, image: np.ndarray, label: str) -> tuple[float, float]:
        """Belirli bir sınıf için (distance, percentile) döndür."""
        dists = self._color_distances(image)
        idx = self._class_index.get(label)
        if idx is None:
            return float("inf"), 100.0
        n = len(dists)
        rank = int(np.sum(dists < dists[idx]))
        percentile = rank / (n - 1) * 100.0
        return float(dists[idx]), percentile

    def get_texture_percentile(self, image: np.ndarray, label: str) -> tuple[float, float]:
        """Belirli bir sınıf için (distance, percentile) döndür."""
        dists = self._texture_distances(image)
        idx = self._class_index.get(label)
        if idx is None:
            return float("inf"), 100.0
        n = len(dists)
        rank = int(np.sum(dists < dists[idx]))
        percentile = rank / (n - 1) * 100.0
        return float(dists[idx]), percentile
