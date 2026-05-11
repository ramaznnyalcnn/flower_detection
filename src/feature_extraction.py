from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from src.data_utils import ImageRecord

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency branch
    cv2 = None

try:
    from scipy.stats import skew
except ImportError:  # pragma: no cover - optional dependency branch
    skew = None

try:
    from skimage import color
    from skimage.feature import hog, local_binary_pattern
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:  # pragma: no cover - old scikit-image spelling
        from skimage.feature import greycomatrix as graycomatrix
        from skimage.feature import greycoprops as graycoprops
except ImportError:  # pragma: no cover - optional dependency branch
    color = None
    hog = None
    local_binary_pattern = None
    graycomatrix = None
    graycoprops = None


DEFAULT_FEATURES = ("hsv", "hog")
AVAILABLE_FEATURES = ("hsv", "hog", "lbp", "glcm", "color_moments", "hu")


def load_rgb_image(path: str | Path, image_size: int = 224) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.asarray(Image.fromarray(image).convert("L"), dtype=np.uint8)


def hsv_histogram(image: np.ndarray, bins: tuple[int, int, int] = (16, 8, 8)) -> np.ndarray:
    if cv2 is not None:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ranges = ((0, 180), (0, 256), (0, 256))
    else:
        hsv = np.asarray(Image.fromarray(image).convert("HSV"), dtype=np.uint8)
        ranges = ((0, 256), (0, 256), (0, 256))

    hist, _ = np.histogramdd(hsv.reshape(-1, 3), bins=bins, range=ranges)
    hist = hist.astype(np.float32).ravel()
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def hog_features(image: np.ndarray) -> np.ndarray:
    if hog is None:
        raise ImportError("scikit-image is required for HOG features.")
    gray = rgb_to_gray(image)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype(np.float32)


def lbp_features(image: np.ndarray, points: int = 24, radius: int = 3) -> np.ndarray:
    if local_binary_pattern is None:
        raise ImportError("scikit-image is required for LBP features.")
    gray = rgb_to_gray(image)
    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def glcm_features(image: np.ndarray) -> np.ndarray:
    if graycomatrix is None or graycoprops is None:
        raise ImportError("scikit-image is required for GLCM features.")
    gray = rgb_to_gray(image)
    quantized = np.floor(gray / 32).astype(np.uint8)
    matrix = graycomatrix(
        quantized,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    values = [graycoprops(matrix, prop).ravel() for prop in props]
    return np.concatenate(values).astype(np.float32)


def color_moment_features(image: np.ndarray) -> np.ndarray:
    pixels = image.reshape(-1, 3).astype(np.float32)
    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)
    if skew is not None:
        skews = skew(pixels, axis=0, nan_policy="omit")
        skews = np.nan_to_num(skews)
    else:
        centered = pixels - means
        denom = np.power(stds + 1e-6, 3)
        skews = (centered ** 3).mean(axis=0) / denom
    return np.concatenate([means, stds, skews]).astype(np.float32)


def hu_moment_features(image: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return np.zeros(7, dtype=np.float32)
    gray = rgb_to_gray(image)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu.astype(np.float32)


def extract_features_from_array(
    image: np.ndarray,
    feature_types: Iterable[str] = DEFAULT_FEATURES,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for feature_type in feature_types:
        if feature_type == "hsv":
            parts.append(hsv_histogram(image))
        elif feature_type == "hog":
            parts.append(hog_features(image))
        elif feature_type == "lbp":
            parts.append(lbp_features(image))
        elif feature_type == "glcm":
            parts.append(glcm_features(image))
        elif feature_type == "color_moments":
            parts.append(color_moment_features(image))
        elif feature_type == "hu":
            parts.append(hu_moment_features(image))
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    return np.concatenate(parts).astype(np.float32)


def extract_features_from_path(
    path: str | Path,
    feature_types: Iterable[str] = DEFAULT_FEATURES,
    image_size: int = 224,
) -> np.ndarray:
    image = load_rgb_image(path, image_size=image_size)
    return extract_features_from_array(image, feature_types=feature_types)


@dataclass(frozen=True)
class FeatureExtractor:
    feature_types: tuple[str, ...] = DEFAULT_FEATURES
    image_size: int = 224

    def transform_path(self, path: str | Path) -> np.ndarray:
        return extract_features_from_path(path, self.feature_types, self.image_size)

    def transform_records(self, records: Iterable[ImageRecord], show_progress: bool = True) -> np.ndarray:
        records = list(records)
        iterator = records
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(records, desc="Extracting features")
            except ImportError:
                pass

        features = [self.transform_path(record.path) for record in iterator]
        if not features:
            raise ValueError("No features were extracted.")
        return np.vstack(features).astype(np.float32)


def validate_feature_names(feature_types: Iterable[str]) -> tuple[str, ...]:
    feature_types = tuple(feature_types)
    invalid = [name for name in feature_types if name not in AVAILABLE_FEATURES]
    if invalid:
        raise ValueError(f"Unknown feature(s): {', '.join(invalid)}")
    return feature_types

