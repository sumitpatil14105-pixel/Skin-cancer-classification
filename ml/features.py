from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import skew
from skimage import measure
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops

SHAPE_NAMES = [
    "area_ratio", "compactness", "solidity", "eccentricity", "extent",
    "major_minor_ratio", "asymmetry", "border_irregularity",
]
COLOR_NAMES = [f"{ch}_{stat}" for ch in ("R", "G", "B", "H", "S", "V")
               for stat in ("mean", "std", "skew")]
CONTRAST_NAMES = ["contrast_R", "contrast_G", "contrast_B"]
TEXTURE_NAMES = ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
                 "glcm_energy", "glcm_correlation", "gray_entropy"]
FEATURE_NAMES = SHAPE_NAMES + COLOR_NAMES + CONTRAST_NAMES + TEXTURE_NAMES


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def _axis(props, which: str) -> float:
    # property was renamed in scikit-image 0.19
    name = f"axis_{which}_length"
    return float(getattr(props, name, None) if hasattr(props, name)
                 else getattr(props, f"{which}_axis_length"))


def _safe_skew(x: np.ndarray) -> float:
    return float(skew(x)) if x.size > 2 and x.std() > 1e-8 else 0.0


def extract_features(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a float vector aligned with FEATURE_NAMES (zeros if no lesion)."""
    m = mask > 0
    h, w = m.shape
    if m.sum() < 50:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    props = measure.regionprops(m.astype(np.uint8))[0]
    area = float(props.area)
    perim = float(props.perimeter)

    minr, minc, maxr, maxc = props.bbox
    crop = m[minr:maxr, minc:maxc]
    asym = 0.5 * ((1 - _iou(crop, crop[:, ::-1])) + (1 - _iou(crop, crop[::-1, :])))
    convex_perim = float(measure.perimeter(props.image_convex)) or 1.0

    shape = [
        area / (h * w),
        4 * math.pi * area / (perim ** 2 + 1e-9),
        float(props.solidity),
        float(props.eccentricity),
        float(props.extent),
        float(_axis(props, 'major') / (_axis(props, 'minor') + 1e-9)),
        asym,
        perim / convex_perim,
    ]

    # colour statistics inside the lesion
    img = rgb.astype(np.float64) / 255.0
    hsv = rgb2hsv(img)
    colour = []
    for arr in (img[..., 0], img[..., 1], img[..., 2],
                hsv[..., 0], hsv[..., 1], hsv[..., 2]):
        v = arr[m]
        colour += [float(v.mean()), float(v.std()), _safe_skew(v)]

    # lesion-vs-surrounding-skin contrast
    ring = ndi.binary_dilation(m, iterations=25) & ~ndi.binary_dilation(m, iterations=5)
    if ring.sum() < 50:
        ring = ~m
    contrast = [float(img[..., c][m].mean() - img[..., c][ring].mean())
                for c in range(3)] if ring.any() else [0.0, 0.0, 0.0]

    # texture (GLCM) on the lesion bounding box
    gray = rgb2gray(img)
    box = gray[minr:maxr, minc:maxc].copy()
    box[~crop] = gray[m].mean()
    q = np.clip((box * 31).astype(np.uint8), 0, 31)
    glcm = graycomatrix(q, distances=[1, 3],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    texture = [float(graycoprops(glcm, p).mean()) for p in
               ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")]
    texture.append(float(measure.shannon_entropy(q[crop])))

    return np.array(shape + colour + contrast + texture, dtype=np.float64)
