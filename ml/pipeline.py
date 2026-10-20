

from dataclasses import dataclass

import cv2
import numpy as np

from .features import extract_features
from .preprocessing import (denoise_radon, load_image, remove_hair,
                            segment_lesion)


@dataclass
class PipelineResult:
    original: np.ndarray       # RGB
    hair_mask: np.ndarray      # bool
    cleaned: np.ndarray        # RGB, hair + noise removed
    mask: np.ndarray           # uint8 {0,255}
    segmented: np.ndarray      # RGB, background blacked out
    overlay: np.ndarray        # RGB, lesion contour on cleaned image
    features: np.ndarray
    seg_ok: bool


def run_pipeline(src) -> PipelineResult:
    original = load_image(src)
    no_hair, hair_mask = remove_hair(original)
    cleaned = denoise_radon(no_hair)
    mask, ok = segment_lesion(cleaned)

    segmented = cleaned.copy()
    segmented[mask == 0] = 0

    overlay = cleaned.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    feats = extract_features(cleaned, mask)
    return PipelineResult(original, hair_mask, cleaned, mask, segmented,
                          overlay, feats, ok)


def features_from_path(path):
    """Helper for joblib workers: returns feature vector or None on failure."""
    try:
        return run_pipeline(path).features
    except Exception:  # unreadable / corrupt image
        return None
