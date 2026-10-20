from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "svm_model.joblib"
LABELS = {0: "Benign", 1: "Malignant"}


class ModelNotTrained(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_model(path: str = str(MODEL_PATH)):
    if not Path(path).exists():
        raise ModelNotTrained(
            "No trained model found. Run:  python -m ml.train  (see README)."
        )
    return joblib.load(path)


def predict(features: np.ndarray) -> dict:
    bundle = load_model()
    pipe = bundle["pipeline"]
    x = np.asarray(features, dtype=np.float64).reshape(1, -1)
    p_malignant = float(pipe.predict_proba(x)[0, 1])
    label = 1 if p_malignant >= bundle.get("threshold", 0.5) else 0
    return {
        "label": LABELS[label],
        "p_malignant": p_malignant,
        "p_benign": 1.0 - p_malignant,
        "metrics": bundle.get("metrics", {}),
    }
