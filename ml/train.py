
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FEATURE_NAMES
from .pipeline import features_from_path

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_folder(local_dir: Path):
    rows = []
    for label, name in ((0, "benign"), (1, "malignant")):
        for p in sorted((local_dir / name).rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                rows.append((str(p), label))
    return rows


def build_matrix(rows, n_jobs: int):
    feats = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(features_from_path)(p) for p, _ in rows)
    keep = [i for i, f in enumerate(feats) if f is not None and np.isfinite(f).all()]
    X = np.stack([feats[i] for i in keep])
    y = np.array([rows[i][1] for i in keep])
    paths = [rows[i][0] for i in keep]
    print(f"usable images: {len(keep)}/{len(rows)}")
    return X, y, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["isic", "folder"], default="isic")
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--size", default="thumbnail_256",
                    help='ISIC file variant: "thumbnail_256" or "full"')
    ap.add_argument("--extra-query", default="", help="extra ISIC search filter")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--out", default=str(ROOT / "models" / "svm_model.joblib"))
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "isic":
        from .isic_api import download_dataset
        rows = download_dataset(data_dir, args.per_class, args.size, args.extra_query)
    else:
        if not args.local_dir:
            ap.error("--local-dir is required with --source folder")
        rows = collect_folder(Path(args.local_dir))
    if len(rows) < 20:
        raise SystemExit(f"Only {len(rows)} labelled images found - need more to train.")

    X, y, _ = build_matrix(rows, args.jobs)
    np.savez_compressed(data_dir / "features.npz", X=X, y=y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced",
                    random_state=args.seed)),
    ])
    grid = {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", 0.1, 0.01, 0.001]}
    cv = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    print("best params:", search.best_params_, f"(cv AUC {search.best_score_:.3f})")

    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_te, proba)
    print(classification_report(y_te, pred, target_names=["benign", "malignant"]))
    print("confusion matrix [rows=true, cols=pred]:\n", confusion_matrix(y_te, pred))
    # print(f"test ROC-AUC: {auc:.3f}")

    report = classification_report(y_te, pred, target_names=["benign", "malignant"],
                                   output_dict=True)
    metrics = {
        "test_auc": float(auc),
        "test_accuracy": float(report["accuracy"]),
        "malignant_recall": float(report["malignant"]["recall"]),
        "malignant_precision": float(report["malignant"]["precision"]),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best, "feature_names": FEATURE_NAMES,
                 "threshold": 0.5, "metrics": metrics}, out)
    (out.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
