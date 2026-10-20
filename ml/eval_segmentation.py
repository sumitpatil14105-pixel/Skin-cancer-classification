
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .pipeline import run_pipeline
from .preprocessing import IMG_SIZE


def load_mask(path) -> np.ndarray:
    m = ImageOps.exif_transpose(Image.open(path)).convert("L")
    w, h = m.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    m = m.crop((left, top, left + s, top + s)).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return np.asarray(m) > 127


def find_mask(masks_dir: Path, isic_id: str):
    hits = sorted(masks_dir.rglob(f"{isic_id}*"))
    hits = [h for h in hits if h.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif"}]
    if not hits:
        return None
    pref = [h for h in hits if "staple" in h.name.lower()]
    return (pref or hits)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    masks_dir = Path(args.masks)
    dices, ious = [], []
    for img in sorted(Path(args.images).glob("*.jpg"))[: args.limit]:
        mpath = find_mask(masks_dir, img.stem)
        if mpath is None:
            continue
        gt = load_mask(mpath)
        pred = run_pipeline(img).mask > 0
        inter = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        dices.append(2 * inter / max(gt.sum() + pred.sum(), 1))
        ious.append(inter / max(union, 1))
    if not dices:
        raise SystemExit("No image/mask pairs matched - check the file naming.")
    print(f"{len(dices)} images | Dice {np.mean(dices):.3f} +/- {np.std(dices):.3f}"
          f" | IoU {np.mean(ious):.3f} +/- {np.std(ious):.3f}")


if __name__ == "__main__":
    main()
