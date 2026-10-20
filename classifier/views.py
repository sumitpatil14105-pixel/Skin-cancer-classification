import time
import uuid
from pathlib import Path

from django.conf import settings
from django.shortcuts import render
from PIL import Image

from ml.classifier import ModelNotTrained, load_model, predict
from ml.pipeline import run_pipeline

from .forms import UploadForm


def _save(arr, name: str, folder: Path, uid: str) -> str:
    """Save an array as PNG under MEDIA_ROOT and return its URL."""
    fname = f"{uid}_{name}.png"
    Image.fromarray(arr).save(folder / fname)
    return f"{settings.MEDIA_URL}results/{fname}"


def _purge_old(folder: Path, max_age_s: int = 3600) -> None:
    """Uploaded images are sensitive: delete saved result images after an hour."""
    cutoff = time.time() - max_age_s
    for p in folder.glob("*.png"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def _model_status():
    try:
        load_model()
        return True
    except ModelNotTrained:
        return False


def index(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            return _analyse(request, form.cleaned_data["image"])
    else:
        form = UploadForm()
    return render(request, "classifier/index.html",
                  {"form": form, "model_ready": _model_status()})


def _analyse(request, upload):
    try:
        result = run_pipeline(upload)
    except Exception:
        form = UploadForm()
        form.add_error("image", "This file could not be processed as an image.")
        return render(request, "classifier/index.html",
                      {"form": form, "model_ready": _model_status()})

    folder = Path(settings.MEDIA_ROOT) / "results"
    folder.mkdir(parents=True, exist_ok=True)
    _purge_old(folder)
    uid = uuid.uuid4().hex[:12]

    hair_vis = result.original.copy()
    hair_vis[result.hair_mask] = (0, 255, 255)

    images = [
        ("Original", _save(result.original, "original", folder, uid)),
        ("Hair detected (Radon)", _save(hair_vis, "hair", folder, uid)),
        ("Hair & noise removed", _save(result.cleaned, "cleaned", folder, uid)),
        ("Binary mask", _save(result.mask, "mask", folder, uid)),
        ("Segmented lesion", _save(result.segmented, "segmented", folder, uid)),
        ("Contour overlay", _save(result.overlay, "overlay", folder, uid)),
    ]

    prediction, error = None, None
    if not result.seg_ok:
        error = ("The lesion could not be segmented reliably in this image, "
                 "so no prediction is shown. Try a centred, well-lit dermoscopic photo.")
    else:
        try:
            prediction = predict(result.features)
            prediction["pct_malignant"] = round(prediction["p_malignant"] * 100, 1)
            prediction["pct_benign"] = round(prediction["p_benign"] * 100, 1)
        except ModelNotTrained as exc:
            error = str(exc)

    return render(request, "classifier/result.html",
                  {"images": images, "prediction": prediction, "error": error})
