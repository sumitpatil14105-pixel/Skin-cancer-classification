import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import measure, morphology, filters
from skimage.transform import iradon, radon

IMG_SIZE = 256
THETAS = np.arange(0.0, 180.0, 1.0)


# 1. Loading
def load_image(src, size: int = IMG_SIZE) -> np.ndarray:
    img = Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


# 2. Enhanced Hair Removal using Multi-Scale Morphology & Frangi Filter
def _hair_candidates(gray: np.ndarray) -> np.ndarray:
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) 
        for k in (5, 9, 13)
    ]
    blackhat_imgs = [cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k) for k in kernels]
    blackhat = np.max(np.stack(blackhat_imgs, axis=0), axis=0)
    
    otsu, _ = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return blackhat > max(float(otsu), 10.0)


def _drop_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.concatenate([[False], sizes >= min_size])
    return keep[lab]


def detect_hair_mask(rgb: np.ndarray, min_line_len: float = 30.0) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cand = _hair_candidates(gray)
    if cand.sum() < 20:
        return np.zeros_like(cand, dtype=bool)


    norm_gray = gray.astype(np.float32) / 255.0
    vesselness = filters.frangi(norm_gray, sigmas=range(1, 4, 1), black_ridges=True)
    
    vessel_mask = vesselness > np.percentile(vesselness, 92)

    hair = cand & vessel_mask
    
    hair = _drop_small(hair, 10)
    hair = ndi.binary_dilation(hair, structure=np.ones((3, 3)), iterations=1)
    hair = ndi.binary_closing(hair, structure=np.ones((2, 2)))
    
    return hair


def remove_hair(rgb: np.ndarray):
    hair = detect_hair_mask(rgb)
    if not hair.any():
        return rgb.copy(), hair
    cleaned = cv2.inpaint(rgb, hair.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    return cleaned, hair


# 3. Radon-domain denoising
def denoise_radon(rgb: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    h = rgb.shape[0]
    out = np.empty_like(rgb)
    for c in range(3):
        chan = rgb[..., c].astype(np.float32)
        sino = radon(chan, theta=THETAS, circle=False)
        sino = ndi.gaussian_filter1d(sino, sigma=sigma, axis=0)
        rec = iradon(sino, theta=THETAS, output_size=h,
                     filter_name="hann", circle=False)
        if rec.std() > 1e-6 and chan.std() > 1e-6:
            rec = (rec - rec.mean()) / rec.std() * chan.std() + chan.mean()
        out[..., c] = np.clip(rec, 0, 255).astype(np.uint8)
    return out


# 4. Binary segmentation
def _vignette_mask(gray: np.ndarray) -> np.ndarray:
    dark = gray < 25
    lab = measure.label(dark, connectivity=1)
    h, w = gray.shape
    out = np.zeros_like(dark)
    for corner in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        l = lab[corner]
        if l:
            out |= lab == l
    return out


def segment_lesion(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    vig = _vignette_mask(blur)
    valid = ~vig

    t, _ = cv2.threshold(blur[valid].reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (blur <= t) & valid

    mask = ndi.binary_opening(mask, structure=morphology.disk(3))
    mask = ndi.binary_closing(mask, structure=morphology.disk(5))

    lab = measure.label(mask, connectivity=2)
    if lab.max() == 0:
        return np.zeros_like(gray), False

    h, w = gray.shape
    centre = np.array([h / 2, w / 2])
    best, best_score = 0, -1.0
    for r in measure.regionprops(lab):
        d = np.linalg.norm(np.array(r.centroid) - centre) / (0.5 * h)
        score = r.area * np.exp(-(d ** 2))
        if score > best_score:
            best, best_score = r.label, score

    mask = ndi.binary_fill_holes(lab == best)
    frac = mask.mean()
    ok = 0.01 <= frac <= 0.90
    return (mask.astype(np.uint8) * 255), ok