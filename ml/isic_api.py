
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import requests

BASE = os.environ.get("ISIC_API_BASE", "https://api.isic-archive.com/api/v2").rstrip("/")
TOKEN = os.environ.get("ISIC_API_TOKEN")


def _session() -> requests.Session:
    s = requests.Session()
    if TOKEN:
        s.headers["Authorization"] = f"Bearer {TOKEN}"
    return s


def label_of(record: dict):
    """0 = benign, 1 = malignant, None = unusable."""
    clin = (record.get("metadata") or {}).get("clinical") or {}
    value = clin.get("diagnosis_1") or clin.get("benign_malignant")
    if not value:
        return None
    value = str(value).strip().lower()
    if value.startswith("malignant"):
        return 1
    if value.startswith("benign"):
        return 0
    return None


def search(diagnosis_1: str, limit: int, extra_query: str = "", session=None):
    """Yield up to `limit` image records whose diagnosis_1 matches."""
    session = session or _session()
    query = f'diagnosis_1:"{diagnosis_1}"'
    if extra_query:
        query += f" AND {extra_query}"
    url = f"{BASE}/images/search/"
    params = {"query": query, "limit": min(100, limit)}
    got = 0
    while url and got < limit:
        r = session.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        for rec in data.get("results", []):
            yield rec
            got += 1
            if got >= limit:
                return
        url, params = data.get("next"), None  # `next` is a full URL


def _download(session, url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return False


def download_dataset(out_dir, per_class: int = 500, size: str = "thumbnail_256",
                     extra_query: str = ""):
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
    rows = []
    for diagnosis in ("Benign", "Malignant"):
        n = 0
        for rec in search(diagnosis, per_class, extra_query, session):
            lab = label_of(rec)
            files = rec.get("files") or {}
            url = (files.get(size) or files.get("full") or {}).get("url")
            if lab is None or not url:
                continue
            dest = img_dir / f"{rec['isic_id']}.jpg"
            if dest.exists() or _download(session, url, dest):
                rows.append((str(dest), lab))
                n += 1
        print(f"{diagnosis}: {n} images")
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)
    return rows
