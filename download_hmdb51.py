"""Download HMDB51 (ZIP repack from Hugging Face) and extract the activity
classes we need into dataset/hmdb51/<Activity>/<clip>.avi.

Source: https://huggingface.co/datasets/MichiganNLP/hmdb
File:   hmdb51_org.zip (~4 GB)

Usage:
  python download_hmdb51.py
  python download_hmdb51.py --chunks 8 --limit-classes 5
"""
import argparse
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

HF_URL = "https://huggingface.co/datasets/MichiganNLP/hmdb/resolve/main/hmdb51_org.zip"
ROOT = Path(__file__).parent
RAW_DIR = ROOT / "dataset" / "hmdb51_raw"
OUT_DIR = ROOT / "dataset" / "hmdb51"

# HMDB51 class folder -> our activity label
CLASS_MAP = {
    "walk": "Walking",
    "run": "Running",
    "jump": "Jumping",
    "climb_stairs": "Climbing Stairs",
    "sit": "Sitting",
    "stand": "Getting Up",
    "fall_floor": "Falling",
}


def get_total_size():
    with requests.get(HF_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        return int(r.headers.get("Content-Length", 0))


def download_range(url, start, end, dest, retries=4):
    dest = Path(dest)
    have = dest.stat().st_size if dest.exists() else 0
    if have >= end - start:
        return
    headers = {"Range": f"bytes={start + have}-{end - 1}"}
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=600) as r:
                if r.status_code in (200, 206):
                    mode = "ab" if have else "wb"
                    with open(dest, mode) as f:
                        for chunk in r.iter_content(1 << 20):
                            if chunk:
                                f.write(chunk)
                    return
                elif r.status_code == 416:
                    return
        except Exception as e:
            print(f"    chunk retry {attempt}/{retries}: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed chunk {start}-{end}")


def download_parallel(zip_path, total, chunks):
    if zip_path.exists() and zip_path.stat().st_size >= total:
        print(f"  {zip_path.name} already complete ({total / 1e9:.2f} GB)")
        return
    bounds = []
    step = total // chunks
    for i in range(chunks):
        start = i * step
        end = total if i == chunks - 1 else (i + 1) * step
        bounds.append((start, end))
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    last = [time.time(), 0]
    with ThreadPoolExecutor(max_workers=chunks) as ex:
        futs = {ex.submit(download_range, HF_URL, s, e, Path(f"{zip_path}.part{i}")): i
                for i, (s, e) in enumerate(bounds)}
        for fut in as_completed(futs):
            fut.result()
    # assemble
    with open(zip_path, "wb") as out:
        for i, (_, _) in enumerate(bounds):
            part = Path(f"{zip_path}.part{i}")
            with open(part, "rb") as p:
                out.write(p.read())
            part.unlink()
    print(f"  downloaded {total / 1e9:.2f} GB in {time.time() - t0:.0f}s")
    if zip_path.stat().st_size != total:
        print(f"  WARN size mismatch {zip_path.stat().st_size} != {total}")


def normalize(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def discover_classes(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    parts = set()
    for n in names:
        p = Path(n)
        if len(p.parts) >= 2 and p.suffix.lower() in (".avi", ".mp4"):
            parts.add(p.parts[1])
    return names, parts


def extract_selected(zip_path, limit_classes=0):
    names, parts = discover_classes(zip_path)
    selected = {}
    for folder in parts:
        key = normalize(folder)
        if key in CLASS_MAP:
            selected[folder] = CLASS_MAP[key]
    print(f"  found {len(parts)} class folders, selecting {len(selected)}: "
          f"{sorted(CLASS_MAP.keys())}")
    if limit_classes:
        selected = dict(list(selected.items())[:limit_classes])

    out_count = {}
    with zipfile.ZipFile(zip_path) as zf:
        for folder, activity in selected.items():
            target = OUT_DIR / activity
            target.mkdir(parents=True, exist_ok=True)
            count = 0
            for n in names:
                p = Path(n)
                if len(p.parts) >= 2 and p.parts[1] == folder and p.suffix.lower() in (".avi", ".mp4"):
                    dest = target / p.name
                    if not dest.exists():
                        with zf.open(n) as src, open(dest, "wb") as out:
                            out.write(src.read())
                    count += 1
            out_count[activity] = count
            print(f"  {activity} <- {folder}: {count} clips")
    return out_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--limit-classes", type=int, default=0)
    ap.add_argument("--skip-download", action="store_true", help="only extract from existing zip")
    args = ap.parse_args()

    zip_path = RAW_DIR / "hmdb51_org.zip"
    total = get_total_size()
    print(f"HMDB51 zip: {total / 1e9:.2f} GB -> {zip_path}")
    if not args.skip_download:
        download_parallel(zip_path, total, args.chunks)

    print("Extracting selected classes...")
    extract_selected(zip_path, args.limit_classes)
    print(f"Done. Data at {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
