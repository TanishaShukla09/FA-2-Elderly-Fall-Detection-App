"""Download the UR Fall Detection Dataset (RGB frames) into dataset/urfall/.

Sources: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
Downloads, for each sequence, the camera-0 RGB frame zip (PNG frames) and,
for fall sequences, the body-height csv used to locate the fall moment.

Layout produced:
  dataset/urfall/fall/01/rgb/fall-01-cam0-rgb-001.png ...
  dataset/urfall/fall/01/height.csv
  dataset/urfall/adl/01/rgb/adl-01-cam0-rgb-001.png ...

Usage:
  python download_urfall.py            # falls + ADL
  python download_urfall.py --falls    # only fall sequences
  python download_urfall.py --workers 6 --limit 5
"""
import argparse
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data"
ROOT = Path(__file__).parent
RAW_DIR = ROOT / "dataset" / "urfall_raw"
OUT_DIR = ROOT / "dataset" / "urfall"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"


def http_get(url, dest, timeout=600):
    headers = {"User-Agent": USER_AGENT}
    existing = dest.stat().st_size if dest.exists() else 0
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        if r.status_code == 416:
            return  # already fully downloaded
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        mode = "ab" if existing else "wb"
        written = 0
        with open(dest, mode) as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
        if total and existing + written < total:
            raise ConnectionError(f"incomplete {existing + written}/{total}")


def download_file(url, dest, retries=6):
    for attempt in range(1, retries + 1):
        try:
            http_get(url, dest)
            if dest.exists() and dest.stat().st_size > 0:
                return True
        except Exception as e:
            print(f"    retry {attempt}/{retries} {Path(url).name}: {e}")
            time.sleep(2 * attempt)
    return False


def extract_zip(zip_path, target):
    target.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.endswith(".png")]
            extracted = list(target.rglob("*.png"))
            if len(extracted) == len(members):
                return
            for old in target.rglob("*.png"):
                old.unlink()
            for member in members:
                zf.extract(member, target)
    except zipfile.BadZipFile as e:
        print(f"    WARN bad zip {zip_path.name}: {e}")


def build_tasks(include_adl):
    tasks = []
    for i in range(1, 31):
        seq = f"fall-{i:02d}"
        tasks.append({
            "zip": f"{seq}-cam0-rgb.zip",
            "csv": f"{seq}-data.csv",
            "rel": f"fall/{i:02d}",
        })
    if include_adl:
        for i in range(1, 41):
            seq = f"adl-{i:02d}"
            tasks.append({
                "zip": f"{seq}-cam0-rgb.zip",
                "csv": None,
                "rel": f"adl/{i:02d}",
            })
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--falls", action="store_true", help="download only fall sequences")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="stop after N sequences (testing)")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(not args.falls)
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"UR Fall: {len(tasks)} sequences -> {OUT_DIR}")
    t0 = time.time()

    def ensure_zip(zip_path):
        for attempt in range(3):
            if not (zip_path.exists() and zip_path.stat().st_size > 1000):
                print(f"  downloading {zip_path.name}")
                if not download_file(f"{BASE_URL}/{zip_path.name}", zip_path):
                    return False
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    if any(m.endswith(".png") for m in zf.namelist()):
                        return True
            except zipfile.BadZipFile:
                pass
            print(f"  re-downloading {zip_path.name} (corrupt, {zip_path.stat().st_size // 1_000_000} MB)")
            zip_path.unlink(missing_ok=True)
        return False

    def one(t):
        zip_name = t["zip"]
        zip_path = RAW_DIR / zip_name
        if not ensure_zip(zip_path):
            return f"  FAILED {zip_name}"
        if t["csv"]:
            csv_path = OUT_DIR / t["rel"] / "height.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not (csv_path.exists() and csv_path.stat().st_size > 10):
                print(f"  downloading {t['csv']}")
                download_file(f"{BASE_URL}/{t['csv']}", csv_path)
        rgb_dir = OUT_DIR / t["rel"] / "rgb"
        extract_zip(zip_path, rgb_dir)
        n = len(list(rgb_dir.rglob("*.png")))
        return f"  {t['rel']}: {n} frames"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, t): t for t in tasks}
        for fut in as_completed(futs):
            print(fut.result())

    print(f"Done in {time.time() - t0:.0f}s. Data at {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
