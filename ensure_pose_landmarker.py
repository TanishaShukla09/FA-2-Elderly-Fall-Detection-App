import urllib.request
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"
URL = ("https://storage.googleapis.com/mediapipe-models/"
       "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")


def ensure_model(force: bool = False) -> Path:
    MODEL_PATH.parent.mkdir(exist_ok=True)
    if MODEL_PATH.exists() and not force:
        print(f"[ensure_pose_landmarker] OK: {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")
        return MODEL_PATH
    print(f"[ensure_pose_landmarker] Downloading PoseLandmarker Lite task -> {MODEL_PATH}")
    urllib.request.urlretrieve(URL, MODEL_PATH)
    print(f"[ensure_pose_landmarker] Saved {MODEL_PATH.stat().st_size} bytes")
    return MODEL_PATH


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()
    ensure_model(force=args.force)
