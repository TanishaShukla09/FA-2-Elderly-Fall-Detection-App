"""
Full automatic pipeline for the Elderly-Fall-Detection project.

    python pipeline.py                 # run every step
    python pipeline.py --limit 2       # quick test on 2 IMVIA videos
    python pipeline.py --skip-nn       # skip the Keras step
    python pipeline.py --skip-features # skip the 10-class extract_feature step

Steps
  [0] ensure_pose_landmarker  -> models/pose_landmarker_lite.task  (auto-download if missing)
  [1] extract_imvia           -> features/imvia_features.npz        (videos -> pose features)
  [2] extract_feature         -> features/features.npz              (10-class, existing images)
  [3] train_model             -> models/fall_model.pkl + plots      (RandomForest, 10-class)
  [4] train_nn                -> models/fall detection.h5, history.pkl, threshold.pkl,
                                 label_encoder.pkl, x_Test.npy, ytest.npy,
                                 training curves / confusion matrix / classification report
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(script, *args):
    cmd = [PY, os.path.join(ROOT, script), *args]
    print(f"\n>>> Running: {' '.join(cmd)}\n", flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(f"Step failed ({script}) with exit code {rc}")


def main():
    parser = argparse.ArgumentParser(description="Run the full fall-detection pipeline")
    parser.add_argument("--skip-landmarker", action="store_true")
    parser.add_argument("--skip-imvia", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-sklearn", action="store_true")
    parser.add_argument("--skip-nn", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N IMVIA videos")
    parser.add_argument("--frame-step", type=int, default=3, help="IMVIA frame sampling step")
    parser.add_argument("--save-frames", action="store_true", help="Dump IMVIA JPEGs")
    parser.add_argument("--only", type=int, nargs="*", default=[], help="Only these IMVIA video numbers")
    args = parser.parse_args()

    imvia_args = []
    if args.limit:
        imvia_args += ["--limit", str(args.limit)]
    if args.frame_step != 3:
        imvia_args += ["--frame-step", str(args.frame_step)]
    if args.save_frames:
        imvia_args.append("--save-frames")
    if args.only:
        imvia_args += ["--only", *map(str, args.only)]

    if not args.skip_landmarker:
        run("ensure_pose_landmarker.py")
    if not args.skip_imvia:
        run("extract_imvia.py", *imvia_args)
    if not args.skip_features:
        run("extract_feature.py")
    if not args.skip_sklearn:
        run("train_model.py")
    if not args.skip_nn:
        run("train_nn.py")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - all artifacts generated under models/, features/, screenshots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
