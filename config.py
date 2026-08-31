"""Centralized configuration for the Elderly Fall Detection system.

All paths, thresholds and window sizes live here so the Streamlit app, the
feature extraction and the training scripts can never drift apart. Change a
value once and every consumer picks it up.
"""
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / "dataset"
FEATURES_DIR = PROJECT_DIR / "features"
MODEL_DIR = PROJECT_DIR / "models"
SCREENSHOT_DIR = PROJECT_DIR / "screenshots"

# ------------------------------------------------------------------
# Model paths
# ------------------------------------------------------------------
FRAME_MODEL_PATH = MODEL_DIR / "fall_model.pkl"        # classical multi-class + binary fall
SEQUENCE_MODEL_PATH = MODEL_DIR / "fall_sequence_model.keras"  # LSTM/GRU fall detector

POSE_LITE_MODEL = "models/pose_landmarker_lite.task"
POSE_FULL_MODEL = "models/pose_landmarker_full.task"
POSE_URL_LITE = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                 "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
POSE_URL_FULL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                 "pose_landmarker_full/float16/1/pose_landmarker_full.task")

# ------------------------------------------------------------------
# Temporal / sequence windows
# ------------------------------------------------------------------
FEATURE_WINDOW = 30        # frames of history used by TemporalWindowExtractor
SEQUENCE_LEN = 20          # frames per input sequence for the LSTM/GRU detector
SEQUENCE_STRIDE = 10       # stride between sequence windows while building data

# ------------------------------------------------------------------
# Live-camera performance
# ------------------------------------------------------------------
# Pose coordinates are normalized, so a smaller inference image does not
# change the feature layout.  512 px is a much better latency/accuracy tradeoff
# for a browser webcam than processing every 640 px frame on the CPU.
LIVE_POSE_MAX_SIDE = 448
# Keep browser capture close to the CPU inference rate.  Without a cap, a
# 30 FPS camera can queue frames faster than Windows CPU inference consumes
# them, which makes the preview appear several seconds behind reality.
LIVE_CAMERA_FPS = 12
# The browser can deliver frames faster than pose inference finishes on a
# Streamlit Cloud CPU.  Analyse a bounded number of frames and let WebRTC
# display the others immediately; this keeps the preview responsive instead
# of accumulating a growing processing queue.
LIVE_PROCESS_FPS = 8
# Public STUN server for WebRTC candidate discovery. It lets the browser
# establish the webcam stream after deployment, rather than relying only on
# local host candidates.
WEBRTC_STUN_URL = "stun:stun.l.google.com:19302"
# The LSTM sees a 20-frame sequence.  Re-evaluating the same sliding sequence
# on every camera frame is costly and provides almost no additional signal.
# Keep adding every frame to its history, but score it at 7-8 Hz on a 30 FPS
# camera.  This caps the alarm decision delay to roughly 0.13 seconds.
LIVE_LSTM_PREDICTION_STRIDE = 4

# ------------------------------------------------------------------
# Fall alarm logic (app)
# ------------------------------------------------------------------
DEFAULT_FALL_THRESHOLD = 0.55      # used only when the model has no tuned threshold
SEQUENCE_LSTM_THRESHOLD = 0.32     # tuned on validation (max val F2) for the LSTM fall detector
CONFIRM_FRAMES = 10                # consecutive suspicious frames before the alarm fires
STATIONARY_CONFIRM_S = 1.5         # seconds of near-zero motion after impact to confirm a fall
STILL_SPEED = 0.02                 # normalized hip speed below this counts as "still"

# ------------------------------------------------------------------
# Record & Train
# ------------------------------------------------------------------
RECORDED_DIR = DATASET_DIR / "recorded"
RECORD_DEFAULT_SECONDS = 10
RECORD_VIDEO_FPS = 30
RECORD_VIDEO_SIZE = (1280, 720)
