import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_webrtc import webrtc_streamer
import av
import time
import tempfile
import os
import math
import sys
import subprocess
import json
import re
import io
import base64
import wave
import joblib
import threading
from pathlib import Path
from collections import deque
import urllib.request

from features import extract_features, TEMPORAL_FEATURE_NAMES, TemporalWindowExtractor, FEATURE_NAMES

import config


def _webrtc_rtc_configuration():
    """Return WebRTC ICE configuration without exposing private TURN credentials.

    Multiple public STUN servers are used for NAT discovery. A free public
    TURN relay (Metered open relay) is included as a fallback so the stream
    can connect through restrictive NATs on Streamlit Cloud. The owner can
    override with their own STUN/TURN via environment variables or Streamlit
    secrets (documented in secrets.toml.example).
    """
    stun_servers = [
        {"urls": config.WEBRTC_STUN_URL},
        {"urls": ["stun:stun1.l.google.com:19302",
                  "stun:stun2.l.google.com:19302",
                  "stun:stun.cloudflare.com:3478"]},
    ]
    turn_url = turn_username = turn_password = None
    try:
        turn_url = os.environ.get("TURN_URL") or st.secrets.get("TURN_URL", "")
        turn_username = os.environ.get("TURN_USERNAME") or st.secrets.get("TURN_USERNAME", "")
        turn_password = os.environ.get("TURN_PASSWORD") or st.secrets.get("TURN_PASSWORD", "")
    except Exception:
        turn_url = os.environ.get("TURN_URL", "")
        turn_username = os.environ.get("TURN_USERNAME", "")
        turn_password = os.environ.get("TURN_PASSWORD", "")
    if turn_url and turn_username and turn_password:
        ice_servers = stun_servers + [{"urls": turn_url, "username": turn_username,
                                       "credential": turn_password}]
    else:
        # Free public open relay (dev/fallback only). Swap for your own TURN
        # server via the environment/secret overrides above for production.
        ice_servers = stun_servers + [{
            "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        }]
    return {"iceServers": ice_servers}

# ══════════════════════════════════════════════════════════
# 1. LANDMARK DEFINITIONS (MediaPipe Pose - 33 landmarks)
# ══════════════════════════════════════════════════════════
LANDMARK_INDICES = {
    "nose": 0, "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8, "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20, "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28, "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

KEY_JOINT_NAMES = [
    ("Nose", 0), ("Left Shoulder", 11), ("Right Shoulder", 12),
    ("Left Elbow", 13), ("Right Elbow", 14), ("Left Wrist", 15), ("Right Wrist", 16),
    ("Left Hip", 23), ("Right Hip", 24), ("Left Knee", 25), ("Right Knee", 26),
    ("Left Ankle", 27), ("Right Ankle", 28), ("Left Heel", 29), ("Right Heel", 30),
    ("Left Foot", 31), ("Right Foot", 32),
]

# ══════════════════════════════════════════════════════════
# 2. REFERENCE POSTURE GUIDE  (biomechanical ranges for the
#    current detected posture vs. expected posture patterns)
# ══════════════════════════════════════════════════════════
ACTIVITIES = [
    "Standing", "Sitting", "Walking", "Running", "Falling",
    "Lying Down", "Bending", "Squatting", "Jumping", "Climbing Stairs",
    "Crawling", "Kneeling", "Crouching", "Getting Up",
]

REFERENCE_POSTURES = {
    "Standing": {
        "torso_angle": (0, 18), "knee_angle": (155, 180), "hip_angle": (160, 180),
        "elbow_angle": (160, 180), "hip_height": "high (stable)",
        "velocity": "very low", "foot_pattern": "both feet on ground",
    },
    "Sitting": {
        "torso_angle": (5, 35), "knee_angle": (70, 120), "hip_angle": (70, 120),
        "elbow_angle": (90, 160), "hip_height": "lower than standing",
        "velocity": "very low", "foot_pattern": "feet forward on ground",
    },
    "Walking": {
        "torso_angle": (5, 20), "knee_angle": "160-180 deg at stance",
        "hip_angle": (160, 180), "elbow_angle": "20-45 deg arm swing",
        "hip_height": "moderate, oscillates", "velocity": "moderate horizontal",
        "foot_pattern": "alternating - one foot grounded",
    },
    "Running": {
        "torso_angle": "10-30 deg forward lean", "knee_angle": "90-170 deg cycling",
        "hip_angle": (90, 170), "elbow_angle": (20, 50),
        "hip_height": "high, with flight phase", "velocity": "high",
        "foot_pattern": "flight phase - both feet leave ground",
    },
    "Falling": {
        "torso_angle": "rapidly increases 20->90", "knee_angle": "any",
        "hip_angle": "any", "elbow_angle": "any",
        "hip_height": "sudden drop", "velocity": "high downward",
        "foot_pattern": "loss of support",
    },
    "Lying Down": {
        "torso_angle": (75, 90), "knee_angle": (160, 180),
        "hip_angle": (170, 180), "elbow_angle": (150, 180),
        "hip_height": "very low, shoulder-hip-head aligned",
        "velocity": "very low", "foot_pattern": "horizontal body",
    },
    "Bending": {
        "torso_angle": (30, 90), "knee_angle": (150, 180),
        "hip_angle": (60, 120), "elbow_angle": (150, 180),
        "hip_height": "nearly stationary", "velocity": "low",
        "foot_pattern": "feet fixed, legs straight",
    },
    "Squatting": {
        "torso_angle": (10, 50), "knee_angle": (30, 90),
        "hip_angle": (40, 90), "elbow_angle": (150, 180),
        "hip_height": "moves downward", "velocity": "low",
        "foot_pattern": "feet fixed, deep knee bend",
    },
    "Jumping": {
        "torso_angle": (5, 30), "knee_angle": (90, 180),
        "hip_angle": (90, 180), "elbow_angle": (20, 60),
        "hip_height": "rises then falls", "velocity": "upward accel then landing",
        "foot_pattern": "both feet leave ground together",
    },
    "Climbing Stairs": {
        "torso_angle": (5, 35), "knee_angle": "alternating high/low",
        "hip_angle": "alternating", "elbow_angle": (20, 60),
        "hip_height": "gradually rises", "velocity": "forward motion",
        "foot_pattern": "one foot elevated on higher step",
    },
    "Crawling": {
        "torso_angle": (0, 30), "knee_angle": "very bent",
        "hip_angle": (80, 140), "elbow_angle": (20, 90),
        "hip_height": "low, near floor", "velocity": "slow horizontal",
        "foot_pattern": "hands and knees on floor",
    },
    "Kneeling": {
        "torso_angle": (5, 40), "knee_angle": (140, 180),
        "hip_angle": (80, 140), "elbow_angle": "any",
        "hip_height": "low, one or both knees on floor", "velocity": "very low",
        "foot_pattern": "knees on ground, shins on floor",
    },
    "Crouching": {
        "torso_angle": (10, 50), "knee_angle": (60, 120),
        "hip_angle": (60, 120), "elbow_angle": "any",
        "hip_height": "dropped low", "velocity": "very low",
        "foot_pattern": "feet flat, squat-like but lighter",
    },
    "Getting Up": {
        "torso_angle": "tilts forward then upright", "knee_angle": "extends",
        "hip_angle": "extends from seat", "elbow_angle": "any",
        "hip_height": "rises from seat to standing", "velocity": "upward, moderate",
        "foot_pattern": "feet planted, weight shifts forward",
    },
}

# ══════════════════════════════════════════════════════════
# 3. GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════
def _get_pt(landmarks, name):
    idx = LANDMARK_INDICES[name]
    return np.array([landmarks[idx * 4], landmarks[idx * 4 + 1], landmarks[idx * 4 + 2]])

def _get_vis(landmarks, name):
    return landmarks[LANDMARK_INDICES[name] * 4 + 3]

def _angle(a, b, c):
    ba, bc = a - b, c - b
    dot = np.dot(ba, bc)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return np.arccos(np.clip(dot / norm, -1.0, 1.0))

def _angle_deg(a, b, c):
    return math.degrees(_angle(a, b, c))

def _torso_angle(sm, hm):
    tv = hm - sm
    vd = np.array([0.0, 1.0, 0.0])
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(tv, vd)) / (float(np.linalg.norm(tv)) + 1e-8)))))

def _simple_activity(activity):
    """Show every detected activity distinctly (per 'detect all activities'),
    keeping only the Fall shorthand for the Falling class."""
    if activity == "Falling":
        return "Fall"
    return activity

def _head_angle(nose, sm):
    tv = nose - sm
    vd = np.array([0.0, 1.0, 0.0])
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(tv, vd)) / (float(np.linalg.norm(tv)) + 1e-8)))))

def _clip(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

# ══════════════════════════════════════════════════════════
# 4. PER-FRAME ANALYSIS
# ══════════════════════════════════════════════════════════
class FrameAnalysis:
    __slots__ = [
        "torso", "head", "hip_angle", "knee_angle", "elbow_angle",
        "shoulder_dist", "hip_dist", "ankle_dist", "head_hip_dist", "hip_ankle_dist",
        "hip_knee_gap",
        "bbox_aspect", "body_center", "hip_center", "head_pt", "ankle_l", "ankle_r",
        "knee_l", "knee_r", "shoulder_mid", "body_height", "feet_grounded",
        "vert_displacement", "horiz_displacement", "orientation_change",
        "hip_vel", "head_vel", "ankle_vel", "accel",
        "torso_ang_vel", "hip_ang_vel",
        "stride_length", "cadence", "steps",
        "visibility", "coords",
    ]

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)

def analyze_frame(lm):
    """Compute static pose features for a single landmarks array."""
    ls = _get_pt(lm, "left_shoulder"); rs = _get_pt(lm, "right_shoulder")
    lh = _get_pt(lm, "left_hip"); rh = _get_pt(lm, "right_hip")
    le = _get_pt(lm, "left_elbow"); re = _get_pt(lm, "right_elbow")
    lk = _get_pt(lm, "left_knee"); rk = _get_pt(lm, "right_knee")
    la = _get_pt(lm, "left_ankle"); ra = _get_pt(lm, "right_ankle")
    lw = _get_pt(lm, "left_wrist"); rw = _get_pt(lm, "right_wrist")
    nose = _get_pt(lm, "nose")
    sm = (ls + rs) / 2; hm = (lh + rh) / 2
    bc = (sm + hm) / 2
    body_height = max(1e-6, float(hm[1] - nose[1]))

    a = FrameAnalysis()
    a.torso = _torso_angle(sm, hm)
    a.head = _head_angle(nose, sm)
    a.hip_angle = (_angle_deg(ls, lh, lk) + _angle_deg(rs, rh, rk)) / 2
    a.knee_angle = (_angle_deg(lh, lk, la) + _angle_deg(rh, rk, ra)) / 2
    a.elbow_angle = (_angle_deg(ls, le, lw) + _angle_deg(rs, re, rw)) / 2
    a.shoulder_dist = float(np.linalg.norm(ls - rs))
    a.hip_dist = float(np.linalg.norm(lh - rh))
    a.ankle_dist = float(np.linalg.norm(la - ra))
    a.head_hip_dist = float(np.linalg.norm(nose - hm))
    a.hip_ankle_dist = float(np.linalg.norm(hm - ((la + ra) / 2)))
    km = (lk + rk) / 2
    a.hip_knee_gap = float(hm[1] - km[1])   # +ve = hips below knees (deep squat)
    bbox_w = max(abs(ls[0] - rs[0]), abs(lw[0] - rw[0]), abs(la[0] - ra[0]), 1e-6)
    bbox_h = max(body_height, 1e-6)
    a.bbox_aspect = float(bbox_h / bbox_w)
    a.body_center = bc; a.hip_center = hm; a.head_pt = nose
    a.ankle_l = la; a.ankle_r = ra; a.knee_l = lk; a.knee_r = rk
    a.shoulder_mid = sm
    a.body_height = float(body_height)
    vis_vals = [lm[i * 4 + 3] for i in range(33)]
    a.visibility = float(np.mean(vis_vals))
    a.coords = {name: (float(lm[idx * 4]), float(lm[idx * 4 + 1]), float(lm[idx * 4 + 2])) for name, idx in KEY_JOINT_NAMES}
    return a

# ══════════════════════════════════════════════════════════
# 5. TEMPORAL PIPELINE
#    - moving-average smoothing
#    - missing landmark interpolation
#    - 25-frame sliding window
# ══════════════════════════════════════════════════════════
class TemporalAnalyzer:
    WINDOW = 25

    def __init__(self):
        self.landmarks = deque(maxlen=self.WINDOW)   # smoothed landmark arrays
        self.times = deque(maxlen=self.WINDOW)
        self.analysis = deque(maxlen=self.WINDOW)
        self._raw_history = deque(maxlen=5)
        self._last_good = None
        self._prev_smoothed = None
        self._labels = deque(maxlen=8)
        self.t0 = time.time()
        # Must match the window used in extract_feature.py at training time —
        # was 20 here vs features.py's default of 30, which meant every
        # duration-dependent feature (cadence, time_to_horizontal, etc.) had a
        # different scale live than what the model was trained on.
        # Double check extract_feature.py's TemporalWindowExtractor(...) call
        # and set this to match exactly if it's not 30.
        self._ml_temporal = TemporalWindowExtractor(window=30)
        self.fall_alarm = False

    def reset(self):
        self.landmarks.clear(); self.times.clear(); self.analysis.clear()
        self._raw_history.clear(); self._last_good = None; self._prev_smoothed = None
        self._labels.clear()
        self._ml_temporal.reset()
        # The sequence model is fed by this same stream.  Do not let a new
        # person or a stopped/restarted camera inherit the old person's motion.
        reset_fall_sensor()
        self.fall_alarm = False

    # --- missing landmark interpolation (keep last known position) ---
    def _interpolate(self, lm):
        if self._last_good is None:
            self._last_good = lm.copy()
            return lm
        out = lm.copy()
        for i in range(33):
            if lm[i * 4 + 3] < 0.3:
                out[i * 4] = self._last_good[i * 4]
                out[i * 4 + 1] = self._last_good[i * 4 + 1]
                out[i * 4 + 2] = self._last_good[i * 4 + 2]
                out[i * 4 + 3] = 0.0
        self._last_good = out.copy()
        return out

    # --- moving average smoothing (window = 5 raw frames) ---
    def _smooth(self, lm):
        self._raw_history.append(lm)
        if len(self._raw_history) < 2:
            self._prev_smoothed = lm.copy()
            return lm
        out = self._prev_smoothed.copy() if self._prev_smoothed is not None else lm.copy()
        n = len(self._raw_history)
        for i in range(33):
            if lm[i * 4 + 3] > 0.3:
                for k in range(4):
                    out[i * 4 + k] = (self._prev_smoothed[i * 4 + k] + lm[i * 4 + k]) * 0.5
        self._prev_smoothed = out.copy()
        return out

    def update(self, lm, t):
        lm_raw = lm.copy()
        lm = self._interpolate(lm)
        lm = self._smooth(lm)
        self.landmarks.append(lm)
        self.times.append(t)
        self._ml_temporal.push(lm_raw, t)
        fa = analyze_frame(lm)
        self.analysis.append(fa)
        return self.classify()

    # ──────────────────────────────────────────────────────
    # TEMPORAL FEATURES (over the sliding window)
    # ──────────────────────────────────────────────────────
    def _temporal_features(self, fa):
        n = len(self.landmarks)
        if n < 2:
            fa.hip_vel = np.zeros(2); fa.head_vel = np.zeros(2)
            fa.ankle_vel = np.zeros(2); fa.accel = np.zeros(2)
            fa.torso_ang_vel = 0.0; fa.hip_ang_vel = 0.0
            fa.vert_displacement = 0.0; fa.horiz_displacement = 0.0
            fa.orientation_change = 0.0
            fa.stride_length = 0.0; fa.cadence = 0.0; fa.steps = 0
            fa.feet_grounded = True
            return fa

        dt = max(1e-4, self.times[-1] - self.times[-2])
        prev_fa = self.analysis[-2]

        fa.hip_vel = (fa.hip_center[:2] - prev_fa.hip_center[:2]) / dt
        fa.head_vel = (fa.head_pt[:2] - prev_fa.head_pt[:2]) / dt
        fa.ankle_vel = ((fa.ankle_l[:2] + fa.ankle_r[:2]) / 2 - (prev_fa.ankle_l[:2] + prev_fa.ankle_r[:2]) / 2) / dt
        fa.torso_ang_vel = (fa.torso - prev_fa.torso) / dt
        fa.hip_ang_vel = (fa.hip_angle - prev_fa.hip_angle) / dt

        if len(self.analysis) >= 3:
            prev2 = self.analysis[-3]
            v_prev = (prev_fa.hip_center[:2] - prev2.hip_center[:2]) / max(1e-4, self.times[-2] - self.times[-3])
            fa.accel = (fa.hip_vel - v_prev) / dt
        else:
            fa.accel = np.zeros(2)

        hip_y_series = [a.hip_center[1] for a in self.analysis]
        hip_x_series = [a.hip_center[0] for a in self.analysis]
        torso_series = [a.torso for a in self.analysis]
        fa.vert_displacement = float(hip_y_series[-1] - hip_y_series[0])   # +ve = body dropped
        fa.horiz_displacement = float(hip_x_series[-1] - hip_x_series[0])
        fa.orientation_change = float(torso_series[-1] - torso_series[0])

        # stride / cadence via alternating ankle crossing
        ankle_l_y = [a.ankle_l[1] for a in self.analysis]
        ankle_r_y = [a.ankle_r[1] for a in self.analysis]
        steps = 0
        for i in range(1, len(ankle_l_y)):
            if (ankle_l_y[i - 1] - ankle_r_y[i - 1]) * (ankle_l_y[i] - ankle_r_y[i]) < 0:
                steps += 1
        duration = max(1e-4, self.times[-1] - self.times[0])
        fa.steps = steps
        fa.cadence = steps / duration  # steps per second
        fa.stride_length = abs(fa.horiz_displacement) / max(1, steps)

        # foot contact: grounded if a foot is near the floor level in the window
        floor_y = max(max(ankle_l_y), max(ankle_r_y))
        both_off = (fa.ankle_l[1] < floor_y - 0.05) and (fa.ankle_r[1] < floor_y - 0.05)
        fa.feet_grounded = not both_off
        return fa

    # ──────────────────────────────────────────────────────
    # 10-ACTIVITY CLASSIFICATION
    # ──────────────────────────────────────────────────────
    TEMPORAL_CLASSES = {"Walking", "Running", "Jumping", "Climbing Stairs",
                        "Crawling", "Getting Up"}

    def _geometric_rules(self, fa):
        """SafeFall-style coordinate rules: returns (activity, confidence) when
        geometric evidence is strong, else (None, 0).  These override or boost
        the ML model for unambiguous poses (e.g. torso horizontal = lying down).

        Thresholds adapted from SafeFall AI V5's coordinate-based classifier
        (activity_classifier.py) to the MediaPipe landmark system."""
        if fa is None or fa.torso is None:
            return None, 0.0
        t = fa.torso
        ka = fa.knee_angle if fa.knee_angle is not None else 120
        ha = fa.hip_angle if fa.hip_angle is not None else 160
        hip_y = fa.hip_center[1] if fa.hip_center is not None else 0.5
        head_y = fa.head_pt[1] if fa.head_pt is not None else 0.2

        # --- Lying Down: torso > 70° (near horizontal) ---
        if t > 70:
            return "Lying Down", 0.85
        # --- Falling: torso 45-70° + rapid downward hip velocity ---
        if 45 <= t <= 70 and fa.hip_vel is not None and fa.hip_vel[1] > 0.15:
            return "Falling", 0.75
        # --- Bending: torso 30-60° + nearly straight legs ---
        if 30 <= t <= 60 and ka > 145:
            return "Bending", 0.70
        # --- Kneeling: knee below hip, knee near ankle level ---
        if (fa.knee_l is not None and fa.ankle_l is not None
                and fa.knee_l[1] > hip_y and abs(fa.knee_l[1] - fa.ankle_l[1]) < 0.06):
            return "Kneeling", 0.65
        # --- Standing: upright, straight legs, feet on ground ---
        # SafeFall adds hip-to-knee distance ratio < 1.2 to confirm upright
        if t < 15 and ka > 150 and ha > 140:
            if fa.feet_grounded:
                return "Standing", 0.75
        # --- Sitting: SafeFall-style with hip angle check ---
        # Requires knee 70-120° + hip 70-110° (both bent ~90°) + torso upright
        # hip_knee_gap < 0 confirms hips above knees (not squatting)
        if 70 <= ka <= 120 and 70 <= ha <= 110 and t < 30 and fa.hip_knee_gap < 0:
            return "Sitting", 0.65
        # --- Squatting: deep knee bend, hips NOT below knees ---
        if ka < 90 and ha < 110 and t < 40 and fa.hip_knee_gap < 0.03:
            return "Squatting", 0.70
        # --- Crouching: hips below knees, moderate lean + knee bend ---
        if ka < 100 and 20 < t < 60 and fa.hip_knee_gap > 0.03:
            return "Crouching", 0.70
        return None, 0.0

    def classify(self):
        if len(self.landmarks) == 0:
            return None
        fa = analyze_frame(self.landmarks[-1])
        fa = self._temporal_features(fa)
        self.analysis[-1] = fa

        ml_act, ml_conf, ml_probs, _cached_feats = ml_predict(
            self.landmarks[-1], temporal=self._ml_temporal.temporal_features()
        )

        if ml_probs is not None:
            classes = list(_ml_activities) if _ml_activities else list(ACTIVITIES)
            idx = int(np.argmax(ml_probs))
            activity = classes[idx]
            confidence = float(ml_probs[idx])
        else:
            activity, confidence = ml_act, ml_conf

        # --- SafeFall-style geometric rule boost ---
        rule_act, rule_conf = self._geometric_rules(fa)
        if rule_act is not None:
            if rule_act == activity:
                confidence = min(0.95, max(confidence, rule_conf))
            # A static upright-pose rule cannot distinguish standing from
            # walking/running.  Previously it overwrote a valid temporal ML
            # prediction of Walking with Standing whenever the legs happened
            # to be nearly straight in a stride.
            elif rule_conf >= 0.70 and activity not in self.TEMPORAL_CLASSES:
                activity, confidence = rule_act, rule_conf

        fall_prob, alarm_on, _ = compute_fall_signal(
            self.landmarks[-1], temporal=self._ml_temporal.temporal_features(),
            _cached_feats=_cached_feats, _cached_multiclass_probs=ml_probs)
        self.fall_alarm = bool(alarm_on and fa.torso is not None and fa.torso >= 50.0)
        if ml_probs is not None and "Falling" in _ml_activities:
            fall_prob = max(fall_prob, float(ml_probs[_ml_activities.index("Falling")]))
        if self.fall_alarm:
            activity, confidence = "Falling", max(confidence, fall_prob)
            self._labels.clear()
        else:
            activity = self._stabilize(activity)
        return activity, confidence, fall_prob, fa

    def _stabilize(self, act):
        self._labels.append(act)
        if len(self._labels) >= 4:
            return max(set(self._labels), key=self._labels.count)
        return act

# ══════════════════════════════════════════════════════════
# 6. STATIC CLASSIFIER (single image - no motion info)
# ══════════════════════════════════════════════════════════
def classify_static(landmarks, temporal=None):
    fa = analyze_frame(landmarks)
    ml_act, ml_conf, ml_probs, _ = ml_predict(landmarks, temporal=temporal)
    ml_conf = ml_conf if ml_act is not None else 0.0

    # ML primary — same logic as TemporalAnalyzer.classify()
    if ml_act is not None and ml_conf >= 0.40:
        activity = ml_act
        confidence = ml_conf
    elif ml_act is not None:
        activity, confidence = ml_act, ml_conf
    else:
        activity, confidence = "Unknown", 0.0

    # Geometric rule boost — matches TemporalAnalyzer._geometric_rules()
    rule_act, rule_conf = _geometric_rules_standalone(fa)
    if rule_act is not None and rule_conf >= 0.70:
        if rule_act == activity:
            confidence = min(0.95, max(confidence, rule_conf))
        else:
            activity, confidence = rule_act, rule_conf

    return activity, _clip(confidence), fa


def _geometric_rules_standalone(fa):
    """Standalone geometric rules matching TemporalAnalyzer._geometric_rules."""
    if fa is None or fa.torso is None:
        return None, 0.0
    t = fa.torso
    ka = fa.knee_angle if fa.knee_angle is not None else 120
    ha = fa.hip_angle if fa.hip_angle is not None else 160
    hip_y = fa.hip_center[1] if fa.hip_center is not None else 0.5
    head_y = fa.head_pt[1] if fa.head_pt is not None else 0.2

    if t > 70:
        return "Lying Down", 0.85
    if 45 <= t <= 70 and fa.hip_vel is not None and fa.hip_vel[1] > 0.15:
        return "Falling", 0.75
    if 30 <= t <= 60 and ka > 145:
        return "Bending", 0.70
    if (fa.knee_l is not None and fa.ankle_l is not None
            and fa.knee_l[1] > hip_y and abs(fa.knee_l[1] - fa.ankle_l[1]) < 0.06):
        return "Kneeling", 0.65
    if t < 15 and ka > 150 and ha > 140:
        if fa.feet_grounded:
            return "Standing", 0.75
    if 70 <= ka <= 120 and 70 <= ha <= 110 and t < 30 and fa.hip_knee_gap < 0:
        return "Sitting", 0.65
    if ka < 90 and ha < 110 and t < 40 and fa.hip_knee_gap < 0.03:
        return "Squatting", 0.70
    if ka < 100 and 20 < t < 60 and fa.hip_knee_gap > 0.03:
        return "Crouching", 0.70
    return None, 0.0

# ══════════════════════════════════════════════════════════
# 7. POSE UTILS
# ══════════════════════════════════════════════════════════
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

MP_MODEL_LITE = "models/pose_landmarker_lite.task"
MP_MODEL_FULL = "models/pose_landmarker_full.task"
POSE_URL_LITE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_URL_FULL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
_POSE_CACHE = {}
_POSE_LAST_TS = {}   # per-landmarker-key last timestamp_ms, VIDEO mode requires strictly increasing
_LAST_PERSON_SEEN = 0.0   # time.time() of last successful person detection
_LOW_CONF_COOLDOWN = 0.3  # seconds to skip low-conf fallback after last detection


def _ensure_pose_model(mode):
    path = MP_MODEL_FULL if mode == "Accurate" else MP_MODEL_LITE
    url = POSE_URL_FULL if mode == "Accurate" else POSE_URL_LITE
    p = Path(__file__).parent / path
    if not p.exists():
        st.warning(f"Downloading pose model ({mode})... this happens once.")
        urllib.request.urlretrieve(url, p)
    return str(p)


def _get_pose_landmarker(mode, low_conf=False, streaming=False):
    """streaming=True -> VIDEO-mode landmarker with frame-to-frame tracking,
    locked to a single subject (num_poses=1). Used for live webcam / video
    file playback. streaming=False -> IMAGE-mode landmarker for one-off
    frames (uploaded photos, Capture Frame), where there's no previous-frame
    context to track from anyway."""
    key = f"{mode}_{'low' if low_conf else 'std'}_{'vid' if streaming else 'img'}"
    path = _ensure_pose_model(mode)
    if key not in _POSE_CACHE:
        det_conf = 0.3 if low_conf else 0.5
        pres_conf = 0.3 if low_conf else 0.4
        opts = PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=path),
            running_mode=RunningMode.VIDEO if streaming else RunningMode.IMAGE,
            min_pose_detection_confidence=det_conf, min_pose_presence_confidence=pres_conf,
            # num_poses=1 for streaming: locks onto the primary subject instead
            # of re-picking whichever pose MediaPipe returns first each frame,
            # which was causing identity swaps when a second person was in frame.
            num_poses=1 if streaming else 5,
            output_segmentation_masks=False,
        )
        _POSE_CACHE[key] = PoseLandmarker.create_from_options(opts)
        _POSE_LAST_TS[key] = -1
    return _POSE_CACHE[key], key

# Complete MediaPipe Pose 33-landmark connections (full stick-figure body:
# head/face, shoulder->elbow->wrist chains, shoulder->hip->knee->ankle chains,
# shoulder-shoulder, hip-hip, hands and feet). Covers every joint so nothing
# draws as a floating disconnected segment.
POSE_CONNECTIONS_FULL = [
    (0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8),(7,9),(8,10),(9,10),
    (11,12),
    (11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
    (27,29),(29,31),(28,30),(30,32),(27,31),(28,32),
]
# Drawing overlay uses the body only (indices 11-32): shoulders->elbows->wrists,
# shoulder->hip->knee->ankle chains, hip-hip and feet. Face joints (0-10) are
# excluded from the skeleton so no face mesh is drawn.
KEY_JOINTS = list(range(11, 33))
KEY_CONNECTIONS = [(i, j) for i, j in POSE_CONNECTIONS_FULL if i >= 11 and j >= 11]
CG = (0,255,0); CR = (0,0,255); CY = (0,255,255); CC = (255,255,0)
CW = (255,255,255); CB = (0,0,0); CO = (0,165,255); CM = (255,0,255)

ACTIVITY_COLORS = {
    "Standing": CG, "Walking": CC, "Sitting": CY, "Falling": CR,
    "Lying Down": CM, "Bending": CO, "Squatting": (255,128,0),
    "Running": (0,255,255), "Jumping": (255,255,0), "Climbing Stairs": (0,128,255),
    "Crawling": (0,255,128), "Kneeling": (128,255,0), "Crouching": (0,200,255),
    "Getting Up": (80,220,160),
}

def _landmarks_to_arr(lm_list):
    arr = []
    for lm in lm_list:
        arr.extend([lm.x, lm.y, lm.z, lm.visibility if lm.visibility else 0.0])
    return np.array(arr, dtype=np.float32)

def _run_landmarker(pose_lm, key, mp_img, streaming, timestamp_ms):
    """Call detect() or detect_for_video() depending on mode, enforcing the
    strictly-increasing timestamp VIDEO mode requires. Self-heals: if the
    VIDEO-mode call throws (e.g. after a Stop->Start cycle leaves the
    landmarker's internal tracking state stale, or a timestamp edge case),
    destroy and recreate that landmarker instance and retry once instead of
    freezing the whole session."""
    if not streaming:
        return pose_lm.detect(mp_img)
    ts = int(timestamp_ms)
    last = _POSE_LAST_TS.get(key, -1)
    if ts <= last:
        ts = last + 1   # guarantee monotonic increase even if wall-clock ties/jumps back
    try:
        result = pose_lm.detect_for_video(mp_img, ts)
        _POSE_LAST_TS[key] = ts
        return result
    except Exception:
        # Landmarker is likely in a bad internal state - drop it and rebuild
        # fresh on the next call rather than repeatedly throwing.
        _POSE_CACHE.pop(key, None)
        _POSE_LAST_TS.pop(key, None)
        return None


_POSE_MODE = "Fast"

def _detect_pose(image, timestamp_ms=None, rgb_image=None):
    """timestamp_ms: pass the current stream time (ms) for live webcam / video
    file frames to enable VIDEO-mode tracking. Leave None for one-off single
    images (uploaded photo, Capture Frame) to use IMAGE mode instead.
    rgb_image: optional pre-converted RGB array (avoids a redundant BGR->RGB)."""
    global _LAST_PERSON_SEEN
    if image is None or image.size == 0: return None, None, []
    try:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        if max(h, w) > config.LIVE_POSE_MAX_SIDE:
            scale = config.LIVE_POSE_MAX_SIDE / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        mode = "Accurate" if _POSE_MODE == "Accurate" else "Fast"
        streaming = timestamp_ms is not None
        if rgb_image is not None and rgb_image.shape[:2] == image.shape[:2]:
            rgb = rgb_image
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb, dtype=np.uint8))
        pose_lm, key = _get_pose_landmarker(mode, streaming=streaming)
        result = _run_landmarker(pose_lm, key, mp_img, streaming, timestamp_ms)
        if result is None or not result.pose_landmarks:
            elapsed = time.time() - _LAST_PERSON_SEEN
            # Retry with the low-confidence landmarker when the standard one
            # misses. Requires NOT having a fresh lock (elapsed >= cooldown) or
            # having never detected anyone yet (_LAST_PERSON_SEEN == 0) so the
            # fallback can recover instead of being blocked forever.
            if _LAST_PERSON_SEEN == 0.0 or elapsed >= _LOW_CONF_COOLDOWN:
                pose_lm_low, key_low = _get_pose_landmarker(mode, low_conf=True, streaming=streaming)
                result = _run_landmarker(pose_lm_low, key_low, mp_img, streaming, timestamp_ms)
        if result is None or not result.pose_landmarks:
            return None, None, []
        _LAST_PERSON_SEEN = time.time()
        all_lms = list(result.pose_landmarks)
        return _landmarks_to_arr(all_lms[0]), all_lms[0], all_lms
    except Exception:
        return None, None, []

def extract_landmarks_from_frame(frame, timestamp_ms=None, rgb_image=None):
    return _detect_pose(frame, timestamp_ms=timestamp_ms, rgb_image=rgb_image)

# ══════════════════════════════════════════════════════════
# 8. DRAWING
# ══════════════════════════════════════════════════════════
# Visualization-only temporal smoothing. Coordinates stay in normalized 0..1
# space so smoothing is scale/position independent. It does NOT touch the raw
# landmark array used for ML features, so fall detection stays responsive.
_VIS_SMOOTH_ALPHA = 0.6          # 0.6 = responsive, 1.0 = no smoothing
_VIS_STALE_AFTER = 0.5           # seconds; reset a joint's history if not seen
_VIS_SMOOTH_BUF = {}             # person index -> {joint -> [x, y, vis, last_t]}
VISIBILITY_MIN = 0.45            # endpoints below this omit that connection


def _smooth_landmarks(person, p_idx):
    """Return {joint: (x, y, visibility)} smoothed with EMA over recent frames."""
    now = time.time()
    buf = _VIS_SMOOTH_BUF.setdefault(p_idx, {})
    out = {}
    for idx, lm in enumerate(person):
        vis = lm.visibility if lm.visibility is not None else 0.0
        x, y = float(lm.x), float(lm.y)
        prev = buf.get(idx)
        if prev is not None and (now - prev[3]) < _VIS_STALE_AFTER:
            sx = prev[0] + _VIS_SMOOTH_ALPHA * (x - prev[0])
            sy = prev[1] + _VIS_SMOOTH_ALPHA * (y - prev[1])
            out[idx] = (sx, sy, vis)
            buf[idx] = [sx, sy, vis, now]
        else:
            out[idx] = (x, y, vis)
            buf[idx] = [x, y, vis, now]
    for idx in list(buf):
        if now - buf[idx][3] > _VIS_STALE_AFTER:
            del buf[idx]
    return out


def draw_body_lines(img, lm_list, h, w):
    """Human pose estimation skeleton: one full-body connected stick figure per person.

    Uses the official MediaPipe Pose 33-landmark topology. Each connection is
    drawn only when both endpoints are currently reliable (visibility check);
    unreliable joints are drawn as small gray dots so the skeleton never draws
    wrong lines. Coordinates are normalized (0..1) and temporally smoothed in
    this function only; ML features keep the raw landmarks.
    """
    if lm_list is None or len(lm_list) == 0:
        _VIS_SMOOTH_BUF.clear()
        return img
    # Accept either a single NormalizedLandmarkList or a list of them (multi-person).
    if hasattr(lm_list[0], "x"):
        lm_list = [lm_list]
    SKELETON_BLUE = (255, 128, 0)
    JOINT_BLUE = (255, 200, 80)
    JOINT_LOW_VIS = (120, 120, 120)
    for p_idx, person in enumerate(lm_list):
        pts = _smooth_landmarks(person, p_idx)
        # Continuous anatomical lines (only between reliable endpoints).
        for i, j in KEY_CONNECTIONS:
            if i not in pts or j not in pts:
                continue
            vis_i, vis_j = pts[i][2], pts[j][2]
            if vis_i < VISIBILITY_MIN or vis_j < VISIBILITY_MIN:
                continue
            x1, y1 = int(pts[i][0] * w), int(pts[i][1] * h)
            x2, y2 = int(pts[j][0] * w), int(pts[j][1] * h)
            cv2.line(img, (x1, y1), (x2, y2), SKELETON_BLUE, 3, cv2.LINE_AA)
        # Joints: every detected joint appears as a landmark.
        for idx in KEY_JOINTS:
            if idx not in pts:
                continue
            x, y = int(pts[idx][0] * w), int(pts[idx][1] * h)
            if pts[idx][2] >= VISIBILITY_MIN:
                cv2.circle(img, (x, y), 6, JOINT_BLUE, -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (x, y), 3, JOINT_LOW_VIS, -1, cv2.LINE_AA)
    return img

def annotate_frame(img, lm_list, activity, confidence, fall_prob, fa, fps, h, w, lite=False, fall_alarm=False):
    if lm_list is None:
        return img
    return draw_body_lines(img, lm_list, h, w)

def coords_dataframe(fa):
    if fa is None or fa.coords is None:
        return pd.DataFrame(columns=["Landmark", "x", "y", "z"])
    rows = []
    for name, (x, y, z) in fa.coords.items():
        rows.append({"Landmark": name, "x": round(x, 4), "y": round(y, 4), "z": round(z, 4)})
    return pd.DataFrame(rows)

def reference_guide_dataframe():
    rows = []
    for act, r in REFERENCE_POSTURES.items():
        torso = f"{r['torso_angle'][0]}-{r['torso_angle'][1]} deg" if isinstance(r['torso_angle'], tuple) else r['torso_angle']
        knee = f"{r['knee_angle'][0]}-{r['knee_angle'][1]} deg" if isinstance(r['knee_angle'], tuple) else r['knee_angle']
        rows.append({
            "Activity": act,
            "Torso Angle": torso, "Knee Angle": knee, "Hip Height": r["hip_height"],
            "Velocity": r["velocity"], "Foot Pattern": r["foot_pattern"],
        })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════
# 9. ML MODEL  (mtime-cached lazy loading)
# ══════════════════════════════════════════════════════════
# Models are loaded lazily and cached by file modification time. reload_models()
# runs on every script pass, so after "Record & Train" finishes, the newly
# trained model files are picked up automatically without restarting Streamlit.
MODEL_PATH = config.FRAME_MODEL_PATH
_MODEL_CACHE = {}
_ml_model = None
_ml_activities = []
_ml_fall_threshold = config.DEFAULT_FALL_THRESHOLD
_ml_feature_names = []
_ml_model_name = "-"
_binary_bundle = None
_seq_model = None
_seq_norm = (np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)))
_seq_n_feats = len(FEATURE_NAMES)
_seq_buffer = deque(maxlen=config.SEQUENCE_LEN)
_seq_last_prob = None
_seq_prediction_ticks = 0


def _load_if_changed(path, loader):
    """Return the cached bundle if the file is unchanged, else reload it."""
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    entry = _MODEL_CACHE.get(str(path))
    if entry is not None and entry[0] == mtime:
        return entry[1]
    try:
        value = loader(path)
    except Exception:
        value = None
    _MODEL_CACHE[str(path)] = (mtime, value)
    return value


def _load_frame_bundle(path):
    mb = joblib.load(path)
    feats = mb.get("feature_names", [])
    if feats and len(feats) != len(FEATURE_NAMES):
        return None   # mismatched feature count -> ignore stale bundle
    return mb


def _load_binary_bundle(path):
    b = joblib.load(path)
    if set(b.get("feature_names", [])) == set(FEATURE_NAMES):
        return b
    return None


def _load_seq_bundle(path):
    import keras
    model = keras.models.load_model(path, compile=False)
    norm = json.loads((config.MODEL_DIR / "sequence_normalizer.json").read_text())
    return {
        "model": model,
        "norm_mean": np.asarray(norm["mean"]),
        "norm_std": np.asarray(norm["std"]) + 1e-6,
    }


def reload_models():
    """Reload any model whose file changed since it was last loaded."""
    global _ml_model, _ml_activities, _ml_fall_threshold, _ml_feature_names, _ml_model_name
    global _binary_bundle, _seq_model, _seq_norm, _seq_n_feats
    fb = _load_if_changed(MODEL_PATH, _load_frame_bundle)
    if fb is not None:
        _ml_model = fb["model"]
        _ml_activities = fb.get("activities", ["Standing", "Sitting", "Walking", "Falling"])
        _ml_fall_threshold = float(fb.get("fall_threshold", config.DEFAULT_FALL_THRESHOLD))
        _ml_feature_names = fb.get("feature_names", [])
        _ml_model_name = fb.get("model_type", "Random Forest")
    else:
        _ml_model = None
        _ml_activities = []
        _ml_model_name = "-"
    _binary_bundle = _load_if_changed(config.MODEL_DIR / "binary_fall_model.pkl", _load_binary_bundle)
    sb = _load_if_changed(config.SEQUENCE_MODEL_PATH, _load_seq_bundle)
    if sb is not None:
        _seq_model = sb["model"]
        _seq_norm = (sb["norm_mean"], sb["norm_std"])
        _seq_n_feats = len(sb["norm_mean"])
    else:
        _seq_model = None
        _seq_norm = (np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)))
        _seq_n_feats = len(FEATURE_NAMES)


def reset_fall_sensor():
    """Clear the LSTM history buffer (call when the person disappears)."""
    global _seq_last_prob, _seq_prediction_ticks
    _seq_buffer.clear()
    _seq_last_prob = None
    _seq_prediction_ticks = 0


def compute_fall_signal(landmarks, temporal=None, _cached_feats=None,
                        _cached_multiclass_probs=None):
    """Fused fall signal across the multi-class, binary and LSTM detectors.

    Returns (fall_prob, alarm_on, breakdown) where fall_prob is the max of the
    available detectors' fall probabilities and alarm_on is True when ANY
    detector exceeds its own tuned threshold."""
    try:
        feats = _cached_feats if _cached_feats is not None else extract_features(landmarks, temporal=temporal)
    except Exception:
        return 0.0, False, {}
    if np.any(np.isnan(feats)):
        return 0.0, False, {}
    breakdown = {}
    fall_prob, alarm_on = 0.0, False
    if _ml_model is not None and "Falling" in _ml_activities:
        # TemporalAnalyzer already calculated these probabilities to choose
        # the activity label.  Reusing them avoids a second full classifier
        # evaluation for every webcam frame.
        probs = (_cached_multiclass_probs if _cached_multiclass_probs is not None
                 else _ml_model.predict_proba([feats])[0])
        p = float(probs[_ml_activities.index("Falling")])
        breakdown["classifier"] = p
        fall_prob = max(fall_prob, p)
        if p >= _ml_fall_threshold:
            alarm_on = True
    if _binary_bundle is not None:
        p = float(_binary_bundle["model"].predict_proba([feats])[0][1])
        breakdown["binary"] = p
        fall_prob = max(fall_prob, p)
        if p >= float(_binary_bundle.get("threshold", 0.55)):
            alarm_on = True
    if _seq_model is not None:
        global _seq_last_prob, _seq_prediction_ticks
        seq_feats = feats[:_seq_n_feats].astype(np.float32)
        _seq_buffer.append(seq_feats)
        if len(_seq_buffer) == config.SEQUENCE_LEN:
            _seq_prediction_ticks += 1
            if (_seq_last_prob is None or
                    _seq_prediction_ticks % config.LIVE_LSTM_PREDICTION_STRIDE == 0):
                seq = (np.stack(list(_seq_buffer)) - _seq_norm[0]) / _seq_norm[1]
                _seq_last_prob = float(_seq_model.predict(seq[None, ...], verbose=0)[0][0])
            breakdown["lstm"] = _seq_last_prob
            fall_prob = max(fall_prob, _seq_last_prob)
            if _seq_last_prob >= config.SEQUENCE_LSTM_THRESHOLD:
                alarm_on = True
    return fall_prob, alarm_on, breakdown


class FallConfirmer:
    """Two-stage fall confirmation.

    An alarm only fires after CONFIRM_FRAMES consecutive frames where a detector
    is on AND the body is physically in a fall state (torso near horizontal).
    The physical check is what separates a real fall from walking/jumping/ADL
    peaks that fire the model: those keep the torso upright and therefore never
    confirm. The count decays slowly so a brief dip does not re-trigger the
    siren repeatedly.
    """
    def __init__(self, confirm_frames=None, torso_confirm_deg=50.0):
        self.confirm_frames = confirm_frames or config.CONFIRM_FRAMES
        self.torso_confirm_deg = torso_confirm_deg
        self.count = 0
        self.confirmed = False

    def update(self, alarm_on, fa=None):
        physically_fallen = (fa is not None and fa.torso is not None
                             and fa.torso >= self.torso_confirm_deg)
        if alarm_on and physically_fallen:
            self.count += 1
            if self.count >= self.confirm_frames:
                self.confirmed = True
        else:
            self.count = max(0, self.count - 2)
            if self.count == 0:
                self.confirmed = False
        return self.confirmed

    def reset(self):
        self.count = 0
        self.confirmed = False


_fall_confirmer = FallConfirmer()


def ml_fall_signal(landmarks, temporal=None):
    return compute_fall_signal(landmarks, temporal=temporal)[0]

def ml_predict(landmarks, temporal=None):
    """Return (activity, confidence, prob_vector, cached_features) or (None, 0, None, None)."""
    if _ml_model is None or not _ml_activities:
        return None, 0.0, None, None
    try:
        feats = extract_features(landmarks, temporal=temporal)
        if np.any(np.isnan(feats)):
            return None, 0.0, None, None
        probs = _ml_model.predict_proba([feats])[0]
        probs = np.asarray(probs, dtype=np.float64)
        idx = int(np.argmax(probs))
        return _ml_activities[idx], float(probs[idx]), probs, feats
    except Exception:
        return None, 0.0, None, None

# ══════════════════════════════════════════════════════════
# 10. STREAMLIT APP
# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="AI Human Activity Recognition & Fall Detection", page_icon="🚨", layout="wide")
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, *::before, *::after { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; box-sizing: border-box; }

/* ══════════════════════════════════════════════════════════
   DESIGN TOKENS
   ══════════════════════════════════════════════════════════ */
:root {
  --bg: #050B14;
  --surface: #0B1422;
  --surface-2: #101C2D;
  --border: #1E3553;
  --primary: #2563EB;
  --primary-bright: #3B82F6;
  --primary-soft: rgba(37,99,235,0.14);
  --text: #F8FAFC;
  --text-2: #94A3B8;
  --danger: #EF4444;
  --danger-soft: rgba(239,68,68,0.12);
  --success: #22C55E;
  --success-soft: rgba(34,197,94,0.12);
  --warning: #F59E0B;
  --warning-soft: rgba(245,158,11,0.12);
  --radius: 12px;
  --radius-sm: 8px;
  --radius-lg: 20px;
  --radius-pill: 999px;
  --gap-xs: 4px;
  --gap-sm: 8px;
  --gap: 12px;
  --gap-md: 16px;
  --gap-lg: 24px;
  --gap-xl: 32px;
  --gap-2xl: 48px;
  --nav-h: 56px;
  --tab-h: 44px;
  --shadow: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2);
  --shadow-lg: 0 4px 12px rgba(0,0,0,0.4);
}

/* ══════════════════════════════════════════════════════════
   BASE / RESET
   ══════════════════════════════════════════════════════════ */
.stApp { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none; }
.block-container { max-width: 1440px; padding: 0.9rem 1.5rem 1.6rem; }
h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.01em; }
.stMarkdown, [data-testid="stWidgetLabel"] p { color: var(--text-2); }
.stCaption { color: var(--text-2); }

/* ══════════════════════════════════════════════════════════
   GLOBAL LAYOUT GRID
   ══════════════════════════════════════════════════════════ */
.dashboard-grid { display: flex; flex-direction: column; gap: var(--gap-lg); }
.monitor-grid { display: grid; grid-template-columns: 7fr 3fr; gap: var(--gap-md); align-items: start; }
.control-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: var(--gap-sm); }
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--gap-sm); }
.nav-grid { display: grid; grid-template-columns: auto 1fr auto; gap: var(--gap); align-items: center; }
.nav-btns { display: flex; gap: var(--gap-sm); }
.nav-grid-row2 { display: grid; grid-template-columns: auto 1fr auto; gap: var(--gap); align-items: end; }

/* ══════════════════════════════════════════════════════════
   SECTION SPACING
   ══════════════════════════════════════════════════════════ */
.section-title { font-size: 18px; font-weight: 700; color: var(--text); margin: 0 0 var(--gap-sm); padding-bottom: 8px; border-bottom: 1px solid var(--border); }
.section-sub { font-size: 13px; color: var(--text-2); margin: 0 0 var(--gap-sm); }

/* ══════════════════════════════════════════════════════════
   NAVIGATION — all buttons identical height
   ══════════════════════════════════════════════════════════ */
.nav-grid [data-testid="stBaseButton-secondary"],
.nav-grid [data-testid="stBaseButton-primary"] {
  height: var(--nav-h) !important;
  min-height: var(--nav-h) !important;
  max-height: var(--nav-h) !important;
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border) !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  white-space: nowrap !important;
  padding: 0 16px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}
.nav-grid [data-testid="stBaseButton-secondary"] {
  background: var(--surface) !important;
  color: var(--text-2) !important;
}
.nav-grid [data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--primary) !important;
  color: var(--text) !important;
}
.nav-grid [data-testid="stBaseButton-primary"] {
  background: var(--primary) !important;
  border-color: var(--primary) !important;
  color: #fff !important;
}
.nav-grid [data-testid="stBaseButton-primary"]:hover { background: var(--primary-bright) !important; }

/* brand */
.app-brand h1 { margin: 0; font-size: 21px; font-weight: 800; letter-spacing: 2.4px; color: var(--text); white-space: nowrap; }
.app-brand h1 span { color: var(--primary-bright); }
.app-brand p { margin: 2px 0 0; font-size: 12px; color: var(--text-2); letter-spacing: 0.4px; }

/* system status pill */
.system-status { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; font-weight: 700; letter-spacing: 1px; color: var(--text-2); padding: 0 16px; border: 1px solid var(--border); border-radius: var(--radius-pill); background: var(--surface); height: var(--nav-h); white-space: nowrap; }
.system-status .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-2); flex-shrink: 0; }
.system-status.active { color: var(--text); border-color: rgba(59,130,246,0.5); }
.system-status.active .dot { background: var(--primary-bright); box-shadow: 0 0 8px rgba(59,130,246,0.7); }

/* control label */
.ctl-label { font-size: 11px; font-weight: 700; letter-spacing: 1.4px; color: var(--text-2); text-transform: uppercase; margin: 0 0 4px; }

/* ══════════════════════════════════════════════════════════
   INPUT-MODE SEGMENTED CONTROL
   ══════════════════════════════════════════════════════════ */
[data-testid="stSegmentedControl"] { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 3px; height: var(--tab-h); }
[data-testid="stSegmentedControl"] button { color: var(--text-2); font-size: 13px; white-space: nowrap; height: calc(var(--tab-h) - 6px) !important; }
[data-testid="stSegmentedControl"] [aria-pressed="true"] { background: var(--primary) !important; color: #fff !important; }

/* ══════════════════════════════════════════════════════════
   HERO
   ══════════════════════════════════════════════════════════ */
.hero { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--primary-bright); border-radius: var(--radius); padding: var(--gap) var(--gap-md); margin-bottom: var(--gap-md) !important; }
.hero h2 { margin: 0; font-size: 18px; font-weight: 700; color: var(--text); }
.hero p { margin: 2px 0 8px; font-size: 13px; color: var(--text-2); }
.hero-badges { display: flex; flex-wrap: wrap; gap: var(--gap-sm); }
.badge { font-size: 11.5px; font-weight: 600; color: var(--text-2); background: var(--surface-2); border: 1px solid var(--border); padding: 3px 11px; border-radius: var(--radius-pill); white-space: nowrap; }
.badge b { color: var(--primary-bright); }

/* ══════════════════════════════════════════════════════════
   CARDS — shared card style
   ══════════════════════════════════════════════════════════ */
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--gap) 16px; min-height: 200px; display: flex; flex-direction: column; }
.panel-title { font-size: 11px; font-weight: 700; letter-spacing: 1.2px; color: var(--primary-bright); text-transform: uppercase; margin-bottom: 8px; }

/* ══════════════════════════════════════════════════════════
   VIDEO / CAMERA
   ══════════════════════════════════════════════════════════ */
.video-frame { background: #04080F; border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
.camera-empty { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 400px; text-align: center; padding: 24px; }
.camera-empty .big { font-size: 14px; font-weight: 700; letter-spacing: 1.5px; color: var(--text); }
.camera-empty .sub { font-size: 13px; color: var(--text-2); margin-top: 6px; max-width: 420px; }

/* ══════════════════════════════════════════════════════════
   DETECTION ROWS
   ══════════════════════════════════════════════════════════ */
.det-row { display: flex; align-items: center; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(30,53,83,0.55); }
.det-row:last-child { border-bottom: none; }
.det-row .lbl { font-size: 13px; color: var(--text-2); }
.det-row .val { font-size: 15px; font-weight: 700; color: var(--text); }
.det-row .val.fall { color: var(--danger); }

/* ══════════════════════════════════════════════════════════
   KPI CARDS — strict 4-col, identical sizing
   ══════════════════════════════════════════════════════════ */
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--gap) 16px; text-align: left; transition: border-color 0.2s, box-shadow 0.2s; }
.kpi:hover { border-color: var(--primary); box-shadow: 0 0 0 1px var(--primary), var(--shadow-lg); }
.kpi .kpi-icon { font-size: 20px; margin-bottom: 4px; }
.kpi .kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: var(--text-2); line-height: 1.4; }
.kpi .kpi-value { font-size: 28px; font-weight: 800; color: var(--text); margin-top: 4px; line-height: 1.2; }
.kpi .kpi-trend { font-size: 11px; font-weight: 600; margin-top: 2px; }
.kpi .kpi-trend.up { color: var(--success); }
.kpi .kpi-trend.down { color: var(--danger); }
.kpi .kpi-trend.neutral { color: var(--text-2); }
.kpi.danger .kpi-value { color: var(--danger); }

/* Notification center */
.notif-bell { position: relative; display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 40px; border-radius: var(--radius-sm); background: var(--surface); border: 1px solid var(--border); cursor: pointer; font-size: 18px; }
.notif-badge { position: absolute; top: -4px; right: -4px; min-width: 18px; height: 18px; border-radius: 9px; background: var(--danger); color: #fff; font-size: 10px; font-weight: 700; display: flex; align-items: center; justify-content: center; padding: 0 4px; }
.notif-panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--gap); max-height: 320px; overflow-y: auto; }
.notif-item { display: flex; align-items: flex-start; gap: 10px; padding: 8px 0; border-bottom: 1px solid rgba(30,53,83,0.55); }
.notif-item:last-child { border-bottom: none; }
.notif-item .notif-icon { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; flex-shrink: 0; }
.notif-item .notif-icon.fall { background: var(--danger-soft); color: var(--danger); }
.notif-item .notif-icon.recovery { background: var(--success-soft); color: var(--success); }
.notif-item .notif-icon.system { background: var(--primary-soft); color: var(--primary-bright); }
.notif-item .notif-body { flex: 1; }
.notif-item .notif-msg { font-size: 12px; color: var(--text); font-weight: 500; }
.notif-item .notif-time { font-size: 11px; color: var(--text-2); margin-top: 2px; }
.notif-item .notif-dismiss { background: none; border: none; color: var(--text-2); cursor: pointer; font-size: 14px; padding: 2px; }
.notif-item .notif-dismiss:hover { color: var(--text); }

/* Contact system */
.contact-panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--gap-md); }
.contact-card { display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(30,53,83,0.55); }
.contact-card:last-child { border-bottom: none; }
.contact-info { display: flex; align-items: center; gap: 10px; }
.contact-avatar { width: 32px; height: 32px; border-radius: 50%; background: var(--surface-2); display: flex; align-items: center; justify-content: center; font-size: 14px; }
.contact-name { font-size: 13px; font-weight: 600; color: var(--text); }
.contact-phone { font-size: 11px; color: var(--text-2); }
.contact-actions { display: flex; gap: 6px; }
.contact-actions button { padding: 5px 10px; border-radius: var(--radius-sm); border: 1px solid var(--border); background: var(--surface-2); color: var(--text-2); font-size: 11px; font-weight: 600; cursor: pointer; }
.contact-actions button:hover { border-color: var(--primary); color: var(--text); }

/* Fall risk gauge */
.risk-gauge { position: relative; width: 100%; height: 12px; background: var(--surface-2); border-radius: 6px; overflow: hidden; }
.risk-gauge .risk-fill { height: 100%; border-radius: 6px; transition: width 0.5s ease, background 0.3s ease; }
.risk-gauge .risk-fill.low { background: var(--success); }
.risk-gauge .risk-fill.medium { background: var(--warning); }
.risk-gauge .risk-fill.high { background: var(--danger); }

/* Confidence bar */
.conf-bar { width: 100%; height: 6px; background: var(--surface-2); border-radius: 3px; overflow: hidden; margin-top: 6px; }
.conf-bar .conf-fill { height: 100%; background: var(--primary-bright); border-radius: 3px; transition: width 0.3s ease; }

/* Vitals mock */
.vital-row { display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(30,53,83,0.4); }
.vital-row:last-child { border-bottom: none; }
.vital-label { font-size: 11px; color: var(--text-2); display: flex; align-items: center; gap: 6px; }
.vital-value { font-size: 13px; font-weight: 700; color: var(--text); }

/* Right panel (analysis sidebar) */
.analysis-panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--gap-md); display: flex; flex-direction: column; gap: var(--gap-md); }
.analysis-card { background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 12px; }
.analysis-card-title { font-size: 10px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: var(--text-2); margin-bottom: 8px; }

/* ══════════════════════════════════════════════════════════
   BUTTONS — consistent heights
   ══════════════════════════════════════════════════════════ */
[data-testid="stBaseButton-secondary"] { border-color: var(--border) !important; background: var(--surface) !important; color: var(--text-2) !important; border-radius: var(--radius-sm) !important; }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--primary) !important; color: var(--text) !important; }
[data-testid="stBaseButton-primary"] { background: var(--primary) !important; border-color: var(--primary) !important; color: #fff !important; border-radius: var(--radius-sm) !important; }
[data-testid="stBaseButton-primary"]:hover { background: var(--primary-bright) !important; }
.control-grid [data-testid="stBaseButton-secondary"],
.control-grid [data-testid="stBaseButton-primary"] {
  height: 44px !important;
  min-height: 44px !important;
  font-size: 14px !important;
}

/* ══════════════════════════════════════════════════════════
   STREAMLIT COLUMN / SECTION SPACING NORMALIZATION
   ══════════════════════════════════════════════════════════ */
[data-testid="stVerticalBlock"] > div { margin-bottom: 0; }
[data-testid="stHorizontalBlock"] { gap: var(--gap) !important; }
[data-testid="stHorizontalBlock"] > div { min-width: 0; }

/* nav columns: all children buttons same height */
[data-testid="stHorizontalBlock"] [data-testid="stBaseButton-secondary"],
[data-testid="stHorizontalBlock"] [data-testid="stBaseButton-primary"] {
  height: var(--nav-h) !important;
  min-height: var(--nav-h) !important;
  max-height: var(--nav-h) !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

/* control buttons: equal height */
div[data-testid="stHorizontalBlock"]:has([data-testid="stBaseButton-primary"][key="start_webcam_btn"]) [data-testid="stBaseButton-secondary"],
div[data-testid="stHorizontalBlock"]:has([data-testid="stBaseButton-primary"][key="start_webcam_btn"]) [data-testid="stBaseButton-primary"] {
  height: 44px !important;
  min-height: 44px !important;
}

/* KPI columns: equal sizing */
div[data-testid="stHorizontalBlock"]:has(.kpi) > div { flex: 1 1 0 !important; }

/* ══════════════════════════════════════════════════════════
   ALERTS / EMERGENCY
   ══════════════════════════════════════════════════════════ */
[data-testid="stAlert"] { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--primary-bright); color: var(--text); }
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p { color: var(--text-2); }
[data-testid="stAlert"][data-baseweb] { color: var(--text); }

.emergency { border-radius: var(--radius); padding: var(--gap) 16px; border: 1px solid var(--border); }
.emergency.normal { background: var(--success-soft); border-color: rgba(34,197,94,0.45); }
.emergency.caution { background: var(--warning-soft); border-color: rgba(245,158,11,0.6); }
.emergency.critical { background: var(--danger-soft); border-color: rgba(239,68,68,0.6); animation: pulse 1.2s ease-in-out infinite; }
.emergency.ok { background: var(--primary-soft); border-color: rgba(37,99,235,0.45); }
.emergency.alarm { background: rgba(239,68,68,0.12); border-color: rgba(239,68,68,0.6); animation: pulse 1.2s ease-in-out infinite; }
.emergency .em-title { font-size: 13px; font-weight: 800; letter-spacing: 1px; }
.emergency.normal .em-title { color: var(--success); }
.emergency.caution .em-title { color: var(--warning); }
.emergency.critical .em-title { color: #fff; }
.emergency.ok .em-title { color: var(--text); }
.emergency.alarm .em-title { color: #fff; }
.emergency .em-sub { font-size: 12px; margin-top: 3px; }
.emergency.normal .em-sub { color: var(--text-2); }
.emergency.caution .em-sub { color: var(--text-2); }
.emergency.critical .em-sub { color: rgba(255,255,255,0.85); }
.emergency.ok .em-sub { color: var(--text-2); }
.emergency.alarm .em-sub { color: rgba(255,255,255,0.85); }

/* Emergency banner (full-width, above monitoring grid) */
.em-banner { display: flex; align-items: center; justify-content: space-between; padding: 14px 20px; border-radius: var(--radius); border: 1px solid var(--border); background: var(--surface); margin-bottom: var(--gap-md); }
.em-banner.normal { background: var(--success-soft); border-color: rgba(34,197,94,0.45); border-left: 3px solid var(--success); }
.em-banner.caution { background: var(--warning-soft); border-color: rgba(245,158,11,0.6); border-left: 3px solid var(--warning); }
.em-banner.critical { background: var(--danger-soft); border-color: rgba(239,68,68,0.6); border-left: 3px solid var(--danger); animation: pulse 1.2s ease-in-out infinite; }
.em-banner .em-left { display: flex; align-items: center; gap: 12px; }
.em-banner .em-status-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.em-banner.normal .em-status-dot { background: var(--success); box-shadow: 0 0 8px rgba(34,197,94,0.7); }
.em-banner.caution .em-status-dot { background: var(--warning); box-shadow: 0 0 8px rgba(245,158,11,0.7); }
.em-banner.critical .em-status-dot { background: var(--danger); box-shadow: 0 0 8px rgba(239,68,68,0.7); }
.em-banner .em-text { font-size: 13px; font-weight: 700; letter-spacing: 0.5px; }
.em-banner.normal .em-text { color: var(--success); }
.em-banner.caution .em-text { color: var(--warning); }
.em-banner.critical .em-text { color: #fff; }
.em-banner .em-meta { font-size: 11px; color: var(--text-2); margin-top: 2px; }
.em-banner .em-right { display: flex; align-items: center; gap: 12px; }
.em-banner .em-duration { font-size: 11px; color: var(--text-2); }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.3);} 50% { box-shadow: 0 0 18px 4px rgba(239,68,68,0.45);} }

/* ══════════════════════════════════════════════════════════
   MISC COMPONENTS
   ══════════════════════════════════════════════════════════ */
.ok-box { background: var(--primary-soft); border: 1px solid rgba(37,99,235,0.45); border-left: 3px solid var(--primary-bright); border-radius: var(--radius-sm); padding: 10px 14px; font-size: 13px; color: var(--text); }

.call-btn { display: flex; align-items: center; justify-content: space-between; gap: 8px; background: var(--primary); color: #fff; padding: 9px 12px; border-radius: var(--radius-sm); text-decoration: none; font-weight: 600; font-size: 13px; margin-top: 6px; border: none; cursor: pointer; }
.call-btn:hover { filter: brightness(1.1); color: #fff; }
.call-btn small { opacity: 0.8; font-weight: 500; }
.call-btn.disabled { background: var(--surface-2); color: var(--text-2); }

.stepper { display: flex; flex-wrap: wrap; gap: var(--gap-sm); margin: 6px 0 14px; }
.step { flex: 1; min-width: 100px; text-align: center; padding: 8px 6px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-sm); }
.step .n { width: 20px; height: 20px; line-height: 20px; border-radius: 50%; background: var(--surface-2); color: var(--text-2); font-size: 11px; font-weight: 700; margin: 0 auto 5px; }
.step .t { font-size: 11px; font-weight: 600; color: var(--text-2); }
.step.done { border-color: rgba(59,130,246,0.55); }
.step.done .n { background: var(--primary); color: #fff; }
.step.done .t { color: var(--text); }

.metric-pill { display: inline-flex; align-items: center; gap: 6px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-pill); padding: 5px 12px; font-weight: 700; font-size: 12px; color: var(--text); }

footer { text-align: center; padding: 0.8rem 0; color: var(--text-2); font-size: 12px; border-top: 1px solid var(--border); margin-top: var(--gap-md); }

[data-testid="stFileUploaderDropzone"] { background: var(--surface); border: 1px dashed var(--border); border-radius: 10px; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--primary); }

[data-testid="stExpander"] details { border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
[data-testid="stExpander"] summary { color: var(--text); font-weight: 600; }

.stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div { background: var(--surface-2); color: var(--text); border-radius: var(--radius-sm); }
[data-testid="stSliderThumb"] { background: var(--primary); }

.stPlotlyChart { background: transparent; }
.stProgress > div > div > div > div { background: var(--primary); }

/* ══════════════════════════════════════════════════════════
   RESPONSIVE — smaller screens
   ══════════════════════════════════════════════════════════ */
@media (max-width: 1400px) {
  .monitor-grid { grid-template-columns: 1fr; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
/* ══════════════════════════════════════════════════════════
   PERFORMANCE DASHBOARD — strict CSS Grid metric cards
   ══════════════════════════════════════════════════════════ */
.perf-section { margin-bottom: var(--gap-lg); }
.perf-section-title { font-size: 14px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: var(--text-2); margin-bottom: var(--gap-sm); padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.perf-grid { display: grid; gap: var(--gap-sm); }
.perf-grid.cols-5 { grid-template-columns: repeat(5, 1fr); }
.perf-grid.cols-4 { grid-template-columns: repeat(4, 1fr); }
.perf-grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
.perf-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 14px 16px; display: flex; flex-direction: column; min-height: 80px; transition: border-color 0.2s, box-shadow 0.2s; }
.perf-card:hover { border-color: rgba(37,99,235,0.4); box-shadow: 0 0 0 1px rgba(37,99,235,0.2); }
.perf-card .pc-label { font-size: 10px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: var(--text-2); margin-bottom: 6px; line-height: 1.2; }
.perf-card .pc-value { font-size: 26px; font-weight: 800; color: var(--text); line-height: 1.1; margin-top: auto; }
.perf-card .pc-value.na { font-size: 14px; font-weight: 500; color: var(--text-2); }
.perf-card .pc-sub { font-size: 11px; color: var(--text-2); margin-top: 4px; }
.perf-card.danger .pc-value { color: var(--danger); }
.perf-card.success .pc-value { color: var(--success); }
.perf-card.primary .pc-value { color: var(--primary-bright); }
.perf-card.winner { border-color: rgba(34,197,94,0.5); }
.perf-card.winner::after { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: var(--success); border-radius: var(--radius-sm) var(--radius-sm) 0 0; }

/* Comparison table */
.perf-table { width: 100%; border-collapse: collapse; }
.perf-table th { font-size: 10px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: var(--text-2); text-align: left; padding: 10px 14px; border-bottom: 2px solid var(--border); background: var(--surface); }
.perf-table td { font-size: 13px; font-weight: 600; color: var(--text); padding: 10px 14px; border-bottom: 1px solid rgba(30,53,83,0.4); }
.perf-table tr:last-child td { border-bottom: none; }
.perf-table tr:hover td { background: rgba(37,99,235,0.05); }
.perf-table .improved { color: var(--success); }
.perf-table .degraded { color: var(--danger); }
.perf-table .metric-name { color: var(--text-2); font-weight: 500; }

/* Model comparison */
.model-compare { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--gap-sm); }
.model-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 14px 16px; position: relative; }
.model-card.best { border-color: rgba(34,197,94,0.5); }
.model-card.best::before { content: "SELECTED"; position: absolute; top: 8px; right: 8px; font-size: 9px; font-weight: 700; letter-spacing: 1px; color: var(--success); background: var(--success-soft); padding: 2px 6px; border-radius: var(--radius-pill); }
.model-card .mc-name { font-size: 13px; font-weight: 700; color: var(--text); margin-bottom: 8px; }
.model-card .mc-metric { display: flex; justify-content: space-between; padding: 3px 0; }
.model-card .mc-metric .lbl { font-size: 11px; color: var(--text-2); }
.model-card .mc-metric .val { font-size: 12px; font-weight: 700; color: var(--text); }

/* Image grid for evaluation outputs */
.perf-image-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--gap-md); }
.perf-image-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; }
.perf-image-card .pic-title { font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: var(--text-2); padding: 10px 14px; border-bottom: 1px solid var(--border); }
.perf-image-card img { width: 100%; display: block; }

@media (max-width: 1400px) {
  .perf-grid.cols-5 { grid-template-columns: repeat(3, 1fr); }
  .perf-grid.cols-4 { grid-template-columns: repeat(2, 1fr); }
  .perf-image-grid { grid-template-columns: 1fr; }
}
@media (max-width: 768px) {
  .perf-grid.cols-5 { grid-template-columns: repeat(2, 1fr); }
  .perf-grid.cols-4 { grid-template-columns: repeat(2, 1fr); }
  .model-compare { grid-template-columns: 1fr; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 11. SESSION STATE
# ══════════════════════════════════════════════════════════
for key in ["prediction_history", "fall_count", "normal_count", "total_predictions",
            "activity_log", "webcam_captured_frame", "fall_history"]:
    if key not in st.session_state:
        if key in ["fall_count", "normal_count", "total_predictions"]:
            st.session_state[key] = 0
        elif key == "webcam_captured_frame":
            st.session_state[key] = None
        else:
            st.session_state[key] = [] if "history" in key or "log" in key else 0
for _k in ["fall_active", "siren_on"]:
    if _k not in st.session_state:
        st.session_state[_k] = False
if "view" not in st.session_state:
    st.session_state.view = "live"
if "mode_tab" not in st.session_state or st.session_state.mode_tab not in ("Live Camera", "Image", "Video"):
    st.session_state.mode_tab = "Live Camera"
if "pose_mode" not in st.session_state:
    st.session_state.pose_mode = "Fast"
_POSE_MODE = st.session_state.get("pose_mode", "Fast")

# ══════════════════════════════════════════════════════════
# 12. HISTORY HELPERS
# ══════════════════════════════════════════════════════════
def log_prediction(activity, confidence):
    st.session_state.total_predictions += 1
    if activity in ("Falling", "Fall"):
        st.session_state.fall_count += 1
    else:
        st.session_state.normal_count += 1
    st.session_state.prediction_history.append(
        {"time": time.strftime("%H:%M:%S"), "activity": activity, "confidence": f"{confidence:.0%}"})

def log_fall_event(fall_prob, status):
    st.session_state.fall_history.append(
        {"time": time.strftime("%H:%M:%S"), "status": status, "fall_prob": f"{fall_prob:.0%}"})
    st.session_state.fall_history = st.session_state.fall_history[-50:]

def _avg_confidence():
    if not st.session_state.prediction_history:
        return 0
    try:
        confs = [float(p["confidence"].replace("%", "")) for p in st.session_state.prediction_history[-20:]]
        return float(np.mean(confs)) if confs else 0
    except Exception:
        return 0

def _fall_contacts_html():
    ec_name = st.session_state.get("ec_name", "").strip()
    ec_phone = st.session_state.get("ec_phone", "").strip()
    amb_phone = st.session_state.get("amb_phone", "").strip()
    doc_name = st.session_state.get("doc_name", "").strip()
    doc_phone = st.session_state.get("doc_phone", "").strip()
    def btn(icon, label, phone):
        if phone:
            return f'<a class="call-btn" href="tel:{phone}">{icon} {label} <small>{phone}</small></a>'
        return f'<span class="call-btn disabled">⚠️ {label} <small>add in Alert &amp; Sound Setup</small></span>'
    rows = [btn("📞", ec_name or "Emergency Contact", ec_phone),
            btn("🚑", "Ambulance", amb_phone),
            btn("🩺", doc_name or "Doctor", doc_phone)]
    return '<div class="call-grid">' + "".join(rows) + "</div>"

def _siren_wav_b64(dur=3.0, sr=16000):
    n = int(sr * dur)
    t = np.arange(n) / sr
    seg = 0.35
    cycle = seg * 2
    freq = np.where((t % cycle) < seg, 620.0, 940.0)
    phase = np.cumsum(freq) / sr
    snd = 0.7 * np.sin(2 * np.pi * phase)
    snd += 0.25 * np.sin(2 * np.pi * phase * 2)
    snd += 0.1 * np.sin(2 * np.pi * phase * 4)
    env = np.minimum(1.0, np.minimum(t / 0.02, (dur - t) / 0.05))
    pcm = (snd * env * 0.72 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()

_SIREN_WAV_B64 = _siren_wav_b64()

SIREN_HTML = """
<div id="siren-root"></div>
<script>
(function(){
  if (window.__sirenReady) return;
  window.__sirenReady = true;
  var B64 = "SIREN_B64_PLACEHOLDER";
  function audioEl(){
    if (!window.__sirenEl) {
      try {
        window.__sirenEl = new Audio("data:audio/wav;base64," + B64);
        window.__sirenEl.loop = true;
        window.__sirenEl.volume = 1;
      } catch(e){}
    }
    return window.__sirenEl;
  }
  window.__initAudio = function(){
    var a = audioEl();
    if (!a) return;
    try {
      var p = a.play();
      if (p && p.then) { p.then(function(){ setTimeout(function(){ try{ a.pause(); a.currentTime = 0; }catch(e){} }, 100); }).catch(function(){}); }
    } catch(e){}
    try { if ("Notification" in window && Notification.permission === "default") Notification.requestPermission(); } catch(e){}
  };
  window.__playSiren = function(){
    var a = audioEl();
    if (!a) return;
    try {
      a.currentTime = 0;
      var p = a.play();
      if (p && p.catch) p.catch(function(){});
      window.__sirenOn = true;
      clearTimeout(window.__sirenTimer);
      window.__sirenTimer = setTimeout(function(){ window.__stopSiren(); }, 25000);
    } catch(e){}
  };
  window.__stopSiren = function(){
    try { var a = window.__sirenEl; if (a) { a.pause(); a.currentTime = 0; } } catch(e){}
    window.__sirenOn = false;
  };
  window.__notify = function(msg){
    try {
      if ("Notification" in window && Notification.permission === "granted") {
        new Notification("🚨 Fall Detected", { body: msg, tag: "fall-alert" });
      }
    } catch(e){}
  };
  document.addEventListener("click", window.__initAudio, {once:false});
  document.addEventListener("pointerdown", window.__initAudio, {once:false});
})();
</script>
""".replace("SIREN_B64_PLACEHOLDER", _SIREN_WAV_B64)

# ══════════════════════════════════════════════════════════
# 13. RECORD & TRAIN HELPERS
# ══════════════════════════════════════════════════════════
PROJECT_DIR = Path(__file__).parent
RECORDED_DIR = PROJECT_DIR / "dataset" / "recorded"
RECORD_ACTIVITIES = [
    "Standing", "Walking", "Sitting", "Jumping", "Lying Down",
    "Falling", "Crawling", "Bending", "Crouching", "Kneeling",
    "Getting Up",
]

def _recorded_counts():
    counts = {}
    if RECORDED_DIR.exists():
        for d in sorted(RECORDED_DIR.iterdir()):
            if d.is_dir():
                counts[d.name] = {
                    "frames": sum(1 for _ in d.glob("*.jpg")),
                    "videos": sum(1 for _ in d.glob("*.avi")) + sum(1 for _ in d.glob("*.mp4")),
                }
    return counts

SESSIONS_FILE = RECORDED_DIR / "sessions.json"

def _load_sessions():
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text())
        except Exception:
            pass
    return {"record_sessions": {}, "record_frames": {}, "train_runs": 0, "train_history": []}

def _save_sessions(s):
    try:
        RECORDED_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_FILE.write_text(json.dumps(s, indent=2))
    except Exception:
        pass

# ══════════════════════════════════════════════════════════
# 14. SMALL UI BUILDERS
# ══════════════════════════════════════════════════════════
def _kpi_html(label, value, danger=False):
    cls = "kpi danger" if danger else "kpi"
    return f'<div class="{cls}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'

def _ok_box(msg):
    st.markdown(f'<div class="ok-box">&#10003; {msg}</div>', unsafe_allow_html=True)

def _fill_kpis(phs):
    avg_conf = _avg_confidence()
    phs[0].markdown(_kpi_html("Total Detections", st.session_state.total_predictions), unsafe_allow_html=True)
    phs[1].markdown(_kpi_html("Fall Events", st.session_state.fall_count, danger=st.session_state.fall_count > 0), unsafe_allow_html=True)
    phs[2].markdown(_kpi_html("Normal Activities", st.session_state.normal_count), unsafe_allow_html=True)
    phs[3].markdown(_kpi_html("Avg Confidence", f"{avg_conf:.0f}%"), unsafe_allow_html=True)

def _kpi_static_row():
    k1, k2, k3, k4 = st.columns(4)
    phs = [k1.empty(), k2.empty(), k3.empty(), k4.empty()]
    _fill_kpis(phs)

def _detection_html(activity, confidence, pose_status, fall_status, fall=False):
    rows = ""
    for lbl, val, is_fall in [("CURRENT ACTIVITY", activity, False),
                              ("CONFIDENCE", f"{confidence:.0%}", False),
                              ("POSE STATUS", pose_status, False),
                              ("FALL STATUS", fall_status, fall)]:
        cls = ' class="val fall"' if is_fall else ' class="val"'
        rows += f'<div class="det-row"><span class="lbl">{lbl}</span><span{cls}>{val}</span></div>'
    return f'<div class="panel"><div class="panel-title">Current Detection</div>{rows}</div>'

def _empty_detection_html():
    return _detection_html("—", 0.0, "Camera offline", "No fall detected")

def _emergency_html(fall, confidence=None, activity=None):
    if fall:
        sub = f'Fall confidence <b>{confidence:.0%}</b>' if confidence is not None else "Immediate attention required"
        if activity:
            sub += f" &middot; {activity}"
        html = (f'<div class="emergency alarm"><div class="em-title">&#9888; FALL DETECTED</div>'
                f'<div class="em-sub">{sub} — check on the person immediately.</div></div>')
        html += '<div class="call-grid">' + "".join(_call_btn_rows()) + "</div>"
        return html
    return ('<div class="emergency ok"><div class="em-title">&#10003; NO FALL DETECTED</div>'
            '<div class="em-sub">Monitoring normally</div></div>')

def _call_btn_rows():
    ec_name = st.session_state.get("ec_name", "").strip()
    ec_phone = st.session_state.get("ec_phone", "").strip()
    amb_phone = st.session_state.get("amb_phone", "").strip()
    doc_name = st.session_state.get("doc_name", "").strip()
    doc_phone = st.session_state.get("doc_phone", "").strip()
    def btn(icon, label, phone):
        if phone:
            return f'<a class="call-btn" href="tel:{phone}">{icon} {label} <small>{phone}</small></a>'
        return f'<span class="call-btn disabled">&#9888;&#65039; {label} <small>set in Alert &amp; Sound Setup</small></span>'
    return [btn("📞", ec_name or "Emergency Contact", ec_phone),
            btn("🚑", "Ambulance", amb_phone),
            btn("🩺", doc_name or "Doctor", doc_phone)]

def _stepper_html(done_flags):
    steps = ["Record", "Review", "Extract Features", "Train", "Evaluate", "Save Model"]
    cells = []
    for i, (s, done) in enumerate(zip(steps, done_flags), 1):
        cls = "step done" if done else "step"
        cells.append(f'<div class="{cls}"><div class="n">{i}</div><div class="t">{s}</div></div>')
    return '<div class="stepper">' + "".join(cells) + "</div>"

def _render_hero():
    n_acts = len(_ml_activities) if _ml_activities else len(ACTIVITIES)
    st.markdown(f"""
    <div class="hero">
      <h2>AI-Powered Human Activity &amp; Fall Detection</h2>
      <p>Real-time pose estimation, activity recognition and fall monitoring.</p>
      <div class="hero-badges">
        <span class="badge">MediaPipe Pose</span>
        <span class="badge">Random Forest</span>
        <span class="badge"><b>33</b> Landmarks</span>
        <span class="badge"><b>{n_acts}</b> Activities</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 15. HEADER / NAVIGATION
# ══════════════════════════════════════════════════════════
def _render_header():
    """Top header: brand / nav / system status on a fixed grid, then an aligned
    control row with POSE MODEL and the input-mode tabs. Returns the active view."""
    view = st.session_state.get("view", "live")
    ml_active = _ml_model is not None
    bcol, ncol, scol = st.columns([1.5, 2.8, 1.2], vertical_alignment="center")
    with bcol:
        st.markdown('<div class="app-brand"><h1>AI <span>CARE</span></h1><p>Human Activity &amp; Fall Detection</p></div>',
                    unsafe_allow_html=True)
    with ncol:
        n1, n2, n3, n4 = st.columns(4)
        if n1.button("Live Monitoring", key="nav_live",
                     type="primary" if view == "live" else "secondary", width="stretch"):
            st.session_state.view = "live"
            st.session_state.mode_tab = "Live Camera"
        if n2.button("Analytics", key="nav_analytics",
                     type="primary" if view == "analytics" else "secondary", width="stretch"):
            st.session_state.view = "analytics"
        if n3.button("Model Performance", key="nav_perf",
                     type="primary" if view == "performance" else "secondary", width="stretch"):
            st.session_state.view = "performance"
        if n4.button("Record & Train", key="nav_train",
                     type="primary" if view == "train" else "secondary", width="stretch"):
            st.session_state.view = "train"
    with scol:
        cls = "system-status active" if ml_active else "system-status"
        label = "SYSTEM ACTIVE" if ml_active else "MODEL NOT LOADED"
        st.markdown(f'<div class="{cls}"><span class="dot"></span>{label}</div>', unsafe_allow_html=True)

    view = st.session_state.get("view", "live")
    if view in ("live", "image", "video"):
        r2a, r2b, r2c = st.columns([1.5, 2.8, 1.2], vertical_alignment="center")
        with r2a:
            st.markdown('<div class="ctl-label">POSE MODEL</div>', unsafe_allow_html=True)
            st.segmented_control("Pose model", ["Fast", "Accurate"],
                                 key="pose_mode", label_visibility="collapsed")
        global _POSE_MODE
        _POSE_MODE = st.session_state.get("pose_mode", "Fast")
        with r2b:
            mode = st.segmented_control(
                "Input mode", ["Live Camera", "Image", "Video"],
                key="mode_tab", label_visibility="collapsed")
            view = {"Live Camera": "live", "Image": "image", "Video": "video"}.get(mode, view)
            st.session_state.view = view
    return view

# ══════════════════════════════════════════════════════════
# 16. LIVE MONITORING
# ══════════════════════════════════════════════════════════
def _risk_gauge_html(risk_pct):
    cls = "low" if risk_pct < 30 else ("medium" if risk_pct < 70 else "high")
    return (f'<div class="risk-gauge"><div class="risk-fill {cls}" style="width:{risk_pct:.0f}%"></div></div>'
            f'<div style="display:flex;justify-content:space-between;margin-top:3px;">'
            f'<span style="font-size:10px;color:var(--text-2);">LOW</span>'
            f'<span style="font-size:11px;font-weight:700;color:var(--text);">{risk_pct:.0f}%</span>'
            f'<span style="font-size:10px;color:var(--text-2);">HIGH</span></div>')

def _confidence_bar_html(confidence):
    return (f'<div class="conf-bar"><div class="conf-fill" style="width:{confidence*100:.0f}%"></div></div>'
            f'<div style="font-size:11px;font-weight:600;color:var(--primary-bright);margin-top:2px;">{confidence:.0%}</div>')

def _vital_row_html(icon, label, value):
    return f'<div class="vital-row"><span class="vital-label">{icon} {label}</span><span class="vital-value">{value}</span></div>'

def _analysis_html(activity, confidence, fall_prob, is_fall, fps_val, pose_status):
    risk_pct = min(100, fall_prob * 100) if is_fall else min(100, fall_prob * 50)
    risk_cls = "low" if risk_pct < 30 else ("medium" if risk_pct < 70 else "high")
    activity_color = "var(--danger)" if is_fall else "var(--success)"
    return f"""
    <div class="analysis-panel">
      <div class="analysis-card">
        <div class="analysis-card-title">Current Activity</div>
        <div style="font-size:18px;font-weight:800;color:{activity_color};margin-bottom:4px;">{activity}</div>
        {_confidence_bar_html(confidence)}
      </div>
      <div class="analysis-card">
        <div class="analysis-card-title">Fall Risk</div>
        {_risk_gauge_html(risk_pct)}
      </div>
      <div class="analysis-card">
        <div class="analysis-card-title">Posture Analysis</div>
        {_vital_row_html("🦴", "Pose", pose_status)}
        {_vital_row_html("📊", "Confidence", f"{confidence:.0%}")}
        {_vital_row_html("⚡", "Fall Signal", f"{fall_prob:.2f}" if fall_prob else "—")}
      </div>
      <div class="analysis-card">
        <div class="analysis-card-title">Vitals (estimated)</div>
        {_vital_row_html("💓", "Heart Rate", f"{68 + int(confidence*20)} bpm")}
        {_vital_row_html("🏃", "Movement", "High" if activity in ("Running","Jumping") else "Moderate" if activity in ("Walking","Climbing Stairs") else "Low")}
        {_vital_row_html("🔥", "Intensity", f"{confidence*100:.0f}%")}
      </div>
      <div class="analysis-card">
        <div class="analysis-card-title">System</div>
        {_vital_row_html("📷", "FPS", f"{fps_val:.0f}")}
        {_vital_row_html("🧠", "Model", "Active" if _ml_model else "N/A")}
        {_vital_row_html("📐", "Pose", "Fast" if st.session_state.get("pose_mode") == "Fast" else "Accurate")}
      </div>
    </div>"""

def _emergency_banner_html(is_fall, fall_prob=None, activity=None, duration_str=""):
    if is_fall:
        cls = "critical"
        title = "⚠ FALL DETECTED"
        sub = f"Fall confidence {fall_prob:.0%}" if fall_prob is not None else "Immediate attention required"
        if activity: sub += f" · {activity}"
    else:
        cls = "normal"
        title = "✓ SYSTEM NORMAL"
        sub = "All monitoring systems operational"
    return (f'<div class="em-banner {cls}">'
            f'<div class="em-left"><div class="em-status-dot"></div>'
            f'<div><div class="em-text">{title}</div><div class="em-meta">{sub}</div></div></div>'
            f'<div class="em-right"><div class="em-duration">{duration_str}</div></div></div>')


class LiveDetector:
    """Thread-safe wrapper around MediaPipe + ML pipeline for WebRTC.

    The browser sends frames via webrtc_streamer.  The video_frame_callback
    calls process() on a background thread.  Results are read via snapshot()
    from the Streamlit main thread."""

    def __init__(self):
        self.lock = threading.Lock()
        # ``async_processing=True`` permits WebRTC to invoke the callback from
        # more than one worker.  MediaPipe's VIDEO mode and TemporalAnalyzer
        # are sequential by design, so never queue a second expensive frame
        # behind one already being analysed.
        self.processing_lock = threading.Lock()
        self.last_processed_at = 0.0
        self.temporal = TemporalAnalyzer()
        self.fall_confirmer = FallConfirmer()
        self.activity = "No Person"
        self.confidence = 0.0
        self.fall_prob = 0.0
        self.fall_alarm = False
        self.fa = None
        self.fps = 0.0
        self.last_frame_time = 0.0
        self.fall_logged = False
        self.display = None
        self._missed_frames = 0
        self._last_landmarks = None
        self._last_all_poses = None
        self._HOLD_FRAMES = 8

    def process(self, image_bgr):
        """Process one frame, dropping excess WebRTC frames to stay live."""
        if not self.processing_lock.acquire(blocking=False):
            return image_bgr
        try:
            now = time.time()
            min_interval = 1.0 / max(1, config.LIVE_PROCESS_FPS)
            if now - self.last_processed_at < min_interval:
                return image_bgr
            self.last_processed_at = now
            return self._process_frame(image_bgr, now)
        finally:
            self.processing_lock.release()

    def _process_frame(self, image_bgr, now):
        h, w = image_bgr.shape[:2]
        if self.last_frame_time > 0:
            dt = max(now - self.last_frame_time, 1e-6)
            inst_fps = 1.0 / dt
            self.fps = inst_fps if self.fps == 0 else 0.9 * self.fps + 0.1 * inst_fps
        self.last_frame_time = now

        try:
            landmarks, _, all_poses = extract_landmarks_from_frame(
                image_bgr, timestamp_ms=now * 1000)
        except Exception:
            return image_bgr

        if landmarks is None:
            if self._missed_frames < self._HOLD_FRAMES and self._last_landmarks is not None:
                self._missed_frames += 1
                landmarks = self._last_landmarks
                all_poses = self._last_all_poses
            else:
                self.temporal.reset()
                self.fall_confirmer.reset()
                self._missed_frames = 0
                self._last_landmarks = None
                self._last_all_poses = None
                display = annotate_frame(image_bgr.copy(), None, "No Person", 0.0, 0.0, None, 0, h, w)
                with self.lock:
                    self.activity = "No Person"
                    self.confidence = 0.0
                    self.fall_prob = 0.0
                    self.fall_alarm = False
                    self.fa = None
                    self.display = display
                return display

        self._missed_frames = 0
        self._last_landmarks = landmarks
        self._last_all_poses = all_poses

        try:
            activity, confidence, fall_prob, fa = self.temporal.update(landmarks, now)
            activity = _simple_activity(activity)
            _is_fall = self.temporal.fall_alarm
            _confirmed = self.fall_confirmer.update(_is_fall, fa)
            display = annotate_frame(image_bgr.copy(), all_poses, activity, confidence,
                                     fall_prob, fa, 0, h, w, lite=True, fall_alarm=_confirmed)

            with self.lock:
                self.activity = activity
                self.confidence = confidence
                self.fall_prob = fall_prob
                self.fall_alarm = _confirmed
                self.fa = fa
                self.display = display

            if _confirmed and not self.fall_logged:
                self.fall_logged = True
            elif not _confirmed and self.fall_logged:
                self.fall_logged = False

            return display
        except Exception as e:
            cv2.putText(image_bgr, f"ML error: {str(e)[:60]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            return image_bgr

    def snapshot(self):
        with self.lock:
            return dict(activity=self.activity, confidence=self.confidence,
                        fall_prob=self.fall_prob, fall_alarm=self.fall_alarm,
                        fa=self.fa, fps=self.fps, display=self.display)

    def reset(self):
        with self.lock:
            self.temporal.reset()
            self.fall_confirmer.reset()
            self.activity = "No Person"
            self.confidence = 0.0
            self.fall_prob = 0.0
            self.fall_alarm = False
            self.fa = None
            self.display = None
            self.fall_logged = False
            self._missed_frames = 0
            self._last_landmarks = None
            self._last_all_poses = None
            self.last_processed_at = 0.0


_live_detector_ref = None

def _webrtc_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    try:
        image_bgr = frame.to_ndarray(format="bgr24")
        det = _live_detector_ref
        if det is None:
            return frame
        annotated = det.process(image_bgr)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
    except Exception:
        # A camera frame should never be able to terminate the WebRTC track.
        return frame


def _close_webcam():
    detector = st.session_state.pop("live_detector", None)
    if detector is not None:
        detector.reset()
    st.session_state.pop("webcam_on", None)


def _render_captured_analysis(container, cap_frame):
    """Static analysis for a captured frame."""
    with container.container():
        st.markdown('<div class="section-title">Captured Frame Analysis</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1], gap="small")
        c_h, c_w = cap_frame.shape[:2]
        c_landmarks, c_pose_lm, c_all_poses = extract_landmarks_from_frame(cap_frame)
        with c1:
            if c_landmarks is not None:
                c_act, c_conf, c_fa = classify_static(c_landmarks, temporal=None)
                c_fp = ml_fall_signal(c_landmarks) or 0.0
                c_disp = annotate_frame(cap_frame.copy(), c_all_poses, c_act, c_conf, c_fp, c_fa, 0, c_h, c_w)
                st.image(cv2.cvtColor(c_disp, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
            else:
                st.warning("No person detected.")
        with c2:
            if c_landmarks is not None:
                c_act = _simple_activity(c_act)
                is_fall = c_act == "Fall" or c_fp >= 0.55
                st.markdown(_analysis_html(c_act, c_conf, c_fp, is_fall, 0, "Static analysis"),
                            unsafe_allow_html=True)
                if is_fall:
                    st.markdown(_emergency_html(True, c_fp, c_act), unsafe_allow_html=True)
                else:
                    st.markdown(_emergency_html(False), unsafe_allow_html=True)
                with st.expander("Body Coordinates"):
                    st.dataframe(coords_dataframe(c_fa), width="stretch", hide_index=True)
                log_prediction(c_act, c_conf)


def render_live():
    global _live_detector_ref
    if not st.session_state.get("live_detector"):
        st.session_state["live_detector"] = LiveDetector()
    detector: LiveDetector = st.session_state["live_detector"]
    _live_detector_ref = detector

    # ── hero ────────────────────────────────────────────────
    n_acts = len(_ml_activities) if _ml_activities else len(ACTIVITIES)
    st.markdown(f"""
    <div class="hero" style="padding:var(--gap-sm) var(--gap-md);max-height:80px;overflow:hidden;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div><h2 style="font-size:16px;margin:0;">Elderly AI Monitor</h2>
        <p style="margin:1px 0 0;font-size:12px;">Real-time posture analysis &amp; fall detection</p></div>
        <div style="display:flex;gap:8px;">
          <span class="badge">MediaPipe Pose</span>
          <span class="badge">HistGB</span>
          <span class="badge"><b>{n_acts}</b> Activities</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── emergency banner ────────────────────────────────────
    snap = detector.snapshot()
    elapsed = time.time() - st.session_state.get("monitor_start", time.time())
    m, s = divmod(int(elapsed), 60)
    hrs, m = divmod(m, 60)
    dur = f"{hrs:02d}:{m:02d}:{s:02d}" if hrs else f"{m:02d}:{s:02d}"
    emergency_slot = st.empty()
    emergency_slot.markdown(_emergency_banner_html(False, duration_str=f"Uptime: {dur}"),
                            unsafe_allow_html=True)

    # ── two-column layout ───────────────────────────────────
    mon_col, side_col = st.columns([0.7, 0.3], gap="small")

    with mon_col:
        try:
            webrtc_ctx = webrtc_streamer(
                key="elderly-fall-live",
                video_frame_callback=_webrtc_frame_callback,
                rtc_configuration=_webrtc_rtc_configuration(),
                media_stream_constraints={
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                              "frameRate": {"ideal": config.LIVE_CAMERA_FPS,
                                            "max": config.LIVE_CAMERA_FPS},
                              "facingMode": "user"},
                    "audio": False,
                },
                async_processing=True,
            )
        except Exception:
            webrtc_ctx = None

        if webrtc_ctx is None or not webrtc_ctx.state.playing:
            st.empty()

        # ── captured-frame analysis ─────────────────────────
        if st.session_state.webcam_captured_frame is not None:
            _render_captured_analysis(st.empty(),
                                      st.session_state.webcam_captured_frame)

    with side_col:
        analysis_slot = st.empty()
        analysis_slot.markdown(
            _analysis_html("—", 0.0, 0.0, False, 0.0, "Awaiting camera"),
            unsafe_allow_html=True)

        with st.expander("⚠ Alert & Sound Setup"):
            st.html(SIREN_HTML)
            st.markdown(
                '<div style="text-align:center;margin:4px 0;">'
                '<button class="call-btn" onclick="window.__initAudio();'
                " this.innerText = '&#10004; Sound Enabled';"
                ' this.disabled = true;"'
                ' style="width:100%;justify-content:center;">'
                '&#128266; Enable Sound</button></div>',
                unsafe_allow_html=True)
            st.markdown('<div class="section-sub" style="margin:6px 0 2px;">'
                        'Emergency contacts</div>', unsafe_allow_html=True)
            st.text_input("Emergency Contact Name", key="ec_name",
                          placeholder="e.g. Daughter Priya")
            st.text_input("Emergency Contact Phone", key="ec_phone",
                          placeholder="+91 98xxxxxx21")
            st.text_input("Ambulance Phone", key="amb_phone", value="108")
            st.text_input("Doctor Name", key="doc_name",
                          placeholder="e.g. Dr. Sharma")
            st.text_input("Doctor Phone", key="doc_phone",
                          placeholder="+91 99xxxxxx87")

    # ── KPIs + Recent Activity (static slots) ───────────────
    k1, k2, k3, k4 = st.columns(4, gap="small")
    kpi_slots = [k1.empty(), k2.empty(), k3.empty(), k4.empty()]
    st.markdown('<div class="section-title">Recent Activity</div>',
                unsafe_allow_html=True)
    activity_slot = st.empty()

    with st.expander("Reference Posture Guide (Biomechanical Ranges)"):
        st.markdown(
            "Expected joint angles and body patterns for each activity "
            "(typical values from biomechanical references). The system "
            "compares current measured angles against these ranges.")
        st.dataframe(reference_guide_dataframe(), width="stretch",
                     hide_index=True)

    # Do not use a ``while webrtc_ctx.state.playing`` loop here.  It blocks
    # Streamlit's script runner, making localhost appear frozen and preventing
    # the Cloud frontend from completing normal component updates.  The video
    # frame callback above continues independently on WebRTC's worker thread.
    # These values refresh whenever Streamlit reruns (for example after a UI
    # action), while the actual video remains live in the component.
    if webrtc_ctx is not None and webrtc_ctx.state.playing:
        if "monitor_start" not in st.session_state:
            st.session_state.monitor_start = time.time()
        snap = detector.snapshot()
        activity = snap["activity"]
        confidence = snap["confidence"]
        fall_prob = snap["fall_prob"]
        is_fall = snap["fall_alarm"]
        fa_obj = snap["fa"]
        if activity != "No Person":
            log_prediction(_simple_activity(activity), confidence)

        if is_fall:
            emergency_slot.markdown(
                _emergency_banner_html(True, fall_prob, _simple_activity(activity)),
                unsafe_allow_html=True)
        pose_status = ("No person detected" if activity == "No Person"
                       else "Full body detected" if is_fall
                       else ("Full body detected"
                             if (fa_obj is not None and hasattr(fa_obj, 'visibility')
                                 and fa_obj.visibility and fa_obj.visibility >= 0.5)
                             else "Partial visibility"))
        analysis_slot.markdown(
            _analysis_html(_simple_activity(activity), confidence,
                           fall_prob, is_fall, 0, pose_status),
            unsafe_allow_html=True)
        _fill_kpis(kpi_slots)
        if st.session_state.prediction_history:
            activity_slot.dataframe(
                pd.DataFrame(st.session_state.prediction_history[-8:][::-1]),
                width="stretch", hide_index=True)
        else:
            activity_slot.info("No detections yet — monitoring active.")
    elif webrtc_ctx is not None:
        st.info("Click START to grant browser camera permission. If it still cannot connect on Streamlit Cloud, add your TURN credentials in the app secrets.")

# ══════════════════════════════════════════════════════════
# 17. IMAGE UPLOAD
# ══════════════════════════════════════════════════════════
def render_image():
    st.markdown('<div class="section-title">Image Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload an image to detect pose, body coordinates and classify activity.</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")
    if uploaded:
        try:
            uploaded.seek(0)
            image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            image = None
        if image is None or image.size == 0:
            st.error("Could not read the uploaded image.")
        else:
            icol1, icol2 = st.columns(2)
            with icol1:
                st.markdown('<div class="panel-title">Original</div>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
            with st.spinner("Analyzing..."):
                landmarks, pose_lm, all_poses = extract_landmarks_from_frame(image)
                if landmarks is not None:
                    activity, confidence, fa = classify_static(landmarks, temporal=None)
                    activity = _simple_activity(activity)
                    fall_prob = ml_fall_signal(landmarks) or 0.0
                    pose_img = annotate_frame(image.copy(), all_poses, activity, confidence, fall_prob, fa, 0,
                                              image.shape[0], image.shape[1])
                else:
                    activity, confidence, fa, fall_prob = None, 0, None, 0
                    pose_img = annotate_frame(image.copy(), None, "No Person", 0.0, 0.0, None, 0,
                                              image.shape[0], image.shape[1])
            with icol2:
                st.markdown('<div class="panel-title">Pose Analysis</div>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB), width="stretch")
            if activity:
                is_fall = activity == "Fall" or fall_prob >= 0.55
                st.markdown(_emergency_html(is_fall, fall_prob if is_fall else None, activity), unsafe_allow_html=True)
                st.markdown(_detection_html(activity, confidence, "Static analysis",
                                            "Fall detected" if is_fall else "No fall detected", fall=is_fall),
                            unsafe_allow_html=True)
                log_prediction(activity, confidence)
                if fa is not None:
                    with st.expander("Body Coordinates & Angles"):
                        st.markdown("### Landmark Coordinates (x, y, z)")
                        st.dataframe(coords_dataframe(fa), width="stretch", hide_index=True)
                        st.markdown("### Joint Angles")
                        ang_rows = [
                            ("Torso", fa.torso), ("Head", fa.head), ("Hip", fa.hip_angle),
                            ("Knee (avg)", fa.knee_angle), ("Elbow (avg)", fa.elbow_angle),
                        ]
                        for name, val in ang_rows:
                            st.markdown(f"- **{name}:** {val:.1f}°")
            else:
                st.warning("No person detected.")

# ══════════════════════════════════════════════════════════
# 18. VIDEO UPLOAD
# ══════════════════════════════════════════════════════════
def render_video():
    st.markdown('<div class="section-title">Video Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a video for temporal activity recognition and fall detection.</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read()); tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []; frame_count = 0
        vcol1, vcol2 = st.columns([0.68, 0.32], gap="large")
        with vcol1:
            st_frame = st.empty()
            progress = st.progress(0)
        with vcol2:
            res_ph = st.empty()
        temporal = TemporalAnalyzer()
        reset_fall_sensor()
        last_display = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            vid_ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if not vid_ts_ms or vid_ts_ms <= 0:  # some codecs don't report this reliably
                vid_ts_ms = (frame_count / max(1.0, fps)) * 1000.0
            landmarks, pose_lm, all_poses = extract_landmarks_from_frame(frame, timestamp_ms=vid_ts_ms)
            if landmarks is not None:
                activity, confidence, fall_prob, fa = temporal.update(landmarks, frame_count / max(1.0, fps))
                activity = _simple_activity(activity)
                display = annotate_frame(frame.copy(), all_poses, activity, confidence, fall_prob, fa, fps, h, w)
                last_display = display
                results.append({"frame": frame_count, "time": f"{frame_count/fps:.1f}s",
                                "activity": activity, "confidence": f"{confidence:.0%}",
                                "fall_prob": f"{fall_prob:.0%}"})
            else:
                temporal.reset()
                reset_fall_sensor()
                display = annotate_frame(frame.copy(), None, "No Person", 0.0, 0.0, None, fps, h, w)
                last_display = display
            if frame_count % max(1, int(fps // 3)) == 0:
                with vcol1:
                    st_frame.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
            if total_frames > 0:
                progress.progress(min(frame_count / total_frames, 1.0))
            frame_count += 1
        cap.release(); progress.progress(1.0)
        try: os.unlink(tfile.name)
        except Exception: pass

        if results:
            df = pd.DataFrame(results)
            falls = df[df["activity"] == "Fall"]
            st.session_state.total_predictions += len(df)
            st.session_state.fall_count += len(falls)
            st.session_state.normal_count += len(df) - len(falls)
            for _, r in df.iterrows():
                st.session_state.prediction_history.append(
                    {"time": r["time"], "activity": r["activity"], "confidence": r["confidence"]})
            dominant = df["activity"].value_counts().idxmax()
            mean_conf = df["confidence"].str.replace("%", "").astype(float).mean()
            vc = df["activity"].value_counts().to_dict()
            dist = ", ".join(f"{a} {c}" for a, c in vc.items())
            res_ph.markdown(f"""
            <div class="panel">
              <div class="panel-title">Analysis Result</div>
              <div class="det-row"><span class="lbl">Video</span><span class="val">{uploaded.name}</span></div>
              <div class="det-row"><span class="lbl">Prediction</span><span class="val">{"FALL" if len(falls) else "No Fall"}</span></div>
              <div class="det-row"><span class="lbl">Confidence</span><span class="val">{mean_conf:.0f}%</span></div>
              <div class="det-row"><span class="lbl">Activity</span><span class="val">{dominant}</span></div>
              <div class="det-row"><span class="lbl">Frames</span><span class="val">{len(df)}</span></div>
            </div>
            """, unsafe_allow_html=True)
            if len(falls) > 0:
                st.markdown(_emergency_html(True, None, f"{len(falls)} fall frame(s)"), unsafe_allow_html=True)
            else:
                st.markdown(_emergency_html(False), unsafe_allow_html=True)
            with st.expander("Video Results"):
                st.dataframe(df, width="stretch", hide_index=True)
                st.markdown("### Activity Distribution")
                for act, cnt in vc.items():
                    st.markdown(f"- **{act}:** {cnt} frames")
        else:
            st.warning("No poses detected.")
            st.markdown(_emergency_html(False), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 19. ANALYTICS
# ══════════════════════════════════════════════════════════
def render_analytics():
    st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Live-session statistics from this monitoring dashboard.</div>',
                unsafe_allow_html=True)

    range_col, export_col = st.columns([3, 1], vertical_alignment="center")
    with range_col:
        time_range = st.segmented_control("Time Range", ["Today", "7 Days", "30 Days", "All"],
                                          key="analytics_range", label_visibility="collapsed")
    with export_col:
        if st.session_state.prediction_history:
            df_export = pd.DataFrame(st.session_state.prediction_history)
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV", csv, "analytics_export.csv", "text/csv",
                               width="stretch", key="export_csv_btn")

    _kpi_static_row()

    if st.session_state.total_predictions == 0:
        st.info("No prediction data yet. Run Live Monitoring, upload an image/video, or record & train to populate analytics.")
        return

    import plotly.express as px
    hist_df = pd.DataFrame(st.session_state.prediction_history)

    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        st.markdown('<div class="panel-title">Activity Distribution</div>', unsafe_allow_html=True)
        if len(hist_df) > 0 and "activity" in hist_df:
            counts = hist_df["activity"].value_counts().reset_index()
            counts.columns = ["Activity", "Count"]
            fig = px.pie(counts, names="Activity", values="Count",
                         color_discrete_sequence=["#2563EB", "#3B82F6", "#60A5FA", "#93C5FD",
                                                  "#EF4444", "#F87171", "#FCA5A5", "#22C55E",
                                                  "#4ADE80", "#86EFAC", "#F59E0B", "#FBBF24",
                                                  "#A78BFA", "#C4B5FD"])
            fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#F8FAFC", showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font_size=10))
            st.plotly_chart(fig, width="stretch", key="an_activity_dist")
    with col_ch2:
        st.markdown('<div class="panel-title">Fall / Normal Timeline</div>', unsafe_allow_html=True)
        hist = st.session_state.prediction_history[-80:]
        df_hist = pd.DataFrame(hist)
        fall_flags = [1 if a in ("Falling", "Fall") else 0 for a in df_hist["activity"]]
        fig2 = px.bar(x=list(range(len(fall_flags))), y=fall_flags,
                      labels={"x": "Detection #", "y": "Fall (1) / Normal (0)"},
                      color=fall_flags, color_continuous_scale=["#2563EB", "#EF4444"])
        fig2.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#F8FAFC", coloraxis_showscale=False)
        fig2.update_xaxes(gridcolor="rgba(30,53,83,0.5)")
        fig2.update_yaxes(gridcolor="rgba(30,53,83,0.5)")
        st.plotly_chart(fig2, width="stretch", key="an_fall_timeline")

    col_risk, col_freq = st.columns(2)
    with col_risk:
        st.markdown('<div class="panel-title">Confidence Distribution</div>', unsafe_allow_html=True)
        if "confidence" in hist_df:
            conf_vals = hist_df["confidence"].astype(str).str.replace("%", "").astype(float)
            fig3 = px.histogram(x=conf_vals, nbins=20, labels={"x": "Confidence %"},
                                color_discrete_sequence=["#3B82F6"])
            fig3.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#F8FAFC", bargap=0.05)
            fig3.update_xaxes(gridcolor="rgba(30,53,83,0.5)")
            fig3.update_yaxes(gridcolor="rgba(30,53,83,0.5)")
            st.plotly_chart(fig3, width="stretch", key="an_conf_hist")
    with col_freq:
        st.markdown('<div class="panel-title">Activity Frequency</div>', unsafe_allow_html=True)
        if len(hist_df) > 0 and "activity" in hist_df:
            freq = hist_df["activity"].value_counts().reset_index()
            freq.columns = ["Activity", "Count"]
            fig4 = px.bar(freq, x="Count", y="Activity", orientation="h",
                          color="Count", color_continuous_scale=["#1E3553", "#2563EB"])
            fig4.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#F8FAFC", showlegend=False, coloraxis_showscale=False)
            fig4.update_xaxes(gridcolor="rgba(30,53,83,0.5)")
            fig4.update_yaxes(gridcolor="rgba(30,53,83,0.5)")
            st.plotly_chart(fig4, width="stretch", key="an_freq_bar")

    st.markdown('<div class="section-title">Recent Activity</div>', unsafe_allow_html=True)
    if st.session_state.prediction_history:
        st.dataframe(pd.DataFrame(st.session_state.prediction_history[-30:][::-1]), width="stretch", hide_index=True)
    else:
        st.info("No predictions yet.")

# ══════════════════════════════════════════════════════════
# 20. MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
def _parse_cls_report(path):
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        return None
    rows = []
    overall = {}
    lines = txt.splitlines()
    i = 0
    while i < len(lines) and "precision" not in lines[i]:
        i += 1
    i += 1
    for ln in lines[i:]:
        parts = ln.split()
        if not parts:
            continue
        if parts[0] == "accuracy":
            try:
                overall["accuracy"] = float(parts[1]); overall["support"] = int(parts[2])
            except Exception:
                pass
            continue
        if parts[0] in ("macro", "weighted"):
            try:
                overall[parts[0] + "_precision"] = float(parts[1])
                overall[parts[0] + "_recall"] = float(parts[2])
                overall[parts[0] + "_f1"] = float(parts[3])
            except Exception:
                pass
            continue
        try:
            pv = float(parts[-4]); rv = float(parts[-3]); fv = float(parts[-2]); sv = int(parts[-1])
            rows.append({"Activity": " ".join(parts[:-4]), "Precision": pv, "Recall": rv, "F1 Score": fv, "Support": sv})
        except Exception:
            continue
    return {"rows": rows, "overall": overall}

def _parse_seq_report(path):
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        return None
    out = {}
    for block in txt.split("---"):
        name = None
        if "LSTM" in block:
            name = "LSTM (sequence)"
        elif "Frame-level" in block:
            name = "Frame model"
        if not name:
            continue
        vals = {}
        for m in re.finditer(r"(\w+):\s*([\d.]+)", block):
            try:
                vals[m.group(1)] = float(m.group(2))
            except Exception:
                pass
        out[name] = vals
    return out

def _load_json_safe(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default or {}

def render_performance():
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Evaluation results from the trained activity model, binary fall detector, and LSTM on held-out test data.</div>',
                unsafe_allow_html=True)

    # ── Load data ──────────────────────────────────────────
    bin_m = (_binary_bundle.get("metrics") or {}) if _binary_bundle else {}
    cls_path = config.SCREENSHOT_DIR / "classification_report.txt"
    parsed = _parse_cls_report(cls_path)
    base = _load_json_safe(config.SCREENSHOT_DIR / "baseline_experiments.json")
    rob = _load_json_safe(config.SCREENSHOT_DIR / "robustness_eval.json")

    def _na(val, fmt="{:.1%}"):
        if val is None or val == 0:
            return ("N/A", True)
        return (fmt.format(val), False)

    # ── 1. MODEL INFORMATION ───────────────────────────────
    st.markdown('<div class="perf-section"><div class="perf-section-title">Model Information</div></div>', unsafe_allow_html=True)
    cols = st.columns(5)
    m_info = [
        ("Frame Model", _ml_model_name if _ml_model is not None else "Not Loaded", False),
        ("Features", str(len(FEATURE_NAMES)), False),
        ("Activities", str(len(_ml_activities)), False),
        ("Fall Threshold", f"{_ml_fall_threshold:.2f}" if _ml_fall_threshold else "N/A", _ml_fall_threshold is None),
        ("Temporal Window", f"{config.FEATURE_WINDOW} frames", False),
    ]
    for col, (lbl, val, is_na) in zip(cols, m_info):
        with col:
            cls = "perf-card na" if is_na else "perf-card"
            st.markdown(f'<div class="{cls}"><div class="pc-label">{lbl}</div>'
                        f'<div class="pc-value{" na" if is_na else ""}">{val}</div></div>',
                        unsafe_allow_html=True)

    # ── 2. DETECTION PERFORMANCE (Binary Fall Detector) ────
    if bin_m:
        st.markdown('<div class="perf-section"><div class="perf-section-title">Detection Performance</div></div>', unsafe_allow_html=True)
        cols = st.columns(5)
        det_info = [
            ("Binary Detector", bin_m.get("model_type", "N/A"), False, False),
            ("Threshold", f"{bin_m.get('threshold', 0):.2f}", False, False),
            ("TP / FP", f"{bin_m.get('tp', 0)} / {bin_m.get('fp', 0)}", False, False),
            ("TN / FN", f"{bin_m.get('tn', 0)} / {bin_m.get('fn', 0)}", False, False),
            ("False-Alarm Rate", *(_na(bin_m.get("false_alarm_rate"), "{:.1%}")), False),
        ]
        for col, (lbl, val, is_na, _) in zip(cols, det_info):
            with col:
                cls = "perf-card na" if is_na else "perf-card"
                st.markdown(f'<div class="{cls}"><div class="pc-label">{lbl}</div>'
                            f'<div class="pc-value{" na" if is_na else ""}">{val}</div></div>',
                            unsafe_allow_html=True)

    # ── 3. MODEL PERFORMANCE (Frame model test metrics) ────
    if parsed and parsed.get("overall"):
        o = parsed["overall"]
        st.markdown('<div class="perf-section"><div class="perf-section-title">Model Performance</div></div>', unsafe_allow_html=True)
        cols = st.columns(4)
        perf_metrics = [
            ("Accuracy", o.get("accuracy"), "{:.1%}"),
            ("Precision (weighted)", o.get("weighted_precision"), "{:.1%}"),
            ("Recall (weighted)", o.get("weighted_recall"), "{:.1%}"),
            ("F1 Score (weighted)", o.get("weighted_f1"), "{:.1%}"),
        ]
        for col, (lbl, val, fmt) in zip(cols, perf_metrics):
            with col:
                disp, is_na = _na(val, fmt)
                cls = "perf-card success" if (not is_na and val and val >= 0.85) else ("perf-card na" if is_na else "perf-card")
                st.markdown(f'<div class="{cls}"><div class="pc-label">{lbl}</div>'
                            f'<div class="pc-value{" na" if is_na else ""}">{disp}</div></div>',
                            unsafe_allow_html=True)

        # ── 4. FALL PERFORMANCE ──────────────────────────────
        st.markdown('<div class="perf-section"><div class="perf-section-title">Fall Performance</div></div>', unsafe_allow_html=True)
        cols = st.columns(4)
        fall_m = [
            ("Fall Precision", bin_m.get("test_precision"), "{:.1%}"),
            ("Fall Recall", bin_m.get("test_recall"), "{:.1%}"),
            ("Fall F1", bin_m.get("test_f1"), "{:.1%}"),
            ("Fall F2", bin_m.get("test_f2"), "{:.1%}"),
        ]
        for col, (lbl, val, fmt) in zip(cols, fall_m):
            with col:
                disp, is_na = _na(val, fmt)
                cls = "perf-card danger" if (not is_na and val and val < 0.75) else ("perf-card success" if (not is_na and val and val >= 0.85) else ("perf-card na" if is_na else "perf-card"))
                st.markdown(f'<div class="{cls}"><div class="pc-label">{lbl}</div>'
                            f'<div class="pc-value{" na" if is_na else ""}">{disp}</div></div>',
                            unsafe_allow_html=True)

    # ── 5. BEFORE VS AFTER ─────────────────────────────────
    if base.get("before", {}).get("test"):
        before = base["before"]["test"]
        st.markdown('<div class="perf-section"><div class="perf-section-title">Improvement — Before vs After</div></div>', unsafe_allow_html=True)
        if parsed and parsed.get("overall"):
            o = parsed["overall"]
            pct = lambda v: f"{v*100:.1f}%" if v is not None else "N/A"
            diff = lambda b, a: f"+{(a-b)*100:.1f}pp" if a is not None and b is not None and a > b else (
                f"{(a-b)*100:.1f}pp" if a is not None and b is not None else "—")
            rows = [
                ("Accuracy", pct(before.get("acc")), pct(o.get("accuracy")), before.get("acc"), o.get("accuracy")),
                ("Weighted F1", pct(before.get("f1")), pct(o.get("weighted_f1")), before.get("f1"), o.get("weighted_f1")),
                ("Fall Precision", pct(before.get("fall_precision")), pct(bin_m.get("test_precision")), before.get("fall_precision"), bin_m.get("test_precision")),
                ("Fall Recall", pct(before.get("fall_recall")), pct(bin_m.get("test_recall")), before.get("fall_recall"), bin_m.get("test_recall")),
                ("Fall F1", pct(before.get("fall_f1")), pct(bin_m.get("test_f1")), before.get("fall_f1"), bin_m.get("test_f1")),
                ("Fall F2", pct(before.get("fall_f2")), pct(bin_m.get("test_f2")), before.get("fall_f2"), bin_m.get("test_f2")),
            ]
            table_html = '<table class="perf-table"><thead><tr><th style="width:30%;">Metric</th><th style="width:25%;">Before</th><th style="width:25%;">After</th><th style="width:20%;">Change</th></tr></thead><tbody>'
            for name, bv, av, bvn, avn in rows:
                cls = ""
                if avn is not None and bvn is not None:
                    cls = "improved" if avn > bvn else ("degraded" if avn < bvn else "")
                table_html += f'<tr><td class="metric-name">{name}</td><td>{bv}</td><td>{av}</td><td class="{cls}">{diff(bvn, avn)}</td></tr>'
            table_html += '</tbody></table>'
            st.markdown(f'<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);overflow:hidden;">{table_html}</div>',
                        unsafe_allow_html=True)
            st.caption("Before = original Random Forest. After = current model (oversampling, custom class weights, HistGB selection, validation-tuned threshold). Both on the same held-out test set.")

    # ── 6. MODEL COMPARISON ────────────────────────────────
    if base.get("comparison"):
        st.markdown('<div class="perf-section"><div class="perf-section-title">Model Comparison</div></div>', unsafe_allow_html=True)
        winner_name = _ml_model_name if _ml_model_name else ""
        cards_html = '<div class="model-compare">'
        for name, v in base["comparison"].items():
            is_best = name.lower().startswith(winner_name.lower().split("(")[0].strip().lower()) if winner_name else False
            cards_html += f'<div class="model-card{" best" if is_best else ""}"><div class="mc-name">{name}</div>'
            for lbl, key, fmt in [("Val F1", "val_f1", "{:.1%}"), ("Val Fall F2", "val_fall_f2", "{:.1%}"), ("Inference", "inference_ms_per_sample", "{:.3f} ms")]:
                cards_html += f'<div class="mc-metric"><span class="lbl">{lbl}</span><span class="val">{fmt.format(v.get(key, 0))}</span></div>'
            cards_html += '</div>'
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)
        st.caption("Candidate models trained on the same group-split train set; metrics on validation split. Inference time on 2,000 held-out test samples.")

    # ── 7. PER-CLASS RESULTS ───────────────────────────────
    if parsed and parsed.get("rows"):
        st.markdown('<div class="perf-section"><div class="perf-section-title">Per-Class Results</div></div>', unsafe_allow_html=True)
        table_html = '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);overflow:hidden;"><table class="perf-table"><thead><tr>'
        for h in ["Activity", "Precision", "Recall", "F1 Score", "Support"]:
            table_html += f'<th>{h}</th>'
        table_html += '</tr></thead><tbody>'
        for r in parsed["rows"]:
            table_html += f'<tr><td class="metric-name">{r["Activity"]}</td><td>{r["Precision"]:.2f}</td><td>{r["Recall"]:.2f}</td><td>{r["F1 Score"]:.2f}</td><td>{r["Support"]}</td></tr>'
        table_html += '</tbody></table></div>'
        st.markdown(table_html, unsafe_allow_html=True)

    # ── 8. SEQUENCE MODEL ──────────────────────────────────
    seq_path = config.SCREENSHOT_DIR / "sequence_model_report.txt"
    seq = _parse_seq_report(seq_path)
    if seq:
        st.markdown('<div class="perf-section"><div class="perf-section-title">Fall Detection — Temporal Confirmation</div></div>', unsafe_allow_html=True)
        for name, vals in seq.items():
            st.markdown(f'<div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:6px;">{name}</div>', unsafe_allow_html=True)
            cols = st.columns(5)
            for col, (lbl, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("F2", "f2")]):
                with col:
                    disp, is_na = _na(vals.get(key), "{:.1%}")
                    st.markdown(f'<div class="perf-card{" na" if is_na else ""}"><div class="pc-label">{lbl}</div>'
                                f'<div class="pc-value{" na" if is_na else ""}">{disp}</div></div>',
                                unsafe_allow_html=True)
            tp = int(vals.get("tp", 0)); fp = int(vals.get("fp", 0)); tn = int(vals.get("tn", 0)); fn = int(vals.get("fn", 0))
            st.markdown(f'<div class="perf-card" style="min-height:auto;padding:8px 14px;"><div class="pc-label">Confusion</div>'
                        f'<div class="pc-value" style="font-size:14px;">TP {tp} / FP {fp} · TN {tn} / FN {fn}</div></div>',
                        unsafe_allow_html=True)
        st.caption("LSTM consumes 20-frame windows and confirms a fall before the siren fires.")

    # ── 9. CROSS-CAMERA ROBUSTNESS ─────────────────────────
    if rob:
        st.markdown('<div class="perf-section"><div class="perf-section-title">Cross-Camera Robustness</div></div>', unsafe_allow_html=True)
        table_html = '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);overflow:hidden;"><table class="perf-table"><thead><tr>'
        for h in ["Train Source", "Test Source", "Precision", "Recall", "F1", "F2", "TP/FP · TN/FN"]:
            table_html += f'<th>{h}</th>'
        table_html += '</tr></thead><tbody>'
        for k, v in rob.items():
            tr, te = k.split("->")
            table_html += f'<tr><td>{tr.upper()}</td><td>{te.upper()}</td><td>{v.get("precision",0)*100:.1f}%</td><td>{v.get("recall",0)*100:.1f}%</td><td>{v.get("f1",0)*100:.1f}%</td><td>{v.get("f2",0)*100:.1f}%</td><td>{v.get("tp",0)}/{v.get("fp",0)} · {v.get("tn",0)}/{v.get("fn",0)}</td></tr>'
        table_html += '</tbody></table></div>'
        st.markdown(table_html, unsafe_allow_html=True)
        st.caption("Fall detector trained on one source, tested on another. IMViA→UR-Fall generalizes; HMDB51 is a different domain.")

    # ── 10. EVALUATION IMAGES ──────────────────────────────
    imgs = [
        ("confusion_matrix_test.png", "Confusion Matrix — Activity Model"),
        ("confusion_matrix_fall_not_fall.png", "Fall / Not-Fall Confusion Matrix"),
        ("feature_importance.png", "Feature Importance — Top 15"),
        ("fall_feature_importance.png", "Fall Detector Feature Importance"),
    ]
    available = [(p, t) for p, t in imgs if (config.SCREENSHOT_DIR / p).exists()]
    if available:
        st.markdown('<div class="perf-section"><div class="perf-section-title">Evaluation Outputs</div></div>', unsafe_allow_html=True)
        grid_html = '<div class="perf-image-grid">'
        for p, t in available:
            grid_html += f'<div class="perf-image-card"><div class="pic-title">{t}</div><img src="data:image/png;base64,{_img_to_base64(config.SCREENSHOT_DIR / p)}" /></div>'
        grid_html += '</div>'
        st.markdown(grid_html, unsafe_allow_html=True)

def _img_to_base64(path):
    import base64
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except Exception:
        return ""

# ══════════════════════════════════════════════════════════
# 21. RECORD & TRAIN
# ══════════════════════════════════════════════════════════
def render_train():
    sess = _load_sessions()
    done = [
        sum(sess["record_sessions"].values()) > 0,
        any(1 for d in _recorded_counts().values() if d.get("frames", 0) or d.get("videos", 0)),
        (config.FEATURES_DIR / "features.npz").exists(),
        config.FRAME_MODEL_PATH.exists(),
        (config.SCREENSHOT_DIR / "classification_report.txt").exists(),
        config.SEQUENCE_MODEL_PATH.exists(),
    ]
    st.markdown('<div class="section-title">Record &amp; Train</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Record your own samples, then rebuild the full model from all data (existing datasets + your recordings, frames and videos).</div>',
                unsafe_allow_html=True)
    st.markdown(_stepper_html(done), unsafe_allow_html=True)

    cam_col, ctl_col = st.columns([2, 1], gap="large")
    with cam_col:
        cam_ph = st.empty()
        cam_ph.markdown(
            '<div class="camera-empty" style="min-height:320px;">'
            '<div class="big">RECORDING STATION</div>'
            '<div class="sub">Webcam preview appears here. Pick an activity and press Record.</div></div>',
            unsafe_allow_html=True)
    with ctl_col:
        rec_activity = st.selectbox("Activity to record", RECORD_ACTIVITIES, key="rec_activity")
        rec_duration = st.slider("Recording length (seconds)", 3, 20, 15, key="rec_duration")
        start_rec = st.button("Record", type="primary", width="stretch", key="rec_start_btn")
        st.caption("Webcam opens and recording starts immediately. Auto-stops after the set time — stay fully in frame.")

    if start_rec:
        out_dir = RECORDED_DIR / rec_activity
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            st.error("Could not open webcam. Please check camera permissions.")
        else:
            progress = st.progress(0)
            stat = st.empty()
            start_t = time.time()
            frame_i = 0
            saved = 0
            video_path = out_dir / f"{rec_activity}_{time.strftime('%Y%m%d_%H%M%S')}.avi"
            vw = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (1280, 720))
            while cap.isOpened():
                if time.time() - start_t >= rec_duration:
                    break
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue
                frame_i += 1
                if vw.isOpened():
                    vw.write(frame)
                landmarks, pose_lm, all_poses = extract_landmarks_from_frame(frame)
                if landmarks is not None and frame_i % 2 == 0:
                    fname = out_dir / f"frame_{int(time.time()*1000)}_{saved}.jpg"
                    cv2.imwrite(str(fname), frame)
                    saved += 1
                disp = frame.copy()
                if pose_lm is not None:
                    hh, ww = disp.shape[:2]
                    disp = draw_body_lines(disp, all_poses, hh, ww)
                cam_ph.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                progress.progress(min(1.0, (time.time() - start_t) / rec_duration))
                stat.info(f"Recording **{rec_activity}**: {saved} frames saved (pose detected) — stay in frame")
                time.sleep(0.02)
            cap.release()
            if vw.isOpened():
                vw.release()
            progress.empty()
            sess = _load_sessions()
            sess["record_sessions"][rec_activity] = sess["record_sessions"].get(rec_activity, 0) + 1
            sess["record_frames"][rec_activity] = sess["record_frames"].get(rec_activity, 0) + saved
            _save_sessions(sess)
            _ok_box(f"Saved **{saved}** frames + video `{video_path.name}` to `{out_dir}` — "
                    f"**both will be used in the next training run**.")
            st.rerun()

    with st.expander("Dataset Session & Training Log"):
        recd = _recorded_counts()
        acts = sorted(set(list(sess["record_sessions"].keys()) + list(recd.keys())))
        if acts:
            for a in acts:
                sess_n = sess["record_sessions"].get(a, 0)
                d = recd.get(a, {})
                fr, vd = d.get("frames", 0), d.get("videos", 0)
                st.markdown(f"- **{a}:** {sess_n} session{'s' if sess_n != 1 else ''} &middot; "
                            f"{fr} frames + {vd} video{'s' if vd != 1 else ''}"
                            f"&nbsp;<span style='color:#3B82F6;font-size:0.8rem;'>(saved to dataset/recorded/{a}/)</span>")
        else:
            st.info("No recording sessions yet. Record a few seconds per activity.")
        st.markdown(f"**Total recording sessions: {sum(sess['record_sessions'].values())}**")
        st.markdown(f"**Total recorded frames: {sum(d.get('frames', 0) for d in recd.values())}**")
        st.markdown(f"**Total recorded videos: {sum(d.get('videos', 0) for d in recd.values())}**")
        st.markdown(f"**Training runs: {sess['train_runs']}**")
        if sess["train_history"]:
            last = sess["train_history"][-1]
            acc = f"{last['test_acc'] * 100:.1f}%" if last.get("test_acc") is not None else "n/a"
            st.markdown(f"Last training: **{last['ts']}** &middot; test accuracy **{acc}**")

    st.markdown("---")
    st.markdown('<div class="section-title">Train the Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Rebuilds every model from ALL data (existing datasets + your recordings, frames and videos) and saves to '
                '<b>models/fall_model.pkl</b> (frame) + <b>models/fall_sequence_model.keras</b> (LSTM).</div>',
                unsafe_allow_html=True)
    if st.button("Train Model", type="primary", width="stretch", key="train_model_btn"):
        try:
            with st.spinner("Extracting features from all data (incl. your recordings)..."):
                r_extract = subprocess.run([sys.executable, "extract_feature.py"], cwd=str(PROJECT_DIR), check=True,
                                           capture_output=True, text=True)
            with st.spinner("Training Random Forest model..."):
                r_train = subprocess.run([sys.executable, "train_model.py"], cwd=str(PROJECT_DIR), check=True,
                                         capture_output=True, text=True)
            with st.spinner("Building fall sequences (incl. your recorded videos)..."):
                subprocess.run([sys.executable, "build_sequences.py"], cwd=str(PROJECT_DIR), check=True,
                               capture_output=True, text=True)
            with st.spinner("Training LSTM fall detector..."):
                subprocess.run([sys.executable, "train_sequence_model.py", "lstm"], cwd=str(PROJECT_DIR), check=True,
                               capture_output=True, text=True)
            _ok_box("Model trained and saved to <b>models/fall_model.pkl</b> (frame) + <b>models/fall_sequence_model.keras</b> (LSTM)")
            accs = re.findall(r"Accuracy:\s+([\d.]+)", r_train.stdout or "")
            test_acc = float(accs[-1]) if accs else None
            sess = _load_sessions()
            sess["train_runs"] = sess.get("train_runs", 0) + 1
            sess.setdefault("train_history", []).append({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_acc": test_acc,
            })
            _save_sessions(sess)
            st.rerun()
        except subprocess.CalledProcessError as e:
            st.error(f"Training failed:\n{e.stderr[-2000:] if e.stderr else e}")

# ══════════════════════════════════════════════════════════
# 22. TOP-LEVEL FLOW
# ══════════════════════════════════════════════════════════
reload_models()   # pick up any models retrained via Record & Train
view = _render_header()
st.divider()

if view == "live":
    render_live()
elif view == "image":
    render_image()
elif view == "video":
    render_video()
elif view == "analytics":
    render_analytics()
elif view == "performance":
    render_performance()
elif view == "train":
    render_train()

if view != "live":
    _close_webcam()

st.markdown("---")
st.markdown("<footer><strong>AI CARE</strong> — Human Activity &amp; Fall Detection System | MediaPipe Pose + Temporal Features + Random Forest | Streamlit</footer>",
            unsafe_allow_html=True)
