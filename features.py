"""Shared pose feature extraction used by BOTH training (extract_feature.py)
and live inference (app.py) so the model input layout can never drift.

Static features describe the current pose (angles + body-size-normalized
lengths). Temporal features describe motion over the last ~20 frames
(velocities, accelerations, displacements) - the signal that distinguishes
a real fall from lying/bending.

Coordinates follow MediaPipe: x,y in [0,1], y points DOWN.
Angles are in radians.
"""
import numpy as np
from collections import deque

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

# Lengths are normalized by shoulder-hip distance so they stay valid at any
# camera distance / zoom. Raw x/z coordinates are deliberately NOT used.
STATIC_FEATURE_NAMES = [
    "shoulder_hip_dist",
    "torso_inclination",
    "left_knee_angle", "right_knee_angle",
    "left_elbow_angle", "right_elbow_angle",
    "left_hip_angle", "right_hip_angle",
    "shoulder_width_ratio", "hip_width_ratio",
    "left_shoulder_hip_knee", "right_shoulder_hip_knee",
    "left_ankle_y_ratio", "right_ankle_y_ratio",
    "left_wrist_y_ratio", "right_wrist_y_ratio",
    "head_to_hip_ratio",
    "left_knee_ankle", "right_knee_ankle",
    "torso_leg_angle",
    "shoulder_hip_knee_left", "shoulder_hip_knee_right",
]

TEMPORAL_FEATURE_NAMES = [
    # --- instantaneous downward motion (normalized by body scale) ---
    "hip_vel_y", "head_vel_y", "ankle_vel_y",
    "body_accel_mag",
    # --- window displacement ---
    "vert_displacement", "horiz_displacement",
    "cadence", "feet_grounded",
    # --- body drop / movement (fall signature: body goes down fast) ---
    "hip_drop_speed", "shoulder_drop_speed", "head_drop_speed",
    "body_center_disp",
    # --- posture transition (standing -> horizontal) ---
    "time_to_horizontal", "posture_change_rate", "standing_to_ground",
    # --- orientation dynamics ---
    "torso_angle_change", "orientation_velocity", "rotation_accel",
    # --- post-fall inactivity (fall is followed by stillness) ---
    "post_fall_motion", "stationary_duration", "movement_after_event",
    # --- speed discrimination (Walking vs Running, Getting Up vs Sitting Down) ---
    "hip_speed_mag", "vert_velocity_dir", "body_speed_mag",
]

FEATURE_NAMES = STATIC_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES
N_STATIC = len(STATIC_FEATURE_NAMES)
N_TEMPORAL = len(TEMPORAL_FEATURE_NAMES)


def _get_pt(landmarks, name):
    idx = LANDMARK_INDICES[name]
    return np.array([landmarks[idx * 4], landmarks[idx * 4 + 1], landmarks[idx * 4 + 2]])


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    dot = np.dot(ba, bc)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return np.arccos(np.clip(dot / norm, -1.0, 1.0))


def extract_features(landmarks, temporal=None):
    """Full feature vector = static(N) + temporal(M).

    landmarks: 33x4 array [x,y,z,visibility]. temporal: optional M-vector
    from a TemporalWindowExtractor; zeros are used when no history exists.
    """
    ls = _get_pt(landmarks, "left_shoulder"); rs = _get_pt(landmarks, "right_shoulder")
    lh = _get_pt(landmarks, "left_hip"); rh = _get_pt(landmarks, "right_hip")
    le = _get_pt(landmarks, "left_elbow"); re = _get_pt(landmarks, "right_elbow")
    lk = _get_pt(landmarks, "left_knee"); rk = _get_pt(landmarks, "right_knee")
    la = _get_pt(landmarks, "left_ankle"); ra = _get_pt(landmarks, "right_ankle")
    lw = _get_pt(landmarks, "left_wrist"); rw = _get_pt(landmarks, "right_wrist")
    nose = _get_pt(landmarks, "nose")
    lf = _get_pt(landmarks, "left_foot_index"); rf = _get_pt(landmarks, "right_foot_index")

    sm = (ls + rs) / 2
    hm = (lh + rh) / 2
    km = (lk + rk) / 2
    shd = np.linalg.norm(sm - hm) + 1e-8
    nose_hip = np.linalg.norm(nose - hm)

    feats = {
        "shoulder_hip_dist": np.linalg.norm(sm - hm),
        "torso_inclination": _angle(sm, hm, hm + np.array([0.0, 1.0, 0.0])),
        "left_knee_angle": _angle(lh, lk, la),
        "right_knee_angle": _angle(rh, rk, ra),
        "left_elbow_angle": _angle(ls, le, lw),
        "right_elbow_angle": _angle(rs, re, rw),
        "left_hip_angle": _angle(ls, lh, lk),
        "right_hip_angle": _angle(rs, rh, rk),
        "shoulder_width_ratio": np.linalg.norm(ls - rs) / shd,
        "hip_width_ratio": np.linalg.norm(lh - rh) / shd,
        "left_shoulder_hip_knee": _angle(ls, lh, lk),
        "right_shoulder_hip_knee": _angle(rs, rh, rk),
        "left_ankle_y_ratio": (la[1] - hm[1]) / shd,
        "right_ankle_y_ratio": (ra[1] - hm[1]) / shd,
        "left_wrist_y_ratio": (lw[1] - sm[1]) / shd,
        "right_wrist_y_ratio": (rw[1] - sm[1]) / shd,
        "head_to_hip_ratio": nose_hip / shd,
        "left_knee_ankle": _angle(lk, la, lf),
        "right_knee_ankle": _angle(rk, ra, rf),
        "torso_leg_angle": _angle(sm, hm, km),
        "shoulder_hip_knee_left": _angle(ls, lh, lk),
        "shoulder_hip_knee_right": _angle(rs, rh, rk),
    }
    static = np.array([feats[k] for k in STATIC_FEATURE_NAMES], dtype=np.float32)
    if temporal is None:
        temporal = np.zeros(N_TEMPORAL, dtype=np.float32)
    return np.concatenate([static, np.asarray(temporal, dtype=np.float32)])


def _quantities(lm):
    """Scalars needed to compute motion features for one frame."""
    ls = _get_pt(lm, "left_shoulder"); rs = _get_pt(lm, "right_shoulder")
    lh = _get_pt(lm, "left_hip"); rh = _get_pt(lm, "right_hip")
    lk = _get_pt(lm, "left_knee"); rk = _get_pt(lm, "right_knee")
    la = _get_pt(lm, "left_ankle"); ra = _get_pt(lm, "right_ankle")
    nose = _get_pt(lm, "nose")
    sm = (ls + rs) / 2
    hm = (lh + rh) / 2
    return {
        "hip": (hm[0], hm[1]),
        "shoulder": (sm[0], sm[1]),
        "head": (nose[0], nose[1]),
        "body_center": ((sm[0] + hm[0]) / 2, (sm[1] + hm[1]) / 2),
        "ankle": ((la[0] + ra[0]) / 2, (la[1] + ra[1]) / 2),
        "ankle_l_y": la[1], "ankle_r_y": ra[1],
        "torso": _angle(sm, hm, hm + np.array([0.0, 1.0, 0.0])),
        "hip_angle": _angle(ls, lh, lk),
        "shd": float(np.linalg.norm(sm - hm)),   # body scale for normalization
    }


class TemporalWindowExtractor:
    """Buffers recent frames so a labeled frame also gets motion features.
    Call push(landmarks, t) every frame; call temporal_features() to get the
    current motion vector. reset() at the start of each clip/sequence.
    """

    def __init__(self, window=30):
        self.window = window
        self.times = deque(maxlen=window)
        self.qs = deque(maxlen=window)

    def reset(self):
        self.times.clear()
        self.qs.clear()

    def push(self, lm, t):
        self.times.append(t)
        self.qs.append(_quantities(lm))

    # fall detection thresholds (radians / normalized units)
    _HORIZONTAL_RAD = 60.0 * np.pi / 180.0
    _VERTICAL_RAD = 30.0 * np.pi / 180.0
    _STILL_SPEED = 0.02      # hip speed (per body-scale per second) below this = still
    _FALL_DROP = 0.15        # hip drop (per body-scale) below this = no real transition

    def temporal_features(self):
        n = len(self.qs)
        z = np.zeros(N_TEMPORAL, dtype=np.float32)
        if n < 2:
            return z
        q = self.qs[-1]
        prev = self.qs[-2]
        dt = max(1e-4, self.times[-1] - self.times[-2])
        duration = max(1e-4, self.times[-1] - self.times[0])
        scale = max(1e-3, float(np.mean([a["shd"] for a in self.qs])))

        # --- instantaneous velocities (normalized by body scale) ---
        hip_vel = (np.array(q["hip"]) - np.array(prev["hip"])) / dt
        head_vel = (np.array(q["head"]) - np.array(prev["head"])) / dt
        ankle_vel = (np.array(q["ankle"]) - np.array(prev["ankle"])) / dt
        accel = np.zeros(2)
        if n >= 3:
            prev2 = self.qs[-3]
            v_prev = (np.array(prev["hip"]) - np.array(prev2["hip"])) / max(1e-4, self.times[-2] - self.times[-3])
            accel = (hip_vel - v_prev) / dt

        hip_y = [a["hip"][1] for a in self.qs]
        hip_x = [a["hip"][0] for a in self.qs]
        sh_y = [a["shoulder"][1] for a in self.qs]
        head_y = [a["head"][1] for a in self.qs]
        torso_s = [a["torso"] for a in self.qs]
        bc_x = [a["body_center"][0] for a in self.qs]
        bc_y = [a["body_center"][1] for a in self.qs]
        vert = float(hip_y[-1] - hip_y[0])      # +ve = hips dropped over window
        horiz = float(hip_x[-1] - hip_x[0])

        # --- cadence / foot contact (existing logic) ---
        ankle_l = [a["ankle_l_y"] for a in self.qs]
        ankle_r = [a["ankle_r_y"] for a in self.qs]
        steps = sum(
            1 for i in range(1, len(ankle_l))
            if (ankle_l[i - 1] - ankle_r[i - 1]) * (ankle_l[i] - ankle_r[i]) < 0
        )
        cadence = steps / duration
        floor_y = max(max(ankle_l), max(ankle_r))
        both_off = (q["ankle_l_y"] < floor_y - 0.05) and (q["ankle_r_y"] < floor_y - 0.05)
        feet_grounded = float(not both_off)

        # --- drop speeds: how fast each body part dropped over the window ---
        hip_drop = max(0.0, hip_y[0] - hip_y[-1]) / duration / scale
        shoulder_drop = max(0.0, sh_y[0] - sh_y[-1]) / duration / scale
        head_drop = max(0.0, head_y[0] - head_y[-1]) / duration / scale
        body_center_disp = (np.hypot(bc_x[-1] - bc_x[0], bc_y[-1] - bc_y[0]) / scale)

        # --- orientation dynamics ---
        torso_angle_change = float(torso_s[-1] - torso_s[0])
        ang_vels = []
        for i in range(1, n):
            dt_i = max(1e-4, self.times[i] - self.times[i - 1])
            ang_vels.append((torso_s[i] - torso_s[i - 1]) / dt_i)
        posture_change_rate = float(np.mean(np.abs(ang_vels)))
        orientation_velocity = float(np.max(np.abs(ang_vels)))
        rot_accel = 0.0
        if len(ang_vels) >= 2:
            rot_accel = float(np.max(np.abs(np.diff(ang_vels))))

        # --- posture transition: standing/vertical -> horizontal ---
        # Find the first moment the torso went horizontal, and when it was still vertical.
        h_idx = next((i for i, t in enumerate(torso_s) if t >= self._HORIZONTAL_RAD), None)
        time_to_horizontal = 0.0
        movement_after_event = 0.0
        if h_idx is not None:
            v_idx = next((j for j in range(h_idx - 1, -1, -1) if torso_s[j] <= self._VERTICAL_RAD), None)
            if v_idx is not None:
                time_to_horizontal = float(self.times[h_idx] - self.times[v_idx])
                hip_speeds_after = []
                for i in range(v_idx, n - 1):
                    dt_i = max(1e-4, self.times[i + 1] - self.times[i])
                    sp = np.hypot(self.qs[i + 1]["hip"][0] - self.qs[i]["hip"][0],
                                  self.qs[i + 1]["hip"][1] - self.qs[i]["hip"][1]) / dt_i / scale
                    hip_speeds_after.append(sp)
                movement_after_event = float(np.mean(hip_speeds_after)) if hip_speeds_after else 0.0
        else:
            time_to_horizontal = duration   # never seen horizontal within window

        early_vertical = min(torso_s[:max(1, n // 3)]) <= self._VERTICAL_RAD
        standing_to_ground = float(
            early_vertical and torso_s[-1] >= self._HORIZONTAL_RAD
            and (vert / scale) >= self._FALL_DROP
        )

        # --- post-fall inactivity ---
        third = max(1, n // 3)
        hip_speeds = []
        for i in range(1, n):
            dt_i = max(1e-4, self.times[i] - self.times[i - 1])
            sp = np.hypot(self.qs[i]["hip"][0] - self.qs[i - 1]["hip"][0],
                          self.qs[i]["hip"][1] - self.qs[i - 1]["hip"][1]) / dt_i / scale
            hip_speeds.append(sp)
        post_fall_motion = float(np.mean(hip_speeds[-third:])) if hip_speeds else 0.0
        still_frames = sum(1 for sp in hip_speeds if sp < self._STILL_SPEED)
        stationary_duration = float(still_frames * (duration / max(1, n - 1)))

        # --- speed discrimination (Walking vs Running, Getting Up vs Sitting Down) ---
        hip_speed_mag = float(np.hypot(hip_vel[0], hip_vel[1]))
        vert_velocity_dir = float(np.sign(hip_vel[1]))  # +1=down, -1=up
        sh_speeds = []
        for i in range(1, n):
            dt_i = max(1e-4, self.times[i] - self.times[i - 1])
            sp = np.hypot(self.qs[i]["shoulder"][0] - self.qs[i - 1]["shoulder"][0],
                          self.qs[i]["shoulder"][1] - self.qs[i - 1]["shoulder"][1]) / dt_i / scale
            sh_speeds.append(sp)
        body_speed_mag = float(np.mean(hip_speeds)) if hip_speeds else 0.0

        return np.array([
            hip_vel[1], head_vel[1], ankle_vel[1],
            float(np.hypot(*accel)),
            vert, horiz,
            cadence, feet_grounded,
            hip_drop, shoulder_drop, head_drop, body_center_disp,
            time_to_horizontal, posture_change_rate, standing_to_ground,
            torso_angle_change, orientation_velocity, rot_accel,
            post_fall_motion, stationary_duration, movement_after_event,
            hip_speed_mag, vert_velocity_dir, body_speed_mag,
        ], dtype=np.float32)
