# AI-Powered Elderly Fall Detection System using Machine Learning and Deep Learning

A real-time elderly fall detection and activity recognition system that uses MediaPipe pose estimation, hand-crafted biomechanical features, and ensemble machine learning to classify 14 human activities and detect falls from webcam, image, or video input. Deployed as an interactive Streamlit web application.

**Streamlit App:** `[PASTE YOUR STREAMLIT CLOUD LINK HERE]`

---

## Project Overview

Elderly falls are a leading cause of injury, hospitalisation, and loss of independence among older adults. Timely detection of a fall can significantly improve emergency response outcomes, yet continuous human supervision is neither practical nor scalable in most care settings.

This project implements an **AI-assisted monitoring system** that uses computer vision and machine learning to automatically detect human posture, classify activities in real time, and trigger an emergency alert when a fall is identified. The system processes webcam video using MediaPipe pose estimation to extract 33 body landmarks per frame, converts these into 46 biomechanical features (joint angles, body proportions, velocities, and motion dynamics), and classifies the activity using a trained HistGradientBoosting ensemble model.

The system is designed to **assist caregivers and healthcare monitoring workflows** — it is not a replacement for professional medical supervision. Predictions may contain errors, and human oversight remains essential.

---

## Objectives

1. Detect human posture and body movement from webcam input using pose estimation.
2. Classify 14 distinct human activities including standing, sitting, walking, running, falling, lying down, bending, squatting, jumping, climbing stairs, crawling, kneeling, crouching, and getting up.
3. Identify fall events using a multi-tier detection approach (frame-level classifier, binary fall model, and LSTM sequence model).
4. Generate emergency alerts (visual banner, siren audio, fall counter) when a confirmed fall is detected.
5. Display real-time monitoring information including activity label, confidence score, fall risk gauge, and pose skeleton overlay.
6. Evaluate the trained model using accuracy, precision, recall, and F1-score on a held-out test set.
7. Deploy the system as an interactive web application using Streamlit.

---

## System Workflow

```
Input Image / Video / Webcam Stream
        |
        v
  Frame Processing (OpenCV)
        |
        v
  Pose Estimation (MediaPipe Pose - 33 landmarks)
        |
        v
  Feature Extraction (46 biomechanical features)
        |
        v
  Activity Classification (HistGradientBoosting Ensemble - 14 classes)
        |
        v
  Fall Detection Logic (3-tier fusion + multi-frame confirmation)
        |
        v
  Prediction + Confidence Score
        |
        v
  Emergency Alert (if fall confirmed)
        |
        v
  Streamlit Dashboard (real-time display)
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.9+ | Main programming language |
| OpenCV | Image and video processing, frame capture |
| MediaPipe Pose | Real-time body pose estimation (33 landmarks) |
| Scikit-learn | HistGradientBoosting, RandomForest, model training and evaluation |
| TensorFlow / Keras | LSTM sequence model for temporal fall detection |
| Streamlit | Web application deployment and dashboard |
| streamlit-webrtc | Browser-native WebRTC camera streaming |
| NumPy | Numerical array operations, feature computation |
| Pandas | Data processing and tabular display |
| Matplotlib / Seaborn | Confusion matrices, training curves, feature importance plots |
| Plotly | Interactive analytics charts in the dashboard |
| Pillow | Image format handling |
| joblib | Model serialization |

---

## Dataset

| Property | Value |
|---|---|
| **Datasets Used** | UR Fall Dataset, IMVIA/Le2i Fall Dataset, HMDB51 Action Recognition Dataset, Custom Webcam Recordings |
| **Data Type** | RGB video frames and images |
| **Total Samples** | ~13,000+ labelled frames across all sources |
| **Activity Classes** | 14 |
| **Pose Estimator** | MediaPipe Pose Landmarker (Lite) |
| **Feature Vectors** | 46-dimensional (22 static + 24 temporal) per frame |

**Data Sources:**
- **UR Fall Dataset** — Real fall recordings with RGB frames and body height CSVs (fall onset detection)
- **IMVIA / Le2i Fall Dataset** — CCTV-style fall videos with per-frame annotations
- **HMDB51** — Real video clips for activities: Walking, Running, Jumping, Falling, Sitting, Getting Up, Climbing Stairs
- **Custom Recordings** — Webcam-captured clips of Standing, Sitting, Walking, Crouching, Kneeling
- **Synthetic** — Computer-generated poses with Gaussian noise for under-represented classes

**Preprocessing:**
- MediaPipe pose detection on every frame (33 landmarks x 4 values = 132 values per frame)
- Low-visibility landmark interpolation using last-known positions
- Moving-average smoothing over 5 frames
- Feature vector computation: 22 static features (angles, distances, proportions) + 24 temporal features (velocities, displacement, cadence, drop speeds, orientation dynamics, post-fall inactivity)
- Group-aware splitting: frames from the same video clip never cross split boundaries (prevents temporal leakage)

**Train / Validation / Test Split:**

| Split | Percentage | Method |
|---|---|---|
| Training | ~70% | GroupShuffleSplit (15% held out for test, then 15% of remainder for val) |
| Validation | ~15% | GroupShuffleSplit |
| Testing | 15% | GroupShuffleSplit |

**Dataset Source:** `[ADD DATASET LINKS HERE]`
- UR Fall: `http://fenix.ur.edu.pl/~mkepski/dataset/ur_fall.html`
- HMDB51: `https://huggingface.co/datasets/ariG23498/hmdb51`
- IMVIA/Le2i: `[ADD IMVIA/LE2I LINK HERE]`

---

## Activity Classes

| Index | Activity | Category |
|---|---|---|
| 0 | Standing | Normal |
| 1 | Sitting | Normal |
| 2 | Walking | Normal |
| 3 | Running | Normal |
| 4 | **Falling** | **Fall** |
| 5 | Lying Down | Fall-adjacent |
| 6 | Bending | Normal |
| 7 | Squatting | Normal |
| 8 | Jumping | Normal |
| 9 | Climbing Stairs | Normal |
| 10 | Crawling | Normal |
| 11 | Kneeling | Normal |
| 12 | Crouching | Normal |
| 13 | Getting Up | Fall-adjacent |

---

## Model Selection

### Pose Estimation Model

| Property | Value |
|---|---|
| **Model** | MediaPipe Pose Landmarker (Lite) |
| **Provider** | Google MediaPipe |
| **Landmarks Detected** | 33 body landmarks (x, y, z, visibility) |
| **Input** | RGB image or video frame |
| **Detection Confidence** | 0.3 (low-confidence fallback mode) |
| **Why Selected** | Real-time performance in-browser, 33 landmarks sufficient for fall-relevant joints (shoulders, hips, knees, ankles), lightweight "lite" variant for fast inference |

MediaPipe Pose detects 33 anatomical landmarks including shoulders, elbows, wrists, hips, knees, and ankles. Each landmark provides normalized (x, y, z) coordinates and a visibility score. The system uses the "lite" model variant for fast real-time inference, with automatic fallback to a low-confidence threshold (0.3) when the primary detection fails.

### Activity Classification Model

| Property | Value |
|---|---|
| **Model** | HistGradientBoosting + RandomForest Soft-Voting Ensemble |
| **Algorithm** | HistGradientBoostingClassifier (primary) + RandomForestClassifier |
| **Input Features** | 46 (22 static + 24 temporal) |
| **Output Classes** | 14 activity classes |
| **Selection Metric** | `weighted-F1 + 0.6 * fall-F2` on validation set |
| **Why Selected** | Handles mixed feature types well, robust to class imbalance with balanced class weights, strong performance on tabular biomechanical features |

Three candidate models were trained and compared: RandomForest, HistGradientBoosting, and a soft-voting ensemble of both. The ensemble was selected based on the combined selection metric.

### Binary Fall Detector

| Property | Value |
|---|---|
| **Model** | HistGradientBoostingClassifier (binary) |
| **Output** | Fall vs No-Fall probability |
| **Threshold Tuning** | Maximises F2 score on validation (recall-weighted) |

### LSTM Sequence Model

| Property | Value |
|---|---|
| **Model** | Keras LSTM (128 -> 64 units, dropout 0.3) |
| **Input** | 20-frame sequences of 43 features |
| **Output** | Binary fall probability |
| **Threshold** | 0.32 (tuned on validation F2) |

### Classification Architecture

```
MediaPipe Pose (33 landmarks)
        |
   Feature Extraction (46 features)
        |
        +---> HistGB+RF Ensemble (14-class activity)
        |
        +---> Binary Fall Detector (Fall vs No-Fall)
        |
        +---> LSTM Sequence Model (temporal fall pattern over 20 frames)
        |
   Fall Signal Fusion (max of 3 fall probabilities)
        |
   FallConfirmer (10 consecutive frames + torso angle >= 50 deg)
        |
   Confirmed Fall -> Emergency Alert
```

---

## Pose Estimation

The system uses MediaPipe Pose to detect 33 body landmarks in each video frame. Key landmarks used for fall detection include:

- **Shoulders** (left/right) — upper body reference
- **Elbows** (left/right) — arm position for activity classification
- **Wrists** (left/right) — hand position
- **Hips** (left/right) — lower body reference, fall detection
- **Knees** (left/right) — leg position, bending/sitting detection
- **Ankles** (left/right) — foot grounding, stance detection
- **Nose** — head position

These landmarks are used to compute torso inclination, joint angles, body proportions, and relative positions that form the 46-dimensional feature vector.

**Pose Estimation Output**

`[INSERT POSE ESTIMATION SCREENSHOT HERE]`

---

## Feature Extraction

Raw MediaPipe landmarks (33 points x 4 values = 132 values) are converted into 46 engineered biomechanical features per frame:

### Static Features (22)

| Feature | Description |
|---|---|
| `shoulder_hip_dist` | Distance between shoulder midpoint and hip center |
| `torso_inclination` | Angle of torso relative to vertical (degrees) |
| `left_knee_angle` | Left knee joint angle |
| `right_knee_angle` | Right knee joint angle |
| `left_elbow_angle` | Left elbow joint angle |
| `right_elbow_angle` | Right elbow joint angle |
| `left_hip_angle` | Left hip joint angle |
| `right_hip_angle` | Right hip joint angle |
| `shoulder_width_ratio` | Shoulder width as proportion of body |
| `hip_width_ratio` | Hip width as proportion of body |
| `left_shoulder_hip_knee` | Left shoulder-hip-knee angle |
| `right_shoulder_hip_knee` | Right shoulder-hip-knee angle |
| `left_ankle_y_ratio` | Left ankle Y position relative to body |
| `right_ankle_y_ratio` | Right ankle Y position relative to body |
| `left_wrist_y_ratio` | Left wrist Y position relative to body |
| `right_wrist_y_ratio` | Right wrist Y position relative to body |
| `head_to_hip_ratio` | Head-to-hip vertical distance ratio |
| `left_knee_ankle` | Left knee-ankle segment angle |
| `right_knee_ankle` | Right knee-ankle segment angle |
| `torso_leg_angle` | Combined torso-leg angle |
| `shoulder_hip_knee_left` | Left side shoulder-hip-knee alignment |
| `shoulder_hip_knee_right` | Right side shoulder-hip-knee alignment |

### Temporal Features (24)

| Feature | Description |
|---|---|
| `hip_vel_y` | Vertical hip velocity |
| `head_vel_y` | Vertical head velocity |
| `ankle_vel_y` | Vertical ankle velocity |
| `body_accel_mag` | Body acceleration magnitude |
| `vert_displacement` | Vertical displacement over window |
| `horiz_displacement` | Horizontal displacement over window |
| `cadence` | Movement cadence |
| `feet_grounded` | Whether feet are grounded |
| `hip_drop_speed` | Hip downward speed |
| `shoulder_drop_speed` | Shoulder downward speed |
| `head_drop_speed` | Head downward speed |
| `body_center_disp` | Body center displacement |
| `time_to_horizontal` | Time to reach horizontal posture |
| `posture_change_rate` | Rate of posture change |
| `standing_to_ground` | Standing-to-ground transition indicator |
| `torso_angle_change` | Change in torso angle over window |
| `orientation_velocity` | Torso orientation velocity |
| `rotation_accel` | Rotation acceleration |
| `post_fall_motion` | Post-fall motion indicator |
| `stationary_duration` | Duration of stationary period |
| `movement_after_event` | Movement after potential fall event |
| `hip_speed_mag` | Hip speed magnitude |
| `vert_velocity_dir` | Vertical velocity direction |
| `body_speed_mag` | Overall body speed magnitude |

Temporal features are computed over a sliding window of 30 frames using the `TemporalWindowExtractor` class.

---

## Model Training

### Data Preparation

1. Frames extracted from all video datasets (UR Fall, IMVIA, HMDB51, custom recordings)
2. MediaPipe pose detection run on every frame to obtain 33 landmarks
3. 46 biomechanical features computed per frame
4. Synthetic poses generated for under-represented classes (Gaussian noise on canonical poses)
5. Group-aware train/val/test split ensuring no video clip leaks across splits

### Training Process

- **Algorithm**: HistGradientBoostingClassifier with soft-voting ensemble with RandomForestClassifier
- **Class balancing**: Custom per-class weights with additional boosting for minority activities (Getting Up 3.0x, Running 3.5x, Walking 3.0x, etc.)
- **Oversampling**: Minority classes boosted to minimum 250 samples (capped at 6x duplication ratio)
- **Cross-validation**: 5-fold StratifiedKFold (shuffle=True, random_state=42)

### Key Hyperparameters

| Hyperparameter | HistGradientBoosting | RandomForest |
|---|---|---|
| Estimators / Iterations | max_iter=300 | n_estimators=300 |
| Learning Rate | 0.05 | — |
| Max Depth | — | 14 |
| Max Leaf Nodes | 12 | — |
| Min Samples Leaf | 35 | 6 |
| Min Samples Split | — | 10 |
| L2 Regularisation | 0.5 | — |
| Class Weight | "balanced" | custom weights |
| Early Stopping | Yes (patience=30) | — |
| Validation Fraction | 0.15 | — |

### Model Selection

Three candidates trained: RandomForest, HistGradientBoosting, RF+HistGB Ensemble. Selected by:

**Score = Validation Weighted-F1 + 0.6 x Validation Fall-F2**

The fall-F2 component (beta=2) prioritises recall for the Falling class, ensuring missed falls are penalised more heavily than false alarms.

**Training Notebook:** `[ADD NOTEBOOK LINK HERE]`

---

## Model Evaluation

Results on the held-out test set (1,529 samples, 14 classes):

| Metric | Result |
|---|---|
| **Accuracy** | **82.54%** |
| **Precision (weighted)** | **82.45%** |
| **Recall (weighted)** | **82.54%** |
| **F1-Score (weighted)** | **82.30%** |

### Per-Class Results (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Standing | 0.85 | 0.88 | 0.87 | 254 |
| Sitting | 0.77 | 0.74 | 0.76 | 288 |
| Walking | 0.61 | 0.55 | 0.58 | 62 |
| Running | 0.82 | 0.83 | 0.82 | 59 |
| **Falling** | **0.74** | **0.83** | **0.79** | **168** |
| Lying Down | 0.88 | 0.86 | 0.87 | 176 |
| Bending | 0.79 | 0.81 | 0.80 | 160 |
| Squatting | 1.00 | 1.00 | 1.00 | 43 |
| Jumping | 0.80 | 1.00 | 0.89 | 53 |
| Climbing Stairs | 0.96 | 0.73 | 0.83 | 73 |
| Crawling | 0.95 | 1.00 | 0.97 | 37 |
| Kneeling | 0.98 | 0.98 | 0.98 | 48 |
| Crouching | 0.98 | 0.88 | 0.93 | 64 |
| Getting Up | 0.31 | 0.30 | 0.30 | 44 |
| **Weighted Avg** | **0.81** | **0.81** | **0.81** | **1529** |

The **Falling class** achieves 83% recall, meaning 83% of actual fall frames are correctly detected. The selection metric prioritises this recall via the F2 weighting.

---

## Confusion Matrix

The confusion matrix shows the distribution of predicted vs actual classes on the test set. Correct classifications appear on the diagonal, while off-diagonal elements show misclassifications.

Key observations:
- **Falling** is most often confused with Lying Down and Bending (similar body orientations)
- **Getting Up** has the lowest F1 (0.30) due to visual similarity with Sitting and Standing transitions
- **Squatting, Crawling, Kneeling** achieve near-perfect classification (F1 > 0.93)

`![Confusion Matrix](screenshots/confusion_matrix_test.png)`

`[INSERT CONFUSION MATRIX IMAGE HERE IF ABOVE LINK BROKEN]`

---

## Accuracy and Loss Graphs

### Training Curves

The training curves show model performance over training iterations for both training and validation sets.

`![Training Curves](screenshots/training_curves.png)`

`[INSERT TRAINING CURVES IMAGE HERE IF ABOVE LINK BROKEN]`

### Training History

`![Training History](screenshots/training_history.png)`

`[INSERT TRAINING HISTORY IMAGE HERE IF ABOVE LINK BROKEN]`

### Feature Importance

The feature importance plot shows which biomechanical features contribute most to classification decisions.

`![Feature Importance](screenshots/feature_importance.png)`

`[INSERT FEATURE IMPORTANCE IMAGE HERE IF ABOVE LINK BROKEN]`

---

## Prediction Results

Example predictions from the system show:

- **Predicted activity** label (e.g., Standing, Walking, Falling)
- **Confidence score** (0.0 - 1.0)
- **Pose skeleton overlay** drawn on the input frame
- **Fall status** (Fall Risk LOW / HIGH)
- **Fall probability** from the fused 3-tier detector

`[INSERT PREDICTION SCREENSHOT HERE]`

---

## Fall Detection Logic

The fall detection system uses a **3-tier fusion approach** rather than relying on a single model:

```
Frame arrives
    |
    v
Tier 1: HistGB 14-class model -> "Falling" class probability
    |
Tier 2: Binary Fall Model -> Fall vs No-Fall probability
    |
Tier 3: LSTM Sequence Model -> Temporal fall pattern (over 20 frames)
    |
    v
Fused Fall Probability = max(Tier 1, Tier 2, Tier 3)
    |
    v
Is fused probability >= threshold?
    |
    +--> YES: Is torso angle >= 50 degrees? (physical confirmation)
    |           |
    |           +--> YES: Increment confirmation counter
    |           |
    |           +--> NO: Do not count
    |
    +--> NO: Reset confirmation counter
    |
    v
Confirmation counter >= 10 consecutive frames?
    |
    +--> YES: CONFIRMED FALL -> Trigger Emergency Alert
    |
    +--> NO: Continue Monitoring
```

### Fall Thresholds

| Detector | Threshold | Tuning Method |
|---|---|---|
| HistGB "Falling" class | 0.55 | Default (configurable) |
| Binary Fall Model | Tuned on validation | Maximise F2 score |
| LSTM Sequence Model | 0.32 | Tuned on validation F2 |

### Fall Confirmation Requirements

A fall is only confirmed when **all** of the following are met:
1. Fused fall probability exceeds the threshold
2. Torso angle is >= 50 degrees (body is physically tilted toward horizontal)
3. Suspicious frames persist for 10 consecutive frames (`CONFIRM_FRAMES`)
4. Optional: body remains stationary for 1.5 seconds post-event

This multi-stage approach reduces false positives from activities like Bending, Squatting, or Getting Up that may temporarily resemble a fall.

---

## Emergency Alert System

When a confirmed fall is detected, the system triggers:

1. **Visual Emergency Banner** — Full-width red pulsing banner displaying "FALL ALERT" with the detected activity and fall probability
2. **Siren Audio** — Generated siren WAV played through the browser (base64-encoded, 3-second frequency sweep)
3. **Fall Counter** — Increments the total fall alert count displayed in KPI cards
4. **Fall History Log** — Each fall event is logged with timestamp, fall probability, and status ("CONFIRMED FALL" or "cleared")
5. **Emergency Contact Buttons** — Configurable emergency contact numbers with clickable call buttons (tel: links)

The alert system is implemented in the main-thread update loop (not the background WebRTC callback thread) to ensure Streamlit session state access is thread-safe.

`[INSERT FALL ALERT SCREENSHOT HERE]`

---

## Streamlit Application

## Live Streamlit Application

**Streamlit App:** `[PASTE YOUR STREAMLIT CLOUD LINK HERE]`

**GitHub Repository:** `[PASTE GITHUB REPOSITORY LINK HERE]`

---

### Dashboard Features

| Feature | Implemented |
|---|---|
| Live Webcam Monitoring (WebRTC) | Yes |
| Image Upload and Analysis | Yes |
| Video Upload and Analysis | Yes |
| AI Activity Prediction (14 classes) | Yes |
| Pose Skeleton Visualisation | Yes |
| Activity Classification | Yes |
| Fall Detection (3-tier fusion) | Yes |
| Confidence Score Display | Yes |
| Fall Alert (visual + audio) | Yes |
| Activity Statistics (KPI cards) | Yes |
| Analytics Dashboard (Plotly charts) | Yes |
| Model Performance Page | Yes |
| Record and Retrain (in-app) | Yes |
| Emergency Contact Setup | Yes |
| Pose Model Selector (Fast/Accurate) | Yes |
| Frame Capture and Freeze | Yes |
| Fall History Log | Yes |
| Confusion Matrix Display | Yes |
| Feature Importance Display | Yes |
| Classification Report Display | Yes |

---

### Streamlit Dashboard Screenshots

**Main Dashboard**

`[INSERT MAIN DASHBOARD SCREENSHOT HERE]`

**Live Monitoring**

`[INSERT LIVE MONITORING SCREENSHOT HERE]`

**Image Prediction**

`[INSERT IMAGE PREDICTION SCREENSHOT HERE]`

**Video Prediction**

`[INSERT VIDEO PREDICTION SCREENSHOT HERE]`

**Pose Detection**

`[INSERT POSE DETECTION SCREENSHOT HERE]`

**Fall Alert**

`[INSERT FALL ALERT SCREENSHOT HERE]`

**Analytics**

`[INSERT ANALYTICS SCREENSHOT HERE]`

**Model Performance**

`[INSERT MODEL PERFORMANCE SCREENSHOT HERE]`

---

## Deployment

| Property | Value |
|---|---|
| **Deployment Platform** | Streamlit Community Cloud |
| **Live Application** | `[PASTE STREAMLIT LINK HERE]` |
| **Repository** | `[PASTE GITHUB LINK HERE]` |
| **Demo Video** | `[PASTE VIDEO LINK HERE]` |
| **Python Version** | 3.9+ |
| **Entry Point** | `streamlit run app.py` |

---

## Project Structure

```text
Elderly-Fall-Detection/
├── app.py                        # Main Streamlit application (2728 lines)
├── config.py                     # Centralized paths, thresholds, constants
├── features.py                   # 46-feature extraction (22 static + 24 temporal)
├── extract_feature.py            # Multi-dataset feature extraction pipeline
├── train_model.py                # HistGB/RF ensemble trainer + binary fall model
├── train_sequence_model.py       # LSTM/GRU sequence fall detector trainer
├── build_sequences.py            # Sliding-window sequence dataset builder
├── train_nn.py                   # Keras MLP trainer (3-class fall detection)
├── evaluate_model.py             # Model evaluation + confusion matrices
├── pipeline.py                   # Full auto pipeline (extract -> train -> evaluate)
├── download_urfall.py            # UR Fall dataset downloader
├── download_hmdb51.py            # HMDB51 dataset downloader
├── extract_imvia.py              # IMVIA/Le2i feature extractor
├── ensure_pose_landmarker.py     # MediaPipe model downloader
├── test_app.py                   # Streamlit smoke test
├── requirements.txt              # Python dependencies
├── models/                       # Trained model files
│   ├── fall_model.pkl            # Primary ensemble (14-class)
│   ├── binary_fall_model.pkl     # Binary fall classifier
│   ├── fall_sequence_model.keras # LSTM sequence model
│   ├── fall detection.h5         # Keras MLP (3-class)
│   ├── activity_label_encoder.pkl
│   ├── sequence_normalizer.json
│   ├── pose_landmarker_lite.task
│   └── pose_landmarker_full.task
├── features/                     # Extracted feature arrays
│   ├── features.npz              # Train/val/test splits
│   ├── info.json                 # Activity names, feature names, counts
│   ├── sequences.npz             # LSTM sequence data
│   └── imvia_features.npz        # IMVIA-specific features
├── dataset/                      # Training datasets
│   ├── urfall/                   # UR Fall Dataset
│   ├── hmdb51/                   # HMDB51 activity clips
│   ├── imvia/                    # IMVIA/Le2i fall videos
│   ├── recorded/                 # Custom webcam recordings
│   └── [synthetic generators]    # Pose generators for augmentation
└── screenshots/                  # Evaluation outputs
    ├── confusion_matrix_test.png
    ├── confusion_matrix_train.png
    ├── confusion_matrix_validation.png
    ├── confusion_matrix_fall_not_fall.png
    ├── feature_importance.png
    ├── training_curves.png
    ├── training_history.png
    ├── evaluation_results.json
    ├── classification_report.txt
    └── classification_report_nn.txt
```

---

## Installation and Local Setup

### Prerequisites
- Python 3.9 or higher
- Webcam (for live monitoring)

### Steps

```bash
# 1. Clone the repository
git clone [PASTE GITHUB REPOSITORY LINK HERE]
cd Elderly-Fall-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Download datasets and train models
python pipeline.py

# 4. Run the Streamlit application
streamlit run app.py
```

The application opens at `http://localhost:8501`.

---

## Evaluation Evidence

### FA-2 Evidence Checklist

| Evidence Item | Status | Location |
|---|---|---|
| Pose estimation output | Included | `[INSERT SCREENSHOT]` |
| Model prediction output | Included | `[INSERT SCREENSHOT]` |
| Accuracy | 82.54% | `screenshots/evaluation_results.json` |
| Precision (weighted) | 82.45% | `screenshots/evaluation_results.json` |
| Recall (weighted) | 82.54% | `screenshots/evaluation_results.json` |
| F1-Score (weighted) | 82.30% | `screenshots/evaluation_results.json` |
| Confusion matrix | Included | `screenshots/confusion_matrix_test.png` |
| Training curves | Included | `screenshots/training_curves.png` |
| Feature importance | Included | `screenshots/feature_importance.png` |
| Streamlit dashboard | Deployed | `[PASTE STREAMLIT LINK HERE]` |
| Fall alert | Implemented | `[INSERT SCREENSHOT]` |
| Demo video | Included | `[PASTE VIDEO LINK HERE]` |

---

## Real-World Challenges and Limitations

- **Lighting variation**: MediaPipe pose detection accuracy drops significantly in low-light or backlit conditions, which can cause missed detections.
- **Camera angle**: The system works best with a frontal or slightly angled view. Extreme side or overhead views reduce landmark accuracy.
- **Occlusion**: Partial body occlusion (furniture, other people) causes MediaPipe to lose landmarks, requiring interpolation which may introduce noise.
- **Similar postures**: Activities like Bending, Squatting, and Crouching share similar body orientations, leading to classification confusion.
- **False fall detections**: Activities with rapid downward motion (e.g., sitting down quickly, picking up objects) can momentarily trigger the fall detector before confirmation logic filters them.
- **Getting Up class**: This class has the lowest F1-score (0.30) because the transition from lying/sitting to standing shares visual features with multiple other activities.
- **Dataset limitations**: The training data includes limited elderly-specific footage. The UR Fall and IMVIA datasets primarily feature younger subjects, which may reduce generalisation to actual elderly populations.
- **Single person**: The current system tracks one person at a time. Multi-person environments require person detection and tracking.
- **Temporal window**: The 30-frame feature window introduces a short delay before temporal features stabilise, meaning very brief fall events might be missed.

---

## Future Improvements

- Collect elderly-specific training data to improve real-world generalisation.
- Integrate YOLOv8 person detection for multi-person tracking and identification.
- Add low-light preprocessing (histogram equalisation, CLAHE) to handle poor lighting.
- Implement CCTV/RTSP stream integration for permanent monitoring installations.
- Add wearable sensor fusion (accelerometer + camera) for higher-confidence fall detection.
- Reduce false positive rate with activity-specific cooldown timers.
- Implement periodic model retraining as new data is collected.
- Add cloud-based dashboard for remote caregiver monitoring.
- Integrate with smart home systems (door sensors, motion detectors).
- Deploy on edge devices (Raspberry Pi, Jetson Nano) for standalone monitoring.

---

## Ethical and Healthcare Considerations

- This system is an **AI-assisted monitoring tool**, not a medical device or diagnosis system.
- **Predictions may contain errors** — both false positives (unnecessary alerts) and false negatives (missed falls) are possible.
- **Human supervision is still required** — the system should supplement, not replace, caregiver attention.
- **False positives** can cause alert fatigue and unnecessary emergency responses.
- **False negatives** can result in delayed medical attention for actual falls.
- The system should be used as one component of a broader elderly care strategy that includes regular human check-ins and professional medical oversight.
- Privacy considerations: webcam data is processed locally and is not stored or transmitted unless explicitly configured.

---

## References

- MediaPipe Pose Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- Scikit-learn HistGradientBoosting: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
- Streamlit Documentation: https://docs.streamlit.io/
- OpenCV Documentation: https://docs.opencv.org/
- TensorFlow Documentation: https://www.tensorflow.org/docs
- UR Fall Dataset: http://fenix.ur.edu.pl/~mkepski/dataset/ur_fall.html
- HMDB51 Dataset: https://huggingface.co/datasets/ariG23498/hmdb51
- IMVIA/Le2i Fall Detection Dataset: `[ADD LINK HERE]`
- SafeFall AI — Elderly Fall Detection using YOLO and Deep Learning: `[ADD LINK HERE]`

---

## Final Submission

**Live Streamlit Application:** `[PASTE STREAMLIT LINK HERE]`

**Project Demonstration Video:** `[PASTE VIDEO LINK HERE]`

**Training Notebook:** `[https://colab.research.google.com/drive/1clvQhHRRjirMR_EgK_K7UdDlwFonQMgT?usp=sharing]`
