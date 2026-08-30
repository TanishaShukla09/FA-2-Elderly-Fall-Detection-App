# FA-2 Rubric Evidence Checklist

This project satisfies the locally verifiable implementation requirements in the FA-2 brief.
Use this checklist when preparing the final submission.

| Rubric requirement | Status | Evidence in this project |
|---|---|---|
| Pose estimation selected and integrated | Complete | `app.py` uses MediaPipe Pose Landmarker with 33 body landmarks. |
| Activity and fall classification | Complete | `train_model.py`, `train_sequence_model.py`, and `app.py` classify activities and fuse three fall signals. |
| Required activity detection | Complete | The dashboard detects Standing, Sitting, Walking, Falling, and additional activities. |
| Emergency fall alert | Complete | Confirmed falls show a visual alert, siren, toast, event log, and emergency-contact controls. |
| 70/15/15 train-validation-test split | Complete | `extract_feature.py` and `build_sequences.py` use group-aware 70/15/15 splits. |
| Prevent temporal data leakage | Complete | Contiguous exported frames and webcam capture sessions are kept in one split group. |
| Evaluation metrics | Complete | Accuracy, precision, recall, F1, classification report, and fall-focused metrics are produced by `train_model.py` and `evaluate_model.py`. |
| Confusion matrix and model plots | Complete | Saved under `screenshots/`, including `confusion_matrix_test.png`, `training_curves.png`, and `feature_importance.png`. |
| Streamlit dashboard | Complete | `app.py` provides live camera, image upload, video upload, predictions, alerts, analytics, and record/retrain controls. |
| Deployment configuration | Complete locally | `.streamlit/config.toml` and `requirements.txt` are present. |
| Streamlit Cloud URL | Student action required | Deploy this repository, then replace the placeholder URL in `README.md`. |
| GitHub repository URL | Student action required | Push the project to GitHub, then replace the placeholder URL in `README.md`. |
| Screen-recorded demonstration | Student action required | Record the running app showing pose, walking, fall alert, uploads, metrics, and dashboard; then add its link to `README.md`. |
| Live dashboard screenshots | Student action required | Capture screenshots during the demonstration and replace the screenshot placeholders in `README.md`. |

## Required final demonstration sequence

1. Open the deployed Streamlit application.
2. Show image upload and pose landmarks.
3. Show a normal Walking prediction (and briefly Standing/Sitting if possible).
4. Show a fall sequence and the confirmed alert.
5. Show video upload, analytics, and the model-performance page.
6. Show `confusion_matrix_test.png` and the current evaluation metrics.
7. State the limitations: camera angle, lighting, occlusion, false alerts, and that this is not a medical device.

## Important accuracy note

After changing frame grouping or adding recordings, run feature extraction and retrain before presenting metrics. The new result may be lower than an earlier value because the new group split prevents neighboring frames from leaking into both train and test sets; this is a more trustworthy evaluation.
