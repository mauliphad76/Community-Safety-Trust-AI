# Community Safety & Trust AI Platform

## Project Overview

Community Safety & Trust AI Platform is a Streamlit-based decision-support application for safety, fraud detection, and digital trust workflows.

The currently implemented AI feature is an Indian currency image classifier that predicts whether an uploaded note appears **REAL** or **FAKE**. The prediction is image-based and is not a definitive bank-grade counterfeit verification.

## Problem Statement

Counterfeit currency can be difficult to identify from visual inspection alone. This project explores how a convolutional neural network (CNN) can analyze currency images and provide a quick Real/Fake indication with a confidence score.

## Project Objectives

- Build a reproducible Indian currency image dataset index.
- Create leakage-safe train, validation, and test splits.
- Train a binary CNN classifier for REAL versus FAKE currency images.
- Evaluate the trained model on an unseen test set.
- Integrate the trained model into a Streamlit dashboard.
- Provide a clear uncertainty state and responsible-use disclaimer.

## Current Features

- Streamlit dashboard with three module cards.
- Working Fake Currency Detector using a trained TensorFlow/Keras CNN.
- Upload support for JPG, JPEG, PNG, AVIF, and WEBP images.
- RGB conversion, 224 x 224 resizing, and pixel normalization to `[0, 1]`.
- REAL/FAKE prediction with confidence percentage.
- `UNCLEAR` result when confidence is below 60%.
- Leakage-safe CSV-based dataset splits.
- SHA-256 duplicate grouping during split creation.
- Saved model, training history, training plot, test results, and confusion matrix.

## System Workflow

1. Currency images are stored under `data/real/` and `data/fake/`.
2. `scripts/build_dataset_index.py` creates image metadata and SHA-256 hashes.
3. `scripts/create_stratified_splits.py` creates reproducible stratified CSV manifests.
4. `scripts/train_currency_cnn.py` trains the CNN using the training and validation manifests.
5. `scripts/evaluate_currency_cnn.py` evaluates the saved model on the unseen test manifest.
6. `app.py` loads the saved model and provides interactive image prediction through Streamlit.

## AI Modules

### Implemented: Fake Currency Detector

- Technology: CNN with TensorFlow/Keras.
- Task: Binary REAL versus FAKE image classification.
- Status: Implemented and integrated into the Streamlit dashboard.

### Planned / Coming Soon: Suspicious Activity Detector

- Planned technology: YOLO with a rule engine.
- Status: Not implemented yet.

### Planned / Coming Soon: Fake News / Scam Detector

- Planned technology: BERT with NLP.
- Status: Not implemented yet.

## Fake Currency Detector Details

The dashboard's **Open Currency Detector** flow:

1. Accepts a supported currency image upload.
2. Displays the uploaded image.
3. Converts the image to RGB.
4. Resizes it to `224 x 224` pixels.
5. Normalizes pixel values to `[0, 1]`.
6. Runs the saved CNN model.
7. Interprets label `0` as REAL and label `1` as FAKE.
8. Displays the prediction and confidence when confidence is at least 60%.
9. Displays `UNCLEAR — Please scan the note again.` below 60% confidence.

The dashboard displays this disclaimer:

> This AI result is an image-based prediction and should not be treated as a definitive bank-grade counterfeit verification.

## Dataset Information

The dataset contains Indian currency images organized by class and denomination.

| Class | Images |
|---|---:|
| REAL | 4,937 |
| FAKE | 2,508 |
| **Total** | **7,445** |

The seven denominations are:

- ₹10
- ₹20
- ₹50
- ₹100
- ₹200
- ₹500
- ₹2000

The dataset is split using the following CSV manifests:

| Split | Total | REAL | FAKE |
|---|---:|---:|---:|
| Training | 5,212 | 3,456 | 1,756 |
| Validation | 1,117 | 741 | 376 |
| Testing | 1,116 | 740 | 376 |

The split process uses random seed `42` and keeps exact duplicate images with the same SHA-256 hash in the same split.

## CNN Training Pipeline

The training pipeline is implemented in `scripts/train_currency_cnn.py`.

- Input shape: `224 x 224 x 3`
- Base preprocessing: RGB conversion, resize, and `[0, 1]` normalization
- Training augmentation: small rotation, zoom, and translation
- Validation and test data: no augmentation
- Loss: binary crossentropy
- Optimizer: Adam
- Metrics: accuracy, precision, and recall
- Class imbalance: class weights calculated from the training CSV
- Training limit: up to 20 epochs
- Early stopping: monitors validation loss and restores the best weights
- Checkpoint: saves the best model by validation loss

The test set is not used during training.

## Model Performance

The saved model was evaluated on the unseen test set of 1,116 images.
The best trained model is stored at `models/currency_cnn_best.keras`.

| Metric | Result |
|---|---:|
| Test accuracy | 88.71% |
| Precision | 89.56% |
| Recall | 75.27% |
| F1-score | 81.79% |

Confusion matrix:

| Actual / Predicted | REAL | FAKE |
|---|---:|---:|
| REAL | 707 | 33 |
| FAKE | 93 | 283 |

The model made 990 correct predictions and 126 incorrect predictions on the test set.

## Project Structure

The following structure reflects the current project files and generated artifacts:

```text
Community_Safety_Trust_AI/
├── app.py
├── requirements.txt
├── assets/
├── data/
│   ├── real/
│   │   ├── 10/
│   │   ├── 20/
│   │   ├── 50/
│   │   ├── 100/
│   │   ├── 200/
│   │   ├── 500/
│   │   └── 2000/
│   └── fake/
│       ├── 10/
│       ├── 20/
│       ├── 50/
│       ├── 100/
│       ├── 200/
│       ├── 500/
│       └── 2000/
├── dataset_metadata/
│   ├── all_images.csv
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   └── split_summary.json
├── models/
│   └── currency_cnn_best.keras
├── modules/
├── results/
│   ├── currency_confusion_matrix.png
│   ├── currency_test_results.json
│   ├── currency_training_history.csv
│   └── currency_training_plot.png
├── scripts/
│   ├── build_dataset_index.py
│   ├── create_stratified_splits.py
│   ├── evaluate_currency_cnn.py
│   └── train_currency_cnn.py
└── venv/
```

The `venv/` directory is the existing local virtual environment and is not required to be committed to version control.

## Installation

The project includes a `requirements.txt` file and an existing Windows virtual environment. From PowerShell, activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell activation is unavailable, use the virtual-environment interpreter directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The required packages include Streamlit, TensorFlow, Pillow, NumPy, Pandas, scikit-learn, OpenCV, Ultralytics, Transformers, and PyTorch. The currently implemented currency pipeline directly uses Streamlit, TensorFlow/Keras, Pillow, NumPy, and Matplotlib.

## How to Run

From the project root, run:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

The application opens in the browser at the local Streamlit URL shown in the terminal. If the default port is busy, specify another port, for example:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

## How to Use the Currency Detector

1. Start the Streamlit application.
2. Select **Open Currency Detector** on the dashboard.
3. Upload a JPG, JPEG, PNG, AVIF, or WEBP currency image.
4. Review the displayed image.
5. Review the REAL/FAKE prediction and confidence percentage.
6. Treat low-confidence results as unclear and scan the note again.
7. Use **Back to Dashboard** to return to the main view.

The denomination folder is not used to determine an uploaded image's prediction. The CNN produces the REAL/FAKE result.

## Future Scope

- Implement the Suspicious Activity Detector with YOLO and a rule engine.
- Implement the Fake News / Scam Detector with BERT and NLP.
- Add broader validation and monitoring for real-world image conditions.
- Improve model calibration and evaluation across additional currency image sources.

These items are future work and are not currently available in the dashboard.

## Limitations / Disclaimer

- The current detector is an image-based binary classifier, not a bank-grade verification system.
- Predictions depend on image quality, lighting, framing, and similarity to the training data.
- A confidence score is not a guarantee of correctness.
- The current application does not implement the planned YOLO activity or BERT scam modules.
- Users should rely on official banking or law-enforcement verification for definitive counterfeit assessment.

## Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- Pillow
- NumPy
- Matplotlib
- Pandas and scikit-learn are included in the project dependencies.
- OpenCV, Ultralytics, Transformers, and PyTorch are included in the project dependencies for planned or future work; they are not used by the current currency detector flow.
- Git/GitHub for source-control workflow

## Author / Team

**Community Safety & Trust AI Project Team**

This repository is structured as a college project and GitHub portfolio project. Add the individual author or team member names here when they are finalized.
