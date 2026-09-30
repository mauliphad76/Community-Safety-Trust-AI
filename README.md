# Community Safety & Trust AI

## Project Overview

Community Safety & Trust AI is an AI-powered platform focused on safety, fraud detection, and digital trust. The current working application is a Streamlit dashboard with an implemented CNN-based Indian currency image detector. Other module cards are visible in the dashboard but are planned interfaces only.

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

## Current Implemented Features

- Streamlit dashboard with a professional interface and three AI module cards.
- Working Fake Currency Detector using a trained TensorFlow/Keras CNN.
- Upload support for JPG, JPEG, PNG, AVIF, and WEBP images.
- RGB conversion, 224 x 224 resizing, and pixel normalization to `[0, 1]`.
- REAL/FAKE prediction with confidence percentage.
- `UNCLEAR` result when confidence is below 60%.
- Back-to-dashboard navigation from the currency detector.

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

## Currency Detection Model Details

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

## Dataset

The dataset contains Indian currency images organized by class and denomination.

| Class | Images |
|---|---:|
| REAL | 4,937 |
| FAKE | 2,508 |
| **Total** | **7,445** |

The dataset has 7,445 images: 4,937 REAL and 2,508 FAKE. The seven denominations are:

- ₹10
- ₹20
- ₹50
- ₹100
- ₹200
- ₹500
- ₹2000

The image dataset is intentionally not included in the GitHub repository because of its size and storage requirements. The `data/` directory is ignored by Git; the dataset metadata and split CSV files are maintained separately under `dataset_metadata/`.

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

The following structure reflects the current project files. `assets/` and `modules/` currently contain no files. The local `data/` and `venv/` directories are not part of the GitHub repository.

```text
Community_Safety_Trust_AI/
├── app.py
├── requirements.txt
├── assets/                         # Empty
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
├── modules/                        # Empty
├── results/
│   ├── currency_test_results.json
│   ├── currency_training_history.csv
│   ├── currency_confusion_matrix.png
│   └── currency_training_plot.png
├── scripts/
│   ├── build_dataset_index.py
│   ├── create_stratified_splits.py
│   ├── evaluate_currency_cnn.py
│   └── train_currency_cnn.py
└── venv/
```

The `venv/` directory is a local virtual environment and should not be committed to version control. The `data/` directory is excluded from Git because the image dataset is large.

## Installation

The project includes a `requirements.txt` file and an existing Windows virtual environment. From PowerShell, activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell activation is unavailable, use the virtual-environment interpreter directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The current application and currency training/evaluation scripts use Python, Streamlit, TensorFlow/Keras, NumPy, Pillow, and Matplotlib.

## How to Run

From the project root, run:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

The application opens in the browser at the local Streamlit URL shown in the terminal. If the default port is busy, specify another port, for example:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Alternatively, when Streamlit is available on your `PATH`, run:

```powershell
streamlit run app.py
```

## Usage

1. Start the Streamlit application.
2. Select **Open Currency Detector** on the dashboard.
3. Upload a JPG, JPEG, PNG, AVIF, or WEBP currency image.
4. Review the displayed image.
5. Review the REAL/FAKE prediction and confidence percentage.
6. Treat low-confidence results as unclear and scan the note again.
7. Use **Back to Dashboard** to return to the main view.

The denomination folder is not used to determine an uploaded image's prediction. The CNN produces the REAL/FAKE result.

## Future Scope / Planned Modules

- **Planned / Not yet implemented:** Suspicious Activity Detection using YOLO, computer vision, and a rule engine.
- **Planned / Not yet implemented:** Fake News / Scam Detection using BERT and NLP.
- **Planned / Not yet implemented:** Additional safety and trust intelligence features.
- Future work may also include broader validation and model calibration across additional currency image sources.

These items are future work and are not currently available in the dashboard.

## Important Disclaimer

- The current detector is an image-based binary classifier, not a bank-grade verification system.
- Predictions depend on image quality, lighting, framing, and similarity to the training data.
- A confidence score is not a guarantee of correctness.
- The current application does not implement the planned YOLO activity or BERT scam modules.
- Users should rely on official banking or law-enforcement verification for definitive counterfeit assessment.

## Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- Pillow
- Matplotlib (training and evaluation plots)

## GitHub / Project Information

[Community Safety & Trust AI repository](https://github.com/mauliphad76/Community-Safety-Trust-AI)

## Author

**Mauli Phad**
