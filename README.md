# Community Safety & Trust AI

## Project Overview

Community Safety & Trust AI is a local Streamlit dashboard for currency screening, YOLO object detection with crowd-count review signals, and English/Hindi fake-news text classification.

All three modules reuse locally saved model artifacts. Their predictions are decision-support signals, not definitive verification or independent fact-checking.

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
- YOLO object detection for uploaded images, with annotated output, object confidence scores, bounding boxes, and per-class counts.
- English and Hindi Fake News Detection using saved TF-IDF vectorizers and Linear SVMs.
- Direct dashboard navigation between all three modules.
- Optional YOLO camera support remains in the code; the current detector page exposes image uploads.
- An optional experimental alert for the generic COCO `knife` label; it is not a weapon or threat determination.

## System Workflow

1. Currency images are stored under `data/real/` and `data/fake/`.
2. `scripts/build_dataset_index.py` creates image metadata and SHA-256 hashes.
3. `scripts/create_stratified_splits.py` creates reproducible stratified CSV manifests.
4. `scripts/train_currency_cnn.py` trains the CNN using the training and validation manifests.
5. `scripts/evaluate_currency_cnn.py` evaluates the saved model on the unseen test manifest.
6. `app.py` loads the saved currency and YOLO models and provides interactive image analysis through Streamlit.
7. The fake-news module loads the saved language-specific SVM and vectorizer for English or Hindi text input.

## AI Modules

### Implemented: Fake Currency Detector

- Technology: CNN with TensorFlow/Keras.
- Task: Binary REAL versus FAKE image classification.
- Status: Implemented and integrated into the Streamlit dashboard.

### Implemented: Object Detection

- Technology: pretrained Ultralytics YOLO with a simple rule engine.
- Task: detect all supported model classes in an uploaded image, show bounding boxes and per-class counts, and indicate when the configurable person-count threshold is met. The demo exposes image uploads only; existing camera support is retained for later.
- Inference defaults: 18% confidence, 0.70 IoU/NMS, 640-pixel inference size, and `max_det=300`. Select 960 or 1280 for high-resolution images when seeking small objects; larger inputs cost more time and may not help low-resolution images.
- The crowding threshold defaults to 5 people and triggers when the detected count is greater than or equal to that value. It is an adjustable review threshold, not a universal definition of crowding.
- The optional display class filter is applied after inference; it does not limit which classes the model checks.
- The COCO model supports 80 object classes. It does not include mirrors; detecting unsupported categories may require custom-trained weights.
- The loaded COCO weights include a generic `knife` class (class ID 43). Its optional confidence-gated visual/audio alert is experimental and does not confirm a weapon, threat, or crime. No gun class or separate weapon model is configured.
- Camera alert sounds are cooldown-limited and repeated reruns of the same input do not replay an uploaded-image alert.
- The module does not infer suspicious behaviour, intent, danger, or criminal activity.
- The local `yolo11n.pt` weights are used when present; otherwise Ultralytics may attempt to obtain them on first use.

### Fake News / Scam Detector Baseline

- Prediction uses the trained language-specific TF-IDF + Linear SVM models. The UI provides English and Hindi selection.
- A TF-IDF + Logistic Regression, Multinomial Naive Bayes, and Linear SVM baseline is available for English and Hindi. Training, validation-based selection, held-out test evaluation, command-line prediction, and Streamlit integration are implemented.
- English and Hindi use the shared labels `REAL = 0`, `FAKE = 1`. Clean datasets and 70/15/15 train/validation/test splits are written under `data/fake_news/`; preprocessing metadata is written under `dataset_metadata/fake_news/`.
- WELFake source labels are mapped as `0 = FAKE` to project label `1` and `1 = REAL` to project label `0`. Hindi labels continue to come from the source filenames: fake `1`, true `0`.
- Rebuild only English outputs and preserve Hindi data with `python scripts/fake_news/preprocess_and_split.py --language english`.
- Run `python scripts/fake_news/train_baseline.py` followed by `python scripts/fake_news/evaluate_baseline.py`; training reads only train splits, model selection uses validation FAKE-class F1, and only selected models are evaluated on test splits.
- Predict with `python scripts/fake_news/predict_baseline.py --language hindi --model linear_svm --text-file news.txt` for UTF-8 Hindi text.
- The reusable interface is `predict_news(language="hindi", text=news_text)` from `modules.fake_news_detection`. It returns the numeric label, `REAL`/`FAKE` prediction, and a signed Linear SVM decision score. That score is not a calibrated probability; positive values favor FAKE and negative values favor REAL. Empty or unsupported inputs raise explicit errors. The classification is not independent fact-checking.
- Transformer training is deferred and is not part of the current local workflow. An older experiment notebook remains archived under `notebooks/fake_news/`; the app uses the saved Linear SVM models instead.

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

Key application and model files are organized as follows. The local `data/` and `venv/` directories are excluded from Git.

```text
Community_Safety_Trust_AI/
├── app.py
├── requirements.txt
├── yolo11n.pt
├── data/
│   ├── real/                       # Local currency training/test images
│   ├── fake/
│   └── fake_news/                  # Local raw, processed, and split CSVs
├── dataset_metadata/
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   └── fake_news/
├── models/
│   ├── currency_cnn_best.keras
│   └── fake_news/baseline/
│       ├── english/{linear_svm.joblib, tfidf_vectorizer.joblib}
│       └── hindi/{linear_svm.joblib, tfidf_vectorizer.joblib}
├── modules/
│   ├── fake_news_detection.py
│   └── suspicious_activity.py
├── results/
│   ├── currency_*.*
│   └── fake_news/
├── scripts/
│   ├── fake_news/
│   ├── evaluate_currency_cnn.py
│   └── train_currency_cnn.py
├── tests/
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
2. Select a module from the dashboard or the top navigation.
3. For currency, upload a JPG, JPEG, PNG, AVIF, or WEBP note image; review the REAL/FAKE result and confidence/unclear state.
4. For suspicious activity, upload an image and select **Analyze Image**; review object detections and crowd-count review indicators.
5. For fake news, select English or Hindi, enter article text, and select **Analyze Article**.
6. Interpret the fake-news SVM score only as a signed decision margin, not as a probability or confidence percentage.
7. Verify all AI results with appropriate human or authoritative review.

The denomination folder is not used to determine an uploaded image's prediction. The CNN produces the REAL/FAKE result.

## Future Scope

- Authentication and additional safety/trust features remain future work.
- Future work may also include broader validation and model calibration across additional currency image sources.

## Important Disclaimer

- The currency detector is an image-based binary classifier, not a bank-grade verification system. Predictions depend on image quality, lighting, framing, and similarity to training data; confidence is not a guarantee.
- YOLO detections and crowd counts are approximate visual signals. They do not determine intent, danger, or criminal activity.
- Fake-news labels are dataset-trained text classifications, not independent fact-checking. The Linear SVM decision score is not a calibrated probability.
- Use appropriate human review and authoritative sources for consequential decisions.

## Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- scikit-learn and joblib
- Ultralytics YOLO and OpenCV
- NumPy
- Pillow
- Matplotlib (training and evaluation plots)

## GitHub / Project Information

[Community Safety & Trust AI repository](https://github.com/mauliphad76/Community-Safety-Trust-AI)

## Author

**Mauli Phad**
