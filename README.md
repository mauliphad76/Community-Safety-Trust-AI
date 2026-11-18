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

- **MongoDB Cloud Atlas Authentication**: Secure Register & Login flow with bcrypt cryptographic salting & hashing, session management, and connection testing.
- **AI Modules Sidebar**: Unified navigation side-drawer displaying user session info, active modules switcher, and live AI engine metrics.
- **Executive High-Contrast Light Theme**: Clean, highly readable, zero-blue UI designed for accessibility and high contrast.
- **Fake Currency Detector**: Trained TensorFlow/Keras CNN (`models/currency_cnn_best.keras`) evaluating Indian banknotes with Single-Side and Dual-Side scans, confidence scores, and `UNCLEAR` safety fallback.
- **Suspicious Activity & Crowd Screening**: Offline-bundled YOLO model (`models/yolo11n.pt`) with bounding boxes, per-class counts, crowd-count threshold alarms, and optional experimental knife detection alerts.
- **Fake News Detection**: Trained TF-IDF + Linear SVM models (`models/fake_news/baseline/`) for both English and Hindi text articles with signed decision margin scores.
- **Sample Inputs Included**: Bundled sample currency images and bilingual news articles under `sample_inputs/` for instant out-of-the-box testing upon cloning.

## System Workflow

1. User authenticates via MongoDB Cloud Atlas (Register / Sign In).
2. The unified sidebar allows switching between AI modules.
3. Currency notes are analyzed by the saved CNN model (`models/currency_cnn_best.keras`).
4. Visual activity and crowd counts are analyzed by the bundled YOLOv11 model (`models/yolo11n.pt`).
5. English and Hindi news texts are classified by the trained TF-IDF + Linear SVM pipelines.
6. All AI predictions provide decision-support signals with responsible AI disclaimers.

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

Key application and model files are organized as follows. All trained models and sample test inputs are bundled in version control. The heavy raw training dataset (`data/`) and local virtual environment (`venv/`) are excluded from Git.

```text
Community_Safety_Trust_AI/
├── app.py                             # Main Streamlit application with AI modules & sidebar
├── requirements.txt                   # Complete dependencies list
├── .env.example                       # Environment template for MongoDB Atlas Cloud
├── .gitignore                         # Excludes secrets, caches, and raw datasets
├── sample_inputs/                     # Ready-to-use demo files for instant testing
│   ├── currency/                      # Sample real & fake currency banknotes
│   └── fake_news/                     # Sample English & Hindi news articles
├── models/                            # Pre-trained and bundled AI models
│   ├── currency_cnn_best.keras        # Best CNN weights for currency verification (5.1 MB)
│   ├── yolo11n.pt                     # Bundled YOLOv11 weights for activity/crowd (5.6 MB)
│   └── fake_news/baseline/            # TF-IDF vectorizers & Linear SVMs (English & Hindi)
├── modules/
│   ├── auth.py                        # MongoDB Cloud Atlas Register & Login with bcrypt
│   ├── fake_news_detection.py         # Fake news inference pipeline
│   └── suspicious_activity.py         # YOLO object detection & crowd counting engine
├── dataset_metadata/                  # Manifests and split distributions
├── results/                           # Evaluation metrics, confusion matrices, and plots
├── scripts/                           # Training, evaluation, and preprocessing pipelines
└── tests/                             # Automated test suite (auth, integration, inference)
```

---

## 🚀 Setup & Run on Another Laptop

Follow these steps to clone and run this project on any other laptop (Windows, macOS, or Linux).

### Step 1: Clone the Repository

```bash
git clone https://github.com/mauliphad76/Community-Safety-Trust-AI.git
cd Community-Safety-Trust-AI
```

### Step 2: Create and Activate a Python Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If script execution is disabled in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or activate via Command Prompt `.\venv\Scripts\activate.bat`)*

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure MongoDB Cloud Atlas (Authentication)

Authentication is powered by **MongoDB Atlas Cloud**. You have two quick options:

- **Option A (Via Web UI - Recommended)**: Simply launch the app (Step 5). If MongoDB is not yet configured, the app will show a friendly **Connect MongoDB Atlas Cloud** form where you can paste your connection string directly!
- **Option B (Via `.env` file)**: Copy `.env.example` to `.env`:
  ```bash
  cp .env.example .env
  ```
  Edit `.env` and paste your connection string:
  ```env
  MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
  MONGODB_DB_NAME=community_safety_trust_ai
  ```

> *Tip: Ensure your MongoDB Atlas cluster has IP Whitelist set to `0.0.0.0/0` (Allow Access from Anywhere) or includes your laptop's current public IP address.*

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

The app will launch in your default web browser at `http://localhost:8501`.

---

## 🧪 Testing the AI Modules Out-of-the-Box

The repository includes pre-bundled models and sample test inputs under `sample_inputs/`:

1. **Sign Up / Log In**:
   - Create your account on the portal or log in with existing credentials.
2. **Fake Currency Detector**:
   - Open **Currency Detector** from the sidebar.
   - Upload `sample_inputs/currency/sample_real_note.jpg` or `sample_fake_note.jpg`.
   - Review the CNN Real/Fake classification and confidence gauge.
3. **Suspicious Activity Detection**:
   - Open **Suspicious Activity** from the sidebar.
   - Upload any image containing people or objects.
   - Select **Analyze Image** to view YOLO bounding boxes, class counts, and crowding alerts.
4. **Fake News Detection**:
   - Open **Fake News** from the sidebar.
   - Copy sample text from `sample_inputs/fake_news/sample_english_news.txt` or `sample_hindi_news.txt`.
   - Select **Analyze Article** to inspect the Linear SVM decision score.

---

## 🔬 Running Automated Tests

To verify that all modules, MongoDB authentication, and inference pipelines are working properly on your machine:

```bash
python -m unittest discover -s tests
```

---

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
