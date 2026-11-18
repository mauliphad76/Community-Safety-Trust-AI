# Fake News Baseline Comparison

Selection uses validation FAKE-class F1 (label 1). Test metrics are shown only for the model selected for that language; test data was not used for tuning or selection.

| Language | Model | Validation accuracy | Validation precision | Validation recall | Validation F1 (FAKE) | Train time (s) | Validation eval (s) | Test eval (s)* | Test accuracy* | Test precision* | Test recall* | Test F1* | Vectorizer bytes | Classifier bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| English | TF-IDF + Logistic Regression | 0.9592 | 0.9577 | 0.9680 | 0.9628 | 56.53 | 0.03 | n/a | n/a | n/a | n/a | n/a | 7,606,978 | 1,600,875 |
| English | TF-IDF + Multinomial Naive Bayes | 0.8961 | 0.8916 | 0.9220 | 0.9065 | 54.28 | 0.03 | n/a | n/a | n/a | n/a | n/a | 7,606,978 | 6,400,791 |
| English | TF-IDF + Linear SVM **(selected)** | 0.9720 | 0.9722 | 0.9768 | 0.9745 | 55.47 | 0.03 | 0.0304 | 0.9692 | 0.9706 | 0.9732 | 0.9719 | 7,606,978 | 1,600,731 |
| Hindi | TF-IDF + Logistic Regression | 0.9261 | 0.9402 | 0.9194 | 0.9297 | 0.79 | 0.01 | n/a | n/a | n/a | n/a | n/a | 1,688,343 | 374,699 |
| Hindi | TF-IDF + Multinomial Naive Bayes | 0.8908 | 0.8408 | 0.9798 | 0.9050 | 0.67 | 0.01 | n/a | n/a | n/a | n/a | n/a | 1,688,343 | 1,496,151 |
| Hindi | TF-IDF + Linear SVM **(selected)** | 0.9400 | 0.9472 | 0.9395 | 0.9433 | 0.73 | 0.01 | 0.0137 | 0.9315 | 0.9537 | 0.9152 | 0.9340 | 1,688,343 | 374,555 |

## Validation classification details

### English

#### TF-IDF + Logistic Regression

- Accuracy: 0.9592
- FAKE precision: 0.9577
- FAKE recall: 0.9680
- FAKE F1: 0.9628
- Macro F1: 0.9588

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| REAL | 0.9610 | 0.9485 | 0.9547 | 4,333 |
| FAKE | 0.9577 | 0.9680 | 0.9628 | 5,218 |

Confusion matrix (rows=actual, columns=predicted; REAL, FAKE): `[[4110, 223], [167, 5051]]`

#### TF-IDF + Multinomial Naive Bayes

- Accuracy: 0.8961
- FAKE precision: 0.8916
- FAKE recall: 0.9220
- FAKE F1: 0.9065
- Macro F1: 0.8948

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| REAL | 0.9020 | 0.8650 | 0.8831 | 4,333 |
| FAKE | 0.8916 | 0.9220 | 0.9065 | 5,218 |

Confusion matrix (rows=actual, columns=predicted; REAL, FAKE): `[[3748, 585], [407, 4811]]`

#### TF-IDF + Linear SVM

- Accuracy: 0.9720
- FAKE precision: 0.9722
- FAKE recall: 0.9768
- FAKE F1: 0.9745
- Macro F1: 0.9718

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| REAL | 0.9719 | 0.9663 | 0.9691 | 4,333 |
| FAKE | 0.9722 | 0.9768 | 0.9745 | 5,218 |

Confusion matrix (rows=actual, columns=predicted; REAL, FAKE): `[[4187, 146], [121, 5097]]`

#### Selected test evaluation: TF-IDF + Linear SVM

- Selected by validation FAKE F1: 0.9745
- Accuracy: 0.9692
- FAKE precision: 0.9706
- FAKE recall: 0.9732
- FAKE F1: 0.9719
- Macro F1: 0.9689

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| REAL | 0.9676 | 0.9645 | 0.9660 | 4,333 |
| FAKE | 0.9706 | 0.9732 | 0.9719 | 5,219 |

Confusion matrix (rows=actual, columns=predicted; REAL, FAKE): `[[4179, 154], [140, 5079]]`

### Hindi

#### TF-IDF + Logistic Regression

- Accuracy: 0.9261
- FAKE precision: 0.9402
- FAKE recall: 0.9194
- FAKE F1: 0.9297
- Macro F1: 0.9259

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| REAL | 0.9109 | 0.9338 | 0.9222 | 438 |
| FAKE | 0.9402 | 0.9194 | 0.9297 | 496 |

Confusion matrix (rows=actual, columns=predicted; REAL, FAKE): `[[409, 29], [40, 456]]`

#### TF-IDF + Multinomial Naive Bayes

- Accuracy: 0.8908
- FAKE precision: 0.8408
- FAKE recall: 0.9798
- FAKE F1: 0.9050
- Macro F1: 0.8883

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| REAL | 0.9719 | 0.7900 | 0.8715 | 438 |
| FAKE | 0.8408 | 0.9798 | 0.9050 | 496 |

Confusion matrix (rows=actual, columns=predicted; REAL, FAKE): `[[346, 92], [10, 486]]`

#### TF-IDF + Linear SVM

- Accuracy: 0.9400
- FAKE precision: 0.9472
- FAKE recall: 0.9395
- FAKE F1: 0.9433
- Macro F1: 0.9398

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| REAL | 0.9321 | 0.9406 | 0.9364 | 438 |
| FAKE | 0.9472 | 0.9395 | 0.9433 | 496 |

Confusion matrix (rows=actual, columns=predicted; REAL, FAKE): `[[412, 26], [30, 466]]`

#### Selected test evaluation: TF-IDF + Linear SVM

- Selected by validation FAKE F1: 0.9433
- Accuracy: 0.9315
- FAKE precision: 0.9537
- FAKE recall: 0.9152
- FAKE F1: 0.9340
- Macro F1: 0.9314

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| REAL | 0.9085 | 0.9499 | 0.9287 | 439 |
| FAKE | 0.9537 | 0.9152 | 0.9340 | 495 |

Confusion matrix (rows=actual, columns=predicted; REAL, FAKE): `[[417, 22], [42, 453]]`


*Test columns are populated only for validation-selected models.*

Detailed per-model classification reports and validation confusion matrices are saved in the language-specific result directories. Test classification reports and confusion matrices are saved only for the selected model in each language.

Evaluation timings include classifier prediction and metric calculation; the shared TF-IDF transformation is performed once per split and is not included in each model's timing.
