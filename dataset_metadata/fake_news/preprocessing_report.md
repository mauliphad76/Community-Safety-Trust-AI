# Fake News Preprocessing Report

- Label mapping: REAL = 0, FAKE = 1
- Split proportions: 70% train, 15% validation, 15% test
- Random state: 42

## English

- Source files: data\fake_news\raw\english\WELFake_Dataset.csv
- Original rows: 72,134
- Cleaned rows: 63,676
- Removed missing/blank text: 0
- Removed exact duplicates: 8,458
- REAL: 28,886
- FAKE: 34,790

| Split | Rows | REAL | FAKE | REAL % | FAKE % |
|---|---:|---:|---:|---:|---:|
| Train | 44,573 | 20,220 | 24,353 | 45.36% | 54.64% |
| Validation | 9,551 | 4,333 | 5,218 | 45.37% | 54.63% |
| Test | 9,552 | 4,333 | 5,219 | 45.36% | 54.64% |

Source label mapping: original 0 (FAKE) -> project 1; original 1 (REAL) -> project 0.
Raw source-label counts: 0=35,028, 1=37,106.

Duplicate leakage across splits: 0 (train-validation 0, train-test 0, validation-test 0).

## Hindi

- Source files: data\fake_news\raw\hindi\hindi_fake_news.csv, data\fake_news\raw\hindi\hindi_true_news.csv
- Original rows: 7,111
- Cleaned rows: 6,225
- Removed missing/blank text: 0
- Removed exact duplicates: 886
- REAL: 2,921
- FAKE: 3,304

| Split | Rows | REAL | FAKE | REAL % | FAKE % |
|---|---:|---:|---:|---:|---:|
| Train | 4,357 | 2,044 | 2,313 | 46.91% | 53.09% |
| Validation | 934 | 438 | 496 | 46.90% | 53.10% |
| Test | 934 | 439 | 495 | 47.00% | 53.00% |

Source label mapping: hindi_fake_news.csv -> 1 (FAKE); hindi_true_news.csv -> 0 (REAL).

Duplicate leakage across splits: 0 (train-validation 0, train-test 0, validation-test 0).

Devanagari-containing rows: 6,220 (99.92%).
