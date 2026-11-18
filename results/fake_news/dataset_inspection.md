# Fake News Dataset Inspection

- Source rows: 79,245
- Unique non-empty texts: 69,896
- Empty text rows: 0
- Duplicate text rows (including within files): 9,349
- Conflicting duplicate-label groups: 0
- Labels: {'0': 40348, '1': 38897}
- Languages: {'english': 72134, 'hindi': 7111}

Label encoding: `0 = REAL`, `1 = FAKE`.

WELFake source labels map as 0=FAKE and 1=REAL; project labels are REAL=0 and FAKE=1. Hindi labels are assigned from the explicit hindi_fake_news (1) and hindi_true_news (0) filenames.

| File | Rows | Columns | Text columns | Label column | Missing values | Duplicates |
|---|---:|---|---|---|---|---:|
| `WELFake_Dataset.csv` | 72,134 | `, title, text, label` | `title, text` | `label` | `{'title': 558, 'text': 783}` | 8,462 |
| `hindi_fake_news.csv` | 3,869 | `Articles, ` | `Articles` | `filename` | `{'': 3868}` | 566 |
| `hindi_true_news.csv` | 3,242 | `Articles` | `Articles` | `filename` | `none` | 321 |

Duplicate text is normalized for Unicode compatibility, whitespace, and case before fingerprinting. No article text is copied into this report.
