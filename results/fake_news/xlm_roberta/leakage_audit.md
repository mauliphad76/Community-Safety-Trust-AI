# XLM-RoBERTa Dataset Leakage Audit

Diagnostic only; this audit does not train models or select checkpoints.

## Exact duplicate overlap

| Language | Train-validation | Train-test | Validation-test |
|---|---:|---:|---:|
| English | 0 | 0 | 0 |
| Hindi | 0 | 0 | 0 |

## English source/topic cue counts

Counts are articles containing a cue (REAL/FAKE); cues can overlap.

| Split | Rows | Source cue | REAL | FAKE | Topic cue | REAL | FAKE |
|---|---:|---|---:|---:|---|---:|---:|
| Train | 44,573 | Reuters | 323 | 14966 | election | 4962 | 6946 |
|  | | Associated Press | 249 | 459 | vote | 4594 | 6388 |
|  | | BBC | 183 | 211 | president | 9484 | 15727 |
|  | | CNN | 1518 | 1303 | congress | 2554 | 4892 |
|  | | New York Times | 960 | 5044 | immigration | 944 | 2179 |
|  | | Washington Post | 804 | 746 | brexit | 216 | 629 |
|  | | Fox News | 1294 | 1041 | covid | 1 | 0 |
|  | | Breitbart | 654 | 2371 | vaccine | 90 | 66 |
|  | | Infowars | 144 | 14 | muslim | 1648 | 2246 |
|  | | Sputnik | 108 | 18 | israel | 675 | 1068 |
|  | | The Guardian | 189 | 113 | russia | 2712 | 3502 |
|  | |  |  |  | economy | 938 | 2088 |
| Validation | 9,551 | Reuters | 71 | 3225 | election | 1077 | 1464 |
|  | | Associated Press | 46 | 87 | vote | 985 | 1440 |
|  | | BBC | 45 | 48 | president | 2062 | 3292 |
|  | | CNN | 339 | 265 | congress | 562 | 1045 |
|  | | New York Times | 222 | 1121 | immigration | 190 | 441 |
|  | | Washington Post | 181 | 139 | brexit | 57 | 124 |
|  | | Fox News | 265 | 205 | covid | 1 | 0 |
|  | | Breitbart | 133 | 490 | vaccine | 17 | 19 |
|  | | Infowars | 24 | 1 | muslim | 355 | 480 |
|  | | Sputnik | 21 | 3 | israel | 161 | 203 |
|  | | The Guardian | 49 | 26 | russia | 608 | 714 |
|  | |  |  |  | economy | 193 | 453 |
| Test | 9,552 | Reuters | 85 | 3232 | election | 1052 | 1455 |
|  | | Associated Press | 49 | 102 | vote | 968 | 1375 |
|  | | BBC | 38 | 57 | president | 2029 | 3396 |
|  | | CNN | 321 | 283 | congress | 507 | 1100 |
|  | | New York Times | 207 | 1069 | immigration | 204 | 436 |
|  | | Washington Post | 176 | 163 | brexit | 50 | 134 |
|  | | Fox News | 267 | 238 | covid | 0 | 0 |
|  | | Breitbart | 136 | 496 | vaccine | 18 | 19 |
|  | | Infowars | 27 | 6 | muslim | 358 | 455 |
|  | | Sputnik | 30 | 3 | israel | 124 | 224 |
|  | | The Guardian | 51 | 26 | russia | 576 | 818 |
|  | |  |  |  | economy | 202 | 460 |

## Hindi source/topic cue counts

Counts are articles containing a cue (REAL/FAKE); cues can overlap.

| Split | Rows | Source cue | REAL | FAKE | Topic cue | REAL | FAKE |
|---|---:|---|---:|---:|---|---:|---:|
| Train | 4,357 | PTI | 1 | 0 | चुनाव | 35 | 45 |
|  | | ANI | 0 | 0 | वोट | 11 | 24 |
|  | | PIB | 0 | 2 | मोदी | 25 | 49 |
|  | | Alt News | 0 | 0 | कांग्रेस | 21 | 46 |
|  | | BOOM | 0 | 0 | भाजपा | 11 | 9 |
|  | | India Today | 0 | 0 | सरकार | 105 | 95 |
|  | | NDTV | 2 | 0 | कोरोना | 99 | 140 |
|  | | Aaj Tak | 19 | 0 | टीका | 10 | 2 |
|  | | The Quint | 0 | 0 | मुस्लिम | 8 | 56 |
|  | |  |  |  | पाकिस्तान | 56 | 68 |
|  | |  |  |  | रूस | 55 | 12 |
|  | |  |  |  | अर्थव्यवस्था | 2 | 0 |
| Validation | 934 | PTI | 1 | 0 | चुनाव | 8 | 17 |
|  | | ANI | 0 | 0 | वोट | 2 | 8 |
|  | | PIB | 0 | 0 | मोदी | 5 | 15 |
|  | | Alt News | 0 | 0 | कांग्रेस | 11 | 10 |
|  | | BOOM | 0 | 0 | भाजपा | 1 | 0 |
|  | | India Today | 0 | 0 | सरकार | 23 | 14 |
|  | | NDTV | 6 | 0 | कोरोना | 26 | 25 |
|  | | Aaj Tak | 7 | 0 | टीका | 4 | 1 |
|  | | The Quint | 0 | 0 | मुस्लिम | 2 | 10 |
|  | |  |  |  | पाकिस्तान | 15 | 14 |
|  | |  |  |  | रूस | 13 | 4 |
|  | |  |  |  | अर्थव्यवस्था | 2 | 1 |
| Test | 934 | PTI | 0 | 0 | चुनाव | 9 | 9 |
|  | | ANI | 0 | 0 | वोट | 3 | 7 |
|  | | PIB | 0 | 0 | मोदी | 7 | 17 |
|  | | Alt News | 0 | 0 | कांग्रेस | 2 | 11 |
|  | | BOOM | 0 | 0 | भाजपा | 2 | 1 |
|  | | India Today | 0 | 0 | सरकार | 20 | 17 |
|  | | NDTV | 2 | 0 | कोरोना | 25 | 20 |
|  | | Aaj Tak | 4 | 0 | टीका | 2 | 1 |
|  | | The Quint | 0 | 0 | मुस्लिम | 0 | 16 |
|  | |  |  |  | पाकिस्तान | 11 | 18 |
|  | |  |  |  | रूस | 10 | 6 |
|  | |  |  |  | अर्थव्यवस्था | 0 | 0 |

## Limitations

- The prepared split schema has text, label, and language only; source, publisher, publication date, and topic identifiers are unavailable.
- Publisher-name mentions are heuristic text cues, not verified publication provenance; articles can quote or discuss a publisher.
- Topic terms are a small manually chosen diagnostic list, not an exhaustive topic annotation or topic-disjoint split.
- Existing stratified random splits may distribute related events/topics across splits; no temporal or event-grouped holdout is available.
