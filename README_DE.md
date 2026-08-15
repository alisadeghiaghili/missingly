# missingly

> **Diese README beschreibt die öffentliche API ab v1.0.0. Für historische Experimente siehe den [`legacy-experiments`](https://github.com/alisadeghiaghili/missingly/tree/legacy-experiments)-Branch.**

> **Fehlende-Daten-Analyse für pandas — Batterien inklusive.**

[![PyPI](https://img.shields.io/pypi/v/missingly)](https://pypi.org/project/missingly/)
[![Python](https://img.shields.io/pypi/pyversions/missingly)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![API stability](https://img.shields.io/badge/API-v1.0.0%20stable-brightgreen)](https://github.com/alisadeghiaghili/missingly/releases/tag/v1.0.0)

🌐 [English](README.md) | Deutsch | [فارسی](README_FA.md)

---

`missingly` ist ein Python-Paket zur **Diagnose, Visualisierung und Imputation fehlender Daten** in pandas DataFrames. Es bietet:

- Einen flüssigen `df.miss.*`-Accessor, der die Ergonomie des R-Pakets `naniar` nachahmt.
- Sklearn-kompatible Transformer (`MissinglyImputer`) für die Verwendung innerhalb von `Pipeline`.
- Einzeilige HTML-Berichte (`create_report`).
- Little-MCAR-Test und ausdrücklich begrenzte Diagnostik beobachteter Fehlwertmuster.
- Zeitreihenbewusstes Lückenanalyse und Imputation.

**Unterstützte Python-Version:** 3.9 und neuer.

---

## Installation

```bash
# Kernpaket
pip install missingly

# Mit interaktiven Plotly-Diagrammen
pip install missingly[interactive]

# Mit Persisch/Arabisch (RTL)-Unterstützung für statische matplotlib-Plots
pip install missingly[rtl]

# Alles (interactive + RTL)
pip install missingly[all]
```

---

## Öffentliche API (v1)

### `df.miss.*`-Accessor

```python
import missingly
import pandas as pd
import numpy as np

df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, 6]})

df.miss.n_miss()
df.miss.pct_miss()
df.miss.miss_var_summary()
df.miss.vis_miss()
df.miss.impute(strategy="mean")
```

### Zusammenfassung & Diagnose

```python
import missingly as mi

mi.n_miss(df)
mi.pct_miss(df)
mi.miss_var_summary(df)
mi.miss_case_summary(df)
mi.mcar_test(df)
# outcome muss vollständig beobachtet sein und eine Zeile pro Zeile in df enthalten.
mi.missingness_association_test(df, outcome)
mi.diagnose_missing(df)
```

### Visualisierung

```python
import missingly as mi
import pandas as pd, numpy as np

df = pd.DataFrame({
    "alter":      [25, np.nan, 47, 33, np.nan],
    "einkommen":  [50000, 62000, np.nan, np.nan, 71000],
    "stadt":      ["Berlin", "München", np.nan, "Hamburg", "Köln"],
})

mi.vis_miss(df)
mi.matrix(df)
mi.bar(df)
mi.miss_case(df)
mi.miss_var_pct(df)
mi.miss_patterns(df)
mi.miss_cooccurrence(df)
mi.upset(df)
mi.heatmap(df)
mi.miss_cluster(df)
mi.dendrogram(df)
```

#### Interaktiv

```python
mi.vis_miss(df, interactive=True)
mi.heatmap(df, interactive=True)
mi.matrix(df, interactive=True)
mi.bar(df, interactive=True)
```

### Imputation

```python
mi.impute_mean(df)
mi.impute_median(df)
mi.impute_mode(df)
mi.impute_knn(df)
mi.impute_mice(df)
mi.impute_rf(df)
mi.impute_gb(df)

dfs = mi.impute_mice(df, n_imputations=5)
```

### Zeitreihen

```python
index = pd.date_range("2024-01-01", periods=14, freq="D")
temp  = [5.1, 4.8, np.nan, np.nan, 6.2, 6.5, np.nan, 7.0,
         7.3, np.nan, np.nan, np.nan, 8.1, 8.4]
ts = pd.DataFrame({"temperatur": temp}, index=index)

summary = mi.miss_ts_summary(ts)
ax = mi.vis_ts_miss(ts)
ts_filled = mi.impute_ts(ts, strategy="linear")
```

Für gemischte kategoriale/numerische Daten steht Gower-KNN nur als
transduktive Standalone-Funktion `mi.impute_knn(..., metric="mixed")` zur
Verfügung. `MissinglyImputer` lehnt `metric="mixed"` ab, weil noch kein
sicherer induktiver Train/Test-Donorvertrag implementiert ist.

### sklearn-Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("impute", mi.MissinglyImputer(strategy="knn")),
    ("model",  LogisticRegression()),
])
pipe.fit(X_train, y_train)
```

---

## Lizenz

Apache 2.0 — Copyright 2025 Ali Sadeghi Aghili. Siehe [LICENSE](LICENSE).
