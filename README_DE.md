# missingly

> **Diese README beschreibt die öffentliche API ab v1.0.0. Für historische Experimente siehe den [`legacy-experiments`](https://github.com/alisadeghiaghili/missingly/tree/legacy-experiments)-Branch.**

> **Fehlende-Daten-Analyse für pandas — Batterien inklusive.**

[![PyPI](https://img.shields.io/pypi/v/missingly)](https://pypi.org/project/missingly/)
[![Python](https://img.shields.io/pypi/pyversions/missingly)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![API stability](https://img.shields.io/badge/API-v1.0.0%20stable-brightgreen)](https://github.com/alisadeghiaghili/missingly/releases/tag/v1.0.0)

🌐 [English](README.md) | Deutsch | [فارسی](README_FA.md)

---

`missingly` ist ein Python-Paket zur **Diagnose, Visualisierung und Imputation fehlender Daten** in pandas DataFrames. Es bietet:

- Einen flüssigen `df.miss.*`-Accessor, der die Ergonomie des R-Pakets `naniar` nachahmt.
- Sklearn-kompatible Transformer (`MissinglyImputer`) für die Verwendung innerhalb von `Pipeline`.
- Einzeilige HTML-Berichte (`create_report`).
- Statistische Tests für MCAR / MAR / MNAR-Mechanismen.
- Zeitreihenbewusstes Lückenanalyse und Imputation.

---

## Installation

```bash
# Kernpaket
pip install missingly

# Mit interaktiven Plotly-Diagrammen
pip install missingly[interactive]

# Mit Persisch/Arabisch (RTL)-Unterstützung für statische matplotlib-Plots
# Erforderlich, wenn Spaltennamen oder Beschriftungen persische/arabische Zeichen enthalten
pip install missingly[rtl]

# Alles (interactive + RTL)
pip install missingly[all]
```

> **Hinweis für Persisch-/Arabisch-Nutzer:** Statische matplotlib-Plots erfordern `missingly[rtl]`
> (installiert `arabic-reshaper` und `python-bidi`) **plus** eine kompatible Schriftart
> wie [Vazirmatn](https://fonts.google.com/specimen/Vazirmatn) auf Ihrem System.
> Interaktive Plotly-Diagramme (`interactive=True`) funktionieren ohne zusätzliche Abhängigkeiten.

---

## Öffentliche API (v1)

Die unten aufgeführten Symbole bilden die **stabile, unterstützte API-Oberfläche** für v1.
Breaking Changes werden über einen Major-Version-Bump angekündigt.

### `df.miss.*`-Accessor

```python
import missingly  # registriert df.miss automatisch
import pandas as pd
import numpy as np

df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, 6]})

df.miss.n_miss()            # 3  — Gesamtzahl fehlender Werte
df.miss.pct_miss()          # 50.0  — % fehlend im gesamten DataFrame
df.miss.miss_var_summary()  # Zusammenfassungstabelle pro Spalte
df.miss.vis_miss()          # Fehlendheits-Matrix-Visualisierung
df.miss.impute(strategy="mean")  # gibt imputierten DataFrame zurück
```

### Zusammenfassung & Diagnose

```python
import missingly as mi

mi.n_miss(df)             # int — Gesamtzahl fehlender Werte
mi.pct_miss(df)           # float — Gesamtprozentsatz fehlender Werte
mi.miss_var_summary(df)   # pd.DataFrame — Aufschlüsselung pro Spalte
mi.miss_case_summary(df)  # pd.DataFrame — Aufschlüsselung pro Zeile
mi.mcar_test(df)          # Ergebnis von Littles MCAR-Test
mi.mar_mnar_test(df)      # MAR-vs-MNAR-Indikator
mi.diagnose_missing(df)   # Mechanismus + Empfehlungs-Dict
```

### Visualisierung

Die Visualisierungsschicht befindet sich in `missingly.visualisation` und wird
über `missingly.visualise` für Rückwärtskompatibilität re-exportiert.

> **Modul-Layout**
> | Modul | Inhalt |
> |---|---|
> | `missingly.visualisation.static` | Alle matplotlib-basierten Funktionen |
> | `missingly.visualisation.interactive` | Alle Plotly-Backends (aufgerufen wenn `interactive=True`) |
> | `missingly.visualisation._base` | Gemeinsame Hilfsfunktionen: `_rtl_safe`, `_safe_labels`, `_nullity`, `_pct_labels` |
> | `missingly.visualise` | Dünne Re-Export-Fassade — verwenden Sie dies im Anwendungscode |

#### Grundlegend

```python
import missingly as mi
import pandas as pd, numpy as np

df = pd.DataFrame({
    "alter":      [25, np.nan, 47, 33, np.nan],
    "einkommen":  [50000, 62000, np.nan, np.nan, 71000],
    "stadt":      ["Berlin", "München", np.nan, "Hamburg", "Köln"],
})

mi.vis_miss(df)          # annotierte Kachelmatrix mit prozentualen Spaltenbeschriftungen
mi.matrix(df)            # rohe Präsenz/Abwesenheits-Heatmap
mi.bar(df)               # Balkendiagramm: Anzahl fehlender Werte pro Spalte
mi.miss_case(df)         # Balkendiagramm: Anzahl fehlender Werte pro Zeile
mi.miss_var_pct(df)      # horizontale Balken: % fehlend pro Variable, sortiert
```

#### Muster

```python
mi.miss_patterns(df)     # horizontale Balken: häufigste N Fehlendheitsmuster
mi.miss_cooccurrence(df) # symmetrische Heatmap: wie oft zwei Spalten gemeinsam fehlen
mi.upset(df)             # UpSet-Plot von überschneidenden Fehlendheits-Mengen
```

#### Korrelation / Clustering

```python
# Nullity-Korrelations-Heatmap (Pearson auf binären Fehlendheitsindikatoren)
mi.heatmap(df)
mi.heatmap(df, mask_insignificant=True)  # nicht signifikante Zellen grau färben

# Hierarchisches Clustering von Zeilen nach Fehlendheitsmuster
ax = mi.miss_cluster(df)

# Dendrogramm von Variablen, geclustert nach Nullity-Korrelation
mi.dendrogram(df)
```

#### Interaktiv

Übergeben Sie `interactive=True` an eine der folgenden Funktionen, um eine Plotly-Figur zu erhalten:

```python
mi.vis_miss(df, interactive=True)
mi.heatmap(df, interactive=True)
mi.matrix(df, interactive=True)
mi.bar(df, interactive=True)
mi.miss_var_pct(df, interactive=True)
mi.miss_cooccurrence(df, interactive=True)
mi.miss_case(df, interactive=True)
mi.upset(df, interactive=True)
mi.miss_patterns(df, interactive=True)
```

### Imputation

```python
mi.impute_mean(df)           # Mittelwert-Imputation
mi.impute_median(df)         # Median-Imputation
mi.impute_mode(df)           # Modalwert-Imputation
mi.impute_knn(df)            # k-NN-Imputation (Euklidische Distanz, numerisch-sicher)
mi.impute_mice(df)           # MICE (IterativeImputer + BayesianRidge)
mi.impute_rf(df)             # Random-Forest-Imputation
mi.impute_gb(df)             # Gradient-Boosting-Imputation

# Multiple Imputation — m Datensätze für Rubin-Pooling generieren
dfs = mi.impute_mice(df, n_imputations=5)
```

#### KNN mit Gower-Distanz (gemischt numerisch + kategorisch)

```python
df_mixed = pd.DataFrame({
    "alter":   [25, np.nan, 35, 40],
    "stadt":   ["London", "Paris", None, "Berlin"],
    "klasse":  ["A", "B", "A", None],
})

# Euklidisches KNN (Standard) — schnell, kodiert Kategorien ordinal
result = mi.impute_knn(df_mixed, n_neighbors=3)

# Gower-KNN — statistisch korrekt für stark kategorische Daten
result = mi.impute_knn(df_mixed, n_neighbors=3, metric="mixed")
```

> **Leistungshinweis:** Gower-Distanz ist **O(n²)** in Speicher und Laufzeit.
> Vermeiden Sie `metric="mixed"` bei Datensätzen mit mehr als ~10.000 Zeilen.

### Multiple Imputation (erweitert)

Für statistisch valide Inferenz nach der Imputation generieren Sie *m* Datensätze
mit `impute_mice(..., n_imputations=m)` und kombinieren Sie die Modellergebnisse
mit Rubins Regeln über die Hilfsfunktionen in `missingly.mi`:

```python
import numpy as np
import pandas as pd
from missingly import impute_mice
from missingly.mi import pool_scalar_estimates
from sklearn.linear_model import LinearRegression

# 1. m imputierte Datensätze generieren
dfs = impute_mice(df, n_imputations=5)

# 2. Modell auf jeden imputierten Datensatz fitten
beta1_ests, beta1_vars = [], []
for d in dfs:
    reg = LinearRegression().fit(d[["x"]], d["y"])
    beta1_ests.append(float(reg.coef_[0]))
    resid = d["y"] - reg.predict(d[["x"]])
    ss_x = float(((d["x"] - d["x"].mean()) ** 2).sum())
    beta1_vars.append(float(np.var(resid, ddof=2)) / ss_x)

# 3. Mit Rubins Regeln zusammenführen
result = pool_scalar_estimates(beta1_ests, beta1_vars)
print(f"Gepooltes beta1 = {result['q_bar']:.3f}  (Gesamtvarianz = {result['t']:.4f})")
```

### Zeitreihen-Fehlendheit

```python
import missingly as mi
import pandas as pd
import numpy as np

index = pd.date_range("2024-01-01", periods=14, freq="D")
temp  = [5.1, 4.8, np.nan, np.nan, 6.2, 6.5, np.nan, 7.0,
         7.3, np.nan, np.nan, np.nan, 8.1, 8.4]
ts = pd.DataFrame({"temperatur": temp}, index=index)

# 1. Lücken zusammenfassen
summary = mi.miss_ts_summary(ts, col="temperatur")

# 2. Fehlendheit über die Zeitachse visualisieren
ax = mi.vis_ts_miss(ts)

# 3. Mit linearer Interpolation imputieren
ts_filled = mi.impute_ts(ts, strategy="linear")
```

**Verfügbare Strategien für `impute_ts`:** `ffill`, `bfill`, `linear`, `time`, `spline`.

### sklearn-Pipeline-Integration

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

MIT — siehe [LICENSE](LICENSE).
