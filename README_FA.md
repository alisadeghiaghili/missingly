<div dir="rtl">

# missingly

> **این README ورژن v1.0.0+ API عمومی را توصیف می‌کند. برای آزمایش‌های تاریخی به شاخه [`legacy-experiments`](https://github.com/alisadeghiaghili/missingly/tree/legacy-experiments) مراجعه کنید.**

> **تحلیل داده‌های گمشده برای pandas — همه چیز در یک جا.**

[![PyPI](https://img.shields.io/pypi/v/missingly)](https://pypi.org/project/missingly/)
[![Python](https://img.shields.io/pypi/pyversions/missingly)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![API stability](https://img.shields.io/badge/API-v1.0.0%20stable-brightgreen)](https://github.com/alisadeghiaghili/missingly/releases/tag/v1.0.0)

🌐 [English](README.md) | [Deutsch](README_DE.md) | فارسی

---

`missingly` یک پکیج Python برای **تشخیص، تجسم، و جایگزینی داده‌های گمشده** در DataFrame‌های pandas است. این پکیج ارائه می‌دهد:

- accessor روان `df.miss.*` که الگوی پکیج `naniar` در R را دنبال می‌کند.
- transformer‌های سازگار با sklearn (`MissinglyImputer`) برای استفاده در `Pipeline`.
- گزارش‌های HTML یک‌مرحله‌ای (`create_report`).
- آزمون‌های آماری برای مکانیسم‌های MCAR / MAR / MNAR.
- تحلیل gap و imputation آگاه از سری‌زمانی.

---

## نصب

```bash
# پکیج اصلی
pip install missingly

# با نمودارهای تعاملی Plotly
pip install missingly[interactive]

# با پشتیبانی از فارسی / عربی (RTL) برای نمودارهای ایستای matplotlib
# لازم است وقتی نام ستون‌ها یا برچسب‌ها حاوی کاراکترهای فارسی/عربی هستند
pip install missingly[rtl]

# همه چیز (interactive + RTL)
pip install missingly[all]
```

> **کاربران فارسی/عربی:** نمودارهای ایستای matplotlib نیاز به `missingly[rtl]` دارند
> (نصب می‌کند `arabic-reshaper` و `python-bidi`) **به‌علاوه** یک فونت سازگار
> مانند [Vazirmatn](https://fonts.google.com/specimen/Vazirmatn) روی سیستم شما.
> نمودارهای تعاملی Plotly (`interactive=True`) بدون هیچ وابستگی اضافه‌ای به درستی کار می‌کنند.

---

## API عمومی (v1)

نمادهای زیر **سطح API پایدار و پشتیبانی‌شده** برای v1 هستند.
تغییرات breaking به این‌ها از طریق یک major-version bump اعلام خواهد شد.

### accessor `df.miss.*`

```python
import missingly  # df.miss را به صورت خودکار ثبت می‌کند
import pandas as pd
import numpy as np

df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, 6]})

df.miss.n_miss()            # 3  — تعداد کل مقادیر گمشده
df.miss.pct_miss()          # 50.0  — درصد گمشده در کل DataFrame
df.miss.miss_var_summary()  # جدول خلاصه به ازای هر ستون
df.miss.vis_miss()          # تجسم ماتریس گمشدگی
df.miss.impute(strategy="mean")  # DataFrame جایگزین‌شده را برمی‌گرداند
```

### خلاصه و تشخیص

```python
import missingly as mi

mi.n_miss(df)             # int — تعداد کل گمشده
mi.pct_miss(df)           # float — درصد کل گمشده
mi.miss_var_summary(df)   # pd.DataFrame — خلاصه به ازای هر ستون
mi.miss_case_summary(df)  # pd.DataFrame — خلاصه به ازای هر سطر
mi.mcar_test(df)          # نتیجه آزمون MCAR لیتل
mi.mar_mnar_test(df)      # شاخص MAR در برابر MNAR
mi.diagnose_missing(df)   # dict مکانیسم + توصیه
```

### تجسم

لایه تجسم در `missingly.visualisation` قرار دارد و از طریق `missingly.visualise` برای سازگاری با نسخه‌های قبلی مجدداً export می‌شود.

> **چیدمان ماژول**
> | ماژول | محتوا |
> |---|---|
> | `missingly.visualisation.static` | همه توابع مبتنی بر matplotlib |
> | `missingly.visualisation.interactive` | backend‌های Plotly (هنگام `interactive=True` فراخوانی می‌شوند) |
> | `missingly.visualisation._base` | helper‌های مشترک: `_rtl_safe`, `_safe_labels`, `_nullity`, `_pct_labels` |
> | `missingly.visualise` | facade بازخروجی نازک — در کد اپلیکیشن از این استفاده کنید |

#### پایه

```python
import missingly as mi
import pandas as pd, numpy as np

df = pd.DataFrame({
    "سن":      [25, np.nan, 47, 33, np.nan],
    "درآمد":   [50000, 62000, np.nan, np.nan, 71000],
    "شهر":     ["تهران", "مشهد", np.nan, "تهران", "اصفهان"],
})

mi.vis_miss(df)          # ماتریس کاشی‌ای با برچسب‌های درصد به ازای هر ستون
mi.matrix(df)            # heatmap حضور/غیاب خام
mi.bar(df)               # نمودار میله‌ای: تعداد گمشده به ازای هر ستون
mi.miss_case(df)         # نمودار میله‌ای: تعداد گمشده به ازای هر سطر
mi.miss_var_pct(df)      # میله‌های افقی: درصد گمشده به ازای هر متغیر، مرتب‌شده
```

#### الگوها

```python
mi.miss_patterns(df)     # میله‌های افقی: رایج‌ترین N الگوی گمشدگی
mi.miss_cooccurrence(df) # heatmap متقارن: میزان هم‌زمان گمشدن دو ستون
mi.upset(df)             # نمودار UpSet از مجموعه‌های گمشدگی متقاطع
```

#### همبستگی / خوشه‌بندی

```python
# heatmap همبستگی nullity (پیرسون روی شاخص‌های گمشدگی باینری)
mi.heatmap(df)
mi.heatmap(df, mask_insignificant=True)  # سلول‌های غیرمعنادار را خاکستری کن

# خوشه‌بندی سلسله‌مراتبی سطرها بر اساس الگوی گمشدگی
ax = mi.miss_cluster(df)

# دندروگرام متغیرها خوشه‌بندی‌شده بر اساس همبستگی nullity
mi.dendrogram(df)
```

#### تعاملی

`interactive=True` را به هر تابع زیر پاس دهید تا یک figure Plotly بگیرید:

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

### Imputation (جایگزینی)

```python
mi.impute_mean(df)           # جایگزینی با میانگین
mi.impute_median(df)         # جایگزینی با میانه
mi.impute_mode(df)           # جایگزینی با مد
mi.impute_knn(df)            # جایگزینی k-NN
mi.impute_mice(df)           # MICE (IterativeImputer + BayesianRidge)
mi.impute_rf(df)             # جایگزینی Random Forest
mi.impute_gb(df)             # جایگزینی Gradient Boosting

# Multiple Imputation — تولید m مجموعه داده برای pooling Rubin
dfs = mi.impute_mice(df, n_imputations=5)
```

#### KNN با فاصله Gower (ترکیب عددی + دسته‌ای)

```python
df_mixed = pd.DataFrame({
    "سن":    [25, np.nan, 35, 40],
    "شهر":   ["لندن", "پاریس", None, "برلین"],
    "درجه":  ["A", "B", "A", None],
})

# KNN اقلیدسی (پیش‌فرض) — سریع
result = mi.impute_knn(df_mixed, n_neighbors=3)

# KNN با فاصله Gower — برای داده‌های غالباً دسته‌ای
result = mi.impute_knn(df_mixed, n_neighbors=3, metric="mixed")
```

> **توجه عملکردی:** فاصله Gower **O(n²)** در حافظه و زمان اجراست.
> از `metric="mixed"` برای مجموعه‌داده‌های با بیش از ۱۰٬۰۰۰ سطر پرهیز کنید.

### Imputation چندگانه (پیشرفته)

برای استنباط آماری معتبر پس از imputation، از `impute_mice(..., n_imputations=m)` استفاده کنید و نتایج مدل را با قوانین Rubin از طریق ابزارهای `missingly.mi` ادغام کنید:

```python
import numpy as np
import pandas as pd
from missingly import impute_mice
from missingly.mi import pool_scalar_estimates
from sklearn.linear_model import LinearRegression

# ۱. تولید m مجموعه داده imputed
dfs = impute_mice(df, n_imputations=5)

# ۲. fit مدل روی هر مجموعه داده
beta1_ests, beta1_vars = [], []
for d in dfs:
    reg = LinearRegression().fit(d[["x"]], d["y"])
    beta1_ests.append(float(reg.coef_[0]))
    resid = d["y"] - reg.predict(d[["x"]])
    ss_x = float(((d["x"] - d["x"].mean()) ** 2).sum())
    beta1_vars.append(float(np.var(resid, ddof=2)) / ss_x)

# ۳. ادغام با قوانین Rubin
result = pool_scalar_estimates(beta1_ests, beta1_vars)
print(f"beta1 ادغام‌شده = {result['q_bar']:.3f}  (واریانس کل = {result['t']:.4f})")
```

### سری‌زمانی

```python
import missingly as mi
import pandas as pd
import numpy as np

index = pd.date_range("2024-01-01", periods=14, freq="D")
temp  = [5.1, 4.8, np.nan, np.nan, 6.2, 6.5, np.nan, 7.0,
         7.3, np.nan, np.nan, np.nan, 8.1, 8.4]
ts = pd.DataFrame({"دما": temp}, index=index)

# ۱. خلاصه gap‌ها
summary = mi.miss_ts_summary(ts, col="دما")

# ۲. تجسم گمشدگی در محور زمان
ax = mi.vis_ts_miss(ts)

# ۳. جایگزینی با درون‌یابی خطی
ts_filled = mi.impute_ts(ts, strategy="linear")
```

**استراتژی‌های موجود برای `impute_ts`:** `ffill`، `bfill`، `linear`، `time`، `spline`.

### یکپارچگی با Pipeline sklearn

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

## مجوز

MIT — به [LICENSE](LICENSE) مراجعه کنید.

</div>
