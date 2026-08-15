<div dir="rtl">

# missingly

> **این README ورژن v1.0.0+ API عمومی را توصیف می‌کند. برای آزمایش‌های تاریخی به شاخه [`legacy-experiments`](https://github.com/alisadeghiaghili/missingly/tree/legacy-experiments) مراجعه کنید.**

> **تحلیل داده‌های گمشده برای pandas — همه چیز در یک جا.**

[![PyPI](https://img.shields.io/pypi/v/missingly)](https://pypi.org/project/missingly/)
[![Python](https://img.shields.io/pypi/pyversions/missingly)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![API stability](https://img.shields.io/badge/API-v1.0.0%20stable-brightgreen)](https://github.com/alisadeghiaghili/missingly/releases/tag/v1.0.0)

🌐 [English](README.md) | [Deutsch](README_DE.md) | فارسی

---

`missingly` یک پکیج Python برای **تشخیص، تجسم، و جایگزینی داده‌های گمشده** در DataFrame‌های pandas است. این پکیج ارائه می‌دهد:

- accessor روان `df.miss.*` که الگوی پکیج `naniar` در R را دنبال می‌کند.
- transformer‌های سازگار با sklearn (`MissinglyImputer`) برای استفاده در `Pipeline`.
- گزارش‌های HTML یک‌مرحله‌ای (`create_report`).
- آزمون لیتل برای MCAR و تشخیص‌های محدودِ الگوی غیبت بر مبنای داده‌های مشاهده‌شده.
- تحلیل gap و imputation آگاه از سری‌زمانی.

**نسخهٔ پایتون پشتیبانی‌شده:** ۳.۹ و بالاتر.

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
# outcome باید کاملاً مشاهده‌شده و هم‌طول df باشد.
mi.missingness_association_test(df, outcome)  # بررسی ارتباط غیبت با پیامد مشاهده‌شده
mi.diagnose_missing(df)   # شواهد MCAR + راهنمای مبتنی بر فرض‌ها
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
mi.heatmap(df)
mi.heatmap(df, mask_insignificant=True)
ax = mi.miss_cluster(df)
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
mi.impute_mean(df)
mi.impute_median(df)
mi.impute_mode(df)
mi.impute_knn(df)
mi.impute_mice(df)
mi.impute_rf(df)
mi.impute_gb(df)

dfs = mi.impute_mice(df, n_imputations=5)
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

summary = mi.miss_ts_summary(ts)
ax = mi.vis_ts_miss(ts)
ts_filled = mi.impute_ts(ts, strategy="linear")
```

KNN با فاصلهٔ Gower برای داده‌های ترکیبی فقط در تابع مستقل و transductive
`mi.impute_knn(..., metric="mixed")` در دسترس است. `MissinglyImputer` مقدار
`metric="mixed"` را رد می‌کند، چون قرارداد امن donor میان train/test هنوز
پیاده‌سازی نشده است.

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

Apache 2.0 — حق نشر ۲۰۲۵ علی صادقی آقیلی. به [LICENSE](LICENSE) مراجعه کنید.

</div>
