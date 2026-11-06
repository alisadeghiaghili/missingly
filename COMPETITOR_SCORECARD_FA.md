# کارنامهٔ رقابتی missingly در برابر همهٔ ابزارهای داده‌گمشده (۰ تا ۱۰)

- تاریخ ارزیابی: ۲۰۲۶-۱۱-۰۶
- نسخهٔ ارزیابی‌شده: missingly `2.0.0` روی `main` (کامیت `ccf9853`). روی PyPI فقط `1.0.0` منتشر شده است.
- دامنه: ۶۷ ابزار در Python، R، SAS، SPSS، Stata، JMP، MATLAB، Mplus، Blimp، Julia، JavaScript، Spark، H2O، ابزارهای GUI (JASP، jamovi، KNIME، Orange، RapidMiner)، DataRobot و ابزارهای کیفیت داده (Great Expectations، Soda).
- جایگزین و دقیق‌شدهٔ این دو مقایسه: جدول‌های «پکیج‌های R (رقبای اصلی)…md» و `COMPETITOR_ANALYSIS.md` (۲۰۲۶-۰۸-۰۸). فهرست اصلاحات در بخش ۸ آمده است.

---

## ۱. خلاصهٔ اجرایی

1. **missingly در «گستره» جزو سه ابزار اول دنیاست، اما در «عمق» تقریباً هیچ‌جا اول نیست.** میانگین ۲۰ بُعد امکانات ۴٫۹۵ از ۱۰ است: رتبهٔ ۳ از ۶۷، پشت mice (۵٫۱۰) و SAS (۵٫۰۵) و جلوتر از Stata و SPSS. ولی فقط در **یک** بُعد امتیاز ۸ یا بیشتر می‌گیرد (ویژوال‌سازی). mice شش بُعد بالای ۸ دارد.
2. **در Python، بیشترین پوشش امکانات را missingly دارد.** فاصله‌اش با نفر دوم (PyPOTS با ۲٫۹۵، بدون احتساب تازه‌واردِ تأییدنشدهٔ mipy-impute) زیاد است. هیچ پکیج Python دیگری ویژوال‌سازی، خلاصه‌سازی، تست MCAR، ایمپیوت، MI، شبیه‌سازی، ارزیابی و گزارش را یک‌جا ندارد.
3. **در امتیاز کل (۰٫۶ امکانات + ۰٫۴ غیرامکاناتی) رتبهٔ ۶ از ۶۷ است.** دلیلش ابعاد غیرامکاناتی است. بلوغ (۲)، پذیرش و جامعه (۱؛ ۱۷ دانلود در ماه)، پشتیبانی (۲) و مجوز BUSL (۴) امتیاز را پایین می‌کشند. بدون این‌ها missingly همسطح Stata است.
4. **شکاف‌های مطلق، به ترتیب اهمیت:**
   - تحلیل حساسیت MNAR: امتیاز ۱، در برابر ۱۰ برای rbmi
   - مقیاس‌پذیری: امتیاز ۲، رتبهٔ ۴۹
   - چندسطحی و Survey: امتیاز ۱
   - روش‌های پیشرفته و Matrix completion: امتیاز ۲
   - سری زمانی: امتیاز ۳، در برابر ۱۰ برای imputeTS
   - موتور MI و Pooling: امتیاز ۵ و ۴، در برابر ۱۰ و ۱۰ برای mice
5. **جایی که missingly واقعاً اول یا نزدیک اول است:**
   - Provenance و بازتولیدپذیری: رتبهٔ ۱
   - خلاصه و الگوهای missingness: رتبهٔ ۲، بعد از naniar
   - گسترهٔ ایمپیوت تک‌مقداری: رتبهٔ ۲
   - ارزیابی و مقایسهٔ ایمپیوترها: رتبهٔ ۲
   - ویژوال‌سازی: رتبهٔ ۳
   - تست مکانیزم: رتبهٔ ۳
   - چندزبانگی و RTL: رتبهٔ ۳، و تنها ابزار برنامه‌نویسی با گزارش RTL فارسی

> جملهٔ صادقانه برای README: «جامع‌ترین جعبه‌ابزار داده‌گمشده در Python (اکتشاف + ایمپیوت + MI پایه + گزارش)». جملهٔ «بهتر از mice/naniar/SAS» با این شواهد قابل دفاع نیست.

---

## ۲. روش ارزیابی

- **مقیاس:**
  - ۰: ندارد
  - ۱ تا ۲: ابتدایی
  - ۳ تا ۴: پایه
  - ۵ تا ۶: خوب، با شکاف‌های جدی
  - ۷ تا ۸: قوی
  - ۹: نزدیک به مرجع
  - ۱۰: خودِ مرجع. برای هر بُعد حداکثر ۱ تا ۲ ابزار ۱۰ می‌گیرند.
- **لنگرها:** برای اینکه عددها بین ابزارها قابل مقایسه باشند، این ابزارها مبنا گرفته شدند:
  - mice: ۱۰ در موتور MI و Pooling
  - naniar: ۱۰ در ویژوال‌سازی و خلاصه
  - imputeTS: ۱۰ در سری زمانی
  - rbmi: ۱۰ در MNAR
  - sklearn: ۱۰ در یکپارچگی با Pipeline
- **ستون‌های امکانات (F):**

| کد | بُعد | کد | بُعد |
|---|---|---|---|
| F1 | ویژوال‌سازی missingness | F11 | شبیه‌سازی/آمپیوت MCAR/MAR/MNAR |
| F2 | خلاصه و تحلیل الگو (shadow، pattern) | F12 | ارزیابی و بنچمارک ایمپیوترها |
| F3 | تشخیص مکانیزم (Little's MCAR و …) | F13 | تشخیص‌های ایمپیوت (همگرایی، overlay) |
| F4 | گسترهٔ ایمپیوت تک‌مقداری | F14 | گزارش‌دهی (HTML/PDF/جدول انتشار) |
| F5 | دادهٔ ترکیبی/دسته‌ای صحیح | F15 | یکپارچگی Pipeline یادگیری ماشین |
| F6 | موتور Multiple Imputation (FCS/JM، PMM، predictor matrix) | F16 | یکپارچگی با دیتافریم (pandas/tidy/polars) |
| F7 | Pooling و استنباط (Rubin، Barnard–Rubin، D1–D3) | F17 | مقیاس و کارایی |
| F8 | تحلیل حساسیت MNAR (delta، tipping point، J2R) | F18 | چندسطحی / Survey / طولی |
| F9 | ایمپیوت سری زمانی | F19 | چندزبانگی و RTL |
| F10 | ML پیشرفته / matrix completion / یادگیری عمیق | F20 | Provenance و بازتولیدپذیری |

- **ستون‌های غیرامکاناتی (N):**

| کد | بُعد | کد | بُعد |
|---|---|---|---|
| N1 | بلوغ و پایداری API | N6 | فعالیت نگهداری (۱۲ ماه اخیر) |
| N2 | پذیرش و جامعه (دانلود، ستاره، ارجاع) | N7 | سهولت استفاده |
| N3 | مستندات و منابع آموزشی (کتاب، vignette) | N8 | آزادی مجوز و هزینه (MIT/BSD=۱۰، GPL=۸، BUSL≈۴، تجاری=۱) |
| N4 | اعتبارسنجی آماری (مقاله، تست مرجع) | N9 | تعامل‌پذیری با اکوسیستم |
| N5 | کیفیت مهندسی (تست، CI، تایپ) | N10 | پشتیبانی |

- **منبع شواهد:**
  - missingly: ممیزی مستقیم کد با ارجاع `file:line`. ادعای README بدون پشتوانهٔ کد «ندارد» حساب شد.
  - رقبا: CRAN/crandb، PyPI، pypistats، cranlogs، GitHub API و مستندات رسمی (آبان ۱۴۰۵ / نوامبر ۲۰۲۶).
  - امتیازها قضاوت کارشناسی کالیبره‌شده‌اند، نه اندازه‌گیری خودکار. محدودیت‌ها در بخش ۱۰ آمده است.

---

## ۳. متادیتای رقبا (نسخه، مجوز، پذیرش، نگهداری)

### Python

| ابزار | نسخه | تاریخ | مجوز | ستاره | دانلود/ماه | وضعیت |
|---|---|---|---|---|---|---|
| **missingly** | 2.0.0 (PyPI: 1.0.0) | 2026-09 | BUSL-1.1 (تا 2030) | — | **۱۷** | بسیار فعال، تک‌نگهدارنده، ۵ ماه سابقه |
| missingno | 0.5.2 | 2023-02-26 | MIT | 4,211 | 248,754 | راکد (آخرین push 2024-05) |
| scikit-learn impute | 1.9.1 | 2026-09-11 | BSD-3 | 68,020 | 184M | بسیار فعال |
| statsmodels MICE | 0.15.0 | 2026-08-27 | BSD-3 | 11,743 | 32M | فعال |
| miceforest | 6.0.5 | 2025-10-27 | MIT | 415 | 31,552 | متوسط |
| autoimpute | 0.14.1 | 2023-12-30 | MIT | 252 | 118 | رهاشده |
| hyperimpute | 0.1.17 | 2023-02-28 | MIT | 202 | 916 | رهاشده |
| fancyimpute | 0.7.0 | 2021-10-21 | Apache-2.0 | 1,279 | 43,794 | رهاشده |
| missingpy | 0.2.0 | 2018-12-10 | GPL-3.0 | 246 | ؟ (مشکوک) | رهاشده |
| missforest (PyPI) | 4.2.3 | 2025-01-03 | MIT | 80 | 1,139 | کم‌فعال |
| scikit-na | 0.4.1 | 2026-08-29 | MIT | 5 | 79 | فعال، پذیرش ناچیز |
| MissMecha | 0.1.2 | 2025-07-29 | MIT | 111 | 18 | کم‌فعال |
| NApy | 2.1 | 2026-10-02 | GPL-3.0 | 9 | 799 | فعال |
| PyPOTS | 2.0 (GitHub) | 2026-10-30 | BSD-3 | 2,093 | 61,227 | بسیار فعال |
| ImputeGAP | 1.1.2 | 2026-01-21 | MIT | 70 | ~82 | بسیار فعال |
| pyampute | 0.0.3 | 2022-08-26 | BSD | 75 | 3,882 | رهاشده |
| ydata-profiling / fg-data-profiling | 4.20.0 | 2026-09-11 | MIT | 13,752 | 661,379 | فعال |
| impyute | 0.0.8 | 2019-04-29 | GPL/MIT | 360 | 862 | رهاشده |
| datawig | 0.2.0 | 2020-07-16 | Apache-2.0 | 492 | 206 | آرشیوشده |
| MIDASpy | 1.4.2 | 2026-03-08 | Apache-2.0 | 160 | 209 | کم‌فعال |
| mipy-impute ⚠ | 2.2.0 | 2026-11-03 | MIT | ؟ | تازه منتشرشده | امتیازها بر اساس ادعای README و تأییدنشده |

### R

| پکیج | نسخه | تاریخ | مجوز | ستاره | دانلود/ماه | وضعیت |
|---|---|---|---|---|---|---|
| naniar | 1.1.0 | 2024-03-05 | MIT | 675 | 18,533 | نیمه‌راکد (آخرین commit 2025-04) |
| visdat | 0.6.1 | 2026-09-25 | MIT | 463 | 19,882 | فعال (rOpenSci) |
| mice | 3.19.0 | 2025-12-10 | GPL≥2 | 521 | 77,824 | بسیار فعال |
| VIM | 7.3.1 | 2026-09-11 | GPL≥2 | 93 | 16,943 | بسیار فعال |
| missForest | 1.6.1 | 2025-10-26 | GPL≥2 | 110 | 9,685 | فعال |
| missRanger | 2.6.1 | 2024-12-07 | GPL≥2 | 73 | 4,374 | آرام |
| mixgb | 2.2.3 | 2026-01-17 | GPL≥3 | 27 | 353 | فعال |
| Amelia | 1.8.3 | 2024-11-08 | GPL≥2 | 67 | 7,306 | خاموش |
| imputeTS | 3.4 | 2025-08-25 | GPL-3 | 174 | 22,251 | کند ولی فعال |
| softImpute | 1.4-3 | 2025-05-12 | GPL-2 | — | 3,013 | کند |
| missMDA | 1.23 | 2026-09-17 | GPL≥2 | — | 7,660 | فعال |
| simputation | 0.2.9 | 2024-12-16 | GPL-3 | 94 | 1,490 | خاموش |
| missMethods | 0.4.0 | 2022-09-16 | GPL-3 | 7 | 644 | رهاشده |
| Hmisc | 5.3-0 | 2026-09-06 | GPL≥2 | 211 | 219,841 | بسیار فعال |
| jomo / mitml | 2.7-6 / 0.4-5 | 2023 | GPL | 31 | ~66,000 | خاموش |
| mi | 1.3.1 | 2026-07-28 | GPL≥2 | — | 19,858 | انتشار 2026 |
| smcfcs | 2.0.2 | 2026-05-01 | GPL-3 | 14 | 3,275 | فعال |
| rbmi | 1.7.1 | 2026-11-05 | Apache-2.0 | 21 | 827 | بسیار فعال |
| MissMech | 1.0.4 | 2024-03-05 | GPL≥2 | — | 531 | خاموش |
| finalfit | 1.1.0 | 2025-09-03 | MIT | 276 | 2,309 | کند |
| norm | 1.0-11.1 | 2023-06-18 | GPL≥2 | — | 18,269 | خاموش |
| misl | 2.0.0 | 2026-04-08 | MIT | — | 321 | تازه (tidymodels) |
| JointAI | 1.1.0 | 2026-01-30 | GPL≥2 | 29 | 683 | بسیار فعال |
| micemd | 1.11.0 | 2026-09-28 | GPL | — | 634 | فعال |

### تجاری، GUI، زبان‌های دیگر و پلتفرم‌ها

| ابزار | دسته | نسخه | مجوز/قیمت | وضعیت |
|---|---|---|---|---|
| SAS/STAT PROC MI + MIANALYZE + SURVEYIMPUTE | تجاری | SAS/STAT 15.x / Viya | تجاری، گران | فعال |
| IBM SPSS Missing Values + MI | تجاری | v31/v32 | اشتراک + ماژول افزودنی | فعال |
| Stata mi + misstable | تجاری | Stata 19 / StataNow | تجاری | فعال (بدون تغییر MI در نسخهٔ ۱۹) |
| JMP Pro | تجاری GUI | 19.1 (2026-01) | تجاری | فعال |
| MATLAB (fillmissing، knnimpute) | تجاری | R2026b (تأییدنشده) | تجاری | فعال |
| Mplus | SEM تجاری | 8.x | تجاری | کند |
| Blimp | Bayesian MI | v3 | رایگان، متن‌بسته | متوسط |
| JASP / jamovi | GUI آزاد | 0.19+ / 2.7 | AGPL | فعال |
| Impute.jl / Mice.jl | Julia | 0.7.1 / 0.4.1 (2026) | MIT | فعال |
| danfo.js | JS | — | MIT | کم‌فعال |
| Spark MLlib Imputer | پلتفرم داده | PySpark 4.3.0 (2026-10-30) | Apache-2.0 | بسیار فعال |
| H2O-3 | پلتفرم ML | 3.46.0.12 (2026-08) | Apache-2.0 | فعال |
| KNIME / Orange / RapidMiner | GUI workflow | 5.11 / 3.40 / 2026 | GPL / GPL / تجاری | فعال |
| DataRobot | AutoML | SaaS | تجاری، گران | فعال |
| Great Expectations / Soda | کیفیت داده | 1.24.1 / 4.30.0 | Apache-2.0 + Cloud پولی | فعال |

---

## ۴. جدول امتیاز امکانات — بخش اول (F1 تا F10)

| ابزار | اکوسیستم | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **missingly** | Py | 8 | 7 | 6 | 7 | 5 | 5 | 4 | 1 | 3 | 2 |
| missingno | Py | 8 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sklearn.impute | Py | 0 | 1 | 0 | 6 | 3 | 3 | 0 | 0 | 1 | 4 |
| statsmodels MICE | Py | 2 | 2 | 0 | 1 | 2 | 5 | 5 | 0 | 0 | 0 |
| miceforest | Py | 0 | 1 | 0 | 3 | 6 | 7 | 0 | 0 | 0 | 6 |
| autoimpute | Py | 2 | 2 | 1 | 6 | 5 | 6 | 6 | 1 | 3 | 1 |
| hyperimpute | Py | 0 | 0 | 0 | 6 | 5 | 2 | 0 | 0 | 0 | 8 |
| fancyimpute | Py | 0 | 0 | 0 | 4 | 1 | 1 | 0 | 0 | 0 | 6 |
| missingpy | Py | 0 | 0 | 0 | 3 | 2 | 0 | 0 | 0 | 0 | 5 |
| missforest (PyPI) | Py | 0 | 0 | 0 | 3 | 5 | 0 | 0 | 0 | 0 | 5 |
| scikit-na | Py | 5 | 6 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| MissMecha | Py | 4 | 3 | 5 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| NApy | Py | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| PyPOTS | Py | 1 | 0 | 0 | 4 | 1 | 2 | 0 | 0 | 9 | 9 |
| ImputeGAP | Py | 4 | 2 | 0 | 5 | 0 | 0 | 0 | 0 | 8 | 7 |
| pyampute | Py | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ydata-profiling | Py | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| impyute | Py | 0 | 0 | 0 | 5 | 1 | 1 | 0 | 0 | 2 | 2 |
| datawig | Py | 0 | 0 | 0 | 4 | 6 | 0 | 0 | 0 | 0 | 6 |
| MIDASpy | Py | 0 | 0 | 0 | 0 | 5 | 6 | 5 | 0 | 0 | 7 |
| mipy-impute ⚠ | Py | 3 | 2 | 2 | 6 | 3 | 4 | 5 | 2 | 2 | 4 |
| pandas / polars built-ins | Py/Rust | 0 | 3 | 0 | 4 | 2 | 0 | 0 | 0 | 3 | 0 |
| naniar | R | 10 | 10 | 5 | 3 | 2 | 0 | 0 | 0 | 2 | 0 |
| visdat | R | 6 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mice | R | 4 | 6 | 1 | 4 | 8 | 10 | 10 | 6 | 2 | 4 |
| VIM | R | 9 | 6 | 1 | 8 | 7 | 3 | 0 | 0 | 1 | 3 |
| missForest | R | 0 | 0 | 0 | 3 | 6 | 0 | 0 | 0 | 0 | 6 |
| missRanger | R | 0 | 0 | 0 | 3 | 7 | 4 | 0 | 0 | 0 | 6 |
| mixgb | R | 0 | 0 | 0 | 3 | 7 | 5 | 2 | 0 | 0 | 6 |
| Amelia | R | 3 | 1 | 0 | 1 | 5 | 7 | 5 | 1 | 6 | 0 |
| imputeTS | R | 7 | 5 | 0 | 6 | 0 | 0 | 0 | 0 | 10 | 1 |
| softImpute | R | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 8 |
| missMDA | R | 1 | 0 | 0 | 3 | 8 | 5 | 2 | 0 | 0 | 7 |
| simputation | R | 0 | 0 | 0 | 7 | 5 | 0 | 0 | 0 | 0 | 2 |
| missMethods | R | 0 | 0 | 0 | 5 | 3 | 0 | 0 | 0 | 0 | 0 |
| Hmisc (aregImpute) | R | 3 | 4 | 0 | 5 | 6 | 6 | 6 | 0 | 0 | 3 |
| jomo | R | 0 | 0 | 0 | 0 | 7 | 7 | 2 | 0 | 0 | 0 |
| mitml | R | 0 | 0 | 0 | 0 | 5 | 6 | 9 | 0 | 0 | 0 |
| mi (Gelman) | R | 3 | 3 | 0 | 0 | 6 | 6 | 5 | 0 | 0 | 0 |
| smcfcs | R | 0 | 0 | 0 | 0 | 6 | 7 | 3 | 2 | 0 | 0 |
| rbmi | R | 0 | 1 | 0 | 0 | 2 | 4 | 7 | 10 | 0 | 0 |
| MissMech | R | 1 | 2 | 9 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| finalfit | R | 5 | 6 | 3 | 0 | 0 | 1 | 2 | 0 | 0 | 0 |
| norm | R | 0 | 1 | 0 | 3 | 1 | 5 | 2 | 0 | 0 | 0 |
| misl | R | 0 | 0 | 0 | 2 | 6 | 5 | 2 | 0 | 0 | 6 |
| JointAI | R | 3 | 3 | 0 | 0 | 7 | 7 | 5 | 1 | 0 | 0 |
| micemd | R | 0 | 0 | 0 | 0 | 5 | 6 | 3 | 0 | 0 | 0 |
| SAS PROC MI/MIANALYZE | SAS | 3 | 6 | 3 | 4 | 6 | 9 | 9 | 9 | 5 | 2 |
| IBM SPSS MVA + MI | SPSS | 4 | 7 | 8 | 5 | 3 | 6 | 7 | 1 | 3 | 1 |
| Stata mi + misstable | Stata | 3 | 7 | 3 | 3 | 6 | 9 | 9 | 4 | 4 | 1 |
| JMP Pro | JMP | 5 | 6 | 3 | 5 | 3 | 2 | 1 | 0 | 3 | 6 |
| MATLAB | MATLAB | 2 | 3 | 0 | 6 | 2 | 0 | 0 | 0 | 6 | 3 |
| Mplus | Mplus | 0 | 4 | 3 | 3 | 4 | 6 | 7 | 3 | 2 | 0 |
| Blimp | Blimp | 1 | 2 | 1 | 2 | 7 | 8 | 7 | 3 | 2 | 1 |
| JASP | GUI | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| jamovi | GUI | 1 | 1 | 2 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| Impute.jl | Julia | 0 | 1 | 0 | 5 | 1 | 0 | 0 | 0 | 4 | 3 |
| Mice.jl | Julia | 0 | 1 | 0 | 2 | 3 | 5 | 5 | 0 | 0 | 0 |
| danfo.js | JS | 0 | 2 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| Spark MLlib Imputer | Spark | 0 | 3 | 0 | 3 | 1 | 0 | 0 | 0 | 2 | 0 |
| H2O-3 | H2O | 0 | 3 | 0 | 3 | 2 | 0 | 0 | 0 | 2 | 5 |
| KNIME Missing Value | KNIME | 1 | 3 | 0 | 5 | 3 | 0 | 0 | 0 | 4 | 0 |
| Orange Impute | Orange | 1 | 2 | 0 | 4 | 3 | 0 | 0 | 0 | 1 | 0 |
| Altair RapidMiner | RapidMiner | 1 | 3 | 0 | 5 | 3 | 0 | 0 | 0 | 3 | 0 |
| DataRobot | DataRobot | 1 | 3 | 0 | 3 | 5 | 0 | 0 | 0 | 2 | 1 |
| Great Expectations | DQ | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Soda | DQ | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## ۵. جدول امتیاز امکانات — بخش دوم (F11 تا F20)

| ابزار | اکوسیستم | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **missingly** | Py | 5 | 6 | 5 | 6 | 6 | 6 | 2 | 1 | 7 | 7 |
| missingno | Py | 0 | 0 | 0 | 0 | 0 | 5 | 2 | 0 | 0 | 0 |
| sklearn.impute | Py | 0 | 2 | 1 | 0 | 10 | 6 | 5 | 0 | 0 | 4 |
| statsmodels MICE | Py | 0 | 0 | 2 | 0 | 1 | 6 | 2 | 2 | 0 | 3 |
| miceforest | Py | 0 | 3 | 6 | 0 | 5 | 5 | 6 | 0 | 0 | 4 |
| autoimpute | Py | 0 | 2 | 4 | 0 | 5 | 4 | 1 | 0 | 0 | 3 |
| hyperimpute | Py | 3 | 4 | 0 | 0 | 7 | 4 | 4 | 0 | 0 | 3 |
| fancyimpute | Py | 0 | 1 | 0 | 0 | 4 | 2 | 2 | 0 | 0 | 2 |
| missingpy | Py | 0 | 0 | 0 | 0 | 6 | 2 | 1 | 0 | 0 | 1 |
| missforest (PyPI) | Py | 0 | 0 | 0 | 0 | 4 | 4 | 3 | 0 | 0 | 2 |
| scikit-na | Py | 0 | 0 | 0 | 4 | 0 | 5 | 2 | 0 | 0 | 1 |
| MissMecha | Py | 8 | 4 | 0 | 0 | 4 | 5 | 2 | 0 | 0 | 4 |
| NApy | Py | 0 | 0 | 0 | 0 | 2 | 3 | 8 | 0 | 0 | 1 |
| PyPOTS | Py | 4 | 6 | 1 | 1 | 5 | 3 | 7 | 1 | 0 | 5 |
| ImputeGAP | Py | 5 | 8 | 2 | 2 | 2 | 3 | 3 | 0 | 0 | 4 |
| pyampute | Py | 7 | 0 | 0 | 0 | 5 | 4 | 1 | 0 | 0 | 3 |
| ydata-profiling | Py | 0 | 0 | 0 | 7 | 0 | 7 | 5 | 0 | 1 | 2 |
| impyute | Py | 0 | 0 | 0 | 0 | 1 | 2 | 1 | 0 | 0 | 1 |
| datawig | Py | 0 | 2 | 0 | 0 | 3 | 5 | 3 | 0 | 0 | 2 |
| MIDASpy | Py | 0 | 2 | 1 | 0 | 2 | 4 | 5 | 0 | 0 | 3 |
| mipy-impute ⚠ | Py | 5 | 6 | 3 | 5 | 1 | 4 | 2 | 3 | 0 | 3 |
| pandas / polars built-ins | Py/Rust | 0 | 0 | 0 | 0 | 2 | 10 | 9 | 0 | 0 | 3 |
| naniar | R | 1 | 0 | 3 | 1 | 1 | 9 | 3 | 1 | 0 | 3 |
| visdat | R | 0 | 0 | 0 | 1 | 1 | 9 | 2 | 0 | 0 | 0 |
| mice | R | 9 | 3 | 8 | 1 | 2 | 5 | 4 | 9 | 0 | 6 |
| VIM | R | 1 | 2 | 3 | 1 | 2 | 6 | 5 | 3 | 0 | 2 |
| missForest | R | 0 | 3 | 2 | 0 | 2 | 4 | 4 | 0 | 0 | 2 |
| missRanger | R | 0 | 3 | 1 | 0 | 2 | 6 | 6 | 0 | 0 | 3 |
| mixgb | R | 0 | 1 | 2 | 0 | 2 | 5 | 6 | 0 | 0 | 3 |
| Amelia | R | 0 | 4 | 7 | 0 | 1 | 3 | 5 | 5 | 0 | 3 |
| imputeTS | R | 0 | 3 | 5 | 0 | 2 | 5 | 3 | 0 | 0 | 2 |
| softImpute | R | 0 | 2 | 0 | 0 | 1 | 2 | 7 | 0 | 0 | 2 |
| missMDA | R | 0 | 4 | 3 | 0 | 2 | 3 | 4 | 1 | 0 | 2 |
| simputation | R | 0 | 0 | 0 | 0 | 2 | 8 | 3 | 0 | 0 | 2 |
| missMethods | R | 8 | 6 | 0 | 0 | 1 | 4 | 3 | 0 | 0 | 2 |
| Hmisc (aregImpute) | R | 0 | 2 | 3 | 4 | 1 | 4 | 4 | 1 | 0 | 3 |
| jomo | R | 0 | 0 | 2 | 0 | 1 | 3 | 5 | 8 | 0 | 3 |
| mitml | R | 0 | 0 | 5 | 2 | 1 | 4 | 3 | 9 | 0 | 3 |
| mi (Gelman) | R | 0 | 0 | 6 | 0 | 1 | 3 | 3 | 1 | 0 | 2 |
| smcfcs | R | 0 | 0 | 3 | 0 | 1 | 3 | 3 | 2 | 0 | 3 |
| rbmi | R | 0 | 2 | 3 | 3 | 0 | 5 | 5 | 6 | 0 | 5 |
| MissMech | R | 0 | 0 | 1 | 0 | 0 | 2 | 2 | 0 | 0 | 2 |
| finalfit | R | 0 | 0 | 0 | 8 | 0 | 8 | 2 | 0 | 0 | 1 |
| norm | R | 0 | 0 | 3 | 0 | 0 | 2 | 2 | 0 | 0 | 3 |
| misl | R | 0 | 1 | 1 | 0 | 6 | 7 | 5 | 0 | 0 | 3 |
| JointAI | R | 0 | 2 | 6 | 2 | 0 | 4 | 2 | 8 | 0 | 4 |
| micemd | R | 0 | 0 | 3 | 0 | 0 | 3 | 3 | 8 | 0 | 3 |
| SAS PROC MI/MIANALYZE | SAS | 2 | 3 | 6 | 6 | 2 | 3 | 6 | 6 | 5 | 6 |
| IBM SPSS MVA + MI | SPSS | 0 | 1 | 4 | 5 | 1 | 2 | 3 | 4 | 8 | 4 |
| Stata mi + misstable | Stata | 1 | 2 | 5 | 4 | 1 | 3 | 5 | 8 | 3 | 6 |
| JMP Pro | JMP | 0 | 2 | 3 | 5 | 1 | 3 | 4 | 1 | 7 | 4 |
| MATLAB | MATLAB | 0 | 1 | 1 | 3 | 2 | 3 | 6 | 1 | 5 | 4 |
| Mplus | Mplus | 4 | 1 | 4 | 1 | 0 | 1 | 4 | 8 | 1 | 5 |
| Blimp | Blimp | 0 | 1 | 6 | 2 | 0 | 2 | 4 | 9 | 1 | 5 |
| JASP | GUI | 0 | 0 | 0 | 4 | 0 | 2 | 1 | 0 | 8 | 3 |
| jamovi | GUI | 0 | 0 | 0 | 3 | 0 | 2 | 1 | 0 | 7 | 3 |
| Impute.jl | Julia | 0 | 1 | 0 | 0 | 2 | 6 | 5 | 1 | 0 | 3 |
| Mice.jl | Julia | 0 | 0 | 2 | 1 | 1 | 6 | 6 | 1 | 0 | 3 |
| danfo.js | JS | 0 | 0 | 0 | 0 | 0 | 4 | 3 | 0 | 0 | 2 |
| Spark MLlib Imputer | Spark | 0 | 0 | 0 | 0 | 8 | 6 | 9 | 0 | 1 | 3 |
| H2O-3 | H2O | 0 | 1 | 0 | 1 | 4 | 4 | 8 | 0 | 1 | 4 |
| KNIME Missing Value | KNIME | 0 | 1 | 0 | 5 | 5 | 3 | 5 | 0 | 2 | 4 |
| Orange Impute | Orange | 0 | 1 | 0 | 1 | 4 | 2 | 2 | 0 | 2 | 3 |
| Altair RapidMiner | RapidMiner | 3 | 2 | 0 | 3 | 5 | 3 | 4 | 0 | 2 | 4 |
| DataRobot | DataRobot | 0 | 2 | 2 | 4 | 8 | 4 | 8 | 1 | 2 | 5 |
| Great Expectations | DQ | 0 | 0 | 0 | 7 | 3 | 6 | 7 | 0 | 1 | 5 |
| Soda | DQ | 0 | 0 | 0 | 4 | 2 | 6 | 7 | 0 | 1 | 4 |

## ۶. جدول امتیاز غیرامکاناتی (N1 تا N10)

| ابزار | اکوسیستم | N1 | N2 | N3 | N4 | N5 | N6 | N7 | N8 | N9 | N10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **missingly** | Py | 2 | 1 | 5 | 3 | 7 | 8 | 8 | 4 | 5 | 2 |
| missingno | Py | 6 | 9 | 5 | 2 | 4 | 2 | 10 | 10 | 5 | 2 |
| sklearn.impute | Py | 10 | 10 | 9 | 8 | 9 | 10 | 8 | 10 | 10 | 6 |
| statsmodels MICE | Py | 9 | 6 | 6 | 6 | 8 | 8 | 5 | 10 | 6 | 5 |
| miceforest | Py | 7 | 6 | 6 | 3 | 6 | 4 | 8 | 10 | 6 | 4 |
| autoimpute | Py | 4 | 3 | 6 | 2 | 4 | 0 | 6 | 10 | 3 | 1 |
| hyperimpute | Py | 4 | 3 | 4 | 6 | 6 | 0 | 6 | 10 | 4 | 1 |
| fancyimpute | Py | 6 | 6 | 4 | 4 | 3 | 0 | 7 | 10 | 3 | 1 |
| missingpy | Py | 2 | 3 | 2 | 1 | 2 | 0 | 7 | 8 | 3 | 0 |
| missforest (PyPI) | Py | 3 | 2 | 3 | 1 | 3 | 5 | 8 | 10 | 3 | 2 |
| scikit-na | Py | 3 | 1 | 5 | 1 | 4 | 6 | 7 | 10 | 4 | 2 |
| MissMecha | Py | 1 | 1 | 5 | 3 | 4 | 5 | 7 | 10 | 3 | 2 |
| NApy | Py | 2 | 1 | 5 | 6 | 5 | 7 | 6 | 8 | 4 | 2 |
| PyPOTS | Py | 7 | 7 | 8 | 7 | 6 | 10 | 5 | 10 | 7 | 5 |
| ImputeGAP | Py | 4 | 2 | 6 | 5 | 6 | 10 | 6 | 10 | 4 | 3 |
| pyampute | Py | 3 | 2 | 6 | 5 | 4 | 0 | 7 | 10 | 4 | 1 |
| ydata-profiling | Py | 8 | 9 | 7 | 3 | 7 | 9 | 9 | 10 | 7 | 6 |
| impyute | Py | 3 | 3 | 4 | 1 | 3 | 0 | 6 | 8 | 3 | 0 |
| datawig | Py | 4 | 4 | 4 | 5 | 4 | 0 | 6 | 10 | 2 | 0 |
| MIDASpy | Py | 5 | 3 | 4 | 7 | 4 | 4 | 4 | 10 | 3 | 2 |
| mipy-impute ⚠ | Py | 1 | 0 | 3 | 1 | 3 | 9 | 7 | 10 | 2 | 1 |
| pandas / polars built-ins | Py/Rust | 10 | 10 | 9 | 5 | 9 | 10 | 9 | 10 | 10 | 7 |
| naniar | R | 8 | 8 | 8 | 7 | 7 | 3 | 9 | 10 | 8 | 5 |
| visdat | R | 8 | 7 | 7 | 6 | 7 | 8 | 9 | 10 | 8 | 7 |
| mice | R | 10 | 10 | 9 | 9 | 8 | 8 | 7 | 8 | 9 | 8 |
| VIM | R | 9 | 7 | 7 | 7 | 6 | 10 | 6 | 8 | 6 | 6 |
| missForest | R | 8 | 7 | 4 | 7 | 5 | 7 | 9 | 8 | 4 | 4 |
| missRanger | R | 6 | 6 | 6 | 4 | 5 | 3 | 9 | 8 | 5 | 4 |
| mixgb | R | 4 | 3 | 5 | 6 | 5 | 7 | 7 | 8 | 5 | 4 |
| Amelia | R | 9 | 6 | 7 | 8 | 5 | 3 | 6 | 8 | 5 | 5 |
| imputeTS | R | 8 | 7 | 7 | 8 | 6 | 6 | 9 | 8 | 6 | 5 |
| softImpute | R | 9 | 5 | 5 | 9 | 5 | 5 | 6 | 8 | 3 | 4 |
| missMDA | R | 9 | 6 | 7 | 8 | 5 | 9 | 6 | 8 | 6 | 6 |
| simputation | R | 6 | 4 | 6 | 3 | 7 | 1 | 9 | 8 | 6 | 3 |
| missMethods | R | 4 | 2 | 5 | 3 | 6 | 0 | 7 | 8 | 4 | 2 |
| Hmisc (aregImpute) | R | 10 | 9 | 8 | 8 | 6 | 10 | 5 | 8 | 8 | 6 |
| jomo | R | 7 | 5 | 5 | 7 | 4 | 1 | 4 | 8 | 6 | 3 |
| mitml | R | 8 | 6 | 7 | 7 | 6 | 1 | 6 | 8 | 7 | 4 |
| mi (Gelman) | R | 7 | 4 | 5 | 7 | 4 | 5 | 4 | 8 | 4 | 3 |
| smcfcs | R | 6 | 4 | 6 | 8 | 5 | 8 | 5 | 8 | 5 | 5 |
| rbmi | R | 7 | 3 | 8 | 9 | 9 | 10 | 4 | 10 | 5 | 7 |
| MissMech | R | 6 | 3 | 5 | 8 | 3 | 1 | 6 | 8 | 3 | 2 |
| finalfit | R | 6 | 5 | 8 | 3 | 5 | 4 | 8 | 10 | 6 | 4 |
| norm | R | 8 | 4 | 3 | 8 | 3 | 1 | 3 | 8 | 2 | 2 |
| misl | R | 2 | 1 | 4 | 2 | 5 | 9 | 8 | 10 | 6 | 2 |
| JointAI | R | 7 | 3 | 8 | 8 | 6 | 8 | 5 | 8 | 6 | 5 |
| micemd | R | 6 | 3 | 5 | 7 | 5 | 9 | 5 | 8 | 8 | 4 |
| SAS PROC MI/MIANALYZE | SAS | 10 | 9 | 9 | 9 | 7 | 7 | 5 | 1 | 5 | 10 |
| IBM SPSS MVA + MI | SPSS | 10 | 9 | 7 | 6 | 5 | 7 | 9 | 1 | 6 | 9 |
| Stata mi + misstable | Stata | 10 | 9 | 10 | 9 | 6 | 8 | 7 | 1 | 5 | 9 |
| JMP Pro | JMP | 9 | 7 | 8 | 5 | 5 | 8 | 9 | 1 | 5 | 8 |
| MATLAB | MATLAB | 10 | 9 | 9 | 5 | 6 | 9 | 8 | 1 | 6 | 9 |
| Mplus | Mplus | 9 | 7 | 7 | 8 | 5 | 6 | 5 | 1 | 4 | 7 |
| Blimp | Blimp | 6 | 5 | 6 | 8 | 4 | 6 | 6 | 5 | 4 | 6 |
| JASP | GUI | 7 | 7 | 6 | 3 | 6 | 8 | 10 | 8 | 5 | 5 |
| jamovi | GUI | 7 | 7 | 6 | 3 | 5 | 8 | 10 | 8 | 5 | 5 |
| Impute.jl | Julia | 5 | 3 | 4 | 1 | 6 | 7 | 7 | 10 | 6 | 4 |
| Mice.jl | Julia | 3 | 2 | 5 | 4 | 5 | 6 | 6 | 9 | 6 | 3 |
| danfo.js | JS | 5 | 4 | 5 | 1 | 4 | 3 | 8 | 10 | 5 | 3 |
| Spark MLlib Imputer | Spark | 8 | 9 | 8 | 3 | 8 | 10 | 6 | 10 | 8 | 7 |
| H2O-3 | H2O | 8 | 6 | 7 | 3 | 6 | 7 | 6 | 10 | 6 | 7 |
| KNIME Missing Value | KNIME | 8 | 7 | 7 | 2 | 6 | 9 | 9 | 8 | 7 | 7 |
| Orange Impute | Orange | 8 | 6 | 7 | 2 | 6 | 8 | 9 | 8 | 6 | 5 |
| Altair RapidMiner | RapidMiner | 8 | 6 | 7 | 2 | 5 | 6 | 8 | 2 | 6 | 8 |
| DataRobot | DataRobot | 6 | 6 | 6 | 3 | 5 | 8 | 8 | 1 | 6 | 9 |
| Great Expectations | DQ | 8 | 8 | 8 | 2 | 8 | 10 | 6 | 10 | 8 | 7 |
| Soda | DQ | 7 | 6 | 7 | 2 | 7 | 8 | 7 | 10 | 7 | 7 |

---

## ۷. جمع‌بندی‌های عددی

### ۷.۱ رتبه‌بندی کلی (۶۷ ابزار)

«هستهٔ missing-data» میانگین ابعاد F1 تا F8 به‌علاوهٔ F13 است: تحلیل، تشخیص، ایمپیوت، MI و استنباط.

| رتبه | ابزار | اکوسیستم | میانگین امکانات (F) | میانگین غیرامکاناتی (N) | هستهٔ missing-data | امتیاز کل (۰٫۶F+۰٫۴N) | تعداد ابعاد ≥۸ |
|---|---|---|---|---|---|---|---|
| 1 | mice | R | 5.1 | 8.6 | 6.3 | 6.50 | 6 |
| 2 | SAS PROC MI/MIANALYZE | SAS | 5.0 | 7.2 | 6.1 | 5.91 | 3 |
| 3 | Stata mi + misstable | Stata | 4.3 | 7.4 | 5.4 | 5.57 | 3 |
| 4 | IBM SPSS MVA + MI | SPSS | 3.9 | 6.9 | 5.0 | 5.07 | 2 |
| 5 | sklearn.impute | Py | 2.3 | 9.0 | 1.6 | 4.98 | 1 |
| 6 | **missingly** | Py | 5.0 | 4.5 | 5.3 | 4.77 | 1 |
| 7 | VIM | R | 3.1 | 7.2 | 4.1 | 4.77 | 2 |
| 8 | Hmisc (aregImpute) | R | 2.8 | 7.8 | 3.7 | 4.77 | 0 |
| 9 | PyPOTS | Py | 3.0 | 7.2 | 1.0 | 4.65 | 2 |
| 10 | pandas / polars built-ins | Py/Rust | 1.8 | 8.9 | 1.0 | 4.64 | 2 |
| 11 | naniar | R | 2.7 | 7.3 | 3.7 | 4.54 | 3 |
| 12 | JMP Pro | JMP | 3.2 | 6.5 | 3.1 | 4.52 | 0 |
| 13 | rbmi | R | 2.6 | 7.2 | 3.0 | 4.47 | 1 |
| 14 | MATLAB | MATLAB | 2.4 | 7.2 | 1.6 | 4.32 | 0 |
| 15 | imputeTS | R | 2.5 | 7.0 | 2.6 | 4.27 | 1 |
| 16 | Amelia | R | 2.9 | 6.2 | 3.3 | 4.19 | 0 |
| 17 | Mplus | Mplus | 3.0 | 5.9 | 3.8 | 4.19 | 1 |
| 18 | JointAI | R | 2.7 | 6.4 | 3.6 | 4.18 | 1 |
| 19 | Blimp | Blimp | 3.2 | 5.6 | 4.1 | 4.16 | 2 |
| 20 | Spark MLlib Imputer | Spark | 1.8 | 7.7 | 0.8 | 4.16 | 2 |
| 21 | missMDA | R | 2.2 | 7.0 | 2.4 | 4.15 | 1 |
| 22 | KNIME Missing Value | KNIME | 2.0 | 7.0 | 1.3 | 4.03 | 0 |
| 23 | miceforest | Py | 2.6 | 6.0 | 2.6 | 3.96 | 0 |
| 24 | Great Expectations | DQ | 1.6 | 7.5 | 0.3 | 3.96 | 0 |
| 25 | ydata-profiling | Py | 1.6 | 7.5 | 1.0 | 3.93 | 0 |
| 26 | ImputeGAP | Py | 2.8 | 5.6 | 1.4 | 3.89 | 2 |
| 27 | DataRobot | DataRobot | 2.5 | 5.8 | 1.6 | 3.85 | 2 |
| 28 | mitml | R | 2.4 | 6.0 | 2.8 | 3.81 | 2 |
| 29 | H2O-3 | H2O | 1.9 | 6.6 | 0.9 | 3.78 | 1 |
| 30 | statsmodels MICE | Py | 1.6 | 6.9 | 2.1 | 3.75 | 0 |
| 31 | visdat | R | 1.1 | 7.7 | 1.0 | 3.74 | 1 |
| 32 | Altair RapidMiner | RapidMiner | 2.0 | 5.8 | 1.3 | 3.55 | 0 |
| 33 | Soda | DQ | 1.4 | 6.8 | 0.3 | 3.53 | 0 |
| 34 | missForest | R | 1.6 | 6.3 | 1.2 | 3.48 | 0 |
| 35 | missRanger | R | 2.0 | 5.6 | 1.7 | 3.47 | 0 |
| 36 | finalfit | R | 1.8 | 5.9 | 1.9 | 3.44 | 2 |
| 37 | mipy-impute ⚠ | Py | 3.2 | 3.7 | 3.3 | 3.43 | 0 |
| 38 | micemd | R | 1.7 | 6.0 | 1.9 | 3.42 | 1 |
| 39 | mixgb | R | 2.1 | 5.4 | 2.1 | 3.42 | 0 |
| 40 | smcfcs | R | 1.6 | 6.0 | 2.3 | 3.39 | 0 |
| 41 | Orange Impute | Orange | 1.3 | 6.5 | 1.1 | 3.38 | 0 |
| 42 | misl | R | 2.2 | 4.9 | 1.8 | 3.28 | 0 |
| 43 | JASP | GUI | 1.1 | 6.5 | 0.4 | 3.26 | 1 |
| 44 | jamovi | GUI | 1.1 | 6.4 | 0.8 | 3.25 | 0 |
| 45 | mi (Gelman) | R | 1.9 | 5.1 | 3.2 | 3.21 | 0 |
| 46 | hyperimpute | Py | 2.3 | 4.4 | 1.4 | 3.14 | 1 |
| 47 | jomo | R | 1.9 | 5.0 | 2.0 | 3.14 | 1 |
| 48 | autoimpute | Py | 2.6 | 3.9 | 3.7 | 3.12 | 0 |
| 49 | Impute.jl | Julia | 1.6 | 5.3 | 0.8 | 3.08 | 0 |
| 50 | softImpute | R | 1.1 | 5.9 | 0.1 | 3.05 | 1 |
| 51 | MIDASpy | Py | 2.0 | 4.6 | 1.9 | 3.04 | 0 |
| 52 | Mice.jl | Julia | 1.8 | 4.9 | 2.0 | 3.04 | 0 |
| 53 | simputation | R | 1.4 | 5.3 | 1.3 | 2.99 | 1 |
| 54 | MissMecha | Py | 2.1 | 4.1 | 1.8 | 2.93 | 1 |
| 55 | missingno | Py | 0.9 | 5.5 | 1.2 | 2.74 | 1 |
| 56 | missMethods | R | 1.6 | 4.1 | 0.9 | 2.60 | 1 |
| 57 | scikit-na | Py | 1.4 | 4.3 | 1.7 | 2.53 | 0 |
| 58 | datawig | Py | 1.6 | 3.9 | 1.1 | 2.49 | 0 |
| 59 | fancyimpute | Py | 1.1 | 4.4 | 0.7 | 2.45 | 0 |
| 60 | pyampute | Py | 1.2 | 4.2 | 0.4 | 2.40 | 0 |
| 61 | MissMech | R | 1.0 | 4.5 | 1.6 | 2.40 | 1 |
| 62 | missforest (PyPI) | Py | 1.3 | 4.0 | 0.9 | 2.38 | 0 |
| 63 | NApy | Py | 0.8 | 4.6 | 0.3 | 2.35 | 1 |
| 64 | norm | R | 1.1 | 4.2 | 1.7 | 2.34 | 0 |
| 65 | danfo.js | JS | 0.7 | 4.8 | 0.6 | 2.34 | 0 |
| 66 | impyute | Py | 0.8 | 3.1 | 0.8 | 1.72 | 0 |
| 67 | missingpy | Py | 1.0 | 2.8 | 0.6 | 1.72 | 0 |

> **هشدار تفسیری:** میانگین امکانات به نفع ابزارهای «همه‌کاره» است. یک ابزار تخصصی مثل imputeTS یا rbmi در حوزهٔ خودش ۱۰ می‌گیرد و در بقیهٔ حوزه‌ها ۰، پس میانگینش پایین می‌آید. برای دیدن «عمق»، ستون «تعداد ابعاد ≥۸» و جدول ۷.۲ را ببینید.

### ۷.۲ بهترین ابزار در هر بُعد و رتبهٔ missingly

| بُعد | بهترین امتیاز | دارندهٔ بهترین امتیاز | missingly | رتبهٔ missingly از 67 |
|---|---|---|---|---|
| F1 | 10 | naniar | 8 | 3 |
| F2 | 10 | naniar | 7 | 2 |
| F3 | 9 | MissMech | 6 | 3 |
| F4 | 8 | VIM | 7 | 2 |
| F5 | 8 | mice, missMDA | 5 | 18 |
| F6 | 10 | mice | 5 | 18 |
| F7 | 10 | mice | 4 | 18 |
| F8 | 10 | rbmi | 1 | 9 |
| F9 | 10 | imputeTS | 3 | 10 |
| F10 | 9 | PyPOTS | 2 | 25 |
| F11 | 9 | mice | 5 | 5 |
| F12 | 8 | ImputeGAP | 6 | 2 |
| F13 | 8 | mice | 5 | 8 |
| F14 | 8 | finalfit | 6 | 4 |
| F15 | 10 | sklearn.impute | 6 | 5 |
| F16 | 10 | pandas / polars built-ins | 6 | 8 |
| F17 | 9 | pandas / polars built-ins, Spark MLlib Imputer | 2 | 49 |
| F18 | 9 | mice, mitml, Blimp | 1 | 17 |
| F19 | 8 | IBM SPSS MVA + MI, JASP | 7 | 3 |
| F20 | 7 | **missingly** | 7 | 1 |
| N1 | 10 | sklearn.impute, pandas / polars built-ins, mice, Hmisc (aregImpute), SAS PROC MI/MIANALYZE, IBM SPSS MVA + MI, Stata mi + misstable, MATLAB | 2 | 62 |
| N2 | 10 | sklearn.impute, pandas / polars built-ins, mice | 1 | 62 |
| N3 | 10 | Stata mi + misstable | 5 | 42 |
| N4 | 9 | mice, softImpute, rbmi, SAS PROC MI/MIANALYZE, Stata mi + misstable | 3 | 41 |
| N5 | 9 | sklearn.impute, pandas / polars built-ins, rbmi | 7 | 8 |
| N6 | 10 | sklearn.impute, PyPOTS, ImputeGAP, pandas / polars built-ins, VIM, Hmisc (aregImpute), rbmi, Spark MLlib Imputer, Great Expectations | 8 | 17 |
| N7 | 10 | missingno, JASP, jamovi | 8 | 16 |
| N8 | 10 | missingno, sklearn.impute, statsmodels MICE, miceforest, autoimpute, hyperimpute, fancyimpute, missforest (PyPI), scikit-na, MissMecha, PyPOTS, ImputeGAP, pyampute, ydata-profiling, datawig, MIDASpy, mipy-impute ⚠, pandas / polars built-ins, naniar, visdat, rbmi, finalfit, misl, Impute.jl, danfo.js, Spark MLlib Imputer, H2O-3, Great Expectations, Soda | 4 | 59 |
| N9 | 10 | sklearn.impute, pandas / polars built-ins | 5 | 33 |
| N10 | 10 | SAS PROC MI/MIANALYZE | 2 | 49 |

### ۷.۳ missingly در برابر «پشتهٔ» هر اکوسیستم

اینجا برای هر بُعد بهترین امتیاز موجود در آن اکوسیستم گرفته شده است. یعنی اگر کاربر همهٔ ابزارهای آن اکوسیستم را کنار هم استفاده کند، چه به دست می‌آورد.

| پشته | میانگین F | توضیح |
|---|---|---|
| پشتهٔ R (naniar + mice + imputeTS + rbmi + MissMech + …) | **۸٫۰۵** | سقف واقعی رقابت. در همه‌چیز جز چندزبانگی جلوتر است. |
| پشتهٔ Python بدون missingly | ۶٫۴۵ | sklearn + PyPOTS + miceforest + missingno + MissMecha + … |
| SAS + SPSS + Stata | ۵٫۷۰ | قوی در MI، Pooling، MNAR و پشتیبانی. ضعیف در ML و ویژوال‌سازی. |
| **missingly به‌تنهایی** | ۴٫۹۵ | یک پکیج، یک API |

- **ابعادی که missingly به‌تنهایی از کل پشتهٔ Python جلوتر است:**
  - F2: خلاصه و الگو (۷ در برابر ۶)
  - F3: تست MCAR (۶ در برابر ۵)
  - F4: گسترهٔ ایمپیوت (۷ در برابر ۶)
  - F19: RTL و چندزبانگی (۷ در برابر ۱)
  - F20: Provenance (۷ در برابر ۵)
- **مساوی با کل پشتهٔ Python:**
  - F1: ویژوال‌سازی (۸، هم‌سطح missingno)
  - F8: MNAR (هر دو ۱؛ Python هیچ ابزار MNAR جدی ندارد)

> بزرگ‌ترین فرصت بازار: **در کل اکوسیستم Python هیچ ابزار حساسیت MNAR وجود ندارد** (بهترین امتیاز ۲ است). اولین پیاده‌سازی معتبر delta-adjustment و tipping-point در Python، جای خالی rbmi و PROC MI MNAR را پر می‌کند.

---

## ۸. اصلاحات مقایسه‌های قبلی

### ۸.۱ جدول‌های فایل «پکیج‌های R (رقبای اصلی)…md»

| ادعای قبلی | واقعیت امروز | امتیاز دقیق |
|---|---|---|
| ردیف «miceforest (R)» با لینک CRAN | **چنین پکیجی روی CRAN وجود ندارد** (crandb: `not_found`). miceforest فقط Python است. معادل‌های R: missRanger و mixgb. | ردیف حذف شد |
| missingly: Diagn/MCAR = ✓✓ | فقط یک تست Little's MCAR (EM، با oracle از naniar) به‌علاوهٔ screen لجستیک. شناسایی MAR/MNAR عمداً حذف شده است. | F3 = ۶ |
| missingly: Multi‑Imp = ✓ | `impute_mice` روی `IterativeImputer` + BayesianRidge ساخته شده است. predictor matrix، visit order، `where` و bounds دارد. PMM داخل موتور، متد per-variable، passive و blocks ندارد. Pooling فقط Rubin کلاسیک است؛ Barnard–Rubin و D1 تا D3 ندارد. | F6 = ۵، F7 = ۴ |
| missingly: Adv Algo = ✓ | RF/GB از طریق IterativeImputer (پیش‌فرض `max_iter=1`). softImpute، SVD، GAIN و VAE ندارد. | F10 = ۲ |
| missingly: Report = ✓✓ | HTML مستقل Jinja2 در en/fa/de با RTL. PDF و جدول انتشار ندارد. | F14 = ۶ |
| missingly: sklearn = ✓✓ | `fit/transform` و clone-safe است. فقط DataFrame می‌پذیرد، `check_estimator` ندارد و چند حالتش `NotImplementedError` می‌دهد. | F15 = ۶ |
| missingly: Time‑Series = ~ | درست است. فقط ffill/bfill/linear/time/spline دارد. Kalman، STL و ARIMA ندارد؛ با grep تأیید شد. | F9 = ۳ |
| «missingly تنها پکیجی است که در Viz و Diagn و sklearn ●● دارد» | با امتیاز کالیبره‌شده درست نیست: Viz = ۸، Diagn = ۶، sklearn = ۶. | — |
| scikit‑na: Viz/Diagn/Report = ✓✓ | ۵ ستاره و ۷۹ دانلود در ماه. قابلیت‌ها محدودتر از ادعاست. | F1 = ۵، F3 = ۴، F14 = ۴ |
| hyperimpute: Multi‑Imp = ✓✓ | چارچوب ایمپیوت تکراری تک‌مقداری است، نه MI با pooling. از ۲۰۲۳ رهاشده است. | F6 = ۲، N6 = ۰ |
| autoimpute: Multi‑Imp = ✓✓ | قابلیت درست است، ولی آخرین انتشار 2023-12 است و رهاشده. | F6 = ۶، N6 = ۰ |
| fancyimpute: Viz = ~ | ویژوال‌سازی ندارد. آخرین انتشار 2021 است. | F1 = ۰ |
| MissMecha: Diagn = ✓✓ | قوتش شبیه‌سازی مکانیزم است، نه تست. | F3 = ۵، F11 = ۸ |
| imputeTS: Multi‑Imp = ~ | فقط ایمپیوت تک‌مقداری دارد. | F6 = ۰ |
| Amelia: Time‑Series = ~ | پشتیبانی ts/cs و polytime دارد. | F9 = ۶ (ارتقا) |
| Stata: Time‑Series = ✓، Report = ✓ | | F9 = ۴، F14 = ۴ |
| naniar و missingno «فعال» فرض شده بودند | naniar آخرین انتشار 2024-03 دارد؛ missingno آخرین انتشار 2023-02. | N6 = ۳ و ۲ |
| در مقایسه‌های قبلی غایب بودند | rbmi (مرجع MNAR)، MissMech، missMDA، JointAI، smcfcs، mitml، Blimp، Mplus، JMP، PyPOTS، ImputeGAP، MIDASpy، Spark، H2O، KNIME و … | اضافه شدند |

### ۸.۲ `COMPETITOR_ANALYSIS.md` (2026-08-08)

**شکاف‌هایی که از آن تاریخ بسته شده‌اند:**
- predictor matrix، visit order، `where` و bounds در `impute_mice`
- بازنویسی Little's MCAR با EM و oracle از R naniar
- Gower داخلی
- قراردادهای typed برای MI (`mi_contracts`)
- `BenchmarkManifest` و fingerprintهای provenance

**هنوز باز هستند:**
- PMM داخل موتور MI (`impute_pmm` جداست)
- Barnard–Rubin و D1 تا D3
- passive terms و blocks
- حساسیت MNAR
- Kalman و STL
- `check_estimator`
- Polars فراتر از `polars_missing_summary` (extra مربوط به `polars` هم در `pyproject.toml` تعریف نشده است)
- مقیاس‌پذیری (`performance.py` سه stub منسوخ است)
- Survey و چندسطحی

اعتبارسنج fixture با R mice وجود دارد، ولی فقط «شاهد» است و parity اثبات‌شده نیست.

---

## ۹. اولویت‌بندی برای بالا بردن امتیاز (اثر بر کارنامه)

| اولویت | کار | ابعاد متأثر | امتیاز فعلی ← هدف واقع‌بینانه | اثر بر رتبه |
|---|---|---|---|---|
| ۱ | delta-adjustment + tipping-point + pattern-mixture روی خروجی MI | F8 | ۱ ← ۶ | اولین ابزار MNAR در Python. رتبه از ۹ به ۲ یا ۳. |
| ۲ | موتور FCS واقعی: PMM داخل حلقه، متد per-variable (logreg/polyreg/polr)، passive، blocks | F5، F6، F13 | ۵/۵/۵ ← ۷/۸/۷ | نزدیک miceforest و Stata |
| ۳ | Barnard–Rubin df + D1/D3 + pooling هر مدل statsmodels | F7 | ۴ ← ۷ | جلوتر از statsmodels MICE |
| ۴ | انتشار 2.0 روی PyPI، مستندات Sphinx، چند vignette، JOSS paper | N1، N2، N3، N4 | ۲/۱/۵/۳ ← ۴/۳/۷/۶ | بزرگ‌ترین اثر بر امتیاز کل |
| ۵ | Kalman/STL با statsmodels state-space | F9 | ۳ ← ۶ | عبور از MATLAB و Amelia |
| ۶ | Polars/Arrow واقعی + اجرای chunked برای خلاصه‌ها | F16، F17 | ۶/۲ ← ۸/۵ | |
| ۷ | softImpute / IterativeSVD و `max_iter` واقعی برای RF | F10 | ۲ ← ۵ | |
| ۸ | تصمیم مجوز: BUSL در برابر MIT/BSD | N8، N2 | ۴ ← ۱۰ | همهٔ رقبای Python آزادند. BUSL مانع پذیرش است. |

شبیه‌سازی امتیاز: اگر اولویت‌های ۱ تا ۵ انجام شوند، میانگین F به حدود ۶٫۰ می‌رسد. این بالاتر از mice (۵٫۱) و SAS است. در امتیاز کل، missingly رتبهٔ ۲ تا ۳ دنیا را می‌گیرد، به شرط اینکه N2 و N4 هم بالا بروند.

---

## ۱۰. محدودیت‌های این ارزیابی

- امتیازها قضاوت کارشناسی کالیبره‌شده‌اند. اندازه‌گیری عددی یا بنچمارک اجرایی پشتشان نیست.
- اعداد بخش R عمدتاً بر اساس دانش از پکیج‌ها هستند. فقط متادیتا (نسخه، تاریخ، دانلود، ستاره) از crandb، cranlogs و GitHub تأیید شده است.
- برای محصولات تجاری متن‌بسته (SAS، SPSS، Stata، JMP، MATLAB، DataRobot)، بُعد N5 یعنی کیفیت مهندسی، حدسی است.
- برای missingly تست‌ها اجرا نشدند، چون محیط Python با pandas روی این ماشین نبود. تعداد ۱۰۳۵ تست شمارش `def test_` است. پوشش ۹۲٫۵۰٪ عدد مستند در `CONVENTIONS.md` است و `CHANGELOG` هنوز ۸۹٫۶۴٪ را نشان می‌دهد.
- ارقام دانلود missingpy و ImputeGAP مشکوک‌اند. امتیازهای mipy-impute از روی README است و تأیید نشده.
- قاعدهٔ پروژه: وضعیت «Parity verified» یا «Better verified» فقط با fixture مستقل عددی قابل ادعاست. این کارنامه هیچ‌کدام از آن دو را اثبات نمی‌کند؛ فقط جایگاه را نشان می‌دهد.

---

پیوست‌ها: شواهد کامل هر امتیاز، با `file:line` برای missingly و URL برای رقبا، در پوشهٔ `competitive-evidence/` است. اسکریپت `build_tables.py` همهٔ جدول‌ها و میانگین‌ها را بازتولید می‌کند.
