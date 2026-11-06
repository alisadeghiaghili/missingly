# نقشهٔ راه رسیدن به سطح بهترین رقیب در هر بُعد

- تاریخ: ۲۰۲۶-۱۱-۰۶
- مبنا: [COMPETITOR_SCORECARD_FA.md](COMPETITOR_SCORECARD_FA.md) و ممیزی کد در [competitive-evidence/missingly_audit.md](competitive-evidence/missingly_audit.md)
- هدف: **هیچ بُعدی از امتیاز بهترین رقیب در همان بُعد پایین‌تر نباشد.** امروز ۲۹ بُعد از ۳۰ زیر بهترین رقیب‌اند و کمترین امتیاز ۱ است.

> **پیش‌نیاز (به‌روزرسانی ۲۰۲۶-۱۱-۰۶):** این نقشهٔ راه بخش «E» از «برنامهٔ جامع کیفیت، اصلاح و هم‌ترازی missingly» است (مخزن مستندات مهندسی، `missingly/docs/PLAN-2026-11-quality-and-remediation.md`). پیش از فاز ۱ این نقشه، سه کار باید تمام شده باشد:
>
> 1. ترمیم build، CI و انتشار (`WP-01` و `WP-02`، معادل قدم ۰.۰). نسخهٔ فعلی با `0.0.0` ساخته می‌شود و setuptools≥77 آن را رد می‌کند.
> 2. اصلاح ۳۸۵ یافتهٔ ممیزی کد نوامبر ۲۰۲۶ در موج‌های ۱ تا ۳. چند یافته مستقیماً ادعای «MI معتبر» را رد می‌کنند: draw پارامتر برای هر imputation انجام نمی‌شود، کدگذاری ترتیبی روی متغیرهای اسمی اعمال می‌شود، و intercept در PMM با شیب نمونه‌گیری‌شده ناسازگار است.
> 3. سبز شدن quality gate سونارکیوب روی کل کد پایتون (کارهای `SQ-*`).
>
> قدم ۰.۲ این نقشه در کارهای `SQ-*` ادغام شده است. قدم ۰.۳ (انتشار 2.x) تا پایان `WP-01` و `WP-41` مسدود است. امتیازهای پیش‌بینی‌شده در بخش ۴ این فرض را دارند که اصلاحات موج‌های ۱ تا ۳ انجام شده باشد.

---

## ۱. هدف عددی هر بُعد

ترتیب جدول بر اساس اندازهٔ فاصله است. فاز اجرای هر بُعد به خاطر وابستگی‌ها تعیین شده است (بخش ۳).

| بُعد | الان | هدف (بهترین رقیب) | مرجع مقایسه | فاصله | فاز |
|---|---|---|---|---|---|
| F8 حساسیت MNAR | 1 | 10 | rbmi | 9 | ۲ |
| N2 پذیرش و جامعه | 1 | 10 | sklearn، mice | 9 | ۸ |
| F18 چندسطحی / Survey | 1 | 9 | mice، mitml، Blimp | 8 | ۶ |
| N1 بلوغ | 2 | 10 | mice، SAS، Stata | 8 | ۰ و ۸ |
| N10 پشتیبانی | 2 | 10 | SAS | 8 | ۸ |
| F9 سری زمانی | 3 | 10 | imputeTS | 7 | ۳ |
| F10 ML پیشرفته | 2 | 9 | PyPOTS | 7 | ۴ |
| F17 مقیاس | 2 | 9 | polars، Spark | 7 | ۵ |
| F7 Pooling | 4 | 10 | mice | 6 | ۱ |
| N4 اعتبارسنجی آماری | 3 | 9 | mice، rbmi، SAS | 6 | ۰ و ۸ |
| N8 مجوز | 4 | 10 | MIT/BSD | 6 | ۰ |
| F6 موتور MI | 5 | 10 | mice | 5 | ۱ |
| N3 مستندات | 5 | 10 | Stata | 5 | ۸ |
| N9 تعامل‌پذیری | 5 | 10 | sklearn | 5 | ۵ و ۸ |
| F11 شبیه‌سازی | 5 | 9 | mice::ampute | 4 | ۷ |
| F15 Pipeline | 6 | 10 | sklearn | 4 | ۰ و ۷ |
| F16 دیتافریم | 6 | 10 | pandas، polars | 4 | ۵ |
| F2 خلاصه و الگو | 7 | 10 | naniar | 3 | ۷ |
| F3 تست مکانیزم | 6 | 9 | MissMech | 3 | ۷ |
| F5 دادهٔ ترکیبی | 5 | 8 | mice، missMDA | 3 | ۱ |
| F13 تشخیص ایمپیوت | 5 | 8 | mice | 3 | ۱ |
| F1 ویژوال‌سازی | 8 | 10 | naniar | 2 | ۷ |
| F12 ارزیابی ایمپیوترها | 6 | 8 | ImputeGAP | 2 | ۷ |
| F14 گزارش | 6 | 8 | finalfit | 2 | ۷ |
| N5 کیفیت مهندسی | 7 | 9 | sklearn، rbmi | 2 | ۰ |
| N6 نگهداری | 8 | 10 | sklearn، VIM | 2 | ۰ و ۸ |
| N7 سهولت استفاده | 8 | 10 | missingno، JASP | 2 | ۸ |
| F4 ایمپیوت تک‌مقداری | 7 | 8 | VIM | 1 | ۱ |
| F19 چندزبانگی | 7 | 8 | SPSS، JASP | 1 | ۷ |
| F20 Provenance | 7 | 7 (خودمان اول هستیم) | missingly | 0 | ۷ (حفظ و ارتقا به ۸) |

---

## ۲. قاعدهٔ «انجام شد» برای هر قدم

یک بُعد فقط وقتی امتیاز جدید می‌گیرد که همهٔ این شرط‌ها در **همان PR** برقرار باشند. این همان سیاست `COMPETITOR_ANALYSIS.md` و قاعدهٔ «doc-vs-code» پروژه است.

1. **کد** با docstring کامل و مثال اجرایی برای هر تابع عمومی.
2. **تست واحد و property-based** (hypothesis). پوشش کل از ۹۲٫۵٪ کمتر نشود.
3. **Oracle مستقل:** خروجی روی دیتاست مرجع با ابزار مرجع آن بُعد مقایسه می‌شود (جدول بخش ۱) و تلورانس در `BenchmarkManifest` ثبت می‌شود. شرط پیشنهادی در CI: `.github/workflows` + `tests/oracles/<tool>/`.
4. **مستندات:** README (en/fa/de)، صفحهٔ docs و یک vignette.
5. **ورودی CHANGELOG** تاریخ‌دار.
6. **به‌روزرسانی کارنامه:** امتیاز جدید در `competitive-evidence/build_tables.py` و `COMPETITOR_SCORECARD_FA.md`، با ارجاع به fixture.

بدون oracle، امتیاز بالا نمی‌رود؛ حتی اگر قابلیت پیاده شده باشد.

---

## ۳. فازها و قدم‌ها

اندازهٔ هر قدم: S یعنی چند روز، M یعنی ۱ تا ۲ هفته، L یعنی ۳ تا ۶ هفته، XL یعنی بیش از ۶ هفته (برای یک نگهدارنده).

### فاز ۰ — زیرساخت اعتبار (پیش‌نیاز همهٔ فازها)

| # | کار | فایل‌ها | معیار پذیرش | بُعد ← امتیاز | اندازه |
|---|---|---|---|---|---|
| 0.0 | **تعمیر فوری `pyproject.toml`:** کامیت `81528c7` همهٔ این‌ها را حذف کرد: extras `docs`/`test`/`quality`/`polars`/`upset`، منبع dynamic version، تنظیمات ruff و coverage، و classifierهای 3.13/3.14. CI روی `main` از ۲۰۲۶-۰۹-۰۵ قرمز است. | `pyproject.toml`، `tests/test_packaging_contract.py` | CI سبز، `python -m build` و `twine check` سالم | پیش‌نیاز همه | S |
| 0.1 | Oracle harness عمومی: اجرای R در CI برای mice، naniar، rbmi، imputeTS، MissMech، missForest و softImpute. Fixtureها با SHA-256 منجمد شوند. | `tests/oracles/`، workflow جدید، `benchmark.py` | هر oracle یک dataset، یک نسخهٔ ابزار و یک تلورانس دارد. شکست oracle، PR را بلاک می‌کند. | N4: 3 ← 5 | M |
| 0.2 | `mypy --strict` روی کل پکیج (الان فقط `benchmark.py` در `ci.yml:57`). بالا بردن gate پوشش از ۸۰ به ۹۲. اضافه کردن mutation testing (mutmut) روی `mi.py` و `diagnostics.py`. | `pyproject.toml`، `ci.yml` | CI سبز با strict | N5: 7 ← 9 | M |
| 0.3 | فرایند انتشار: انتشار 2.x روی PyPI (الان فقط 1.0.0 آنجاست)، ورودی تاریخ‌دار CHANGELOG، سیاست SemVer و deprecation مکتوب، انتشار ماهانه. | `CHANGELOG.md`، `CONTRIBUTING.md` | نسخه روی PyPI با tag متناظر | N1: 2 ← 3، N6: 8 ← 9 | S |
| 0.4 | **تصمیم مجوز (تصمیم شما):** BUSL-1.1 → MIT/BSD/Apache-2.0. همهٔ رقبای Python مجوز آزاد دارند و BUSL مانع پذیرش سازمانی است. | `LICENSE`، `pyproject.toml` | — | N8: 4 ← 10 (اگر BUSL بماند، ۴ می‌ماند) | S |
| 0.5 | شروع سازگاری `check_estimator`: پذیرش numpy، `set_output`، و تست `parametrize_with_checks` روی حالت‌های امن. | `transformer.py` | `check_estimator` برای mean/median/knn/pmm پاس شود | F15: 6 ← 7 | M |

پس از فاز ۰: کل ۵٫۲۸، غیرامکاناتی ۵٫۷۰.

### فاز ۱ — موتور FCS هم‌سطح mice (F5، F6، F7، F13، F4)

این بزرگ‌ترین شکاف الگوریتمی است و پیش‌نیاز فازهای ۲ و ۶.

| # | کار | فایل‌ها | معیار پذیرش (oracle) | بُعد ← امتیاز | اندازه |
|---|---|---|---|---|---|
| 1.1 | **متد per-variable داخل حلقهٔ FCS:** `method={"bmi": "pmm", "hyp": "logreg", "chl": "norm", ...}` با متدهای `pmm`، `norm`، `norm.boot`، `logreg`، `polyreg`، `polr`، `cart`، `rf` و `midastouch`. کدگذاری ordinal برای categoricalها (`impute.py:1313-1345`) حذف شود. | `impute.py` (`impute_mice`، `_run_mice_with_history` حدود خط ۶۹۰)، `mi_contracts.py` (`FCSKernel`) | روی `nhanes`، `nhanes2` و `boys`: میانگین، SD و جدول فراوانی ایمپیوت‌ها در تلورانس mice. تخمین pooled رگرسیون در ±۰٫۰۵ SE. | F5: 5 ← 8، F6: 5 ← 8 | XL |
| 1.2 | **PMM داخل موتور:** match type 0/1/2، `donors=5` و ridge. `impute_pmm` (`impute.py:1437`) روی همان kernel سوار شود. | `impute.py` | oracle `mice(method="pmm")` | F4: 7 ← 8 | M |
| 1.3 | **Blocks، passive imputation و post-processing:** مثلاً `passive={"bmi": lambda d: d.wgt / (d.hgt/100)**2}`، squeeze و bounds به ازای هر متغیر، و `post` (پایهٔ delta در فاز ۲). | `impute.py`، `mi_contracts.py` | oracle `mice` با `blocks`، passive `~I()` و `post` | F6: 8 ← 10 | L |
| 1.4 | **شیء MIData کامل:** معادل `complete(action="long"/"broad")` و `with()`/`pool()` و ذخیره و بازخوانی. | `mi_contracts.py`، `mi_workflow.py` | گردش کامل round-trip | (F6) | M |
| 1.5 | **Pooling کامل:** Barnard–Rubin با `dfcom` (`mi.py:179` و `338-343`)، آزمون‌های چندمتغیره D1/D2/D3، pooling هر نتیجهٔ statsmodels (OLS، GLM، Logit، MixedLM، GEE) و Cox با lifelines به‌صورت اختیاری. جدول خروجی با `estimate`، `se`، `df`، `CI`، `p`، `riv`، `lambda` و `fmi`. | `mi.py`، `mi_workflow.py` | oracle `mice::pool` و `mitml::testModels`، تطبیق تا ۱e-6 | F7: 4 ← 10 | L |
| 1.6 | **تشخیص‌ها:** trace میانگین و SD برای هر زنجیره، `stripplot`، `densityplot`، `xyplot`، overimputation و posterior predictive check. | `diagnostics.py` (`mice_convergence` در خط ۱۱۱۸)، `visualisation/` | معادل نمودارهای mice و ggmice | F13: 5 ← 8 | M |

پس از فاز ۱: میانگین F ۵٫۹۰، کل ۵٫۸۲.

### فاز ۲ — تحلیل حساسیت MNAR (کمترین امتیاز: F8 = 1)

در کل اکوسیستم Python هیچ رقیبی بالاتر از ۲ ندارد. این فاز missingly را تنها گزینهٔ Python می‌کند.

| # | کار | معیار پذیرش (oracle) | اندازه |
|---|---|---|---|
| 2.1 | **Delta adjustment:** `delta_adjust(midata, {"y": -2.0})` روی متغیر و pattern، با استفاده از `post` فاز ۱.۳. | oracle `mice(post=...)` روی `nhanes`. نمونهٔ مستند در FIMD §9.2. | M |
| 2.2 | **Tipping-point analysis:** شبکهٔ delta، تخمین pooled و p برای هر delta، نقطهٔ واژگونی و نمودار آن. | تطبیق با خروجی دستی mice و مثال‌های rbmi | M |
| 2.3 | **Pattern-mixture با identifying restrictions:** CCMV، NCMV و ACMV برای دادهٔ monotone. | مثال‌های مقالهٔ Molenberghs و Kenward، با محاسبهٔ مستقل | L |
| 2.4 | **Reference-based** (هم‌سطح rbmi): J2R، CR، CIR، LMCF، MAR، شامل MMRM، conditional mean imputation با jackknife/bootstrap و approximate Bayesian. | oracle `rbmi` روی دیتاست `antidepressant_data`: تخمین و SE در تلورانس | XL |
| 2.5 | **Selection model** (Heckman / Diggle–Kenward) به‌صورت اختیاری. | نتیجه‌های منتشرشده | L |

F8: 1 ← 10 (پس از 2.4). کل ۶٫۰۹.

### فاز ۳ — سری زمانی هم‌سطح imputeTS (F9: 3 ← 10)

| # | کار | معیار پذیرش | اندازه |
|---|---|---|---|
| 3.1 | `impute_ts` با استراتژی‌های `interpolation` (linear، spline، stine)، `ma` (simple، linear، exponential)، `locf`/`nocb` با `maxgap`، `mean`/`median`/`mode` و `random`. | oracle `imputeTS::na_*` روی `tsAirgap` و `tsNH4` | M |
| 3.2 | `kalman`: هم StructTS هم auto-ARIMA از طریق `statsmodels.tsa` (`UnobservedComponents`، `SARIMAX`) با smoothing. | oracle `na_kalman` (تلورانس RMSE) | L |
| 3.3 | `seadec` و `seasplit` با STL. | oracle `na_seadec` و `na_seasplit` | M |
| 3.4 | `statsNA` و نمودارهای `ggplot_na_distribution`، `gapsize`، `intervals` و `imputations`. `timeseries.py:345` گسترش پیدا کند. | معادل بصری و عددی | M |
| 3.5 | سری زمانی چندمتغیره: backend اختیاری PyPOTS (extra `ts-deep`). | بنچمارک روی دیتاست‌های PyPOTS | M |

پس از فاز ۳: کل ۶٫۳۰.

### فاز ۴ — ML پیشرفته و matrix completion (F10: 2 ← 9)

| # | کار | معیار پذیرش | اندازه |
|---|---|---|---|
| 4.1 | `softimpute` و `iterative_svd`، با الگوریتم Mazumder–Hastie–Tibshirani. | oracle R `softImpute` | M |
| 4.2 | **missForest واقعی:** معیار توقف OOB و همگرایی؛ الان `max_iter` پیش‌فرض ۱ است (`impute.py:2057-2066`). | oracle R `missForest` (NRMSE و PFC) | M |
| 4.3 | MI بر پایهٔ XGBoost یا LightGBM، شبیه mixgb و miceforest، به‌صورت extra اختیاری. | مقایسه با miceforest و mixgb | L |
| 4.4 | extra `deep` (PyTorch): GAIN، MIWAE، not-MIWAE و یک مدل diffusion. می‌تواند adapter روی پیاده‌سازی‌های موجود باشد. | بنچمارک هم‌سطح hyperimpute و PyPOTS | XL |

پس از فاز ۴: کل ۶٫۵۱.

### فاز ۵ — مقیاس و دیتافریم (F16: 6 ← 10، F17: 2 ← 9، بخشی از N9)

| # | کار | معیار پذیرش | اندازه |
|---|---|---|---|
| 5.1 | **Polars namespace کامل** `df.miss` (با `register_dataframe_namespace`) برای خلاصه‌ها، shadow، ایمپیوت پایه و نمودارها. extra `polars` در `pyproject.toml` تعریف شود (الان نیست). | خروجی polars و pandas برابر | L |
| 5.2 | ورود و خروج Arrow و DataFrame Interchange. پذیرش LazyFrame با streaming برای خلاصه‌ها. | تست هم‌ارزی | M |
| 5.3 | آماره‌های بسندهٔ chunked برای خلاصه‌ها و Little's MCAR (EM روی chunkها). اجرای موازی زنجیره‌های MI با joblib. KNN تقریبی به‌صورت اختیاری. stubهای `performance.py` حذف یا واقعی شوند. | خروجی chunked با خروجی in-memory دقیقاً برابر | L |
| 5.4 | بنچمارک مقیاس در CI: ۱ میلیون ردیف در هر PR و ۱۰ میلیون ردیف هفتگی، با ثبت زمان، RSS، نسخه و سخت‌افزار. | گزارش منتشرشده | M |
| 5.5 | adapter خلاصه برای Dask و Spark. | تست هم‌ارزی | M |

پس از فاز ۵: کل ۶٫۸۴.

### فاز ۶ — چندسطحی، Survey و طولی (F18: 1 ← 9)

پیش‌نیاز: فاز ۱.

| # | کار | معیار پذیرش | اندازه |
|---|---|---|---|
| 6.1 | متدهای دوسطحی در FCS: `2l.norm`، `2l.pmm`، `2l.bin` و `2lonly.mean`، با متغیر cluster در predictor matrix (کد ۲-۲). | oracle `mice` و `miceadds` روی `popmis` | XL |
| 6.2 | Joint model چندسطحی به سبک jomo/pan به‌صورت اختیاری، و pooling چندسطحی با MixedLM. | oracle `jomo` و `mitml` | L |
| 6.3 | Survey: وزن‌ها، strata و PSU در ایمپیوت (hot-deck وزن‌دار به سبک SURVEYIMPUTE) و در pooling. | مقایسه با مثال‌های SAS SURVEYIMPUTE | L |
| 6.4 | ابزارهای پنل: تبدیل wide↔long و ایمپیوت آگاه از زمان در طولی. | تست‌ها | M |

پس از فاز ۶: میانگین F ۸٫۰۰، کل ۷٫۰۸.

### فاز ۷ — تکمیل اکتشاف و گزارش (F1، F2، F3، F11، F12، F14، F15، F19، F20)

| # | کار | معیار پذیرش | بُعد ← امتیاز | اندازه |
|---|---|---|---|---|
| 7.1 | نمودارهای VIM: `marginplot`، aggregation plot، spine plot و parallel coordinates. نمودارهای naniar: `geom_miss_point`، `gg_miss_fct`، `gg_miss_span` و `gg_miss_case_cumsum`. تست regression پیکسلی. | معادل بصری مستند | F1: 8 ← 10 | M |
| 7.2 | معادل‌های naniar: `miss_var_run`، `miss_var_span`، `miss_case_table`، `miss_var_table`، جدول `md.pattern`، `group_by` در همه‌جا، و `replace_with_na_*`. | oracle naniar با ستون‌ها و مقادیر یکسان | F2: 7 ← 10 | M |
| 7.3 | تست‌های MCAR: Jamshidian–Jalal (نرمال و ناپارامتری)، Hawkins، t-test جداگانه برای هر متغیر به سبک SPSS MVA، و آزمون MCAR برای دادهٔ ترکیبی. | oracle `MissMech::TestMCARNormality` | F3: 6 ← 9 | L |
| 7.4 | `ampute` سازگار با mice: `patterns`، `freq`، `weights`، `type` (LEFT، RIGHT، MID، TAIL)، کالیبراسیون نسبت، و خانواده‌های MNAR به سبک MissMecha. `simulate.py` بازنویسی شود. | oracle `mice::ampute` (نسبت و الگوها) | F11: 5 ← 9 | M |
| 7.5 | Harness بنچمارک: مکانیزم × نرخ × دیتاست × روش، و گزارش leaderboard. روی `compare.py` ساخته شود. | هم‌سطح ImputeGAP | F12: 6 ← 8 | M |
| 7.6 | گزارش PDF (extra، با weasyprint) و جدول‌های انتشار (LaTeX، Word و markdown) به سبک finalfit و gtsummary. | خروجی معتبر | F14: 6 ← 8 | M |
| 7.7 | `check_estimator` کامل، پشتیبانی sparse، `MissingIndicator`، و `MultipleImputationEstimator` که MI را داخل Pipeline اجرا و پیش‌بینی‌ها را pool کند. | `parametrize_with_checks` سبز | F15: 7 ← 10 | L |
| 7.8 | پیام‌های خطا و همهٔ نمودارها چندزبانه با RTL، به‌علاوهٔ یک زبان دیگر (مثلاً عربی). | تست RTL | F19: 7 ← 8 | S |
| 7.9 | Audit trail برای هر سلول (کدام سلول، با کدام متد و کدام seed) و replay از manifest. | تست بازتولید bit-for-bit | F20: 7 ← 8 | M |

پس از فاز ۷: میانگین F ۹٫۰۵، یعنی **همهٔ ۲۰ بُعد امکانات به سطح بهترین رقیب یا بالاتر می‌رسند.** کل ۷٫۷۱.

### فاز ۸ — ابعاد غیرامکاناتی

| # | کار | بُعد ← امتیاز |
|---|---|---|
| 8.1 | سایت مستندات Sphinx با گالری، vignette برای هر workflow، و کتاب آنلاین «Flexible Imputation of Missing Data with Python» به سبک FIMD، به فارسی و انگلیسی. | N3: 5 ← 9 |
| 8.2 | مقالهٔ JOSS، به‌علاوهٔ مقالهٔ اعتبارسنجی با شبیه‌سازی پوشش ۹۵٪ CI و bias در برابر mice و rbmi. | N4: 5 ← 9 |
| 8.3 | API یکنواخت: نوع بازگشتی `impute_mice` الان با flagها تغییر می‌کند. پیش‌فرض‌های خوب و `df.miss.report()` یک‌خطی. | N7: 8 ← 10 |
| 8.4 | پل‌های ارتباطی: rpy2 برای تبدیل `mids` ↔ MIData، به‌علاوهٔ statsmodels، lifelines، mlflow و Arrow. | N9: 5 ← 10 |
| 8.5 | انتشار ماهانه و پاسخ به issue در کمتر از ۷ روز. | N6: 9 ← 10 |
| 8.6 | جامعه: ارائه در PyData و useR، notebookهای Kaggle، پست‌های آموزشی، جذب ۲ تا ۳ هم‌نگهدارنده، و سند governance. | N2: 1 ← 4، N10: 2 ← 5، N1: 3 ← 5 |

پس از فاز ۸: میانگین F ۹٫۰۵، میانگین N ۸٫۱۰، کل ۸٫۶۷. برای مقایسه: mice ۶٫۵۰ و SAS ۵٫۹۱. یعنی **رتبهٔ ۱ از ۶۷.**

---

## ۴. خلاصهٔ پیش‌بینی امتیاز

| پس از فاز | میانگین F | میانگین N | امتیاز کل | ابعاد زیر بهترین رقیب |
|---|---|---|---|---|
| امروز | 4.95 | 4.50 | 4.77 | ۲۹ از ۳۰ |
| ۰ زیرساخت | 5.00 | 5.70 | 5.28 | ۲۷ |
| ۱ موتور FCS | 5.90 | 5.70 | 5.82 | ۲۲ |
| ۲ MNAR | 6.35 | 5.70 | 6.09 | ۲۱ |
| ۳ سری زمانی | 6.70 | 5.70 | 6.30 | ۲۰ |
| ۴ ML پیشرفته | 7.05 | 5.70 | 6.51 | ۱۹ |
| ۵ مقیاس | 7.60 | 5.70 | 6.84 | ۱۷ |
| ۶ چندسطحی | 8.00 | 5.70 | 7.08 | ۱۶ |
| ۷ اکتشاف و گزارش | 9.05 | 5.70 | 7.71 | ۸ (همه N) |
| ۸ غیرامکاناتی | 9.05 | 8.10 | **8.67** | ۴ (N1، N2، N3، N10) |

---

## ۵. چه چیزی را نمی‌شود با کد بست

چهار بُعد بعد از همهٔ فازها هنوز زیر بهترین رقیب می‌مانند، چون به زمان و جامعه وابسته‌اند:

- **N1 (بلوغ، هدف ۱۰):** mice و SAS بیش از ۲۰ سال سابقه دارند. هر سال API پایدار حدود ۱ امتیاز می‌آورد.
- **N2 (پذیرش، هدف ۱۰):** sklearn ماهانه ۱۸۴ میلیون دانلود دارد. شاخص واقع‌بینانه برای ۲ سال آینده حدود ۱۰ هزار دانلود در ماه است، معادل امتیاز ۴ تا ۵.
- **N10 (پشتیبانی، هدف ۱۰):** امتیاز ۱۰ پشتیبانی تجاری SAS است. با هم‌نگهدارنده‌ها و SLA، حدود ۵ تا ۶ ممکن است.
- **N3 (مستندات، هدف ۱۰):** امتیاز ۱۰ دفترچهٔ Stata MI است. کتاب آنلاین ۸.۱ به ۹ می‌رسد.

بنابراین «کمترین امتیاز = بهترین رقیب» برای ۲۶ بُعد با این نقشه قابل دستیابی است. برای این ۴ بُعد، هدف درست «بهترین در Python» است، نه «بهترین در دنیا».

---

## ۶. ترتیب پیشنهادی شروع

1. **0.4 تصمیم مجوز.** تصمیم شماست و روی جذب مشارکت‌کننده اثر مستقیم دارد.
2. **0.1 Oracle harness.** بدون آن هیچ امتیازی قابل ادعا نیست.
3. **1.1 و 1.2.** متد per-variable و PMM، چون پیش‌نیاز MNAR و چندسطحی‌اند.
4. **2.1 و 2.2.** delta و tipping-point، که سریع‌ترین راه برای خارج شدن F8 از ۱ هستند (۱ ← ۶) و جای خالی Python را پر می‌کنند.

بعد از این‌ها، فازها به همان ترتیب جدول.
