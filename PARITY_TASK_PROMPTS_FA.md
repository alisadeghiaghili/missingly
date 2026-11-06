# دستور کارهای مرحله‌به‌مرحله — فاز ۰ و فاز ۱

- مبنا: [PARITY_ROADMAP_FA.md](PARITY_ROADMAP_FA.md) (فاز ۰ زیرساخت، فاز ۱ موتور FCS هم‌سطح mice)
- تاریخ: ۲۰۲۶-۱۱-۰۶ — وضعیت `main` در کامیت `ccf9853`

## نحوهٔ استفاده

- دستور کارها را **به ترتیب**، هر کدام در **یک جلسهٔ کاری تازه** و در پوشهٔ `D:\missingly` اجرا کنید. ستون «سطح»: `L1` یعنی کار ساده و مکانیکی، `L2` یعنی نیازمند قضاوت آماری یا طراحی.
- هر دستور کار یک برنچ، یک PR و یک push دارد. دستور کار بعدی را فقط **بعد از merge شدن PR قبلی** بدهید، چون هر دستور کار از `main` به‌روز شروع می‌کند.
- دستور کار‌هایی که با «الف/ب» شکسته شده‌اند عمداً کوچک نگه داشته شده‌اند تا در یک جلسهٔ کاری تمام شوند.
- **بلوک «قواعد مشترک» در بخش ۰ را بالای هر دستور کار paste کنید.** خود دستور کار‌ها هم به آن ارجاع می‌دهند، ولی paste کردن مطمئن‌تر است.
- **جایگاه در برنامهٔ کلی (۲۰۲۶-۱۱-۰۶):** این دستور کارها زیرمجموعهٔ «برنامهٔ جامع کیفیت، اصلاح و هم‌ترازی missingly» در مخزن مستندات مهندسی هستند (`missingly/docs/PLAN-2026-11-quality-and-remediation.md`). ترتیب آن برنامه بر جدول زیر مقدم است:
  - `PAR-0.0` همان بسته‌های `WP-01` و `WP-02` است و یک‌بار انجام می‌شود.
  - `PAR-0.2a/b` با کارهای `SQ-*` (quality gate سونارکیوب) ادغام شده‌اند.
  - `PAR-0.5a/b` بعد از `WP-24` اجرا می‌شود.
  - فاز ۱ (`PAR-1.*`) فقط بعد از موج‌های ۱ تا ۳ اصلاحات شروع می‌شود. تست‌های رگرسیون `WP-11`، `WP-21`، `WP-22`، `WP-23` و `WP-25` جزو معیار پذیرش `PAR-1.*` هستند و نباید حذف یا ضعیف شوند.
  - هر PR علاوه بر بند ۴ قواعد مشترک، باید quality gate سونارکیوب را هم سبز کند.

| ترتیب | شناسه | سطح | موضوع | وابسته به |
|---|---|---|---|---|
| 1 | PAR-0.0 | L1 | **تعمیر `pyproject.toml`** (CI روی `main` از ۵ سپتامبر قرمز است) | — |
| 2 | PAR-0.1 | L2 | قالب عمومی Oracle و قرارداد fixtureها | 0.0 |
| 3 | PAR-0.2a | L1 | mypy روی کل پکیج (غیر strict) | 0.0 |
| 4 | PAR-0.2b | L1 | mypy strict روی ماژول‌های آماری + gate پوشش ۹۲ | 0.2a |
| 5 | PAR-0.3 | L1 | آماده‌سازی انتشار 2.x (بدون publish) | 0.0 |
| 6 | PAR-0.4 | L1 | تغییر مجوز — **فقط اگر تصمیم گرفتید** | — |
| 7 | PAR-0.5a | L2 | `MissinglyImputer`: ورودی numpy، `set_output`، `feature_names_in_` | 0.0 |
| 8 | PAR-0.5b | L2 | `parametrize_with_checks` برای استراتژی‌های امن | 0.5a |
| 9 | PAR-1.1a | L2 | هستهٔ FCS با متد per-variable: `norm`، `norm.boot`، `pmm` | 0.1 |
| 10 | PAR-1.1b | L2 | متدهای دسته‌ای داخل FCS: `logreg`، `polyreg`، `polr` | 1.1a |
| 11 | PAR-1.1c | L2 | متدهای درختی: `cart`، `rf`، `midastouch` | 1.1a |
| 12 | PAR-1.2 | L2 | PMM کامل (match type، donors) و یکپارچه‌سازی `impute_pmm` | 1.1a |
| 13 | PAR-1.3a | L2 | `blocks` و `post` (پایهٔ delta-adjustment در فاز ۲) | 1.1b |
| 14 | PAR-1.3b | L2 | passive imputation | 1.3a |
| 15 | PAR-1.4 | L2 | `MIData` کامل: `complete(long/broad)`، `with`/`pool`، ذخیره و بارگذاری | 1.1a |
| 16 | PAR-1.5a | L2 | Pooling: Barnard–Rubin با `dfcom` و جدول pooled کامل | 1.4 |
| 17 | PAR-1.5b | L2 | Pooling نتیجه‌های statsmodels (OLS، GLM، Logit، MixedLM، GEE) | 1.5a |
| 18 | PAR-1.5c | L2 | آزمون‌های چندمتغیره D1/D2/D3 | 1.5b |
| 19 | PAR-1.6 | L2 | نمودارهای تشخیصی MI | 1.4 |

---

## ۰. قواعد مشترک (بالای هر دستور کار paste شود)

```text
قواعد مشترک برای این task در ریپوی D:\missingly (GitHub: alisadeghiaghili/missingly):

1. شروع:
   - استاندارد مهندسی Consilient (`ENGINEERING-STANDARDS.md` و `ROADMAP.md` اکوسیستم) را رعایت کن.
   - CONVENTIONS.md (بخش "Non-negotiable execution rules")، TESTING.md و CONTRIBUTING.md را بخوان.
   - `git switch main && git pull --ff-only && git switch -c <نام برنچ داده‌شده>`

2. TDD اجباری:
   - اول تست‌های شکست‌خورده را بنویس و جداگانه commit کن (`test: ...`). بعد پیاده‌سازی (`feat:` یا `fix:`).
   - تست smoke هرگز شاهد درستی آماری نیست. برای هر ادعای آماری یا یک oracle عددی مستقل
     (خروجی R یا محاسبهٔ دستی دقیق) لازم است، یا ادعا نباید نوشته شود.

3. مستندات، هم‌زمان با کد:
   - هر تابع یا کلاس عمومی جدید یا تغییرکرده docstring کامل numpydoc دارد: خلاصه، Parameters (نوع، پیش‌فرض، دامنهٔ مجاز)،
     Returns، Raises، Notes (فرمول یا مرجع مقاله)، See Also، و Examples اجرایی با doctest
     (حداقل دو مثال: ساده و واقع‌گرایانه).
   - CHANGELOG.md: ورودی زیر [Unreleased] در بخش درست (Added/Changed/Fixed).
   - README.md (en): اگر API عمومی تغییر کرده. README_FA.md و README_DE.md هم همان تغییر.
   - docs/source: صفحه یا بخش مربوط به‌روز شود.
   - CONVENTIONS.md: وضعیت آیتم roadmap. COMPETITOR_ANALYSIS.md: فقط با شاهد.
   - هیچ ادعایی در README یا docs ننویس که کد و تست پشتش نیست (قاعدهٔ doc-vs-code).

4. قبل از push، همه باید سبز باشند:
   - ruff check missingly tests
   - mypy (طبق تنظیم فعلی CI)
   - pytest tests/ با پوشش، بدون کمتر شدن از عدد فعلی
   - python -m doctest روی فایل‌های تغییرکرده

5. پایان:
   - commitها با Conventional Commits (`test:`، `feat:`، `fix:`، `docs:`، `ci:`؛ فقط تغییرات CI از نوع `ci:` باشند).
   - `git push -u origin <branch>`
   - `gh pr create` با: هدف، تغییرات، شواهد (اعداد oracle و تعداد تست)، شکاف‌های باقی‌مانده
     نسبت به رقیب نام‌برده، و چک‌لیست مستندات.
   - merge نکن. خلاصهٔ نهایی را با لینک PR گزارش کن.

6. بی‌نشانی ابزار:
   - در فایل‌ها، پیام‌های commit، نام برنچ و متن PR هیچ اشاره‌ای به ابزار خودکار، مدل یا دستیار نباشد؛
     هیچ trailer از نوع Co-Authored-By یا «Generated with» اضافه نشود.

7. محدوده:
   - فقط همین task. تغییر نامرتبط ممنوع. oracle provenance را با اصلاح correctness در یک PR مخلوط نکن.
   - اگر کاری خارج از محدوده لازم شد، در گزارش نهایی بنویس؛ انجامش نده.
```

---

## فاز ۰ — زیرساخت

### PAR-0.0 — تعمیر `pyproject.toml` (اولویت فوری)

```text
[قواعد مشترک را اینجا paste کن]

برنچ: fix/pyproject-regression

مشکل (تأییدشده): کامیت 81528c7 ("chore(pyproject): update license metadata to BUSL-1.1 for PyPI")
کل pyproject.toml را بازنویسی کرد و این‌ها حذف شدند:
- extras: docs، test، quality، polars، upset. CI (ci.yml) هنوز `.[test,all]`، `.[test,polars]` و `.[quality]`
  را نصب می‌کند، و از همان کامیت CI روی main شکست می‌خورد.
- [tool.setuptools.dynamic] version = {attr = "missingly._version.__version__"}، در حالی که
  `dynamic = ["version"]` هنوز اعلام شده است.
- classifierهای Python 3.13 و 3.14، در حالی که CI روی 3.9 تا 3.14 تست می‌کند.
- [tool.ruff]، [tool.ruff.lint] (select = ["E9", "F", "BLE001"])، [tool.coverage.run] و [tool.coverage.report].
- کف نسخهٔ وابستگی‌ها پایین آورده شد: مثلاً pandas>=2.0.0 به pandas>=1.3.0.
نسخهٔ سالم قبلی: `git show 78e6aa0:pyproject.toml`.

کار:
1. اول یک تست قرارداد بنویس: `tests/test_packaging_contract.py`. این تست pyproject.toml را با tomllib/tomli
   می‌خواند و بررسی می‌کند که:
   - هر extra که در .github/workflows/*.yml با `.[...]` نصب می‌شود در pyproject تعریف شده باشد؛
   - dynamic version منبع داشته باشد؛
   - classifierهای Python با matrix CI یکی باشند.
   این تست الان باید شکست بخورد. commit کن.
2. pyproject.toml را اصلاح کن: همهٔ بخش‌های حذف‌شده را از 78e6aa0 برگردان. license = "BUSL-1.1"، license-files
   و classifierهای جدید Intended Audience و Topic را نگه دار.
   کف وابستگی‌ها: نسخهٔ 78e6aa0 را برگردان (pandas>=2.0.0 و غیره)، مگر اینکه CHANGELOG دلیل مستندی برای پایین آوردن داشته باشد.
   مشکل `License :: Other/Proprietary License` همراه با فیلد `license` (طبق PEP 639 classifier مجوز نباید
   هم‌زمان با license expression باشد) را بررسی کن و اگر `python -m build` یا `twine check` خطا داد، classifier را حذف کن.
3. `python -m build` و `twine check dist/*` را اجرا کن. pip install -e '.[test,all]'، '.[quality]' و '.[polars]'
   باید بدون هشدار "does not provide the extra" نصب شوند.
4. pytest کامل. تعداد تست‌ها و پوشش را گزارش کن.
5. CHANGELOG، زیر Fixed: "Restored optional extras, dynamic version source and tool configuration lost in 81528c7".
6. push و PR. در توضیح PR لینک run شکست‌خوردهٔ قبلی را بگذار (`gh run list --branch main`).
```

### PAR-0.1 — قالب عمومی Oracle

```text
[قواعد مشترک را اینجا paste کن]

برنچ: ci/oracle-fixture-contract

زمینه: الان دو oracle مستقل وجود دارد:
- tests/fixtures/little_mcar (naniar mcar_test)
- tests/fixtures/mice_numeric_fcs (mice)
که هر کدام generate_*.R، validate_*.py، benchmark_manifest.json و یک CSV دارند، و job جداگانه در
.github/workflows/r-oracle.yml. فازهای بعدی به حدود ۱۵ oracle دیگر نیاز دارند (mice pool، mitml، rbmi، imputeTS،
MissMech، softImpute، missForest و ...). باید یک قالب یکسان و قابل‌تکرار داشته باشیم.

کار:
1. تست قرارداد `tests/test_oracle_fixture_contract.py`. برای هر پوشه در tests/fixtures/ که فایل
   benchmark_manifest.json دارد، بررسی کند که:
   - README.md، یک generate_*.R (یا generate_*.py برای oracleهای Python)، یک validate_*.py و dataset با SHA-256
     ثبت‌شده در manifest وجود داشته باشد؛
   - SHA-256 واقعی فایل dataset با manifest برابر باشد؛
   - manifest نسخهٔ دقیق ابزار مرجع، seed، تلورانس‌ها (abs/rel) و estimand را داشته باشد
     (از اسکیمای BenchmarkManifest در missingly/benchmark.py استفاده کن، شمای جدید نساز).
   این تست باید برای دو fixture فعلی سبز باشد. اگر نیست، fixture را درست کن، نه تست را.
2. یک راهنمای `tests/fixtures/ORACLE_GUIDE.md`: گام‌به‌گام چطور یک oracle جدید اضافه شود، شامل الگوی generate،
   validate، manifest، job در r-oracle.yml، و قاعدهٔ «بدون oracle، امتیاز کارنامه بالا نمی‌رود».
3. r-oracle.yml را طوری بازآرایی کن که اضافه کردن oracle جدید فقط یک ورودی matrix باشد: پوشه، بستهٔ R با نسخهٔ
   قفل‌شده، و نام اسکریپت‌ها. رفتار دو job فعلی دقیقاً حفظ شود و artifactها همان نام را داشته باشند.
4. یک اسکریپت کمکی `tests/fixtures/new_oracle.py <name> <r-package>@<version>` که اسکلت پوشه را بسازد.
5. مستندات: TESTING.md (بخش Oracles) و CHANGELOG.
6. push و PR. CI و r-oracle هر دو باید سبز باشند.
```

### PAR-0.2a — mypy روی کل پکیج

```text
[قواعد مشترک را اینجا paste کن]

برنچ: ci/mypy-whole-package

زمینه: CI فقط `mypy missingly/benchmark.py --follow-imports=skip` را اجرا می‌کند (ci.yml، job quality) و
در pyproject تنظیم [tool.mypy] وجود ندارد. py.typed منتشر می‌شود، پس کاربران انتظار type درست دارند.

کار:
1. در pyproject.toml بخش [tool.mypy] بساز: python_version = "3.9"، warn_unused_ignores، warn_redundant_casts،
   check_untyped_defs = true، و ignore_missing_imports فقط برای seaborn، plotly، arabic_reshaper، bidi و upsetplot
   (به‌صورت per-module override، نه global).
2. `mypy missingly` را اجرا کن و همهٔ خطاها را درست کن. اولویت با اصلاح type است، نه `# type: ignore`.
   هر ignore باقی‌مانده کد خطای مشخص و توضیح یک‌خطی داشته باشد.
   هیچ تغییر رفتاری مجاز نیست: اگر اصلاح type رفتار را تغییر می‌دهد، متوقف شو و در PR گزارش کن.
3. ci.yml: گام mypy را به `mypy missingly` تغییر بده.
4. نوع بازگشتی impute_mice الان Union پنج‌گانه است (impute.py حدود خط ۱۱۱۵). با @overload و Literal برای
   return_result و return_history، نوع دقیق برای هر ترکیب flag تعریف کن. این تغییر فقط در سطح type است و رفتار عوض نمی‌شود.
5. CHANGELOG (Changed) و CONVENTIONS.md (آیتم T2).
6. push و PR، با فهرست تعداد خطاهای اصلاح‌شده به تفکیک ماژول.
```

### PAR-0.2b — mypy strict روی ماژول‌های آماری و gate پوشش

```text
[قواعد مشترک را اینجا paste کن]

برنچ: ci/strict-typing-and-coverage-gate

کار:
1. override با strict = true برای این ماژول‌ها: missingly.mi، missingly.mi_contracts، missingly.mi_workflow،
   missingly.diagnostics، missingly.provenance، missingly.benchmark، missingly.compare و missingly._distance.
   همهٔ خطاها را درست کن، بدون تغییر رفتار.
2. پوشش: عدد واقعی فعلی را با pytest --cov روی Python 3.12 اندازه بگیر. CONVENTIONS.md عدد 92.50 را می‌گوید و
   CHANGELOG عدد 89.64 را. هر دو را با عدد واقعی یکی کن.
   در ci.yml مقدار --cov-fail-under را از 80 به floor(عدد واقعی) برسان، حداقل 92.
   اگر عدد واقعی زیر 92 است، تست‌های واقعی (نه تست بی‌محتوا) برای خطوط پوشش‌داده‌نشدهٔ mi.py و diagnostics.py
   اضافه کن تا به 92 برسد.
3. mutation testing: یک workflow جدا به‌صورت schedule هفتگی و workflow_dispatch با mutmut روی missingly/mi.py و
   missingly/diagnostics.py که گزارش survived mutants را artifact کند. PR را بلاک نکند.
4. مستندات: TESTING.md (بخش‌های Typing، Coverage gate، Mutation testing)، CONVENTIONS.md (T1، T2، T3) و CHANGELOG.
5. push و PR.
```

### PAR-0.3 — آماده‌سازی انتشار 2.x

```text
[قواعد مشترک را اینجا paste کن]

برنچ: docs/release-2.x-readiness

زمینه: روی PyPI فقط 1.0.0 هست. _version.py مقدار 2.0.0 را دارد، ولی CHANGELOG ورودی تاریخ‌دار 2.0.0 ندارد و یک
بلوک بزرگ [Unreleased] بالای 1.0.0 است. تاریخ 0.1.0 در CHANGELOG (2025-01-10) با اولین commit گیت (2026-04-21)
تناقض دارد. تنها tag موجود v1.0.0 است.

کار:
1. CHANGELOG را مرتب کن: بلوک [Unreleased] فعلی را به [2.1.0] تبدیل کن (2.0.0 فقط تغییر مجوز بود و بخش جدا
   می‌گیرد). تاریخ‌ها از `git log` استخراج شوند. تناقض تاریخ 0.1.0 در یک یادداشت توضیح داده شود، نه با پاک کردن.
   در صورت نیاز _version.py را به 2.1.0 ببر.
2. CONTRIBUTING.md: بخش «Versioning and deprecation policy» با SemVer، این‌که چه چیزی breaking حساب می‌شود،
   دورهٔ حداقل دو minor برای deprecation با FutureWarning (از missingly/_deprecation.py)، و cadence انتشار ماهانه.
3. release.yml را بخوان و یک چک‌لیست انتشار در RELEASING.md بنویس: tag، build، twine check، trusted publishing، و
   بررسی نصب از PyPI در یک venv تمیز.
4. یک تست قرارداد که بررسی کند `_version.__version__` با بالاترین نسخهٔ تاریخ‌دار CHANGELOG برابر است.
5. **tag نزن و منتشر نکن.** در گزارش نهایی دقیقاً بنویس مالک ریپو چه دستورهایی را برای انتشار اجرا کند.
6. push و PR.
```

### PAR-0.4 — تغییر مجوز (فقط پس از تصمیم شما)

```text
[قواعد مشترک را اینجا paste کن]

برنچ: chore/license-<spdx>   (مثلاً chore/license-mit)

تصمیم مالک: مجوز از BUSL-1.1 به <SPDX انتخابی: MIT یا BSD-3-Clause یا Apache-2.0> تغییر می‌کند.

کار:
1. LICENSE را با متن رسمی مجوز جدید جایگزین کن. NOTICE را بازبینی کن (برای Apache-2.0 لازم است؛ برای MIT/BSD
   فقط attributionهای شخص ثالث بماند). XeroGraph_LICENSE و هر attribution شخص ثالث دست‌نخورده بماند.
2. pyproject.toml: license = "<SPDX>"، و classifier مجوز طبق PEP 639 تنظیم شود.
3. README.md، README_FA.md و README_DE.md: badge و بخش License. هر اشاره به Change Date 2030 حذف شود.
4. هر فایل دیگری که BUSL را نام می‌برد: `grep -rni busl` روی کل ریپو به‌جز .git.
5. تست قرارداد: فیلد license در pyproject با SPDX شناسه‌ای که در LICENSE آمده یکی باشد.
6. CHANGELOG، زیر Changed: اعلام تغییر مجوز و این‌که از کدام نسخه به بعد اعمال می‌شود.
7. push و PR.
```

### PAR-0.5a — `MissinglyImputer`: ورودی numpy و set_output

```text
[قواعد مشترک را اینجا paste کن]

برنچ: feat/transformer-array-input-set-output

زمینه: MissinglyImputer (missingly/transformer.py) فقط pandas DataFrame می‌پذیرد و آرایه را رد می‌کند.
`set_output`، `feature_names_in_`، `n_features_in_` و `get_feature_names_out` کامل نیستند یا وجود ندارند.
حالت‌های ناامن (transductive) عمداً NotImplementedError می‌دهند (transformer.py:500-521). این سیاست درست است و باید حفظ شود.

کار:
1. تست‌های شکست‌خورده:
   - fit و transform روی np.ndarray (float با nan) برای استراتژی‌های mean، median، mode، knn و pmm؛
   - `set_output(transform="pandas")` و `"default"`؛
   - `feature_names_in_` فقط وقتی ورودی DataFrame است؛ `n_features_in_` همیشه؛
   - `get_feature_names_out()`؛
   - داخل Pipeline و ColumnTransformer و GridSearchCV.
2. پیاده‌سازی با sklearn.utils.validation (validate_data یا check_array بسته به نسخهٔ sklearn؛ حداقل sklearn پروژه
   را در نظر بگیر). ورودی sparse صریحاً با پیام روشن رد شود (این PR آن را پشتیبانی نمی‌کند).
3. docstring کلاس: بخش‌های «Input containers» و «Output containers» با مثال doctest برای numpy و pandas.
4. README (en/fa/de): بخش sklearn با مثال numpy. CHANGELOG (Added). CONVENTIONS.md (A3).
5. push و PR.
```

### PAR-0.5b — `parametrize_with_checks`

```text
[قواعد مشترک را اینجا paste کن]

برنچ: test/sklearn-estimator-checks

کار:
1. `tests/test_sklearn_estimator_checks.py` با `sklearn.utils.estimator_checks.parametrize_with_checks` روی
   MissinglyImputer برای strategy در mean، median، most_frequent/mode، knn و pmm.
2. هر check که شکست می‌خورد:
   - اگر باگ است: تست بازتولید جدا و اصلاح (fix:).
   - اگر سیاست عمدی پروژه است (مثلاً رد sparse): از `expected_failed_checks` یا `_more_tags`/`__sklearn_tags__`
     طبق نسخهٔ sklearn استفاده کن، با دلیل مکتوب کنار هر مورد.
3. CI روی کمینه و بیشینهٔ sklearn پشتیبانی‌شده اجرا شود (یک job کوچک با pin نسخه).
4. docs: صفحهٔ «sklearn compatibility» با جدول checkهای پاس‌شده و استثناهای مستند. CHANGELOG و CONVENTIONS.md
   (R5: «Estimator checks» انجام شد).
5. push و PR. در PR تعداد checkهای پاس‌شده و استثناها را بنویس.
```

---

## فاز ۱ — موتور FCS هم‌سطح mice

> اصل طراحی این فاز: یک kernel FCS واحد با `method` per-variable، که `impute_mice` و `MIData` روی آن سوار شوند.
> API عمومی فعلی `impute_mice` (تابع در missingly/impute.py حدود خط ۱۱۱۵) نشکند؛ پارامترهای جدید اختیاری‌اند و
> رفتار پیش‌فرض قدیمی با FutureWarning اعلام شود اگر قرار است عوض شود.

### PAR-1.1a — هستهٔ FCS با `method` per-variable: `norm`، `norm.boot`، `pmm`

```text
[قواعد مشترک را اینجا paste کن]

برنچ: feat/fcs-per-variable-methods-numeric

زمینه:
- impute_mice الان یک estimator برای همهٔ ستون‌ها دارد: sklearn IterativeImputer + BayesianRidge(sample_posterior=True)
  در مسیر سریع، یا حلقهٔ دستی _run_mice_with_history (impute.py حدود خط ۶۹۰) وقتی predictor_matrix، visit_sequence،
  where یا bounds داده شود.
- قرارداد FCSKernel در mi_contracts.py:308 تعریف شده است.
- هدف: هم‌سطح mice(method=...) شویم. مرجع الگوریتم‌ها: van Buuren, FIMD 2nd ed., §3.2 (norm، norm.boot) و §3.4 (pmm).

کار:
1. oracle جدید با ORACLE_GUIDE.md: tests/fixtures/mice_methods_numeric/
   - dataset nhanes از mice (CSV با SHA-256)؛
   - generate_mice_reference.R با mice 3.19.0 قفل‌شده: m=20، maxit=20، seed ثابت، برای method = "norm"، "norm.boot"
     و "pmm" جداگانه. خروجی: میانگین و SD ایمپیوت‌ها برای هر متغیر، و pooled coef از lm(chl ~ age + bmi) با pool().
   - تلورانس: چون RNG دو زبان متفاوت است، مقایسهٔ «توزیعی» نه سلول‌به‌سلول. تفاوت pooled estimate کمتر از 0.1×SE
     pooled شدهٔ mice، و تفاوت SD ایمپیوت‌ها کمتر از ۱۵٪. تلورانس‌ها را در manifest با استدلال بنویس.
2. تست‌های شکست‌خورده: impute_mice(df, method={"bmi": "pmm", "chl": "norm"}, n_imputations=20, ...) و method به‌صورت
   رشتهٔ واحد برای همهٔ ستون‌ها، به‌علاوهٔ تست‌های property (hypothesis):
   - هیچ مقدار مشاهده‌شده تغییر نکند؛
   - pmm فقط مقادیر مشاهده‌شده را برگرداند؛
   - با seed یکسان، خروجی bit-identical باشد؛
   - با seed متفاوت، ایمپیوت‌ها متفاوت باشند.
3. پیاده‌سازی:
   - ماژول جدید missingly/fcs/ (یا _fcs.py) با رجیستری متدها: هر متد یک تابع خالص
     (y_obs, X_obs, X_mis, rng) -> draws که قرارداد FCSKernel را رعایت کند.
     - norm: Bayesian linear regression با draw از posterior σ² و β (الگوریتم 3.1 در FIMD)؛
     - norm.boot: bootstrap؛
     - pmm: predictive mean matching type 1 با donors=5 (پارامترهای کامل در PAR-1.2).
   - impute_mice پارامتر method بگیرد: str یا dict[str, str]، پیش‌فرض None یعنی رفتار فعلی. اعتبارسنجی: نام متد
     ناشناخته → InvalidStrategyError؛ متد عددی روی ستون categorical → خطای typed صریح.
   - predictor_matrix، visit_sequence، where و bounds روی kernel جدید هم کار کنند.
4. مستندات: docstring کامل impute_mice (جدول متدها با مرجع FIMD)، vignette در docs با مثال nhanes، README سه زبانه
   (بخش MICE)، CHANGELOG، CONVENTIONS.md (R4) و COMPETITOR_ANALYSIS.md (ردیف mice، فقط با عدد oracle).
5. push و PR. در PR جدول «missingly در برابر mice» برای هر متد و هر متغیر بگذار.
```

### PAR-1.1b — متدهای دسته‌ای داخل FCS: `logreg`، `polyreg`، `polr`

```text
[قواعد مشترک را اینجا paste کن]

برنچ: feat/fcs-categorical-methods

زمینه: الان impute_mice ستون‌های categorical را ordinal-encode می‌کند و با BayesianRidge رگرس می‌کند
(impute.py:1313-1345). خود CHANGELOG این را «unvalidated categorical FCS approximation» می‌نامد.
impute_logreg (حدود خط ۱۵۷۹)، impute_polyreg (حدود ۱۶۹۹) و impute_polr (حدود ۱۸۱۷) جدا و تک‌مرحله‌ای‌اند.
هدف: هم‌سطح mice شدن برای logreg، polyreg و polr با draw از posterior (FIMD §3.6).

کار:
1. oracle: tests/fixtures/mice_methods_categorical/ با nhanes2 (age سه‌سطحی factor، hyp باینری) و یک ستون ترتیبی
   ساختگی ثبت‌شده با seed (یا boys$gen که ordered factor است). خروجی mice: جدول فراوانی ایمپیوت‌ها و pooled
   glm(hyp ~ age + bmi, family=binomial).
   تلورانس: اختلاف نسبت هر سطح کمتر از ۵ واحد درصد؛ pooled log-odds کمتر از 0.15×SE.
2. تست‌های شکست‌خورده:
   - method پیش‌فرض برای category باینری logreg، برای nominal polyreg و برای ordered polr باشد (مثل defaultMethod در mice)؛
   - خروجی همیشه یکی از سطوح موجود باشد و dtype categorical و ترتیب سطوح حفظ شود؛
   - برای logreg، posterior draw با augmentation در حالت separation (طبق mice: augment) جلوی واگرایی را بگیرد.
3. پیاده‌سازی در رجیستری FCS از PAR-1.1a. polr با statsmodels OrderedModel. مسیر ordinal-encoding حذف شود.
   **تغییر رفتار:** خروجی پیش‌فرض برای ستون‌های categorical عوض می‌شود. در CHANGELOG زیر «Changed» با برچسب
   BREAKING توضیح بده و قرارداد `_mice_conditional_model_contract` را به‌روز کن.
   impute_logreg، impute_polyreg و impute_polr عمومی همان kernel را صدا بزنند (حذف کد تکراری).
4. مستندات مثل PAR-1.1a، به‌علاوهٔ بخش «Categorical variables» در vignette.
5. push و PR با جدول مقایسه.
```

### PAR-1.1c — متدهای درختی: `cart`، `rf`، `midastouch`

```text
[قواعد مشترک را اینجا paste کن]

برنچ: feat/fcs-tree-methods

کار:
1. oracle: tests/fixtures/mice_methods_tree/ روی nhanes2 برای method = "cart"، "rf" و "midastouch" با همان قالب و
   تلورانس توزیعی PAR-1.1a. برای rf، تفاوت پیاده‌سازی (ranger در R در برابر sklearn در Python) را در manifest
   مستند کن و تلورانس را بر همین اساس تعیین کن.
2. تست‌ها: هر سه متد هم برای ستون عددی و هم categorical کار کنند؛ cart و rf از donorهای برگ (FIMD §3.5)
   نمونه بگیرند، نه از پیش‌بینی میانگین.
3. پیاده‌سازی در رجیستری FCS. impute_rf و impute_gb فعلی (impute.py حدود ۱۹۹۷ و ۲۰۶۸، روی IterativeImputer با
   max_iter=1) دست نخورند. در docstringشان See Also به method="rf" اضافه کن و تفاوتشان را توضیح بده
   (آن‌ها missForest-like تک‌مقداری‌اند).
4. مستندات و CHANGELOG. push و PR.
```

### PAR-1.2 — PMM کامل

```text
[قواعد مشترک را اینجا paste کن]

برنچ: feat/pmm-matching-types

زمینه: impute_pmm (impute.py:1437) تک‌زنجیره و جدا از MICE است. در PAR-1.1a یک pmm پایه داخل FCS آمد.

کار:
1. پارامترهای pmm: matchtype ∈ {0, 1, 2} (FIMD §3.4.2)، donors (پیش‌فرض 5)، ridge (پیش‌فرض 1e-5)، و exclude
   (مقادیری که donor نباشند)، از طریق method_params={"bmi": {"donors": 10}}.
2. oracle: همان fixture mice_methods_numeric، به‌علاوهٔ اجراهای matchtype 0 و 2 و donors=1 و 10.
3. impute_pmm عمومی و استراتژی "pmm" در MissinglyImputer هر دو از همین kernel استفاده کنند. رفتار fit و transform
   ترانسفورمر (donor pool فقط از داده‌های train) حفظ شود و تستش باشد (anti-leakage).
4. مستندات (جدول matchtype با توضیح)، CHANGELOG. push و PR.
```

### PAR-1.3a — `blocks` و `post`

```text
[قواعد مشترک را اینجا paste کن]

برنچ: feat/fcs-blocks-and-post

کار:
1. `blocks`: dict[str, list[str]] که چند متغیر را با هم ایمپیوت کند (در این PR فقط با متد multivariate norm برای
   block‌های عددی؛ متدهای دیگر روی block با خطای صریح رد شوند). predictor_matrix در سطح block تعریف شود، مثل mice.
2. `post`: dict[str, Callable[[pd.Series, PostContext], pd.Series]] که بعد از هر draw روی ایمپیوت‌های آن متغیر اجرا شود.
   PostContext شامل iteration، imputation index و rng است.
   - کاربردها: squeeze، گرد کردن، و **delta adjustment** (پایهٔ فاز ۲: post={"chl": lambda s, ctx: s + 10}).
   - تابع کمکی عمومی missingly.fcs.squeeze(lower, upper).
3. oracle: mice(blocks=...) روی nhanes، و mice(post=...) با `imp[[j]][, i] <- squeeze(...)` و delta = +10 روی chl؛
   مقایسهٔ pooled mean(chl).
4. تست: post فقط روی سلول‌های ایمپیوت‌شده اثر بگذارد، نه مشاهده‌شده‌ها؛ ترتیب اجرا مستند و تست شده باشد.
5. مستندات: vignette با بخش «Post-processing and delta adjustment» (اشاره به فاز ۲)، CHANGELOG. push و PR.
```

### PAR-1.3b — Passive imputation

```text
[قواعد مشترک را اینجا paste کن]

برنچ: feat/fcs-passive-imputation

کار:
1. `passive`: dict[str, Callable[[pd.DataFrame], pd.Series]]، مثلاً {"bmi": lambda d: d["wgt"] / (d["hgt"]/100)**2}.
   متغیر passive در هر iteration بعد از به‌روز شدن ورودی‌هایش از روی آن‌ها بازمحاسبه شود و خودش مستقیم ایمپیوت نشود.
   predictor_matrix باید جلوی feedback loop را بگیرد: اگر bmi ورودی پیش‌بینی hgt یا wgt باشد، خطای صریح
   (FIMD §6.4).
2. oracle: mice با method["bmi"] <- "~ I(wgt / (hgt/100)^2)" روی boys (wgt، hgt، bmi). مقایسهٔ میانگین bmi ایمپیوت‌شده
   و pooled lm.
3. تست‌ها: سازگاری درونی (bmi دقیقاً برابر فرمول در هر dataset ایمپیوت‌شده)، تشخیص feedback loop، و seed.
4. مستندات و CHANGELOG. push و PR.
```

### PAR-1.4 — `MIData` کامل

```text
[قواعد مشترک را اینجا paste کن]

برنچ: feat/midata-complete-with-pool

زمینه: MIData در mi_contracts.py:344 وجود دارد. mi_with و mi_fit در mi_workflow.py هستند (خطوط ۴۰ و ۹۵).
هدف: معادل کامل شیء mids در mice.

کار:
1. MIData:
   - complete(action: int | "long" | "broad" | "all", include_original=False)، معادل mice::complete؛
     ستون‌های .imp و .id در حالت long.
   - with_(fn)، که نتیجه را در MIResults (یک لیست typed) برمی‌گرداند؛ از mi_with فعلی استفاده کند.
   - where، method، predictor_matrix، visit_sequence، seed، chainMean و chainVar (برای PAR-1.6) نگه‌داری شوند.
   - to_parquet(path) و from_parquet(path)، که round-trip را با dataframe_fingerprint از provenance.py تأیید کنند.
     خواندن فایل دستکاری‌شده خطای صریح بدهد.
   - from_long(df, imp_col=".imp", id_col=".id")، برای وارد کردن خروجی mice::complete(long) یا Stata mi.
2. impute_mice(..., return_result=True) یک MIData کامل برگرداند. سازگاری عقب‌رو با ImputationResult حفظ شود
   (یا با FutureWarning مسیر مهاجرت داده شود).
3. تست‌ها: round-trip، fingerprint، from_long روی خروجی واقعی mice (fixture CSV از R)، و برابری complete("long")
   با mice.
4. مستندات: صفحهٔ «MIData» در docs با مثال کامل شبیه mice (imp ← impute، fit ← with، pool) و README سه‌زبانه.
   CHANGELOG. push و PR.
```

### PAR-1.5a — Barnard–Rubin و جدول pooled کامل

```text
[قواعد مشترک را اینجا paste کن]

برنچ: feat/pooling-barnard-rubin

زمینه: mi.py:33 (pool_scalar_estimates) و mi.py:199 (pool_linear_regression_results) df کلاسیک Rubin را
محاسبه می‌کنند: (m-1)(1+1/r)^2 (mi.py حدود ۱۷۹ و ۳۳۸-۳۴۳). dfcom ندارند. fmi با df قدیمی محاسبه می‌شود.

کار:
1. oracle: tests/fixtures/mice_pool/ با nhanes، m=20 و fit lm(chl ~ age + bmi)؛ خروجی summary(pool(fit)) و
   pool(fit)$pooled با ستون‌های estimate، ubar، b، t، dfcom، df، riv، lambda و fmi.
   برای جدا کردن pooling از RNG، **ورودی oracle همان ۲۰ dataset ایمپیوت‌شدهٔ R است** (CSV long). پس تلورانس
   تقریباً دقیق است: 1e-8 نسبی.
2. تست‌های شکست‌خورده: df با Barnard–Rubin (1999) و dfcom (پیش‌فرض n - k)، طبق فرمول mice::pool. fmi و lambda
   طبق تعریف mice. CI با t(df). وقتی dfcom=∞ (پارامتر صریح)، نتیجه به df کلاسیک برگردد (تست برابری).
3. پیاده‌سازی: تابع جدید pool(results, dfcom=None) که PooledTable با ستون‌های بالا به‌علاوهٔ se، statistic، p_value،
   conf_low و conf_high برگرداند. pool_scalar_estimates و pool_linear_regression_results روی آن سوار شوند و خروجی
   قبلی‌شان با FutureWarning اعلام شود اگر df عوض می‌شود.
4. مستندات: docstring با فرمول‌ها (LaTeX در Notes) و مرجع Barnard & Rubin 1999 و FIMD §5.2. صفحهٔ docs «Pooling».
   CHANGELOG (Changed با توضیح تغییر df). COMPETITOR_ANALYSIS.md (Barnard–Rubin: Parity verified با لینک fixture).
5. push و PR.
```

### PAR-1.5b — Pooling نتیجه‌های statsmodels

```text
[قواعد مشترک را اینجا paste کن]

برنچ: feat/pool-statsmodels-results

کار:
1. pool() هر نتیجهٔ statsmodels را بپذیرد که params، cov_params() و df_resid (یا nobs) دارد: OLS، WLS، GLM
   (binomial، poisson)، Logit، Probit، MixedLM و GEE. یک Protocol برای «estimable result» در mi_contracts.py
   تعریف شود تا کاربر نتیجهٔ دلخواه را هم بتواند pool کند.
   dfcom خودکار: df_resid اگر هست؛ برای GEE و MixedLM با هشدار صریح ∞، طبق رفتار mice.
2. mi_workflow: mi_fit(midata, formula, model="ols" | "glm" | "logit" | "mixedlm" | "gee", **kw) و pool_linear_models
   با FutureWarning به pool منتقل شود.
3. oracle: mice_pool fixture گسترش یابد به glm(hyp ~ bmi, binomial) روی nhanes2، و lmer یا glmer (lme4) روی یک
   dataset خوشه‌ای کوچک ثبت‌شده. ورودی همان datasetهای ایمپیوت‌شدهٔ R باشد.
4. تست‌ها برای هر نوع مدل. نتیجهٔ بدون cov_params خطای typed بدهد.
5. مستندات (جدول مدل‌های پشتیبانی‌شده)، CHANGELOG. push و PR.
```

### PAR-1.5c — آزمون‌های چندمتغیره D1/D2/D3

```text
[قواعد مشترک را اینجا paste کن]

برنچ: feat/multiparameter-tests-d1-d2-d3

کار:
1. توابع D1(fit1, fit0) (Wald چندمتغیره، Li، Raghunathan و Rubin 1991)، D2(statistics) (ترکیب chi²) و D3(fit1, fit0)
   (likelihood ratio، Meng و Rubin 1992)، برای مقایسهٔ مدل تو در تو و آزمون مشترک چند ضریب.
   خروجی: statistic، df1، df2، p_value و riv.
2. oracle: mice::D1، D2، D3 و mitml::testModels روی nhanes2، با مدل‌های fit0: chl ~ age و fit1: chl ~ age + bmi + hyp.
   ورودی، datasetهای ایمپیوت‌شدهٔ R است. تلورانس 1e-6.
3. تست‌ها: حالت‌های مرزی (m=2، riv≈0)، و مدل‌های غیرتودرتو → خطای صریح.
4. مستندات (فرمول‌ها و راهنمای انتخاب D1 در برابر D2 در برابر D3، با ارجاع به FIMD §5.3)، CHANGELOG و
   COMPETITOR_ANALYSIS.md. push و PR.
```

### PAR-1.6 — نمودارهای تشخیصی MI

```text
[قواعد مشترک را اینجا paste کن]

برنچ: feat/mi-diagnostic-plots

زمینه: mice_convergence (diagnostics.py:1118) فقط R-hat روی میانگین per-iteration دارد. plot_mice_convergence در
diagnostics.py حدود ۱۲۴۲، و vis_impute_dist و miss_impute_compare هم وجود دارند.

کار:
1. MIData.chainMean و chainVar (از PAR-1.4) → plot_mi_trace(midata): trace میانگین و SD برای هر متغیر و هر
   زنجیره، مثل plot(imp) در mice.
2. stripplot_mi، densityplot_mi و xyplot_mi: مشاهده‌شده در برابر ایمپیوت‌شده با رنگ‌بندی ثابت mice
   (observed آبی، imputed قرمز). نسخهٔ static (matplotlib) و نسخهٔ interactive (plotly، extra). RTL برای برچسب‌ها.
3. overimpute(midata، variables): برای هر مقدار مشاهده‌شده آن را حذف کن، m بار ایمپیوت کن، و نمودار observed در برابر
   پیش‌بینی با بازهٔ ۹۰٪ رسم کن (Honaker و King؛ مثل Amelia::overimpute). خروجی عددی هم بدهد (پوشش بازه) که تست شود.
4. تست‌ها: contract (محورها، تعداد خطوط برابر m، برچسب‌ها)، و پوشش overimpute روی دادهٔ شبیه‌سازی MCAR نزدیک ۹۰٪
   (بازهٔ مجاز آماری مستند، با seed ثابت).
5. مستندات: گالری در docs با تصویر هر نمودار، README سه‌زبانه و CHANGELOG. push و PR.
```

---

## پس از پایان فاز ۱

- امتیازهای جدید را فقط برای ابعادی که oracle دارند در `competitive-evidence/build_tables.py` به‌روز کنید.
  پیش‌بینی: F4 = ۸، F5 = ۸، F6 = ۱۰، F7 = ۱۰، F13 = ۸.
- سپس `COMPETITOR_SCORECARD_FA.md` را بازتولید کنید.
- دستور کار‌های فاز ۲ (MNAR) روی `post` از PAR-1.3a و `MIData` از PAR-1.4 بنا می‌شوند.
