"""Build calibrated 0-10 scorecard tables (markdown) for the missingly competitor report."""
import sys

# name | ecosystem | F1..F20 | N1..N10   (calibrated scores, 2026-11-06)
RAW = """
**missingly**|Py|8 7 6 7 5 5 4 1 3 2 5 6 5 6 6 6 2 1 7 7|2 1 5 3 7 8 8 4 5 2
missingno|Py|8 3 0 0 0 0 0 0 0 0 0 0 0 0 0 5 2 0 0 0|6 9 5 2 4 2 10 10 5 2
sklearn.impute|Py|0 1 0 6 3 3 0 0 1 4 0 2 1 0 10 6 5 0 0 4|10 10 9 8 9 10 8 10 10 6
statsmodels MICE|Py|2 2 0 1 2 5 5 0 0 0 0 0 2 0 1 6 2 2 0 3|9 6 6 6 8 8 5 10 6 5
miceforest|Py|0 1 0 3 6 7 0 0 0 6 0 3 6 0 5 5 6 0 0 4|7 6 6 3 6 4 8 10 6 4
autoimpute|Py|2 2 1 6 5 6 6 1 3 1 0 2 4 0 5 4 1 0 0 3|4 3 6 2 4 0 6 10 3 1
hyperimpute|Py|0 0 0 6 5 2 0 0 0 8 3 4 0 0 7 4 4 0 0 3|4 3 4 6 6 0 6 10 4 1
fancyimpute|Py|0 0 0 4 1 1 0 0 0 6 0 1 0 0 4 2 2 0 0 2|6 6 4 4 3 0 7 10 3 1
missingpy|Py|0 0 0 3 2 0 0 0 0 5 0 0 0 0 6 2 1 0 0 1|2 3 2 1 2 0 7 8 3 0
missforest (PyPI)|Py|0 0 0 3 5 0 0 0 0 5 0 0 0 0 4 4 3 0 0 2|3 2 3 1 3 5 8 10 3 2
scikit-na|Py|5 6 4 0 0 0 0 0 0 0 0 0 0 4 0 5 2 0 0 1|3 1 5 1 4 6 7 10 4 2
MissMecha|Py|4 3 5 2 2 0 0 0 0 0 8 4 0 0 4 5 2 0 0 4|1 1 5 3 4 5 7 10 3 2
NApy|Py|0 1 0 0 2 0 0 0 0 0 0 0 0 0 2 3 8 0 0 1|2 1 5 6 5 7 6 8 4 2
PyPOTS|Py|1 0 0 4 1 2 0 0 9 9 4 6 1 1 5 3 7 1 0 5|7 7 8 7 6 10 5 10 7 5
ImputeGAP|Py|4 2 0 5 0 0 0 0 8 7 5 8 2 2 2 3 3 0 0 4|4 2 6 5 6 10 6 10 4 3
pyampute|Py|2 1 1 0 0 0 0 0 0 0 7 0 0 0 5 4 1 0 0 3|3 2 6 5 4 0 7 10 4 1
ydata-profiling|Py|5 4 0 0 0 0 0 0 0 0 0 0 0 7 0 7 5 0 1 2|8 9 7 3 7 9 9 10 7 6
impyute|Py|0 0 0 5 1 1 0 0 2 2 0 0 0 0 1 2 1 0 0 1|3 3 4 1 3 0 6 8 3 0
datawig|Py|0 0 0 4 6 0 0 0 0 6 0 2 0 0 3 5 3 0 0 2|4 4 4 5 4 0 6 10 2 0
MIDASpy|Py|0 0 0 0 5 6 5 0 0 7 0 2 1 0 2 4 5 0 0 3|5 3 4 7 4 4 4 10 3 2
mipy-impute ⚠|Py|3 2 2 6 3 4 5 2 2 4 5 6 3 5 1 4 2 3 0 3|1 0 3 1 3 9 7 10 2 1
pandas / polars built-ins|Py/Rust|0 3 0 4 2 0 0 0 3 0 0 0 0 0 2 10 9 0 0 3|10 10 9 5 9 10 9 10 10 7
naniar|R|10 10 5 3 2 0 0 0 2 0 1 0 3 1 1 9 3 1 0 3|8 8 8 7 7 3 9 10 8 5
visdat|R|6 3 0 0 0 0 0 0 0 0 0 0 0 1 1 9 2 0 0 0|8 7 7 6 7 8 9 10 8 7
mice|R|4 6 1 4 8 10 10 6 2 4 9 3 8 1 2 5 4 9 0 6|10 10 9 9 8 8 7 8 9 8
VIM|R|9 6 1 8 7 3 0 0 1 3 1 2 3 1 2 6 5 3 0 2|9 7 7 7 6 10 6 8 6 6
missForest|R|0 0 0 3 6 0 0 0 0 6 0 3 2 0 2 4 4 0 0 2|8 7 4 7 5 7 9 8 4 4
missRanger|R|0 0 0 3 7 4 0 0 0 6 0 3 1 0 2 6 6 0 0 3|6 6 6 4 5 3 9 8 5 4
mixgb|R|0 0 0 3 7 5 2 0 0 6 0 1 2 0 2 5 6 0 0 3|4 3 5 6 5 7 7 8 5 4
Amelia|R|3 1 0 1 5 7 5 1 6 0 0 4 7 0 1 3 5 5 0 3|9 6 7 8 5 3 6 8 5 5
imputeTS|R|7 5 0 6 0 0 0 0 10 1 0 3 5 0 2 5 3 0 0 2|8 7 7 8 6 6 9 8 6 5
softImpute|R|0 0 0 1 0 0 0 0 0 8 0 2 0 0 1 2 7 0 0 2|9 5 5 9 5 5 6 8 3 4
missMDA|R|1 0 0 3 8 5 2 0 0 7 0 4 3 0 2 3 4 1 0 2|9 6 7 8 5 9 6 8 6 6
simputation|R|0 0 0 7 5 0 0 0 0 2 0 0 0 0 2 8 3 0 0 2|6 4 6 3 7 1 9 8 6 3
missMethods|R|0 0 0 5 3 0 0 0 0 0 8 6 0 0 1 4 3 0 0 2|4 2 5 3 6 0 7 8 4 2
Hmisc (aregImpute)|R|3 4 0 5 6 6 6 0 0 3 0 2 3 4 1 4 4 1 0 3|10 9 8 8 6 10 5 8 8 6
jomo|R|0 0 0 0 7 7 2 0 0 0 0 0 2 0 1 3 5 8 0 3|7 5 5 7 4 1 4 8 6 3
mitml|R|0 0 0 0 5 6 9 0 0 0 0 0 5 2 1 4 3 9 0 3|8 6 7 7 6 1 6 8 7 4
mi (Gelman)|R|3 3 0 0 6 6 5 0 0 0 0 0 6 0 1 3 3 1 0 2|7 4 5 7 4 5 4 8 4 3
smcfcs|R|0 0 0 0 6 7 3 2 0 0 0 0 3 0 1 3 3 2 0 3|6 4 6 8 5 8 5 8 5 5
rbmi|R|0 1 0 0 2 4 7 10 0 0 0 2 3 3 0 5 5 6 0 5|7 3 8 9 9 10 4 10 5 7
MissMech|R|1 2 9 1 0 0 0 0 0 0 0 0 1 0 0 2 2 0 0 2|6 3 5 8 3 1 6 8 3 2
finalfit|R|5 6 3 0 0 1 2 0 0 0 0 0 0 8 0 8 2 0 0 1|6 5 8 3 5 4 8 10 6 4
norm|R|0 1 0 3 1 5 2 0 0 0 0 0 3 0 0 2 2 0 0 3|8 4 3 8 3 1 3 8 2 2
misl|R|0 0 0 2 6 5 2 0 0 6 0 1 1 0 6 7 5 0 0 3|2 1 4 2 5 9 8 10 6 2
JointAI|R|3 3 0 0 7 7 5 1 0 0 0 2 6 2 0 4 2 8 0 4|7 3 8 8 6 8 5 8 6 5
micemd|R|0 0 0 0 5 6 3 0 0 0 0 0 3 0 0 3 3 8 0 3|6 3 5 7 5 9 5 8 8 4
SAS PROC MI/MIANALYZE|SAS|3 6 3 4 6 9 9 9 5 2 2 3 6 6 2 3 6 6 5 6|10 9 9 9 7 7 5 1 5 10
IBM SPSS MVA + MI|SPSS|4 7 8 5 3 6 7 1 3 1 0 1 4 5 1 2 3 4 8 4|10 9 7 6 5 7 9 1 6 9
Stata mi + misstable|Stata|3 7 3 3 6 9 9 4 4 1 1 2 5 4 1 3 5 8 3 6|10 9 10 9 6 8 7 1 5 9
JMP Pro|JMP|5 6 3 5 3 2 1 0 3 6 0 2 3 5 1 3 4 1 7 4|9 7 8 5 5 8 9 1 5 8
MATLAB|MATLAB|2 3 0 6 2 0 0 0 6 3 0 1 1 3 2 3 6 1 5 4|10 9 9 5 6 9 8 1 6 9
Mplus|Mplus|0 4 3 3 4 6 7 3 2 0 4 1 4 1 0 1 4 8 1 5|9 7 7 8 5 6 5 1 4 7
Blimp|Blimp|1 2 1 2 7 8 7 3 2 1 0 1 6 2 0 2 4 9 1 5|6 5 6 8 4 6 6 5 4 6
JASP|GUI|1 1 0 1 1 0 0 0 0 0 0 0 0 4 0 2 1 0 8 3|7 7 6 3 6 8 10 8 5 5
jamovi|GUI|1 1 2 2 1 0 0 0 0 0 0 0 0 3 0 2 1 0 7 3|7 7 6 3 5 8 10 8 5 5
Impute.jl|Julia|0 1 0 5 1 0 0 0 4 3 0 1 0 0 2 6 5 1 0 3|5 3 4 1 6 7 7 10 6 4
Mice.jl|Julia|0 1 0 2 3 5 5 0 0 0 0 0 2 1 1 6 6 1 0 3|3 2 5 4 5 6 6 9 6 3
danfo.js|JS|0 2 0 2 1 0 0 0 0 0 0 0 0 0 0 4 3 0 0 2|5 4 5 1 4 3 8 10 5 3
Spark MLlib Imputer|Spark|0 3 0 3 1 0 0 0 2 0 0 0 0 0 8 6 9 0 1 3|8 9 8 3 8 10 6 10 8 7
H2O-3|H2O|0 3 0 3 2 0 0 0 2 5 0 1 0 1 4 4 8 0 1 4|8 6 7 3 6 7 6 10 6 7
KNIME Missing Value|KNIME|1 3 0 5 3 0 0 0 4 0 0 1 0 5 5 3 5 0 2 4|8 7 7 2 6 9 9 8 7 7
Orange Impute|Orange|1 2 0 4 3 0 0 0 1 0 0 1 0 1 4 2 2 0 2 3|8 6 7 2 6 8 9 8 6 5
Altair RapidMiner|RapidMiner|1 3 0 5 3 0 0 0 3 0 3 2 0 3 5 3 4 0 2 4|8 6 7 2 5 6 8 2 6 8
DataRobot|DataRobot|1 3 0 3 5 0 0 0 2 1 0 2 2 4 8 4 8 1 2 5|6 6 6 3 5 8 8 1 6 9
Great Expectations|DQ|0 3 0 0 0 0 0 0 0 0 0 0 0 7 3 6 7 0 1 5|8 8 8 2 8 10 6 10 8 7
Soda|DQ|0 3 0 0 0 0 0 0 0 0 0 0 0 4 2 6 7 0 1 4|7 6 7 2 7 8 7 10 7 7
"""

F = [f"F{i}" for i in range(1, 21)]
N = [f"N{i}" for i in range(1, 11)]
CORE = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F13"]  # missing-data core

rows = []
for line in RAW.strip().splitlines():
    name, eco, f, n = line.split("|")
    fv = list(map(int, f.split())); nv = list(map(int, n.split()))
    assert len(fv) == 20 and len(nv) == 10, name
    rows.append({"name": name, "eco": eco, **dict(zip(F, fv)), **dict(zip(N, nv))})

def mean(r, keys):
    return sum(r[k] for k in keys) / len(keys)

for r in rows:
    r["Fm"] = mean(r, F); r["Nm"] = mean(r, N); r["Core"] = mean(r, CORE)
    r["Total"] = 0.6 * r["Fm"] + 0.4 * r["Nm"]
    r["Top"] = sum(1 for k in F if r[k] >= 8)

def table(keys, title):
    out = [f"| ابزار | اکوسیستم | " + " | ".join(keys) + " |",
           "|" + "---|" * (len(keys) + 2)]
    for r in rows:
        out.append(f"| {r['name']} | {r['eco']} | " + " | ".join(str(r[k]) for k in keys) + " |")
    return "\n".join(out)

def best_line(k):
    m = max(r[k] for r in rows)
    who = [r["name"] for r in rows if r[k] == m]
    ms = next(r for r in rows if r["name"] == "**missingly**")[k]
    rank = 1 + sum(1 for r in rows if r[k] > ms)
    return m, ", ".join(who), ms, rank

out = []
out.append("<!--A-->\n" + table(F[:10], "A"))
out.append("<!--B-->\n" + table(F[10:], "B"))
out.append("<!--C-->\n" + table(N, "C"))
ranked = sorted(rows, key=lambda r: -r["Total"])
agg = ["| رتبه | ابزار | اکوسیستم | میانگین امکانات (F) | میانگین غیرامکاناتی (N) | هستهٔ missing-data | امتیاز کل (۰٫۶F+۰٫۴N) | تعداد ابعاد ≥۸ |",
       "|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(ranked, 1):
    agg.append(f"| {i} | {r['name']} | {r['eco']} | {r['Fm']:.1f} | {r['Nm']:.1f} | {r['Core']:.1f} | {r['Total']:.2f} | {r['Top']} |")
out.append("<!--D-->\n" + "\n".join(agg))
bl = ["| بُعد | بهترین امتیاز | دارندهٔ بهترین امتیاز | missingly | رتبهٔ missingly از " + str(len(rows)) + " |", "|---|---|---|---|---|"]
for k in F + N:
    m, who, ms, rank = best_line(k)
    bl.append(f"| {k} | {m} | {who} | {ms} | {rank} |")
out.append("<!--E-->\n" + "\n".join(bl))
core_rank = sorted(rows, key=lambda r: -r["Core"])
out.append("<!--CORE-->\n" + "\n".join(f"{i}. {r['name']} {r['Core']:.2f}" for i, r in enumerate(core_rank[:12], 1)))
py = [r for r in ranked if r["eco"].startswith("Py")]
out.append("<!--PY-->\n" + "\n".join(f"{i}. {r['name']} {r['Total']:.2f} F={r['Fm']:.2f}" for i, r in enumerate(sorted(py, key=lambda r: -r['Total']), 1)))
pyF = sorted(py, key=lambda r: -r["Fm"])
out.append("<!--PYF-->\n" + "\n".join(f"{i}. {r['name']} F={r['Fm']:.2f}" for i, r in enumerate(pyF[:8], 1)))
allF = sorted(rows, key=lambda r: -r["Fm"])
out.append("<!--ALLF-->\n" + "\n".join(f"{i}. {r['name']} F={r['Fm']:.2f}" for i, r in enumerate(allF[:12], 1)))
sys.stdout.reconfigure(encoding="utf-8")
print("\n\n".join(out))
print("\nN_TOOLS", len(rows))
