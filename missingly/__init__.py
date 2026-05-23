"""missingly - A Python library for missing data analysis.

Inspired by the R package naniar.

Public API (v1)
---------------
- ``df.miss.*`` accessor (pandas chaining API)
- Summary: bind_shadow, n_miss, n_complete, pct_miss, pct_complete,
  miss_var_summary, miss_case_summary
- Diagnosis: mcar_test, mar_mnar_test, diagnose_missing
- Visualisation: matrix, bar, miss_case, miss_var_pct, vis_miss,
  miss_which, upset, miss_patterns, miss_cooccurrence, heatmap,
  miss_cluster
- Report: create_report
- Time series: miss_ts_summary, vis_ts_miss, vis_miss_over_time,
  impute_ts
- Evaluation: compare_imputations, cv_compare_imputations
- sklearn: MissinglyImputer, FittedImputer, make_imputer
"""

__version__ = "0.1.0"

from .summary import (
    bind_shadow,
    n_miss,
    n_complete,
    pct_miss,
    pct_complete,
    miss_var_summary,
    miss_case_summary,
)
from .visualise import (
    matrix,
    bar,
    upset,
    dendrogram,
    scatter_miss,
    miss_case,
    vis_impute_dist,
    vis_miss_fct,
    vis_miss_cumsum_var,
    vis_miss_cumsum_case,
    vis_miss_span,
    vis_parallel_coords,
    heatmap,
    vis_miss,
    miss_var_pct,
    miss_cluster,
    miss_which,
    miss_patterns,
    miss_cooccurrence,
    miss_row_profile,
    shadow_scatter,
    vis_miss_by_group,
    miss_impute_compare,
)
from .stats import (
    mcar_test,
    mar_mnar_test,
    diagnose_missing,
)
from .manipulation import (
    replace_with_na,
    replace_with_na_all,
    clean_names,
    remove_empty,
    coalesce_columns,
    miss_as_feature,
)
from .impute import (
    impute_mean,
    impute_median,
    impute_mode,
    impute_knn,
    impute_mice,
    impute_rf,
    impute_gb,
    FittedImputer,
    make_imputer,
)
from .transformer import MissinglyImputer
from .report import create_report
from .compare import compare_imputations, cv_compare_imputations
from .timeseries import (
    miss_ts_summary,
    gap_table,
    vis_ts_miss,
    vis_gap_lengths,
    vis_miss_over_time,
    impute_ts,
)
from .simulate import (
    simulate_mcar,
    simulate_mar,
    simulate_mnar,
    simulate_mixed,
)
from .performance import (
    chunk_apply,
    memory_usage_mb,
    optimize_dtypes,
)
from .stats_extra import (
    hotelling_test,
    pattern_monotone_test,
    missing_correlation_matrix,
)

# Backward-compat aliases (defined here to avoid pytest collecting
# stats_extra.py as a test module via test_* name pattern)
test_hotelling = hotelling_test
test_pattern_monotone = pattern_monotone_test

# Register the df.miss accessor
from . import accessor  # noqa: F401

# ---------------------------------------------------------------------------
# v1 Public API surface
# ---------------------------------------------------------------------------
__all__ = [
    # ── summary ──────────────────────────────────────────────────────────────
    "bind_shadow",
    "n_miss",
    "n_complete",
    "pct_miss",
    "pct_complete",
    "miss_var_summary",
    "miss_case_summary",
    # ── diagnosis ────────────────────────────────────────────────────────────
    "mcar_test",
    "mar_mnar_test",
    "diagnose_missing",
    # ── visualisation (core) ─────────────────────────────────────────────────
    "matrix",
    "bar",
    "miss_case",
    "miss_var_pct",
    "vis_miss",
    "miss_which",
    "upset",
    "miss_patterns",
    "miss_cooccurrence",
    "heatmap",
    "miss_cluster",
    # ── report ───────────────────────────────────────────────────────────────
    "create_report",
    # ── time series ──────────────────────────────────────────────────────────
    "miss_ts_summary",
    "vis_ts_miss",
    "vis_miss_over_time",
    "impute_ts",
    # ── evaluation ───────────────────────────────────────────────────────────
    "compare_imputations",
    "cv_compare_imputations",
    # ── sklearn-style components ─────────────────────────────────────────────
    "MissinglyImputer",
    "FittedImputer",
    "make_imputer",
    # ── impute helpers ────────────────────────────────────────────────────────
    "impute_mean",
    "impute_median",
    "impute_mode",
    "impute_knn",
    "impute_mice",
    "impute_rf",
    "impute_gb",
    # ── manipulation helpers ──────────────────────────────────────────────────
    "replace_with_na",
    "replace_with_na_all",
    # Legacy / power-user API (subject to change) ─────────────────────────────
    # These symbols are experimental or have been superseded.  They emit a
    # FutureWarning on every call and may be moved, renamed, or removed in a
    # future release without a major-version bump.
    "clean_names",               # experimental / legacy
    "remove_empty",              # experimental / legacy
    "coalesce_columns",          # experimental / legacy
    "miss_as_feature",           # experimental / legacy
    "gap_table",                 # experimental / legacy
    "vis_gap_lengths",           # experimental / legacy
    "simulate_mcar",             # experimental / legacy
    "simulate_mar",              # experimental / legacy
    "simulate_mnar",             # experimental / legacy
    "simulate_mixed",            # experimental / legacy
    "chunk_apply",               # experimental / legacy
    "memory_usage_mb",           # experimental / legacy
    "optimize_dtypes",           # experimental / legacy
    "hotelling_test",            # experimental / legacy
    "test_hotelling",            # experimental / legacy (alias)
    "pattern_monotone_test",     # experimental / legacy
    "test_pattern_monotone",     # experimental / legacy (alias)
    "missing_correlation_matrix",  # experimental / legacy
    # visualisation extras (power-user)
    "dendrogram",
    "scatter_miss",
    "miss_row_profile",
    "shadow_scatter",
    "vis_miss_fct",
    "vis_miss_by_group",
    "vis_impute_dist",
    "miss_impute_compare",
    "vis_miss_cumsum_var",
    "vis_miss_cumsum_case",
    "vis_miss_span",
    "vis_parallel_coords",
    # accessor module
    "accessor",
]
