"""missingly — comprehensive missing-data analysis and imputation toolkit.

Public API
----------

**Visualisation**::

    matrix, bar, miss_case, miss_var_pct, vis_miss, miss_which,
    upset, miss_patterns, miss_cooccurrence, heatmap, miss_cluster, ...

**Summary / stats**::

    summarise, miss_scan_count, miss_summary,
    miss_var_summary, miss_case_summary,
    n_miss, n_complete, pct_miss, pct_complete, bind_shadow

**Diagnosis**::

    mcar_test, mar_mnar_test, diagnose_missing

**Imputation**::

    impute_mean, impute_median, impute_mode,
    impute_knn, impute_mice, impute_rf, impute_gb, make_imputer

**sklearn-style**::

    MissinglyImputer, FittedImputer, make_imputer

**Comparison**::

    compare_imputations, cv_compare_imputations

**Time-series**::

    miss_ts_summary, vis_ts_miss, vis_miss_over_time, impute_ts

**MI Workflow (R-style mice::with / mice::pool)**::

    mi_with, mi_fit, pool_linear_models

**Evaluation**::

    compare_imputations, cv_compare_imputations

**Report**::

    create_report

**Accessor**::

    df.miss.n_miss(), df.miss.vis_miss(), ...
"""

from __future__ import annotations

import warnings as _warnings

# ---------------------------------------------------------------------------
# Deprecation helper
# ---------------------------------------------------------------------------
from missingly._deprecation import deprecated_api as _deprecated_api

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
from missingly.visualise import (
    matrix,
    bar,
    miss_case,
    miss_var_pct,
    vis_miss,
    miss_which,
    upset,
    miss_patterns,
    miss_cooccurrence,
    heatmap,
    dendrogram,
    miss_cluster,
    miss_row_profile,
    shadow_scatter,
    vis_miss_fct,
    vis_miss_by_group,
    vis_impute_dist,
    miss_impute_compare,
    scatter_miss,
    vis_miss_cumsum_var,
    vis_miss_cumsum_case,
    vis_miss_span,
    vis_parallel_coords,
)

# ---------------------------------------------------------------------------
# Summary / stats  (canonical source: diagnostics)
# ---------------------------------------------------------------------------
from missingly.diagnostics import (
    bind_shadow,
    n_miss,
    n_complete,
    pct_miss,
    pct_complete,
    miss_var_summary,
    miss_case_summary,
    summarise,
    miss_scan_count,
    miss_summary,
    mcar_test,
    mar_mnar_test,
    diagnose_missing,
)

# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------
from missingly.impute import (
    impute_mean,
    impute_median,
    impute_mode,
    impute_knn,
    impute_mice,
    impute_rf,
    impute_gb,
    make_imputer,
    FittedImputer,
)

# ---------------------------------------------------------------------------
# sklearn-style transformer
# ---------------------------------------------------------------------------
from missingly.transformer import (
    MissinglyImputer,
)

# ---------------------------------------------------------------------------
# Multiple Imputation pooling utilities
# ---------------------------------------------------------------------------
from missingly.mi import (
    pool_scalar_estimates,
    pool_linear_regression_results,
)

# ---------------------------------------------------------------------------
# MI Workflow (R-style mice::with / mice::pool)
# ---------------------------------------------------------------------------
from missingly.mi_workflow import (
    mi_with,
    mi_fit,
    pool_linear_models,
)

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
from missingly.compare import (
    compare_imputations,
    cv_compare_imputations,
)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
from missingly.report import create_report

# ---------------------------------------------------------------------------
# Time-series
# ---------------------------------------------------------------------------
from missingly.timeseries import (
    miss_ts_summary,
    vis_ts_miss,
    vis_miss_over_time,
    impute_ts,
)

# ---------------------------------------------------------------------------
# Simulation (v1 stable)
# ---------------------------------------------------------------------------
from missingly.simulate import (
    simulate_mcar,
    simulate_mar,
    simulate_mnar,
    simulate_mixed,
)

# ---------------------------------------------------------------------------
# Manipulation
# ---------------------------------------------------------------------------
from missingly.manipulation import (
    replace_with_na,
    replace_with_na_all,
    add_any_miss_var,
    bind_shadow_matrix,
)

# ---------------------------------------------------------------------------
# Legacy / experimental shims
# ---------------------------------------------------------------------------
from missingly.stats_extra import (
    hotelling_test as _hotelling_test,
    pattern_monotone_test as _pattern_monotone_test,
    missing_correlation_matrix as _missing_correlation_matrix,
)
from missingly.timeseries import (
    gap_table as _gap_table,
    vis_gap_lengths as _vis_gap_lengths,
)
from missingly.manipulation import (
    clean_names as _clean_names,
    remove_empty as _remove_empty,
    coalesce_columns as _coalesce_columns,
    miss_as_feature as _miss_as_feature,
)
from missingly.performance import (
    chunk_apply as _chunk_apply,
    memory_usage_mb as _memory_usage_mb,
    optimize_dtypes as _optimize_dtypes,
)
from missingly.simulate import (
    simulate_mcar as _simulate_mcar,
    simulate_mar as _simulate_mar,
    simulate_mnar as _simulate_mnar,
    simulate_mixed as _simulate_mixed,
)

hotelling_test = _deprecated_api(since="0.2.0")(_hotelling_test)
pattern_monotone_test = _deprecated_api(since="0.2.0")(_pattern_monotone_test)
missing_correlation_matrix = _deprecated_api(since="0.2.0")(_missing_correlation_matrix)
gap_table = _deprecated_api(since="0.2.0")(_gap_table)
vis_gap_lengths = _deprecated_api(since="0.2.0")(_vis_gap_lengths)
clean_names = _deprecated_api(since="0.2.0")(_clean_names)
remove_empty = _deprecated_api(since="0.2.0")(_remove_empty)
coalesce_columns = _deprecated_api(since="0.2.0")(_coalesce_columns)
miss_as_feature = _deprecated_api(since="0.2.0")(_miss_as_feature)
chunk_apply = _deprecated_api(since="0.2.0")(_chunk_apply)
memory_usage_mb = _deprecated_api(since="0.2.0")(_memory_usage_mb)
optimize_dtypes = _deprecated_api(since="0.2.0")(_optimize_dtypes)
simulate_mcar = _deprecated_api(since="0.2.0")(_simulate_mcar)
simulate_mar = _deprecated_api(since="0.2.0")(_simulate_mar)
simulate_mnar = _deprecated_api(since="0.2.0")(_simulate_mnar)
simulate_mixed = _deprecated_api(since="0.2.0")(_simulate_mixed)

test_hotelling = hotelling_test
test_pattern_monotone = pattern_monotone_test

# ---------------------------------------------------------------------------
# Accessor (side-effect import — registers df.miss)
# ---------------------------------------------------------------------------
import missingly.accessor  # noqa: F401

__all__ = [
    # Visualisation
    "matrix", "bar", "miss_case", "miss_var_pct", "vis_miss", "miss_which",
    "upset", "miss_patterns", "miss_cooccurrence", "heatmap", "dendrogram",
    "miss_cluster", "miss_row_profile", "shadow_scatter", "vis_miss_fct",
    "vis_miss_by_group", "vis_impute_dist", "miss_impute_compare",
    "scatter_miss", "vis_miss_cumsum_var", "vis_miss_cumsum_case",
    "vis_miss_span", "vis_parallel_coords",
    # Summary
    "bind_shadow", "n_miss", "n_complete", "pct_miss", "pct_complete",
    "miss_var_summary", "miss_case_summary", "summarise",
    "miss_scan_count", "miss_summary",
    # Diagnosis
    "mcar_test", "mar_mnar_test", "diagnose_missing",
    # Imputation
    "impute_mean", "impute_median", "impute_mode",
    "impute_knn", "impute_mice", "impute_rf", "impute_gb", "make_imputer",
    # sklearn-style
    "MissinglyImputer", "FittedImputer",
    # MI pooling
    "pool_scalar_estimates", "pool_linear_regression_results",
    # MI Workflow
    "mi_with", "mi_fit", "pool_linear_models",
    # Comparison
    "compare_imputations", "cv_compare_imputations",
    # Report
    "create_report",
    # Time-series
    "miss_ts_summary", "vis_ts_miss", "vis_miss_over_time", "impute_ts",
    # Simulation (experimental)
    "simulate_mcar", "simulate_mar", "simulate_mnar", "simulate_mixed",
    # Manipulation
    "replace_with_na", "replace_with_na_all", "add_any_miss_var", "bind_shadow_matrix",
    # Legacy / experimental
    "hotelling_test", "pattern_monotone_test", "missing_correlation_matrix",
    "gap_table", "vis_gap_lengths",
    "clean_names", "remove_empty", "coalesce_columns", "miss_as_feature",
    "chunk_apply", "memory_usage_mb", "optimize_dtypes",
    # Backward-compat aliases
    "test_hotelling", "test_pattern_monotone",
]

__version__ = "0.3.0"
