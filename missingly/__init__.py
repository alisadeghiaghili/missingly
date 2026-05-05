"""missingly - A Python library for missing data analysis.

Inspired by the R package naniar.
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

# Backward-compat aliases (defined here, NOT in stats_extra, to avoid
# pytest collecting stats_extra.py as a test module)
test_hotelling = hotelling_test
test_pattern_monotone = pattern_monotone_test

# Register the df.miss accessor
from . import accessor  # noqa: F401

__all__ = [
    # summary
    "bind_shadow", "n_miss", "n_complete", "pct_miss", "pct_complete",
    "miss_var_summary", "miss_case_summary",
    # visualise
    "matrix", "bar", "miss_case", "miss_var_pct", "vis_miss", "miss_which",
    "upset", "miss_patterns", "miss_cooccurrence",
    "heatmap", "dendrogram", "miss_cluster",
    "miss_row_profile", "miss_impute_compare",
    "shadow_scatter",
    "vis_miss_fct", "vis_miss_by_group",
    "vis_impute_dist",
    "scatter_miss", "vis_miss_cumsum_var", "vis_miss_cumsum_case",
    "vis_miss_span", "vis_parallel_coords",
    # stats
    "mcar_test", "mar_mnar_test", "diagnose_missing",
    # stats_extra
    "hotelling_test", "test_hotelling",
    "pattern_monotone_test", "test_pattern_monotone",
    "missing_correlation_matrix",
    # manipulation
    "replace_with_na", "replace_with_na_all", "clean_names",
    "remove_empty", "coalesce_columns", "miss_as_feature",
    # impute
    "impute_mean", "impute_median", "impute_mode",
    "impute_knn", "impute_mice", "impute_rf", "impute_gb",
    "FittedImputer", "make_imputer",
    # transformer
    "MissinglyImputer",
    # report
    "create_report",
    # compare
    "compare_imputations", "cv_compare_imputations",
    # timeseries
    "miss_ts_summary", "gap_table", "vis_ts_miss",
    "vis_gap_lengths", "vis_miss_over_time", "impute_ts",
    # simulate
    "simulate_mcar", "simulate_mar", "simulate_mnar", "simulate_mixed",
    # performance
    "chunk_apply", "memory_usage_mb", "optimize_dtypes",
    # accessor
    "accessor",
]
