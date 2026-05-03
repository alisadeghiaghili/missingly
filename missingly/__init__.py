# missingly - A Python library for missing data analysis
#
# Author: Ali Sadeghi Aghili <alisadeghiaghili@gmail.com>
# Inspired by the R package naniar

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
from .compare import compare_imputations
from .timeseries import (
    miss_ts_summary,
    gap_table,
    vis_ts_miss,
    vis_gap_lengths,
    vis_miss_over_time,
    impute_ts,
)
# Register the df.miss accessor
from . import accessor  # noqa: F401

__all__ = [
    # summary
    "bind_shadow", "n_miss", "n_complete", "pct_miss", "pct_complete",
    "miss_var_summary", "miss_case_summary",
    # visualise — basic
    "matrix", "bar", "miss_case", "miss_var_pct", "vis_miss", "miss_which",
    # visualise — pattern
    "upset", "miss_patterns", "miss_cooccurrence",
    # visualise — correlation / clustering
    "heatmap", "dendrogram", "miss_cluster",
    # visualise — row/variable profiles
    "miss_row_profile", "miss_impute_compare",
    # visualise — shadow / MAR
    "shadow_scatter",
    # visualise — factor / group
    "vis_miss_fct", "vis_miss_by_group",
    # visualise — imputation diagnostics
    "vis_impute_dist",
    # visualise — misc
    "scatter_miss", "vis_miss_cumsum_var", "vis_miss_cumsum_case",
    "vis_miss_span", "vis_parallel_coords",
    # stats
    "mcar_test", "mar_mnar_test", "diagnose_missing",
    # manipulation
    "replace_with_na", "replace_with_na_all", "clean_names",
    "remove_empty", "coalesce_columns", "miss_as_feature",
    # impute — stateless
    "impute_mean", "impute_median", "impute_mode",
    "impute_knn", "impute_mice", "impute_rf", "impute_gb",
    # impute — stateful
    "FittedImputer", "make_imputer",
    # transformer
    "MissinglyImputer",
    # report
    "create_report",
    # compare
    "compare_imputations",
    # timeseries
    "miss_ts_summary", "gap_table", "vis_ts_miss",
    "vis_gap_lengths", "vis_miss_over_time", "impute_ts",
    # accessor
    "accessor",
]
