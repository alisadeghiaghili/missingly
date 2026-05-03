# missingly - A Python library for missing data analysis
#
# Author: Ali Sadeghi Aghili <alisadeghiaghili@gmail.com>
# inspired by the R package naniar

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
)
from .stats import (
    mcar_test,
    mar_mnar_test,
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
)
from .report import (
    create_report,
)
from .compare import (
    compare_imputations,
)

__all__ = [
    "bind_shadow",
    "n_miss",
    "n_complete",
    "pct_miss",
    "pct_complete",
    "miss_var_summary",
    "miss_case_summary",
    "matrix",
    "bar",
    "upset",
    "dendrogram",
    "scatter_miss",
    "miss_case",
    "vis_impute_dist",
    "vis_miss_fct",
    "vis_miss_cumsum_var",
    "vis_miss_cumsum_case",
    "vis_miss_span",
    "vis_parallel_coords",
    "heatmap",
    "vis_miss",
    "miss_var_pct",
    "miss_cluster",
    "miss_which",
    "mcar_test",
    "mar_mnar_test",
    "replace_with_na",
    "replace_with_na_all",
    "clean_names",
    "remove_empty",
    "coalesce_columns",
    "miss_as_feature",
    "impute_mean",
    "impute_median",
    "impute_mode",
    "impute_knn",
    "impute_mice",
    "impute_rf",
    "impute_gb",
    "create_report",
    "compare_imputations",
]
