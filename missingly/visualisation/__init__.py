"""Public re-exports for the visualisation sub-package.

All public plot functions are defined in:
  * :mod:`missingly.visualisation.static`   – matplotlib-based
  * :mod:`missingly.visualisation.interactive` – Plotly wrappers

This ``__init__`` re-exports every public name so callers may write either::

    from missingly.visualisation import matrix, heatmap
    # or
    from missingly import visualise; visualise.matrix(...)
"""
from missingly.visualisation.static import (
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

__all__ = [
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
    "dendrogram",
    "miss_cluster",
    "miss_row_profile",
    "shadow_scatter",
    "vis_miss_fct",
    "vis_miss_by_group",
    "vis_impute_dist",
    "miss_impute_compare",
    "scatter_miss",
    "vis_miss_cumsum_var",
    "vis_miss_cumsum_case",
    "vis_miss_span",
    "vis_parallel_coords",
]
