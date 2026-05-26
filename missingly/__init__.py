"""missingly — comprehensive missing-data analysis and imputation toolkit.

Public API
----------

**Visualisation**

.. autosummary::
   :nosignatures:

   matrix
   bar
   miss_case
   miss_var_pct
   vis_miss
   miss_which
   upset
   miss_patterns
   miss_cooccurrence
   heatmap
   dendrogram
   miss_cluster
   miss_row_profile
   shadow_scatter
   vis_miss_fct
   vis_miss_by_group
   vis_impute_dist
   miss_impute_compare
   scatter_miss
   vis_miss_cumsum_var
   vis_miss_cumsum_case
   vis_miss_span
   vis_parallel_coords

**Summary / stats**

.. autosummary::
   :nosignatures:

   miss_case_summary
   miss_var_summary

**Imputation**

.. autosummary::
   :nosignatures:

   impute_mean
   impute_median
   impute_mode
   impute_knn
   impute_mice
   impute_rf
   impute_gb
   make_imputer

**Multiple Imputation pooling utilities**

.. autosummary::
   :nosignatures:

   pool_scalar_estimates
   pool_linear_regression_results

**Comparison**

.. autosummary::
   :nosignatures:

   compare_imputation_methods

**Simulation**

.. autosummary::
   :nosignatures:

   simulate_mcar
   simulate_mar
   simulate_mnar

**Manipulation**

.. autosummary::
   :nosignatures:

   add_any_miss_var
   bind_shadow_matrix

**Accessor**

The ``DataFrame.miss`` accessor exposes the full API without extra imports::

    df.miss.summary()
    df.miss.vis_miss()
"""

from __future__ import annotations

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
# Summary / stats  (only names that actually exist in missingly.summary)
# ---------------------------------------------------------------------------
from missingly.summary import (
    miss_case_summary,
    miss_var_summary,
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
)

# ---------------------------------------------------------------------------
# Multiple Imputation pooling utilities
# ---------------------------------------------------------------------------
from missingly.mi import (
    pool_scalar_estimates,
    pool_linear_regression_results,
)

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
from missingly.compare import compare_imputation_methods

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
from missingly.simulate import (
    simulate_mcar,
    simulate_mar,
    simulate_mnar,
)

# ---------------------------------------------------------------------------
# Manipulation
# ---------------------------------------------------------------------------
from missingly.manipulation import (
    add_any_miss_var,
    bind_shadow_matrix,
)

# ---------------------------------------------------------------------------
# Accessor (side-effect import — registers df.miss)
# ---------------------------------------------------------------------------
import missingly.accessor  # noqa: F401

__all__ = [
    # Visualisation
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
    # Summary / stats
    "miss_case_summary",
    "miss_var_summary",
    # Imputation
    "impute_mean",
    "impute_median",
    "impute_mode",
    "impute_knn",
    "impute_mice",
    "impute_rf",
    "impute_gb",
    "make_imputer",
    # Multiple Imputation pooling utilities
    "pool_scalar_estimates",
    "pool_linear_regression_results",
    # Comparison
    "compare_imputation_methods",
    # Simulation
    "simulate_mcar",
    "simulate_mar",
    "simulate_mnar",
    # Manipulation
    "add_any_miss_var",
    "bind_shadow_matrix",
]

__version__ = "0.2.0"
