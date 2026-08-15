"""Global configuration for the missingly package.

All hardcoded thresholds and behavioural flags previously scattered across
``impute.py`` are centralised here as a single :class:`MissinglyConfig`
dataclass. A package-level singleton :data:`config` is provided so that
callers can adjust settings once and have them take effect everywhere::

    import missingly
    missingly.config.large_df_threshold = 100_000
    missingly.config.strict_mode = True

Each imputation function still accepts a ``strict_mode`` keyword argument
that takes precedence over the global setting for that one call.

Notes
-----
This module has **no** imports from other ``missingly`` sub-modules to
avoid circular imports. It imports only from the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = ["MissinglyConfig", "config"]


@dataclass
class MissinglyConfig:
    """Package-wide configuration settings.

    All attributes can be mutated at runtime; changes take effect on the
    next function call that reads them.

    Attributes
    ----------
    large_df_threshold : int
        Row count above which imputation functions emit a
        :class:`UserWarning` about potentially long runtimes or high memory
        usage. Default is 50 000.
    knn_cat_neighbors_threshold : int
        When KNN is applied to a DataFrame where categorical columns
        outnumber numeric columns, a :class:`UserWarning` is emitted if
        ``n_neighbors`` exceeds this value. Default is 5.
    strict_mode : bool
        Global default for the ``strict_mode`` parameter accepted by
        ``impute_logreg``, ``impute_rf``, and ``impute_gb``. When ``True``,
        estimator failures raise
        :class:`~missingly.exceptions.ImputationError` instead of falling
        back to a column mean/mode. Default is ``False``. The imputation
        functions validate this value at call time and reject values other
        than exactly ``True`` or ``False``. Individual call-site
        ``strict_mode`` kwargs override this setting.

    Examples
    --------
    Raise on any imputation failure globally:

    >>> import missingly
    >>> missingly.config.strict_mode = True

    Lower the large-DataFrame warning threshold:

    >>> missingly.config.large_df_threshold = 10_000

    Reset to defaults:

    >>> missingly.config = missingly.MissinglyConfig()
    """

    large_df_threshold: int = 50_000
    knn_cat_neighbors_threshold: int = 5
    strict_mode: bool = False


#: Package-level singleton.  Import and mutate this object to change
#: behaviour globally:  ``missingly.config.strict_mode = True``.
config: MissinglyConfig = MissinglyConfig()
