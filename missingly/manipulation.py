"""Deprecated shim — manipulation logic moved to ``missingly.utils.manipulation``.

This module is kept for backward compatibility only.  All symbols are
re-exported from :mod:`missingly.utils.manipulation`.  Import from there
directly for new code::

    # Preferred (new code)
    from missingly.utils.manipulation import replace_with_na

    # Still works but will emit DeprecationWarning in v0.4.0
    from missingly.manipulation import replace_with_na

def bind_shadow_matrix(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Return the shadow matrix of a DataFrame as a standalone DataFrame.

    Each column of the returned DataFrame corresponds to one column of
    the input, renamed ``<col>_NA``, and contains ``True`` where the
    original value is missing and ``False`` where it is present.

    Unlike :func:`~missingly.summary.bind_shadow` (which concatenates the
    shadow alongside the original data), this function returns **only**
    the shadow matrix — useful when you want to analyse or visualise the
    missingness pattern independently.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  Not modified in place.
    missing_values : list, optional
        Additional scalar sentinels treated as missing alongside ``NaN``.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_rows, n_cols)`` with boolean dtype, column names
        ``["<original_col>_NA", ...]``, and the same index as *df*.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [np.nan, 2.0, 3.0]})
    >>> bind_shadow_matrix(df)
        a_NA   b_NA
    0  False   True
    1   True  False
    2  False  False

    With a sentinel value:

    >>> df2 = pd.DataFrame({'x': [0, -99, 2], 'y': [1, 2, -99]})
    >>> bind_shadow_matrix(df2, missing_values=[-99])
        x_NA   y_NA
    0  False  False
    1   True  False
    2  False   True
    """
    shadow = df.isnull()
    if missing_values is not None:
        shadow = shadow | df.isin(missing_values)
    shadow.columns = [f"{col}_NA" for col in df.columns]
    return shadow


@deprecated_api(
    "Use `from data_quality_toolkit.cleaning import clean_names` instead.",
    since="0.2.0",
)
def clean_names(*args, **kwargs):
    """Legacy shim — emits :class:`FutureWarning`.

    .. deprecated:: 0.2.0
        Moved to ``data_quality_toolkit.cleaning``.
    """
    warnings.warn(
        "clean_names moved to data_quality_toolkit.cleaning and will be "
        "removed from missingly in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from data_quality_toolkit.cleaning import clean_names as _clean
    except ImportError as exc:
        raise ImportError(
            "data_quality_toolkit is required for clean_names(). "
            "Install it with: pip install data-quality-toolkit"
        ) from exc
    return _clean(*args, **kwargs)


@deprecated_api(
    "Use `from data_quality_toolkit.cleaning import remove_empty` instead.",
    since="0.2.0",
)
def remove_empty(*args, **kwargs):
    """Legacy shim — emits :class:`FutureWarning`.

    .. deprecated:: 0.2.0
        Moved to ``data_quality_toolkit.cleaning``.
    """
    warnings.warn(
        "remove_empty moved to data_quality_toolkit.cleaning and will be "
        "removed from missingly in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from data_quality_toolkit.cleaning import remove_empty as _remove
    except ImportError as exc:
        raise ImportError(
            "data_quality_toolkit is required for remove_empty(). "
            "Install it with: pip install data-quality-toolkit"
        ) from exc
    return _remove(*args, **kwargs)


@deprecated_api(
    "Use `from data_quality_toolkit.cleaning import coalesce_columns` instead.",
    since="0.2.0",
)
def coalesce_columns(*args, **kwargs):
    """Legacy shim — emits :class:`FutureWarning`.

    .. deprecated:: 0.2.0
        Moved to ``data_quality_toolkit.cleaning``.
    """
    warnings.warn(
        "coalesce_columns moved to data_quality_toolkit.cleaning and will be "
        "removed from missingly in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from data_quality_toolkit.cleaning import coalesce_columns as _coal
    except ImportError as exc:
        raise ImportError(
            "data_quality_toolkit is required for coalesce_columns(). "
            "Install it with: pip install data-quality-toolkit"
        ) from exc
    return _coal(*args, **kwargs)

from __future__ import annotations

# Re-export everything from the canonical location so existing imports
# continue to work without modification.
from missingly.utils.manipulation import (  # noqa: F401
    replace_with_na,
    replace_with_na_all,
    add_any_miss_var,
    bind_shadow_matrix,
    clean_names,
    remove_empty,
    coalesce_columns,
    miss_as_feature,
)

__all__ = [
    "replace_with_na",
    "replace_with_na_all",
    "add_any_miss_var",
    "bind_shadow_matrix",
    "clean_names",
    "remove_empty",
    "coalesce_columns",
    "miss_as_feature",
]
