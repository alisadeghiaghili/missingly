"""Sklearn-compatible imputation transformer for missingly.

This module exposes :class:`MissinglyImputer`, a ``BaseEstimator`` /
``TransformerMixin`` subclass that wraps all missingly imputation
strategies in a standard scikit-learn ``fit`` / ``transform`` interface.

This allows ``MissinglyImputer`` to be used inside
``sklearn.pipeline.Pipeline``, ``GridSearchCV``, ``cross_val_score``,
and any other sklearn-compatible tooling without modification.

Design decisions
----------------
* ``fit`` learns the parameters required by each strategy (e.g. column
  means for ``"mean"``, or trains the KNN / RF / GB model on the
  training set) and stores them so ``transform`` can apply them to
  unseen data without re-fitting.
* The underlying sklearn imputers are stored as instance attributes so
  they are serialisable via ``pickle`` / ``joblib`` (required for
  production deployment).
* Categorical columns are handled consistently with the rest of
  missingly: ordinal-encoded for KNN/MICE, classifier-based for RF/GB.
* Column dtypes are classified once at ``fit`` time by the shared layer in
  :mod:`missingly._dtypes`, the same one the functions in
  :mod:`missingly.impute` use: numeric columns (NumPy or nullable) are
  modelled as ``float64`` and restored to their dtype after ``transform``;
  boolean and categorical columns are imputed as categories; datetime,
  timedelta and mixed-type ``object`` columns are never modelled by the
  model-based strategies (see the *Notes* of :class:`MissinglyImputer`).
* ``fit_transform`` is inherited from ``TransformerMixin`` and calls
  ``fit`` then ``transform``, so it is equivalent to calling both
  separately.
* Fitted state is tracked via ``_is_fitted`` (a bool set to ``True``
  at the end of ``fit``). ``check_is_fitted`` inspects this attribute,
  avoiding the false-positive that occurs when mutable defaults such as
  ``[]`` are set in ``__init__``.
* ``GradientBoostingRegressor`` and ``GradientBoostingClassifier`` do not
  accept NaN in feature matrices.  Before fitting or predicting with GB,
  ``_fill_nan_with_col_means`` is applied to feature arrays.
* ``estimator_kwargs`` is stored as a single ``Optional[dict]`` parameter
  (not ``**kwargs``) so that ``get_params()`` / ``set_params()`` /
  ``clone()`` work correctly per the sklearn estimator contract.
* Strategies exposed through this transformer must be inductive: ``fit`` may
  learn only from training rows and ``transform`` may not refit or use an
  evaluation row as a donor.  Unsupported inductive variants fail loudly.
* ``strategy="hotdeck"`` with ``hotdeck_method="random"`` samples only from
  the training donor pool. Sequential and weighted hot-deck remain available
  as standalone, transductive functions but are not sklearn-transformer modes.

Examples
--------
>>> import pandas as pd, numpy as np
>>> from sklearn.pipeline import Pipeline
>>> from sklearn.ensemble import RandomForestClassifier
>>> from missingly.transformer import MissinglyImputer
>>>
>>> X_train = pd.DataFrame({'age': [25, np.nan, 35], 'city': ['A', 'B', np.nan]})
>>> X_test  = pd.DataFrame({'age': [np.nan, 40],     'city': ['A', np.nan]})
>>>
>>> imputer = MissinglyImputer(strategy='mean')
>>> imputer.fit(X_train)
MissinglyImputer()
>>> X_train_imputed = imputer.transform(X_train)
>>> X_test_imputed  = imputer.transform(X_test)   # uses train statistics only
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.preprocessing import OrdinalEncoder

from ._dtypes import (
    _OPAQUE_REASONS,
    ColumnKind,
    classify_column,
    classify_frame,
    column_mode,
    fill_with_statistic,
    is_numeric_kind,
    resolve_opaque_targets,
    restore_numeric,
    to_float_array,
)
from .exceptions import InsufficientDataError, UnsupportedDtypeError
from .impute import (
    _LEGACY_ENCODED,
    _fill_nan_with_col_means,
    _restore_categorical_dtype,
)


_VALID_STRATEGIES = frozenset(
    {
        "mean", "median", "mode", "knn", "mice", "rf", "gb", "pmm",
        "logreg", "polyreg", "polr",
        "hotdeck",
    }
)
# Strategies that fill every column with one learned value (a statistic or a mode).
_SIMPLE_STRATEGIES = frozenset({"mean", "median", "mode"})
# Strategies that fit a model on the encoded frame: opaque (datetime, timedelta,
# mixed-type object) columns cannot be modelled there.
_MODEL_STRATEGIES = frozenset({"knn", "mice", "rf", "gb"})
# Strategies whose imputed integer cells are rounded to the nearest integer; the
# others (mean, median, mode, hot-deck) fill with a statistic or an observed
# value and never round.
_ROUNDING_STRATEGIES = frozenset({"knn", "mice", "pmm", "rf", "gb"})
_NOT_FITTED_REASON = (
    "the column had no missing value at fit time, so no model was trained for it"
)


class MissinglyImputer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible imputation transformer.

    Wraps all missingly imputation strategies in a standard
    ``fit`` / ``transform`` interface so the imputer can be embedded
    in ``sklearn.pipeline.Pipeline`` without data leakage.

    ``fit`` learns imputation parameters from the **training** data only.
    ``transform`` applies those learned parameters to any DataFrame
    (train or test), ensuring that test-set information never leaks into
    the imputed values.

    Parameters
    ----------
    strategy : str, optional
        Imputation strategy.  One of:

        * ``"mean"``      — fill numeric with mean, categorical with mode.
        * ``"median"``    — fill numeric with median, categorical with mode.
        * ``"mode"``      — fill all columns with most frequent value.
        * ``"knn"``       — Euclidean k-Nearest Neighbours (default k=5).
        * ``"mice"``      — Multiple Imputation by Chained Equations.
        * ``"rf"``        — Random Forest.
        * ``"gb"``        — Gradient Boosting.
        * ``"pmm"``       — Predictive Mean Matching using training donors;
          a predictor-free numeric target uses deterministic train-donor draws.
        * ``"hotdeck"``   — Train-donor random hot-deck.

        ``"logreg"``, ``"polyreg"``, ``"polr"``, mixed-metric KNN, and
        sequential/weighted hot-deck are available as standalone transductive
        functions, but intentionally raise :class:`NotImplementedError` here:
        their current implementations do not yet have a safe inductive
        train/transform contract.

        Default is ``"mean"``.
    n_neighbors : int, optional
        Number of neighbours for ``strategy="knn"``.  Default 5.
    metric : {"euclidean", "mixed"}, optional
        Distance metric for ``strategy="knn"``. ``"mixed"`` is rejected in
        this transformer until a train-only Gower donor implementation exists;
        use the standalone transductive API when that behavior is intended.
    max_iter : int, optional
        Maximum EM/MICE/RF/GB iterations.  Default 10.
    random_state : int, optional
        Random seed for reproducibility.  Default 0.
    estimator_kwargs : dict or None, optional
        Additional keyword arguments forwarded to the underlying sklearn
        estimator (e.g. ``{'n_estimators': 200}`` for RF/GB strategies).
        Stored as-is so that ``get_params()`` / ``set_params()`` /
        ``clone()`` behave correctly.
    hotdeck_method : {"random", "sequential", "weighted"}, optional
        Hot-deck variant. Only ``"random"`` is inductive and accepted here;
        the other values raise :class:`NotImplementedError`.
    hotdeck_direction : {"forward", "backward", "nearest"}, optional
        Direction for ``hotdeck_method="sequential"``.
        Default is ``"nearest"``.
    hotdeck_n_donors : int, optional
        Donor pool size for ``hotdeck_method="weighted"``.
        Default is 5.

    Notes
    -----
    ``MissinglyImputer`` accepts pandas DataFrames only. Array and sparse
    containers are rejected so column names, dtypes, and train/transform schema
    checks remain explicit. DataFrame column labels must be unique.

    **Dtype handling.** The kind of every column is decided once, at ``fit``
    time, by the dtype layer shared with :mod:`missingly.impute`:

    * *Numeric* (``int*``, ``uint*``, ``Int*``, ``UInt*``, ``float*``,
      ``Float*``): modelled as ``float64`` with ``NaN`` for missing, so
      ``pd.NA`` never reaches NumPy or scikit-learn.  Columns without missing
      values are returned exactly as given (an ``int64`` ID column stays
      ``int64``).  Imputed cells are cast back to the column dtype.
    * *Integer targets* of ``"knn"``, ``"mice"``, ``"pmm"``, ``"rf"`` and
      ``"gb"`` are rounded to the nearest integer (ties to even) and clipped to
      the dtype range, so ``Int64`` stays ``Int64``.  ``"mean"`` and
      ``"median"`` never round: a nullable integer column keeps its dtype when
      the fill value is integral and is returned as ``Float64`` otherwise.
      ``"mode"`` and ``"hotdeck"`` copy observed values, so the dtype is kept.
    * *Boolean and categorical* (``bool``, ``boolean``, ``category``,
      ``string``, ``object``): imputed as categories and returned in their own
      dtype (a ``boolean`` target stays ``boolean``).
    * *Opaque* (datetime with or without a time zone, timedelta, ``object``
      columns mixing value types): ``"mean"``, ``"median"``, ``"mode"``,
      ``"pmm"`` and ``"hotdeck"`` fill them with observed values.  The
      model-based strategies (``"knn"``, ``"mice"``, ``"rf"``, ``"gb"``) pass a
      complete opaque column through unchanged and do not use it as a
      predictor; an incomplete one raises
      :class:`~missingly.exceptions.UnsupportedDtypeError`, except an
      incomplete datetime column, which is still imputed as a nominal category
      but emits a ``FutureWarning`` because it will raise in a future release.
      Missing cells that appear at ``transform`` time in an opaque column that
      was complete at ``fit`` time raise
      :class:`~missingly.exceptions.UnsupportedDtypeError`, except in a
      datetime column, which is filled with its training mode and emits a
      ``FutureWarning`` for the same reason.
    * ``"mean"``, ``"median"`` and ``"mode"`` raise
      :class:`~missingly.exceptions.InsufficientDataError` for a column with no
      observed value.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> frame = pd.DataFrame({"score": [1.0, None, 3.0]})
    >>> pipe = Pipeline([("imputer", MissinglyImputer(
    ...     strategy="hotdeck", hotdeck_method="random", random_state=0
    ... ))])
    >>> result = pipe.fit_transform(frame)
    >>> bool(result.isna().any().any())
    False
    """

    def __init__(
        self,
        strategy: str = "mean",
        n_neighbors: int = 5,
        metric: str = "euclidean",
        max_iter: int = 10,
        random_state: int = 0,
        estimator_kwargs: Optional[dict] = None,
        hotdeck_method: str = "random",
        hotdeck_direction: str = "nearest",
        hotdeck_n_donors: int = 5,
    ) -> None:
        self._validate_strategy(strategy)
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs
        self.hotdeck_method = hotdeck_method
        self.hotdeck_direction = hotdeck_direction
        self.hotdeck_n_donors = hotdeck_n_donors
        self._is_fitted: bool = False

    @staticmethod
    def _validate_strategy(strategy: str) -> None:
        """Validate a public imputation strategy name.

        Parameters
        ----------
        strategy : str
            Requested imputation strategy.

        Returns
        -------
        None
            The function returns normally only for supported strategies.

        Raises
        ------
        ValueError
            If ``strategy`` is not a supported strategy.

        Examples
        --------
        >>> MissinglyImputer._validate_strategy("mean")
        >>> try:
        ...     MissinglyImputer._validate_strategy("unknown")
        ... except ValueError as error:
        ...     "strategy" in str(error) and "unknown" in str(error)
        True
        """
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_VALID_STRATEGIES)}; "
                f"got {strategy!r}"
            )

    def set_params(self, **params: object) -> "MissinglyImputer":
        """Set estimator parameters while preserving strategy validation.

        Parameters
        ----------
        **params : object
            Estimator parameters accepted by scikit-learn's ``BaseEstimator``.
            When provided, ``strategy`` must be one of the documented
            ``MissinglyImputer`` strategies.

        Returns
        -------
        MissinglyImputer
            This estimator, as required by the scikit-learn estimator API.

        Raises
        ------
        ValueError
            If ``strategy`` is unsupported or another parameter name is
            unknown to ``BaseEstimator``.

        Examples
        --------
        >>> imputer = MissinglyImputer().set_params(strategy="median")
        >>> imputer.strategy
        'median'
        """
        if "strategy" in params:
            strategy = params["strategy"]
            if not isinstance(strategy, str):
                raise ValueError("strategy must be a string")
            self._validate_strategy(strategy)
        return super().set_params(**params)

    # ------------------------------------------------------------------
    # sklearn fitted-state contract
    # ------------------------------------------------------------------

    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Replace Python None with np.nan in object/string columns."""
        result = df.copy()
        obj_cols = result.select_dtypes(include=["object", "string"]).columns
        if len(obj_cols):
            result[obj_cols] = result[obj_cols].where(
                result[obj_cols].notna(), other=np.nan
            )
        return result

    @staticmethod
    def _label_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Return categorical columns as ``object`` with ``NaN`` for missing cells.

        Parameters
        ----------
        df : pd.DataFrame
            Columns of any non-numeric kind (``bool``, ``boolean``, ``string``,
            ``category``, ``object``, datetime).

        Returns
        -------
        pd.DataFrame
            ``object`` copy in which ``pd.NA`` and ``NaT`` are ``numpy.nan``, the
            representation scikit-learn's ordinal encoder understands.

        Examples
        --------
        >>> frame = pd.DataFrame({"flag": pd.array([True, None], dtype="boolean")})
        >>> MissinglyImputer._label_frame(frame)["flag"].tolist()
        [True, nan]
        """
        columns = {}
        for column in df.columns:
            values = df[column].astype(object).to_numpy(copy=True)
            values[df[column].isna().to_numpy()] = np.nan
            columns[column] = values
        return pd.DataFrame(columns, index=df.index, columns=df.columns, dtype=object)

    @staticmethod
    def _fill_label(series: pd.Series, value: object, dtype: object) -> pd.Series:
        """Fill the missing cells of a categorical column with one label.

        Parameters
        ----------
        series : pd.Series
            Column to fill; not modified.
        value : object
            Label learned at fit time (a mode).
        dtype : dtype
            Fit-time dtype to restore.

        Returns
        -------
        pd.Series
            Filled column in ``dtype``.

        Examples
        --------
        >>> filled = MissinglyImputer._fill_label(
        ...     pd.Series(["a", None]), "a", pd.StringDtype()
        ... )
        >>> filled.tolist(), str(filled.dtype)
        (['a', 'a'], 'string')
        """
        filled = series.astype(object)
        filled.loc[series.isna()] = value
        return _restore_categorical_dtype(filled, dtype)

    def _modelled_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the columns an imputer models, numeric ones as ``float64``.

        Parameters
        ----------
        X : pd.DataFrame
            Frame with the fitted schema.

        Returns
        -------
        pd.DataFrame
            Fit-time ``model_cols_`` of ``X``; numeric columns (NumPy or nullable)
            are ``float64`` with ``NaN`` for missing, every other column keeps its
            dtype.  Opaque pass-through columns are left out.

        Examples
        --------
        >>> frame = pd.DataFrame({"n": pd.array([1, None], dtype="Int64")})
        >>> imputer = MissinglyImputer().fit(frame)
        >>> str(imputer._modelled_frame(frame)["n"].dtype)
        'float64'
        """
        modelled = X[self.model_cols_].copy()
        for column in self.numeric_cols_:
            modelled[column] = to_float_array(X[column])
        return modelled

    def _coerce_transform_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Represent fitted categorical columns as objects before encoding.

        Parameters
        ----------
        df : pd.DataFrame
            Schema-validated transform input in fit-time column order.

        Returns
        -------
        pd.DataFrame
            Copy with categorical columns represented as ``object`` so an
            all-missing serving batch cannot select sklearn's numeric branch.

        Examples
        --------
        A one-row ``float64`` all-missing categorical column is converted before
        it reaches the fitted ``OrdinalEncoder``.
        """
        result = df.copy()
        for column in self.cat_cols_:
            if pd.api.types.is_numeric_dtype(result[column]):
                result[column] = result[column].astype(object)
        return result

    def _encode_cats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordinal-encode fitted categorical columns using object input.

        Parameters
        ----------
        df : pd.DataFrame
            Input with the fitted schema and column order.

        Returns
        -------
        pd.DataFrame
            Numeric feature matrix suitable for sklearn estimators.

        Examples
        --------
        Missing categorical values remain encoded as ``np.nan``; observed
        unseen values receive the encoder's configured unknown representation.
        """
        df_work = df.copy()
        if self.cat_cols_ and self.encoder_ is not None:
            df_work[self.cat_cols_] = self.encoder_.transform(
                self._label_frame(df_work[self.cat_cols_])
            )
        return df_work.astype(float)

    def _restore_observed_unknown_categories(
        self,
        result: pd.DataFrame,
        original: pd.DataFrame,
    ) -> pd.DataFrame:
        """Restore observed categories outside the fitted training vocabulary.

        Parameters
        ----------
        result : pd.DataFrame
            Decoded imputation result produced from encoded features.
        original : pd.DataFrame
            Canonicalized transform input before encoding.

        Returns
        -------
        pd.DataFrame
            Result in which observed unknown categories are preserved rather
            than treated as values requiring imputation.

        Examples
        --------
        If training observed ``"a"`` and ``"b"`` while a serving row contains
        ``"unseen"``, the returned row still contains ``"unseen"``.
        """
        if not self.cat_cols_ or self.encoder_ is None:
            return result

        restored = result.copy()
        for position, column in enumerate(self.cat_cols_):
            source = original[column]
            known_values = pd.Index(self.encoder_.categories_[position])
            unknown_mask = source.notna() & ~source.isin(known_values)
            if not unknown_mask.any():
                continue

            if isinstance(restored[column].dtype, pd.CategoricalDtype):
                additions = pd.Index(source.loc[unknown_mask].unique()).difference(
                    restored[column].cat.categories
                )
                if len(additions):
                    restored[column] = restored[column].cat.add_categories(additions)
            restored.loc[unknown_mask, column] = source.loc[unknown_mask]
        return restored

    def _decode_cats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse-transform ordinal-encoded categorical columns."""
        if not self.cat_cols_ or self.encoder_ is None:
            return df
        result = df.copy()
        cat_array = result[self.cat_cols_].to_numpy().copy().astype(float)
        for i, col in enumerate(self.cat_cols_):
            n_cats = len(self.encoder_.categories_[i])
            cat_array[:, i] = np.clip(np.round(cat_array[:, i]), 0, n_cats - 1)
        decoded = self.encoder_.inverse_transform(cat_array)
        for i, col in enumerate(self.cat_cols_):
            result[col] = decoded[:, i]
            result[col] = _restore_categorical_dtype(result[col], self.cat_dtypes_[col])
        return result

    def _learn_schema(self, X: pd.DataFrame) -> None:
        """Record the kind of every column and split modelled from opaque ones.

        Model-based strategies (``knn``, ``mice``, ``rf``, ``gb``) cannot model
        datetime, timedelta or mixed-type ``object`` columns: complete ones are
        passed through and are not predictors, incomplete ones raise
        :class:`~missingly.exceptions.UnsupportedDtypeError`, except incomplete
        datetime columns, which the previous release imputed as nominal
        categories and which now only warn.  Mode-based strategies (``mean``,
        ``median``, ``mode``, ``pmm``, ``hotdeck``) fill every non-numeric kind
        with observed values, so nothing is set aside for them.

        Parameters
        ----------
        X : pd.DataFrame
            Normalised training frame.

        Raises
        ------
        UnsupportedDtypeError
            For an incomplete opaque column the strategy cannot model.

        Warns
        -----
        FutureWarning
            For an incomplete datetime column of a model-based strategy.

        Examples
        --------
        >>> frame = pd.DataFrame({"n": [1.0, None], "t": pd.to_timedelta([1, 2], unit="D")})
        >>> imputer = MissinglyImputer(strategy="knn")
        >>> imputer._learn_schema(frame)
        >>> imputer.passthrough_cols_, imputer.model_cols_
        (('t',), ['n'])
        """
        kinds = classify_frame(X)
        opaque: List[Hashable] = []
        if self.strategy in _MODEL_STRATEGIES:
            kinds, legacy = resolve_opaque_targets(
                X, self.strategy, kinds=kinds, deprecated=_LEGACY_ENCODED
            )
            opaque = [
                column
                for column, kind in kinds.items()
                if kind in _OPAQUE_REASONS and column not in legacy
            ]
        self.column_kinds_: Dict[Hashable, ColumnKind] = kinds
        self.passthrough_cols_: Tuple[Hashable, ...] = tuple(opaque)
        self.model_cols_: List[str] = [c for c in X.columns if c not in opaque]
        self.numeric_cols_: List[str] = [
            c for c in self.model_cols_ if is_numeric_kind(kinds[c])
        ]
        self.cat_cols_: List[str] = [
            c for c in self.model_cols_ if c not in self.numeric_cols_
        ]
        self.cat_dtypes_: Dict[str, object] = {c: X[c].dtype for c in self.cat_cols_}
        # Datetime columns set aside because they were complete keep a mode, for the
        # deprecated serving-time case in which they suddenly have gaps.
        self.fill_values_: Dict[Hashable, Any] = {
            column: column_mode(X[column], ColumnKind.DATETIME)
            for column in opaque
            if kinds[column] is ColumnKind.DATETIME and X[column].notna().any()
        }

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    @staticmethod
    def _require_unique_columns(X: pd.DataFrame, *, operation: str) -> None:
        """Reject ambiguous DataFrame schemas before estimator validation.

        Parameters
        ----------
        X : pd.DataFrame
            Input frame whose column labels are checked.
        operation : str
            Public operation name used in the actionable error message.

        Raises
        ------
        ValueError
            If ``X`` has duplicate column labels.

        Returns
        -------
        None
            This method returns after validating that the column labels are unique.

        Examples
        --------
        >>> frame = pd.DataFrame([[1.0, 2.0]], columns=["x", "x"])
        >>> try:
        ...     MissinglyImputer._require_unique_columns(frame, operation="fit")
        ... except ValueError as error:
        ...     "fit X columns must be unique" in str(error)
        True
        """
        if not X.columns.is_unique:
            raise ValueError(
                f"{operation} X columns must be unique; "
                "duplicate labels are not supported."
            )

    def fit(self, X: pd.DataFrame, y=None) -> "MissinglyImputer":
        """Learn imputation parameters from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data that may contain missing values.
        y : ignored
            Present for sklearn API compatibility.

        Returns
        -------
        self
            Fitted transformer.

        Raises
        ------
        TypeError
            If ``X`` is not a pandas DataFrame.
        ValueError
            If ``X`` has duplicate column labels.
        InsufficientDataError
            For ``"mean"``, ``"median"`` and ``"mode"``: a column has no
            observed value.
        UnsupportedDtypeError
            For the model-based strategies: an incomplete timedelta,
            mixed-type ``object`` or other opaque column.

        Warns
        -----
        FutureWarning
            For the model-based strategies: an incomplete datetime column,
            which is imputed as a nominal category for now and will raise
            :class:`~missingly.exceptions.UnsupportedDtypeError` later.

        Examples
        --------
        >>> frame = pd.DataFrame({"score": [1.0, None, 3.0]})
        >>> result = MissinglyImputer(strategy="mean").fit(frame).transform(frame)
        >>> bool(result.isna().any().any())
        False
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"MissinglyImputer expects a pandas DataFrame; got {type(X)}"
            )
        self._require_unique_columns(X, operation="fit")
        X = self._normalize(X)

        self.feature_names_in_: List[str] = X.columns.tolist()
        self._learn_schema(X)
        self.imputer_ = None
        self.encoder_: Optional[OrdinalEncoder] = None
        self.rf_reg_models_: Dict[
            str, RandomForestRegressor | GradientBoostingRegressor
        ] = {}
        self.rf_clf_models_: Dict[
            str, RandomForestClassifier | GradientBoostingClassifier
        ] = {}
        self.cat_label_encoders_: Dict[str, OrdinalEncoder] = {}
        self._knn_train_df_: Optional[pd.DataFrame] = None
        self._train_df_: Optional[pd.DataFrame] = None
        self._tree_feature_fill_values_: Optional[np.ndarray] = None

        self._fit_strategy(self._modelled_frame(X))

        self._is_fitted = True
        return self

    def _fit_strategy(self, X: pd.DataFrame) -> None:
        """Dispatch to the fit routine of the configured strategy.

        Parameters
        ----------
        X : pd.DataFrame
            Modelled training frame (numeric columns ``float64``).

        Raises
        ------
        NotImplementedError
            For strategies and variants without an inductive implementation.
        """
        strategy = self.strategy

        if strategy in _SIMPLE_STRATEGIES:
            self._fit_simple(X, num_strategy=None if strategy == "mode" else strategy)
        elif strategy == "knn":
            if self.metric == "mixed":
                raise NotImplementedError(
                    "MissinglyImputer(metric='mixed') is not available in "
                    "inductive mode because the current Gower implementation "
                    "would use transform rows as donors. Use metric='euclidean' "
                    "or the standalone transductive function explicitly."
                )
            self._fit_knn(X)
        elif strategy == "mice":
            self._fit_mice(X)
        elif strategy == "pmm":
            self._train_df_ = X.copy()
        elif strategy in ("rf", "gb"):
            self._fit_tree(X)
        elif strategy in ("logreg", "polyreg", "polr"):
            raise NotImplementedError(
                f"MissinglyImputer(strategy={strategy!r}) has no inductive "
                "implementation and would refit during transform. Use the "
                "standalone function explicitly or choose an inductive strategy."
            )
        elif strategy == "hotdeck":
            if self.hotdeck_method != "random":
                raise NotImplementedError(
                    "Only random hot-deck is available in MissinglyImputer "
                    "because sequential and weighted variants need an explicit "
                    "transductive donor policy."
                )
            self._train_df_ = X.copy()

    def _fit_encoder(self, X: pd.DataFrame) -> None:
        """Fit the ordinal encoder of the categorical columns, if there are any."""
        if not self.cat_cols_:
            return
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        self.encoder_.fit(self._label_frame(X[self.cat_cols_]))

    def _fit_simple(self, X: pd.DataFrame, num_strategy: Optional[str]) -> None:
        """Learn one fill value per column for mean, median and mode.

        Parameters
        ----------
        X : pd.DataFrame
            Modelled training frame.
        num_strategy : {"mean", "median"} or None
            Statistic for numeric columns; ``None`` fills every column with its
            most frequent value.  Non-numeric columns always use the mode.

        Raises
        ------
        InsufficientDataError
            If a column has no observed value.
        """
        for column in X.columns:
            if X[column].isna().all():
                raise InsufficientDataError(
                    column=str(column), n_observed=0, n_required=1
                )
        statistic_cols = self.numeric_cols_ if num_strategy else []
        if statistic_cols:
            imputer = SimpleImputer(strategy=num_strategy)
            imputer.fit(X[statistic_cols])
            self.imputer_ = imputer
            self.fill_values_.update(zip(statistic_cols, imputer.statistics_))
        for column in X.columns:
            if column not in self.fill_values_:
                self.fill_values_[column] = column_mode(
                    X[column], self.column_kinds_[column]
                )

    def _fit_knn(self, X: pd.DataFrame) -> None:
        """Fit KNN imputer — euclidean or mixed metric."""
        if self.metric == "mixed":
            self._knn_train_df_ = X.copy()
            return
        self._fit_encoder(X)
        X_enc = self._encode_cats(X)
        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        imputer.fit(X_enc)
        self.imputer_ = imputer

    def _fit_mice(self, X: pd.DataFrame) -> None:
        """Fit ordinal encoder + IterativeImputer."""
        self._fit_encoder(X)
        X_enc = self._encode_cats(X)
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            random_state=self.random_state,
            imputation_order="roman",
        )
        imputer.fit(X_enc)
        self.imputer_ = imputer

    def _fit_tree(self, X: pd.DataFrame) -> None:
        """Fit per-column regressor/classifier for rf and gb strategies."""
        cat_set = set(self.cat_cols_)
        self._fit_encoder(X)

        X_enc = self._encode_cats(X).values
        fill_values = np.nanmean(X_enc, axis=0)
        self._tree_feature_fill_values_ = np.where(
            np.isfinite(fill_values), fill_values, 0.0
        )
        col_index = {col: i for i, col in enumerate(X.columns)}

        for col in X.columns:
            train_mask = X[col].notna()
            if train_mask.sum() < 2:
                continue

            feature_idx = [col_index[c] for c in X.columns if c != col]
            X_train = X_enc[np.ix_(np.where(train_mask)[0], feature_idx)]

            if self.strategy == "gb":
                X_train = _fill_nan_with_col_means(X_train)

            if col in cat_set:
                self._fit_tree_classifier(col, X_train, X.loc[train_mask, col])
            else:
                self._fit_tree_regressor(col, X_train, X.loc[train_mask, col])

    def _fit_tree_classifier(
        self, col: str, X_train: np.ndarray, labels: pd.Series
    ) -> None:
        """Fit the classifier of one categorical column (labels are observed)."""
        est_kwargs = self.estimator_kwargs or {}
        enc_y = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        y_train_enc = (
            enc_y.fit_transform(labels.to_numpy(dtype=object).reshape(-1, 1))
            .ravel()
            .astype(int)
        )
        clf = (
            RandomForestClassifier(random_state=self.random_state, **est_kwargs)
            if self.strategy == "rf"
            else GradientBoostingClassifier(
                random_state=self.random_state, **est_kwargs
            )
        )
        clf.fit(X_train, y_train_enc)
        self.rf_clf_models_[col] = clf
        self.cat_label_encoders_[col] = enc_y

    def _fit_tree_regressor(
        self, col: str, X_train: np.ndarray, target: pd.Series
    ) -> None:
        """Fit the regressor of one numeric column (target is ``float64``)."""
        est_kwargs = self.estimator_kwargs or {}
        reg = (
            RandomForestRegressor(random_state=self.random_state, **est_kwargs)
            if self.strategy == "rf"
            else GradientBoostingRegressor(random_state=self.random_state, **est_kwargs)
        )
        reg.fit(X_train, target.to_numpy())
        self.rf_reg_models_[col] = reg

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def _require_fitted_imputer(
        self,
    ) -> SimpleImputer | KNNImputer | IterativeImputer:
        """Return the strategy-specific sklearn imputer after fit.

        Returns
        -------
        SimpleImputer or KNNImputer or IterativeImputer
            The fitted estimator required by the active transform path.

        Raises
        ------
        RuntimeError
            If a fitted-state invariant has been violated.

        Examples
        --------
        >>> imputer = MissinglyImputer().fit(pd.DataFrame({"x": [1.0, None]}))
        >>> type(imputer._require_fitted_imputer()).__name__
        'SimpleImputer'
        """
        if self.imputer_ is None:
            raise RuntimeError("Fitted transformer is missing its sklearn imputer.")
        return self.imputer_

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Apply fitted imputation to a DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Data to impute.  Must have the same columns as training data.
        y : ignored

        Returns
        -------
        pd.DataFrame
            Imputed copy of *X*.

        Raises
        ------
        TypeError
            If ``X`` is not a pandas DataFrame.
        ValueError
            If ``X`` has duplicate labels or does not match the fitted schema.
        UnsupportedDtypeError
            If a timedelta or mixed-type ``object`` column that was complete at
            ``fit`` time has missing values and the strategy is model-based
            (``"knn"``, ``"mice"``, ``"rf"``, ``"gb"``).

        Warns
        -----
        FutureWarning
            For a model-based strategy: a datetime column that was complete at
            ``fit`` time has missing values; they are filled with the training
            mode for now and this will raise
            :class:`~missingly.exceptions.UnsupportedDtypeError` later.

        Examples
        --------
        >>> train = pd.DataFrame({"score": [1.0, 3.0]})
        >>> result = MissinglyImputer().fit(train).transform(
        ...     pd.DataFrame({"score": [None]})
        ... )
        >>> float(result.loc[0, "score"])
        2.0
        """
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self)

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"MissinglyImputer expects a pandas DataFrame; got {type(X)}"
            )
        self._require_unique_columns(X, operation="transform")

        missing_cols = set(self.feature_names_in_) - set(X.columns)
        extra_cols = set(X.columns) - set(self.feature_names_in_)
        if missing_cols:
            raise ValueError(
                f"Columns present during fit but missing in transform: "
                f"{sorted(missing_cols)}"
            )
        if extra_cols:
            raise ValueError(
                f"Columns present in transform but not seen during fit: "
                f"{sorted(extra_cols)}"
            )

        X = X.loc[:, self.feature_names_in_]
        X = self._coerce_numeric(self._normalize(X))
        self._require_servable(X)
        modelled = self._coerce_transform_categoricals(self._modelled_frame(X))
        return self._restore_schema(X, self._transform_modelled(modelled))

    def _coerce_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Give fit-time numeric columns a numeric dtype at transform time.

        A serving batch in which a numeric column is entirely missing arrives as
        an ``object`` column of ``None``; it becomes ``float64`` here.

        Parameters
        ----------
        X : pd.DataFrame
            Normalised transform input in fit-time column order.

        Returns
        -------
        pd.DataFrame
            Copy of ``X`` in which every fit-time numeric column is numeric.

        Examples
        --------
        >>> imputer = MissinglyImputer().fit(pd.DataFrame({"n": [1.0, None]}))
        >>> str(imputer._coerce_numeric(pd.DataFrame({"n": [None]}))["n"].dtype)
        'float64'
        """
        result = X.copy()
        for column in self.numeric_cols_:
            if not pd.api.types.is_numeric_dtype(result[column]):
                result[column] = pd.to_numeric(result[column])
        return result

    def _require_servable(self, X: pd.DataFrame) -> None:
        """Reject (or warn about) missing cells in a column the models set aside.

        Parameters
        ----------
        X : pd.DataFrame
            Normalised transform input in fit-time column order.

        Raises
        ------
        UnsupportedDtypeError
            If a timedelta or mixed-type ``object`` column that was complete at
            fit time has missing values now: no model exists for it.

        Warns
        -----
        FutureWarning
            For a datetime column that was complete at fit time and has missing
            values now.  The previous release imputed it as a nominal category;
            the gaps are filled with the training mode for now and this will
            raise :class:`~missingly.exceptions.UnsupportedDtypeError` in a
            future release.

        Examples
        --------
        >>> frame = pd.DataFrame({"n": [1.0, 2.0], "t": pd.to_timedelta([1, 2], unit="D")})
        >>> imputer = MissinglyImputer(strategy="knn").fit(frame)
        >>> gap = frame.assign(t=pd.to_timedelta([1, None], unit="D"))
        >>> try:
        ...     imputer.transform(gap)
        ... except UnsupportedDtypeError as error:
        ...     print(error.column)
        t
        """
        for column in self.passthrough_cols_:
            if not X[column].isna().any():
                continue
            if column not in self.fill_values_:
                raise UnsupportedDtypeError(
                    column=str(column),
                    dtype=X[column].dtype,
                    strategy=self.strategy,
                    reason=_NOT_FITTED_REASON,
                )
            warnings.warn(
                f"Column {column!r} has dtype {X[column].dtype} with missing values "
                f"that were not present at fit time; filling them with the training "
                f"mode with {self.strategy!r} is deprecated and will raise "
                "UnsupportedDtypeError in a future release.",
                FutureWarning,
                stacklevel=3,
            )

    def _transform_modelled(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute the modelled columns with the fitted strategy.

        Parameters
        ----------
        X : pd.DataFrame
            Modelled transform input (numeric columns ``float64``).

        Returns
        -------
        pd.DataFrame
            Imputed modelled columns; numeric ones are still ``float64``.
        """
        strategy = self.strategy
        result = X.copy()

        if strategy in _SIMPLE_STRATEGIES:
            result = self._transform_simple(X)

        elif strategy in ("knn", "mice"):
            X_enc = self._encode_cats(X)
            arr = self._require_fitted_imputer().transform(X_enc)
            df_enc = pd.DataFrame(arr, index=X.index, columns=X_enc.columns)
            result = self._decode_cats(df_enc)
            result = self._restore_observed_unknown_categories(result, X)

        elif strategy == "pmm":
            result = self._transform_pmm(X)

        elif strategy in ("rf", "gb"):
            result = self._transform_tree(X)

        elif strategy == "hotdeck":
            result = self._transform_hotdeck(X)

        return result[self.model_cols_]

    def _transform_simple(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill mean, median and mode with the values learned at fit time."""
        result = X.copy()
        numeric = set(self.numeric_cols_)
        for column in X.columns:
            if not X[column].isna().any():
                continue
            value = self.fill_values_[column]
            if column in numeric:
                result[column] = fill_with_statistic(X[column], float(value))
            else:
                # mode keeps the dtype of the serving column (v1 behaviour), mean
                # and median restore the fit-time dtype.
                dtype = (
                    X[column].dtype
                    if self.strategy == "mode"
                    else self.cat_dtypes_[column]
                )
                result[column] = self._fill_label(X[column], value, dtype)
        return result

    def _restore_schema(
        self, original: pd.DataFrame, imputed: pd.DataFrame
    ) -> pd.DataFrame:
        """Return the imputed frame in the dtypes of the input.

        Complete numeric and boolean columns are copied from the input, so
        their dtype and values are exactly preserved; imputed numeric cells are
        cast back to the column dtype (rounded to the nearest integer only for
        strategies that predict values); pass-through columns are re-attached.

        Parameters
        ----------
        original : pd.DataFrame
            Normalised transform input in fit-time column order.
        imputed : pd.DataFrame
            Result of :meth:`_transform_modelled`.

        Returns
        -------
        pd.DataFrame
            Frame with the fit-time column order and the input dtypes.

        Examples
        --------
        >>> frame = pd.DataFrame({"n": pd.array([1, None, 3], dtype="Int64")})
        >>> str(MissinglyImputer(strategy="knn", n_neighbors=1).fit_transform(frame)["n"].dtype)
        'Int64'
        """
        result = imputed.copy()
        for column in self.passthrough_cols_:
            source = original[column]
            if source.isna().any():
                source = self._fill_label(
                    source, self.fill_values_[column], source.dtype
                )
            result[column] = source.array
        round_integers = self.strategy in _ROUNDING_STRATEGIES
        for column in self.numeric_cols_:
            source = original[column]
            if source.isna().any():
                source = restore_numeric(
                    source,
                    imputed[column],
                    classify_column(source),
                    round_integers=round_integers,
                )
            result[column] = source.array
        for column in self.cat_cols_:
            if self.column_kinds_[column] is not ColumnKind.BOOLEAN:
                continue
            if original[column].isna().any():
                result[column] = _restore_categorical_dtype(
                    result[column], self.cat_dtypes_[column]
                )
            else:
                result[column] = original[column].array
        return result[self.feature_names_in_]

    def _transform_hotdeck(self, X: pd.DataFrame) -> pd.DataFrame:
        """Dispatch hot-deck transform to the appropriate variant.

        Samples every missing recipient only from observed values in the
        training DataFrame stored during ``fit``. Evaluation rows are never
        concatenated with the donor pool.
        """
        if self._train_df_ is None:
            raise RuntimeError("Random hot-deck transformer has no fitted donor pool.")
        result = X.copy()
        rng = np.random.default_rng(self.random_state)
        for column in X.columns:
            missing_mask = result[column].isna()
            if not missing_mask.any():
                continue
            donors = self._train_df_[column].dropna()
            if donors.empty:
                raise ValueError(
                    f"Column {column!r} has no observed training donors for hot-deck imputation."
                )
            result.loc[missing_mask, column] = rng.choice(
                donors.to_numpy(), size=int(missing_mask.sum()), replace=True
            )
            result[column] = _restore_categorical_dtype(
                result[column], self.cat_dtypes_.get(column, X[column].dtype)
            )
        return result

    def _transform_tree(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted per-column tree models to impute missing values."""
        is_gb = self.strategy == "gb"
        cat_set = set(self.cat_cols_)
        X_enc = self._encode_cats(X).values
        col_index = {col: i for i, col in enumerate(X.columns)}
        result = X.copy()

        for col in X.columns:
            missing_mask = result[col].isnull()
            if not missing_mask.any():
                continue

            feature_idx = [col_index[c] for c in X.columns if c != col]
            X_pred = X_enc[np.ix_(np.where(missing_mask)[0], feature_idx)]

            if is_gb:
                X_pred = self._fill_tree_features_from_training(X_pred, feature_idx)

            if col in cat_set and col in self.rf_clf_models_:
                clf = self.rf_clf_models_[col]
                enc_y = self.cat_label_encoders_[col]
                y_pred_enc = clf.predict(X_pred).reshape(-1, 1)
                y_pred = enc_y.inverse_transform(y_pred_enc).ravel()
                result.loc[missing_mask, col] = y_pred
                result[col] = _restore_categorical_dtype(
                    result[col], self.cat_dtypes_[col]
                )
            elif col not in cat_set and col in self.rf_reg_models_:
                reg = self.rf_reg_models_[col]
                result.loc[missing_mask, col] = reg.predict(X_pred)

        return result

    def _fill_tree_features_from_training(
        self,
        features: np.ndarray,
        feature_idx: List[int],
    ) -> np.ndarray:
        """Fill transform-time tree features using fit-time column means.

        Parameters
        ----------
        features : numpy.ndarray
            Encoded feature rows to predict.
        feature_idx : list of int
            Positions of the corresponding original feature columns.

        Returns
        -------
        numpy.ndarray
            Copy of ``features`` with NaNs replaced only by fit-time values.

        Examples
        --------
        >>> import numpy as np
        >>> imputer = MissinglyImputer(strategy="gb")
        >>> imputer._tree_feature_fill_values_ = np.array([1.0, 2.0])
        >>> imputer._fill_tree_features_from_training(np.array([[np.nan]]), [1])
        array([[2.]])
        """
        if self._tree_feature_fill_values_ is None:
            raise RuntimeError("Tree feature fill values are unavailable before fit.")
        result = features.copy()
        for position, original_idx in enumerate(feature_idx):
            result[np.isnan(result[:, position]), position] = (
                self._tree_feature_fill_values_[original_idx]
            )
        return result

    def _transform_pmm(
        self,
        X: pd.DataFrame,
        n_nearest_donors: int = 5,
    ) -> pd.DataFrame:
        """Apply PMM imputation using stored training data as donor pool."""
        from .impute import _fill_feature_matrix

        train_df = self._train_df_
        if train_df is None:
            raise RuntimeError("PMM transformer has no fitted training data.")
        rng = np.random.default_rng(self.random_state)
        result = X.copy()

        num_cols = self.numeric_cols_
        cat_cols = self.cat_cols_

        for col in num_cols:
            missing_mask = result[col].isna()
            if not missing_mask.any():
                continue

            feature_cols = [c for c in num_cols if c != col]
            if not feature_cols:
                donors = train_df.loc[train_df[col].notna(), col].to_numpy()
                result.loc[missing_mask, col] = rng.choice(
                    donors,
                    size=int(missing_mask.sum()),
                    replace=True,
                )
                continue

            train_obs_mask = ~train_df[col].isna()
            if train_obs_mask.sum() == 0:
                warnings.warn(
                    f"PMM: column {col!r} has no observed training values; "
                    "skipping imputation for this column.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            X_train = train_df.loc[train_obs_mask, feature_cols].values.astype(np.float64)
            y_train = train_df.loc[train_obs_mask, col].values.astype(np.float64)
            X_miss = result.loc[missing_mask, feature_cols].values.astype(np.float64)

            X_train_clean, X_miss_clean = _fill_feature_matrix(X_train, X_miss)

            model = BayesianRidge()
            model.fit(X_train_clean, y_train)
            y_pred = model.predict(X_miss_clean)

            k = min(n_nearest_donors, len(y_train))
            distances = np.abs(y_train[np.newaxis, :] - y_pred[:, np.newaxis])
            donor_idx = np.argpartition(distances, k - 1, axis=1)[:, :k]
            rand_col = rng.integers(0, k, size=len(y_pred))
            chosen_idx = donor_idx[np.arange(len(y_pred)), rand_col]
            imputed_vals = y_train[chosen_idx]

            miss_indices = missing_mask[missing_mask].index
            result.loc[miss_indices, col] = imputed_vals

        for col in cat_cols:
            gaps = result[col].isna()
            observed = train_df[col].dropna()
            if gaps.any() and len(observed) > 0:
                result.loc[gaps, col] = column_mode(observed, self.column_kinds_[col])

        return result

    # ------------------------------------------------------------------
    # sklearn API utilities
    # ------------------------------------------------------------------

    def get_feature_names_out(self) -> np.ndarray:
        """Return fit-time feature names in scikit-learn's array format.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of feature names in their fit-time order.

        Examples
        --------
        >>> import pandas as pd
        >>> names = MissinglyImputer().fit(
        ...     pd.DataFrame({"score": [1.0, None]})
        ... ).get_feature_names_out()
        >>> names.tolist()
        ['score']
        """
        return np.asarray(self.feature_names_in_, dtype=object)
