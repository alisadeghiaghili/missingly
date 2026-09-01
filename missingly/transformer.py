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
from typing import Dict, List, Optional

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

from .impute import (
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

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> pipe = Pipeline([
    ...     ("imputer", MissinglyImputer(strategy="hotdeck", hotdeck_method="weighted")),
    ...     ("clf",     LogisticRegression()),
    ... ])
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
            categorical = df_work[self.cat_cols_].astype(object)
            categorical = categorical.where(categorical.notna(), other=np.nan)
            df_work[self.cat_cols_] = self.encoder_.transform(categorical)
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

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

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
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"MissinglyImputer expects a pandas DataFrame; got {type(X)}"
            )
        X = self._normalize(X)

        self.feature_names_in_: List[str] = X.columns.tolist()
        self.numeric_cols_: List[str] = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_: List[str] = X.select_dtypes(exclude=[np.number]).columns.tolist()
        self.cat_dtypes_: Dict[str, object] = {c: X[c].dtype for c in self.cat_cols_}
        self.imputer_ = None
        self.cat_imputer_ = None
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

        strategy = self.strategy

        if strategy == "mean":
            self._fit_simple(X, num_strategy="mean")
        elif strategy == "median":
            self._fit_simple(X, num_strategy="median")
        elif strategy == "mode":
            self.imputer_ = SimpleImputer(strategy="most_frequent")
            self.imputer_.fit(X)
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
            self._fit_pmm(X)
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

        self._is_fitted = True
        return self

    def _fit_simple(self, X: pd.DataFrame, num_strategy: str) -> None:
        """Fit mean/median numeric imputer and mode categorical imputer."""
        if self.numeric_cols_:
            self.imputer_ = SimpleImputer(strategy=num_strategy)
            self.imputer_.fit(X[self.numeric_cols_])
        if self.cat_cols_:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            cat_imputer.fit(X[self.cat_cols_])
            self.cat_imputer_ = cat_imputer

    def _fit_knn(self, X: pd.DataFrame) -> None:
        """Fit KNN imputer — euclidean or mixed metric."""
        if self.metric == "mixed":
            self._knn_train_df_ = X.copy()
            return
        if self.cat_cols_:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            )
            self.encoder_.fit(X[self.cat_cols_])
        X_enc = self._encode_cats(X)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer_.fit(X_enc)

    def _fit_mice(self, X: pd.DataFrame) -> None:
        """Fit ordinal encoder + IterativeImputer."""
        if self.cat_cols_:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            )
            self.encoder_.fit(X[self.cat_cols_])
        X_enc = self._encode_cats(X)
        self.imputer_ = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            random_state=self.random_state,
            imputation_order="roman",
        )
        self.imputer_.fit(X_enc)

    def _fit_pmm(self, X: pd.DataFrame) -> None:
        """Fit PMM — store training data for donor matching."""
        self._train_df_ = X.copy()
        if self.cat_cols_:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            )
            self.encoder_.fit(X[self.cat_cols_])

    def _fit_tree(self, X: pd.DataFrame) -> None:
        """Fit per-column regressor/classifier for rf and gb strategies."""
        est_kwargs = self.estimator_kwargs or {}
        is_rf = self.strategy == "rf"
        cat_set = set(self.cat_cols_)

        if self.cat_cols_:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            )
            self.encoder_.fit(X[self.cat_cols_])

        X_enc = self._encode_cats(X).values
        fill_values = np.nanmean(X_enc, axis=0)
        self._tree_feature_fill_values_ = np.where(
            np.isfinite(fill_values), fill_values, 0.0
        )
        col_index = {col: i for i, col in enumerate(X.columns)}

        for col in X.columns:
            missing_mask = X[col].isnull()
            train_mask = ~missing_mask
            if train_mask.sum() < 2:
                continue

            feature_idx = [col_index[c] for c in X.columns if c != col]
            X_train = X_enc[np.ix_(np.where(train_mask)[0], feature_idx)]

            if not is_rf:
                X_train = _fill_nan_with_col_means(X_train)

            if col in cat_set:
                enc_y = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                )
                y_train_enc = enc_y.fit_transform(
                    X.loc[train_mask, col].values.reshape(-1, 1)
                ).ravel().astype(int)
                clf = (
                    RandomForestClassifier(random_state=self.random_state, **est_kwargs)
                    if is_rf
                    else GradientBoostingClassifier(random_state=self.random_state, **est_kwargs)
                )
                clf.fit(X_train, y_train_enc)
                self.rf_clf_models_[col] = clf
                self.cat_label_encoders_[col] = enc_y
            else:
                y_train = X.loc[train_mask, col].values
                reg = (
                    RandomForestRegressor(random_state=self.random_state, **est_kwargs)
                    if is_rf
                    else GradientBoostingRegressor(random_state=self.random_state, **est_kwargs)
                )
                reg.fit(X_train, y_train)
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
        """
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self)

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"MissinglyImputer expects a pandas DataFrame; got {type(X)}"
            )

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
        X = self._normalize(X)
        X = self._coerce_transform_categoricals(X)
        result = X.copy()
        strategy = self.strategy

        if strategy in ("mean", "median"):
            if self.numeric_cols_ and self.imputer_ is not None:
                result[self.numeric_cols_] = self.imputer_.transform(
                    X[self.numeric_cols_]
                )
            if self.cat_cols_ and self.cat_imputer_ is not None:
                imputed = self.cat_imputer_.transform(X[self.cat_cols_])
                for i, col in enumerate(self.cat_cols_):
                    filled = pd.Series(imputed[:, i], index=X.index)
                    result[col] = _restore_categorical_dtype(filled, self.cat_dtypes_[col])

        elif strategy == "mode":
            imputer = self._require_fitted_imputer()
            arr = imputer.transform(X)
            result = pd.DataFrame(arr, index=X.index, columns=X.columns)
            for col in X.columns:
                result[col] = _restore_categorical_dtype(result[col], X[col].dtype)

        elif strategy == "knn":
            X_enc = self._encode_cats(X)
            arr = self._require_fitted_imputer().transform(X_enc)
            df_enc = pd.DataFrame(arr, index=X.index, columns=X_enc.columns)
            result = self._decode_cats(df_enc)
            result = self._restore_observed_unknown_categories(result, X)

        elif strategy == "mice":
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
            if result[col].isna().any():
                observed = train_df[col].dropna()
                if len(observed) > 0:
                    result[col] = result[col].fillna(observed.mode().iloc[0])

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
