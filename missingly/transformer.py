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

Example
-------
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

from .impute import _fill_nan_with_col_means


_VALID_STRATEGIES = frozenset(
    {"mean", "median", "mode", "knn", "mice", "rf", "gb"}
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

        * ``"mean"``   — fill numeric with mean, categorical with mode.
        * ``"median"`` — fill numeric with median, categorical with mode.
        * ``"mode"``   — fill all columns with most frequent value.
        * ``"knn"``    — k-Nearest Neighbours (default k=5).
        * ``"mice"``   — Multiple Imputation by Chained Equations.
        * ``"rf"``     — Random Forest (regressor for numeric,
          classifier for categorical).
        * ``"gb"``     — Gradient Boosting (regressor for numeric,
          classifier for categorical).

        Default is ``"mean"``.
    n_neighbors : int, optional
        Number of neighbours for ``strategy="knn"``.  Default 5.
    max_iter : int, optional
        Maximum EM/MICE/RF/GB iterations.  Default 10.
    random_state : int, optional
        Random seed for reproducibility.  Default 0.
    **estimator_kwargs
        Additional keyword arguments forwarded to the underlying
        sklearn estimator (e.g. ``n_estimators=200`` for RF/GB).

    Attributes
    ----------
    strategy : str
    n_neighbors : int
    max_iter : int
    random_state : int
    feature_names_in_ : list[str] or None
        Column names seen during ``fit``.  ``None`` before fitting.
    numeric_cols_ : list[str]
        Numeric column names seen during ``fit``.
    cat_cols_ : list[str]
        Categorical column names seen during ``fit``.
    imputer_ : fitted sklearn imputer or None
        The underlying fitted imputer (for mean/median/mode/knn/mice).
    cat_imputer_ : fitted SimpleImputer or None
        Categorical imputer (for mean/median).
    encoder_ : fitted OrdinalEncoder or None
        Encoder for categorical columns (knn/mice).
    cat_dtypes_ : dict
        Original dtypes of categorical columns.
    rf_reg_models_ : dict[str, fitted estimator]
        Per-column fitted regressors (rf/gb strategies).
    rf_clf_models_ : dict[str, fitted estimator]
        Per-column fitted classifiers (rf/gb strategies).
    cat_label_encoders_ : dict[str, OrdinalEncoder]
        Per-column label encoders for classification targets (rf/gb).

    Example
    -------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> pipe = Pipeline([
    ...     ("imputer", MissinglyImputer(strategy="knn", n_neighbors=3)),
    ...     ("clf",     LogisticRegression()),
    ... ])
    >>> pipe.fit(X_train, y_train)
    >>> pipe.predict(X_test)
    """

    def __init__(
        self,
        strategy: str = "mean",
        n_neighbors: int = 5,
        max_iter: int = 10,
        random_state: int = 0,
        **estimator_kwargs,
    ) -> None:
        """Initialise the imputer with the chosen strategy and hyperparameters."""
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_VALID_STRATEGIES)}; "
                f"got {strategy!r}"
            )
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs

        # Fitted state — None until fit() is called.
        # IMPORTANT: do NOT use mutable defaults like [] here; sklearn's
        # check_is_fitted looks for attributes ending in '_' and treats any
        # truthy value as "fitted", so [] would pass the check even before fit.
        self._is_fitted: bool = False
        self.feature_names_in_: Optional[List[str]] = None
        self.numeric_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.imputer_ = None
        self.cat_imputer_ = None
        self.encoder_: Optional[OrdinalEncoder] = None
        self.cat_dtypes_: Dict[str, object] = {}
        self.rf_reg_models_: Dict[str, object] = {}
        self.rf_clf_models_: Dict[str, object] = {}
        self.cat_label_encoders_: Dict[str, OrdinalEncoder] = {}

    # ------------------------------------------------------------------
    # sklearn fitted-state contract
    # ------------------------------------------------------------------

    def __sklearn_is_fitted__(self) -> bool:
        """Return whether this estimator has been fitted.

        sklearn calls this method (when present) in ``check_is_fitted``
        instead of inspecting attributes.  Returning ``False`` before
        ``fit`` guarantees a ``NotFittedError`` is raised on ``transform``.
        """
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

    def _encode_cats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordinal-encode categorical columns using the fitted encoder."""
        df_work = df.copy()
        if self.cat_cols_ and self.encoder_ is not None:
            df_work[self.cat_cols_] = self.encoder_.transform(
                df[self.cat_cols_]
            )
        return df_work.astype(float)

    def _decode_cats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse-transform ordinal-encoded categorical columns."""
        if not self.cat_cols_ or self.encoder_ is None:
            return df
        result = df.copy()
        rounded = np.round(result[self.cat_cols_].to_numpy()).clip(0).copy()
        decoded = self.encoder_.inverse_transform(rounded)
        for i, col in enumerate(self.cat_cols_):
            result[col] = decoded[:, i]
            try:
                result[col] = result[col].astype(self.cat_dtypes_[col])
            except (ValueError, TypeError):
                pass
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
            The fitted imputer instance.

        Raises
        ------
        TypeError
            If *X* is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"MissinglyImputer expects a pandas DataFrame; got {type(X)}"
            )
        X = self._normalize(X)

        self.feature_names_in_ = X.columns.tolist()
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()
        self.cat_dtypes_ = {c: X[c].dtype for c in self.cat_cols_}

        strategy = self.strategy

        if strategy == "mean":
            self._fit_simple(X, num_strategy="mean")

        elif strategy == "median":
            self._fit_simple(X, num_strategy="median")

        elif strategy == "mode":
            self.imputer_ = SimpleImputer(strategy="most_frequent")
            self.imputer_.fit(X)

        elif strategy == "knn":
            self._fit_knn(X)

        elif strategy == "mice":
            self._fit_mice(X)

        elif strategy in ("rf", "gb"):
            self._fit_tree(X)

        self._is_fitted = True
        return self

    def _fit_simple(self, X: pd.DataFrame, num_strategy: str) -> None:
        """Fit mean/median numeric imputer and mode categorical imputer."""
        if self.numeric_cols_:
            self.imputer_ = SimpleImputer(strategy=num_strategy)
            self.imputer_.fit(X[self.numeric_cols_])
        if self.cat_cols_:
            self.cat_imputer_ = SimpleImputer(strategy="most_frequent")
            self.cat_imputer_.fit(X[self.cat_cols_])

    def _fit_knn(self, X: pd.DataFrame) -> None:
        """Fit ordinal encoder + KNNImputer on the full encoded DataFrame."""
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
        """Fit ordinal encoder + IterativeImputer on the full encoded DataFrame."""
        if self.cat_cols_:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            )
            self.encoder_.fit(X[self.cat_cols_])
        X_enc = self._encode_cats(X)
        estimator = BayesianRidge()
        self.imputer_ = IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            random_state=self.random_state,
            imputation_order="roman",
        )
        self.imputer_.fit(X_enc)

    def _fit_tree(self, X: pd.DataFrame) -> None:
        """Fit per-column regressor/classifier for rf and gb strategies.

        For GB, any NaN remaining in the feature matrix is filled with
        column means (``_fill_nan_with_col_means``) because
        ``GradientBoostingRegressor`` / ``GradientBoostingClassifier`` do
        not natively support NaN features.
        """
        is_rf = self.strategy == "rf"
        is_gb = self.strategy == "gb"
        cat_set = set(self.cat_cols_)

        if self.cat_cols_:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            )
            self.encoder_.fit(X[self.cat_cols_])

        X_enc = self._encode_cats(X).values
        col_index = {col: i for i, col in enumerate(X.columns)}

        for col in X.columns:
            missing_mask = X[col].isnull()
            train_mask = ~missing_mask
            if train_mask.sum() < 2:
                continue

            feature_idx = [col_index[c] for c in X.columns if c != col]
            X_train = X_enc[np.ix_(np.where(train_mask)[0], feature_idx)]

            if is_gb:
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
                    RandomForestClassifier(
                        random_state=self.random_state, **self.estimator_kwargs
                    )
                    if is_rf
                    else GradientBoostingClassifier(
                        random_state=self.random_state, **self.estimator_kwargs
                    )
                )
                clf.fit(X_train, y_train_enc)
                self.rf_clf_models_[col] = clf
                self.cat_label_encoders_[col] = enc_y
            else:
                y_train = X.loc[train_mask, col].values
                reg = (
                    RandomForestRegressor(
                        random_state=self.random_state, **self.estimator_kwargs
                    )
                    if is_rf
                    else GradientBoostingRegressor(
                        random_state=self.random_state, **self.estimator_kwargs
                    )
                )
                reg.fit(X_train, y_train)
                self.rf_reg_models_[col] = reg

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Apply fitted imputation to a DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Data to impute.  Must have the same columns as the training
            DataFrame passed to ``fit``.
        y : ignored
            Present for sklearn API compatibility.

        Returns
        -------
        pd.DataFrame
            Fully-imputed copy of *X*.  The original is not modified.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If ``transform`` is called before ``fit``.
        TypeError
            If *X* is not a pandas DataFrame.
        ValueError
            If *X* has columns that differ from those seen during ``fit``.
        """
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self)

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"MissinglyImputer expects a pandas DataFrame; got {type(X)}"
            )

        missing_cols = set(self.feature_names_in_) - set(X.columns)
        if missing_cols:
            raise ValueError(
                f"Columns present during fit but missing in transform: "
                f"{sorted(missing_cols)}"
            )

        X = self._normalize(X)
        result = X.copy()
        strategy = self.strategy

        if strategy in ("mean", "median"):
            if self.numeric_cols_ and self.imputer_ is not None:
                result[self.numeric_cols_] = self.imputer_.transform(
                    X[self.numeric_cols_]
                )
            if self.cat_cols_ and self.cat_imputer_ is not None:
                result[self.cat_cols_] = self.cat_imputer_.transform(
                    X[self.cat_cols_]
                )

        elif strategy == "mode":
            arr = self.imputer_.transform(X)
            result = pd.DataFrame(arr, index=X.index, columns=X.columns)

        elif strategy in ("knn", "mice"):
            X_enc = self._encode_cats(X)
            arr = self.imputer_.transform(X_enc)
            df_enc = pd.DataFrame(arr, index=X.index, columns=X_enc.columns)
            result = self._decode_cats(df_enc)

        elif strategy in ("rf", "gb"):
            result = self._transform_tree(X)

        return result[self.feature_names_in_]

    def _transform_tree(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted per-column tree models to impute missing values.

        For GB models, NaN in the feature matrix is filled with column
        means before calling ``predict``.
        """
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
                X_pred = _fill_nan_with_col_means(X_pred)

            if col in cat_set and col in self.rf_clf_models_:
                clf = self.rf_clf_models_[col]
                enc_y = self.cat_label_encoders_[col]
                y_pred_enc = clf.predict(X_pred).reshape(-1, 1)
                y_pred = enc_y.inverse_transform(y_pred_enc).ravel()
                result.loc[missing_mask, col] = y_pred
            elif col not in cat_set and col in self.rf_reg_models_:
                reg = self.rf_reg_models_[col]
                result.loc[missing_mask, col] = reg.predict(X_pred)

        return result

    # ------------------------------------------------------------------
    # sklearn API utilities
    # ------------------------------------------------------------------

    def get_feature_names_out(self) -> List[str]:
        """Return the output feature names (same order as input).

        Returns
        -------
        list[str]
            Feature names as seen during ``fit``.
        """
        return list(self.feature_names_in_)
