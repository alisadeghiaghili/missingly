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
        * ``"rf"``     — Random Forest.
        * ``"gb"``     — Gradient Boosting.

        Default is ``"mean"``.
    n_neighbors : int, optional
        Number of neighbours for ``strategy="knn"``.  Default 5.
    metric : str, optional
        Distance metric for ``strategy="knn"``.  One of
        ``"euclidean"`` (default) or ``"mixed"``.
    max_iter : int, optional
        Maximum EM/MICE/RF/GB iterations.  Default 10.
    random_state : int, optional
        Random seed for reproducibility.  Default 0.
    estimator_kwargs : dict or None, optional
        Additional keyword arguments forwarded to the underlying sklearn
        estimator (e.g. ``{'n_estimators': 200}`` for RF/GB strategies).
        Stored as-is so that ``get_params()`` / ``set_params()`` /
        ``clone()`` behave correctly.

    Example
    -------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> pipe = Pipeline([
    ...     ("imputer", MissinglyImputer(strategy="knn", n_neighbors=3)),
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
    ) -> None:
        """Initialise the imputer.

        All parameters are stored verbatim (no coercion) so that
        ``get_params()`` / ``set_params()`` round-trip correctly.
        """
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_VALID_STRATEGIES)}; "
                f"got {strategy!r}"
            )
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs
        # Fitted-state sentinel — NOT a trailing-underscore attribute so that
        # sklearn's check_is_fitted does not treat it as a fitted attribute.
        self._is_fitted: bool = False

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
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"MissinglyImputer expects a pandas DataFrame; got {type(X)}"
            )
        X = self._normalize(X)

        # Fitted-state attributes — only set inside fit(), never in __init__.
        self.feature_names_in_: List[str] = X.columns.tolist()
        self.numeric_cols_: List[str] = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_: List[str] = X.select_dtypes(exclude=[np.number]).columns.tolist()
        self.cat_dtypes_: Dict[str, object] = {c: X[c].dtype for c in self.cat_cols_}
        self.imputer_ = None
        self.cat_imputer_ = None
        self.encoder_: Optional[OrdinalEncoder] = None
        self.rf_reg_models_: Dict[str, object] = {}
        self.rf_clf_models_: Dict[str, object] = {}
        self.cat_label_encoders_: Dict[str, OrdinalEncoder] = {}
        self._knn_train_df_: Optional[pd.DataFrame] = None

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

        elif strategy == "knn":
            if self.metric == "mixed":
                from .impute import _impute_knn_gower
                train_df = self._knn_train_df_
                n_train = len(train_df)
                combined = pd.concat([train_df, X], ignore_index=True)
                imputed_combined = _impute_knn_gower(
                    combined, n_neighbors=self.n_neighbors
                )
                result = imputed_combined.iloc[n_train:].copy()
                result.index = X.index
            else:
                X_enc = self._encode_cats(X)
                arr = self.imputer_.transform(X_enc)
                df_enc = pd.DataFrame(arr, index=X.index, columns=X_enc.columns)
                result = self._decode_cats(df_enc)

        elif strategy == "mice":
            X_enc = self._encode_cats(X)
            arr = self.imputer_.transform(X_enc)
            df_enc = pd.DataFrame(arr, index=X.index, columns=X_enc.columns)
            result = self._decode_cats(df_enc)

        elif strategy in ("rf", "gb"):
            result = self._transform_tree(X)

        return result[self.feature_names_in_]

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
        """Return feature names as seen during fit.

        Returns
        -------
        list[str]
        """
        return list(self.feature_names_in_)
