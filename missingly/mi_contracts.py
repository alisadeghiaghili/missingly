"""Typed, validated contracts for multiple-imputation workflows.

These objects are intentionally independent from a particular imputation
algorithm.  They establish the data and provenance boundary that a future
fully conditional specification (FCS) engine will consume without changing
the legacy :func:`missingly.impute.impute_mice` return contract prematurely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "FCSKernel",
    "ImputationPlan",
    "ImputationResult",
    "MIData",
    "MissingnessSchema",
]


def _require_dataframe(frame: pd.DataFrame, name: str) -> None:
    """Validate a DataFrame boundary without coercing caller-owned data."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError(f"{name} must not have duplicate column labels")
    if frame.index.has_duplicates:
        raise ValueError(f"{name} must not have duplicate index labels")


def _require_nonempty_strings(values: Sequence[str], name: str) -> None:
    """Validate ordered unique nonempty string settings."""
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise TypeError(f"{name} must contain nonempty strings")


@dataclass(frozen=True)
class MissingnessSchema:
    """Capture the original DataFrame structure and cells eligible for imputation.

    Parameters
    ----------
    columns : tuple of str
        Ordered unique column names from the original data.
    dtypes : tuple of str
        Ordered pandas dtype strings corresponding to ``columns``.
    index : pandas.Index
        Exact original index. Duplicate labels are not allowed because result
        alignment must be unambiguous.
    original_mask : pandas.DataFrame
        Boolean mask with ``True`` exactly where original values were missing.

    Examples
    --------
    >>> frame = pd.DataFrame({"x": [1.0, np.nan]}, index=["a", "b"])
    >>> schema = MissingnessSchema.from_dataframe(frame)
    >>> schema.original_mask["x"].tolist()
    [False, True]
    """

    columns: Tuple[str, ...]
    dtypes: Tuple[str, ...]
    index: pd.Index
    original_mask: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate structural and mask invariants."""
        _require_nonempty_strings(self.columns, "columns")
        if len(self.columns) != len(self.dtypes):
            raise ValueError("columns and dtypes must have equal length")
        if not isinstance(self.index, pd.Index):
            raise TypeError("index must be a pandas Index")
        if self.index.has_duplicates:
            raise ValueError("index must not have duplicate labels")
        _require_dataframe(self.original_mask, "original_mask")
        if tuple(self.original_mask.columns) != self.columns:
            raise ValueError("original_mask columns must exactly match columns")
        if not self.original_mask.index.equals(self.index):
            raise ValueError("original_mask index must exactly match index")
        if not all(pd.api.types.is_bool_dtype(dtype) for dtype in self.original_mask.dtypes):
            raise TypeError("original_mask must contain only boolean columns")

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame) -> "MissingnessSchema":
        """Create an immutable structural contract from a DataFrame.

        Parameters
        ----------
        frame : pandas.DataFrame
            Original incomplete input. It is never modified.

        Returns
        -------
        MissingnessSchema
            Schema with a deep-copied boolean missing-value mask.

        Examples
        --------
        >>> frame = pd.DataFrame({"score": [1.0, np.nan]})
        >>> MissingnessSchema.from_dataframe(frame).columns
        ('score',)
        """
        _require_dataframe(frame, "frame")
        if frame.shape[1] == 0:
            raise ValueError("frame must contain at least one column")
        if any(not isinstance(column, str) for column in frame.columns):
            raise TypeError("frame columns must be strings")
        return cls(
            columns=tuple(frame.columns),
            dtypes=tuple(str(dtype) for dtype in frame.dtypes),
            index=frame.index.copy(),
            original_mask=frame.isna().copy(deep=True),
        )

    def validate_frame(self, frame: pd.DataFrame, *, name: str = "frame") -> None:
        """Ensure a frame has the exact structure required by this schema.

        Parameters
        ----------
        frame : pandas.DataFrame
            Frame to validate against this schema.
        name : str, default="frame"
            Name included in actionable validation errors.

        Raises
        ------
        TypeError
            If ``frame`` is not a DataFrame.
        ValueError
            If columns, index, or dtype strings differ from the original.

        Examples
        --------
        >>> original = pd.DataFrame({"x": [1.0, np.nan]})
        >>> schema = MissingnessSchema.from_dataframe(original)
        >>> schema.validate_frame(original.copy())
        """
        _require_dataframe(frame, name)
        if tuple(frame.columns) != self.columns:
            raise ValueError(f"{name} columns must exactly match the original schema")
        if not frame.index.equals(self.index):
            raise ValueError(f"{name} index must exactly match the original schema")
        if tuple(str(dtype) for dtype in frame.dtypes) != self.dtypes:
            raise ValueError(f"{name} dtypes must exactly match the original schema")


@dataclass(frozen=True)
class ImputationPlan:
    """Describe a reproducible, explicit plan for multiple imputation.

    Parameters
    ----------
    methods : mapping of str to str
        Per-column method identifiers. Keys must be schema column names and
        values must be nonempty public method identifiers.
    n_imputations : int
        Number of completed data sets to generate.
    max_iter : int
        Maximum conditional-model iterations per chain.
    seed_sequence : tuple of int
        One deterministic seed per imputation chain.
    constraints : mapping of str to str, optional
        Human-readable bounds, passive terms, or other future kernel rules.
    provenance : mapping of str to Any, optional
        Raw-data-free reproducibility metadata such as an analysis digest.

    Examples
    --------
    >>> plan = ImputationPlan({"x": "pmm"}, 2, 5, (101, 202))
    >>> plan.n_imputations
    2
    """

    methods: Mapping[str, str]
    n_imputations: int
    max_iter: int
    seed_sequence: Tuple[int, ...]
    constraints: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate reproducibility-critical plan settings."""
        if not isinstance(self.methods, Mapping) or not self.methods:
            raise ValueError("methods must be a nonempty mapping")
        for column, method in self.methods.items():
            _require_nonempty_strings((column,), "method column names")
            _require_nonempty_strings((method,), "method identifiers")
        for name, value in (("n_imputations", self.n_imputations), ("max_iter", self.max_iter)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed_sequence, tuple) or len(self.seed_sequence) != self.n_imputations:
            raise ValueError("seed_sequence must be a tuple with one seed per imputation")
        if len(set(self.seed_sequence)) != len(self.seed_sequence):
            raise ValueError("seed_sequence must not contain duplicate seeds")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seed_sequence):
            raise TypeError("seed_sequence must contain integers")
        for name, value in (("constraints", self.constraints), ("provenance", self.provenance)):
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")

    def validate_schema(self, schema: MissingnessSchema) -> None:
        """Check that all planned methods target columns in a schema.

        Parameters
        ----------
        schema : MissingnessSchema
            Input schema against which method keys are validated.

        Raises
        ------
        TypeError
            If ``schema`` is not a :class:`MissingnessSchema`.
        ValueError
            If a method or constraint references an unknown column.

        Examples
        --------
        >>> schema = MissingnessSchema.from_dataframe(pd.DataFrame({"x": [1.0]}))
        >>> ImputationPlan({"x": "pmm"}, 1, 2, (7,)).validate_schema(schema)
        """
        if not isinstance(schema, MissingnessSchema):
            raise TypeError("schema must be a MissingnessSchema")
        unknown = (set(self.methods) | set(self.constraints)) - set(schema.columns)
        if unknown:
            raise ValueError(f"plan references unknown schema columns: {sorted(unknown)}")


class FCSKernel(Protocol):
    """Protocol for one conditional model step in a future FCS engine.

    Implementations receive the current chain state and return values only for
    the target column's missing rows. Kernels must be explicit about their
    stochastic generator and must not mutate ``state`` in place.
    """

    def impute(
        self,
        state: pd.DataFrame,
        target: str,
        mask: pd.Series,
        rng: np.random.Generator,
    ) -> pd.Series:
        """Generate one target-column conditional draw.

        Parameters
        ----------
        state : pandas.DataFrame
            Current completed chain state.
        target : str
            Column being imputed.
        mask : pandas.Series
            Boolean target-column missingness mask.
        rng : numpy.random.Generator
            Chain-specific random generator.

        Returns
        -------
        pandas.Series
            Draws aligned to the ``True`` entries of ``mask``.
        """


@dataclass(frozen=True)
class MIData:
    """Bind an incomplete input, schema, plan, and validated completed data sets.

    Parameters
    ----------
    original : pandas.DataFrame
        Original incomplete data, structurally matching ``schema``.
    schema : MissingnessSchema
        Original structure and eligible-cell mask.
    plan : ImputationPlan
        Reproducible per-column method and seed configuration.
    imputations : tuple of pandas.DataFrame
        Completed data sets. Observed original cells must be unchanged.

    Examples
    --------
    >>> original = pd.DataFrame({"x": [1.0, np.nan]})
    >>> schema = MissingnessSchema.from_dataframe(original)
    >>> plan = ImputationPlan({"x": "pmm"}, 1, 2, (9,))
    >>> MIData(original, schema, plan, (pd.DataFrame({"x": [1.0, 2.0]}),)).n_imputations
    1
    """

    original: pd.DataFrame
    schema: MissingnessSchema
    plan: ImputationPlan
    imputations: Tuple[pd.DataFrame, ...]

    def __post_init__(self) -> None:
        """Validate structural preservation and observed-cell immutability."""
        self.schema.validate_frame(self.original, name="original")
        self.plan.validate_schema(self.schema)
        if not isinstance(self.imputations, tuple) or len(self.imputations) != self.plan.n_imputations:
            raise ValueError("imputations must be a tuple with one frame per planned imputation")
        observed = ~self.schema.original_mask
        for position, imputed in enumerate(self.imputations):
            self.schema.validate_frame(imputed, name=f"imputations[{position}]")
            if imputed.where(observed).compare(self.original.where(observed)).shape[0]:
                raise ValueError("imputations must not change originally observed cells")

    @property
    def n_imputations(self) -> int:
        """Return the number of completed data sets in the contract."""
        return len(self.imputations)


@dataclass(frozen=True)
class ImputationResult:
    """Attach diagnostic traces and validation metadata to completed MI data.

    Parameters
    ----------
    data : MIData
        Validated original and completed data sets.
    histories : tuple of mappings, optional
        Per-chain, per-variable numerical diagnostic traces.
    warnings : tuple of str, optional
        Explicit non-fatal algorithm warnings.
    validation : mapping of str to bool, optional
        Named validation outcomes such as convergence or bounds checks.

    Examples
    --------
    >>> original = pd.DataFrame({"x": [1.0, np.nan]})
    >>> schema = MissingnessSchema.from_dataframe(original)
    >>> plan = ImputationPlan({"x": "pmm"}, 1, 2, (9,))
    >>> data = MIData(original, schema, plan, (pd.DataFrame({"x": [1.0, 2.0]}),))
    >>> ImputationResult(data, histories=({"x": (1.5, 1.6)},)).validation["result_valid"]
    True
    """

    data: MIData
    histories: Tuple[Mapping[str, Tuple[float, ...]], ...] = ()
    warnings: Tuple[str, ...] = ()
    validation: Mapping[str, bool] = field(default_factory=lambda: {"result_valid": True})

    def __post_init__(self) -> None:
        """Validate trace alignment and explicit result diagnostics."""
        if not isinstance(self.data, MIData):
            raise TypeError("data must be MIData")
        if self.histories and len(self.histories) != self.data.n_imputations:
            raise ValueError("histories must be empty or contain one mapping per imputation")
        for history in self.histories:
            unknown = set(history) - set(self.data.schema.columns)
            if unknown:
                raise ValueError(f"histories reference unknown schema columns: {sorted(unknown)}")
            for trace in history.values():
                if not isinstance(trace, tuple) or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) for value in trace
                ):
                    raise TypeError("history traces must be tuples of finite numbers")
        if not isinstance(self.warnings, tuple) or any(not isinstance(item, str) for item in self.warnings):
            raise TypeError("warnings must be a tuple of strings")
        if not isinstance(self.validation, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, bool)
            for name, value in self.validation.items()
        ):
            raise TypeError("validation must map string names to booleans")
