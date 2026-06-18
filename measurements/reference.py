from __future__ import annotations

import warnings

import numpy as np


def _validate_n(n: int) -> int:
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    return n


def average_reference_matrix(n: int, sparse: bool = True):
    """Return matrix ``I - 1/n 11^T`` for average reference."""
    n = _validate_n(n)
    if sparse:
        try:
            from scipy.sparse import csr_matrix, eye
        except ImportError:  # pragma: no cover - scipy is available in tests
            warnings.warn(
                "scipy is not available; returning a dense reference matrix",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            ones = np.ones((n, n), dtype=float) / float(n)
            return eye(n, format="csr") - csr_matrix(ones)
    return np.eye(n, dtype=float) - np.ones((n, n), dtype=float) / float(n)


def apply_average_reference(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - values.mean(axis=0, keepdims=True)


def _single_reference_matrix(n: int, reference_index: int, sparse: bool):
    n = _validate_n(n)
    reference_index = int(reference_index)
    if reference_index < 0 or reference_index >= n:
        raise ValueError(f"reference_index must be in [0, {n})")
    if sparse:
        try:
            from scipy.sparse import csr_matrix, eye
        except ImportError:  # pragma: no cover - scipy is available in tests
            warnings.warn(
                "scipy is not available; returning a dense reference matrix",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            rows = np.arange(n, dtype=np.int64)
            cols = np.full(n, reference_index, dtype=np.int64)
            data = -np.ones(n, dtype=float)
            correction = csr_matrix((data, (rows, cols)), shape=(n, n))
            return eye(n, format="csr") + correction

    matrix = np.eye(n, dtype=float)
    matrix[:, reference_index] -= 1.0
    return matrix


def apply_reference(values, reference: str = "average", reference_index: int | None = None) -> np.ndarray:
    """Apply ``none``, ``average`` or ``single`` electrode referencing.

    ``single`` requires ``reference_index``. The returned array has the same
    shape as ``values`` and does not mutate the input.
    """

    values = np.asarray(values, dtype=float)
    if reference == "none":
        return values.copy()
    if reference == "average":
        return apply_average_reference(values)
    if reference == "single":
        if reference_index is None:
            raise ValueError("reference_index is required for single reference")
        reference_index = int(reference_index)
        if reference_index < 0 or reference_index >= values.shape[0]:
            raise ValueError(f"reference_index must be in [0, {values.shape[0]})")
        return values - values[reference_index]
    raise ValueError("reference must be one of: 'none', 'average', 'single'")


def reference_matrix(
    n: int,
    reference: str = "average",
    reference_index: int | None = None,
    sparse: bool = True,
):
    """Return a matrix implementing the requested electrode reference."""
    n = _validate_n(n)
    if reference == "none":
        if sparse:
            try:
                from scipy.sparse import eye
            except ImportError:  # pragma: no cover - scipy is available in tests
                warnings.warn(
                    "scipy is not available; returning a dense reference matrix",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                return eye(n, format="csr")
        return np.eye(n, dtype=float)
    if reference == "average":
        return average_reference_matrix(n, sparse=sparse)
    if reference == "single":
        if reference_index is None:
            raise ValueError("reference_index is required for single reference")
        return _single_reference_matrix(n, reference_index=reference_index, sparse=sparse)
    raise ValueError("reference must be one of: 'none', 'average', 'single'")
