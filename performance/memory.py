from __future__ import annotations

import resource
from typing import Any

import numpy as np


def get_process_memory_mb() -> float | None:
    """Return current process resident memory in MB when available."""
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))
    except Exception:
        pass

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except Exception:
        return None

    # Linux reports ru_maxrss in KB, macOS in bytes. The project runtime is
    # normally Linux/WSL, but guard against implausibly large byte values.
    value = float(usage.ru_maxrss)
    if value <= 0.0:
        return None
    if value > 10**8:
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def estimate_array_memory_mb(array: Any) -> float:
    """Estimate numpy-like array storage size in MB."""
    if hasattr(array, "nbytes"):
        return float(array.nbytes) / (1024.0 * 1024.0)
    np_array = np.asarray(array)
    return float(np_array.nbytes) / (1024.0 * 1024.0)
