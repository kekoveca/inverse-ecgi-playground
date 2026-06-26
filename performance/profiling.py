from __future__ import annotations

import cProfile
import pstats
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


def profile_callable(func: Callable, *args, name: str | None = None, **kwargs) -> tuple[Any, dict]:
    """Run a callable and return ``(result, timing_metadata)``."""
    label = name or getattr(func, "__name__", "callable")
    start = perf_counter()
    result = func(*args, **kwargs)
    return result, {"name": label, "elapsed_s": perf_counter() - start}


def run_cprofile(func: Callable, output_path, *args, sort_by: str = "cumulative", **kwargs) -> Any:
    """Run ``func`` under cProfile and save stats to ``output_path``."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    result = profiler.runcall(func, *args, **kwargs)
    profiler.dump_stats(str(path))

    text_path = path.with_suffix(path.suffix + ".txt")
    with text_path.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(profiler, stream=handle).sort_stats(sort_by)
        stats.print_stats(80)
    return result
