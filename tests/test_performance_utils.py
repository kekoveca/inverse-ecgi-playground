import json
import time

import numpy as np

from performance import (
    PerformanceTimer,
    estimate_array_memory_mb,
    get_process_memory_mb,
    save_timing_csv,
    save_timing_json,
)


def test_performance_timer_records_elapsed_time():
    timer = PerformanceTimer()

    with timer.time("sleep", case="unit"):
        time.sleep(0.001)

    assert len(timer.records) == 1
    assert timer.records[0].name == "sleep"
    assert timer.records[0].elapsed_s > 0.0
    assert timer.records[0].metadata["case"] == "unit"
    assert timer.summary()["num_records"] == 1


def test_save_timing_csv_and_json(tmp_path):
    timer = PerformanceTimer()
    timer.add_record("stage", 0.125, size=3)

    csv_path = save_timing_csv(timer, tmp_path / "timing.csv")
    json_path = save_timing_json(timer, tmp_path / "timing.json")

    assert csv_path.exists()
    assert "stage" in csv_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["num_records"] == 1
    assert payload["records"][0]["name"] == "stage"


def test_estimate_array_memory_mb():
    array = np.zeros((128,), dtype=np.float64)

    assert estimate_array_memory_mb(array) == array.nbytes / (1024.0 * 1024.0)


def test_get_process_memory_mb_returns_float_or_none():
    value = get_process_memory_mb()

    assert value is None or isinstance(value, float)
    if value is not None:
        assert value > 0.0
