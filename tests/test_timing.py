import time

from Pelagia.processing.timing import collect_result_timings, measure_phase


def test_collect_result_timings_records_repeated_phases():
    @collect_result_timings()
    def timed_operation():
        for _ in range(2):
            with measure_phase("codec.encode"):
                time.sleep(0.001)
        return {"frame_count": 2}

    result = timed_operation()
    timings = result["timings"]

    assert timings["schema_version"] == 1
    assert timings["unit_count"] == 2
    assert timings["total_ms"] >= timings["measured_ms"] > 0
    assert timings["phases_ms"]["codec.encode"] > 0
    assert timings["phase_counts"]["codec.encode"] == 2
    assert timings["average_phase_ms"]["codec.encode"] > 0
    assert timings["phase_percent"]["codec.encode"] > 0
    assert timings["average_unit_ms"] > 0


def test_measure_phase_is_optional_outside_collector():
    with measure_phase("unused"):
        pass
