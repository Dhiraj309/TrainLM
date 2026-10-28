import pytest

from trainlm.optimization import (
    BatchPrefetchGeometry,
    BatchPrefetchMeasurement,
    select_batch_prefetch_geometry,
)


TOKENS = 4096 * 2 * 32 * 8


def measurement(name, micro_batch, accumulation, prefetch, throughput, **values):
    defaults = dict(
        peak_hbm_gib=12.0,
        input_idle_fraction=0.02,
        graph_stable=True,
        cpu_fallback_count=0,
        real_data=True,
    )
    return BatchPrefetchMeasurement(
        geometry=BatchPrefetchGeometry(
            geometry_id=name,
            sequence_length=4096,
            micro_batch_per_device=micro_batch,
            gradient_accumulation_steps=accumulation,
            data_parallel_replicas=8,
            prefetch_depth=prefetch,
        ),
        global_tokens_per_second=throughput,
        **{**defaults, **values},
    )


def select(values):
    return select_batch_prefetch_geometry(
        values,
        expected_tokens_per_update=TOKENS,
        maximum_peak_hbm_gib=16.0,
    )


def test_selects_fastest_matched_production_geometry():
    result = select(
        (
            measurement("mb2-ga32-p16", 2, 32, 16, 900_000),
            measurement("mb1-ga64-p16", 1, 64, 16, 850_000),
        )
    )
    assert result.selected.geometry.geometry_id == "mb2-ga32-p16"


def test_invalid_candidates_have_actionable_rejections():
    result = select(
        (
            measurement("valid", 2, 32, 16, 800_000),
            measurement("fake", 2, 32, 16, 1_000_000, real_data=False),
            measurement("idle", 2, 32, 16, 1_000_000, input_idle_fraction=0.1),
            measurement("hbm", 2, 32, 16, 1_000_000, peak_hbm_gib=17.0),
        )
    )
    assert result.selected.geometry.geometry_id == "valid"
    assert result.rejected == (
        ("fake", "measurement did not use real data"),
        ("hbm", "peak HBM exceeds budget"),
        ("idle", "input idle fraction exceeds budget"),
    )


def test_ties_are_independent_of_measurement_order():
    values = (
        measurement("b", 2, 32, 16, 900_000),
        measurement("a", 1, 64, 16, 900_000),
    )
    assert select(values) == select(tuple(reversed(values)))
    assert select(values).selected.geometry.geometry_id == "a"


def test_mismatched_token_geometry_is_not_compared():
    result = select(
        (
            measurement("valid", 2, 32, 16, 800_000),
            measurement("wrong", 2, 16, 16, 2_000_000),
        )
    )
    assert result.rejected == (("wrong", "scheduled token geometry does not match"),)


def test_no_eligible_geometry_is_an_error():
    with pytest.raises(ValueError, match="No batch/prefetch geometry"):
        select((measurement("fake", 2, 32, 16, 1_000_000, real_data=False),))


def test_geometry_materializes_parallel_loader_configuration():
    geometry = measurement("production", 2, 32, 16, 900_000).geometry
    assert geometry.parallel_loader_kwargs() == {
        "loader_prefetch_size": 16,
        "device_prefetch_size": 8,
        "host_to_device_transfer_threads": 1,
        "batches_per_execution": 1,
    }


@pytest.mark.parametrize(
    "field",
    [
        "device_prefetch_depth",
        "host_to_device_transfer_threads",
        "batches_per_execution",
    ],
)
def test_parallel_loader_geometry_values_must_be_positive(field):
    values = dict(
        geometry_id="invalid",
        sequence_length=4096,
        micro_batch_per_device=2,
        gradient_accumulation_steps=32,
        data_parallel_replicas=8,
        prefetch_depth=16,
        **{field: 0},
    )
    with pytest.raises(ValueError, match=field):
        BatchPrefetchGeometry(**values)
