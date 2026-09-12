import json

import pytest

from trainlm.benchmark import (
    BenchmarkResult,
    evaluate_repeated_parity,
    load_repeated_parity_evaluation,
)


def result(tokens, *, fingerprint="sha256:same", **overrides):
    values = dict(
        schema_version=1,
        run_id=f"run-{tokens}",
        workload_id="laughlm-135m-v5e8",
        workload_version=1,
        framework="trainlm",
        framework_revision="test",
        backend="pytorch-xla",
        accelerator_type="v5e-8",
        measurement_kind="steady_state",
        cache_state="warm",
        device_synchronized=True,
        device_count=8,
        host_count=1,
        data_parallel_replicas=8,
        warmup_steps=3,
        measured_steps=10,
        scheduled_tokens_per_update=1_048_576,
        total_step_seconds_median=1.0,
        device_step_seconds_median=1.0,
        compile_seconds=1.0,
        peak_hbm_gib=5.0,
        input_idle_fraction=0.01,
        collective_seconds_median=0.05,
        compile_count=1,
        unexpected_compile_count=0,
        cpu_fallback_count=0,
        metadata={"hlo_fingerprint": fingerprint},
    )
    values.update(overrides)
    return BenchmarkResult.from_measurement(
        supervised_tokens_per_replica=(tokens // 8,) * 8,
        ignored_tokens_per_replica=(0,) * 8,
        measurement_wall_seconds=1.0,
        measurement_device_seconds=1.0,
        non_embedding_flops_per_token=4_200_000.0,
        logits_inclusive_flops_per_token=4_500_000.0,
        peak_flops_per_device=1_000_000_000_000.0,
        **values,
    )


def test_three_hard_and_preferred_runs_pass():
    evaluation = evaluate_repeated_parity(
        (result(980_000), result(1_000_000), result(1_020_000))
    )
    assert evaluation.passed
    assert evaluation.preferred_passed
    assert evaluation.median_global_tokens_per_second == 1_000_000
    assert evaluation.throughput_relative_spread == 0.04


def test_every_run_must_clear_hard_thresholds():
    evaluation = evaluate_repeated_parity(
        (result(900_000), result(980_000), result(1_000_000))
    )
    assert not evaluation.passed
    assert "run 1 is below the hard throughput gate" in evaluation.reasons


def test_matching_hlo_and_geometry_are_required():
    evaluation = evaluate_repeated_parity(
        (
            result(980_000),
            result(990_000, fingerprint="sha256:different"),
            result(1_000_000, scheduled_tokens_per_update=42),
        )
    )
    assert not evaluation.passed
    assert evaluation.hlo_fingerprint is None
    assert "run 3 does not match the reference run geometry" in evaluation.reasons


def test_exactly_three_runs_are_required():
    try:
        evaluate_repeated_parity((result(980_000),))
    except ValueError as error:
        assert "exactly three" in str(error)
    else:
        raise AssertionError("single run was accepted")


def _write_manifest(tmp_path, paths):
    manifest = tmp_path / "repeated.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "result_paths": paths,
                "thresholds": {
                    "hard_throughput": 912_600.0,
                    "hard_mfu": 0.478,
                    "preferred_throughput": 963_300.0,
                    "preferred_mfu": 0.504,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_manifest_loads_three_distinct_results(tmp_path):
    paths = []
    for index, tokens in enumerate((980_000, 1_000_000, 1_020_000), start=1):
        name = f"run-{index}.json"
        (tmp_path / name).write_text(result(tokens).to_json(), encoding="utf-8")
        paths.append(name)

    evaluation = load_repeated_parity_evaluation(
        _write_manifest(tmp_path, paths)
    )

    assert evaluation.passed
    assert evaluation.preferred_passed
    assert evaluation.median_global_tokens_per_second == 1_000_000


def test_manifest_rejects_duplicate_and_escaping_result_paths(tmp_path):
    (tmp_path / "run.json").write_text(result(1_000_000).to_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct"):
        load_repeated_parity_evaluation(
            _write_manifest(tmp_path, ["run.json", "run.json", "run.json"])
        )

    outside = tmp_path.parent / "outside-result.json"
    outside.write_text(result(1_000_000).to_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_repeated_parity_evaluation(
            _write_manifest(tmp_path, ["run.json", "../outside-result.json", "third.json"])
        )


def test_manifest_rejects_unknown_threshold_fields(tmp_path):
    manifest = _write_manifest(tmp_path, ["one.json", "two.json", "three.json"])
    values = json.loads(manifest.read_text(encoding="utf-8"))
    values["thresholds"]["unversioned"] = 1
    manifest.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(ValueError, match="versioned parity gate"):
        load_repeated_parity_evaluation(manifest)
