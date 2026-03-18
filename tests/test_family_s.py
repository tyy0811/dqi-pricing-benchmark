"""Tests for Family S synthetic structure-sweep instances."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, ".")


def test_family_s_manifest_is_loadable_and_has_instances():
    from src.family_s_manifest import load_family_s_manifest

    manifest = load_family_s_manifest()
    assert manifest["family"] == "S"
    assert "instances" in manifest
    assert len(manifest["instances"]) >= 6


def test_family_s_all_instances_load_and_have_expected_dimensions():
    from src.family_s_manifest import load_family_s_family

    family = load_family_s_family()
    assert len(family) >= 6
    for inst in family:
        assert inst["family"] == "S"
        B = np.asarray(inst["B"], dtype=np.int8)
        v = np.asarray(inst["v"], dtype=np.int8)
        assert B.ndim == 2
        assert v.ndim == 1
        assert B.shape[0] == int(inst["m_rows"])
        assert B.shape[1] == int(inst["n_bits"])
        assert v.shape[0] == int(inst["m_rows"])


def test_family_s_structure_metrics_compute_for_every_instance():
    from src.family_s_manifest import load_family_s_family
    from src.structure_metrics import compute_structure_metrics

    family = load_family_s_family()
    for inst in family:
        B = np.asarray(inst["B"], dtype=np.int8)
        ell = int(inst.get("ell_default", 2))
        metrics = compute_structure_metrics(B, ell=ell)
        assert "row_weight_mean" in metrics
        assert "col_weight_mean" in metrics
        assert "decoder_collision_rate" in metrics
        assert "ambiguous_syndrome_count" in metrics
        assert "rank_deficiency" in metrics
        assert "decodable_syndrome_fraction" in metrics
        assert "approximate_distance_proxy" in metrics


def test_family_s_structure_metrics_schema_has_frozen_numeric_columns():
    from src.family_s_manifest import load_family_s_family
    from src.structure_metrics import compute_structure_metrics

    required = {
        "row_weight_min",
        "row_weight_mean",
        "row_weight_max",
        "col_weight_min",
        "col_weight_mean",
        "col_weight_max",
        "rank",
        "rank_deficiency",
        "decoder_collision_rate",
        "ambiguous_syndrome_count",
        "short_cycle_proxy",
        "distance_proxy",
    }

    inst = load_family_s_family()[0]
    metrics = compute_structure_metrics(np.asarray(inst["B"], dtype=np.int8), ell=int(inst["ell_default"]))
    assert required.issubset(metrics.keys())
    for key in required:
        value = metrics[key]
        assert value is None or isinstance(value, (int, float, np.floating, np.integer))


def test_family_s_regeneration_from_manifest_parameters_is_deterministic():
    from src.family_s_generator import regenerate_family_s_from_manifest
    from src.family_s_manifest import load_family_s_family

    loaded = {inst["instance_id"]: inst for inst in load_family_s_family()}
    regenerated = regenerate_family_s_from_manifest(write_artifacts=False)

    assert len(regenerated) == len(loaded)
    for inst in regenerated:
        original = loaded[inst["instance_id"]]
        np.testing.assert_array_equal(np.asarray(inst["B"], dtype=np.int8), np.asarray(original["B"], dtype=np.int8))
        np.testing.assert_array_equal(np.asarray(inst["v"], dtype=np.int8), np.asarray(original["v"], dtype=np.int8))


def test_structure_sweep_benchmark_runs_for_one_instance_without_schema_break(tmp_path):
    from src.benchmark import run_structure_sweep_benchmark

    out = run_structure_sweep_benchmark(
        instance_ids=["S_dense_low_collision"],
        n_dqi_samples=32,
        ell=2,
        seed=11,
        output_dir=tmp_path,
        write_json=False,
    )
    assert len(out) == 1
    record = out[0]
    assert record["family"] == "S"
    assert record["instance_id"] == "S_dense_low_collision"
    assert "structure_metrics" in record
    assert "dqi_by_decoder" in record
    assert "bruteforce" in record["dqi_by_decoder"]
    brute = record["dqi_by_decoder"]["bruteforce"]
    assert "decision_metrics" in brute
    assert "resources" in brute
    assert "decoder_diagnostics" in brute
