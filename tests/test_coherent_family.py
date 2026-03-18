"""Tests for Stage 2A coherent tiny-instance artifact family."""

from __future__ import annotations

from pathlib import Path
import copy
import sys

import numpy as np

sys.path.insert(0, ".")


def test_coherent_artifact_files_exist():
    base = Path("data/instances/coherent")
    required = [
        "coherent_manifest.json",
        "C1.json",
        "C1.npz",
        "C2.json",
        "C2.npz",
        "C3.json",
        "C3.npz",
    ]
    for name in required:
        assert (base / name).exists(), f"missing {name}"


def test_load_coherent_manifest_and_family_order():
    from src.coherent_reference import load_coherent_family, load_coherent_manifest

    manifest = load_coherent_manifest()
    assert manifest["canonical_instance_order"] == ["C1", "C2", "C3"]

    family = load_coherent_family()
    assert [rec["meta"]["instance_id"] for rec in family] == ["C1", "C2", "C3"]


def test_validate_all_coherent_instances_pass():
    from src.coherent_reference import load_coherent_family, validate_coherent_instance

    for rec in load_coherent_family():
        out = validate_coherent_instance(rec["meta"], rec["arrays"], strict=True)
        assert out["ok"] is True


def test_manifest_order_mismatch_fails_clearly():
    from src.coherent_reference import load_coherent_manifest, validate_coherent_manifest

    manifest = load_coherent_manifest()
    broken = copy.deepcopy(manifest)
    broken["canonical_instance_order"] = ["C2", "C1", "C3"]

    try:
        validate_coherent_manifest(broken, base_dir="data/instances/coherent")
    except ValueError as exc:
        assert "canonical_instance_order" in str(exc)
    else:
        raise AssertionError("Expected ValueError for canonical order mismatch")


def test_shape_mismatch_between_json_and_npz_fails_clearly():
    from src.coherent_reference import load_coherent_instance, validate_coherent_instance

    rec = load_coherent_instance("C1")
    bad_arrays = dict(rec["arrays"])
    bad_arrays["B"] = bad_arrays["B"][:3]

    try:
        validate_coherent_instance(rec["meta"], bad_arrays, strict=True)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for shape mismatch")


def test_non_binary_b_or_v_fails_clearly():
    from src.coherent_reference import load_coherent_instance, validate_coherent_instance

    rec = load_coherent_instance("C1")
    bad_arrays = dict(rec["arrays"])
    bad_B = bad_arrays["B"].copy()
    bad_B[0, 0] = 2
    bad_arrays["B"] = bad_B

    try:
        validate_coherent_instance(rec["meta"], bad_arrays, strict=True)
    except ValueError as exc:
        assert "binary" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-binary B")


def test_invalid_optimizer_set_fails_clearly():
    from src.coherent_reference import load_coherent_instance, validate_coherent_instance

    rec = load_coherent_instance("C1")
    bad_meta = copy.deepcopy(rec["meta"])
    bad_meta["optimizer_set"] = [1, 0]

    try:
        validate_coherent_instance(bad_meta, rec["arrays"], strict=True)
    except ValueError as exc:
        assert "optimizer_set" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid optimizer_set")


def test_manifest_missing_path_fails_clearly():
    from src.coherent_reference import load_coherent_manifest, validate_coherent_manifest

    manifest = load_coherent_manifest()
    broken = copy.deepcopy(manifest)
    broken["instances"][0]["json_path"] = "missing_file.json"

    try:
        validate_coherent_manifest(broken, base_dir="data/instances/coherent")
    except ValueError as exc:
        assert "json_path" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing manifest json_path")


def test_c1_and_c3_are_exact_max_xorsat_truth_consistent():
    from src.coherent_reference import load_coherent_instance
    from src.reduction import surrogate_objective

    for instance_id in ("C1", "C3"):
        rec = load_coherent_instance(instance_id)
        meta = rec["meta"]
        arr = rec["arrays"]
        n_bits = int(meta["n_bits"])
        B = np.asarray(arr["B"], dtype=np.int8)
        v = np.asarray(arr["v"], dtype=np.int8)
        weights = np.ones(B.shape[0], dtype=np.float64)

        g_expected = np.array(
            [surrogate_objective(x, B, v, weights, n_bits) for x in range(2 ** n_bits)],
            dtype=np.float64,
        )
        f_table = np.asarray(arr["truth_table_F"], dtype=np.float64)
        g_table = np.asarray(arr["truth_table_G"], dtype=np.float64)

        np.testing.assert_allclose(g_table, g_expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(f_table, g_table, rtol=1e-10, atol=1e-10)


def test_c2_has_pricing_and_exact_linkage_paths():
    from src.coherent_reference import load_coherent_instance

    rec = load_coherent_instance("C2")
    meta = rec["meta"]

    assert meta["pricing_source_path"] is not None
    assert meta["exact_encoding_source_path"] is not None
    assert (Path("data/instances/coherent") / meta["pricing_source_path"]).exists()
    assert (Path("data/instances/coherent") / meta["exact_encoding_source_path"]).exists()


def test_all_c_instances_mark_coherent_supported_true():
    from src.coherent_reference import load_coherent_family

    family = load_coherent_family()
    assert family
    for rec in family:
        assert rec["meta"]["coherent_supported"] is True


def test_all_c_instances_satisfy_coherent_size_limits():
    from src.coherent_reference import coherent_supported_for_instance, load_coherent_family

    for rec in load_coherent_family():
        meta = rec["meta"]
        assert coherent_supported_for_instance(
            n_bits=int(meta["n_bits"]),
            m_rows=int(meta["m_rows"]),
            ell_allowed=meta["ell_allowed"],
        )
