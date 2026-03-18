"""Tests for Stage A paired pricing artifact contract."""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, ".")

from src.ilp_to_xorsat_exact import (  # noqa: E402
    paired_pricing_artifact_from_dict,
    paired_pricing_artifact_to_dict,
)
from src.pricing_ilp import REQUIRED_COMPARATIVE_SUBSET, solve_pricing_pair  # noqa: E402
from src.problem_generator import scaling_instance, small_instance  # noqa: E402
from src.reduction import surrogate_artifact_from_dict, surrogate_artifact_to_dict  # noqa: E402


def _build_pair(instance_id: str = "pricing_3feat_6bit") -> dict:
    prob = small_instance(3)
    return solve_pricing_pair(
        prob,
        instance_id=instance_id,
        k=15,
        source_metadata={
            "instance_id": instance_id,
            "pricing_instance_path": "data/instances/pricing_3feat_6bit.json",
        },
    )


def _assert_shapes(enc: dict) -> None:
    n = int(enc["n"])
    m = int(enc["m"])
    assert np.asarray(enc["B"], dtype=np.int8).shape == (m, n)
    assert np.asarray(enc["v"], dtype=np.int8).shape == (m,)
    assert np.asarray(enc["term_weights"], dtype=np.float64).shape == (m,)


def test_solve_pricing_pair_contains_exactly_two_encodings_and_required_subset() -> None:
    paired = _build_pair()

    assert set(paired.keys()) == {
        "paired_artifact_version",
        "instance_id",
        "pairing_kind",
        "source_metadata",
        "encodings",
    }
    assert set(paired["encodings"].keys()) == {"wht_truncated", "ilp_derived"}

    for enc in ("wht_truncated", "ilp_derived"):
        assert REQUIRED_COMPARATIVE_SUBSET.issubset(paired["encodings"][enc].keys())


def test_paired_encodings_share_instance_id_and_n_bits() -> None:
    paired = _build_pair()
    wht = paired["encodings"]["wht_truncated"]
    ilp = paired["encodings"]["ilp_derived"]

    assert paired["instance_id"] == wht["source_metadata"]["instance_id"]
    assert paired["instance_id"] == ilp["source_metadata"]["instance_id"]
    assert int(wht["n"]) == int(ilp["n"])
    assert int(wht["m"]) == int(ilp["m"])
    assert str(wht["bit_order"]) == str(ilp["bit_order"])

    _assert_shapes(wht)
    _assert_shapes(ilp)


def test_ilp_derived_roundtrips_through_paired_json_serialization() -> None:
    paired = _build_pair()
    payload = paired_pricing_artifact_to_dict(paired)
    restored = paired_pricing_artifact_from_dict(json.loads(json.dumps(payload)))

    ilp = restored["encodings"]["ilp_derived"]
    assert isinstance(ilp["B"], np.ndarray)
    assert isinstance(ilp["v"], np.ndarray)
    assert isinstance(ilp["term_weights"], np.ndarray)
    _assert_shapes(ilp)


def test_wht_payload_stays_legacy_surrogate_compatible() -> None:
    paired = _build_pair()
    wht = paired["encodings"]["wht_truncated"]

    payload = surrogate_artifact_to_dict(wht)
    restored = surrogate_artifact_from_dict(payload)
    _assert_shapes(restored)


def test_solve_pricing_pair_supports_p6_and_p7_instances() -> None:
    for n_features in (6, 7):
        instance_id = f"pricing_{n_features}feat_{2 * n_features}bit"
        paired = solve_pricing_pair(
            scaling_instance(n_features),
            instance_id=instance_id,
            k=15,
            source_metadata={
                "instance_id": instance_id,
                "pricing_instance_path": f"data/instances/{instance_id}.json",
            },
        )

        assert paired["instance_id"] == instance_id
        assert set(paired["encodings"].keys()) == {"wht_truncated", "ilp_derived"}
        _assert_shapes(paired["encodings"]["wht_truncated"])
        _assert_shapes(paired["encodings"]["ilp_derived"])
