"""Tests for frozen pricing-family instance generation and scaling builders."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from src.problem_generator import (
    TIER_STANDARD,
    default_instance,
    decode_bitstring,
    encode_tiers,
    scaling_instance,
)


def test_scaling_instance_p6_and_p7_extend_p5_catalog_exactly():
    base = default_instance()
    p6 = scaling_instance(6)
    p7 = scaling_instance(7)

    assert p6.n_features == 6
    assert p6.n_bits == 12
    assert p7.n_features == 7
    assert p7.n_bits == 14

    for derived in (p6, p7):
        for idx in range(5):
            assert derived.features[idx].name == base.features[idx].name
            assert derived.features[idx].prices == base.features[idx].prices

        for idx in range(3):
            assert derived.segments[idx].name == base.segments[idx].name
            assert derived.segments[idx].budget_cap == base.segments[idx].budget_cap
            assert derived.segments[idx].weight == base.segments[idx].weight


def test_scaling_instance_p7_embeds_p5_optimum_feasibly_with_new_features_disabled():
    base = default_instance()
    x_opt, _f_opt = base.brute_force_solve()
    p7 = scaling_instance(7)

    base_tiers = decode_bitstring(x_opt, base.n_features)
    embedded_tiers = base_tiers + [TIER_STANDARD, TIER_STANDARD]
    embedded_x = encode_tiers(embedded_tiers, p7.n_features)

    summary = p7.solution_summary(embedded_x)
    assert summary["n_invalid"] == 0
    assert summary["n_bundle_violations"] == 0
    assert summary["n_luxury"] <= p7.max_luxury


def test_scaling_instance_adds_fourth_segment_and_relaxes_luxury_cap():
    p6 = scaling_instance(6)
    p7 = scaling_instance(7)

    assert len(p6.segments) == 4
    assert len(p7.segments) == 4
    assert p6.max_luxury == 3
    assert p7.max_luxury == 3
