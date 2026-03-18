"""Tests for the fixed 3-period pricing problem model."""

from __future__ import annotations

import numpy as np
import pytest
import sys

sys.path.insert(0, ".")

from src.problem_generator import (
    BundleConstraint,
    CustomerSegment,
    Feature,
    TIER_PREMIUM,
    TIER_STANDARD,
    encode_tiers,
)
from src.problem_generator_multiperiod import MultiPeriodPricingProblem
from tests._multiperiod_helpers import make_tiny_multiperiod_problem


def test_multiperiod_problem_roundtrip_and_contract():
    prob = make_tiny_multiperiod_problem()
    payload = prob.to_dict()
    clone = MultiPeriodPricingProblem.from_dict(payload)

    assert prob.n_periods == 3
    assert clone.n_periods == 3
    assert clone.period_segment_multipliers.shape == (3, 2)
    assert clone.initial_inventory == 3
    assert clone.inventory_consumption.tolist() == [1, 1]


def test_multiperiod_problem_rejects_non_3_period_inputs():
    with pytest.raises(ValueError, match="exactly 3 periods"):
        MultiPeriodPricingProblem(
            features=[Feature("Seats", 400, 700, 1000)],
            segments=[CustomerSegment("Budget", 900, 1.0)],
            bundle_constraints=[],
            period_segment_multipliers=np.array([[1.0], [1.0]], dtype=np.float64),
            initial_inventory=2,
            inventory_consumption=np.array([1], dtype=np.int64),
        )


def test_evaluate_horizon_tracks_inventory_and_period_weights():
    prob = make_tiny_multiperiod_problem()
    x_std = encode_tiers([TIER_STANDARD, TIER_STANDARD], prob.n_features)
    x_mix = encode_tiers([TIER_PREMIUM, TIER_STANDARD], prob.n_features)
    x_high = encode_tiers([TIER_PREMIUM, TIER_PREMIUM], prob.n_features)

    out = prob.evaluate_horizon([x_std, x_mix, x_high])

    assert out["n_periods"] == 3
    assert out["capacity_usage"]["inventory_consumed_per_period"] == [0, 1, 2]
    assert out["capacity_usage"]["remaining_inventory_after_period"] == [3, 2, 0]
    assert len(out["per_period_configuration"]) == 3
    assert out["per_period_configuration"][0]["period_idx"] == 0
    assert out["per_period_configuration"][2]["period_idx"] == 2


def test_build_period_truth_table_is_period_specific():
    prob = make_tiny_multiperiod_problem()
    t0 = prob.build_period_truth_table(0)
    t2 = prob.build_period_truth_table(2)

    assert t0.shape == t2.shape == (prob.n_configurations,)
    assert not np.allclose(t0, t2)
