"""Shared helpers for fixed 3-period pricing tests."""

from __future__ import annotations

import numpy as np

from src.problem_generator import (
    BundleConstraint,
    CustomerSegment,
    Feature,
    TIER_PREMIUM,
)
from src.problem_generator_multiperiod import MultiPeriodPricingProblem


def make_tiny_multiperiod_problem() -> MultiPeriodPricingProblem:
    return MultiPeriodPricingProblem(
        features=[
            Feature("Seats", 400, 700, 1000),
            Feature("Roof", 300, 600, 900),
        ],
        segments=[
            CustomerSegment("Budget", 900, 0.6),
            CustomerSegment("Premium", 1500, 0.4),
        ],
        bundle_constraints=[
            BundleConstraint(
                name="Premium roof requires premium seats",
                feature_a=1,
                tier_a=TIER_PREMIUM,
                feature_b=0,
                allowed_tiers_b={TIER_PREMIUM, 2},
            )
        ],
        period_segment_multipliers=np.array(
            [
                [1.0, 1.0],
                [1.2, 0.8],
                [0.7, 1.3],
            ],
            dtype=np.float64,
        ),
        initial_inventory=3,
        inventory_consumption=np.array([1, 1], dtype=np.int64),
        per_period_capacity_caps=[2, 2, 2],
        smoothing_penalty=5.0,
    )


def make_identical_multiplier_multiperiod_problem() -> MultiPeriodPricingProblem:
    prob = make_tiny_multiperiod_problem()
    prob.period_segment_multipliers[:, :] = 1.0
    prob._period_truth_tables.clear()
    return prob
