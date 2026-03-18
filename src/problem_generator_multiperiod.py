"""Minimal fixed-3-period pricing extension with shared-segment demand shifts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.problem_generator import (
    BundleConstraint,
    CustomerSegment,
    Feature,
    PricingProblem,
    TIER_INVALID,
    TIER_STANDARD,
    decode_bitstring,
)


class MultiPeriodPricingProblem:
    """Strictly 3-period pricing problem with shared horizon inventory.

    The feature-price structure and segment identities are shared across all
    periods. Demand varies only through period-specific multipliers applied to
    the shared segment weights.
    """

    def __init__(
        self,
        features: list[Feature],
        segments: list[CustomerSegment],
        bundle_constraints: list[BundleConstraint],
        period_segment_multipliers: np.ndarray,
        initial_inventory: int,
        inventory_consumption: np.ndarray,
        *,
        max_luxury: int = 2,
        penalty_invalid: float = 50_000.0,
        penalty_bundle: float = 50_000.0,
        penalty_luxury_cap: float = 50_000.0,
        per_period_capacity_caps: list[int] | None = None,
        smoothing_penalty: float = 0.0,
        inventory_violation_penalty: float = 1_000_000.0,
    ) -> None:
        multipliers = np.asarray(period_segment_multipliers, dtype=np.float64)
        if multipliers.shape[0] != 3:
            raise ValueError("MultiPeriodPricingProblem is fixed to exactly 3 periods.")
        if multipliers.shape[1] != len(segments):
            raise ValueError("period_segment_multipliers must align with shared segments.")

        inventory = np.asarray(inventory_consumption, dtype=np.int64)
        if inventory.shape != (len(features),):
            raise ValueError("inventory_consumption must have one entry per feature.")

        if per_period_capacity_caps is not None and len(per_period_capacity_caps) != 3:
            raise ValueError("per_period_capacity_caps must have exactly 3 entries.")

        self.features = features
        self.segments = segments
        self.bundle_constraints = bundle_constraints
        self.period_segment_multipliers = multipliers
        self.initial_inventory = int(initial_inventory)
        self.inventory_consumption = inventory
        self.max_luxury = int(max_luxury)
        self.penalty_invalid = float(penalty_invalid)
        self.penalty_bundle = float(penalty_bundle)
        self.penalty_luxury_cap = float(penalty_luxury_cap)
        self.per_period_capacity_caps = (
            [int(v) for v in per_period_capacity_caps]
            if per_period_capacity_caps is not None
            else None
        )
        self.smoothing_penalty = float(smoothing_penalty)
        self.inventory_violation_penalty = float(inventory_violation_penalty)
        self.n_periods = 3

        self.n_features = len(features)
        self.n_bits = 2 * self.n_features
        self.n_configurations = 2 ** self.n_bits

        self._period_truth_tables: dict[int, np.ndarray] = {}
        self.inventory_consumption_by_config = np.array(
            [
                self.inventory_consumption_for_tiers(decode_bitstring(x, self.n_features))
                for x in range(self.n_configurations)
            ],
            dtype=np.int64,
        )

    def _segments_for_period(self, period_idx: int) -> list[CustomerSegment]:
        if period_idx < 0 or period_idx >= 3:
            raise IndexError("period_idx must be in {0, 1, 2}.")
        period_segments = []
        for seg, mult in zip(self.segments, self.period_segment_multipliers[period_idx]):
            period_segments.append(
                CustomerSegment(
                    name=seg.name,
                    budget_cap=seg.budget_cap,
                    weight=float(seg.weight * mult),
                )
            )
        return period_segments

    def _period_problem(self, period_idx: int) -> PricingProblem:
        return PricingProblem(
            features=self.features,
            segments=self._segments_for_period(period_idx),
            bundle_constraints=self.bundle_constraints,
            max_luxury=self.max_luxury,
            penalty_invalid=self.penalty_invalid,
            penalty_bundle=self.penalty_bundle,
            penalty_luxury_cap=self.penalty_luxury_cap,
        )

    def inventory_consumption_for_tiers(self, tiers: list[int]) -> int:
        """Return inventory consumed by a configuration in one period.

        Standard and invalid tiers consume no shared stock in this first-pass
        model. Premium and luxury tiers consume the configured per-feature
        inventory units.
        """
        total = 0
        for tier, units in zip(tiers, self.inventory_consumption):
            if tier not in {TIER_STANDARD, TIER_INVALID}:
                total += int(units)
        return int(total)

    def evaluate_period(self, period_idx: int, x: int) -> float:
        """Return the period-specific objective for one configuration."""
        return float(self._period_problem(period_idx).evaluate(int(x)))

    def build_period_truth_table(self, period_idx: int) -> np.ndarray:
        """Compute the period-specific truth table for tiny sanity checks."""
        if period_idx not in self._period_truth_tables:
            table = np.zeros(self.n_configurations, dtype=np.float64)
            for x in range(self.n_configurations):
                table[x] = self.evaluate_period(period_idx, x)
            self._period_truth_tables[period_idx] = table
        return self._period_truth_tables[period_idx].copy()

    def evaluate_horizon(self, x_by_period: list[int]) -> dict:
        """Evaluate a fixed 3-period sequence under shared inventory."""
        if len(x_by_period) != 3:
            raise ValueError("x_by_period must have exactly 3 entries.")

        remaining_inventory = int(self.initial_inventory)
        consumed_by_period: list[int] = []
        remaining_after_period: list[int] = []
        per_period_configuration: list[dict] = []
        total_objective = 0.0
        total_penalty = 0.0

        for period_idx, x in enumerate(x_by_period):
            x = int(x)
            tiers = decode_bitstring(x, self.n_features)
            period_objective = self.evaluate_period(period_idx, x)
            used = int(self.inventory_consumption_by_config[x])

            if self.per_period_capacity_caps is not None and used > self.per_period_capacity_caps[period_idx]:
                total_penalty += self.inventory_violation_penalty

            remaining_inventory -= used
            if remaining_inventory < 0:
                total_penalty += self.inventory_violation_penalty

            consumed_by_period.append(used)
            remaining_after_period.append(max(remaining_inventory, 0))
            total_objective += period_objective
            per_period_configuration.append(
                {
                    "period_idx": period_idx,
                    "x": x,
                    "bitstring": format(x, f"0{self.n_bits}b"),
                    "tiers": tiers,
                    "inventory_consumed": used,
                    "remaining_inventory": max(remaining_inventory, 0),
                    "period_objective": float(period_objective),
                }
            )

        smoothing_cost = 0.0
        for left, right in zip(x_by_period, x_by_period[1:]):
            if int(left) != int(right):
                smoothing_cost += self.smoothing_penalty

        horizon_revenue = float(total_objective - total_penalty - smoothing_cost)
        return {
            "n_periods": 3,
            "horizon_revenue": horizon_revenue,
            "per_period_configuration": per_period_configuration,
            "capacity_usage": {
                "inventory_consumed_per_period": consumed_by_period,
                "remaining_inventory_after_period": remaining_after_period,
            },
            "smoothing_penalty_total": float(smoothing_cost),
        }

    def to_dict(self) -> dict:
        """Serialize the fixed 3-period problem to a compact dictionary."""
        return {
            "n_periods": 3,
            "features": [{"name": f.name, "prices": list(f.prices)} for f in self.features],
            "segments": [
                {"name": s.name, "budget_cap": s.budget_cap, "weight": s.weight}
                for s in self.segments
            ],
            "bundle_constraints": [
                {
                    "name": bc.name,
                    "feature_a": bc.feature_a,
                    "tier_a": bc.tier_a,
                    "feature_b": bc.feature_b,
                    "allowed_tiers_b": sorted(bc.allowed_tiers_b),
                }
                for bc in self.bundle_constraints
            ],
            "period_segment_multipliers": self.period_segment_multipliers.tolist(),
            "initial_inventory": self.initial_inventory,
            "inventory_consumption": self.inventory_consumption.tolist(),
            "max_luxury": self.max_luxury,
            "penalty_invalid": self.penalty_invalid,
            "penalty_bundle": self.penalty_bundle,
            "penalty_luxury_cap": self.penalty_luxury_cap,
            "per_period_capacity_caps": self.per_period_capacity_caps,
            "smoothing_penalty": self.smoothing_penalty,
            "inventory_violation_penalty": self.inventory_violation_penalty,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MultiPeriodPricingProblem":
        """Reconstruct a fixed 3-period problem from serialized data."""
        if int(payload.get("n_periods", 3)) != 3:
            raise ValueError("MultiPeriodPricingProblem is fixed to exactly 3 periods.")

        features = [
            Feature(
                name=f["name"],
                price_standard=f["prices"][0],
                price_premium=f["prices"][1],
                price_luxury=f["prices"][2],
            )
            for f in payload["features"]
        ]
        segments = [
            CustomerSegment(
                name=s["name"],
                budget_cap=s["budget_cap"],
                weight=s["weight"],
            )
            for s in payload["segments"]
        ]
        bundle_constraints = [
            BundleConstraint(
                name=bc["name"],
                feature_a=bc["feature_a"],
                tier_a=bc["tier_a"],
                feature_b=bc["feature_b"],
                allowed_tiers_b=set(bc["allowed_tiers_b"]),
            )
            for bc in payload["bundle_constraints"]
        ]
        return cls(
            features=features,
            segments=segments,
            bundle_constraints=bundle_constraints,
            period_segment_multipliers=np.asarray(payload["period_segment_multipliers"], dtype=np.float64),
            initial_inventory=int(payload["initial_inventory"]),
            inventory_consumption=np.asarray(payload["inventory_consumption"], dtype=np.int64),
            max_luxury=int(payload.get("max_luxury", 2)),
            penalty_invalid=float(payload.get("penalty_invalid", 50_000.0)),
            penalty_bundle=float(payload.get("penalty_bundle", 50_000.0)),
            penalty_luxury_cap=float(payload.get("penalty_luxury_cap", 50_000.0)),
            per_period_capacity_caps=payload.get("per_period_capacity_caps"),
            smoothing_penalty=float(payload.get("smoothing_penalty", 0.0)),
            inventory_violation_penalty=float(payload.get("inventory_violation_penalty", 1_000_000.0)),
        )

    def save_json(self, path: str | Path) -> None:
        """Save the problem definition as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
