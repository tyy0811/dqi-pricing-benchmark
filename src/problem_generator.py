"""
problem_generator.py — Discrete Vehicle Package Pricing Problem Generator

Part of dqi-pricing-benchmark: a benchmark mapping discrete automotive pricing
into DQI-compatible binary optimization (Jordan et al., arXiv:2408.08292).

This module remains the single-period pricing problem surface. For the separate
fixed 3-period extension, see ``src.problem_generator_multiperiod``.

Encoding: 2 bits per feature (00=standard, 01=premium, 10=luxury, 11=invalid).
Objective: expected revenue across customer segments minus constraint penalties.

Usage:
    from src.problem_generator import PricingProblem, default_instance
    prob = default_instance()
    truth_table = prob.build_truth_table()
    x_opt, f_opt = prob.brute_force_solve()
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Feature:
    """A vehicle optional feature with 3 price tiers."""
    name: str
    price_standard: float
    price_premium: float
    price_luxury: float

    @property
    def prices(self) -> tuple[float, float, float]:
        return (self.price_standard, self.price_premium, self.price_luxury)


@dataclass
class CustomerSegment:
    """A customer segment defined by a budget cap and population weight."""
    name: str
    budget_cap: float
    weight: float


@dataclass
class BundleConstraint:
    """
    A constraint of the form:
      if feature_a is at tier_a, then feature_b must be in allowed_tiers_b.

    tier_a: int in {0, 1, 2} (standard, premium, luxury)
    allowed_tiers_b: set of int, e.g. {1, 2} means "at least premium"
    """
    name: str
    feature_a: int          # index of the triggering feature
    tier_a: int             # tier that triggers the constraint
    feature_b: int          # index of the constrained feature
    allowed_tiers_b: set    # set of allowed tiers for feature_b


# ---------------------------------------------------------------------------
# 2-bit encoding utilities
# ---------------------------------------------------------------------------

# Tier codes: 2 bits per feature
TIER_STANDARD = 0   # bits: 00
TIER_PREMIUM  = 1   # bits: 01
TIER_LUXURY   = 2   # bits: 10
TIER_INVALID  = 3   # bits: 11


def tier_from_bits(a: int, b: int) -> int:
    """Convert 2 raw bits (a, b) to a tier index (0-3).

    Bit layout per feature:
        a  b  →  tier
        0  0  →  0 (standard)
        0  1  →  1 (premium)
        1  0  →  2 (luxury)
        1  1  →  3 (invalid)
    """
    return 2 * a + b


def bits_from_tier(tier: int) -> tuple[int, int]:
    """Inverse of tier_from_bits."""
    return (tier >> 1) & 1, tier & 1


def decode_bitstring(x: int, n_features: int) -> list[int]:
    """Decode an integer bitstring into a list of tier indices.

    Bit ordering: feature 0 uses the two highest-order bits, etc.
    For n_features=5, x is a 10-bit integer:
        bits [9,8] → feature 0
        bits [7,6] → feature 1
        ...
        bits [1,0] → feature 4

    Returns:
        List of tier indices (0=standard, 1=premium, 2=luxury, 3=invalid).
    """
    n_bits = 2 * n_features
    tiers = []
    for i in range(n_features):
        shift = n_bits - 2 * (i + 1)
        a = (x >> (shift + 1)) & 1
        b = (x >> shift) & 1
        tiers.append(tier_from_bits(a, b))
    return tiers


def encode_tiers(tiers: list[int], n_features: int) -> int:
    """Encode a list of tier indices into an integer bitstring."""
    n_bits = 2 * n_features
    x = 0
    for i, tier in enumerate(tiers):
        a, b = bits_from_tier(tier)
        shift = n_bits - 2 * (i + 1)
        x |= (a << (shift + 1)) | (b << shift)
    return x


def bitstring_to_array(x: int, n_bits: int) -> np.ndarray:
    """Convert integer x to a numpy array of 0/1 bits (MSB first)."""
    return np.array([(x >> (n_bits - 1 - j)) & 1 for j in range(n_bits)],
                    dtype=np.int8)


def array_to_bitstring(arr: np.ndarray) -> int:
    """Convert a numpy bit array (MSB first) back to an integer."""
    x = 0
    for bit in arr:
        x = (x << 1) | int(bit)
    return x


def compute_r_max(
    features: list[Feature],
    segments: list[CustomerSegment],
) -> float:
    """Compute R_max: max expected revenue over all 2-bit tier configurations."""
    n_features = len(features)
    n_configurations = 2 ** (2 * n_features)

    r_max = 0.0
    for x in range(n_configurations):
        tiers = decode_bitstring(x, n_features)

        price = 0.0
        for feat, tier in zip(features, tiers):
            if tier != TIER_INVALID:
                price += feat.prices[tier]

        revenue = 0.0
        for seg in segments:
            if price <= seg.budget_cap:
                revenue += seg.weight * price

        if revenue > r_max:
            r_max = revenue

    return float(r_max)


def recommended_penalties_from_r_max(r_max: float) -> dict[str, float]:
    """Return plan-aligned penalty magnitudes derived from R_max."""
    return {
        "penalty_invalid": 10.0 * r_max,
        "penalty_bundle": 5.0 * r_max,
        "penalty_luxury_cap": 5.0 * r_max,
    }


def recommended_penalties(
    features: list[Feature],
    segments: list[CustomerSegment],
) -> dict[str, float]:
    """Compute recommended penalty magnitudes for a feature/segment setup."""
    r_max = compute_r_max(features, segments)
    return recommended_penalties_from_r_max(r_max)


# ---------------------------------------------------------------------------
# Pricing problem
# ---------------------------------------------------------------------------

class PricingProblem:
    """
    Discrete vehicle package pricing problem.

    The true objective F(x) is:
        F(x) = expected_revenue(x) - penalty_invalid(x)
               - penalty_bundle(x) - penalty_luxury_cap(x)

    where x is a bitstring of length 2 * n_features.
    """

    def __init__(
        self,
        features: list[Feature],
        segments: list[CustomerSegment],
        bundle_constraints: list[BundleConstraint],
        max_luxury: int = 2,
        penalty_invalid: float = 50_000.0,
        penalty_bundle: float = 50_000.0,
        penalty_luxury_cap: float = 50_000.0,
    ):
        self.features = features
        self.segments = segments
        self.bundle_constraints = bundle_constraints
        self.max_luxury = max_luxury

        # Penalty magnitudes (large enough that violations are never optimal)
        self.penalty_invalid = penalty_invalid
        self.penalty_bundle = penalty_bundle
        self.penalty_luxury_cap = penalty_luxury_cap

        self.n_features = len(features)
        self.n_bits = 2 * self.n_features
        self.n_configurations = 2 ** self.n_bits

        # Cache for truth table
        self._truth_table: Optional[np.ndarray] = None

    # -----------------------------------------------------------------------
    # Core evaluation
    # -----------------------------------------------------------------------

    def total_price(self, tiers: list[int]) -> float:
        """Compute the total package price given tier choices.

        Invalid tiers (3) contribute 0 to price (but are penalized separately).
        """
        total = 0.0
        for feat, tier in zip(self.features, tiers):
            if tier == TIER_INVALID:
                continue
            total += feat.prices[tier]
        return total

    def expected_revenue(self, tiers: list[int]) -> float:
        """Compute expected revenue across all customer segments.

        Revenue = sum_k  w_k * Price(x) * 1[Price(x) <= cap_k]
        """
        price = self.total_price(tiers)
        revenue = 0.0
        for seg in self.segments:
            if price <= seg.budget_cap:
                revenue += seg.weight * price
        return revenue

    def count_invalid(self, tiers: list[int]) -> int:
        """Count how many features are in the invalid (11) state."""
        return sum(1 for t in tiers if t == TIER_INVALID)

    def count_luxury(self, tiers: list[int]) -> int:
        """Count how many features are at the luxury tier."""
        return sum(1 for t in tiers if t == TIER_LUXURY)

    def bundle_violations(self, tiers: list[int]) -> int:
        """Count how many bundle constraints are violated."""
        violations = 0
        for bc in self.bundle_constraints:
            # Constraint fires only if feature_a is at the trigger tier
            if tiers[bc.feature_a] == bc.tier_a:
                if tiers[bc.feature_b] not in bc.allowed_tiers_b:
                    violations += 1
        return violations

    def evaluate(self, x: int) -> float:
        """Evaluate the true objective F(x) for a bitstring x (integer)."""
        tiers = decode_bitstring(x, self.n_features)

        # Revenue component
        rev = self.expected_revenue(tiers)

        # Penalty: invalid states
        n_inv = self.count_invalid(tiers)
        pen_inv = self.penalty_invalid * n_inv

        # Penalty: bundle constraint violations
        n_bundle_viol = self.bundle_violations(tiers)
        pen_bundle = self.penalty_bundle * n_bundle_viol

        # Penalty: too many luxury features
        n_lux = self.count_luxury(tiers)
        pen_lux = self.penalty_luxury_cap * max(0, n_lux - self.max_luxury)

        return rev - pen_inv - pen_bundle - pen_lux

    def evaluate_array(self, bits: np.ndarray) -> float:
        """Evaluate F(x) from a numpy bit array (MSB first)."""
        return self.evaluate(array_to_bitstring(bits))

    # -----------------------------------------------------------------------
    # Truth table and brute-force solver
    # -----------------------------------------------------------------------

    def build_truth_table(self) -> np.ndarray:
        """Compute F(x) for all 2^n_bits configurations.

        Returns:
            1D numpy array of shape (2^n_bits,) with F(x) values.
        """
        if self._truth_table is not None:
            return self._truth_table

        table = np.zeros(self.n_configurations, dtype=np.float64)
        for x in range(self.n_configurations):
            table[x] = self.evaluate(x)
        self._truth_table = table
        return table

    def brute_force_solve(self) -> tuple[int, float]:
        """Find the global optimum by exhaustive search.

        Returns:
            (x_opt, f_opt): optimal bitstring (int) and its objective value.
        """
        table = self.build_truth_table()
        x_opt = int(np.argmax(table))
        f_opt = float(table[x_opt])
        return x_opt, f_opt

    def feasible_mask(self) -> np.ndarray:
        """Boolean mask: True if configuration has no violations.

        Returns:
            1D boolean numpy array of shape (2^n_bits,).
        """
        table = self.build_truth_table()
        mask = np.ones(self.n_configurations, dtype=bool)
        for x in range(self.n_configurations):
            tiers = decode_bitstring(x, self.n_features)
            if self.count_invalid(tiers) > 0:
                mask[x] = False
            elif self.bundle_violations(tiers) > 0:
                mask[x] = False
            elif self.count_luxury(tiers) > self.max_luxury:
                mask[x] = False
        return mask

    # -----------------------------------------------------------------------
    # Analysis helpers
    # -----------------------------------------------------------------------

    def solution_summary(self, x: int) -> dict:
        """Return a human-readable summary of a solution."""
        tiers = decode_bitstring(x, self.n_features)
        tier_names = {0: "Standard", 1: "Premium", 2: "Luxury", 3: "INVALID"}

        feature_choices = {}
        for feat, tier in zip(self.features, tiers):
            price = feat.prices[tier] if tier != TIER_INVALID else 0.0
            feature_choices[feat.name] = {
                "tier": tier_names[tier],
                "price": price,
            }

        total_price = self.total_price(tiers)
        segments_buying = []
        for seg in self.segments:
            if total_price <= seg.budget_cap:
                segments_buying.append(seg.name)

        return {
            "bitstring": format(x, f"0{self.n_bits}b"),
            "tiers": [tier_names[t] for t in tiers],
            "feature_choices": feature_choices,
            "total_price": total_price,
            "expected_revenue": self.expected_revenue(tiers),
            "segments_buying": segments_buying,
            "n_invalid": self.count_invalid(tiers),
            "n_bundle_violations": self.bundle_violations(tiers),
            "n_luxury": self.count_luxury(tiers),
            "objective_F": self.evaluate(x),
        }

    def top_k_solutions(self, k: int = 10) -> list[dict]:
        """Return the top-k solutions by objective value."""
        table = self.build_truth_table()
        top_indices = np.argsort(table)[-k:][::-1]
        return [self.solution_summary(int(idx)) for idx in top_indices]

    def statistics(self) -> dict:
        """Return summary statistics of the problem landscape."""
        table = self.build_truth_table()
        mask = self.feasible_mask()

        feasible_values = table[mask]
        return {
            "n_features": self.n_features,
            "n_bits": self.n_bits,
            "n_configurations": self.n_configurations,
            "n_feasible": int(mask.sum()),
            "n_infeasible": int((~mask).sum()),
            "f_max": float(table.max()),
            "f_min": float(table.min()),
            "f_mean": float(table.mean()),
            "f_feasible_max": float(feasible_values.max()) if mask.any() else None,
            "f_feasible_min": float(feasible_values.min()) if mask.any() else None,
            "f_feasible_mean": float(feasible_values.mean()) if mask.any() else None,
        }

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize problem definition to a dictionary."""
        return {
            "features": [
                {"name": f.name, "prices": list(f.prices)}
                for f in self.features
            ],
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
            "max_luxury": self.max_luxury,
            "penalty_invalid": self.penalty_invalid,
            "penalty_bundle": self.penalty_bundle,
            "penalty_luxury_cap": self.penalty_luxury_cap,
        }

    def save_instance(self, path: str | Path) -> None:
        """Save problem definition + truth table + optimum to JSON."""
        x_opt, f_opt = self.brute_force_solve()
        summary = self.solution_summary(x_opt)
        tier_names = {0: "Standard", 1: "Premium", 2: "Luxury", 3: "INVALID"}
        tiers = decode_bitstring(x_opt, self.n_features)

        data = self.to_dict()
        data["solution"] = {
            "x_opt": x_opt,
            "x_opt_bits": format(x_opt, f"0{self.n_bits}b"),
            "x_opt_tiers": [tier_names[t] for t in tiers],
            "x_opt_price": summary["total_price"],
            "segments_buying_at_opt": summary["segments_buying"],
            "f_opt": f_opt,
        }
        data["statistics"] = self.statistics()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "PricingProblem":
        """Reconstruct a PricingProblem from a serialized dictionary."""
        features = [
            Feature(name=f["name"],
                    price_standard=f["prices"][0],
                    price_premium=f["prices"][1],
                    price_luxury=f["prices"][2])
            for f in d["features"]
        ]
        segments = [
            CustomerSegment(name=s["name"],
                            budget_cap=s["budget_cap"],
                            weight=s["weight"])
            for s in d["segments"]
        ]
        bundle_constraints = [
            BundleConstraint(
                name=bc["name"],
                feature_a=bc["feature_a"],
                tier_a=bc["tier_a"],
                feature_b=bc["feature_b"],
                allowed_tiers_b=set(bc["allowed_tiers_b"]),
            )
            for bc in d["bundle_constraints"]
        ]
        return cls(
            features=features,
            segments=segments,
            bundle_constraints=bundle_constraints,
            max_luxury=d.get("max_luxury", 2),
            penalty_invalid=d.get("penalty_invalid", 50_000.0),
            penalty_bundle=d.get("penalty_bundle", 50_000.0),
            penalty_luxury_cap=d.get("penalty_luxury_cap", 50_000.0),
        )


# ---------------------------------------------------------------------------
# Pre-built instances
# ---------------------------------------------------------------------------

def default_instance() -> PricingProblem:
    """The canonical 5-feature demo instance.

    Features: heated seats, panoramic roof, premium audio,
              adaptive cruise, ambient lighting.
    Segments: budget (cap 3000), mid-range (cap 5000), premium (cap 8000).
    Constraints: luxury audio requires ≥premium seats,
                 luxury cruise requires ≥premium roof,
                 at most 2 luxury features.
    """
    features = [
        Feature("Heated Seats",     500,  900, 1400),
        Feature("Panoramic Roof",   800, 1300, 1900),
        Feature("Premium Audio",    400,  800, 1200),
        Feature("Adaptive Cruise",  600, 1100, 1700),
        Feature("Ambient Lighting", 300,  600, 1000),
    ]

    segments = [
        CustomerSegment("Budget",    3000, 0.50),
        CustomerSegment("Mid-range", 5000, 0.35),
        CustomerSegment("Premium",   8000, 0.15),
    ]

    # Feature indices: 0=seats, 1=roof, 2=audio, 3=cruise, 4=lighting
    bundle_constraints = [
        BundleConstraint(
            name="Luxury audio requires ≥premium seats",
            feature_a=2,            # audio
            tier_a=TIER_LUXURY,     # luxury audio triggers
            feature_b=0,            # seats
            allowed_tiers_b={TIER_PREMIUM, TIER_LUXURY},
        ),
        BundleConstraint(
            name="Luxury cruise requires ≥premium roof",
            feature_a=3,            # cruise
            tier_a=TIER_LUXURY,     # luxury cruise triggers
            feature_b=1,            # roof
            allowed_tiers_b={TIER_PREMIUM, TIER_LUXURY},
        ),
    ]

    penalties = recommended_penalties(features, segments)

    return PricingProblem(
        features=features,
        segments=segments,
        bundle_constraints=bundle_constraints,
        max_luxury=2,
        penalty_invalid=penalties["penalty_invalid"],
        penalty_bundle=penalties["penalty_bundle"],
        penalty_luxury_cap=penalties["penalty_luxury_cap"],
    )


def _base_feature_catalog() -> list[Feature]:
    """Return the frozen 5-feature base catalog shared by P5/P6/P7."""
    return [
        Feature("Heated Seats", 500, 900, 1400),
        Feature("Panoramic Roof", 800, 1300, 1900),
        Feature("Premium Audio", 400, 800, 1200),
        Feature("Adaptive Cruise", 600, 1100, 1700),
        Feature("Ambient Lighting", 300, 600, 1000),
    ]


def _base_segments() -> list[CustomerSegment]:
    """Return the frozen 3-segment demand profile shared by P3/P4/P5."""
    return [
        CustomerSegment("Budget", 3000, 0.50),
        CustomerSegment("Mid-range", 5000, 0.35),
        CustomerSegment("Premium", 8000, 0.15),
    ]


def _base_bundle_constraints() -> list[BundleConstraint]:
    """Return the frozen bundle rules shared by the original pricing family."""
    return [
        BundleConstraint(
            name="Luxury audio requires ≥premium seats",
            feature_a=2,
            tier_a=TIER_LUXURY,
            feature_b=0,
            allowed_tiers_b={TIER_PREMIUM, TIER_LUXURY},
        ),
        BundleConstraint(
            name="Luxury cruise requires ≥premium roof",
            feature_a=3,
            tier_a=TIER_LUXURY,
            feature_b=1,
            allowed_tiers_b={TIER_PREMIUM, TIER_LUXURY},
        ),
    ]


def small_instance(n_features: int = 3) -> PricingProblem:
    """A smaller instance for quick testing.

    Uses the first n_features from the default set.
    Adjusts constraints to only include relevant ones.
    """
    all_features = _base_feature_catalog()

    features = all_features[:n_features]

    segments = _base_segments()

    # Only include constraints that reference features within range
    all_constraints = _base_bundle_constraints()
    bundle_constraints = [
        bc for bc in all_constraints
        if bc.feature_a < n_features and bc.feature_b < n_features
    ]

    penalties = recommended_penalties(features, segments)

    return PricingProblem(
        features=features,
        segments=segments,
        bundle_constraints=bundle_constraints,
        max_luxury=min(2, n_features),
        penalty_invalid=penalties["penalty_invalid"],
        penalty_bundle=penalties["penalty_bundle"],
        penalty_luxury_cap=penalties["penalty_luxury_cap"],
    )


def scaling_instance(n_features: int) -> PricingProblem:
    """Return a strict-superset scaling instance for 3 through 7 features."""
    if n_features < 3 or n_features > 7:
        raise ValueError("scaling_instance supports 3 <= n_features <= 7")
    if n_features <= 5:
        return default_instance() if n_features == 5 else small_instance(n_features)

    features = _base_feature_catalog()
    features.append(Feature("Head-Up Display", 400, 650, 850))
    if n_features == 7:
        features.append(Feature("Wireless Charging", 250, 450, 650))

    segments = _base_segments()
    segments.append(CustomerSegment("Tech-Forward", 9000, 0.12))

    bundle_constraints = _base_bundle_constraints()
    bundle_constraints.append(
        BundleConstraint(
            name="Premium head-up display requires adaptive cruise",
            feature_a=5,
            tier_a=TIER_PREMIUM,
            feature_b=3,
            allowed_tiers_b={TIER_STANDARD, TIER_PREMIUM, TIER_LUXURY},
        )
    )
    if n_features == 7:
        bundle_constraints.append(
            BundleConstraint(
                name="Luxury wireless charging requires premium audio",
                feature_a=6,
                tier_a=TIER_LUXURY,
                feature_b=2,
                allowed_tiers_b={TIER_PREMIUM, TIER_LUXURY},
            )
        )

    penalties = recommended_penalties(features, segments)

    return PricingProblem(
        features=features,
        segments=segments,
        bundle_constraints=bundle_constraints,
        max_luxury=3,
        penalty_invalid=penalties["penalty_invalid"],
        penalty_bundle=penalties["penalty_bundle"],
        penalty_luxury_cap=penalties["penalty_luxury_cap"],
    )


# ---------------------------------------------------------------------------
# Frozen instance generation (for reproducible benchmarks)
# ---------------------------------------------------------------------------

def generate_frozen_instances(output_dir: str | Path = "data/instances") -> None:
    """Generate frozen instances at 3, 4, and 5 features.

    Saves each as a JSON file with problem definition, truth table stats,
    and brute-force optimal solution.
    """
    output_dir = Path(output_dir)
    for n in [3, 4, 5, 6, 7]:
        prob = scaling_instance(n)

        filename = f"pricing_{n}feat_{2*n}bit.json"
        prob.save_instance(output_dir / filename)
        stats = prob.statistics()
        x_opt, f_opt = prob.brute_force_solve()
        print(f"[{n} features, {2*n} bits] "
              f"feasible: {stats['n_feasible']}/{stats['n_configurations']}, "
              f"F* = {f_opt:.2f}, "
              f"x* = {format(x_opt, f'0{2*n}b')}")


def compute_feasibility_stats(prob: PricingProblem) -> dict[str, float | int]:
    """Compute aggregate feasibility diagnostics over the full search space.

    The returned violation counts are counts of configurations that violate each
    rule category at least once.
    """
    total = prob.n_configurations
    valid_count = 0
    invalid_tier_count = 0
    bundle_violations = 0
    luxury_cap_violations = 0

    for x in range(total):
        tiers = decode_bitstring(x, prob.n_features)
        has_invalid = prob.count_invalid(tiers) > 0
        has_bundle = prob.bundle_violations(tiers) > 0
        has_luxury_cap_violation = prob.count_luxury(tiers) > prob.max_luxury

        if has_invalid:
            invalid_tier_count += 1
        if has_bundle:
            bundle_violations += 1
        if has_luxury_cap_violation:
            luxury_cap_violations += 1
        if not (has_invalid or has_bundle or has_luxury_cap_violation):
            valid_count += 1

    return {
        "total_count": int(total),
        "valid_count": int(valid_count),
        "valid_fraction": float(valid_count / total if total else 0.0),
        "invalid_tier_count": int(invalid_tier_count),
        "bundle_violations": int(bundle_violations),
        "luxury_cap_violations": int(luxury_cap_violations),
    }
