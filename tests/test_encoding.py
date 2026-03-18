"""
test_encoding.py — Tests for 2-bit tier encoding and problem evaluation.

Run: python -m pytest tests/test_encoding.py -v
"""

import numpy as np
import sys
sys.path.insert(0, ".")

from src.problem_generator import (
    tier_from_bits, bits_from_tier,
    decode_bitstring, encode_tiers,
    bitstring_to_array, array_to_bitstring,
    PricingProblem, default_instance, small_instance,
    compute_r_max, recommended_penalties_from_r_max,
    TIER_STANDARD, TIER_PREMIUM, TIER_LUXURY, TIER_INVALID,
)


# ---------------------------------------------------------------------------
# Encoding roundtrip tests
# ---------------------------------------------------------------------------

def test_tier_from_bits():
    assert tier_from_bits(0, 0) == TIER_STANDARD  # 0
    assert tier_from_bits(0, 1) == TIER_PREMIUM   # 1
    assert tier_from_bits(1, 0) == TIER_LUXURY    # 2
    assert tier_from_bits(1, 1) == TIER_INVALID   # 3


def test_bits_from_tier_roundtrip():
    for tier in range(4):
        a, b = bits_from_tier(tier)
        assert tier_from_bits(a, b) == tier


def test_encode_decode_roundtrip():
    n = 5
    for _ in range(100):
        tiers = [np.random.randint(0, 4) for _ in range(n)]
        x = encode_tiers(tiers, n)
        recovered = decode_bitstring(x, n)
        assert recovered == tiers, f"{tiers} -> {x} -> {recovered}"


def test_bitstring_array_roundtrip():
    n_bits = 10
    for x in [0, 1, 512, 1023, 42]:
        arr = bitstring_to_array(x, n_bits)
        assert len(arr) == n_bits
        assert array_to_bitstring(arr) == x


def test_all_standard_is_zero_bits():
    """All-standard = all bits zero."""
    tiers = [TIER_STANDARD] * 5
    x = encode_tiers(tiers, 5)
    assert x == 0


def test_all_luxury():
    """All luxury = bits 10 10 10 10 10."""
    tiers = [TIER_LUXURY] * 5
    x = encode_tiers(tiers, 5)
    assert x == 0b10_10_10_10_10


# ---------------------------------------------------------------------------
# Objective evaluation tests
# ---------------------------------------------------------------------------

def test_all_standard_no_penalty():
    """All standard should have zero penalties."""
    prob = default_instance()
    tiers = [TIER_STANDARD] * 5
    x = encode_tiers(tiers, 5)
    summary = prob.solution_summary(x)

    assert summary["n_invalid"] == 0
    assert summary["n_bundle_violations"] == 0
    assert summary["n_luxury"] == 0
    assert summary["objective_F"] >= 0


def test_invalid_states_penalized():
    """Any configuration with tier=3 (invalid) should have large penalty."""
    prob = default_instance()

    # One invalid feature
    tiers = [TIER_STANDARD, TIER_STANDARD, TIER_STANDARD,
             TIER_STANDARD, TIER_INVALID]
    x = encode_tiers(tiers, 5)
    f_invalid = prob.evaluate(x)

    # Same but with standard instead
    tiers_ok = [TIER_STANDARD] * 5
    x_ok = encode_tiers(tiers_ok, 5)
    f_ok = prob.evaluate(x_ok)

    # Invalid should be much worse
    assert f_invalid < f_ok - 10_000


def test_bundle_constraint_violation():
    """Luxury audio without premium seats should be penalized."""
    prob = default_instance()

    # Luxury audio (idx 2), standard seats (idx 0)
    tiers = [TIER_STANDARD, TIER_STANDARD, TIER_LUXURY,
             TIER_STANDARD, TIER_STANDARD]
    x = encode_tiers(tiers, 5)
    summary = prob.solution_summary(x)
    assert summary["n_bundle_violations"] == 1

    # Luxury audio with premium seats: no violation
    tiers_ok = [TIER_PREMIUM, TIER_STANDARD, TIER_LUXURY,
                TIER_STANDARD, TIER_STANDARD]
    x_ok = encode_tiers(tiers_ok, 5)
    summary_ok = prob.solution_summary(x_ok)
    assert summary_ok["n_bundle_violations"] == 0


def test_luxury_cap():
    """More than 2 luxury features should be penalized."""
    prob = default_instance()

    # 3 luxury features
    tiers = [TIER_LUXURY, TIER_LUXURY, TIER_LUXURY,
             TIER_STANDARD, TIER_STANDARD]
    x = encode_tiers(tiers, 5)
    summary = prob.solution_summary(x)
    assert summary["n_luxury"] == 3
    # Should have luxury cap penalty
    assert summary["objective_F"] < 0


def test_demand_tradeoff_exists():
    """Verify that the optimal solution is NOT all-luxury.

    This confirms the demand model creates a real tradeoff.
    """
    prob = default_instance()
    x_opt, f_opt = prob.brute_force_solve()
    tiers_opt = decode_bitstring(x_opt, prob.n_features)

    # The optimum should not be all-luxury (because budget segments
    # won't buy, and luxury cap kicks in)
    all_luxury = [TIER_LUXURY] * 5
    assert tiers_opt != all_luxury, \
        "Optimal is all-luxury — demand model isn't creating tradeoff!"


def test_brute_force_is_global_optimum():
    """The brute-force solution should truly be the best."""
    prob = default_instance()
    table = prob.build_truth_table()
    x_opt, f_opt = prob.brute_force_solve()

    assert f_opt == table.max()
    assert table[x_opt] == f_opt


def test_truth_table_shape():
    prob = default_instance()
    table = prob.build_truth_table()
    assert table.shape == (1024,)  # 2^10


def test_feasible_mask_consistency():
    """Feasible configs should have non-negative F (with our penalty scheme)."""
    prob = default_instance()
    table = prob.build_truth_table()
    mask = prob.feasible_mask()

    # All feasible configs should have F >= 0
    assert (table[mask] >= 0).all()

    # All infeasible configs should have very negative F
    assert (table[~mask] < 0).all()


# ---------------------------------------------------------------------------
# Instance scaling tests
# ---------------------------------------------------------------------------

def test_small_instance_sizes():
    for n in [3, 4, 5]:
        prob = small_instance(n) if n < 5 else default_instance()
        assert prob.n_features == n
        assert prob.n_bits == 2 * n
        assert prob.n_configurations == 2 ** (2 * n)

        table = prob.build_truth_table()
        assert table.shape == (2 ** (2 * n),)

        x_opt, f_opt = prob.brute_force_solve()
        assert f_opt == table.max()


def test_factory_penalties_follow_rmax_recommendation():
    """default_instance/small_instance should use R_max-derived penalties."""
    for prob in [small_instance(3), default_instance()]:
        r_max = compute_r_max(prob.features, prob.segments)
        rec = recommended_penalties_from_r_max(r_max)

        np.testing.assert_allclose(prob.penalty_invalid, rec["penalty_invalid"], rtol=1e-12)
        np.testing.assert_allclose(prob.penalty_bundle, rec["penalty_bundle"], rtol=1e-12)
        np.testing.assert_allclose(prob.penalty_luxury_cap, rec["penalty_luxury_cap"], rtol=1e-12)

        data = prob.to_dict()
        np.testing.assert_allclose(data["penalty_invalid"], rec["penalty_invalid"], rtol=1e-12)
        np.testing.assert_allclose(data["penalty_bundle"], rec["penalty_bundle"], rtol=1e-12)
        np.testing.assert_allclose(data["penalty_luxury_cap"], rec["penalty_luxury_cap"], rtol=1e-12)


# ---------------------------------------------------------------------------
# Serialization test
# ---------------------------------------------------------------------------

def test_serialization_roundtrip(tmp_path):
    prob = default_instance()
    path = tmp_path / "test_instance.json"
    prob.save_instance(path)

    import json
    with open(path) as f:
        data = json.load(f)

    prob2 = PricingProblem.from_dict(data)
    assert prob2.n_features == prob.n_features

    # Same truth table
    t1 = prob.build_truth_table()
    t2 = prob2.build_truth_table()
    np.testing.assert_array_equal(t1, t2)


if __name__ == "__main__":
    # Run all tests manually if pytest not available
    test_funcs = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in test_funcs:
        try:
            if "tmp_path" in fn.__code__.co_varnames:
                import tempfile, pathlib
                with tempfile.TemporaryDirectory() as td:
                    fn(pathlib.Path(td))
            else:
                fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
