"""
test_qiskit.py — Tests for Qiskit DQI circuit port.

Run: python -m pytest tests/test_qiskit.py -v
"""

import numpy as np
import sys
sys.path.insert(0, ".")

# Skip if qiskit not installed
pytest = __import__('pytest')

try:
    from src import dqi_circuit_qiskit as qiskit_mod
    from src.dqi_circuit_qiskit import (
        build_dqi_circuit,
        compare_frameworks,
        qiskit_subset_validation_report,
        qiskit_transpiled_structural_count,
        run_qiskit_statevector,
    )
    from src.dqi_state import run_dqi_subset_reference
    from src.resources import qiskit_subset_resource_report
    QISKIT_AVAILABLE = bool(getattr(qiskit_mod, "QISKIT_AVAILABLE", False))
except ImportError:
    QISKIT_AVAILABLE = False


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
def test_circuit_builds():
    """Circuit should build without errors."""
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    v = np.array([0, 1, 0], dtype=np.int8)
    alpha = np.array([0.5, 0.5, 0.5, 0.5])
    alpha /= np.linalg.norm(alpha)

    circuit = build_dqi_circuit(B, v, alpha, ell=1)
    assert circuit is not None
    assert circuit.num_qubits > 0


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
def test_statevector_normalized():
    """Qiskit statevector should be normalized."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 0], dtype=np.int8)
    alpha = np.array([0.7, 0.7])
    alpha /= np.linalg.norm(alpha)

    probs = run_qiskit_statevector(B, v, alpha, ell=1)

    np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-6)


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
def test_qiskit_subset_validation_report_exposes_normalized_probability_surface():
    """Subset validation should expose normalized distribution and structural counts."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 0], dtype=np.int8)
    alpha = np.array([1.0, 0.0], dtype=np.float64)

    out = qiskit_subset_validation_report(B=B, v=v, alpha=alpha, ell=1)

    assert set(out) >= {
        "prob_dist",
        "transpiled_depth",
        "cx_count",
        "qubit_count",
        "gate_counts",
        "syndrome_qargs",
    }
    assert out["status"] == "ok"
    assert out["prob_dist"] is not None
    np.testing.assert_allclose(np.sum(out["prob_dist"]), 1.0, rtol=1e-6, atol=1e-9)
    assert isinstance(out["gate_counts"], dict)
    assert out["syndrome_qargs"] == [2, 3]


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
def test_qiskit_subset_validation_report_handles_tiny_case_marginalization_consistently():
    """Tiny subset case should preserve the NumPy syndrome marginal after Hadamards."""
    B = np.array(
        [
            [0, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )
    v = np.array([1, 1, 0, 0], dtype=np.int8)
    ell = 1
    alpha = np.ones(ell + 1, dtype=np.float64)
    alpha /= np.linalg.norm(alpha)

    numpy_out = run_dqi_subset_reference(B=B, v=v, alpha=alpha, ell=ell)
    qiskit_out = qiskit_subset_validation_report(B=B, v=v, alpha=alpha, ell=ell)

    assert qiskit_out["status"] == "ok"
    np.testing.assert_allclose(qiskit_out["prob_dist"], numpy_out["prob_dist"], atol=1e-9)
    np.testing.assert_allclose(np.sum(qiskit_out["prob_dist"]), 1.0, atol=1e-9)


def test_compare_frameworks_is_explicitly_disabled():
    """Framework comparison should not claim unsupported distributional validation."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 0], dtype=np.int8)
    alpha = np.array([1.0, 0.0], dtype=np.float64)

    result = compare_frameworks(B, v, alpha, ell=1)
    assert result["status"] == "disabled"
    assert result["comparable"] is False
    assert "approximate Dicke" in result["reason"] or "missing decode/uncompute" in result["reason"]


def test_qiskit_structural_count_contract():
    """Structural transpiled resource extraction should return scoped counts."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 0], dtype=np.int8)
    alpha = np.array([1.0, 0.0], dtype=np.float64)

    out = qiskit_transpiled_structural_count(B=B, v=v, alpha=alpha, ell=1)
    assert out["category"] == "qiskit_transpiled_structural_count"
    assert "scope" in out
    assert out["scope"]["decode_uncompute_included"] is False
    assert out["scope"]["dicke_prep_status"] in {"approximate_structural", "unknown"}
    assert out["status"] in {"ok", "not_available", "error"}


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
def test_qiskit_subset_validation_report_runtime_error_is_structured(monkeypatch):
    """Subset validation should degrade to a structured payload on runtime errors."""
    from src import dqi_circuit_qiskit as mod

    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 0], dtype=np.int8)
    alpha = np.array([1.0, 0.0], dtype=np.float64)

    def _boom(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(mod, "qiskit_subset_syndrome_distribution_with_metadata", _boom)

    out = qiskit_subset_validation_report(B=B, v=v, alpha=alpha, ell=1)
    assert out["status"] == "runtime_error"
    assert "subset_statevector_failed" in str(out["reason"])
    assert out["prob_dist"] is None
    assert out["syndrome_qargs"] is None
    assert out["distribution_basis"] == "syndrome_register_after_final_hadamards"
    assert out["scope"]["scope"] == "structural_subset_only"
    assert out["scope"]["decode_uncompute_included"] is False


def test_subset_reference_and_resource_report_expose_explicit_scope_contract():
    """Subset-facing reports should surface explicit includes/excludes metadata."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 0], dtype=np.int8)
    alpha = np.array([1.0, 0.0], dtype=np.float64)

    numpy_out = run_dqi_subset_reference(B=B, v=v, alpha=alpha, ell=1)
    assert numpy_out["scope"] == "structural_subset_only"
    assert numpy_out["decode_uncompute_included"] is False
    assert numpy_out["includes"] == [
        "approximate_low_weight_superposition",
        "phase_kick",
        "syndrome_compute",
        "final_hadamards",
    ]
    assert numpy_out["excludes"] == ["decode_uncompute"]

    resource_out = qiskit_subset_resource_report(B=B, v=v, alpha=alpha, ell=1)
    assert isinstance(resource_out["scope"], dict)
    assert resource_out["scope"]["scope"] == "structural_subset_only"
    assert resource_out["scope"]["decode_uncompute_included"] is False
    assert resource_out["scope"]["includes"] == numpy_out["includes"]
    assert resource_out["scope"]["excludes"] == numpy_out["excludes"]
    assert resource_out["decode_uncompute_included"] is False


def test_qiskit_subset_resource_report_keeps_required_resource_keys():
    """Subset resource report should keep the reviewer-facing resource fields."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 0], dtype=np.int8)
    alpha = np.array([1.0, 0.0], dtype=np.float64)

    resource_out = qiskit_subset_resource_report(B=B, v=v, alpha=alpha, ell=1)
    assert set(resource_out) >= {
        "qubit_count",
        "transpiled_depth",
        "cx_count",
        "gate_counts",
        "scope",
    }
    assert isinstance(resource_out["gate_counts"], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
