"""Tests for resource estimation modules."""

import json
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from src.dicke_circuit import dicke_resources, dicke_superposition_upper_bound
from src.resources import (
    dqi_resource_estimate,
    resource_estimate_from_artifact,
    resource_report,
    summarize_resource_scope,
)


def test_dicke_resources_trivial_case():
    """t=0 returns zero CNOTs (trivial state |00...0>)."""
    result = dicke_resources(m=10, t=0)
    assert result["cnot_est"] == 0
    assert result["single_qubit_rot_est"] == 0
    assert result["depth_est"] == 0
    assert result["method"] == "bartschi_eidenbenz_scs"
    assert "reference" in result


def test_dicke_resources_symmetry_t_vs_m_minus_t():
    """t and m-t should have same CNOT count due to t(m-t) symmetry."""
    m = 10
    result_t3 = dicke_resources(m=m, t=3)
    result_t7 = dicke_resources(m=m, t=7)
    assert result_t3["cnot_est"] == result_t7["cnot_est"]
    assert result_t3["cnot_est"] == 3 * 7  # = 21


def test_dicke_resources_scaling_pattern():
    """CNOT count follows t(m-t) pattern."""
    m = 10
    for t in range(1, m):
        result = dicke_resources(m=m, t=t)
        expected_cnot = t * (m - t)
        assert result["cnot_est"] == expected_cnot, f"Failed for t={t}"


def test_dicke_superposition_upper_bound_keys():
    """Returns expected keys."""
    result = dicke_superposition_upper_bound(m=10, ell=3)
    assert "cnot_est" in result
    assert "single_qubit_rot_est" in result
    assert "depth_est" in result
    assert "method" in result
    assert "per_weight_breakdown" in result
    assert result["method"] == "bartschi_eidenbenz_scs_upper_bound"


def test_dicke_superposition_upper_bound_aggregation():
    """Upper bound is exact sum of per-t costs."""
    m = 10
    ell = 3
    result = dicke_superposition_upper_bound(m=m, ell=ell)
    expected_cnot = sum(dicke_resources(m, t)["cnot_est"] for t in range(ell + 1))
    expected_single = sum(dicke_resources(m, t)["single_qubit_rot_est"] for t in range(ell + 1))
    assert result["cnot_est"] == expected_cnot
    assert result["single_qubit_rot_est"] == expected_single
    assert len(result["per_weight_breakdown"]) == ell + 1


def test_dqi_resource_estimate_schema():
    """All required fields are present."""
    B = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)
    v = np.array([0, 1], dtype=np.int8)
    ell = 2

    result = dqi_resource_estimate(B, v, ell, instance_id="test_instance")

    # Instance metadata
    assert result["instance_id"] == "test_instance"
    assert result["n"] == 3
    assert result["m"] == 2
    assert result["ell"] == 2

    # Dicke estimates
    assert "dicke_method" in result
    assert "dicke_cnot_est" in result
    assert "dicke_single_qubit_rot_est" in result
    assert "dicke_depth_est" in result

    # Other stages
    assert "syndrome_cnot_count" in result
    assert "phase_z_count" in result
    assert "final_h_count" in result

    # Aggregates
    assert "total_gate_est" in result
    assert "total_depth_est" in result
    assert "total_gate_est_with_decode" in result
    assert "total_depth_est_with_decode" in result

    # Qubit breakdown
    assert "n_problem_qubits" in result
    assert "m_error_qubits" in result
    assert "n_syndrome_qubits" in result
    assert "ancilla_qubits_est" in result
    assert "total_qubits_est" in result
    assert "total_qubits_est_with_decode" in result

    # Register assumptions
    assert "register_assumptions" in result

    # Stage tracking
    assert "decode_uncompute" in result["included_stages"]
    assert result["excluded_stages"] == []
    assert "decode_uncompute_model" in result
    assert "decode_uncompute_status" in result
    assert "decode_uncompute_gate_est" in result
    assert "decode_uncompute_depth_est" in result
    assert "decode_uncompute_ancilla_est" in result

    # Provenance
    assert "assumptions" in result
    assert "reference" in result


def test_dqi_resource_estimate_gate_counts():
    """Gate counts are computed correctly from B and v."""
    B = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
    ], dtype=np.int8)
    v = np.array([1, 0, 1], dtype=np.int8)
    n = 3

    result = dqi_resource_estimate(B, v, ell=2)

    assert result["syndrome_cnot_count"] == 5  # sum(B) = 5
    assert result["phase_z_count"] == 2  # sum(v) = 2
    assert result["final_h_count"] == n  # = 3


def test_dqi_resource_estimate_qubit_counts():
    """Qubit counts are computed correctly."""
    B = np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=np.int8)
    v = np.array([0, 1], dtype=np.int8)

    result = dqi_resource_estimate(B, v, ell=2)

    assert result["n_problem_qubits"] == 4
    assert result["m_error_qubits"] == 2
    assert result["n_syndrome_qubits"] == 4
    assert result["ancilla_qubits_est"] == 0
    assert result["total_qubits_est"] == 2 + 4 + 0


def test_dqi_resource_estimate_included_excluded_stages():
    """Correct stages are listed as included/excluded."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 0], dtype=np.int8)

    result = dqi_resource_estimate(B, v, ell=1)

    assert "dicke_prep" in result["included_stages"]
    assert "phase_kick" in result["included_stages"]
    assert "syndrome_compute" in result["included_stages"]
    assert "final_hadamard" in result["included_stages"]
    assert "decode_uncompute" in result["included_stages"]
    assert "decode_uncompute" not in result["excluded_stages"]


def test_decode_inclusive_totals_dominate_decode_exclusive_totals():
    """Decode-inclusive totals should be >= decode-excluding totals."""
    B = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)
    v = np.array([1, 0], dtype=np.int8)

    result = dqi_resource_estimate(B, v, ell=2)

    assert result["total_gate_est_with_decode"] >= result["total_gate_est"]
    assert result["total_depth_est_with_decode"] >= result["total_depth_est"]
    assert result["total_qubits_est_with_decode"] >= result["total_qubits_est"]


def test_bp1_decode_model_reports_analytic_status_and_fields():
    """BP1 decode resource path should be explicit analytic estimate."""
    B = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.int8)
    v = np.array([0, 1, 1], dtype=np.int8)
    result = dqi_resource_estimate(
        B,
        v,
        ell=1,
        decode_uncompute_model="bp1_classical_oracle_estimate",
    )

    assert result["decode_uncompute_model"] == "bp1_classical_oracle_estimate"
    assert result["decode_uncompute_status"] == "analytic_estimate"
    assert result["decode_uncompute_gate_est"] >= 0
    assert result["decode_uncompute_depth_est"] >= 0
    assert result["decode_uncompute_ancilla_est"] >= 0


def test_resource_groups_and_confidence_labels_present():
    """Grouped resource sections should distinguish implemented vs estimated totals."""
    B = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)
    v = np.array([0, 1], dtype=np.int8)
    result = dqi_resource_estimate(B, v, ell=2)

    assert "implemented_resources" in result
    assert "estimated_resources" in result
    assert "total_resources" in result
    assert "analytic_estimate" in result
    assert "qiskit_transpiled_structural_count" in result
    assert "resource_category_note" in result

    impl = result["implemented_resources"]
    for key in ["dicke_prep", "phase_kick", "syndrome_compute", "hadamards"]:
        assert key in impl
        assert "confidence" in impl[key]

    est = result["estimated_resources"]
    assert "decode_uncompute" in est
    assert "coherent_weight_register_overhead" in est
    assert "confidence" in est["decode_uncompute"]
    assert "confidence" in est["coherent_weight_register_overhead"]


def test_resource_report_has_additive_validation_scope_and_summary():
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 1], dtype=np.int8)

    result = dqi_resource_estimate(B, v, ell=1)
    assert (
        result["resource_validation_scope"]
        == "analytic_full_pipeline_plus_qiskit_structural_subset"
    )

    summary = summarize_resource_scope(result)
    assert summary["analytic_scope"] == "full_pipeline_estimate"
    assert summary["qiskit_scope"] == "structural_subset_only"
    assert "decode-inclusive analytic estimate" in summary["note"].lower()

    impl = result["implemented_resources"]
    est = result["estimated_resources"]
    totals = result["total_resources"]
    for key in [
        "total_without_decode",
        "total_with_decode",
        "total_with_decode_and_weight_register_est",
    ]:
        assert key in totals
        assert "gate_est" in totals[key]
        assert "depth_est" in totals[key]
        assert "qubits_est" in totals[key]

    allowed_labels = {
        "exact_formula",
        "analytic_upper_bound",
        "implementation_proxy",
        "not_included",
    }
    for section in [impl["dicke_prep"], impl["phase_kick"], impl["syndrome_compute"], impl["hadamards"],
                    est["decode_uncompute"], est["coherent_weight_register_overhead"]]:
        assert section["confidence"] in allowed_labels

    qiskit_struct = result["qiskit_transpiled_structural_count"]
    assert qiskit_struct["category"] == "qiskit_transpiled_structural_count"
    assert qiskit_struct["status"] in {"ok", "not_available", "error", "not_included"}


def test_stage3_resource_aliases_and_cost_fields_present():
    """Stage 3 join-facing aliases and cost fields should be present."""
    B = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)
    v = np.array([0, 1], dtype=np.int8)
    result = dqi_resource_estimate(B, v, ell=2)

    assert "analytic_estimates" in result
    assert "total_estimated_resources" in result
    assert "prefix_gate_est" in result
    assert "prefix_depth_est" in result
    assert "decode_model_gate_est" in result
    assert "decode_model_depth_est" in result
    assert "rank" in result
    assert "rank_deficiency" in result
    assert "decodable_syndrome_fraction" in result


def test_weight_register_overhead_fields_are_explicit():
    """Weight-register overhead should be explicit even when approximate."""
    B = np.array([[1, 1], [0, 1], [1, 0]], dtype=np.int8)
    v = np.array([1, 0, 1], dtype=np.int8)
    result = dqi_resource_estimate(B, v, ell=3)

    assert "weight_register_qubits" in result
    assert "weight_register_overhead_status" in result
    assert "coherent_weight_register_gate_est" in result
    assert result["weight_register_qubits"] >= 0
    assert result["coherent_weight_register_gate_est"] >= 0
    assert result["weight_register_overhead_status"] in {
        "analytic_upper_bound",
        "implementation_proxy",
        "not_included",
    }


def test_structure_metrics_required_fields_present():
    """Resource report should expose required structure-sensitive metrics."""
    B = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.int8)
    v = np.array([1, 0, 1], dtype=np.int8)
    result = dqi_resource_estimate(B, v, ell=2)

    required = {
        "row_weight_mean",
        "row_weight_max",
        "col_weight_mean",
        "col_weight_max",
        "syndrome_density",
        "number_of_unique_syndromes_decodable_at_ell",
        "decoder_collision_rate",
        "estimated_lookup_table_size",
    }
    assert required.issubset(result.keys())


def test_decode_cost_drivers_reflect_collision_rate():
    """Collision rate should produce nonzero guard overhead when collisions exist."""
    B_low_collision = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 0], dtype=np.int8)
    low = dqi_resource_estimate(B_low_collision, v, ell=1)
    low_drv = low["decode_cost_drivers"]
    assert low_drv["decoder_collision_rate"] == 0.0
    assert low_drv["collision_guard_gate_est"] == 0

    B_with_collision = np.array([[1, 0], [1, 0]], dtype=np.int8)
    high = dqi_resource_estimate(B_with_collision, v, ell=1)
    high_drv = high["decode_cost_drivers"]
    assert high_drv["decoder_collision_rate"] > 0.0
    assert high_drv["collision_guard_gate_est"] > 0


def test_dqi_resource_estimate_rejects_nonbinary_v():
    """ValueError raised for non-binary v."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v_bad = np.array([0, 2], dtype=np.int8)

    with pytest.raises(ValueError, match="binary"):
        dqi_resource_estimate(B, v_bad, ell=1)


def test_dqi_resource_estimate_rejects_shape_mismatch():
    """ValueError raised for B/v dimension mismatch."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v_wrong = np.array([0, 1, 1], dtype=np.int8)

    with pytest.raises(ValueError, match="rows"):
        dqi_resource_estimate(B, v_wrong, ell=1)


def test_resource_report_metadata_presence():
    """Required metadata fields are present."""
    B = np.array([[1, 0], [0, 1]], dtype=np.int8)
    v = np.array([0, 1], dtype=np.int8)

    result = dqi_resource_estimate(B, v, ell=1)

    assert "included_stages" in result
    assert "excluded_stages" in result
    assert "assumptions" in result
    assert "reference" in result
    assert isinstance(result["assumptions"], list)
    assert len(result["assumptions"]) > 0


def test_resource_estimate_from_artifact():
    """resource_estimate_from_artifact extracts B, v and computes resources."""
    artifact = {
        "artifact_version": "1.0",
        "n": 4,
        "m": 2,
        "k_selected": 2,
        "B": [[1, 0, 1, 0], [0, 1, 0, 1]],
        "v": [0, 1],
        "source_metadata": {
            "instance_id": "test_artifact",
            "repo_commit": "abc123",
        },
    }

    result = resource_estimate_from_artifact(artifact, ell=2)

    assert result["instance_id"] == "test_artifact"
    assert result["n"] == 4
    assert result["m"] == 2
    assert result["k_selected"] == 2
    assert result["artifact_version"] == "1.0"
    assert result["repo_commit"] == "abc123"


def test_resource_report_round_trip_with_artifact(tmp_path):
    """Load artifact from file, compute resources, verify schema."""
    artifact = {
        "artifact_version": "1.0",
        "n": 4,
        "m": 2,
        "k_selected": 2,
        "B": [[1, 0, 1, 0], [0, 1, 0, 1]],
        "v": [0, 1],
        "source_metadata": {"instance_id": "roundtrip_test"},
    }

    artifact_path = tmp_path / "test_artifact.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f)

    result = resource_report(str(artifact_path), ell=2)

    assert result["instance_id"] == "roundtrip_test"
    assert result["n"] == 4
    assert result["m"] == 2
    assert "total_gate_est" in result
    assert "total_qubits_est" in result
