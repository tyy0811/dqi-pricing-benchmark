"""Tests for Stage 1 toy-pricing ILP exact reference path."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from src.problem_generator import small_instance


def test_build_toy_pricing_ilp_model_schema():
    from src.pricing_ilp import build_toy_pricing_ilp_model

    prob = small_instance(3)
    model = build_toy_pricing_ilp_model(prob, instance_id="pricing_3feat_6bit")

    required = {
        "schema_version",
        "instance_id",
        "instance_hash",
        "n_features",
        "n_bits",
        "feature_order",
        "bit_order",
        "bitstring_convention",
        "codeword_mapping",
        "valid_codewords",
        "invalid_codewords",
        "segments",
        "bundle_constraints",
        "max_luxury",
        "penalties",
        "objective_formula_label",
        "objective_semantics",
    }
    assert required.issubset(model.keys())
    assert model["bit_order"] == "msb_first"
    assert model["bitstring_convention"] == "feature_major_msb_first"


def test_evaluate_toy_pricing_model_exact_returns_full_domain_fields():
    from src.ilp_to_xorsat_exact import evaluate_toy_pricing_model_exact
    from src.pricing_ilp import build_toy_pricing_ilp_model

    prob = small_instance(3)
    model = build_toy_pricing_ilp_model(prob, instance_id="pricing_3feat_6bit")
    out = evaluate_toy_pricing_model_exact(model)

    assert out["n_states"] == 2 ** prob.n_bits
    assert len(out["F_table"]) == out["n_states"]
    assert out["state_enumeration_order"] == "integer_ascending"
    assert out["constant_offset_included_in_F_table"] is True
    assert out["F_star"] == max(out["F_table"])


def test_build_ilp_exact_reference_artifact_schema_and_exact_flag():
    from src.ilp_to_xorsat_exact import build_ilp_exact_reference_artifact
    from src.pricing_ilp import build_toy_pricing_ilp_model

    prob = small_instance(3)
    model = build_toy_pricing_ilp_model(prob, instance_id="pricing_3feat_6bit")
    artifact = build_ilp_exact_reference_artifact(model)

    assert artifact["encoding_type"] == "ilp_exact_reference"
    assert artifact["reference_path"] == "model_table_full_wht"
    assert artifact["exact_full_spectrum"] is True
    assert artifact["exact_reconstructable"] is True
    assert artifact["m_terms"] == (2 ** artifact["n_bits"]) - 1
    assert artifact["B"].shape == (artifact["m_terms"], artifact["n_bits"])
    assert artifact["v"].shape == (artifact["m_terms"],)
    assert artifact["term_weights"].shape == (artifact["m_terms"],)


def test_validate_exact_reconstruction_full_domain():
    from src.ilp_to_xorsat_exact import validate_exact_reconstruction
    from src.pricing_ilp import build_toy_pricing_ilp_model

    prob = small_instance(3)
    model = build_toy_pricing_ilp_model(prob, instance_id="pricing_3feat_6bit")
    report = validate_exact_reconstruction(model, atol=1e-9, rtol=1e-9)

    assert report["ok"] is True
    assert report["domain_size"] == 2 ** model["n_bits"]
    assert report["max_abs_err"] <= 1e-9


def test_exact_artifact_hash_is_deterministic_for_same_instance():
    from src.ilp_to_xorsat_exact import build_ilp_exact_reference_artifact
    from src.pricing_ilp import build_toy_pricing_ilp_model

    prob = small_instance(3)
    model = build_toy_pricing_ilp_model(prob, instance_id="pricing_3feat_6bit")

    artifact_1 = build_ilp_exact_reference_artifact(model)
    artifact_2 = build_ilp_exact_reference_artifact(model)
    assert artifact_1["artifact_hash"] == artifact_2["artifact_hash"]
