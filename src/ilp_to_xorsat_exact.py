"""Toy-pricing exact reference encoding path.

This module is intentionally scoped to the repository's toy pricing family.
It is not a general ILP compiler.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np

from src.problem_generator import decode_bitstring
from src.reduction import (
    coefficients_to_max_xorsat,
    surrogate_objective,
    walsh_hadamard_transform,
)


SCHEMA_VERSION = "1.0"


def _normalize_for_hash(value):
    if isinstance(value, dict):
        return {str(k): _normalize_for_hash(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def _canonical_hash(payload: dict) -> str:
    normalized = _normalize_for_hash(payload)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _total_price(tiers: Iterable[int], features: list[dict]) -> float:
    total = 0.0
    for tier, feat in zip(tiers, features):
        if tier == 3:
            continue
        total += float(feat["prices"][tier])
    return float(total)


def _expected_revenue(price: float, segments: list[dict]) -> float:
    revenue = 0.0
    for seg in segments:
        if price <= float(seg["budget_cap"]):
            revenue += float(seg["weight"]) * price
    return float(revenue)


def _bundle_violations(tiers: list[int], bundle_constraints: list[dict]) -> int:
    violations = 0
    for bc in bundle_constraints:
        a = int(bc["feature_a"])
        tier_a = int(bc["tier_a"])
        b = int(bc["feature_b"])
        allowed = set(int(v) for v in bc["allowed_tiers_b"])
        if tiers[a] == tier_a and tiers[b] not in allowed:
            violations += 1
    return violations


def _evaluate_toy_objective_for_x(x: int, ilp_model: dict) -> float:
    n_features = int(ilp_model["n_features"])
    tiers = decode_bitstring(x, n_features)

    price = _total_price(tiers, ilp_model["features"])
    revenue = _expected_revenue(price, ilp_model["segments"])
    n_invalid = sum(1 for t in tiers if t == 3)
    n_bundle = _bundle_violations(tiers, ilp_model["bundle_constraints"])
    n_luxury = sum(1 for t in tiers if t == 2)
    penalty = (
        float(ilp_model["penalties"]["invalid"]) * n_invalid
        + float(ilp_model["penalties"]["bundle"]) * n_bundle
        + float(ilp_model["penalties"]["luxury_cap"]) * max(0, n_luxury - int(ilp_model["max_luxury"]))
    )
    return float(revenue - penalty)


def evaluate_toy_pricing_model_exact(ilp_model: dict) -> dict:
    """Evaluate toy pricing model exactly over the full assignment domain."""
    n_bits = int(ilp_model["n_bits"])
    n_states = 2 ** n_bits
    f_table = np.zeros(n_states, dtype=np.float64)
    for x in range(n_states):
        f_table[x] = _evaluate_toy_objective_for_x(x, ilp_model)

    f_star = float(np.max(f_table))
    optimizer_indices = np.flatnonzero(np.isclose(f_table, f_star)).astype(np.int64)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_instance_id": ilp_model["instance_id"],
        "source_model_hash": ilp_model["instance_hash"],
        "n_bits": n_bits,
        "n_states": int(n_states),
        "state_enumeration_order": "integer_ascending",
        "F_table": f_table,
        "F_table_dtype": str(f_table.dtype),
        "optimizer_indices": optimizer_indices,
        "F_star": f_star,
        "constant_offset": float(np.mean(f_table)),
        "constant_offset_included_in_F_table": True,
    }
    payload["table_hash"] = _canonical_hash(
        {
            "schema_version": payload["schema_version"],
            "source_instance_id": payload["source_instance_id"],
            "source_model_hash": payload["source_model_hash"],
            "n_bits": payload["n_bits"],
            "n_states": payload["n_states"],
            "F_table": f_table.tolist(),
        }
    )
    return payload


def _signed_coefficients_from_artifact(artifact: dict) -> np.ndarray:
    """Recover signed Fourier coefficients from parity sign convention."""
    signs = np.where(np.asarray(artifact["v"], dtype=np.int8) == 0, 1.0, -1.0)
    return np.asarray(artifact["term_weights"], dtype=np.float64) * signs


def reconstruct_exact_objective_table(artifact: dict) -> np.ndarray:
    """Reconstruct full objective table from exact full-spectrum artifact."""
    n_bits = int(artifact["n_bits"])
    n_states = 2 ** n_bits
    table = np.zeros(n_states, dtype=np.float64)
    for x in range(n_states):
        table[x] = float(artifact["constant_offset"]) + surrogate_objective(
            x,
            np.asarray(artifact["B"], dtype=np.int8),
            np.asarray(artifact["v"], dtype=np.int8),
            np.asarray(artifact["term_weights"], dtype=np.float64),
            n_bits,
        )
    return table


def build_ilp_exact_reference_artifact(ilp_model: dict) -> dict:
    """Build exact full-spectrum parity artifact for the toy pricing family.

    This path is scoped to the repository's toy pricing model and should not be
    interpreted as a general ILP-to-XORSAT compiler.
    """
    table = evaluate_toy_pricing_model_exact(ilp_model)
    f_table = np.asarray(table["F_table"], dtype=np.float64)
    n_bits = int(table["n_bits"])
    n_states = int(table["n_states"])

    f_hat = walsh_hadamard_transform(f_table)
    indices = np.arange(1, n_states, dtype=np.int64)
    coeffs = f_hat[1:]
    B, v, term_weights = coefficients_to_max_xorsat(indices, coeffs, n_bits=n_bits)

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "encoding_type": "ilp_exact_reference",
        "encoding_label": "toy_pricing_ilp_full_spectrum_reference",
        "reference_path": "model_table_full_wht",
        "spectrum_source": "full_exact_wht_of_F_table",
        "exact_full_spectrum": True,
        "exact_reconstructable": True,
        "source_instance_id": ilp_model["instance_id"],
        "source_model_hash": ilp_model["instance_hash"],
        "source_table_hash": table["table_hash"],
        "objective_semantics": ilp_model["objective_semantics"],
        "bit_order": ilp_model["bit_order"],
        "bitstring_convention": ilp_model["bitstring_convention"],
        "feature_order": ilp_model["feature_order"],
        "term_indexing_convention": "integer_mask_msb_first_excluding_constant",
        "parity_semantics": "row_i_applies_sign_factor_minus_one_pow_dot_Bi_x",
        "weight_sign_convention": "v=0_positive_coeff_v=1_negative_coeff",
        "n_bits": n_bits,
        "m_terms": int(len(indices)),
        "constant_offset": float(f_hat[0]),
        "B": B,
        "v": v,
        "term_weights": term_weights,
        "indices": indices,
        "coefficients": coeffs,
    }

    hash_payload = {
        "schema_version": artifact["schema_version"],
        "encoding_type": artifact["encoding_type"],
        "reference_path": artifact["reference_path"],
        "source_instance_id": artifact["source_instance_id"],
        "source_model_hash": artifact["source_model_hash"],
        "n_bits": artifact["n_bits"],
        "m_terms": artifact["m_terms"],
        "constant_offset": artifact["constant_offset"],
        "indices": artifact["indices"].tolist(),
        "coefficients": np.asarray(artifact["coefficients"], dtype=np.float64).tolist(),
    }
    artifact["artifact_hash"] = _canonical_hash(hash_payload)
    return artifact


def validate_exact_reconstruction(
    ilp_model: dict,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> dict:
    """Validate full-domain exact reconstruction from exact reference artifact."""
    table = evaluate_toy_pricing_model_exact(ilp_model)
    artifact = build_ilp_exact_reference_artifact(ilp_model)
    reconstructed = reconstruct_exact_objective_table(artifact)
    source = np.asarray(table["F_table"], dtype=np.float64)
    abs_err = np.abs(source - reconstructed)
    return {
        "ok": bool(np.allclose(source, reconstructed, atol=atol, rtol=rtol)),
        "domain_size": int(source.size),
        "max_abs_err": float(np.max(abs_err)),
        "mean_abs_err": float(np.mean(abs_err)),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def canonicalize_ilp_encoding(
    ilp_payload: dict,
    *,
    instance_id: str,
    source_metadata: dict,
    k: int | None = None,
) -> dict:
    """Map ILP exact-reference payload into Stage-A canonical comparative fields.

    Preserves original ILP payload fields as encoding-specific extras while exposing
    the shared comparative subset (`artifact_version/mode/bit_order/n/m/B/v/...`).
    """
    required = (
        "schema_version",
        "bit_order",
        "n_bits",
        "m_terms",
        "B",
        "v",
        "term_weights",
        "indices",
        "coefficients",
    )
    for key in required:
        if key not in ilp_payload:
            raise KeyError(f"ILP payload missing required key: {key}")

    B_full = np.asarray(ilp_payload["B"], dtype=np.int8)
    v_full = np.asarray(ilp_payload["v"], dtype=np.int8)
    w_full = np.asarray(ilp_payload["term_weights"], dtype=np.float64)
    idx_full = np.asarray(ilp_payload["indices"], dtype=np.int64)
    coeff_full = np.asarray(ilp_payload["coefficients"], dtype=np.float64)

    if k is None:
        take = int(len(idx_full))
        sel = np.arange(take, dtype=np.int64)
    else:
        if int(k) < 1:
            raise ValueError("k must be >= 1 for canonicalize_ilp_encoding")
        take = int(min(int(k), len(idx_full)))
        # Stable top-|coeff| selection to mirror existing WHT truncation semantics.
        sel = np.argsort(np.abs(coeff_full))[-take:][::-1].astype(np.int64)

    B_sel = B_full[sel]
    v_sel = v_full[sel]
    w_sel = w_full[sel]
    idx_sel = idx_full[sel]
    coeff_sel = coeff_full[sel]

    source = dict(source_metadata or {})
    source["instance_id"] = instance_id
    if "source_instance_id" in ilp_payload:
        source.setdefault("ilp_source_instance_id", ilp_payload["source_instance_id"])
    if "source_model_hash" in ilp_payload:
        source.setdefault("source_model_hash", ilp_payload["source_model_hash"])

    diagnostics = {
        "exact_full_spectrum_source": bool(ilp_payload.get("exact_full_spectrum", True)),
        "selected_term_fraction": float(len(sel) / max(1, len(idx_full))),
        "reference_path": ilp_payload.get("reference_path"),
        "source_table_hash": ilp_payload.get("source_table_hash"),
    }

    out = dict(ilp_payload)
    out["artifact_version"] = str(ilp_payload["schema_version"])
    out["mode"] = "ilp_derived_exact_reference"
    out["n"] = int(ilp_payload["n_bits"])
    out["m"] = int(len(sel))
    out["B"] = B_sel
    out["v"] = v_sel
    out["term_weights"] = w_sel
    out["indices"] = idx_sel
    out["coefficients"] = coeff_sel
    out["diagnostics"] = diagnostics
    out["source_metadata"] = source
    out["full_m_terms"] = int(ilp_payload["m_terms"])
    return out


def _jsonify_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonify_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify_value(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonify_value(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def paired_pricing_artifact_to_dict(paired: dict) -> dict:
    """Convert paired pricing artifact to JSON-friendly payload."""
    return _jsonify_value(dict(paired))


def paired_pricing_artifact_from_dict(payload: dict) -> dict:
    """Load paired pricing artifact and restore ndarray-backed canonical arrays."""
    out = dict(payload)
    encodings = dict(out.get("encodings", {}))
    restored: dict[str, dict] = {}
    for key, enc in encodings.items():
        data = dict(enc)
        if "B" in data:
            data["B"] = np.asarray(data["B"], dtype=np.int8)
        if "v" in data:
            data["v"] = np.asarray(data["v"], dtype=np.int8)
        if "term_weights" in data:
            data["term_weights"] = np.asarray(data["term_weights"], dtype=np.float64)
        restored[key] = data
    out["encodings"] = restored
    return out
