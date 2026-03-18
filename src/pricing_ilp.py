"""Canonical toy-pricing ILP-style model contract.

This module is intentionally scoped to the repository's toy pricing family.
It does not attempt to be a general ILP compiler.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np

from src.ilp_to_xorsat_exact import (
    build_ilp_exact_reference_artifact,
    canonicalize_ilp_encoding,
)
from src.reduction import build_surrogate_artifact

if TYPE_CHECKING:
    from src.problem_generator import PricingProblem


SCHEMA_VERSION = "1.0"
SOURCE_CODE_VERSION = "stage1_toy_ilp_v1"
PAIRED_ARTIFACT_VERSION = "1.0"
REQUIRED_COMPARATIVE_SUBSET = {
    "artifact_version",
    "mode",
    "bit_order",
    "n",
    "m",
    "B",
    "v",
    "term_weights",
    "diagnostics",
    "source_metadata",
}
ALLOWED_PAIRED_TOP_LEVEL = {
    "paired_artifact_version",
    "instance_id",
    "pairing_kind",
    "source_metadata",
    "encodings",
}
ENCODING_KEYS = {"wht_truncated", "ilp_derived"}
PROVENANCE_KEYS = ("instance_id", "pricing_instance_path")


def _normalize_for_hash(value):
    """Normalize nested data into a hash-stable representation."""
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


def build_toy_pricing_ilp_model(
    prob: "PricingProblem",
    instance_id: str | None = None,
) -> dict:
    """Build a canonical toy-family ILP-style model dict from PricingProblem."""
    source = prob.to_dict()
    feature_order = [f["name"] for f in source["features"]]
    model = {
        "schema_version": SCHEMA_VERSION,
        "source_code_version": SOURCE_CODE_VERSION,
        "instance_id": instance_id or f"pricing_{prob.n_features}feat_{prob.n_bits}bit",
        "n_features": int(prob.n_features),
        "n_bits": int(prob.n_bits),
        "feature_order": feature_order,
        "bit_order": "msb_first",
        "bitstring_convention": "feature_major_msb_first",
        "codeword_mapping": {
            0: "standard",
            1: "premium",
            2: "luxury",
            3: "invalid",
        },
        "valid_codewords": [0, 1, 2],
        "invalid_codewords": [3],
        "features": source["features"],
        "segments": source["segments"],
        "bundle_constraints": source["bundle_constraints"],
        "max_luxury": int(source["max_luxury"]),
        "penalties": {
            "invalid": float(source["penalty_invalid"]),
            "bundle": float(source["penalty_bundle"]),
            "luxury_cap": float(source["penalty_luxury_cap"]),
        },
        "objective_formula_label": (
            "expected_revenue_minus_invalid_bundle_luxury_penalties"
        ),
        "objective_semantics": (
            "maximize_F_over_feature_major_msb_first_bitstrings"
        ),
    }
    hash_payload = {
        k: v for k, v in model.items() if k != "instance_hash"
    }
    model["instance_hash"] = _canonical_hash(hash_payload)
    return model


def _build_wht_encoding(
    prob: "PricingProblem",
    *,
    instance_id: str,
    k: int,
    source_metadata: dict,
) -> dict:
    return build_surrogate_artifact(
        prob,
        k=int(k),
        source_metadata=source_metadata,
        created_by="solve_pricing_pair",
    )


def _build_ilp_encoding(
    prob: "PricingProblem",
    *,
    instance_id: str,
    k: int,
    source_metadata: dict,
) -> dict:
    ilp_model = build_toy_pricing_ilp_model(prob, instance_id=instance_id)
    exact_reference = build_ilp_exact_reference_artifact(ilp_model)
    return canonicalize_ilp_encoding(
        exact_reference,
        instance_id=instance_id,
        source_metadata=source_metadata,
        k=int(k),
    )


def _validate_bit_order_alignment_descriptor(desc: dict, *, n: int) -> None:
    if not isinstance(desc, dict):
        raise ValueError("invalid bit_order_alignment descriptor for encoding 'ilp_derived'")
    kind = desc.get("kind")
    if kind not in {"identity", "permutation"}:
        raise ValueError("invalid bit_order_alignment descriptor for encoding 'ilp_derived'")
    source = desc.get("source_bit_order")
    target = desc.get("target_bit_order")
    verified = desc.get("verified")
    if not isinstance(source, list) or not isinstance(target, list):
        raise ValueError("invalid bit_order_alignment descriptor for encoding 'ilp_derived'")
    if len(source) != int(n) or len(target) != int(n):
        raise ValueError("invalid bit_order_alignment descriptor for encoding 'ilp_derived'")
    if verified is not True:
        raise ValueError("invalid bit_order_alignment descriptor for encoding 'ilp_derived'")
    if kind == "permutation":
        perm = desc.get("permutation")
        if not isinstance(perm, list) or len(perm) != int(n):
            raise ValueError("invalid bit_order_alignment descriptor for encoding 'ilp_derived'")
        try:
            perm_int = [int(v) for v in perm]
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("invalid bit_order_alignment descriptor for encoding 'ilp_derived'") from exc
        if sorted(perm_int) != list(range(int(n))):
            raise ValueError("invalid bit_order_alignment descriptor for encoding 'ilp_derived'")


def _validate_encoding_shape(enc: dict, encoding: str) -> None:
    n = int(enc["n"])
    m = int(enc["m"])
    B = np.asarray(enc["B"], dtype=np.int8)
    v = np.asarray(enc["v"], dtype=np.int8)
    term_weights = np.asarray(enc["term_weights"], dtype=np.float64)
    if B.shape != (m, n):
        raise ValueError(f"invalid shape for encoding '{encoding}': B must have shape (m, n)")
    if v.shape != (m,):
        raise ValueError(f"invalid shape for encoding '{encoding}': v must have shape (m,)")
    if term_weights.shape != (m,):
        raise ValueError(
            f"invalid shape for encoding '{encoding}': term_weights must have shape (m,)"
        )


def _validate_paired_pricing_artifact(paired: dict) -> None:
    for field in ALLOWED_PAIRED_TOP_LEVEL:
        if field not in paired:
            raise ValueError(f"paired artifact missing required top-level field: {field}")
    if set(paired.keys()) != ALLOWED_PAIRED_TOP_LEVEL:
        raise ValueError(
            "paired artifact has invalid top-level keys: expected "
            "{'paired_artifact_version', 'instance_id', 'pairing_kind', 'source_metadata', 'encodings'}"
        )

    encodings = paired.get("encodings", {})
    if set(encodings.keys()) != ENCODING_KEYS:
        raise ValueError(
            "paired artifact has invalid encoding set: expected {'wht_truncated', 'ilp_derived'}"
        )

    # Required subset checks first (as locked).
    for encoding in ("wht_truncated", "ilp_derived"):
        payload = encodings[encoding]
        for field in REQUIRED_COMPARATIVE_SUBSET:
            if field not in payload:
                raise ValueError(
                    f"encoding '{encoding}' missing required comparative field: {field}"
                )

    top_source = paired.get("source_metadata", {}) or {}
    top_instance_id = paired["instance_id"]
    wht = encodings["wht_truncated"]
    ilp = encodings["ilp_derived"]

    # Lineage/provenance checks second.
    for key in PROVENANCE_KEYS:
        wht_val = (wht.get("source_metadata") or {}).get(key)
        ilp_val = (ilp.get("source_metadata") or {}).get(key)
        top_val = top_source.get(key)
        if wht_val != ilp_val:
            raise ValueError(
                f"lineage mismatch for key '{key}': expected shared provenance across encodings"
            )
        if top_val is not None and wht_val != top_val:
            raise ValueError(
                f"provenance mismatch for key '{key}': top-level and encoding lineage disagree"
            )
    if (wht.get("source_metadata") or {}).get("instance_id") != top_instance_id:
        raise ValueError(
            "provenance mismatch for key 'instance_id': top-level and encoding lineage disagree"
        )
    if (ilp.get("source_metadata") or {}).get("instance_id") != top_instance_id:
        raise ValueError(
            "provenance mismatch for key 'instance_id': top-level and encoding lineage disagree"
        )

    # Cross-encoding consistency third.
    if int(wht["n"]) != int(ilp["n"]) or int(wht["m"]) != int(ilp["m"]):
        raise ValueError("dimension mismatch across encodings: expected shared n and m")
    if str(wht["bit_order"]) != str(ilp["bit_order"]):
        ilp_meta = ilp.get("source_metadata") or {}
        align = ilp_meta.get("bit_order_alignment")
        if align is None:
            raise ValueError("bit_order mismatch requires source_metadata.bit_order_alignment")
        _validate_bit_order_alignment_descriptor(align, n=int(wht["n"]))

    # Shape checks last.
    _validate_encoding_shape(wht, "wht_truncated")
    _validate_encoding_shape(ilp, "ilp_derived")


def solve_pricing_pair(
    prob: "PricingProblem",
    instance_id: str,
    k: int,
    source_metadata: dict,
) -> dict:
    source = dict(source_metadata or {})
    source["instance_id"] = instance_id

    wht = _build_wht_encoding(
        prob,
        instance_id=instance_id,
        k=k,
        source_metadata=dict(source),
    )
    ilp = _build_ilp_encoding(
        prob,
        instance_id=instance_id,
        k=k,
        source_metadata=dict(source),
    )
    paired = {
        "paired_artifact_version": PAIRED_ARTIFACT_VERSION,
        "instance_id": instance_id,
        "pairing_kind": "pricing_wht_vs_ilp_exact",
        "source_metadata": dict(source),
        "encodings": {
            "wht_truncated": wht,
            "ilp_derived": ilp,
        },
    }
    _validate_paired_pricing_artifact(paired)
    return paired
