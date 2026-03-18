"""Load and validate fixed coherent tiny-instance artifacts for Stage 2A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.dqi_state import COHERENT_STATE_SIZE_CAP, joint_state_size_coherent, run_dqi_statevector
from src.weights import heuristic_alpha_weights, paper_alpha_weights, uniform_weights


DEFAULT_BASE_DIR = Path("data/instances/coherent")
CANONICAL_INSTANCE_ORDER = ["C1", "C2", "C3"]
COHERENT_LIMITS = {
    "max_n_bits": 6,
    "max_m_rows": 8,
    "max_ell": 2,
}


def _resolve_under_base(base_dir: Path, relative_path: str) -> Path:
    path = (base_dir / relative_path).resolve()
    base = base_dir.resolve()
    if base not in path.parents and path != base:
        raise ValueError(f"path escapes coherent base dir: {relative_path!r}")
    return path


def _truth_table_from_weighted_encoding(
    *,
    B: np.ndarray,
    v: np.ndarray,
    n_bits: int,
    row_weights: np.ndarray | None,
    constant_offset: float | None,
) -> np.ndarray:
    B = np.asarray(B, dtype=np.int8)
    v = np.asarray(v, dtype=np.int8)
    m = B.shape[0]
    if row_weights is None:
        w = np.ones(m, dtype=np.float64)
    else:
        w = np.asarray(row_weights, dtype=np.float64)
    offset = float(constant_offset) if constant_offset is not None else 0.0

    out = np.zeros(2 ** n_bits, dtype=np.float64)
    for x in range(2 ** n_bits):
        x_bits = np.array([(x >> (n_bits - 1 - j)) & 1 for j in range(n_bits)], dtype=np.int8)
        parities = (B @ x_bits) % 2
        signs = (-1) ** (v + parities)
        out[x] = offset + float(np.sum(w * signs))
    return out


def coherent_supported_for_instance(
    *,
    n_bits: int,
    m_rows: int,
    ell_allowed: Iterable[int],
) -> bool:
    """Stage 2A coherent support rule for tiny coherent-family artifacts."""
    if n_bits > COHERENT_LIMITS["max_n_bits"]:
        return False
    if m_rows > COHERENT_LIMITS["max_m_rows"]:
        return False
    if len(list(ell_allowed)) == 0:
        return False
    if max(int(e) for e in ell_allowed) > COHERENT_LIMITS["max_ell"]:
        return False
    return True


def _check_binary_matrix(name: str, arr: np.ndarray) -> None:
    unique = np.unique(arr)
    if not set(unique.tolist()).issubset({0, 1}):
        raise ValueError(f"{name} must be binary; got values {unique.tolist()}")


def _validate_optimizer_set(
    optimizer_set: list[int],
    *,
    f_table: np.ndarray,
    f_star: float,
    n_bits: int,
    strict: bool,
) -> None:
    if optimizer_set != sorted(optimizer_set):
        raise ValueError("optimizer_set must be sorted ascending")
    if len(set(optimizer_set)) != len(optimizer_set):
        raise ValueError("optimizer_set must contain unique indices")
    max_index = 2 ** n_bits
    for idx in optimizer_set:
        if idx < 0 or idx >= max_index:
            raise ValueError("optimizer_set contains out-of-range index")
        if not np.isclose(f_table[idx], f_star):
            raise ValueError("optimizer_set index does not achieve F_star")

    if strict:
        expected = np.flatnonzero(np.isclose(f_table, f_star)).astype(np.int64).tolist()
        if optimizer_set != expected:
            raise ValueError("optimizer_set must match full argmax set in strict mode")


def validate_coherent_manifest(
    manifest: dict,
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> dict:
    """Validate manifest schema, ordering, and file references."""
    required_top = {
        "manifest_version",
        "schema_version",
        "family",
        "description",
        "canonical_instance_order",
        "instances",
    }
    if not required_top.issubset(manifest.keys()):
        missing = sorted(required_top.difference(manifest.keys()))
        raise ValueError(f"Manifest missing required keys: {missing}")

    order = list(manifest["canonical_instance_order"])
    if order != CANONICAL_INSTANCE_ORDER:
        raise ValueError(
            f"canonical_instance_order must be {CANONICAL_INSTANCE_ORDER}, got {order}"
        )
    entries = list(manifest["instances"])
    entry_ids = [entry.get("instance_id") for entry in entries]
    if entry_ids != order:
        raise ValueError("manifest instances must appear in canonical_instance_order")

    base = Path(base_dir)
    for entry in entries:
        for key in ("instance_id", "family", "json_path", "npz_path", "n_bits", "m_rows", "ell_default", "coherent_supported"):
            if key not in entry:
                raise ValueError(f"manifest entry missing required field: {key}")
        json_path = _resolve_under_base(base, entry["json_path"])
        npz_path = _resolve_under_base(base, entry["npz_path"])
        if not json_path.exists():
            raise ValueError(f"manifest json_path missing: {entry['json_path']}")
        if not npz_path.exists():
            raise ValueError(f"manifest npz_path missing: {entry['npz_path']}")

    return {"ok": True}


def validate_coherent_instance(
    meta: dict,
    arrays: dict,
    *,
    strict: bool = True,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> dict:
    """Validate one coherent tiny-family instance metadata+arrays payload."""
    required_meta = {
        "schema_version",
        "instance_id",
        "family",
        "source_type",
        "description",
        "seed",
        "n_bits",
        "m_rows",
        "ell_default",
        "ell_allowed",
        "encoding_backend",
        "objective_type",
        "objective_relation",
        "coherent_supported",
        "pricing_source_path",
        "exact_encoding_source_path",
        "wht_exact_available",
        "F_star",
        "G_star",
        "optimizer_set",
        "npz_is_canonical",
    }
    if not required_meta.issubset(meta.keys()):
        missing = sorted(required_meta.difference(meta.keys()))
        raise ValueError(f"instance metadata missing required keys: {missing}")

    if meta.get("npz_is_canonical") is not True:
        raise ValueError("npz_is_canonical must be true")

    n_bits = int(meta["n_bits"])
    m_rows = int(meta["m_rows"])
    ell_default = int(meta["ell_default"])
    ell_allowed = [int(e) for e in meta["ell_allowed"]]

    if ell_default not in ell_allowed:
        raise ValueError("ell_default must be contained in ell_allowed")
    if not coherent_supported_for_instance(n_bits=n_bits, m_rows=m_rows, ell_allowed=ell_allowed):
        raise ValueError("instance violates coherent tiny-family size limits")
    if bool(meta["coherent_supported"]) is not True:
        raise ValueError("coherent_supported must be true for C-family instances")

    required_arrays = {"B", "v", "truth_table_F", "truth_table_G"}
    if not required_arrays.issubset(arrays.keys()):
        missing = sorted(required_arrays.difference(arrays.keys()))
        raise ValueError(f"NPZ missing required arrays: {missing}")

    B = np.asarray(arrays["B"], dtype=np.int8)
    v = np.asarray(arrays["v"], dtype=np.int8)
    f_table = np.asarray(arrays["truth_table_F"], dtype=np.float64)
    g_table = np.asarray(arrays["truth_table_G"], dtype=np.float64)

    if B.shape != (m_rows, n_bits):
        raise ValueError(f"B shape mismatch: expected {(m_rows, n_bits)}, got {B.shape}")
    if v.shape != (m_rows,):
        raise ValueError(f"v shape mismatch: expected {(m_rows,)}, got {v.shape}")
    _check_binary_matrix("B", B)
    _check_binary_matrix("v", v)

    expected_len = 2 ** n_bits
    if len(f_table) != expected_len or len(g_table) != expected_len:
        raise ValueError("truth table length mismatch with n_bits")

    if meta.get("truth_tables_mirrored_in_json", False):
        if "truth_table_F" in meta and not np.allclose(np.asarray(meta["truth_table_F"], dtype=np.float64), f_table):
            raise ValueError("JSON truth_table_F does not match NPZ canonical array")
        if "truth_table_G" in meta and not np.allclose(np.asarray(meta["truth_table_G"], dtype=np.float64), g_table):
            raise ValueError("JSON truth_table_G does not match NPZ canonical array")

    f_star = float(meta["F_star"])
    g_star = float(meta["G_star"])
    if not np.isclose(f_star, float(np.max(f_table))):
        raise ValueError("F_star mismatch with truth_table_F maximum")
    if not np.isclose(g_star, float(np.max(g_table))):
        raise ValueError("G_star mismatch with truth_table_G maximum")

    optimizer_set = [int(x) for x in meta["optimizer_set"]]
    _validate_optimizer_set(
        optimizer_set,
        f_table=f_table,
        f_star=f_star,
        n_bits=n_bits,
        strict=strict,
    )

    relation = meta["objective_relation"]
    if relation == "identical":
        if not np.allclose(f_table, g_table, atol=1e-9, rtol=1e-9):
            raise ValueError("objective_relation=identical requires truth_table_F == truth_table_G")
    elif relation == "affine":
        if "affine_a" not in meta or "affine_b" not in meta:
            raise ValueError("objective_relation=affine requires affine_a and affine_b")
        a = float(meta["affine_a"])
        b = float(meta["affine_b"])
        if not np.allclose(f_table, a * g_table + b, atol=1e-9, rtol=1e-9):
            raise ValueError("affine objective relation failed: F != a*G + b")
    elif relation == "weighted_xorsat_exact":
        if "row_weights" not in arrays or "constant_offset" not in arrays:
            raise ValueError("weighted_xorsat_exact requires row_weights and constant_offset arrays")
        row_weights = np.asarray(arrays["row_weights"], dtype=np.float64)
        if row_weights.shape != (m_rows,):
            raise ValueError("row_weights shape mismatch")
        constant_offset = float(np.asarray(arrays["constant_offset"]).reshape(()))
        reconstructed = _truth_table_from_weighted_encoding(
            B=B,
            v=v,
            n_bits=n_bits,
            row_weights=row_weights,
            constant_offset=constant_offset,
        )
        if not np.allclose(f_table, reconstructed, atol=1e-9, rtol=1e-9):
            raise ValueError("weighted_xorsat_exact reconstruction failed for truth_table_F")
        if not np.allclose(g_table, reconstructed, atol=1e-9, rtol=1e-9):
            raise ValueError("weighted_xorsat_exact reconstruction failed for truth_table_G")
    else:
        raise ValueError(f"Unsupported objective_relation={relation!r}")

    if meta["objective_type"] == "max_xorsat_exact":
        reconstructed_unweighted = _truth_table_from_weighted_encoding(
            B=B,
            v=v,
            n_bits=n_bits,
            row_weights=None,
            constant_offset=0.0,
        )
        if not np.allclose(g_table, reconstructed_unweighted, atol=1e-9, rtol=1e-9):
            raise ValueError("max_xorsat_exact instance has inconsistent truth_table_G")
        if meta["instance_id"] in {"C1", "C3"} and not np.allclose(f_table, g_table, atol=1e-9, rtol=1e-9):
            raise ValueError("C1/C3 must satisfy truth_table_F == truth_table_G")

    if meta["instance_id"] == "C2":
        base = Path(base_dir)
        pricing_path = meta.get("pricing_source_path")
        exact_path = meta.get("exact_encoding_source_path")
        if not pricing_path or not exact_path:
            raise ValueError("C2 requires non-null pricing_source_path and exact_encoding_source_path")
        resolved_pricing = _resolve_under_base(base, str(pricing_path))
        resolved_exact = _resolve_under_base(base, str(exact_path))
        if not resolved_pricing.exists() or not resolved_exact.exists():
            raise ValueError("C2 linkage paths must resolve to existing files")
        with open(resolved_pricing, "r") as f:
            json.load(f)
        with open(resolved_exact, "r") as f:
            json.load(f)

    if meta["instance_id"] == "C3":
        sm = meta.get("structure_metrics", {})
        for key in ("rank_B_gf2", "decoder_collision_rate", "ambiguous_syndrome_count"):
            if key not in sm:
                raise ValueError(f"C3 missing structure_metrics.{key}")

    return {
        "ok": True,
        "instance_id": meta["instance_id"],
        "n_bits": n_bits,
        "m_rows": m_rows,
        "ell_default": ell_default,
    }


def load_coherent_manifest(base_dir: str | Path = DEFAULT_BASE_DIR) -> dict:
    """Load coherent manifest JSON."""
    base = Path(base_dir)
    manifest_path = base / "coherent_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    validate_coherent_manifest(manifest, base_dir=base)
    return manifest


def load_coherent_instance(
    instance_id: str,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> dict:
    """Load one coherent instance record (metadata + canonical NPZ arrays)."""
    base = Path(base_dir)
    manifest = load_coherent_manifest(base)
    entries = {entry["instance_id"]: entry for entry in manifest["instances"]}
    if instance_id not in entries:
        raise KeyError(f"Unknown coherent instance_id={instance_id!r}")
    entry = entries[instance_id]

    json_path = _resolve_under_base(base, entry["json_path"])
    npz_path = _resolve_under_base(base, entry["npz_path"])

    with open(json_path, "r") as f:
        meta = json.load(f)
    arrays = dict(np.load(npz_path, allow_pickle=False))
    validate_coherent_instance(meta, arrays, strict=True, base_dir=base)
    return {
        "meta": meta,
        "arrays": arrays,
        "manifest_entry": entry,
        "json_path": json_path,
        "npz_path": npz_path,
    }


def load_coherent_family(base_dir: str | Path = DEFAULT_BASE_DIR) -> list[dict]:
    """Load all coherent instances in canonical manifest order."""
    base = Path(base_dir)
    manifest = load_coherent_manifest(base)
    order = manifest["canonical_instance_order"]
    return [load_coherent_instance(instance_id, base) for instance_id in order]


def _coherent_parts(instance: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "meta" in instance and "arrays" in instance:
        return dict(instance["meta"]), dict(instance["arrays"])
    raise ValueError("instance must be a coherent record with keys {'meta','arrays'}")


def _alpha_vector_for_mode(*, alpha_mode: str, B: np.ndarray, v: np.ndarray, ell: int) -> np.ndarray:
    if alpha_mode == "uniform":
        return uniform_weights(ell)
    if alpha_mode == "paper":
        return paper_alpha_weights(ell)
    if alpha_mode == "heuristic":
        return heuristic_alpha_weights(B, v, ell)
    raise ValueError(f"unsupported alpha_mode={alpha_mode!r}")


def _active_truth_tables(
    *,
    meta: dict[str, Any],
    arrays: dict[str, Any],
    m_active: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_bits = int(meta["n_bits"])
    B_full = np.asarray(arrays["B"], dtype=np.int8)
    v_full = np.asarray(arrays["v"], dtype=np.int8)
    B = B_full[:m_active]
    v = v_full[:m_active]
    relation = str(meta.get("objective_relation", "identical"))
    if relation == "weighted_xorsat_exact":
        row_weights = np.asarray(arrays.get("row_weights"), dtype=np.float64)[:m_active]
        constant_offset = float(np.asarray(arrays.get("constant_offset")).reshape(()))
        table = _truth_table_from_weighted_encoding(
            B=B,
            v=v,
            n_bits=n_bits,
            row_weights=row_weights,
            constant_offset=constant_offset,
        )
        return table, table.copy()

    table = _truth_table_from_weighted_encoding(
        B=B,
        v=v,
        n_bits=n_bits,
        row_weights=None,
        constant_offset=0.0,
    )
    return table, table.copy()


def _best_top_g_sampled_f(*, samples: np.ndarray, f_values: np.ndarray, g_values: np.ndarray) -> float | None:
    if len(samples) == 0:
        return None
    order = sorted(
        range(len(samples)),
        key=lambda i: (
            -float(g_values[i]),
            -float(f_values[i]),
            int(samples[i]),
        ),
    )
    return float(f_values[order[0]])


def supports_tiny_coherent(
    instance: dict[str, Any],
    *,
    ell: int,
    m_active: int,
    cap_limit: int = COHERENT_STATE_SIZE_CAP,
) -> tuple[bool, str]:
    """Return whether coherent exact execution is supported with machine-readable reason."""
    meta, arrays = _coherent_parts(instance)
    n_bits = int(meta["n_bits"])
    m_rows = int(meta["m_rows"])

    if not bool(meta.get("coherent_supported", False)):
        return False, "instance_not_marked_coherent_supported"
    if int(ell) < 0:
        return False, "ell_negative"
    if int(ell) > int(COHERENT_LIMITS["max_ell"]):
        return False, "ell_out_of_bounds"
    if int(m_active) < 1 or int(m_active) > m_rows:
        return False, "m_active_out_of_bounds"
    if int(m_active) > int(COHERENT_LIMITS["max_m_rows"]):
        return False, "m_active_out_of_bounds"
    if n_bits > int(COHERENT_LIMITS["max_n_bits"]):
        return False, "n_bits_out_of_bounds"

    # Defensive check: manifest m_rows and payload matrix row count must align.
    if np.asarray(arrays["B"]).shape[0] < int(m_active):
        return False, "m_active_out_of_bounds"

    joint_size = int(joint_state_size_coherent(m=int(m_active), n=n_bits, ell=int(ell)))
    if joint_size > int(cap_limit):
        return False, "coherent_exact_limit_exceeded"
    return True, "supported"


def evaluate_coherent_exact(
    instance: dict[str, Any],
    *,
    alpha_mode: str,
    ell: int,
    m_active: int,
    trial_seed: int | None,
    n_dqi_samples: int = 256,
) -> dict[str, Any]:
    """Evaluate coherent exact tiny case with oracle decoder."""
    supported, reason = supports_tiny_coherent(instance, ell=ell, m_active=m_active)
    if not supported:
        raise ValueError(reason)

    meta, arrays = _coherent_parts(instance)
    n_bits = int(meta["n_bits"])
    B = np.asarray(arrays["B"], dtype=np.int8)[:m_active]
    v = np.asarray(arrays["v"], dtype=np.int8)[:m_active]
    f_table, g_table = _active_truth_tables(meta=meta, arrays=arrays, m_active=int(m_active))
    alpha = _alpha_vector_for_mode(alpha_mode=alpha_mode, B=B, v=v, ell=int(ell))

    samples, probabilities, success_prob = run_dqi_statevector(
        B=B,
        v=v,
        alpha=alpha,
        ell=int(ell),
        n_samples=int(n_dqi_samples),
        seed=trial_seed,
        decoder_mode="oracle",
        execution_mode="coherent",
    )
    samples = np.asarray(samples, dtype=np.int64)
    f_values = np.asarray(f_table[samples], dtype=np.float64)
    g_values = np.asarray(g_table[samples], dtype=np.float64)
    best_f = float(np.max(f_values)) if len(f_values) > 0 else None
    best_top_g_sampled_f = _best_top_g_sampled_f(samples=samples, f_values=f_values, g_values=g_values)
    f_star = float(meta.get("F_star")) if meta.get("F_star") is not None else None
    top1_regret = None
    if best_f is not None and f_star is not None:
        top1_regret = float(f_star - best_f)
    return {
        "best_F": best_f,
        "best_sampled_F": best_f,
        "best_top_G_sampled_F": best_top_g_sampled_f,
        "F_star": f_star,
        "top1_regret": top1_regret,
        "success_probability": float(success_prob),
        "postselection_success": float(success_prob),
        "candidate_probabilities": [float(x) for x in np.asarray(probabilities, dtype=np.float64)],
        "n_bits": n_bits,
    }


def evaluate_mixture_exact(
    instance: dict[str, Any],
    *,
    alpha_mode: str,
    ell: int,
    m_active: int,
    trial_seed: int | None,
    n_dqi_samples: int = 256,
) -> dict[str, Any]:
    """Evaluate mixture exact tiny case with oracle decoder."""
    meta, arrays = _coherent_parts(instance)
    n_bits = int(meta["n_bits"])
    m_rows = int(meta["m_rows"])
    if int(m_active) < 1 or int(m_active) > m_rows:
        raise ValueError("m_active_out_of_bounds")
    if int(ell) < 0:
        raise ValueError("ell_negative")

    B = np.asarray(arrays["B"], dtype=np.int8)[:m_active]
    v = np.asarray(arrays["v"], dtype=np.int8)[:m_active]
    f_table, g_table = _active_truth_tables(meta=meta, arrays=arrays, m_active=int(m_active))
    alpha = _alpha_vector_for_mode(alpha_mode=alpha_mode, B=B, v=v, ell=int(ell))

    samples, probabilities, success_prob = run_dqi_statevector(
        B=B,
        v=v,
        alpha=alpha,
        ell=int(ell),
        n_samples=int(n_dqi_samples),
        seed=trial_seed,
        decoder_mode="oracle",
        execution_mode="mixture",
    )
    samples = np.asarray(samples, dtype=np.int64)
    f_values = np.asarray(f_table[samples], dtype=np.float64)
    g_values = np.asarray(g_table[samples], dtype=np.float64)
    best_f = float(np.max(f_values)) if len(f_values) > 0 else None
    best_top_g_sampled_f = _best_top_g_sampled_f(samples=samples, f_values=f_values, g_values=g_values)
    f_star = float(meta.get("F_star")) if meta.get("F_star") is not None else None
    top1_regret = None
    if best_f is not None and f_star is not None:
        top1_regret = float(f_star - best_f)
    return {
        "best_F": best_f,
        "best_sampled_F": best_f,
        "best_top_G_sampled_F": best_top_g_sampled_f,
        "F_star": f_star,
        "top1_regret": top1_regret,
        "success_probability": float(success_prob),
        "postselection_success": float(success_prob),
        "candidate_probabilities": [float(x) for x in np.asarray(probabilities, dtype=np.float64)],
        "n_bits": n_bits,
    }
