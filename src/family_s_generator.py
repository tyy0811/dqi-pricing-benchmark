"""Family S synthetic structure-sweep artifact generator.

Family S is a compact deterministic synthetic family for exercising structure
metrics (row/column weights, rank behavior, collision tendency, ambiguity,
and an approximate distance proxy) on canonical (B, v) artifacts.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.structure_metrics import compute_structure_metrics


DEFAULT_FAMILY_S_DIR = Path("data/instances/family_s")
MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = "1"
FAMILY_S_SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "1.0"


_FAMILY_S_SPECS: list[dict[str, Any]] = [
    {
        "instance_id": "S_dense_low_collision",
        "description": "Dense balanced matrix with low intended collision tendency.",
        "n_bits": 8,
        "m_rows": 10,
        "seed": 1101,
        "ell_default": 2,
        "generation_params": {
            "target_row_weight": 5,
            "row_weight_jitter": 1,
            "column_skew": 0.05,
            "duplicate_rows": 0,
            "dependent_rows": 0,
            "enforce_full_rank": True,
            "target_rank_max": None,
            "unique_base_rows": True,
        },
        "intended_structure_tags": [
            "dense",
            "low_collision",
            "balanced_columns",
            "near_full_rank",
        ],
    },
    {
        "instance_id": "S_dense_high_collision",
        "description": "Dense skewed matrix with duplicated/dependent rows for high ambiguity.",
        "n_bits": 8,
        "m_rows": 10,
        "seed": 1102,
        "ell_default": 2,
        "generation_params": {
            "target_row_weight": 5,
            "row_weight_jitter": 1,
            "column_skew": 0.75,
            "duplicate_rows": 3,
            "dependent_rows": 2,
            "enforce_full_rank": False,
            "target_rank_max": 6,
            "unique_base_rows": False,
        },
        "intended_structure_tags": [
            "dense",
            "high_collision",
            "high_ambiguity",
            "rank_reduced",
            "column_skewed",
        ],
    },
    {
        "instance_id": "S_sparse_low_collision",
        "description": "Sparse near-uniform matrix with low intended collision tendency.",
        "n_bits": 10,
        "m_rows": 12,
        "seed": 1103,
        "ell_default": 2,
        "generation_params": {
            "target_row_weight": 2,
            "row_weight_jitter": 0,
            "column_skew": 0.0,
            "duplicate_rows": 0,
            "dependent_rows": 0,
            "enforce_full_rank": True,
            "target_rank_max": None,
            "unique_base_rows": True,
        },
        "intended_structure_tags": [
            "sparse",
            "low_collision",
            "near_uniform_columns",
            "near_full_rank",
        ],
    },
    {
        "instance_id": "S_sparse_high_collision",
        "description": "Sparse matrix with duplicates and dependencies to force ambiguity.",
        "n_bits": 10,
        "m_rows": 12,
        "seed": 1104,
        "ell_default": 2,
        "generation_params": {
            "target_row_weight": 2,
            "row_weight_jitter": 0,
            "column_skew": 0.65,
            "duplicate_rows": 4,
            "dependent_rows": 3,
            "enforce_full_rank": False,
            "target_rank_max": 6,
            "unique_base_rows": False,
        },
        "intended_structure_tags": [
            "sparse",
            "high_collision",
            "high_ambiguity",
            "rank_reduced",
            "column_skewed",
        ],
    },
    {
        "instance_id": "S_balanced_rank_full",
        "description": "Balanced medium-density matrix with full-rank target.",
        "n_bits": 8,
        "m_rows": 8,
        "seed": 1105,
        "ell_default": 2,
        "generation_params": {
            "target_row_weight": 3,
            "row_weight_jitter": 1,
            "column_skew": 0.1,
            "duplicate_rows": 0,
            "dependent_rows": 0,
            "enforce_full_rank": True,
            "target_rank_max": None,
            "unique_base_rows": True,
        },
        "intended_structure_tags": [
            "balanced",
            "full_rank",
            "medium_density",
            "low_to_medium_collision",
        ],
    },
    {
        "instance_id": "S_rank_deficient",
        "description": "Matrix with intentional rank deficiency via dependent rows.",
        "n_bits": 8,
        "m_rows": 10,
        "seed": 1106,
        "ell_default": 2,
        "generation_params": {
            "target_row_weight": 3,
            "row_weight_jitter": 1,
            "column_skew": 0.2,
            "duplicate_rows": 2,
            "dependent_rows": 4,
            "enforce_full_rank": False,
            "target_rank_max": 5,
            "unique_base_rows": False,
        },
        "intended_structure_tags": [
            "rank_deficient",
            "high_ambiguity",
            "medium_density",
        ],
    },
    {
        "instance_id": "S_column_skewed_dense",
        "description": "Dense regime with strong column skew and moderate collisions.",
        "n_bits": 10,
        "m_rows": 12,
        "seed": 1107,
        "ell_default": 2,
        "generation_params": {
            "target_row_weight": 5,
            "row_weight_jitter": 1,
            "column_skew": 0.85,
            "duplicate_rows": 1,
            "dependent_rows": 2,
            "enforce_full_rank": False,
            "target_rank_max": 8,
            "unique_base_rows": False,
        },
        "intended_structure_tags": [
            "dense",
            "column_skewed",
            "medium_to_high_collision",
        ],
    },
    {
        "instance_id": "S_near_uniform_sparse",
        "description": "Sparse regime with near-uniform columns and moderate rank.",
        "n_bits": 10,
        "m_rows": 10,
        "seed": 1108,
        "ell_default": 2,
        "generation_params": {
            "target_row_weight": 2,
            "row_weight_jitter": 1,
            "column_skew": 0.02,
            "duplicate_rows": 1,
            "dependent_rows": 1,
            "enforce_full_rank": True,
            "target_rank_max": None,
            "unique_base_rows": True,
        },
        "intended_structure_tags": [
            "sparse",
            "near_uniform_columns",
            "medium_collision",
            "near_full_rank",
        ],
    },
]


def build_family_s_specs() -> list[dict[str, Any]]:
    """Return canonical Family S spec list."""
    return copy.deepcopy(_FAMILY_S_SPECS)


def _gf2_rank(matrix: np.ndarray) -> int:
    a = (np.asarray(matrix, dtype=np.uint8) & 1).copy()
    rows, cols = a.shape
    rank = 0
    pivot_row = 0

    for col in range(cols):
        pivot = None
        for r in range(pivot_row, rows):
            if a[r, col]:
                pivot = r
                break
        if pivot is None:
            continue

        if pivot != pivot_row:
            a[[pivot_row, pivot]] = a[[pivot, pivot_row]]

        for r in range(rows):
            if r != pivot_row and a[r, col]:
                a[r, :] ^= a[pivot_row, :]

        rank += 1
        pivot_row += 1
        if pivot_row == rows:
            break

    return int(rank)


def _column_probabilities(n_bits: int, skew: float) -> np.ndarray:
    skew = float(max(0.0, skew))
    idx = np.arange(n_bits, dtype=np.float64)
    weights = np.exp(-skew * idx)
    weights /= np.sum(weights)
    return weights


def _sample_row(
    *,
    n_bits: int,
    row_weight: int,
    probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    row = np.zeros(n_bits, dtype=np.int8)
    cols = rng.choice(n_bits, size=row_weight, replace=False, p=probs)
    row[cols] = 1
    return row


def _target_row_weight(
    *,
    n_bits: int,
    base_weight: int,
    jitter: int,
    rng: np.random.Generator,
) -> int:
    if jitter <= 0:
        return int(max(1, min(n_bits, base_weight)))
    delta = int(rng.integers(-jitter, jitter + 1))
    return int(max(1, min(n_bits, base_weight + delta)))


def _build_candidate_matrix(spec: dict[str, Any], seed: int) -> np.ndarray:
    params = spec["generation_params"]
    n_bits = int(spec["n_bits"])
    m_rows = int(spec["m_rows"])

    base_weight = int(params["target_row_weight"])
    jitter = int(params["row_weight_jitter"])
    duplicate_rows = int(params["duplicate_rows"])
    dependent_rows = int(params["dependent_rows"])
    unique_base_rows = bool(params["unique_base_rows"])
    base_rows_target = int(max(1, m_rows - duplicate_rows - dependent_rows))

    rng = np.random.default_rng(seed)
    probs = _column_probabilities(n_bits, float(params["column_skew"]))

    rows: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    tries = 0
    max_tries = 4000
    while len(rows) < base_rows_target:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                f"Failed to generate base rows for {spec['instance_id']} under constraints."
            )
        w = _target_row_weight(n_bits=n_bits, base_weight=base_weight, jitter=jitter, rng=rng)
        row = _sample_row(n_bits=n_bits, row_weight=w, probs=probs, rng=rng)
        key = tuple(int(x) for x in row)
        if unique_base_rows and key in seen:
            continue
        rows.append(row)
        seen.add(key)

    for _ in range(dependent_rows):
        if len(rows) < 2:
            dep = rows[0].copy()
        else:
            i = int(rng.integers(0, len(rows)))
            j = int(rng.integers(0, len(rows)))
            dep = (rows[i] ^ rows[j]).astype(np.int8)
            if int(np.sum(dep)) == 0:
                dep = rows[i].copy()
        rows.append(dep)

    for _ in range(duplicate_rows):
        idx = int(rng.integers(0, len(rows)))
        rows.append(rows[idx].copy())

    while len(rows) < m_rows:
        w = _target_row_weight(n_bits=n_bits, base_weight=base_weight, jitter=jitter, rng=rng)
        rows.append(_sample_row(n_bits=n_bits, row_weight=w, probs=probs, rng=rng))
    rows = rows[:m_rows]

    order = rng.permutation(m_rows)
    B = np.asarray([rows[int(i)] for i in order], dtype=np.int8)

    target_rank_max = params.get("target_rank_max")
    if target_rank_max is not None:
        target_rank_max = int(target_rank_max)
        adjust_iter = 0
        while _gf2_rank(B) > target_rank_max and adjust_iter < m_rows * 4:
            dst = m_rows - 1 - (adjust_iter % m_rows)
            a = adjust_iter % m_rows
            b = (a + 1) % m_rows
            B[dst] = (B[a] ^ B[b]).astype(np.int8)
            if int(np.sum(B[dst])) == 0:
                B[dst] = B[a].copy()
            adjust_iter += 1

    return B


def _generate_matrix_with_rank_policy(spec: dict[str, Any]) -> np.ndarray:
    n_bits = int(spec["n_bits"])
    m_rows = int(spec["m_rows"])
    params = spec["generation_params"]
    enforce_full_rank = bool(params.get("enforce_full_rank", False))
    max_rank = int(min(m_rows, n_bits))

    seed = int(spec["seed"])
    best = _build_candidate_matrix(spec, seed)
    if not enforce_full_rank:
        return best

    for attempt in range(64):
        candidate = _build_candidate_matrix(spec, seed + attempt)
        if _gf2_rank(candidate) == max_rank:
            return candidate
        if _gf2_rank(candidate) > _gf2_rank(best):
            best = candidate
    return best


def generate_family_s_instance(spec: dict[str, Any]) -> dict[str, Any]:
    """Generate one deterministic Family S artifact dictionary from spec."""
    n_bits = int(spec["n_bits"])
    m_rows = int(spec["m_rows"])
    seed = int(spec["seed"])
    ell_default = int(spec.get("ell_default", 2))

    B = _generate_matrix_with_rank_policy(spec)
    rng = np.random.default_rng(seed + 913)
    v = rng.integers(0, 2, size=m_rows, dtype=np.int8)
    structure_metrics = compute_structure_metrics(B, ell=ell_default)

    return {
        "schema_version": FAMILY_S_SCHEMA_VERSION,
        "family": "S",
        "instance_id": spec["instance_id"],
        "description": spec["description"],
        "source_type": "synthetic_structured_bv",
        "objective_type": "max_xorsat_synthetic",
        "seed": seed,
        "n_bits": n_bits,
        "m_rows": m_rows,
        "ell_default": ell_default,
        "ell_allowed": [1, 2],
        "B": np.asarray(B, dtype=np.int8).tolist(),
        "v": np.asarray(v, dtype=np.int8).tolist(),
        "generation_params": copy.deepcopy(spec["generation_params"]),
        "intended_structure_tags": list(spec["intended_structure_tags"]),
        "structure_metrics": structure_metrics,
    }


def _manifest_from_artifacts(
    *,
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    order = [a["instance_id"] for a in artifacts]
    entries = []
    for a in artifacts:
        entries.append(
            {
                "instance_id": a["instance_id"],
                "family": "S",
                "path": f"{a['instance_id']}.json",
                "source_type": a["source_type"],
                "objective_type": a["objective_type"],
                "n_bits": int(a["n_bits"]),
                "m_rows": int(a["m_rows"]),
                "seed": int(a["seed"]),
                "ell_default": int(a["ell_default"]),
                "generation_params": copy.deepcopy(a["generation_params"]),
                "intended_structure_tags": list(a["intended_structure_tags"]),
            }
        )

    return {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": FAMILY_S_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "family": "S",
        "description": "Canonical synthetic structure-sweep family for benchmark diagnostics.",
        "canonical_instance_order": order,
        "instances": entries,
    }


def write_family_s_artifacts(
    *,
    base_dir: str | Path = DEFAULT_FAMILY_S_DIR,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Generate canonical Family S artifacts and write manifest + instance files."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    artifacts = [generate_family_s_instance(spec) for spec in build_family_s_specs()]
    manifest = _manifest_from_artifacts(artifacts=artifacts)

    for artifact in artifacts:
        path = base / f"{artifact['instance_id']}.json"
        if path.exists() and not overwrite:
            continue
        with open(path, "w") as f:
            json.dump(artifact, f, indent=2, sort_keys=True)

    with open(base / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return manifest


def regenerate_family_s_from_manifest(
    *,
    base_dir: str | Path = DEFAULT_FAMILY_S_DIR,
    write_artifacts: bool = False,
) -> list[dict[str, Any]]:
    """Regenerate Family S instances from manifest parameters deterministically."""
    base = Path(base_dir)
    manifest_path = base / MANIFEST_FILENAME
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    out: list[dict[str, Any]] = []
    for entry in manifest["instances"]:
        spec = {
            "instance_id": entry["instance_id"],
            "description": f"Regenerated artifact for {entry['instance_id']}",
            "n_bits": int(entry["n_bits"]),
            "m_rows": int(entry["m_rows"]),
            "seed": int(entry["seed"]),
            "ell_default": int(entry.get("ell_default", 2)),
            "generation_params": copy.deepcopy(entry["generation_params"]),
            "intended_structure_tags": list(entry.get("intended_structure_tags", [])),
        }
        artifact = generate_family_s_instance(spec)
        artifact["description"] = spec["description"]
        out.append(artifact)

        if write_artifacts:
            with open(base / entry["path"], "w") as f:
                json.dump(artifact, f, indent=2, sort_keys=True)

    return out


if __name__ == "__main__":
    write_family_s_artifacts()
