"""Loader and validator for canonical Family S synthetic structure-sweep artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from src.family_s_generator import DEFAULT_FAMILY_S_DIR, MANIFEST_FILENAME


REQUIRED_MANIFEST_TOP_LEVEL = {
    "manifest_version",
    "schema_version",
    "generator_version",
    "family",
    "description",
    "canonical_instance_order",
    "instances",
}

REQUIRED_MANIFEST_ENTRY = {
    "instance_id",
    "family",
    "path",
    "source_type",
    "objective_type",
    "n_bits",
    "m_rows",
    "seed",
    "ell_default",
    "generation_params",
    "intended_structure_tags",
}

REQUIRED_INSTANCE_FIELDS = {
    "schema_version",
    "family",
    "instance_id",
    "description",
    "source_type",
    "objective_type",
    "seed",
    "n_bits",
    "m_rows",
    "ell_default",
    "ell_allowed",
    "B",
    "v",
    "generation_params",
    "intended_structure_tags",
    "structure_metrics",
}


def _resolve_under_base(base_dir: Path, relative_path: str) -> Path:
    path = (base_dir / relative_path).resolve()
    base = base_dir.resolve()
    if path != base and base not in path.parents:
        raise ValueError(f"path escapes family_s base dir: {relative_path!r}")
    return path


def _validate_binary(name: str, arr: np.ndarray) -> None:
    values = set(np.unique(arr).tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"{name} must be binary, got values={sorted(values)}")


def validate_family_s_instance(payload: dict) -> dict:
    """Validate one Family S artifact payload."""
    missing = sorted(REQUIRED_INSTANCE_FIELDS.difference(payload.keys()))
    if missing:
        raise ValueError(f"Family S instance missing required fields: {missing}")

    if payload["family"] != "S":
        raise ValueError("Family S instance must have family='S'")

    n_bits = int(payload["n_bits"])
    m_rows = int(payload["m_rows"])
    ell_default = int(payload["ell_default"])
    ell_allowed = [int(v) for v in payload["ell_allowed"]]
    if ell_default not in ell_allowed:
        raise ValueError("ell_default must be included in ell_allowed")

    B = np.asarray(payload["B"], dtype=np.int8)
    v = np.asarray(payload["v"], dtype=np.int8)
    if B.shape != (m_rows, n_bits):
        raise ValueError(f"B shape mismatch, expected {(m_rows, n_bits)}, got {B.shape}")
    if v.shape != (m_rows,):
        raise ValueError(f"v shape mismatch, expected {(m_rows,)}, got {v.shape}")
    _validate_binary("B", B)
    _validate_binary("v", v)

    if not isinstance(payload["generation_params"], dict):
        raise ValueError("generation_params must be a dictionary")
    if not isinstance(payload["intended_structure_tags"], list):
        raise ValueError("intended_structure_tags must be a list")

    return {
        "ok": True,
        "instance_id": str(payload["instance_id"]),
        "n_bits": n_bits,
        "m_rows": m_rows,
    }


def validate_family_s_manifest(
    manifest: dict,
    *,
    base_dir: str | Path = DEFAULT_FAMILY_S_DIR,
) -> dict:
    """Validate Family S manifest schema, paths, and deterministic order."""
    missing_top = sorted(REQUIRED_MANIFEST_TOP_LEVEL.difference(manifest.keys()))
    if missing_top:
        raise ValueError(f"Family S manifest missing required fields: {missing_top}")

    if manifest["family"] != "S":
        raise ValueError("Family S manifest must have family='S'")

    entries = list(manifest["instances"])
    order = list(manifest["canonical_instance_order"])
    if [e.get("instance_id") for e in entries] != order:
        raise ValueError("manifest instance order must exactly match canonical_instance_order")

    base = Path(base_dir)
    for entry in entries:
        missing = sorted(REQUIRED_MANIFEST_ENTRY.difference(entry.keys()))
        if missing:
            raise ValueError(f"Family S manifest entry missing fields: {missing}")

        instance_path = _resolve_under_base(base, str(entry["path"]))
        if not instance_path.exists():
            raise ValueError(f"Family S instance path missing: {entry['path']}")
        with open(instance_path, "r") as f:
            payload = json.load(f)

        validate_family_s_instance(payload)

        # Manifest metadata must agree with artifact metadata.
        for key in (
            "instance_id",
            "family",
            "source_type",
            "objective_type",
            "n_bits",
            "m_rows",
            "seed",
            "ell_default",
        ):
            if str(payload[key]) != str(entry[key]):
                raise ValueError(
                    f"Manifest/artifact mismatch for {entry['instance_id']} field {key}: "
                    f"manifest={entry[key]!r}, artifact={payload[key]!r}"
                )

    return {"ok": True, "count": len(entries)}


def load_family_s_manifest(base_dir: str | Path = DEFAULT_FAMILY_S_DIR) -> dict:
    """Load and validate Family S manifest."""
    base = Path(base_dir)
    path = base / MANIFEST_FILENAME
    with open(path, "r") as f:
        manifest = json.load(f)
    validate_family_s_manifest(manifest, base_dir=base)
    return manifest


def load_family_s_instance(
    instance_id: str,
    *,
    base_dir: str | Path = DEFAULT_FAMILY_S_DIR,
) -> dict:
    """Load one Family S instance artifact by id."""
    manifest = load_family_s_manifest(base_dir=base_dir)
    entries = {entry["instance_id"]: entry for entry in manifest["instances"]}
    if instance_id not in entries:
        raise KeyError(f"Unknown Family S instance_id={instance_id!r}")

    base = Path(base_dir)
    path = _resolve_under_base(base, entries[instance_id]["path"])
    with open(path, "r") as f:
        payload = json.load(f)
    validate_family_s_instance(payload)
    return payload


def load_family_s_family(
    *,
    base_dir: str | Path = DEFAULT_FAMILY_S_DIR,
    instance_ids: Iterable[str] | None = None,
) -> list[dict]:
    """Load Family S instances in deterministic order."""
    manifest = load_family_s_manifest(base_dir=base_dir)
    order = list(manifest["canonical_instance_order"])
    if instance_ids is None:
        target = order
    else:
        requested = [str(i) for i in instance_ids]
        unknown = sorted(set(requested).difference(order))
        if unknown:
            raise KeyError(f"Unknown Family S instance_id values: {unknown}")
        # Keep canonical ordering even when a subset is requested.
        target = [instance_id for instance_id in order if instance_id in set(requested)]

    return [load_family_s_instance(instance_id, base_dir=base_dir) for instance_id in target]
