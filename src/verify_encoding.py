"""Stage E paired-encoding validation helpers."""

from __future__ import annotations

from typing import Any

PAIR_VALIDATION_FLAG = "expects_paired_encoding_validation"

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


def should_validate_paired_encoding(row: dict[str, Any]) -> bool:
    return bool(row.get(PAIR_VALIDATION_FLAG, False))


def _fail(message: str) -> tuple[bool, str, str]:
    return False, "encoding_validation_failed", message


def _validate_bit_order_compatibility(wht: dict[str, Any], ilp: dict[str, Any]) -> tuple[bool, str | None]:
    wht_order = wht.get("bit_order")
    ilp_order = ilp.get("bit_order")
    if wht_order == ilp_order:
        return True, None

    align = (ilp.get("source_metadata") or {}).get("bit_order_alignment")
    if not isinstance(align, dict):
        return False, "bit_order mismatch without alignment"
    if align.get("verified") is not True:
        return False, "bit_order alignment is not verified"
    return True, None


def validate_paired_encoding_row(
    row: dict[str, Any],
    paired_artifact: dict[str, Any] | None,
) -> tuple[bool, str | None, str | None]:
    """Validate a Stage A paired artifact for Stage E row joinability."""
    if not should_validate_paired_encoding(row):
        return True, None, None

    if not isinstance(paired_artifact, dict):
        return _fail("missing paired artifact")

    encodings = paired_artifact.get("encodings")
    if not isinstance(encodings, dict):
        return _fail("missing encodings block")
    if set(encodings.keys()) != {"wht_truncated", "ilp_derived"}:
        return _fail("encodings must contain exactly wht_truncated and ilp_derived")

    wht = encodings["wht_truncated"]
    ilp = encodings["ilp_derived"]
    if not isinstance(wht, dict) or not isinstance(ilp, dict):
        return _fail("encoding payloads must be objects")

    for name, payload in (("wht_truncated", wht), ("ilp_derived", ilp)):
        missing = sorted(REQUIRED_COMPARATIVE_SUBSET.difference(payload.keys()))
        if missing:
            return _fail(f"{name} missing required fields: {missing}")

    row_instance = row.get("instance_id")
    top_instance = paired_artifact.get("instance_id")
    if row_instance is not None and top_instance is not None and str(row_instance) != str(top_instance):
        return _fail("instance_id mismatch between row and paired artifact")

    for key in ("instance_id", "pricing_instance_path"):
        wv = (wht.get("source_metadata") or {}).get(key)
        iv = (ilp.get("source_metadata") or {}).get(key)
        if wv is not None and iv is not None and wv != iv:
            return _fail(f"lineage mismatch for key '{key}'")

    if int(wht.get("n")) != int(ilp.get("n")):
        return _fail("n mismatch between encodings")
    if int(wht.get("m")) != int(ilp.get("m")):
        return _fail("m mismatch between encodings")

    bit_ok, bit_reason = _validate_bit_order_compatibility(wht, ilp)
    if not bit_ok:
        return _fail(bit_reason or "bit_order incompatibility")

    return True, None, None
