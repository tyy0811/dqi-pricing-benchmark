"""Stage E faithfulness normalization utilities."""

from __future__ import annotations

from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def can_compute_faithfulness(row: dict[str, Any]) -> tuple[bool, str | None]:
    """Return whether row has sufficient verified provenance for computation."""
    if bool(row.get("has_verified_provenance", False)):
        return True, None
    return False, "insufficient_provenance"


def compute_surrogate_best_gap(row: dict[str, Any]) -> float | None:
    """Compute best-top-G sampled minus best sampled objective gap."""
    best_sampled = _to_float(row.get("best_sampled_F"))
    best_top_g = _to_float(row.get("best_top_G_sampled_F"))
    if best_sampled is None or best_top_g is None:
        return None
    return float(best_top_g - best_sampled)


def normalize_faithfulness_fields(row: dict[str, Any]) -> dict[str, float | None]:
    """Return Stage E faithfulness fields with safe, provenance-aware defaults."""
    out = {
        "decision_distortion": _to_float(row.get("decision_distortion")),
        "spearman_rho_F_G": _to_float(row.get("spearman_rho_F_G")),
        "retained_energy_eta": _to_float(row.get("retained_energy_eta")),
        "best_top_G_sampled_F": _to_float(row.get("best_top_G_sampled_F")),
        "surrogate_best_vs_true_best_sampled_gap": _to_float(
            row.get("surrogate_best_vs_true_best_sampled_gap")
        ),
    }

    if out["surrogate_best_vs_true_best_sampled_gap"] is None:
        ok, _reason = can_compute_faithfulness(row)
        if ok:
            out["surrogate_best_vs_true_best_sampled_gap"] = compute_surrogate_best_gap(row)

    return out
