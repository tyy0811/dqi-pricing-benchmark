"""
reduction.py — Walsh-Hadamard transform and max-XORSAT surrogate construction.

Implements Fourier truncation of the pricing objective F(x) to produce
a sparse parity-style surrogate G(x) in max-XORSAT format (B, v).

Bit ordering convention: MSB-first throughout. For n-bit strings:
  - Bit 0 is the most significant (leftmost)
  - Integer x corresponds to bits [x>>(n-1), ..., x>>0] & 1
  - Fourier index S uses the same convention for parity mask

Reference: Jordan et al. (2024) arXiv:2408.08292, Section 1.2
"""

from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
from scipy.stats import spearmanr
from typing import TYPE_CHECKING
from src.structure_metrics import compute_structure_metrics

if TYPE_CHECKING:
    from src.problem_generator import PricingProblem


def walsh_hadamard_transform(f: np.ndarray) -> np.ndarray:
    """Compute the Walsh-Hadamard transform of a truth table.

    Uses the fast Walsh-Hadamard transform (FWHT) in O(n * 2^n).

    Args:
        f: 1D array of length 2^n containing f(x) values.

    Returns:
        f_hat: Fourier coefficients, same shape as f.
               f_hat[S] = (1/2^n) * sum_x f(x) * (-1)^(<S,x>)
    """
    n = int(np.log2(len(f)))
    assert len(f) == 2 ** n, f"Length must be a power of 2, got {len(f)}"

    f_hat = f.astype(np.float64).copy()

    # In-place FWHT (Hadamard ordering)
    h = 1
    while h < len(f_hat):
        for i in range(0, len(f_hat), h * 2):
            for j in range(i, i + h):
                x = f_hat[j]
                y = f_hat[j + h]
                f_hat[j] = x + y
                f_hat[j + h] = x - y
        h *= 2

    # Normalize
    f_hat /= len(f)
    return f_hat


def top_k_coefficients(
    f_hat: np.ndarray,
    k: int,
    exclude_constant: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the top-K Fourier coefficients by magnitude.

    Args:
        f_hat: Fourier coefficients from walsh_hadamard_transform.
        k: Number of coefficients to select.
        exclude_constant: If True, exclude the constant term (index 0).

    Returns:
        indices: Array of indices (S values) for top-K terms.
        coefficients: Corresponding coefficient values.

    Raises:
        ValueError: If k exceeds available non-constant coefficients.
    """
    max_k = len(f_hat) - 1 if exclude_constant else len(f_hat)
    if k > max_k:
        raise ValueError(f"k={k} exceeds available coefficients ({max_k})")

    if exclude_constant:
        # Work with indices 1 to 2^n - 1
        magnitudes = np.abs(f_hat[1:])
        top_k_local = np.argsort(magnitudes)[-k:][::-1]
        indices = top_k_local + 1  # Shift back to global indices
    else:
        magnitudes = np.abs(f_hat)
        indices = np.argsort(magnitudes)[-k:][::-1]

    coefficients = f_hat[indices]
    return indices, coefficients


def coefficients_to_max_xorsat(
    indices: np.ndarray,
    coefficients: np.ndarray,
    n_bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Fourier coefficients to max-XORSAT format (B, v, weights).

    Each Fourier term with index S becomes a parity constraint:
        - b_i = binary representation of S (parity mask)
        - v_i = 0 if coeff > 0 (prefer even parity), 1 if coeff < 0 (prefer odd)
        - weight_i = |coeff|

    Args:
        indices: Array of m index values (S ⊆ [n]).
        coefficients: Array of m Fourier coefficients.
        n_bits: Number of bits in x.

    Returns:
        B: (m, n) binary matrix of parity masks.
        v: (m,) binary vector of preferred parities.
        weights: (m,) positive weights (absolute values of coefficients).
    """
    m = len(indices)
    B = np.zeros((m, n_bits), dtype=np.int8)
    v = np.zeros(m, dtype=np.int8)
    weights = np.abs(coefficients)

    for i, idx in enumerate(indices):
        # Convert index to binary mask (MSB first)
        for j in range(n_bits):
            B[i, j] = (idx >> (n_bits - 1 - j)) & 1

        # Preferred parity: 0 if coeff > 0, 1 if coeff < 0
        v[i] = 0 if coefficients[i] > 0 else 1

    return B, v, weights


def _energy_fraction(f_hat: np.ndarray, coeffs: np.ndarray) -> float:
    """Return Parseval energy fraction captured by selected coefficients."""
    total_energy = np.sum(f_hat[1:] ** 2)
    captured_energy = np.sum(coeffs ** 2)
    return float(captured_energy / total_energy) if total_energy > 0 else 0.0


def build_surrogate_artifact(
    prob: "PricingProblem",
    k: int = 15,
    *,
    exclude_constant: bool = True,
    source_metadata: dict | None = None,
    created_by: str | None = None,
    repo_commit: str | None = None,
    random_seed: int | None = None,
) -> dict:
    """Build canonical exploratory WHT-topK surrogate artifact.

    Canonical coefficient magnitude key is ``term_weights``.
    """
    f = prob.build_truth_table()
    f_hat = walsh_hadamard_transform(f)
    indices, coeffs = top_k_coefficients(
        f_hat,
        k=k,
        exclude_constant=exclude_constant,
    )
    B, v, term_weights = coefficients_to_max_xorsat(indices, coeffs, prob.n_bits)

    g = np.array([surrogate_objective(x, B, v, term_weights, prob.n_bits)
                  for x in range(len(f))], dtype=np.float64)
    rho, _ = spearmanr(f, g)

    metadata = dict(source_metadata or {})
    metadata.setdefault("instance_id", f"pricing_{prob.n_features}feat_{prob.n_bits}bit")
    if created_by is not None:
        metadata.setdefault("created_by", created_by)
    if repo_commit is not None:
        metadata.setdefault("repo_commit", repo_commit)
    if random_seed is not None:
        metadata.setdefault("random_seed", int(random_seed))
    metadata.setdefault(
        "created_at",
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    return {
        "artifact_version": "1.0",
        "mode": "exploratory_wht_topk",
        "story_role": "exploratory",
        "benchmark_status": "non_mainline_surrogate",
        "bit_order": "msb_first",
        "n": int(prob.n_bits),
        "m": int(len(indices)),
        "k_requested": int(k),
        "k_selected": int(len(indices)),
        "exclude_constant": bool(exclude_constant),
        "B": B,
        "v": v,
        "term_weights": term_weights,
        "indices": indices,
        "coefficients": coeffs,
        "truncation_metadata": {
            "sort_key": "abs_coeff_desc",
            "tie_break": "stable_numpy_argsort",
        },
        "diagnostics": {
            "energy_fraction": _energy_fraction(f_hat, coeffs),
            "spearman_rho": float(rho),
        },
        "source_metadata": metadata,
        # Compatibility fields for existing call sites.
        "f_hat": f_hat,
    }


def surrogate_artifact_to_dict(artifact: dict) -> dict:
    """Convert artifact to a JSON-friendly dictionary."""
    out = dict(artifact)
    for key in ("B", "v", "term_weights", "indices", "coefficients", "f_hat"):
        if key in out and out[key] is not None:
            out[key] = np.asarray(out[key]).tolist()
    return out


def surrogate_artifact_from_dict(payload: dict) -> dict:
    """Load artifact from dict payload with list-or-array numeric fields."""
    out = dict(payload)
    out["B"] = np.asarray(out["B"], dtype=np.int8)
    out["v"] = np.asarray(out["v"], dtype=np.int8)

    if "term_weights" in out:
        out["term_weights"] = np.asarray(out["term_weights"], dtype=np.float64)
    elif "weights" in out:
        out["term_weights"] = np.asarray(out["weights"], dtype=np.float64)
    else:
        raise KeyError("Artifact payload missing 'term_weights' (or legacy 'weights').")

    out["indices"] = np.asarray(out["indices"], dtype=np.int64)
    out["coefficients"] = np.asarray(out["coefficients"], dtype=np.float64)
    if "f_hat" in out and out["f_hat"] is not None:
        out["f_hat"] = np.asarray(out["f_hat"], dtype=np.float64)
    return out


def surrogate_objective(
    x: int,
    B: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
    n_bits: int,
) -> float:
    """Evaluate the weighted surrogate objective G(x).

    G(x) = sum_i weight_i * (-1)^(v_i + b_i · x)

    Args:
        x: Input bitstring as integer.
        B: (m, n) parity mask matrix.
        v: (m,) preferred parity vector.
        weights: (m,) coefficient weights.
        n_bits: Number of bits.

    Returns:
        G(x) value.
    """
    x_bits = np.array([(x >> (n_bits - 1 - j)) & 1 for j in range(n_bits)],
                      dtype=np.int8)
    parities = (B @ x_bits) % 2  # b_i · x mod 2
    signs = (-1) ** (v + parities)
    return float(np.sum(weights * signs))


def surrogate_objective_unweighted(
    x: int,
    B: np.ndarray,
    v: np.ndarray,
    n_bits: int,
) -> int:
    """Evaluate the unweighted surrogate f_unw(x) = sum_i (-1)^(v_i + b_i · x).

    This is the native max-XORSAT objective: counts satisfied - violated.

    Returns:
        Integer in range [-m, m].
    """
    x_bits = np.array([(x >> (n_bits - 1 - j)) & 1 for j in range(n_bits)],
                      dtype=np.int8)
    parities = (B @ x_bits) % 2
    signs = (-1) ** (v + parities)
    return int(np.sum(signs))


def count_satisfied(
    x: int,
    B: np.ndarray,
    v: np.ndarray,
    n_bits: int,
) -> int:
    """Count the number of satisfied XORSAT constraints.

    Constraint i is satisfied when b_i · x ≡ v_i (mod 2).

    Returns:
        Number of satisfied constraints (0 to m).
    """
    x_bits = np.array([(x >> (n_bits - 1 - j)) & 1 for j in range(n_bits)],
                      dtype=np.int8)
    parities = (B @ x_bits) % 2
    return int(np.sum(parities == v))


def build_surrogate(
    prob: "PricingProblem",
    k: int = 15,
    ell: int | None = None,
) -> dict:
    """Build the complete max-XORSAT surrogate from a pricing problem.

    Args:
        prob: PricingProblem instance.
        k: Number of top Fourier coefficients to keep.
        ell: Optional decode radius for structure metrics.

    Returns:
        Dictionary with keys:
            'B': (m, n) parity mask matrix
            'v': (m,) preferred parity vector
            'weights': (m,) coefficient weights
            'indices': (m,) Fourier indices
            'coefficients': (m,) raw Fourier coefficients
            'f_hat': full Fourier transform
            'n_bits': number of bits
            'm': number of constraints
    """
    artifact = build_surrogate_artifact(prob, k=k)

    out = {
        'B': artifact["B"],
        'v': artifact["v"],
        'weights': artifact["term_weights"],  # legacy compatibility alias
        'term_weights': artifact["term_weights"],
        'indices': artifact["indices"],
        'coefficients': artifact["coefficients"],
        'f_hat': artifact["f_hat"],
        'n_bits': artifact["n"],
        'n': artifact["n"],
        'm': artifact["m"],
        'artifact': artifact,
    }
    if ell is not None:
        out["structure_metrics"] = compute_structure_metrics(out["B"], ell=ell)
    return out


def surrogate_faithfulness(
    prob: "PricingProblem",
    k: int = 15,
) -> tuple[float, float]:
    """Compute surrogate quality metrics.

    Args:
        prob: PricingProblem instance.
        k: Number of top coefficients.

    Returns:
        rho: Spearman rank correlation between F and G.
        eta: Fraction of Parseval energy captured by top-K.
    """
    artifact = build_surrogate_artifact(prob, k=k)
    rho = artifact["diagnostics"]["spearman_rho"]
    eta = artifact["diagnostics"]["energy_fraction"]
    return float(rho), float(eta)


def surrogate_structure_metrics(
    prob: "PricingProblem",
    k: int = 15,
    ell: int = 3,
) -> dict:
    """Compute structure-sensitive diagnostics for a surrogate instance."""
    surrogate = build_surrogate(prob, k=k)
    return compute_structure_metrics(surrogate["B"], ell=ell)
