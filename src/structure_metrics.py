"""
structure_metrics.py — Structural diagnostics for surrogate parity systems.

These metrics explain decoder behavior and decode/uncompute resource scaling.
"""

from __future__ import annotations

import numpy as np

from src.decoder_bruteforce import BoundedDistanceDecoder


def _int_bit_count(value: int) -> int:
    """Compatibility wrapper for Python runtimes without ``int.bit_count``."""
    if hasattr(value, "bit_count"):
        return int(value.bit_count())
    return int(bin(int(value)).count("1"))


def _gf2_rank(matrix: np.ndarray) -> int:
    """Return rank over GF(2) via row-reduction."""
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


def _approximate_distance_proxy(
    B: np.ndarray,
    *,
    max_search_bits: int = 18,
) -> float | None:
    """Approximate distance proxy from small-nullspace search on B^T.

    For moderate ``m`` this computes the minimum Hamming weight of a nonzero
    error pattern ``y`` such that ``B^T y = 0 (mod 2)``. When ``m`` is too
    large for deterministic brute force, returns ``None``.
    """
    b = (np.asarray(B, dtype=np.uint8) & 1)
    m = int(b.shape[0])
    if m <= 0:
        return None
    if m > int(max_search_bits):
        return None

    bt = b.T
    best: int | None = None
    for mask in range(1, 1 << m):
        wt = _int_bit_count(mask)
        if best is not None and wt >= best:
            continue
        y = np.array([(mask >> (m - 1 - j)) & 1 for j in range(m)], dtype=np.uint8)
        syndrome = (bt @ y) % 2
        if int(np.sum(syndrome)) == 0:
            best = wt
            if best == 1:
                break

    if best is None:
        # No nonzero nullspace witness found in searched domain.
        return float(m + 1)
    return float(best)


def _short_cycle_proxy(B: np.ndarray) -> float:
    """Return overlap-based short-cycle proxy in bipartite parity graph.

    Counts the fraction of row pairs sharing at least two variable nodes, which
    acts as a lightweight proxy for short even cycles that destabilize decoding.
    """
    b = (np.asarray(B, dtype=np.uint8) & 1)
    m = int(b.shape[0])
    if m < 2:
        return 0.0

    total_pairs = 0
    short_cycle_pairs = 0
    for i in range(m):
        for j in range(i + 1, m):
            total_pairs += 1
            overlap = int(np.sum(b[i] & b[j]))
            if overlap >= 2:
                short_cycle_pairs += 1
    if total_pairs == 0:
        return 0.0
    return float(short_cycle_pairs / total_pairs)


def compute_structure_metrics(B: np.ndarray, ell: int) -> dict:
    """Compute structure-sensitive surrogate metrics for one instance."""
    B = np.asarray(B, dtype=np.int8)
    if B.ndim != 2:
        raise ValueError(f"B must be 2D, got shape {B.shape}")
    if ell < 0:
        raise ValueError(f"ell must be non-negative, got {ell}")

    m, n = B.shape
    row_w = np.sum(B, axis=1, dtype=np.int64)
    col_w = np.sum(B, axis=0, dtype=np.int64)

    rank = _gf2_rank(B)
    unique_syndromes_full = int(2 ** rank)
    syndrome_space_size = int(2 ** n)
    syndrome_density = float(unique_syndromes_full / syndrome_space_size)

    decoder = BoundedDistanceDecoder(B, ell)
    estimated_lookup_table_size = int(decoder.num_patterns)
    decodable_unique = int(decoder.decodable_syndrome_count)
    ambiguous = int(decoder.ambiguous_syndrome_count)
    decoder_collision_rate = float(
        ambiguous / estimated_lookup_table_size
        if estimated_lookup_table_size > 0
        else 0.0
    )
    decodable_syndrome_fraction = float(
        decodable_unique / syndrome_space_size
        if syndrome_space_size > 0
        else 0.0
    )
    rank_deficiency = int(max(0, n - rank))
    approx_distance_proxy = _approximate_distance_proxy(B)
    short_cycle_proxy = _short_cycle_proxy(B)

    return {
        "row_weight_min": int(np.min(row_w)) if m > 0 else 0,
        "row_weight_mean": float(np.mean(row_w)) if m > 0 else 0.0,
        "row_weight_max": int(np.max(row_w)) if m > 0 else 0,
        "col_weight_min": int(np.min(col_w)) if n > 0 else 0,
        "col_weight_mean": float(np.mean(col_w)) if n > 0 else 0.0,
        "col_weight_max": int(np.max(col_w)) if n > 0 else 0,
        "syndrome_density": syndrome_density,
        "number_of_unique_syndromes_decodable_at_ell": decodable_unique,
        "decoder_collision_rate": decoder_collision_rate,
        "estimated_lookup_table_size": estimated_lookup_table_size,
        "ambiguous_syndrome_count": ambiguous,
        "syndrome_space_size": syndrome_space_size,
        "rank_B_gf2": rank,
        "rank": rank,
        "rank_deficiency": rank_deficiency,
        "short_cycle_proxy": short_cycle_proxy,
        "decodable_syndrome_fraction": decodable_syndrome_fraction,
        "approximate_distance_proxy": approx_distance_proxy,
        "distance_proxy": approx_distance_proxy,
        # Canonical stage-c aliases for direct join usage.
        "structure_row_weight_min": int(np.min(row_w)) if m > 0 else 0,
        "structure_row_weight_mean": float(np.mean(row_w)) if m > 0 else 0.0,
        "structure_row_weight_max": int(np.max(row_w)) if m > 0 else 0,
        "structure_col_weight_min": int(np.min(col_w)) if n > 0 else 0,
        "structure_col_weight_mean": float(np.mean(col_w)) if n > 0 else 0.0,
        "structure_col_weight_max": int(np.max(col_w)) if n > 0 else 0,
        "structure_rank": rank,
        "structure_rank_deficiency": rank_deficiency,
        "structure_collision_rate": decoder_collision_rate,
        "structure_ambiguous_syndrome_count": ambiguous,
        "structure_short_cycle_proxy": short_cycle_proxy,
        "structure_distance_proxy": approx_distance_proxy,
        "ell": int(ell),
        "m": int(m),
        "n": int(n),
    }
