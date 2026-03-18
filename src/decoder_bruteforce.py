"""
decoder_bruteforce.py — Bounded-distance brute-force decoder for max-XORSAT.

For small m (≤ 20) and ell (≤ 4), we enumerate all error patterns e with
|e| ≤ ell, compute their syndromes s = B^T @ e mod 2, and build a lookup table.

Convention: We store parity masks as rows of B (shape m×n), so syndrome is
B^T @ e (length n). This is equivalent to Be in the paper's convention when
B stores masks as columns.

Reference: Jordan et al. (2024) arXiv:2408.08292, Section 8.1.2 Step 4
"""

from __future__ import annotations

import numpy as np
from itertools import combinations
from math import ceil, log2
from typing import Optional


def hamming_weight(x: np.ndarray) -> int:
    """Count the number of 1s in a binary array."""
    return int(np.sum(x))


def compute_syndrome(B: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Compute syndrome s = B^T @ e mod 2.

    Args:
        B: (m, n) binary matrix.
        e: (m,) binary error vector.

    Returns:
        s: (n,) syndrome vector.
    """
    return (B.T @ e) % 2


def syndrome_to_tuple(s: np.ndarray) -> tuple:
    """Convert syndrome array to hashable tuple."""
    return tuple(s.astype(int))


class BoundedDistanceDecoder:
    """Brute-force bounded-distance decoder.

    Enumerates all error patterns of weight ≤ ell and builds a syndrome
    lookup table. Decoding looks up the syndrome and returns the corresponding
    error pattern (or None if not found or ambiguous due to collision).

    Attributes:
        B: (m, n) parity check matrix.
        m: Number of constraints (rows of B).
        n: Number of variables (columns of B).
        ell: Maximum error weight to decode.
        lookup: Dict mapping syndrome tuple to error pattern (None if ambiguous).
        num_collisions: Number of syndrome collisions detected.
    """

    def __init__(self, B: np.ndarray, ell: int):
        """Initialize decoder and build lookup table.

        Args:
            B: (m, n) binary parity matrix.
            ell: Maximum Hamming weight of errors to decode.
        """
        self.B = B.astype(np.int8)
        self.m, self.n = B.shape
        self.ell = ell
        self.lookup: dict[tuple, Optional[np.ndarray]] = {}
        self.num_collisions = 0
        self._build_lookup_table()

    def _build_lookup_table(self) -> None:
        """Enumerate all weight-≤-ell patterns and store syndromes."""
        # Weight 0: all-zeros error
        e_zero = np.zeros(self.m, dtype=np.int8)
        s_zero = compute_syndrome(self.B, e_zero)
        self.lookup[syndrome_to_tuple(s_zero)] = e_zero.copy()

        # Weights 1 to ell
        for weight in range(1, self.ell + 1):
            for positions in combinations(range(self.m), weight):
                e = np.zeros(self.m, dtype=np.int8)
                for pos in positions:
                    e[pos] = 1
                s = compute_syndrome(self.B, e)
                s_key = syndrome_to_tuple(s)

                if s_key not in self.lookup:
                    self.lookup[s_key] = e.copy()
                elif self.lookup[s_key] is not None:
                    # Collision: different error patterns map to same syndrome
                    # Mark as ambiguous (None) so decode() returns None
                    if not np.array_equal(self.lookup[s_key], e):
                        self.lookup[s_key] = None
                        self.num_collisions += 1

    def decode(self, syndrome: np.ndarray) -> Optional[np.ndarray]:
        """Decode a syndrome to recover the error pattern.

        Args:
            syndrome: (n,) binary syndrome vector.

        Returns:
            e: (m,) error pattern if found and unambiguous, None otherwise.
               Returns None if syndrome not in lookup or marked ambiguous.
        """
        s_key = syndrome_to_tuple(syndrome.astype(np.int8))
        result = self.lookup.get(s_key, None)
        # Return None for both missing syndromes and ambiguous ones
        return result if result is not None else None

    def decode_result(self, syndrome: np.ndarray) -> dict:
        """Decode syndrome and return canonical structured result payload."""
        recovered = self.decode(syndrome)
        if recovered is None:
            return {
                "decoder_name": "bruteforce",
                "decoded_error": None,
                "success": False,
                "within_radius": False,
                "distance": None,
                "num_flips": 0,
                "metadata": {
                    "ell": int(self.ell),
                    "num_collisions": int(self.num_collisions),
                },
            }

        distance = int(hamming_weight(recovered))
        return {
            "decoder_name": "bruteforce",
            "decoded_error": recovered.copy(),
            "success": True,
            "within_radius": bool(distance <= self.ell),
            "distance": distance,
            "num_flips": distance,
            "metadata": {
                "ell": int(self.ell),
                "num_collisions": int(self.num_collisions),
            },
        }

    def decode_int(self, syndrome_int: int) -> Optional[np.ndarray]:
        """Decode a syndrome given as an integer (MSB first).

        Args:
            syndrome_int: Syndrome as integer.

        Returns:
            e: (m,) error pattern if found, None otherwise.
        """
        s = np.array([(syndrome_int >> (self.n - 1 - j)) & 1
                      for j in range(self.n)], dtype=np.int8)
        return self.decode(s)

    @property
    def num_patterns(self) -> int:
        """Number of error patterns in lookup table."""
        return len(self.lookup)

    @property
    def decodable_syndrome_count(self) -> int:
        """Number of syndromes that decode to a unique error."""
        return int(sum(1 for v in self.lookup.values() if v is not None))

    @property
    def ambiguous_syndrome_count(self) -> int:
        """Number of syndromes marked ambiguous in the lookup table."""
        return int(sum(1 for v in self.lookup.values() if v is None))


def build_decoder(B: np.ndarray, ell: int) -> BoundedDistanceDecoder:
    """Convenience function to create a decoder."""
    return BoundedDistanceDecoder(B, ell)


def lookup_table_reversible_resource_estimate(B: np.ndarray, ell: int) -> dict:
    """Conservative reversible-logic estimate for lookup-table decode/uncompute.

    Model:
    - Build a reversible read-only map from syndrome bits (n) to error bits (m).
    - Implemented as a serial equality-check network over populated lookup rows.
    - Each row pays a full n-bit comparator cost; decodable rows also pay m-bit
      controlled write cost for the recovered error.
    - Costs are intentionally conservative to avoid understating decode overhead.
    """
    B = np.asarray(B, dtype=np.int8)
    m, n = B.shape
    decoder = BoundedDistanceDecoder(B, ell)

    mapping_size = int(decoder.num_patterns)
    decodable_syndromes = int(decoder.decodable_syndrome_count)
    ambiguous_syndromes = int(decoder.ambiguous_syndrome_count)
    syndrome_space_size = int(2 ** n)

    comparator_gate_per_entry = 12 * n + 8
    write_gate_per_decodable_entry = 4 * m + 4
    gate_est = (
        mapping_size * comparator_gate_per_entry
        + decodable_syndromes * write_gate_per_decodable_entry
    )

    comparator_depth_per_entry = 6 * n + 2
    write_depth_per_decodable_entry = 2 * m + 1
    depth_est = (
        mapping_size * comparator_depth_per_entry
        + decodable_syndromes * write_depth_per_decodable_entry
    )

    ancilla_est = (
        n  # comparator workspace
        + m  # temporary decoded-write workspace
        + int(ceil(log2(max(1, mapping_size))))  # row index scratch
    )

    return {
        "decode_uncompute_model": "lookup_table_reversible",
        "decode_uncompute_status": "analytic_estimate_conservative",
        "decode_uncompute_gate_est": int(gate_est),
        "decode_uncompute_depth_est": int(depth_est),
        "decode_uncompute_ancilla_est": int(ancilla_est),
        "mapping_size": mapping_size,
        "decodable_syndromes": decodable_syndromes,
        "ambiguous_syndromes": ambiguous_syndromes,
        "syndrome_space_size": syndrome_space_size,
        "assumptions": [
            "Serial reversible lookup over populated syndrome rows",
            "n-bit comparators per row with explicit uncompute",
            "m-bit controlled write for uniquely decodable rows",
            "Conservative analytic estimate, not compiled gate synthesis",
        ],
    }
