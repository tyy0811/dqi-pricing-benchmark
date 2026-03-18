"""Deterministic hard-decision BP1-style decoder for small DQI instances."""

from __future__ import annotations

from typing import Optional

import numpy as np
from math import ceil, log2

from src.decoder_bruteforce import compute_syndrome, hamming_weight


class BP1Decoder:
    """Hard-decision bit-flip decoder with deterministic iteration order."""

    def __init__(self, B: np.ndarray, ell: int, max_iter: int = 16):
        self.B = B.astype(np.int8)
        self.m, self.n = self.B.shape
        self.ell = int(ell)
        self.max_iter = int(max_iter)

    def _residual(self, e: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
        """Return residual vector for equation B^T e = syndrome (mod 2)."""
        lhs = compute_syndrome(self.B, e)
        return (lhs ^ syndrome.astype(np.int8)).astype(np.int8)

    def _choose_flip(self, residual: np.ndarray) -> tuple[int | None, int]:
        """Choose best variable index to flip based on unsatisfied-check reduction."""
        unsat_before = int(np.sum(residual))
        best_idx = None
        best_gain = 0

        for j in range(self.m):
            residual_after = residual ^ self.B[j, :]
            gain = unsat_before - int(np.sum(residual_after))
            if gain > best_gain:
                best_gain = gain
                best_idx = j

        return best_idx, best_gain

    def decode(self, syndrome: np.ndarray) -> Optional[np.ndarray]:
        """Compatibility decode API returning decoded error or None."""
        result = self.decode_result(syndrome)
        return result["decoded_error"]

    def decode_result(self, syndrome: np.ndarray) -> dict:
        """Decode syndrome and return canonical structured result payload."""
        syndrome = syndrome.astype(np.int8)
        e = np.zeros(self.m, dtype=np.int8)
        num_flips = 0
        iterations = 0

        for it in range(self.max_iter):
            iterations = it + 1
            residual = self._residual(e, syndrome)
            if int(np.sum(residual)) == 0:
                distance = int(hamming_weight(e))
                return {
                    "decoder_name": "bp1",
                    "decoded_error": e.copy(),
                    "success": True,
                    "within_radius": bool(distance <= self.ell),
                    "distance": distance,
                    "num_flips": int(num_flips),
                    "metadata": {
                        "ell": int(self.ell),
                        "iterations": int(iterations),
                        "max_iter": int(self.max_iter),
                        "unsatisfied_final": 0,
                    },
                }

            idx, gain = self._choose_flip(residual)
            if idx is None or gain <= 0:
                break

            e[idx] ^= 1
            num_flips += 1

        residual_final = self._residual(e, syndrome)
        return {
            "decoder_name": "bp1",
            "decoded_error": None,
            "success": False,
            "within_radius": False,
            "distance": None,
            "num_flips": int(num_flips),
            "metadata": {
                "ell": int(self.ell),
                "iterations": int(iterations),
                "max_iter": int(self.max_iter),
                "unsatisfied_final": int(np.sum(residual_final)),
            },
        }


def bp1_classical_oracle_resource_estimate(
    B: np.ndarray,
    ell: int,
    max_iter: int = 16,
) -> dict:
    """Analytic operation-count proxy for BP1 decode/uncompute resources.

    This is not a synthesized reversible circuit. It estimates the cost of
    evaluating the BP1 update rule as a classical oracle inside a larger
    quantum routine.
    """
    B = np.asarray(B, dtype=np.int8)
    m, n = B.shape
    max_iter = int(max_iter)

    per_iter_residual_eval = 2 * m * n
    per_iter_flip_selection = m * (2 * n + 1)
    per_iter_update = n + 2 * m
    op_count_proxy = max_iter * (
        per_iter_residual_eval + per_iter_flip_selection + per_iter_update
    )

    gate_est = 4 * op_count_proxy
    depth_est = max_iter * (5 * n + 3 * m + 6)
    ancilla_est = n + m + int(ceil(log2(max(2, max_iter + 1))))

    return {
        "decode_uncompute_model": "bp1_classical_oracle_estimate",
        "decode_uncompute_status": "analytic_estimate",
        "decode_uncompute_gate_est": int(gate_est),
        "decode_uncompute_depth_est": int(depth_est),
        "decode_uncompute_ancilla_est": int(ancilla_est),
        "op_count_proxy": int(op_count_proxy),
        "max_iter": max_iter,
        "ell": int(ell),
        "not_synthesized_reversible": True,
        "assumptions": [
            "BP1 treated as a classical oracle proxy, not compiled reversible logic",
            "Each iteration scans all m variables against n-bit residual checks",
            "Gate/depth estimates are conservative scaling proxies from operation counts",
        ],
    }
