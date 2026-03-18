"""Tiny-instance exact oracle decoder."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from src.decoder_bruteforce import compute_syndrome, hamming_weight


class TinyOracleDecoder:
    """Exact syndrome decoder restricted to tiny instances."""

    def __init__(
        self,
        B: np.ndarray,
        ell: int,
        *,
        max_m_for_oracle: int = 10,
    ) -> None:
        self.B = np.asarray(B, dtype=np.int8)
        self.m, self.n = self.B.shape
        self.ell = int(ell)
        self.max_m_for_oracle = int(max_m_for_oracle)
        self._tiny_supported = self.m <= self.max_m_for_oracle

    def _exact_decode(self, syndrome: np.ndarray) -> np.ndarray | None:
        for wt in range(self.m + 1):
            for combo in combinations(range(self.m), wt):
                e = np.zeros(self.m, dtype=np.int8)
                for idx in combo:
                    e[idx] = 1
                if np.array_equal(compute_syndrome(self.B, e), syndrome):
                    return e
        return None

    def decode(self, syndrome: np.ndarray) -> np.ndarray | None:
        result = self.decode_result(syndrome)
        return result["decoded_error"]

    def decode_result(self, syndrome: np.ndarray) -> dict:
        syndrome = np.asarray(syndrome, dtype=np.int8)
        if not self._tiny_supported:
            return {
                "decoder_name": "oracle",
                "decoded_error": None,
                "success": False,
                "within_radius": False,
                "distance": None,
                "num_flips": 0,
                "metadata": {
                    "decoder_status": "skipped_tiny_only",
                    "status_reason_code": "oracle_tiny_cap_exceeded",
                    "run_status": "skipped",
                    "max_m_for_oracle": int(self.max_m_for_oracle),
                    "m": int(self.m),
                },
            }

        decoded = self._exact_decode(syndrome)
        if decoded is None:
            return {
                "decoder_name": "oracle",
                "decoded_error": None,
                "success": False,
                "within_radius": False,
                "distance": None,
                "num_flips": 0,
                "metadata": {
                    "decoder_status": "completed",
                    "status_reason_code": None,
                    "run_status": "completed",
                    "max_m_for_oracle": int(self.max_m_for_oracle),
                    "m": int(self.m),
                },
            }

        dist = int(hamming_weight(decoded))
        return {
            "decoder_name": "oracle",
            "decoded_error": decoded.copy(),
            "success": True,
            "within_radius": bool(dist <= self.ell),
            "distance": dist,
            "num_flips": dist,
            "metadata": {
                "decoder_status": "completed",
                "status_reason_code": None,
                "run_status": "completed",
                "max_m_for_oracle": int(self.max_m_for_oracle),
                "m": int(self.m),
            },
        }
