"""Deterministic BP + OSD-lite decoder for small DQI instances."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from src.decoder_bp1 import BP1Decoder
from src.decoder_bruteforce import compute_syndrome, hamming_weight


class BPOSDLiteDecoder:
    """BP-first decoder with a bounded deterministic OSD-lite fallback."""

    def __init__(
        self,
        B: np.ndarray,
        ell: int,
        *,
        bp_max_iter: int = 16,
        osd_order_cap: int = 2,
        osd_candidate_budget: int = 64,
        tie_break: str = "lowest_weight_then_lexicographic",
    ) -> None:
        self.B = np.asarray(B, dtype=np.int8)
        self.m, self.n = self.B.shape
        self.ell = int(ell)
        self.bp_max_iter = int(bp_max_iter)
        self.osd_order_cap = max(0, int(osd_order_cap))
        self.osd_candidate_budget = max(1, int(osd_candidate_budget))
        self.tie_break = str(tie_break)
        self._bp = BP1Decoder(self.B, ell=self.ell, max_iter=self.bp_max_iter)

    def _rank_bits(self, syndrome: np.ndarray) -> list[int]:
        """Rank error bits by parity overlap with the target syndrome."""
        s = np.asarray(syndrome, dtype=np.int8)
        scores = []
        for bit in range(self.m):
            overlap = int(np.dot(self.B[bit].astype(np.int64), s.astype(np.int64)))
            scores.append((overlap, -bit, bit))
        # Higher overlap first, then smaller bit index (deterministic).
        scores.sort(reverse=True)
        return [bit for _overlap, _neg_bit, bit in scores]

    def _candidate_masks(self, ranked_bits: list[int]) -> list[np.ndarray]:
        pool = ranked_bits[: min(len(ranked_bits), max(4, 2 * self.osd_order_cap + 2))]
        candidates: list[np.ndarray] = []
        for order in range(self.osd_order_cap + 1):
            for combo in combinations(pool, order):
                e = np.zeros(self.m, dtype=np.int8)
                for bit in combo:
                    e[bit] = 1
                candidates.append(e)
                if len(candidates) >= self.osd_candidate_budget:
                    return candidates
        return candidates

    def _osd_search(self, syndrome: np.ndarray) -> tuple[np.ndarray | None, int]:
        ranked = self._rank_bits(syndrome)
        candidates = self._candidate_masks(ranked)
        best: np.ndarray | None = None
        checked = 0
        for cand in candidates:
            checked += 1
            if np.array_equal(compute_syndrome(self.B, cand), syndrome):
                if best is None:
                    best = cand.copy()
                    continue
                cur_w = int(hamming_weight(cand))
                best_w = int(hamming_weight(best))
                if cur_w < best_w:
                    best = cand.copy()
                elif cur_w == best_w:
                    # deterministic tie-break: lexicographic vector order
                    if tuple(cand.tolist()) < tuple(best.tolist()):
                        best = cand.copy()
        return best, checked

    def decode(self, syndrome: np.ndarray) -> np.ndarray | None:
        result = self.decode_result(syndrome)
        return result["decoded_error"]

    def decode_result(self, syndrome: np.ndarray) -> dict:
        syndrome = np.asarray(syndrome, dtype=np.int8)

        bp_result = self._bp.decode_result(syndrome)
        bp_meta = bp_result.get("metadata", {}) or {}
        if bool(bp_result.get("success", False)):
            decoded = bp_result.get("decoded_error")
            distance = int(hamming_weight(np.asarray(decoded, dtype=np.int8)))
            return {
                "decoder_name": "bp_osd_lite",
                "decoded_error": np.asarray(decoded, dtype=np.int8).copy(),
                "success": True,
                "within_radius": bool(distance <= self.ell),
                "distance": distance,
                "num_flips": int(bp_result.get("num_flips", distance)),
                "metadata": {
                    "osd_order_cap": int(self.osd_order_cap),
                    "osd_candidate_budget": int(self.osd_candidate_budget),
                    "tie_break": self.tie_break,
                    "fallback_flag": False,
                    "iterations_used": int(bp_meta.get("iterations", 0)),
                },
            }

        decoded, checked = self._osd_search(syndrome)
        if decoded is None:
            return {
                "decoder_name": "bp_osd_lite",
                "decoded_error": None,
                "success": False,
                "within_radius": False,
                "distance": None,
                "num_flips": int(bp_result.get("num_flips", 0)),
                "metadata": {
                    "osd_order_cap": int(self.osd_order_cap),
                    "osd_candidate_budget": int(self.osd_candidate_budget),
                    "tie_break": self.tie_break,
                    "fallback_flag": True,
                    "iterations_used": int(bp_meta.get("iterations", 0)),
                    "osd_candidates_checked": int(checked),
                },
            }

        distance = int(hamming_weight(decoded))
        return {
            "decoder_name": "bp_osd_lite",
            "decoded_error": decoded.copy(),
            "success": True,
            "within_radius": bool(distance <= self.ell),
            "distance": distance,
            "num_flips": distance,
            "metadata": {
                "osd_order_cap": int(self.osd_order_cap),
                "osd_candidate_budget": int(self.osd_candidate_budget),
                "tie_break": self.tie_break,
                "fallback_flag": True,
                "iterations_used": int(bp_meta.get("iterations", 0)),
                "osd_candidates_checked": int(checked),
            },
        }
