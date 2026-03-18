"""Compatibility tests for structure metrics on Python runtimes without int.bit_count."""

from __future__ import annotations

import numpy as np

from src.structure_metrics import _approximate_distance_proxy


def test_approximate_distance_proxy_runs_without_int_bit_count_support() -> None:
    B = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )

    out = _approximate_distance_proxy(B, max_search_bits=8)
    assert out is not None
