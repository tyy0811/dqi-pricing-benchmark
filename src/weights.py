"""Compatibility shim for alpha-weight utilities.

Canonical exports:
- uniform_weights(...)
- paper_alpha_weights(...)
- heuristic_alpha_weights(...)

Legacy aliases remain for compatibility:
- optimal_weights(...) -> heuristic_alpha_weights(...)
"""

from __future__ import annotations

from src.weights_heuristic import (
    build_weight_matrix_heuristic,
    compare_weights,
    heuristic_alpha_weights,
    optimal_weights,  # compatibility alias
)
from src.weights_paper import (
    max_safe_ell,
    paper_alpha_matrix,
    paper_alpha_weights,
    semicircle_prediction,
    uniform_weights,
    validate_dqi_parameters,
)

__all__ = [
    "max_safe_ell",
    "validate_dqi_parameters",
    "uniform_weights",
    "paper_alpha_matrix",
    "paper_alpha_weights",
    "heuristic_alpha_weights",
    "semicircle_prediction",
    "build_weight_matrix_heuristic",
    "compare_weights",
    # Compatibility aliases:
    "optimal_weights",
]
