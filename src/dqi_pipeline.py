"""
dqi_pipeline.py — DQI sampling and evaluation pipeline.

Orchestrates the full DQI workflow:
1. Build surrogate (B, v) from pricing problem
2. Run DQI statevector simulation
3. Evaluate samples on both surrogate G and true objective F
4. Report metrics

Reference: Jordan et al. (2024) arXiv:2408.08292
"""

from __future__ import annotations

import numpy as np
from typing import Optional, TYPE_CHECKING

from src.reduction import (
    build_surrogate,
    surrogate_objective,
    surrogate_objective_unweighted,
    count_satisfied,
    surrogate_faithfulness,
)
from src.weights import (
    heuristic_alpha_weights,
    paper_alpha_weights,
    uniform_weights,
)
from src.dqi_state import (
    coherent_supported_for_instance,
    decoder_diagnostics,
    run_dqi_statevector,
    weight_register_bits,
)

if TYPE_CHECKING:
    from src.problem_generator import PricingProblem


def _is_missing_or_nan(value: object) -> bool:
    """Return True when a reviewer-facing metric value is unusable."""
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def has_complete_alpha_metrics(report: dict) -> bool:
    """Return True only when the compact appendix fields are all usable.

    This is a pure reviewer-facing gate. It does not infer, repair, or enrich
    incomplete rows.
    """
    success_value = report.get("success_prob")
    if _is_missing_or_nan(success_value):
        success_value = report.get("postselection_success")
    if _is_missing_or_nan(success_value):
        success_value = report.get("decoder_success_rate")

    # Use best_F with fallback to best_sampled_F
    best_f_value = report.get("best_F")
    if _is_missing_or_nan(best_f_value):
        best_f_value = report.get("best_sampled_F")

    required_values = [
        report.get("alpha_mode"),
        report.get("execution_mode"),
        report.get("status"),
        best_f_value,
        success_value,
        report.get("top1_regret"),
    ]
    return not any(_is_missing_or_nan(value) for value in required_values)


def compare_candidate_distributions_tvd(run_a: dict, run_b: dict) -> dict:
    """Compute TVD between two run payload candidate distributions."""
    probs_a = run_a.get("probabilities")
    probs_b = run_b.get("probabilities")
    if probs_a is None or probs_b is None:
        return {"ok": False, "reason": "missing_probabilities", "tvd": None}

    pa = np.asarray(probs_a, dtype=np.float64)
    pb = np.asarray(probs_b, dtype=np.float64)
    if pa.ndim != 1 or pb.ndim != 1:
        return {"ok": False, "reason": "probabilities_must_be_1d", "tvd": None}

    if len(pa) != len(pb):
        max_len = max(len(pa), len(pb))
        pa = np.pad(pa, (0, max_len - len(pa)))
        pb = np.pad(pb, (0, max_len - len(pb)))

    sa = float(np.sum(pa))
    sb = float(np.sum(pb))
    if sa <= 0 or sb <= 0:
        return {"ok": False, "reason": "non_positive_probability_mass", "tvd": None}

    pa = pa / sa
    pb = pb / sb
    tvd = float(0.5 * np.sum(np.abs(pa - pb)))
    return {"ok": True, "reason": None, "tvd": tvd}


class DQIPipeline:
    """Full DQI pipeline for pricing optimization.

    Attributes:
        prob: PricingProblem instance.
        k: Number of Fourier terms (surrogate sparsity).
        ell: Maximum Dicke weight.
        alpha_mode: Alpha constructor mode ("uniform", "paper", or "heuristic").
        use_optimal_weights: Backward-compatibility bool alias for alpha_mode.
        execution_mode: Statevector execution mode ("mixture" or "coherent").
        decoder_mode: Decoder selector ("bruteforce" or "bp1").
        surrogate: Cached surrogate (B, v, weights).
    """

    def __init__(
        self,
        prob: "PricingProblem",
        k: int = 15,
        ell: int = 3,
        alpha_mode: str = "heuristic",
        use_optimal_weights: Optional[bool] = None,
        execution_mode: str = "mixture",
        decoder_mode: str = "bruteforce",
    ):
        """Initialize DQI pipeline.

        Args:
            prob: PricingProblem instance.
            k: Number of top Fourier terms for surrogate.
            ell: Maximum Dicke weight bound.
            alpha_mode: Alpha path ("uniform", "paper", or "heuristic").
            use_optimal_weights: Backward-compatible alias for alpha_mode.
            execution_mode: "mixture" or "coherent".
            decoder_mode: "bruteforce" or "bp1".
        """
        self.prob = prob
        self.k = k
        self.ell = ell
        self.alpha_mode = self._resolve_alpha_mode(
            alpha_mode=alpha_mode,
            use_optimal_weights=use_optimal_weights,
        )
        self.use_optimal_weights = self.alpha_mode == "heuristic"
        self.execution_mode = self._resolve_execution_mode(execution_mode)
        self.decoder_mode = self._resolve_decoder_mode(decoder_mode)

        # Build surrogate
        self.surrogate = build_surrogate(prob, k=k)
        self.B = self.surrogate['B']
        self.v = self.surrogate['v']
        self.weights = self.surrogate['weights']

        self.n_bits = prob.n_bits
        self.m = self.B.shape[0]  # actual number of parity terms
        self.decoder_stats = decoder_diagnostics(
            B=self.B,
            ell=self.ell,
            decoder_mode=self.decoder_mode,
        )

    @staticmethod
    def _resolve_alpha_mode(alpha_mode: str, use_optimal_weights: Optional[bool]) -> str:
        """Resolve alpha mode, honoring backward-compatible bool selector."""
        if use_optimal_weights is not None:
            alpha_mode = "heuristic" if use_optimal_weights else "uniform"

        allowed = {"uniform", "paper", "heuristic"}
        if alpha_mode not in allowed:
            raise ValueError(
                f"Invalid alpha_mode={alpha_mode!r}. Must be one of {sorted(allowed)}."
            )
        return alpha_mode

    @staticmethod
    def _resolve_execution_mode(execution_mode: str) -> str:
        """Validate pipeline execution mode."""
        allowed = {"mixture", "coherent"}
        if execution_mode not in allowed:
            raise ValueError(
                f"Invalid execution_mode={execution_mode!r}. Must be one of {sorted(allowed)}."
            )
        return execution_mode

    @staticmethod
    def _resolve_decoder_mode(decoder_mode: str) -> str:
        """Validate decoder selector mode."""
        allowed = {"bruteforce", "bp1", "bp_osd_lite", "oracle"}
        if decoder_mode not in allowed:
            raise ValueError(
                f"Invalid decoder_mode={decoder_mode!r}. Must be one of {sorted(allowed)}."
            )
        return decoder_mode

    def get_alpha_vector(self) -> np.ndarray:
        """Get alpha vector according to configured alpha mode."""
        if self.alpha_mode == "uniform":
            return uniform_weights(self.ell)
        if self.alpha_mode == "paper":
            return paper_alpha_weights(self.ell)
        return heuristic_alpha_weights(self.B, self.v, self.ell)

    def get_weights(self) -> np.ndarray:
        """Backward-compatible alias for get_alpha_vector()."""
        return self.get_alpha_vector()

    def evaluate_F(self, x: int) -> float:
        """Evaluate true objective F(x)."""
        return self.prob.evaluate(x)

    def evaluate_G_weighted(self, x: int) -> float:
        """Evaluate weighted surrogate objective G(x)."""
        return surrogate_objective(x, self.B, self.v, self.weights, self.n_bits)

    def evaluate_G_unweighted(self, x: int) -> int:
        """Evaluate native unweighted max-XORSAT objective."""
        return surrogate_objective_unweighted(x, self.B, self.v, self.n_bits)

    def evaluate_num_satisfied(self, x: int) -> int:
        """Count satisfied surrogate constraints."""
        return count_satisfied(x, self.B, self.v, self.n_bits)

    def evaluate_G(self, x: int) -> float:
        """Backward-compatible alias for weighted surrogate objective."""
        return self.evaluate_G_weighted(x)

    def run(
        self,
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> dict:
        """Run DQI pipeline and evaluate samples.

        Args:
            n_samples: Number of samples to draw.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with:
                samples: Array of sampled x values.
                F_values: True objective values.
                G_weighted_values: Weighted surrogate objective values.
                G_unweighted_values: Unweighted max-XORSAT objective values.
                satisfied_counts: Number of satisfied constraints.
                G_values: Backward-compatible alias of G_weighted_values.
                best_F: Best true objective found.
                best_x: Configuration achieving best F.
                best_G_weighted: Best weighted surrogate objective found.
                best_G_unweighted: Best unweighted surrogate objective found.
                best_satisfied_count: Best satisfied-constraint count.
                best_G: Backward-compatible alias of best_G_weighted.
                probabilities: Full DQI probability distribution.
                success_prob: Decoding success probability.
        """
        alpha = self.get_alpha_vector()
        coherent_supported = coherent_supported_for_instance(
            m=self.m,
            n=self.n_bits,
            ell=self.ell,
        )
        coherent_exact_small_instance = (
            self.execution_mode == "coherent" and coherent_supported
        )
        alpha_source_label = {
            "uniform": "uniform_l2",
            "paper": "paper_tridiagonal_eigenvector",
            "heuristic": "heuristic_proxy_eigenvector",
        }[self.alpha_mode]
        state_model = (
            "coherent_joint_state"
            if self.execution_mode == "coherent"
            else "mixture_probabilities"
        )

        # Run DQI statevector simulation
        try:
            samples, probabilities, success_prob = run_dqi_statevector(
                B=self.B,
                v=self.v,
                alpha=alpha,
                ell=self.ell,
                n_samples=n_samples,
                seed=seed,
                decoder_mode=self.decoder_mode,
                execution_mode=self.execution_mode,
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"DQI pipeline failed during decode/postselection: {exc}"
            ) from exc

        # Evaluate samples
        F_values = np.array([self.evaluate_F(int(x)) for x in samples])
        G_weighted_values = np.array([self.evaluate_G_weighted(int(x)) for x in samples])
        G_unweighted_values = np.array([self.evaluate_G_unweighted(int(x)) for x in samples])
        satisfied_counts = np.array([self.evaluate_num_satisfied(int(x)) for x in samples])

        # Find best
        best_idx = np.argmax(F_values)

        return {
            'samples': samples,
            'F_values': F_values,
            'G_values': G_weighted_values,  # backward-compatible alias
            'G_weighted_values': G_weighted_values,
            'G_unweighted_values': G_unweighted_values,
            'satisfied_counts': satisfied_counts,
            'best_F': float(F_values[best_idx]),
            'best_x': int(samples[best_idx]),
            'best_G': float(np.max(G_weighted_values)),  # backward-compatible alias
            'best_G_weighted': float(np.max(G_weighted_values)),
            'best_G_unweighted': int(np.max(G_unweighted_values)),
            'best_satisfied_count': int(np.max(satisfied_counts)),
            'probabilities': probabilities,
            'alpha': alpha,
            'alpha_mode': self.alpha_mode,
            'alpha_vector': alpha,
            'alpha_source_label': alpha_source_label,
            'alpha_norm': float(np.linalg.norm(alpha)),
            'alpha_is_paper_derived': self.alpha_mode == "paper",
            'execution_mode': self.execution_mode,
            'coherent_supported': coherent_supported,
            'coherent_exact_small_instance': coherent_exact_small_instance,
            'weight_register_bits': weight_register_bits(self.ell),
            'decode_semantics': 'projective_surrogate',
            'state_model': state_model,
            'postselection_kind': 'projective_surrogate',
            'decoder_mode': self.decoder_stats['decoder_mode'],
            'decoder_success_rate': float(self.decoder_stats['decoder_success_rate']),
            'decode_exact_recovery_rate': self.decoder_stats['decode_exact_recovery_rate'],
            'ambiguous_syndrome_rate': self.decoder_stats['ambiguous_syndrome_rate'],
            'flip_count_stats': self.decoder_stats['flip_count_stats'],
            'confidence_stats': self.decoder_stats['confidence_stats'],
            'success_prob': success_prob,
        }

    def run_with_metrics(
        self,
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> dict:
        """Run pipeline with full metrics including ground truth comparison.

        Returns extended results with approximation ratio and other metrics.
        """
        results = self.run(n_samples=n_samples, seed=seed)

        # Ground truth
        x_opt, f_opt = self.prob.brute_force_solve()

        # Surrogate faithfulness
        rho, eta = surrogate_faithfulness(self.prob, k=self.k)

        # Extended metrics
        results.update({
            'f_opt': f_opt,
            'x_opt': x_opt,
            'approximation_ratio': results['best_F'] / f_opt if f_opt > 0 else 0,
            'spearman_rho': rho,
            'spearman_rho_weighted': rho,  # explicit alias: rho(F, G_weighted)
            'energy_fraction': eta,
            'unique_samples': len(np.unique(results['samples'])),
        })

        return results

    def summary(self, results: dict) -> str:
        """Generate human-readable summary of results."""
        weight_mode = {
            "uniform": "uniform weights",
            "paper": "paper-derived tridiagonal eigenvector weights",
            "heuristic": "heuristic proxy eigenvector weights",
        }[self.alpha_mode]
        weight_note = (
            "Weight note: paper mode uses the finite tridiagonal matrix eigenvector."
            if self.alpha_mode == "paper"
            else "Weight note: non-paper modes are benchmark baselines."
        )
        lines = [
            "=" * 60,
            "DQI Pipeline Results",
            "=" * 60,
            f"Problem: {self.n_bits} bits, {self.m} surrogate terms, ell={self.ell}",
            f"Weight mode: {weight_mode}",
            f"Decoder mode: {results.get('decoder_mode', self.decoder_mode)}",
            weight_note,
            f"Samples: {len(results['samples'])} ({results.get('unique_samples', '?')} unique)",
            f"Decoding success prob: {results.get('success_prob', '?'):.4f}",
            "",
            f"Ground truth F*: {results.get('f_opt', '?'):.2f}",
            f"Best F found:    {results['best_F']:.2f}",
            f"Approx ratio:    {results.get('approximation_ratio', '?'):.4f}",
            "",
            f"Best G (weighted):   {results.get('best_G_weighted', results.get('best_G', '?')):.2f}",
            f"Best G (unweighted): {results.get('best_G_unweighted', '?')}",
            f"Best satisfied:      {results.get('best_satisfied_count', '?')}/{self.m}",
            f"Spearman rho(F,G_w): {results.get('spearman_rho_weighted', results.get('spearman_rho', '?')):.4f}",
            f"Energy eta:        {results.get('energy_fraction', '?'):.4f}",
            "",
            f"Best config:     {format(results['best_x'], f'0{self.n_bits}b')}",
            "=" * 60,
        ]
        return "\n".join(lines)
