# Artifact Index

Last validation pass: 2026-03-11

## Matrix Outputs

- Matrix A
  - Per-run JSON: `results/matrix_a/runs/`
  - Manifest: `results/matrix_a/manifest.json`
  - Aggregate table: `results/tables/matrix_a_summary.csv`
  - Aggregate report: `results/reports/matrix_a_report.json`
- Matrix B
  - Per-run JSON: `results/matrix_b/runs/`
  - Manifest: `results/matrix_b/manifest.json`
  - Aggregate table: `results/tables/matrix_b_summary.csv`
  - Aggregate report: `results/reports/matrix_b_report.json`
- Matrix C
  - Per-run JSON: `results/matrix_c/runs/`
  - Manifest: `results/matrix_c/manifest.json`
  - Aggregate table: `results/tables/matrix_c_summary.csv`
  - Aggregate report: `results/reports/matrix_c_report.json`

## Notebook Outputs

- `notebooks/08a_paired_encoding_comparison.ipynb`
  - One-line interpretation: ILP-vs-WHT paired comparisons are available on P3/P4/P5 with explicit unavailable annotations on unsupported metric slices.
- `notebooks/08b_coherent_validation.ipynb`
  - One-line interpretation: Coherent execution validation for Family C instances (C1, C2, C3).
- `notebooks/09_decoder_phase_diagram.ipynb`
  - One-line interpretation: Decoder performance varies by structure regime, with unavailable/failed decoder slices retained visibly rather than dropped.
- `notebooks/10_alpha_finite_size_validation.ipynb`
  - One-line interpretation: Coherent-vs-mixture alpha checks on C1/C2 are largely consistent where supported, while C3 coherent failures remain explicit.
- `notebooks/11_decision_faithfulness_dashboard.ipynb`
  - One-line interpretation: Global faithfulness metrics and decision metrics are presented together, including cases where high rho coexists with non-trivial regret.
- `notebooks/12_quality_vs_cost.ipynb`
  - One-line interpretation: Stage-2 quality joined with Stage-3 resource/structure fields supports a compact Pareto-style quality-vs-cost review.

## Key Result Files

- `results/tables/matrix_a_summary.csv`
- `results/tables/matrix_b_summary.csv`
- `results/tables/matrix_c_summary.csv`
- `results/reports/matrix_a_report.json`
- `results/reports/matrix_b_report.json`
- `results/reports/matrix_c_report.json`
- `results/quality_vs_cost_smoke.json` (optional prior quality-vs-cost report)

## Public Caveat Notes

- Some run directories contain additional historical JSON files beyond the current aggregate report `record_count`; aggregate tables/reports should be treated as canonical for this stack.
- Resource totals are primarily model-based estimates (`analytic_upper_bound` / `implementation_proxy`) and should not be presented as hardware-validated counts.
- Structure fields are unavailable outside structure-bearing runs and are explicitly marked unavailable in notebook outputs.
