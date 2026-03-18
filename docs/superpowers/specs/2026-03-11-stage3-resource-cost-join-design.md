# Stage 3 Resource-Cost Join Design

**Date:** 2026-03-11
**Status:** Approved
**Scope:** Stage 3 support for Stage 2 paired benchmark only

## Objective
Add only the resource/cost machinery required to support Stage 2 comparisons:
- encoding-aware resource summaries
- decoder-aware resource estimates for implemented/plausible decoder models only
- coherent alpha overhead on tiny instances
- one joined quality-vs-cost report embedded in benchmark output

Out of scope:
- full reversible-circuit implementations for all decoders
- automatic artifact writing during normal benchmark runs
- broad benchmark matrix expansion

## Architecture
- Add `src/resources_compare.py` as a pure join layer.
- Keep `src/resources.py` as source of resource estimates and confidence/status labels.
- Keep `src/metrics_decision.py` as source of decision/faithfulness metrics.
- Keep `run_benchmark(...)` thin:
  - `results["quality_vs_cost"] = build_quality_vs_cost_summary(results)`
- Embedded `results["quality_vs_cost"]` is canonical.
- Optional manual export helper is allowed but non-canonical:
  - `export_quality_cost_summary(results, out_path)`

## Data Contract: `results["quality_vs_cost"]`

```python
{
  "schema_version": "1",
  "generated_from_benchmark_version": "unknown_or_commit",
  "row_count": int,
  "objectives": {
    "maximize": ["quality.best_F_over_F_star"],
    "minimize": ["resources.total_estimated_resources.total_gate_est_with_decode"]
  },
  "rows": [ ... ],
  "pareto": {
    "objective": {
      "maximize": ["quality.best_F_over_F_star"],
      "minimize": ["resources.total_estimated_resources.total_gate_est_with_decode"]
    },
    "row_ids": [ ... ]
  },
  "notes": [ ... ]
}
```

### Row Contract
Each row must be JSON-serializable and include all required blocks/keys (use `None` for unavailable values; do not omit keys):

```python
{
  "row_id": str,
  "row_status": "complete" | "missing_resources" | "missing_quality" | "skipped",
  "instance_id": str,
  "instance_family": str,
  "encoding_artifact_id": str,
  "run_label": str,
  "provenance": {
    "encoding_backend": str,
    "decoder": str,
    "decoder_resource_model": str,
    "alpha_mode": str,
    "execution_mode": str,
  },
  "quality": {
    "best_F": float | None,
    "F_star": float | None,
    "best_F_over_F_star": float | None,
    "best_G": float | None,
    "top1_regret": float | None,
    "topk_regret": float | None,
    "optimum_recall_at_k": float | None,
    "best_top_G_sampled_F": float | None,
    "decision_distortion": float | None,
    "decode_success": float | None,
    "postselection_success": float | None,
  },
  "faithfulness": {
    "spearman_rho_F_G": float | None,
    "retained_energy_eta": float | None,
  },
  "collision_structure": {
    "n": int | None,
    "m": int | None,
    "rank": int | None,
    "rank_deficiency": int | None,
    "row_weight_mean": float | None,
    "row_weight_max": int | None,
    "col_weight_mean": float | None,
    "col_weight_max": int | None,
    "syndrome_density": float | None,
    "ambiguous_syndrome_count": int | None,
    "collision_rate": float | None,
    "decodable_syndrome_fraction": float | None,
  },
  "resources": {
    "implemented_resources": dict,
    "analytic_estimates": dict,
    "total_estimated_resources": dict,
    "coherent_alpha_overhead": {
      "status": str,
      "weight_register_qubits": int | None,
      "coherent_prep_overhead": int | None,
      "controlled_sector_overhead": int | None,
    },
    "resource_status": {
      "decode_included": bool,
      "decode_model_status": str,
      "weight_register_status": str,
      "transpiled_subset_available": bool,
    },
  },
  "status": {
    "decode_confidence": str | None,
    "coherent_supported": bool | None,
  },
}
```

## Determinism Rules
1. Row inclusion:
- skip missing run sections silently
- include only rows with both `quality` and `resources` blocks populated (with keys present)
- mark row with `row_status`

2. Deterministic `row_id`:
- stable concatenation of:
  - `instance_id`
  - `encoding_artifact_id`
  - `decoder`
  - `alpha_mode`
  - `execution_mode`
  - `run_label`

3. Pareto semantics:
- API accepts structured objectives:
  - `maximize_keys: list[str]`
  - `minimize_keys: list[str]`
- rows with missing objective values are excluded from Pareto evaluation
- ties are retained
- output `pareto.row_ids` sorted deterministically by `row_id`

## Decoder Resource Modeling Scope
Implemented/allowed:
- bounded-distance lookup (`lookup_table_reversible`)
- BP1 classical-oracle estimate (`bp1_classical_oracle_estimate`)
- optional GJE only if naturally present in existing codebase

Intentionally omitted:
- reversible BP+OSD-lite circuit model
- claims of coherent/reversible implementation for BP+OSD-lite

## Integration Plan
1. Create `src/resources_compare.py`:
- `build_quality_vs_cost_summary(results: dict) -> dict`
- deterministic row builder helpers
- structured Pareto helper with deterministic ordering
- optional `export_quality_cost_summary(results, out_path)`

2. Update `src/benchmark.py`:
- import join helper
- set `results["quality_vs_cost"]` after quality/resource sections are built
- no automatic file output

3. Update `src/structure_metrics.py` only if required for missing fields:
- add explicit aliases/calculations needed by row schema (`rank`, `rank_deficiency`, `decodable_syndrome_fraction`)
- keep existing fields backward compatible

4. Notebook alignment:
- notebook/report helpers consume embedded `results["quality_vs_cost"]`
- no dependency on a second auto-generated artifact

## Tests
Add tests covering:
1. `quality_vs_cost` section exists in benchmark payload and is JSON-serializable.
2. every row includes required keys/blocks; required quality keys always present (nullable).
3. `resources` always contains:
- `implemented_resources`
- `analytic_estimates`
- `total_estimated_resources`
- `coherent_alpha_overhead`
- `resource_status`
4. decode-inclusive totals are never less than decode-exclusive totals.
5. resource outputs include confidence/status labels.
6. structured Pareto objective is present and valid.
7. deterministic `row_id` ordering for identical input.
8. missing-run resilience: absent optional run section does not fail summary generation.

## Acceptance Criteria
- Stage 3 supports Stage 2 paired benchmark with explicit implemented-vs-estimated labels.
- `run_benchmark(...)` includes canonical embedded `quality_vs_cost` summary.
- no automatic side-effect file writes in normal benchmark runs.
- joined payload is compact and reviewer-safe for notebook/report use.
