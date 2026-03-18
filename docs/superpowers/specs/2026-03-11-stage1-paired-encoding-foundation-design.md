# Stage 1 Paired Encoding Foundation Design

**Date:** 2026-03-11
**Status:** Approved

## Goal
Add a toy-pricing-specific exact reference encoding path and a shared metrics layer so the repository can compare WHT-truncated and ILP-exact-reference encodings on the same frozen pricing instances.

## Scope
In scope:
- Toy pricing family only (mutually exclusive feature tiers, hard constraints, simple revenue terms, existing penalties, optional small pairwise interactions if already present).
- Canonical toy-model representation for encoding reference.
- Exact full-spectrum parity export over existing bit variables.
- Shared decision metrics API used by comparison and benchmark paths.
- One small paired comparison path for P3/P4/P5.

Out of scope:
- General ILP compiler.
- New broad benchmark matrix expansion.
- Unrelated notebook rewrites unless schema touch requires adapter updates.

## Design Summary
Stage 1 introduces four modules:

1. `src/pricing_ilp.py`
- Provides a canonical, serializable toy-pricing model view derived from frozen pricing instances.
- No solver binding and no hidden exact-enumeration logic.

2. `src/ilp_to_xorsat_exact.py`
- Provides exact full-domain evaluation over toy model assignments.
- Converts exact full objective table to untruncated max-XORSAT-compatible artifact using existing WHT primitives.
- Explicitly documented as toy-family-only reference path.

3. `src/metrics_decision.py`
- Provides one stable metrics entrypoint for decision-quality reporting.
- Returns all expected keys, with unavailable metrics represented as `None`.

4. `src/encoding_compare.py`
- Runs paired backend comparison with all non-encoding knobs fixed.
- Returns JSON-serializable report with compatibility checks, artifact summaries, metrics, and signed deltas.

## Data Contracts

### Canonical Toy Model (`src/pricing_ilp.py`)
Primary function:
- `build_toy_pricing_ilp_model(prob, instance_id=None) -> dict`

Required fields:
- `schema_version`
- `generator_version` or `source_code_version` (best effort)
- `instance_id`
- `instance_hash` (canonical hash)
- `n_features`
- `n_bits`
- `feature_order`
- `bit_order` (`"msb_first"`)
- `bitstring_convention` (`"feature_major_msb_first"`)
- `codeword_mapping` (`{0: "standard", 1: "premium", 2: "luxury", 3: "invalid"}`)
- `valid_codewords` (`[0, 1, 2]`)
- `invalid_codewords` (`[3]`)
- `segments`
- `bundle_constraints`
- `max_luxury`
- `penalties` (`invalid`, `bundle`, `luxury_cap`)
- `objective_formula_label`
- `objective_semantics`

### Exact Table Evaluation (`src/ilp_to_xorsat_exact.py`)
Primary function:
- `evaluate_toy_pricing_model_exact(ilp_model) -> dict`

Required fields:
- `schema_version`
- `source_instance_id`
- `source_model_hash`
- `n_bits`
- `n_states`
- `state_enumeration_order` (`"integer_ascending"`)
- `F_table` (full objective values on complete assignment domain)
- `F_table_dtype`
- `optimizer_indices`
- `F_star`
- `constant_offset`
- `constant_offset_included_in_F_table` (`True`)

Invariant:
- `F_table[x]` always uses the same integer-to-bit interpretation defined by `bit_order` and `bitstring_convention`.

### Exact Reference Artifact (`src/ilp_to_xorsat_exact.py`)
Primary function:
- `build_ilp_exact_reference_artifact(ilp_model) -> dict`

Required fields:
- `schema_version`
- `encoding_type` (`"ilp_exact_reference"`)
- `encoding_label`
- `reference_path` (`"model_table_full_wht"`)
- `spectrum_source` (`"full_exact_wht_of_F_table"`)
- `exact_full_spectrum` (`True`)
- `exact_reconstructable` (`True`)
- `source_instance_id`
- `source_model_hash`
- `artifact_hash` (canonical hash, timestamp-independent)
- `objective_semantics`
- `bit_order`
- `bitstring_convention`
- `feature_order`
- `term_indexing_convention`
- `parity_semantics`
- `weight_sign_convention`
- `n_bits`
- `m_terms`
- `constant_offset`
- `B`
- `v`
- `term_weights`

Docstring requirements:
- State clearly this is a toy-pricing-family reference encoder, not a general ILP compiler.
- Include exact reconstruction statement:
  from `(B, v, term_weights, constant_offset)` the full objective table is reconstructable over the full assignment domain, within numeric tolerance.

## Compatibility Predicate
Two artifacts are `artifacts_compatible_for_paired_run=True` iff all are true:
- identical `source_instance_id` and `instance_hash` (or model hash lineage),
- identical `codeword_mapping`,
- identical `bit_order` and `bitstring_convention`,
- identical `objective_semantics`,
- same decoder mode and candidate-generation interface for compared runs,
- same alpha mode, execution mode, evaluation budget, and RNG seed policy where stochastic sampling exists.

If any check fails, comparison report must include:
- `artifacts_compatible_for_paired_run=False`
- explicit `compatibility_notes`.

## Hashing Rules
Canonical hashes (`instance_hash`, `source_model_hash`, `artifact_hash`) are computed from canonical JSON payloads with:
- sorted keys,
- explicit `schema_version` included,
- no incidental runtime fields (timestamps, wall-clock durations),
- canonicalized numeric representation for floating values before hashing (stable precision/serialization rule defined in implementation).

## Exactness Invariant
`validate_exact_reconstruction(ilp_model, atol, rtol)` must validate full-domain exactness:
1. Build full `F_table` on all assignments.
2. Build exact artifact.
3. Reconstruct full `F_table_reconstructed` from artifact plus `constant_offset`.
4. Assert full-table equality within tolerance.

This validation is over complete domain, not sampled subsets.

## Shared Metrics Contract (`src/metrics_decision.py`)
Primary function:
- `compute_decision_metrics(...) -> dict`

Top-level shape:
- `metrics_schema_version`
- `core_decision`
- `faithfulness`
- `pipeline`
- `timing`
- `metadata`
- `metric_goal`
- `metric_availability`

Required metrics (always present, values may be `None` when unavailable):
- `best_sampled_F`
- `best_sampled_G`
- `top1_regret`
- `topk_regret`
- `optimum_recall_at_k`
- `precision_at_k_high_F`
- `best_top_G_sampled_F`
- `surrogate_best_vs_true_best_sampled_gap`
- `spearman_rho_F_G`
- `retained_energy_eta`
- `decision_distortion`
- `decode_success`
- `postselection_success`

Required metadata:
- `k_eval`
- `sample_count`
- `unique_sample_count`
- `candidate_source_label`

Metric direction map (`metric_goal`) required for interpretability, e.g.:
- `"best_sampled_F": "higher_is_better"`
- `"top1_regret": "lower_is_better"`

Metric availability notes (`metric_availability`) distinguish:
- `not_applicable`
- `not_computed`
- `insufficient_samples`
- `mode_incompatible`

## Comparison Runner Contract (`src/encoding_compare.py`)
Primary function:
- `run_paired_encoding_comparison(...) -> dict`

Required report fields:
- `comparison_id`
- `comparison_metadata`
- `instance_metadata`
- `backend_a_name`
- `backend_b_name`
- `backend_a_artifact_summary`
- `backend_b_artifact_summary`
- `artifacts_compatible_for_paired_run`
- `compatibility_notes`
- `backend_a_metrics`
- `backend_b_metrics`
- `metric_deltas`

Pre-run artifact summaries include:
- `n_bits`
- `m_terms`
- row-weight/density structure summaries
- `retained_energy_eta` (where applicable)
- `exact_full_spectrum`
- `constant_offset`
- `artifact_hash`

Delta naming is explicit and signed, for example:
- `delta_best_sampled_F_a_minus_b`
- `delta_top1_regret_a_minus_b`

## Benchmark Integration (Stage 1 Minimal)
- Add one optional paired path for P3/P4/P5 only.
- Keep existing benchmark matrix unchanged.
- Reuse `metrics_decision` instead of duplicating metric logic.
- Add new optional result section; do not replace existing keys/contracts.

## Testing Requirements

### New tests
1. `tests/test_ilp_exact_reference.py`
- toy model build on frozen pricing family,
- exact artifact generation path runs,
- full-domain exact reconstruction invariant passes.

2. `tests/test_metrics_decision.py`
- metrics payload contains required grouped sections and keys,
- unavailable metrics are present as `None`,
- metric goal and availability maps are present.

3. `tests/test_encoding_compare.py`
- paired WHT vs ILP exact report on same instance/settings,
- JSON-serializable output,
- schema-compatible backend metrics and explicit deltas.

### Additional stability test
4. Hash determinism test (in ILP exact test module):
- build exact artifact twice from same frozen instance,
- assert identical canonical hash,
- assert timestamp-free metadata is hash-stable.

## Acceptance Criteria
- Repository can generate both WHT-truncated and ILP-exact-reference encodings for one frozen pricing instance.
- ILP-exact-reference path is explicit toy-family-only and not presented as general compiler.
- Shared metrics panel exists and is reused by paired comparison path.
- Comparison reports are paired, schema-compatible, and JSON-serializable.
- Exact reconstruction and hash stability are test-backed.
