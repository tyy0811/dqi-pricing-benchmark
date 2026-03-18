# Notebook 04/05 Repositioning Design

**Date:** 2026-03-10
**Status:** Approved (staged review complete)

## Goal
Reposition the repository around two explicit stories:
- BMW-fit pricing workflow: pricing instance -> exact CP-SAT solve -> surrogate artifact handoff to DQI.
- Paper-faithful DQI microscope: fixed DQI instances -> decoder sensitivity -> postselection and best-`F` effects.

## Scope
This design cycle covers:
- Rename and repurpose notebook 04/05 to new roles.
- Add pricing model and BP1 decoder modules.
- Realign reduction and baseline interfaces to a frozen contract.
- Split weights into paper/heuristic modules with compatibility shim.

Out of scope for this cycle:
- Full `06_resource_analysis`, `07_qiskit_parity`, `08_noise_appendix` implementation.

## Frozen Notebook Sequence
1. `notebooks/01_reproduce_toy_dqi.ipynb`
2. `notebooks/02_pricing_instance_demo.ipynb`
3. `notebooks/03_paper_faithful_dqi.ipynb`
4. `notebooks/04_pricing_pipeline.ipynb`
5. `notebooks/05_decoder_comparison.ipynb`
6. `notebooks/06_resource_analysis.ipynb`
7. `notebooks/07_qiskit_parity.ipynb`
8. `notebooks/08_noise_appendix.ipynb`

Legacy files with overlapping purpose (for this scope):
- Replace/rename `notebooks/04_qiskit_comparison.ipynb` -> `notebooks/04_pricing_pipeline.ipynb`
- Replace/rename `notebooks/05_resource_analysis.ipynb` -> `notebooks/05_decoder_comparison.ipynb`

## Architecture and Ownership
- `src/pricing_model_ortools.py`: owns CP-SAT model construction, solve, and solution extraction.
- `src/classical_baselines.py`: owns baseline orchestration and standardized baseline result interfaces.
- `src/reduction.py`: owns exploratory WHT-topK reduction and surrogate artifact creation.
- `src/decoder_bp1.py`: owns deterministic hard-decision bit-flip decoding only.
- `src/weights_paper.py`: paper/demo-derived alpha weight logic.
- `src/weights_heuristic.py`: heuristic alpha weight logic.
- `src/weights.py`: compatibility shim only.

Terminology policy:
- **pricing instance** = original constrained pricing problem.
- **surrogate artifact / DQI instance** = reduced object derived from WHT truncation.

## Data Contracts

### Surrogate Artifact Contract (`src/reduction.py`)
Canonical field names:
- `artifact_version`
- `mode` (e.g. `exploratory_wht_topk`)
- `bit_order` (`msb_first`)
- `n`, `m`, `k_requested`, `k_selected`
- `exclude_constant`
- `B`, `v`, `term_weights`, `indices`, `coefficients`
- `truncation_metadata` (sorting criterion, tie behavior)
- `diagnostics` (retained spectral mass, optional faithfulness metrics)
- `source_metadata` (`instance_id` plus optional reproducibility fields)

Reproducibility metadata (optional but supported):
- `created_by`, `created_at`, `repo_commit`, `random_seed`

Compatibility:
- Existing helper functions remain available.
- Legacy `weights` output is retained only via wrappers; canonical key is `term_weights`.

### Pricing Solve Result Contract (`src/pricing_model_ortools.py`)
Canonical output object:
- `status`
- `objective_value`
- `optimized_objective_value`
- `x_opt`
- `decoded_config`
- `solve_time_s`
- `constraint_summary`
- `instance_id`
- `metadata`

`objective_value` semantics:
- `objective_value` is always the raw pricing objective `F(x_opt)`.
- `optimized_objective_value` is the exact scalar optimized by the solver (can differ if the model uses ranking/scaling transforms).

### Decoder Result Contract (`src/decoder_bruteforce.py`, `src/decoder_bp1.py`)
Both decoders expose the same shape:
- `decoder_name`
- `decoded_error`
- `success`
- `within_radius`
- `distance`
- `num_flips`
- `metadata`

Semantics:
- Expected decode misses must return structured `success=False` results (not ad-hoc exceptions).
- Contract violations or malformed inputs fail loudly.

## Notebook Data Flow

### Notebook 04 (`04_pricing_pipeline.ipynb`)
- Define/load pricing instance.
- Run required CP-SAT solve (hard dependency, fail fast if OR-Tools missing).
- Show standardized pricing solve result and decoded configuration explanation.
- Build/export surrogate artifact for DQI handoff.
- Report key bridge metrics for pricing->DQI transition.

### Notebook 05 (`05_decoder_comparison.ipynb`)
- Load one or more surrogate artifacts.
- Run brute-force and BP1 decoders under shared settings.
- Compare decoder success rates, within-radius behavior, postselection effect, and best-`F` impact.

## Error Handling Rules
- Notebook 04 fails early if `ortools` is unavailable.
- Decoder failures that are expected in normal operation are structured outputs.
- Artifact schema and shape violations raise explicit errors.

## Testing Requirements
- Add surrogate artifact schema tests and wrapper compatibility checks.
- Add round-trip artifact test: build -> serialize -> reload -> equality checks for key fields.
- Add BP1 decoder deterministic behavior and schema tests.
- Extend baseline tests for standardized CP-SAT result schema and objective semantics.
- Add/adjust weights shim tests to confirm forwarding to paper/heuristic modules.

## Risks and Controls
Primary risk: interface drift between pricing model, reduction artifact, notebooks, and baselines.
Control: frozen contracts above plus schema tests and round-trip serialization tests.

## Acceptance Criteria
- Notebook 04/05 filenames and responsibilities match the frozen sequence.
- CP-SAT path is shared between pricing model and baseline orchestration.
- Surrogate artifacts use canonical schema with `term_weights`.
- Decoder comparison uses standardized decoder-result schema including `within_radius`.
- Tests cover schema conformance, compatibility, and round-trip behavior.
