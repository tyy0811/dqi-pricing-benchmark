# Coherent Execution Mode and Benchmark Diagnostics Design

**Date:** 2026-03-10
**Status:** Approved

## Goal
Add a tiny-instance coherent reference path alongside the existing branch-mixture DQI statevector path, and extend benchmark/notebook diagnostics so reviewers can directly assess:
- uniform vs paper alpha
- heuristic vs paper alpha
- mixture vs coherent execution mode (tiny instances only)

## Scope
In scope:
- `src/dqi_state.py`: coherent execution path, execution selector, coherent support helpers
- `src/dqi_pipeline.py`: execution mode threading, run metadata, TVD helper
- `src/benchmark.py`: explicit mixture keys, optional coherent paper run, pairwise diagnostics, stable unsupported payload
- `tests/test_dqi_coherent_vs_mixture.py`: coherent-vs-mixture regression tests
- `tests/test_benchmark.py`: benchmark schema and diagnostics checks
- `notebooks/03_paper_faithful_dqi.ipynb`: explicit tiny coherent check section
- `notebooks/05_decoder_comparison.ipynb`: paper-alpha decoder comparison with coherent optional/tiny-only framing
- `notebooks/_helpers.py`: only if schema formatting updates are required

Out of scope:
- Fully reversible/physical coherent decoder simulation
- Coherent execution for medium/large benchmark instances

## Scientific and Modeling Contract

### Execution Modes
- `execution_mode="mixture"`: existing fast path using `|alpha_t|^2` convex mixture over fixed-`t` branches
- `execution_mode="coherent"`: explicit joint state simulation for tiny instances in the repo's simplified surrogate model

### Coherent Register Contract
Coherent path simulates `|t>|e>|s>`:
- `t`: weight register (`weight_register_bits = ceil(log2(ell+1))`)
- `e`: error-pattern register (`m` bits)
- `s`: syndrome/output register (`n` bits)

Final candidate distribution is over `x`, where `x` is the final computational basis value of register `s` after final Hadamard.

### Decode/Uncompute Semantics
Coherent mode decode/uncompute is explicitly a projective/postselected surrogate step (non-unitary), aligned with existing simplified formalism.

Required metadata:
- `decode_semantics = "projective_surrogate"`
- `postselection_kind = "projective_surrogate"`
- `state_model = "coherent_joint_state"` or `"mixture_probabilities"`

### Coherent Support Bound
Support is determined by actual simulated joint space size:
- `joint_num_bits_coherent = weight_register_bits + m + n`
- `joint_state_size_coherent = 2 ** joint_num_bits_coherent`
- coherent supported iff `joint_state_size_coherent <= 4096` and coherent assumptions hold

No implicit fallback from coherent to mixture.

## Payload and Schema Contract

### Per-run payload additions
All DQI run payloads include:
- `execution_mode`
- `coherent_supported`
- `coherent_exact_small_instance`
- `weight_register_bits`
- `decode_semantics`
- `state_model`
- `postselection_kind`

### Unsupported coherent payload (benchmark)
If coherent run is attempted but unsupported, benchmark stores a stable payload with:
- always: `status`, `reason`, `execution_mode`, `coherent_supported`, `coherent_exact_small_instance`, `coherent_comparison_attempted`
- attempted config/context: `alpha_mode`, `k`, `ell`, `n_bits`, `m`, `joint_num_bits`, `joint_state_size`, `cap_limit`, `weight_register_bits`, `decoder_mode`, `alpha_vector`

## Benchmark Output Contract
For same `(k, ell)` runs:
- `dqi_uniform_mixture`
- `dqi_paper_mixture`
- `dqi_heuristic_mixture`
- `dqi_paper_coherent` (or unsupported payload)

Backward-compatible aliases remain:
- `dqi_uniform -> dqi_uniform_mixture`
- `dqi_paper -> dqi_paper_mixture`
- `dqi_heuristic -> dqi_heuristic_mixture`

Benchmark-level diagnostics:
- `delta_best_F_paper_vs_heuristic`
- `delta_success_prob_paper_vs_heuristic`
- `delta_best_F_paper_mixture_vs_coherent`
- `tvd_paper_mixture_vs_coherent` (final candidate distribution over `x`)
- `same_top_sample_flag_paper_mixture_vs_coherent`
- `coherent_comparison_attempted`

`same_top_sample_flag_paper_mixture_vs_coherent` definition:
- same best sampled candidate by true `F`
- ties broken by weighted `G`
- then lexicographic `x`

## Notebook Requirements

### Notebook 03
Add section title exactly:
- **Alpha ablation and coherent-vs-mixture check (tiny-instance exact surrogate model)**

Show side-by-side comparison for:
- uniform mixture
- paper mixture
- heuristic mixture
- paper coherent (when supported)

### Notebook 05
Add decoder comparison under at least paper-alpha mode.
Report stability of decoder conclusions across alpha modes.
Coherent mode must be clearly optional and tiny-instance only; do not imply coherent validation for larger frozen instances.

## Testing Requirements

### New tests
`tests/test_dqi_coherent_vs_mixture.py`:
- selector path difference (coherent not same code path as mixture)
- tiny supported coherent run returns normalized candidate distribution
- coherent phase sensitivity on crafted tiny case (reject fake mixture-equivalent coherent path)
- oversize coherent run raises clear bounded-size error
- TVD helper correctness: zero for identical distributions, bounded in `[0,1]`

### Updated tests
`tests/test_benchmark.py`:
- assert new run keys and alias consistency
- assert coherent run or stable unsupported schema
- assert new diagnostics keys and type/None behavior
- assert metadata consistency: `coherent_supported`, `coherent_exact_small_instance`, `coherent_comparison_attempted`

## Acceptance Criteria
- Repo has a truthful tiny-instance coherent reference implementation.
- Existing mixture path remains default and unchanged for main benchmark speed.
- Benchmark answers reviewer question directly:
  - whether alpha ablation is paper-derived
  - how mixture differs from coherent on tiny instances
- Outputs are reviewer-safe and explicit about surrogate/coherent semantics limits.
