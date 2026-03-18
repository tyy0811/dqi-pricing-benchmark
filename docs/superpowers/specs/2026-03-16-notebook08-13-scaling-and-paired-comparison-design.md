# Notebook 08 and 13 Scaling Design

**Date:** 2026-03-16  
**Status:** Approved for planning  
**Scope:** Design only; no implementation in this document.

---

## 1. Decision

This phase updates the reviewer-facing pricing notebooks using the current canonical benchmark artifacts only.

- `notebooks/08_paired_encoding_comparison.ipynb` becomes a strict canonical paired-comparison reader over `results/metrics_master.json`
- `notebooks/13_scaling_analysis.ipynb` is added as a reviewer-facing scaling notebook spanning P3 through P7
- the current canonical master is the ground truth for execution outcomes, coverage, and boundary rows
- frozen artifacts remain preferred inputs wherever available; recomputation is fallback-only and must be explicit

This phase continues from the current validated branch state and does not revisit earlier phases.

---

## 2. Non-Goals

This phase does **not** include:

- changes to canonical completed values for P3, P4, or P5
- new benchmark matrix integration work
- refactoring shared helper layers unless implementation is concretely blocked
- schema redesign for `metrics_master`
- silent dropping of unavailable or failed rows from scaling-boundary analysis
- claims about P7 execution success if canonical completed rows do not exist

---

## 3. Canonical Source of Truth

### 3.1 Execution and coverage truth

`results/metrics_master.json` is the authoritative source for:

- completed vs unavailable vs failed execution status
- paired-comparison eligibility in Notebook 08
- completed decoder/timing coverage in Notebook 13
- tractability-boundary rows in Notebook 13

No notebook logic may assume older P3 through P5 row counts or old status mixes.

### 3.2 Artifact-first reader policy

Reviewer-facing notebooks prefer frozen artifacts over recomputation.

- Notebook 08 should derive matched pricing comparisons directly from the canonical master
- Notebook 13 Section 1 should use frozen CP-SAT metadata when available
- Notebook 13 Section 2 should use frozen paired or truth-table artifacts for encoding diagnostics
- Notebook 13 Section 4 should use analytic estimation from `src/resources.py`

Recomputation is allowed only where explicitly approved in this design.

---

## 4. Notebook 08 Design

### 4.1 Role

Notebook 08 remains a reviewer-facing paired-encoding notebook, but all reviewer-visible counts, tables, and verdict numbers must come from the current canonical master rather than hard-coded pricing instance assumptions or stale snapshot counts.

### 4.2 Canonical filter

The paired candidate slice must be restricted to:

- pricing-family rows only
- the intended reviewer-facing execution slice, `execution_mode == "mixture"` via normalized execution-mode handling
- `run_status == "completed"`

Unavailable, failed, or boundary-only rows must not enter the matched paired comparison.

### 4.3 Matching contract

Notebook 08 must define one explicit pairing contract for `wht_truncated` vs `ilp_derived`.

Primary key:

- use `comparison_key` only if:
  - the column exists
  - values are non-null within the candidate slice
  - values are unique within the candidate slice for each encoding side

Fallback key:

- if `comparison_key` is unsafe, fall back to the smallest deterministic composite key available from:
  - `instance_id`
  - `decoder`
  - `alpha_mode`
  - `execution_mode_norm`
  - `ell`
  - `stage`
  - `family`
  - `canonical_trial_seed`

The notebook must normalize the relevant columns once before pairing and restrict the final key to columns actually present.

### 4.4 Duplicate and ambiguity policy

Before merge, the notebook must assert or report that each encoding contributes at most one completed row per final pairing key.

If either side has duplicate completed rows for the final key:

- do not collapse or arbitrarily choose rows
- emit an explicit unavailable or audit table
- skip paired-delta computation for that slice

If `comparison_key` is rejected, the notebook should show the rejection reason, such as:

- missing values detected
- duplicates detected
- non-unique within the candidate slice

### 4.5 Pairing audit panel

Before the paired delta table, Notebook 08 should render a visible pairing-audit panel showing:

- completed pricing rows considered
- candidate WHT rows
- candidate ILP rows
- matched pairs
- dropped unmatched WHT rows
- dropped unmatched ILP rows
- active key strategy and any fallback reason

This panel is part of the reviewer-facing transparency contract.

### 4.6 Dynamic outputs

The notebook must remove hard-coded assumptions such as:

- explicit pricing-instance lists like `["P3", "P4", "P5"]`
- fixed matched-record totals

The notebook must recompute dynamically from canonical rows:

- matched-record count
- summary statistics
- paired delta tables
- inline verdict numbers and wording

P6 must be included automatically if matched completed rows exist.

P7 contributes only if completed matched rows exist for both encodings. If no such rows exist, P7 is excluded from the paired benchmark and the notebook must include this explicit note near the end:

`P7 (n=14) is excluded from this paired comparison because no completed matched DQI rows are present in the current canonical master; see Notebook 13 for encoding diagnostics and tractability/resource analysis at n=14.`

### 4.7 Failure model

Notebook 08 must fail soft, not fail opaque.

If pairing is unavailable or ambiguous, the notebook should still render reviewer-facing audit or placeholder outputs with clear reasons rather than raising unhelpful execution errors or silently emitting misleading comparisons.

---

## 5. Notebook 13 Design

### 5.1 Role

Notebook 13 is a reviewer-facing scaling notebook for pricing instances P3 through P7 that keeps execution facts, encoding diagnostics, and analytic estimates in clearly separated provenance layers.

### 5.2 Section 1: Instance summary

Section 1 reads all pricing-family frozen instance JSONs for P3 through P7 and displays:

- Instance
- Features
- n (bits)
- Total configs
- Feasible
- Feasible %
- F*
- CP-SAT status
- CP-SAT time (s)
- Model type
- Variables

CP-SAT provenance rule:

- use frozen CP-SAT metadata whenever available
- if any required field is missing, recompute only the missing CP-SAT fields in a deterministic lightweight fallback path
- never overwrite frozen fields that already exist
- if fallback recomputation occurs, emit a short provenance note

### 5.3 Section 2: Encoding quality vs scale

Section 2 uses frozen paired or truth-table diagnostics, not DQI completion status, to compare `wht_truncated` and `ilp_derived` at canonical `m=15`.

Required table columns:

- Instance
- n
- Encoding
- Spearman rho(F,G)
- Retained energy eta
- Decision distortion
- Top-1 regret
- Top-k regret

Plot policy:

- the main `rho vs n` and `eta vs n` plots use only canonical `m=15` points
- if a secondary P7 `m=20` artifact exists, include it only as a clearly labeled diagnostic row in the section table
- do not place the P7 `m=20` diagnostic point on the main scaling plots

The markdown should explicitly state that:

- execution coverage comes from the canonical master
- encoding diagnostics come from frozen paired or truth-table artifacts

### 5.4 Section 3: Decoder scaling

Section 3 uses only canonical pricing-family rows with `run_status == "completed"` and compares exact-statevector decoder behavior.

Primary comparison:

- `bp1` vs `bruteforce`

Per-instance selection rule:

- prefer `ell=2` if completed rows exist for both decoders
- otherwise fall back to `ell=1`
- label fallback instances explicitly

Required table columns:

- Instance
- n
- ell
- Decoder
- Postselection success
- Best F
- Top-1 regret
- Wall-clock (s)

Legacy rows lacking `wall_clock_s` must show a readable placeholder like `—`.

Coverage text must distinguish:

- completed comparison instances
- instances with no completed DQI rows available
- slices that are not applicable

### 5.5 Section 4: Resource estimation

Section 4 uses analytic resource estimation only and includes all pricing instances P3 through P7 with consistent reference `ell=2`.

Required table columns:

- Instance
- n
- m
- ell
- Total qubits
- CNOTs (syndrome)
- Dicke prep gates
- Estimated depth

If P7 has no executed DQI validation, it must be marked analytic-only in this section.

### 5.6 Section 5: Execution timing and tractability boundary

Section 5 splits execution evidence into completed timing summaries and boundary rows.

Part A uses completed canonical rows with timing data and groups by:

- `instance_id`
- `encoding`
- `decoder`
- `ell`

Required columns:

- Instance
- Encoding
- Decoder
- ell
- Mean wall-clock (s)
- Peak memory (MB)
- Runs

Part B uses canonical rows with `run_status in {"unavailable", "failed"}` and displays:

- Instance
- n
- ell
- Config (encoding/decoder/alpha)
- run_status
- boundary_reason
- estimated_qubits
- estimated_memory_gb

The notebook must not silently drop boundary rows. If P7 has only boundary rows, the notebook must say so explicitly.

### 5.7 Section 6: Scaling verdicts

Section 6 computes four explicit verdict lines from measured notebook values only:

1. WHT encoding quality trend with `n`
2. ILP-derived encoding behavior at the largest available `n`
3. BP1 vs brute-force trend with `n`
4. statevector tractability boundary at the largest attempted scale

Verdicts must be cautious and must reflect actual coverage. If P7 has no completed DQI rows, the tractability verdict must say so directly.

### 5.8 Section 7: Frozen outputs

Notebook 13 must be executed end to end, saved with persisted outputs, and write its generated plots into `results/plots/`.

---

## 6. Validation Contract

Before work is considered complete:

1. execute Notebook 08 end to end and save outputs
2. execute Notebook 13 end to end and save outputs
3. verify expected plots are present in `results/plots/`
4. run `python3 -m pytest tests/ -x`

The final patch summary must report:

- Notebook 08 matched-record count
- whether P6 was included
- whether P7 was excluded from the paired comparison
- whether Notebook 13 includes P7 encoding and resource diagnostics
- whether Notebook 13 boundary analysis includes explicit P7 rows
- final pytest result

---

## 7. Implementation Boundary

Implementation should follow the minimal-change path:

- update Notebook 08 in place
- create Notebook 13
- reuse existing notebook/helper patterns where possible
- avoid unrelated refactors

New shared helper work is out of scope unless a concrete blocker makes it necessary to complete Notebook 08 or 13 correctly.
