# Notebook 06 Reader-Only Panel Separation Design

**Date:** 2026-03-12
**Status:** Approved for planning
**Scope:** `notebooks/06_resource_analysis.ipynb` reader-only refactor with strict same-model vs cross-model panel separation.

---

## 1. Decision

Notebook 06 remains a reader-only notebook and will enforce strict panel contracts:

- Main decoder-behavior panel is **same-model authoritative** and uses only `gain_over_bp1`.
- Cross-model comparison is moved to a **separate diagnostic panel** using only `gain_over_bruteforce`.
- No fallback from same-model to cross-model metric in the same panel.

This removes misleading mixed semantics while preserving section structure.

---

## 2. Goals

1. Remove any remaining generator/writer semantics from Notebook 06.
2. Load only canonical artifacts (`resources_master`, `metrics_master`, `run_manifest`).
3. Prevent lookup-table resource cost + brute-force mixture success from appearing as one decoder path.
4. Keep panel-level placeholders for missing/empty/unusable slices.

---

## 3. Scope Boundaries

### 3.1 In scope

- Notebook 06 cell logic and markdown labels.
- Panel contracts for structure/cost/decoder behavior.
- Placeholder reason contract for deterministic notebook/test behavior.

### 3.2 Out of scope

- Changes to canonical artifact writers.
- Scientific metric definitions in `src/`.
- Broad notebook-suite restructuring.

---

## 4. Canonical Inputs (Reader-Only)

Notebook 06 reads only:

- `results/resources_master.parquet|json` via helper loader
- `results/metrics_master.parquet|json` via helper loader
- `results/run_manifest.json` via helper loader

Notebook 06 must not write authoritative artifacts (including `../results/resources_scaling.json`).

---

## 5. Root and Loader Contract

- Use shared root detection pattern (`repo_root()`) in notebook bootstrap.
- Use helper loaders only; no notebook-local path probing.
- Optional missing artifacts produce panel placeholders (not notebook-global abort).

---

## 6. Alignment Contract for Decoder/Resource Panels

### 6.1 Aligned identity key definition (frozen)

Alignment starts with this ordered key policy:

1. **Primary key:** `slot_id` (if present in both canonical sources)
2. **Fallback identity key:**
   - minimum required fields:
     - `instance_id`
     - `encoding`
     - `decoder`
     - `alpha_mode`
     - `execution_mode`
   - optional strengthening field:
     - `canonical_trial_seed` (used when present in both)

### 6.2 Fallback-key resolution when required fields are absent

If any fallback minimum field is absent in either source, alignment is invalid for the panel and must render placeholder with reason `missing_required_fields`.

### 6.3 Duplicate identity behavior (frozen)

If either source has duplicate rows over resolved aligned identity:

- render placeholder with reason `duplicate_aligned_identity`
- do not auto-deduplicate
- do not silently select winners

---

## 7. Status Normalization Source (Frozen)

Before alignment/eligibility filtering, panel logic materializes normalized status per source:

- resource source status -> normalized resource status field
- decoder-study source status -> normalized decoder-study status field

Eligibility checks use normalized statuses only (`completed` gate), not raw status literals.

---

## 8. Panel Semantics

### 8.1 Main panel (authoritative same-model)

- Uses only `gain_over_bp1`.
- No fallback to `gain_over_bruteforce`.
- If same-model metric is absent/non-usable, render placeholder.

### 8.2 Cross-model panel (diagnostic only)

- Separate panel uses only `gain_over_bruteforce`.
- Panel title/label must include:
  - **cross-model**
  - **non-authoritative**
  - **not used for claims**

---

## 9. Usable-Row Contract

A row is usable for a given panel only if all are true:

1. Row aligns under Section 6 key policy.
2. Both source statuses normalize to `completed`.
3. Required decoder-study/resource fields for the panel are present.
4. Panel metric value is numeric and finite:
   - main panel: `gain_over_bp1`
   - diagnostic panel: `gain_over_bruteforce`

If no usable rows remain, render placeholder.

---

## 10. Placeholder Reason Contract (Frozen Enum)

Panel placeholders use a closed reason-code set:

- `missing_artifact`
- `empty_aligned_set`
- `missing_required_fields`
- `non_usable_metric_values`
- `duplicate_aligned_identity`

Notebook may include human-readable `reason/details`, but reason-code semantics must map to this enum for stable smoke tests.

---

## 11. Section Structure Preservation

Notebook 06 section order and narrative remain mostly intact. Changes are constrained to:

- stricter panel contracts
- explicit same-model vs cross-model labeling
- placeholder reason normalization

---

## 12. Acceptance Criteria

1. Notebook 06 no longer authors authoritative artifacts.
2. Main structure-vs-decode panel uses only `gain_over_bp1` and never falls back.
3. Cross-model `gain_over_bruteforce` appears only in a separately labeled diagnostic panel.
4. Duplicate aligned identities in either source produce placeholder (no auto-dedupe).
5. Missing same-model metric produces placeholder for main panel.
6. Notebook executes with panel-level placeholders when optional slices are unavailable.
