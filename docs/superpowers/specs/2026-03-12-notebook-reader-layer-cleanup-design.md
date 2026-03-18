# Notebook Reader Layer Cleanup Design

**Date:** 2026-03-12
**Status:** Approved for planning
**Scope:** Reader-layer hardening only (helpers + notebook bootstrap/guards). No scientific-method rewrites.

---

## 1. Decision

Adopt a helper-first notebook reader contract:

- `notebooks/_helpers.py` is the single compatibility layer for root detection, loader policy, status normalization, optional-column guards, and placeholder contracts.
- Notebooks `06, 08, 09, 10, 11, 12` remain analysis readers; they stop implementing path/bootstrap and schema drift logic inline.
- Comparable Stage E panels continue to read only `quality_cost_master` for comparable slices.

This keeps notebook structure/markdown mostly intact while removing fragile path and schema assumptions.

---

## 2. In Scope and Out of Scope

### 2.1 In scope

- Standardize notebook bootstrap root resolution.
- Remove notebook-local `sys.path.insert(0, "..")` and cwd/parent probing.
- Freeze one status normalization bridge for old/new producer vocabularies.
- Freeze one loader policy for required vs optional artifacts.
- Freeze one optional-column guard API (numeric/object coercion).
- Freeze one placeholder/unavailable panel schema and usage contract.

### 2.2 Out of scope

- No movement of scientific transforms/ranking/statistical logic into `_helpers.py`.
- No scientific interpretation changes.
- No redesign of notebook narrative structure.

---

## 3. Frozen Root Resolver Contract

### 3.1 Single entrypoint

`repo_root()` is the only notebook-facing root resolver.

Notebook bootstrap pattern is frozen to:

```python
from notebooks._helpers import repo_root
ROOT = repo_root()
```

No notebook-local fallback/probing logic is allowed.

### 3.2 Marker list and search order (frozen)

`repo_root()` probes candidate directories using this ordered marker policy:

- **Required markers:** `src/`, `notebooks/`
- **Optional confidence markers:** `pyproject.toml`, `.git`

Candidate directory traversal order:

1. `Path.cwd().resolve()`
2. `Path(__file__).resolve().parent` (helpers/notebooks context)
3. Parent chain for each start candidate up to filesystem root, de-duplicated in first-seen order

Acceptance rule:

- Candidate is repo root if required markers exist.
- Optional markers are recorded for diagnostics only (not required for acceptance).

### 3.3 Failure mode (frozen)

If unresolved, `repo_root()` raises a clear `RuntimeError` that includes:

- required/optional marker definitions
- full visited candidate directory list
- per-candidate marker check summary

---

## 4. Shared Loader Policy Contract

### 4.1 Single internal policy helper

Every public `load_*` wrapper must call one internal loader-policy helper that centrally handles:

- `required` semantics
- missing file behavior
- default empty return shape
- consistent error message formatting

No notebook may implement file-policy branching.

### 4.2 Public loaders covered

- `load_metrics_master(results_root, required=False)`
- `load_run_manifest(results_root, required=False)`
- `load_quality_cost_master(results_root, required=False)`
- `load_quality_cost_unavailable(results_root, required=False)`

### 4.3 Empty return types (frozen)

- Table/data loaders return empty `pd.DataFrame()` when optional artifacts are absent/unreadable.
- JSON/report loaders return `{}` when optional artifacts are absent/unreadable.
- No loader returns `None` unless explicitly documented.

---

## 5. Status Normalization Contract

### 5.1 Canonical bridge

`normalize_run_status(...)` remains the single status bridge, null-safe for scalar or Series inputs.

Required mapping:

- `ok -> completed`
- `completed -> completed`
- `failed -> failed`
- `error -> failed`
- `skipped -> not_applicable`
- `unsupported -> not_applicable`
- `unavailable -> unavailable`
- `not_in_experiment_matrix -> not_in_experiment_matrix`

Unknown non-empty values pass through normalized casing policy (lower/trim) unless explicitly remapped.
Null/empty values normalize to `unavailable`.

### 5.2 Materialization helper (frozen)

Add one DataFrame-level helper to materialize normalized status once per frame (e.g. `run_status_norm`) with missing-column fallback to unavailable semantics.

Notebook rule:

- filter/bucket on normalized status fields, not raw status literals
- raw `run_status` may remain for display/transparency columns only

---

## 6. Guarded Optional-Column Contract

### 6.1 Shared guard/accessor API (frozen)

Notebooks must use helper-level optional-column guard/accessor utilities; notebook-local `if col not in df.columns` drift is disallowed for protected fields.

The helper API must support:

- materialize missing columns
- numeric coercion (`errors="coerce"`)
- object default materialization

### 6.2 Protected fields

At minimum, guard/accessor coverage includes:

- `metric_availability`
- `coherent_mode_record`
- coherent-vs-mixture delta fields:
  - `coherent_vs_mixture_delta_best_F`
  - `coherent_vs_mixture_delta_best_top_G_sampled_F`
  - `coherent_vs_mixture_delta_postselection_success`
  - `coherent_vs_mixture_distribution_distance`
- decoder-study `structure_*` fields
- quality-vs-cost resource fields used in panels:
  - `qubits`, `gates`, `depth`, `resource_status`, `resource_confidence`, `resource_comparability_basis`

---

## 7. Placeholder/Unavailable Contract

### 7.1 Frozen panel-level behavior

For `required=False` reads and optional slices:

- missing/empty artifacts produce panel-level placeholders/unavailable tables
- notebook execution continues; no notebook-global abort

### 7.2 Frozen placeholder schema

Placeholder/unavailable table contract is stable across notebooks with core fields:

- `panel`
- `status` (`unavailable`)
- `reason`
- optional `details`

All notebooks use helper-provided constructors/render paths for this schema.

---

## 8. Notebook Change Rules

### 8.1 Target notebooks

- `notebooks/06_resource_analysis.ipynb`
- `notebooks/08_paired_encoding_comparison.ipynb`
- `notebooks/09_decoder_phase_diagram.ipynb`
- `notebooks/10_alpha_finite_size_validation.ipynb`
- `notebooks/11_decision_faithfulness_dashboard.ipynb`
- `notebooks/12_quality_vs_cost.ipynb`

### 8.2 Bootstrap and import hygiene

- Remove `sys.path.insert(...)` and notebook-side root probing.
- Use `repo_root()` exclusively.

### 8.3 Stage E source boundary

Comparable Stage E panels may read comparable rows only from `quality_cost_master`.
No notebook fallback joins from other tables for comparable slices.

---

## 9. Verification and Hygiene Gates

### 9.1 Lint-style hygiene checks

Add one explicit hygiene check that enforces:

- no `sys.path.insert(...)` in target notebooks
- no notebook-local root probing/fallback logic
- no raw-status filtering where normalized helper is available

### 9.2 Minimum execution sanity scope

Run targeted smoke execution for notebooks:

- `06_resource_analysis.ipynb`
- `08_paired_encoding_comparison.ipynb`
- `09_decoder_phase_diagram.ipynb`
- `10_alpha_finite_size_validation.ipynb`
- `11_decision_faithfulness_dashboard.ipynb`
- `12_quality_vs_cost.ipynb`

Acceptance criterion:

- missing/empty optional artifacts render panel placeholders/unavailable tables
- no hard notebook abort caused by optional artifact absence

---

## 10. Success Criteria

The design is complete when:

1. Root detection is centralized and deterministic via `repo_root()` only.
2. Loader fallback/required policy is centralized behind one helper.
3. Empty return shapes are consistent (`DataFrame` or `{}`; never implicit `None`).
4. `run_status_norm` is materialized consistently via helper contract and used for filtering.
5. Guarded optional fields are materialized/coerced through shared helper accessors.
6. Placeholder schema and panel-level unavailable behavior are uniform across all six notebooks.
7. Stage E comparable panels use only `quality_cost_master` for comparable data.
