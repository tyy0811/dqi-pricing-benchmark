# Stage 2A Coherent Tiny-Instance Family Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fixed coherent tiny-instance artifacts (`C1/C2/C3`), deterministic manifest-driven loading, strict schema/content validation, and tests for Stage 2A reproducibility.

**Architecture:** Commit immutable artifact files under `data/instances/coherent/`, with NPZ as canonical numeric data and JSON as metadata plus optional mirrored tables. Implement `src/coherent_reference.py` as the single loader/validator API enforcing manifest order, linkage integrity, exactness rules, and coherent size constraints. Keep scope narrow to artifact/loader/test work only (no dynamic Family D implementation).

**Tech Stack:** Python 3.11, NumPy, JSON/NPZ file IO, pytest

---

## File Structure and Responsibilities

- Create: `data/instances/coherent/coherent_manifest.json`
  - Canonical ordered membership, relative paths, optional file checksums.
- Create: `data/instances/coherent/C1.json`, `C1.npz`
  - Explicit tiny max-XORSAT exact sanity artifact.
- Create: `data/instances/coherent/C2.json`, `C2.npz`
  - Tiny pricing-ILP-exact linked artifact with explicit relation semantics.
- Create: `data/instances/coherent/C3.json`, `C3.npz`
  - Tiny structured stress artifact with structure metrics.
- Create: `src/coherent_reference.py`
  - Manifest loader, instance loader, family loader, instance/manifest validators.
- Create: `tests/test_coherent_family.py`
  - Positive and negative validation tests.
- Modify (if needed): `src/dqi_state.py`
  - Reuse/export coherent support check only if utility import path is required.
- Modify (if needed): `tests/test_dqi_coherent_vs_mixture.py`
  - Only if shared coherent limit constants need harmonization with Stage 2A loader checks.

Skills to apply during execution: `@test-driven-development`, `@verification-before-completion`.

## Chunk 1: Add Stage 2A Coherent Artifacts

### Task 1: Author fixed `C1/C2/C3` artifact files and manifest

**Files:**
- Create: `data/instances/coherent/coherent_manifest.json`
- Create: `data/instances/coherent/C1.json`
- Create: `data/instances/coherent/C1.npz`
- Create: `data/instances/coherent/C2.json`
- Create: `data/instances/coherent/C2.npz`
- Create: `data/instances/coherent/C3.json`
- Create: `data/instances/coherent/C3.npz`

- [ ] **Step 1: Add failing presence test for coherent artifact set**

```python
def test_coherent_artifact_files_exist():
    from pathlib import Path
    base = Path("data/instances/coherent")
    required = [
        "coherent_manifest.json",
        "C1.json", "C1.npz",
        "C2.json", "C2.npz",
        "C3.json", "C3.npz",
    ]
    for name in required:
        assert (base / name).exists(), f"missing {name}"
```

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest tests/test_coherent_family.py::test_coherent_artifact_files_exist -v`  
Expected: FAIL (files and/or test module missing).

- [ ] **Step 3: Create coherent artifacts and manifest**

Implementation notes:
- Manifest must include:
  - `manifest_version`, `schema_version`, `family`, `description`,
  - `canonical_instance_order: ["C1", "C2", "C3"]`,
  - ordered `instances` entries with relative paths and shape metadata.
- JSON files must include:
  - required Stage 2A schema fields (`objective_relation`, linkage paths, `npz_is_canonical=true`).
- NPZ files must include canonical arrays:
  - `B`, `v`, `truth_table_F`, `truth_table_G` (+ optional `optimizer_set`, `row_weights`, `constant_offset`).

- [ ] **Step 4: Run presence test to verify GREEN**

Run: `python -m pytest tests/test_coherent_family.py::test_coherent_artifact_files_exist -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/instances/coherent/
git commit -m "data: add fixed coherent tiny-family artifacts C1 C2 C3"
```

## Chunk 2: Implement Coherent Loader/Validator API

### Task 2: Add manifest and instance loader APIs

**Files:**
- Create: `src/coherent_reference.py`
- Modify: `tests/test_coherent_family.py`

- [ ] **Step 1: Add failing loader API tests**

```python
def test_load_coherent_manifest_and_family_order():
    from src.coherent_reference import load_coherent_manifest, load_coherent_family
    manifest = load_coherent_manifest()
    assert manifest["canonical_instance_order"] == ["C1", "C2", "C3"]
    family = load_coherent_family()
    assert [x["meta"]["instance_id"] for x in family] == ["C1", "C2", "C3"]
```

- [ ] **Step 2: Run RED test**

Run: `python -m pytest tests/test_coherent_family.py::test_load_coherent_manifest_and_family_order -v`  
Expected: FAIL (`src.coherent_reference` missing).

- [ ] **Step 3: Implement minimal loader functions**

```python
def load_coherent_manifest(base_dir="data/instances/coherent") -> dict: ...
def load_coherent_instance(instance_id: str, base_dir="data/instances/coherent") -> dict: ...
def load_coherent_family(base_dir="data/instances/coherent") -> list[dict]: ...
```

Requirements:
- resolve paths relative to base directory only,
- enforce canonical order from manifest,
- return both `meta` (JSON) and `arrays` (NPZ) payload per instance.

- [ ] **Step 4: Run targeted tests**

Run: `python -m pytest tests/test_coherent_family.py -k "manifest_and_family_order" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/coherent_reference.py tests/test_coherent_family.py
git commit -m "feat: add coherent family loaders with manifest ordering"
```

### Task 3: Add validator APIs and strict checks

**Files:**
- Modify: `src/coherent_reference.py`
- Modify: `tests/test_coherent_family.py`

- [ ] **Step 1: Add failing positive validation tests**

```python
def test_validate_all_coherent_instances_pass():
    from src.coherent_reference import load_coherent_family, validate_coherent_instance
    for rec in load_coherent_family():
        out = validate_coherent_instance(rec["meta"], rec["arrays"], strict=True)
        assert out["ok"] is True
```

- [ ] **Step 2: Add failing negative tests**

```python
def test_manifest_order_mismatch_fails_clearly(): ...
def test_shape_mismatch_between_json_and_npz_fails_clearly(): ...
def test_non_binary_b_or_v_fails_clearly(): ...
def test_invalid_optimizer_set_fails_clearly(): ...
```

- [ ] **Step 3: Run RED set**

Run: `python -m pytest tests/test_coherent_family.py -k "validate_all_coherent_instances_pass or mismatch or non_binary or optimizer" -v`  
Expected: FAIL before validator completeness.

- [ ] **Step 4: Implement validator logic**

Required checks:
- manifest structure/paths/order consistency,
- JSON vs NPZ agreement for dimensions and optional mirrored tables,
- `B` and `v` binary-value enforcement,
- coherent limits (`n_bits<=6`, `m_rows<=8`, `max(ell_allowed)<=2`) and support check,
- `F_star`/`G_star` maxima consistency,
- `optimizer_set` sorted/unique/range/argmax consistency,
- objective relation reconstruction:
  - `"identical"` exact equality,
  - `"affine"` exact `a*G+b` with saved metadata,
  - `"weighted_xorsat_exact"` exact reconstruction with `row_weights` + `constant_offset`,
- C1/C3 exact max-XORSAT rules,
- C2 linkage path existence/loading.

- [ ] **Step 5: Run targeted validator tests**

Run: `python -m pytest tests/test_coherent_family.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/coherent_reference.py tests/test_coherent_family.py
git commit -m "feat: add strict coherent artifact validation rules"
```

## Chunk 3: Stage 2A Test Coverage and Integration Guardrails

### Task 4: Add semantic tests for C1/C2/C3 requirements

**Files:**
- Modify: `tests/test_coherent_family.py`

- [ ] **Step 1: Add failing semantic tests**

```python
def test_c1_and_c3_are_exact_max_xorsat_truth_consistent(): ...
def test_c2_has_pricing_and_exact_linkage_paths(): ...
def test_all_c_instances_mark_coherent_supported_true(): ...
def test_all_c_instances_satisfy_coherent_size_limits(): ...
```

- [ ] **Step 2: Run RED semantics subset**

Run: `python -m pytest tests/test_coherent_family.py -k "c1_and_c3 or c2_has_pricing or coherent_size_limits or coherent_supported_true" -v`  
Expected: FAIL until full semantics wired.

- [ ] **Step 3: Implement minimal missing code/data fixes**

Potential touchpoints:
- coherent artifact metadata corrections,
- validator helper refinements in `src/coherent_reference.py`.

- [ ] **Step 4: Run full coherent-family test module**

Run: `python -m pytest tests/test_coherent_family.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_coherent_family.py src/coherent_reference.py data/instances/coherent/
git commit -m "test: enforce coherent family C1 C2 C3 semantic invariants"
```

## Chunk 4: Verification Before Completion

### Task 5: Run minimal relevant regression suite

**Files:**
- Verify: `tests/test_coherent_family.py`
- Verify: `tests/test_dqi_coherent_vs_mixture.py` (coherent-mode compatibility)

- [ ] **Step 1: Run coherent family tests**

Run: `python -m pytest tests/test_coherent_family.py -v`  
Expected: PASS.

- [ ] **Step 2: Run coherent execution compatibility tests**

Run: `python -m pytest tests/test_dqi_coherent_vs_mixture.py -v`  
Expected: PASS.

- [ ] **Step 3: Optional benchmark smoke (no matrix expansion)**

Run: `python -m pytest tests/test_benchmark.py -k "stage1_paired_encoding" -v`  
Expected: PASS.

- [ ] **Step 4: Confirm no out-of-scope dynamic-family implementation**

Run: `rg -n "dynamic_manifest|Family D|D1|D2|period|stochastic|scenario tree|RL" src tests data/instances/coherent`  
Expected: no new dynamic-family implementation code in Stage 2A.

- [ ] **Step 5: Final commit**

```bash
git add src/coherent_reference.py tests/test_coherent_family.py data/instances/coherent/
git commit -m "feat: stage2a coherent tiny-family artifacts and deterministic loader"
```

## Notes for Execution

- Keep scope strict: only artifacts + manifest + loader/validator + tests.
- Do not implement Family D in this plan.
- If git metadata is unavailable (`.git` missing), execute all non-git steps and report commit steps as blocked.
- Use `@verification-before-completion`: no pass/completion claims without fresh command output.
