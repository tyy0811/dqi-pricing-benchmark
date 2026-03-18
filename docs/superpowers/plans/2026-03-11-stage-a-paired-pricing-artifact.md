# Stage A Paired Pricing Artifact Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Stage A paired pricing artifact production that emits both `wht_truncated` and `ilp_derived` encodings in one validated schema while preserving legacy WHT artifact compatibility.

**Architecture:** Keep notebook 04 as a thin orchestrator and move pairing logic into `src/` modules. Build both encodings through existing reduction/ILP paths, normalize ILP payload into canonical comparative fields, and enforce a strict paired artifact validator before writing outputs. Add paired serializer helpers for JSON round-trip and focused tests for schema, lineage, shapes, and legacy compatibility.

**Tech Stack:** Python 3.11, NumPy, nbformat/Jupyter notebook JSON structure, pytest.

---

## File Structure Map

- Create: `tests/test_pricing_pairing.py`
  - Purpose: Stage A contract tests for paired schema, serialization round-trip, lineage/provenance, shape checks, and legacy surrogate compatibility.
- Modify: `src/pricing_ilp.py`
  - Purpose: add `solve_pricing_pair(...)` entrypoint and paired artifact validator.
- Modify: `src/ilp_to_xorsat_exact.py`
  - Purpose: add ILP-to-canonical adapter and paired JSON serializer/deserializer helpers.
- Modify: `notebooks/04_pricing_pipeline.ipynb`
  - Purpose: switch export to paired artifact, preserve legacy WHT export, and add validation cell.
- Modify: `docs/superpowers/specs/2026-03-11-stage-a-paired-pricing-artifact-design.md`
  - Purpose: keep spec in sync only if implementation reveals necessary clarifications.

## Chunk 1: Core Schema + Serialization

### Task 1: Add failing paired-contract tests first (@superpowers/test-driven-development)

**Files:**
- Create: `tests/test_pricing_pairing.py`
- Read: `tests/test_ilp_exact_reference.py`
- Read: `tests/test_reduction.py`

- [ ] **Step 1: Add required comparative subset constant and top-level contract test**

```python
REQUIRED_COMPARATIVE_SUBSET = {
    "artifact_version",
    "mode",
    "bit_order",
    "n",
    "m",
    "B",
    "v",
    "term_weights",
    "diagnostics",
    "source_metadata",
}


def test_solve_pricing_pair_contract_top_level_and_encoding_set():
    ...
    assert set(paired.keys()) == {
        "paired_artifact_version",
        "instance_id",
        "pairing_kind",
        "source_metadata",
        "encodings",
    }
    assert set(paired["encodings"].keys()) == {"wht_truncated", "ilp_derived"}
```

- [ ] **Step 2: Add lineage/provenance and cross-encoding consistency tests**

```python
def test_solve_pricing_pair_lineage_and_shared_dimensions():
    ...
    assert wht["source_metadata"]["instance_id"] == paired["instance_id"]
    assert ilp["source_metadata"]["instance_id"] == paired["instance_id"]
    assert wht["n"] == ilp["n"]
    assert wht["m"] == ilp["m"]
```

- [ ] **Step 3: Add shape-integrity tests for both encodings**

```python
def _assert_encoding_shapes(enc: dict):
    n = int(enc["n"])
    m = int(enc["m"])
    assert enc["B"].shape == (m, n)
    assert enc["v"].shape == (m,)
    assert enc["term_weights"].shape == (m,)
```

- [ ] **Step 4: Add paired serializer round-trip test (B/v/term_weights restore to ndarray)**

```python
def test_paired_artifact_roundtrip_restores_array_fields_and_preserves_extras():
    ...
    assert isinstance(restored["encodings"]["wht_truncated"]["B"], np.ndarray)
    assert isinstance(restored["encodings"]["wht_truncated"]["v"], np.ndarray)
    assert isinstance(restored["encodings"]["wht_truncated"]["term_weights"], np.ndarray)
```

- [ ] **Step 5: Add legacy WHT compatibility test via existing surrogate helpers**

```python
def test_wht_payload_remains_legacy_surrogate_compatible():
    ...
    payload = surrogate_artifact_to_dict(paired["encodings"]["wht_truncated"])
    restored = surrogate_artifact_from_dict(payload)
    assert restored["B"].shape == (restored["m"], restored["n"])
```

- [ ] **Step 6: Run the new tests to verify they fail before implementation**

Run: `python -m pytest tests/test_pricing_pairing.py -v`
Expected: FAIL on missing `solve_pricing_pair` / paired serializer helpers.

- [ ] **Step 7: Commit test skeleton (if using git)**

```bash
git add tests/test_pricing_pairing.py
git commit -m "test: add failing Stage A paired artifact contract tests"
```

### Task 2: Implement ILP canonicalization + paired serializer helpers

**Files:**
- Modify: `src/ilp_to_xorsat_exact.py`
- Read: `src/reduction.py`
- Test: `tests/test_pricing_pairing.py`

- [ ] **Step 1: Add ILP canonicalization adapter function**

```python
def canonicalize_ilp_encoding(ilp_payload: dict, *, instance_id: str, source_metadata: dict) -> dict:
    # Map existing exact-reference keys to canonical comparative keys:
    # schema_version -> artifact_version
    # encoding label -> mode
    # n_bits -> n
    # m_terms -> m
    # keep B/v/term_weights/diagnostics/source_metadata
    # preserve extras nested under this encoding payload only
    ...
```

- [ ] **Step 2: Add paired serializer helper to JSON-friendly dict**

```python
def paired_pricing_artifact_to_dict(paired: dict) -> dict:
    # Convert B, v, term_weights arrays to lists inside each encoding payload.
    ...
```

- [ ] **Step 3: Add paired deserializer helper with ndarray restore policy**

```python
def paired_pricing_artifact_from_dict(payload: dict) -> dict:
    # Restore B, v, term_weights as np.ndarray for each encoding payload.
    ...
```

- [ ] **Step 4: Add strict bit-order alignment descriptor validator**

```python
def _validate_bit_order_alignment_descriptor(desc: dict, *, n: int) -> None:
    # Validate kind, source_bit_order, target_bit_order, permutation (if required), verified.
    ...
```

- [ ] **Step 5: Run focused tests for serialization and compatibility**

Run: `python -m pytest tests/test_pricing_pairing.py -k "roundtrip or legacy or subset" -v`
Expected: PASS for implemented serialization/canonicalization tests.

- [ ] **Step 6: Commit helper implementation (if using git)**

```bash
git add src/ilp_to_xorsat_exact.py tests/test_pricing_pairing.py
git commit -m "feat: add paired artifact serializers and ILP canonical adapter"
```

## Chunk 2: Orchestration + Notebook Integration

### Task 3: Implement `solve_pricing_pair(...)` and validator in `src/pricing_ilp.py`

**Files:**
- Modify: `src/pricing_ilp.py`
- Read: `src/reduction.py`
- Read: `src/ilp_to_xorsat_exact.py`
- Test: `tests/test_pricing_pairing.py`

- [ ] **Step 1: Add `solve_pricing_pair(...)` with two encoding builders**

```python
def solve_pricing_pair(prob, instance_id: str, k: int, source_metadata: dict) -> dict:
    wht = _build_wht_encoding(...)
    ilp = _build_ilp_encoding(...)
    paired = {
        "paired_artifact_version": "1.0",
        "instance_id": instance_id,
        "pairing_kind": "pricing_wht_vs_ilp_exact",
        "source_metadata": dict(source_metadata),
        "encodings": {
            "wht_truncated": wht,
            "ilp_derived": ilp,
        },
    }
    _validate_paired_pricing_artifact(paired)
    return paired
```

- [ ] **Step 2: Implement validator with required ordering of checks**

Validation order:
1. Required comparative subset presence.
2. Lineage/provenance checks.
3. Cross-encoding consistency (`n`, `m`, `bit_order` or descriptor).
4. Shape checks for `B/v/term_weights`.

- [ ] **Step 3: Ensure ValueError message set matches locked spec**

Implement exact message templates listed in design doc for deterministic test assertions.

- [ ] **Step 4: Run full paired test module**

Run: `python -m pytest tests/test_pricing_pairing.py -v`
Expected: PASS.

- [ ] **Step 5: Commit orchestration changes (if using git)**

```bash
git add src/pricing_ilp.py src/ilp_to_xorsat_exact.py tests/test_pricing_pairing.py
git commit -m "feat: add solve_pricing_pair with strict paired artifact validation"
```

### Task 4: Update Notebook 04 export flow and add validation cell

**Files:**
- Modify: `notebooks/04_pricing_pipeline.ipynb`
- Read: `notebooks/_helpers.py` (if helper reuse is useful)

- [ ] **Step 1: Replace one-sided export cell with paired orchestrator call**

Notebook code change target:
```python
from src.pricing_ilp import solve_pricing_pair
from src.ilp_to_xorsat_exact import (
    paired_pricing_artifact_to_dict,
    paired_pricing_artifact_from_dict,
)
```

- [ ] **Step 2: Write paired artifact to results path**

```python
paired_path = ROOT / "results" / "paired_pricing" / f"{instance_id}.paired.json"
paired_path.parent.mkdir(parents=True, exist_ok=True)
paired_path.write_text(json.dumps(paired_pricing_artifact_to_dict(paired), indent=2))
```

- [ ] **Step 3: Preserve legacy WHT artifact using existing surrogate serializer helper**

```python
legacy_path = ROOT / "data" / "instances" / f"surrogate_{instance_id}_k15.json"
legacy_wht = paired["encodings"]["wht_truncated"]
legacy_path.write_text(json.dumps(surrogate_artifact_to_dict(legacy_wht), indent=2))
```

- [ ] **Step 4: Add validation cell with 4 assertion blocks**

Cell checks:
- paired schema
- encoding membership
- cross-encoding consistency + lineage/provenance
- array/shape integrity

- [ ] **Step 5: Execute notebook 04 to verify outputs exist and parse**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/04_pricing_pipeline.ipynb`
Expected:
- `results/paired_pricing/<instance_id>.paired.json` exists
- `data/instances/surrogate_<instance_id>_k15.json` exists
- validation cell passes with no error output

- [ ] **Step 6: Commit notebook integration (if using git)**

```bash
git add notebooks/04_pricing_pipeline.ipynb
git commit -m "feat: emit paired pricing artifact in notebook 04 with legacy WHT export"
```

### Task 5: Verification and completion checks (@superpowers/verification-before-completion)

**Files:**
- Verify: `results/paired_pricing/*.paired.json`
- Verify: `data/instances/surrogate_*_k15.json`

- [ ] **Step 1: Run targeted tests**

Run: `python -m pytest tests/test_pricing_pairing.py tests/test_ilp_exact_reference.py -v`
Expected: PASS.

- [ ] **Step 2: Verify notebook output artifacts parse**

Run:
```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/paired_pricing')
files=sorted(p.glob('*.paired.json'))
assert files, 'no paired artifacts'
for fp in files:
    with open(fp) as f:
        _=json.load(f)
print('paired artifacts parse OK:', [str(x) for x in files])
PY
```
Expected: paired artifact JSON parse success.

- [ ] **Step 3: Verify legacy surrogate parse compatibility**

Run:
```bash
python - <<'PY'
import json
from src.reduction import surrogate_artifact_from_dict
from pathlib import Path
paths=sorted(Path('data/instances').glob('surrogate_*_k15.json'))
assert paths, 'no legacy surrogate files'
for p in paths:
    payload=json.loads(p.read_text())
    art=surrogate_artifact_from_dict(payload)
    assert art['B'].shape == (art['m'], art['n'])
print('legacy surrogate compatibility OK')
PY
```
Expected: no assertion failures.

- [ ] **Step 4: Final commit (if using git)**

```bash
git add src tests notebooks docs/superpowers/specs/2026-03-11-stage-a-paired-pricing-artifact-design.md docs/superpowers/plans/2026-03-11-stage-a-paired-pricing-artifact.md
git commit -m "feat: Stage A paired pricing artifact contract with notebook 04 integration"
```
