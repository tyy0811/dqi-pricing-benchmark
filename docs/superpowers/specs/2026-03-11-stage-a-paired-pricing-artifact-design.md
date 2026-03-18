# Stage A Paired Pricing Artifact Design

## Scope
Implement Stage A paired artifact production for pricing instances with minimal, composable changes:
- Add paired entrypoint in `src/pricing_ilp.py`.
- Add ILP canonicalization + paired serializers in `src/ilp_to_xorsat_exact.py`.
- Update `notebooks/04_pricing_pipeline.ipynb` to emit paired + legacy artifacts.
- Add `tests/test_pricing_pairing.py`.

No broad repo rewrites. Notebook analysis style remains unchanged where possible.

## Goals
1. Produce one paired JSON artifact containing both encodings for the same pricing instance.
2. Preserve full canonical per-encoding payloads (including optional extras).
3. Enforce a shared required comparative subset on each encoding.
4. Keep Notebook 05 compatibility by preserving legacy WHT artifact export.

## Non-Goals
- No matrix reruns.
- No schema migration across unrelated modules.
- No new top-level encoding-specific fields in paired JSON.

## Paired Artifact Contract

### Top-level (exact key set)
Top-level keys must be exactly:
- `paired_artifact_version`
- `instance_id`
- `pairing_kind`
- `source_metadata`
- `encodings`

### Encodings
`encodings` must contain exactly:
- `wht_truncated`: full canonical artifact dictionary
- `ilp_derived`: full canonical artifact dictionary

Encoding-specific extras are allowed, but must remain nested under each encoding payload.

### Required comparative subset per encoding
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
```

### Shared consistency checks
- `set(encodings.keys()) == {"wht_truncated", "ilp_derived"}`
- Top-level `instance_id` matches both encoding lineages.
- Same pricing provenance across encodings (at minimum `instance_id`, `pricing_instance_path`; optional hash if present in both).
- Same `n`.
- Same `m`.
- Same `bit_order`, or explicit alignment descriptor at:
  `encodings["ilp_derived"]["source_metadata"]["bit_order_alignment"]`.
- No encoding-specific fields promoted to top level.

### Bit-order alignment descriptor schema
Allowed only at `encodings["ilp_derived"]["source_metadata"]["bit_order_alignment"]`:
```python
{
  "kind": "identity" | "permutation",
  "source_bit_order": [...],
  "target_bit_order": [...],
  "permutation": [...],  # required if kind == "permutation"
  "verified": True,
}
```

## Module Design

### `src/pricing_ilp.py`
Add high-level API:
```python
def solve_pricing_pair(prob, instance_id: str, k: int, source_metadata: dict) -> dict:
    ...
```
Internal helpers (private):
- `_build_wht_encoding(...)`
- `_build_ilp_encoding(...)`
- `_validate_paired_pricing_artifact(...)`

Behavior:
- Build WHT artifact using existing `build_surrogate_artifact(...)` path.
- Build ILP exact artifact using existing toy ILP path and canonicalize to shared subset field names.
- Preserve optional per-encoding extras.
- Validate pair contract before returning.

### `src/ilp_to_xorsat_exact.py`
Add:
- `canonicalize_ilp_encoding(...)` (or equivalent adapter name) to produce canonical comparative fields for ILP-derived payload.
- `paired_pricing_artifact_to_dict(...)`
- `paired_pricing_artifact_from_dict(...)`

Round-trip restore policy:
- `B -> np.ndarray`
- `v -> np.ndarray`
- `term_weights -> np.ndarray`
- Encoding extras preserved unchanged under their encoding payload.

## Notebook 04 Update
Update only artifact-production and validation cells:
- Replace one-sided surrogate export call with `solve_pricing_pair(...)`.
- Write paired artifact to:
  - `results/paired_pricing/<instance_id>.paired.json`
- Keep legacy WHT export to:
  - `data/instances/surrogate_<instance_id>_k15.json`
- Legacy export must use existing surrogate serializer/deserializer helpers functionally (no manual field copying contract).

Add validation cell with 4 assertion blocks:
1. Paired schema
2. Encoding membership
3. Cross-encoding consistency + lineage/provenance
4. Array/shape integrity (`B`, `v`, `term_weights` against `(m, n)`)

## Testing Plan
Add `tests/test_pricing_pairing.py` covering:
1. Top-level exact key set.
2. Exact encoding membership.
3. Required subset presence on each encoding.
4. Literal lineage checks:
   - `encodings["wht_truncated"]["source_metadata"]["instance_id"] == paired["instance_id"]`
   - `encodings["ilp_derived"]["source_metadata"]["instance_id"] == paired["instance_id"]`
5. Provenance alignment checks for required shared keys.
6. Cross-encoding consistency (`n`, `m`, `bit_order` or valid alignment descriptor).
7. Shape checks:
   - `B.shape == (m, n)`
   - `v.shape == (m,)`
   - `term_weights.shape == (m,)`
8. JSON round-trip via paired serializer/deserializer, including ndarray restoration for `B/v/term_weights`.
9. Legacy WHT compatibility test via existing surrogate serializer/deserializer helper path.

## Error Semantics (ValueError)
Use explicit messages:
- `paired artifact missing required top-level field: {field}`
- `paired artifact has invalid encoding set: expected {'wht_truncated', 'ilp_derived'}`
- `encoding '{encoding}' missing required comparative field: {field}`
- `lineage mismatch for key '{key}': expected shared provenance across encodings`
- `provenance mismatch for key '{key}': top-level and encoding lineage disagree`
- `dimension mismatch across encodings: expected shared n and m`
- `invalid shape for encoding '{encoding}': B must have shape (m, n)`
- `invalid shape for encoding '{encoding}': v must have shape (m,)`
- `invalid shape for encoding '{encoding}': term_weights must have shape (m,)`
- `bit_order mismatch requires source_metadata.bit_order_alignment`
- `invalid bit_order_alignment descriptor for encoding 'ilp_derived'`

## Acceptance Criteria
- Running Notebook 04 emits `results/paired_pricing/<instance>.paired.json`.
- Paired JSON contains both encodings in one schema.
- Legacy `surrogate_<instance>_k15.json` still exists.
- New paired serialization tests pass.
