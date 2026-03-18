# Stage 2A Coherent Tiny-Instance Family Design

**Date:** 2026-03-11
**Status:** Approved

## Goal
Define a fixed, reproducible coherent tiny-instance family (`C1/C2/C3`) for Stage 2 Matrix B experiments, with deterministic loading and strict validation.

## Scope
In scope:
- Commit fixed coherent artifacts under `data/instances/coherent/`.
- Add deterministic manifest-driven loader and validator.
- Add tests for positive loading/validation and negative failure cases.

Out of scope:
- Dynamic pricing family generation (Family D).
- Full Stage 2 alpha/decoder/coherent matrix implementation.
- New stochastic/control/RL/sampling frameworks.

## Artifacts and Layout
Directory:
- `data/instances/coherent/`

Files:
- `C1.json`, `C1.npz`
- `C2.json`, `C2.npz`
- `C3.json`, `C3.npz`
- `coherent_manifest.json`

## C-Family Definitions

### C1
- `source_type = explicit_bv`
- `n_bits = 4`
- `m_rows = 4`
- `ell_default = 1`
- `coherent_supported = true`
- `objective_type = max_xorsat_exact`
- must satisfy `truth_table_F == truth_table_G`

### C2
- `source_type = pricing_ilp_exact`
- `n_bits = 4`
- `m_rows <= 6`
- `ell_default = 1`
- `coherent_supported = true`
- `objective_type = pricing_exact`
- must preserve linkage to saved toy-pricing schema and exact encoding source
- includes `wht_exact_available` flag (no forced exact WHT requirement)

### C3
- `source_type = synthetic_structured_bv`
- `n_bits = 6`
- `m_rows = 8`
- `ell_default = 2`
- `coherent_supported = true`
- `objective_type = max_xorsat_exact`
- includes structure metrics (`rank_B_gf2`, `decoder_collision_rate`, `ambiguous_syndrome_count`)
- must satisfy `truth_table_F == truth_table_G`

## Manifest Schema (`coherent_manifest.json`)
Required top-level fields:
- `manifest_version`
- `schema_version`
- `family` (must be `coherent_tiny_reference`)
- `description`
- `canonical_instance_order` (must be `["C1", "C2", "C3"]`)
- `instances` (ordered list matching canonical order)

Each `instances[]` entry:
- `instance_id`
- `family` (must be `C`)
- `json_path` (relative path)
- `npz_path` (relative path)
- `n_bits`
- `m_rows`
- `ell_default`
- `coherent_supported`
- optional `json_sha256`
- optional `npz_sha256`

## Per-Instance JSON Schema (`C*.json`)
Required fields:
- `schema_version`
- `instance_id`
- `family` (must be `C`)
- `source_type`
- `description`
- `seed`
- `n_bits`
- `m_rows`
- `ell_default`
- `ell_allowed`
- `encoding_backend`
- `objective_type`
- `objective_relation`
- `coherent_supported`
- `pricing_source_path` (string or `null`)
- `exact_encoding_source_path` (string or `null`)
- `wht_exact_available` (bool)
- `F_star`
- `G_star`
- `optimizer_set`
- `npz_is_canonical` (must be `true`)
- optional `truth_tables_mirrored_in_json`
- optional mirrored `truth_table_F` and `truth_table_G`
- optional `truth_table_length`

Additional required fields for `C3`:
- `structure_metrics` with:
  - `rank_B_gf2`
  - `decoder_collision_rate`
  - `ambiguous_syndrome_count`

## Per-Instance NPZ Schema (`C*.npz`)
NPZ is canonical for numeric arrays.

Required arrays:
- `B` shape `(m_rows, n_bits)` dtype int8 (values must be in `{0,1}`)
- `v` shape `(m_rows,)` dtype int8 (values must be in `{0,1}`)
- `truth_table_F` shape `(2**n_bits,)` dtype float64
- `truth_table_G` shape `(2**n_bits,)` dtype float64

Optional arrays:
- `optimizer_set` int64 indices
- `row_weights` float64 shape `(m_rows,)`
- `constant_offset` float64 scalar

## Objective Relation Rules
`objective_relation` must be one of:
- `"identical"`:
  - require `truth_table_F == truth_table_G`
- `"affine"`:
  - require saved scalar metadata `affine_a` and `affine_b` in JSON
  - require `truth_table_F == affine_a * truth_table_G + affine_b`
- `"weighted_xorsat_exact"`:
  - require `row_weights` and `constant_offset` in NPZ
  - require exact reconstruction of `truth_table_F` from `(B, v, row_weights, constant_offset)`

For `C2`, validator must enforce one of:
- exact identical case, or
- weighted exact reconstruction case.

## Optimizer Set Semantics
`optimizer_set` is a set of bitstring indices (integers), not decoded structured assignments.
Each entry must satisfy:
- integer in `[0, 2**n_bits)`
- sorted ascending
- unique
- corresponds to `F_star`
Strict validation mode may require equality with the full argmax set.

## Loader and Validator API
Create `src/coherent_reference.py` with:
- `load_coherent_manifest(base_dir="data/instances/coherent")`
- `load_coherent_instance(instance_id, base_dir="data/instances/coherent")`
- `load_coherent_family(base_dir="data/instances/coherent")`
- `validate_coherent_instance(instance_meta, arrays, *, strict=True)`
- `validate_coherent_manifest(manifest, base_dir=...)`

Determinism:
- `load_coherent_family` must return instances in `canonical_instance_order`.
- no ad hoc glob-based ordering.

## Validation Rules
Hard failures on:
- missing manifest entries or path resolution failures
- `instances` order mismatch vs `canonical_instance_order`
- JSON/NPZ dimension mismatches
- non-binary values in `B`/`v`
- `ell_default` not in `ell_allowed`
- any `ell` in `ell_allowed` violating coherent limits
- inconsistent `F_star`/`G_star` vs table maxima
- invalid `optimizer_set`
- broken `C2` linkage paths
- unsupported or invalid `objective_relation` reconstruction

Coherent size limits for C-family:
- `n_bits <= 6`
- `m_rows <= 8`
- `max(ell_allowed) <= 2`
- and must pass `coherent_supported_for_instance(...)`

Exact checks:
- C1/C3: recompute `truth_table_G` from encoding and enforce
  `truth_table_F == truth_table_G`.
- C2: enforce declared `objective_relation` reconstruction rule exactly.

## Tests
Add `tests/test_coherent_family.py` covering:
- all `C1/C2/C3` load successfully
- deterministic manifest order
- coherent support true for all three
- C1/C3 exact max-XORSAT truth consistency
- C2 linkage and exactness-rule validation
- coherent size-limit compliance

Add negative tests:
- broken manifest path/order fails clearly
- JSON/NPZ shape mismatch fails clearly
- non-binary `B`/`v` fails clearly
- invalid `optimizer_set` fails clearly

## Acceptance Criteria
- `C1/C2/C3` are concretely defined and committed.
- coherent family loads reproducibly from manifest only.
- validation catches schema/content/linkage/coherent-limit violations.
- Matrix B can run without ad hoc C-family generation.
- implementation remains limited to Stage 2A artifact+loader+tests scope.

## Follow-On Boundary
Family D (dynamic toy pricing) is a separate design cycle and not implemented in Stage 2A.
