# DQI Pricing Benchmark Results and Validation Report

## P Dataset Definitions

The **P family** represents **automotive feature pricing problems** of increasing scale.

### Problem Structure

Each P instance models: *"What price tiers should we offer for each car feature to maximize expected revenue across customer segments?"*

**Components:**
- **Features**: Car add-ons (Heated Seats, Panoramic Roof, Premium Audio, etc.)
- **Tiers per feature**: 3 price levels (Standard, Premium, Luxury) encoded as 2 bits
- **Customer segments**: Budget, Mid-range, Premium, Tech-Forward (with budget caps and weights)
- **Constraints**: Bundle rules ("Luxury audio requires ≥Premium seats"), max luxury cap

### P Instance Specifications

| Instance | Features | Bits (n) | Configurations | Feasible | Feasible % | Optimal F* |
|----------|----------|----------|----------------|----------|------------|------------|
| **P3** | 3 | 6 | 64 | 23 | 35.9% | $3,000 |
| **P4** | 4 | 8 | 256 | 57 | 22.3% | $2,800 |
| **P5** | 5 | 10 | 1,024 | 154 | 15.0% | $3,000 |
| **P6** | 6 | 12 | 4,096 | 520 | 12.7% | $3,360 |
| **P7** | 7 | 14 | 16,384 | 1,272 | 7.8% | $3,100 |

### Features by Instance

| Instance | Features |
|----------|----------|
| P3 | Heated Seats, Panoramic Roof, Premium Audio |
| P4 | + Adaptive Cruise |
| P5 | + Ambient Lighting |
| P6 | + Head-Up Display |
| P7 | + Additional feature (7th) |

### Key Observations

1. **Exponential growth**: Each additional feature doubles the search space (2 bits per feature)
2. **Decreasing feasibility**: More features → more constraint violations → fewer valid configurations
3. **Non-monotonic optimal**: F* doesn't simply increase with more features (depends on constraint interactions)

### Validation Scope

- **P3-P6**: Full DQI statevector validation (executed and verified)
- **P7**: Analytic resource estimates only (statevector simulation exceeds 16GB memory budget)

---

## Notebook Validation Status

**Validated:** 2026-03-17

| Notebook | Status | Notes |
|----------|--------|-------|
| 07 - Qiskit Structural Parity | ✅ Pass | Executes without errors |
| 08 - Paired Encoding Comparison | ✅ Pass | Executes without errors |
| 09 - 3-Period Pricing Extension | ✅ Pass | Removed unused smoothing_penalty_total column |
| 09 - Decoder Phase Diagram | ✅ Pass | Executes without errors |
| 10 - Alpha Finite-Size Validation | ✅ Pass | Executes without errors |
| 11 - Decision Faithfulness Dashboard | ✅ Pass | Now shows ilp_derived_exact data (6/6 rows for key metrics) |
| 12 - Quality vs Cost | ✅ Pass | Executes without errors |
| 13 - Scaling Analysis | ✅ Pass | Added P7 scope note (analytic-only due to memory limits) |

### Data Generation Summary

- **Decoder study:** 48 runs generated (P3-P6 with bp1/bruteforce decoders)
- **ILP-derived exact:** 16 runs generated (P3-P6 × 2 decoders × 2 alpha modes)
- **quality_cost_master.parquet:** 36 rows with 4 encoding types
- **metrics_master.json:** 152 P family rows total
