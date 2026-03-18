# BMW Reviewer Summary

This repository supports four reviewer-facing claims.

## 1. Pricing pipeline works

The pricing pipeline produces usable benchmark surfaces and reviewer-facing notebook artifacts, including the pricing pipeline walkthrough in [04_pricing_pipeline.ipynb](/Users/zenith/Desktop/dqi-pricing-benchmark/notebooks/04_pricing_pipeline.ipynb).

## 2. Exact-reference / ILP-derived encoding beats WHT

The paired encoding comparison shows the exact-reference / ILP-derived encoding as the stronger reviewer-facing baseline relative to WHT. See [08_paired_encoding_comparison.ipynb](/Users/zenith/Desktop/dqi-pricing-benchmark/notebooks/08_paired_encoding_comparison.ipynb).

## 3. Decoder robustness matters

Decoder choice materially affects downstream behavior, so robustness is part of the benchmark story rather than an implementation detail. See [05_decoder_comparison.ipynb](/Users/zenith/Desktop/dqi-pricing-benchmark/notebooks/05_decoder_comparison.ipynb).

## 4. 3-period extension shows positive dynamic-vs-static uplift

The compact 3-period scenario matrix includes positive dynamic-vs-static uplift cases while also retaining near-zero cases and smoothing tradeoff cases for interpretability. See [09_3period_pricing_extension.ipynb](/Users/zenith/Desktop/dqi-pricing-benchmark/notebooks/09_3period_pricing_extension.ipynb).

## Scope Note

Qiskit evidence is **structural subset parity/resource evidence**, not full decode-inclusive DQI parity. The tiny validated subset case and the larger family-S structural/resource context are documented in [07_qiskit_structural_parity.ipynb](/Users/zenith/Desktop/dqi-pricing-benchmark/notebooks/07_qiskit_structural_parity.ipynb).
