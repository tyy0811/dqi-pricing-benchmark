#!/usr/bin/env python3
"""Generate missing paired artifacts for P3 and P4 pricing instances.

This script generates pricing_3feat_6bit.paired.json and pricing_4feat_8bit.paired.json
which are needed by notebook 13 for the WHT vs ILP encoding quality comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure repo-root imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return super().default(obj)

from src.pricing_ilp import solve_pricing_pair
from src.problem_generator import PricingProblem


CANONICAL_M_VALUE = 15  # Matches notebook 13's canonical_m_value


def load_instance(path: Path) -> PricingProblem:
    """Load a frozen pricing instance JSON into a PricingProblem."""
    with open(path, "r") as f:
        data = json.load(f)
    return PricingProblem.from_dict(data)


def generate_paired_artifact(instance_id: str, instance_path: Path, output_dir: Path) -> Path:
    """Generate a paired artifact for the given instance."""
    prob = load_instance(instance_path)

    source_metadata = {
        "instance_id": instance_id,
        "pricing_instance_path": str(instance_path.relative_to(REPO_ROOT)),
    }

    paired = solve_pricing_pair(
        prob,
        instance_id=instance_id,
        k=CANONICAL_M_VALUE,
        source_metadata=source_metadata,
    )

    output_path = output_dir / f"{instance_id}.paired.json"
    with open(output_path, "w") as f:
        json.dump(paired, f, indent=2, cls=NumpyEncoder)

    return output_path


def main():
    instances_dir = REPO_ROOT / "data" / "instances"
    output_dir = REPO_ROOT / "results" / "paired_pricing"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Missing artifacts for P3 and P4
    missing_instances = [
        ("pricing_3feat_6bit", instances_dir / "pricing_3feat_6bit.json"),
        ("pricing_4feat_8bit", instances_dir / "pricing_4feat_8bit.json"),
    ]

    for instance_id, instance_path in missing_instances:
        output_path = output_dir / f"{instance_id}.paired.json"
        if output_path.exists():
            print(f"[SKIP] {instance_id}: artifact already exists at {output_path}")
            continue

        if not instance_path.exists():
            print(f"[ERROR] {instance_id}: instance file not found at {instance_path}")
            continue

        print(f"[GENERATING] {instance_id}...")
        try:
            generated_path = generate_paired_artifact(instance_id, instance_path, output_dir)
            print(f"[SUCCESS] {instance_id}: generated {generated_path}")
        except Exception as e:
            print(f"[ERROR] {instance_id}: {e}")
            raise


if __name__ == "__main__":
    main()
