from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alpha_theory_validation import run_alpha_validation


def _split_csv(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    parsed = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    return parsed or None


def _split_int_csv(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    parsed = str(raw).split(",")
    values = tuple(int(part.strip()) for part in parsed if part.strip())
    return values or None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Stage D Family-C alpha validation.")
    p.add_argument("--output-root", default="results")
    p.add_argument("--instance-ids", default="C1,C2,C3")
    p.add_argument("--alpha-modes", default="uniform,paper,heuristic")
    p.add_argument("--execution-modes", default="coherent,mixture")
    p.add_argument("--ell-values", default="")
    p.add_argument("--m-active-values", default="")
    p.add_argument("--trial-seed", type=int, default=42)
    p.add_argument("--n-dqi-samples", type=int, default=256)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = run_alpha_validation(
        output_root=args.output_root,
        instance_ids=_split_csv(args.instance_ids),
        alpha_modes=_split_csv(args.alpha_modes),
        execution_modes=_split_csv(args.execution_modes),
        ell_values=_split_int_csv(args.ell_values),
        m_active_values=_split_int_csv(args.m_active_values),
        trial_seed=int(args.trial_seed),
        n_dqi_samples=int(args.n_dqi_samples),
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
