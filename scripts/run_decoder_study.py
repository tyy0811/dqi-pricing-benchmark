from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.decoder_study_runner import run_decoder_study


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Stage C decoder-study raw emitter.")
    p.add_argument("--output-root", default="results")
    p.add_argument("--p-instances", default="P3,P4,P5,P6,P7")
    p.add_argument("--s-instances", default="")
    p.add_argument("--alpha-modes", default="paper,uniform")
    p.add_argument("--decoders", default="bruteforce,bp1,bp_osd_lite,oracle")
    p.add_argument("--n-dqi-samples", type=int, default=128)
    p.add_argument("--top-k-by-surrogate", type=int, default=10)
    p.add_argument("--trial-seed", type=int, default=42)
    p.add_argument("--oracle-max-m", type=int, default=10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = run_decoder_study(
        output_root=args.output_root,
        p_instances=_split_csv(args.p_instances),
        s_instances=None if not str(args.s_instances).strip() else _split_csv(args.s_instances),
        alpha_modes=_split_csv(args.alpha_modes),
        decoders=_split_csv(args.decoders),
        n_dqi_samples=int(args.n_dqi_samples),
        top_k_by_surrogate=int(args.top_k_by_surrogate),
        trial_seed=int(args.trial_seed),
        oracle_max_m=int(args.oracle_max_m),
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
