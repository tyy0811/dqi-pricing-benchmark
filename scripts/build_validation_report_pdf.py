#!/usr/bin/env python3

import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "DQI_Pricing_Benchmark_Results_and_Validation_Report.md"
OUTPUT = ROOT / "DQI_Pricing_Benchmark_Results_and_Validation_Report.pdf"
HEADER = ROOT / "scripts" / "pandoc_report_pdf_header.tex"


LONGTABLE_RE = re.compile(r"\\begin\{longtable\}\[\]\{@\{\}([lrc]+)@\{\}\}")


def weighted_widths(spec: str) -> list[float]:
    weights = []
    count = len(spec)
    for idx, col in enumerate(spec):
        if count == 2 and col in {"l", "c"}:
            weights.append(1.0 if idx == 0 else 2.6)
            continue

        if col == "r":
            weights.append(1.0)
        elif col == "c":
            weights.append(1.2 if idx == 0 else 1.4)
        else:
            weights.append(1.2 if idx == 0 else (1.6 if idx == 1 else 2.2))

    usable = 0.96
    total = sum(weights)
    return [usable * weight / total for weight in weights]


def replacement_spec(spec: str) -> str:
    pieces = []
    for col, width in zip(spec, weighted_widths(spec)):
        if col == "r":
            prefix = r">{\raggedleft\arraybackslash}p"
        elif col == "c":
            prefix = r">{\centering\arraybackslash}p"
        else:
            prefix = r">{\raggedright\arraybackslash}p"
        pieces.append(f"{prefix}{{{width:.4f}\\linewidth}}")
    return "@{}" + "".join(pieces) + "@{}"


def patch_longtables(tex: str) -> str:
    return LONGTABLE_RE.sub(
        lambda match: "\\begin{longtable}[]{" + replacement_spec(match.group(1)) + "}",
        tex,
    )


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not HEADER.exists():
        raise FileNotFoundError(HEADER)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tex_path = tmp / "report.tex"

        run(
            [
                "pandoc",
                str(SOURCE),
                "--from=gfm",
                "--standalone",
                "--to=latex",
                "--pdf-engine=xelatex",
                "-V",
                "fontsize=9pt",
                "-H",
                str(HEADER),
                "-o",
                str(tex_path),
            ],
            cwd=ROOT,
        )

        tex_path.write_text(patch_longtables(tex_path.read_text()))

        run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                str(tex_path.name),
            ],
            cwd=tmp,
        )

        produced_pdf = tex_path.with_suffix(".pdf")
        OUTPUT.write_bytes(produced_pdf.read_bytes())

    print(OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
