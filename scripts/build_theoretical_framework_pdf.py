#!/usr/bin/env python3

import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "Theoretical_Framework.md"
OUTPUT = ROOT / "Theoretical_Framework.pdf"
HEADER = ROOT / "scripts" / "pandoc_theoretical_framework_header.tex"

LONGTABLE_RE = re.compile(r"\\begin\{longtable\}\[\]\{@\{\}([lrcp{}0-9.\\\-\s\(\)\*\+\w]+)@\{\}\}")


def replacement_spec(spec: str) -> str:
    cleaned = "".join(ch for ch in spec if ch in "lrc")
    if not cleaned:
        return "@{}" + spec + "@{}"

    count = len(cleaned)
    if count == 2:
        widths = [0.28, 0.68]
    elif count == 5:
        widths = [0.16, 0.12, 0.24, 0.12, 0.32]
    else:
        base = 0.96 / count
        widths = [base] * count

    pieces = []
    for col, width in zip(cleaned, widths):
        if col == "r":
            prefix = r">{\raggedleft\arraybackslash}p"
        elif col == "c":
            prefix = r">{\centering\arraybackslash}p"
        else:
            prefix = r">{\raggedright\arraybackslash}p"
        pieces.append(f"{prefix}{{{width:.4f}\\linewidth}}")

    return "@{}" + "".join(pieces) + "@{}"


def patch_tex(tex: str) -> str:
    tex = tex.replace("⟨", r"\langle ")
    tex = tex.replace("⟩", r"\rangle ")
    tex = tex.replace(r"\inGF", r"\in GF")
    tex = tex.replace(r"\toGF", r"\to GF")
    tex = tex.replace(r"->", r"\to ")

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
        tex_path = tmp / "theoretical_framework.tex"

        run(
            [
                "pandoc",
                str(SOURCE),
                "--from=markdown+tex_math_dollars",
                "--standalone",
                "--to=latex",
                "-H",
                str(HEADER),
                "-o",
                str(tex_path),
            ],
            cwd=ROOT,
        )

        tex_path.write_text(patch_tex(tex_path.read_text()))

        run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                str(tex_path.name),
            ],
            cwd=tmp,
        )

        OUTPUT.write_bytes(tex_path.with_suffix(".pdf").read_bytes())

    print(OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
