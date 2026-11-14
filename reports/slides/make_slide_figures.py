"""Dark versions of the final report's figures, for the slides (background #222222).

Reuses the drawing code in reports/final/make_figures.py and swaps only the
colours: the palette's dark steps, validated on #222222 (help/hurt pair
#3987e5/#e66767, ordinal ramp #1c5cab -> #3987e5 -> #9ec5f4), light ink, a
transparent background.

Run from the repository root:  PYTHONPATH=src python reports/slides/make_slide_figures.py
Writes reports/slides/figures/*.pdf.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt

from acl import analysis

spec = importlib.util.spec_from_file_location(
    "make_figures", Path(__file__).parents[1] / "final" / "make_figures.py")
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)

mf.OUT = Path(__file__).parent / "figures"
mf.INK, mf.INK2, mf.GRID = "#e6e6e6", "#b3b3b3", "#3d3d3d"
mf.BLUE, mf.RED = "#3987e5", "#e66767"
mf.RAMP = ["#1c5cab", "#3987e5", "#9ec5f4"]
plt.rcParams.update({
    "text.color": mf.INK, "axes.labelcolor": mf.INK, "axes.edgecolor": mf.INK2,
    "xtick.color": mf.INK2, "ytick.color": mf.INK2, "legend.labelcolor": mf.INK,
    "savefig.transparent": True,
})


def main() -> None:
    mf.OUT.mkdir(parents=True, exist_ok=True)
    d = analysis.load(300, "ndcg")
    mf.fig_helps_hurts(d)
    mf.fig_degree(d)
    mf.fig_heldout(d)
    print(f"-> {mf.OUT}")


if __name__ == "__main__":
    main()
