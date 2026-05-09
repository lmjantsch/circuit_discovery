"""
Heatmap views of how strongly each weight axis (q, k, v, gate, up) is
associated with EAP-frnorm sweep performance (CPR).

Two heatmaps are drawn in one figure:

1. Sensitivity heatmap (5 (model, task) blocks x 5 axes).
   Cell value = max(CPR) - min(CPR) across the 8-point sweep of that axis
   (others held at 1.0). Higher = the axis matters more for performance,
   irrespective of the response shape (handles non-monotone U-curves).

2. Landscape grid (5 small heatmaps, one per (model, task)).
   Each cell = CPR for a single (axis, weight value) combination.
   Reveals where in value-space the gain (or loss) sits.

Both panels are annotated with the underlying numbers.

Source root is configurable via the GRID_DIR_NAME env var (default
`frozen_grid_search`).

Run:
  /raid/conda/envs/dacslab_djk_mib/bin/python \
      -m experiments.mib.visualization.plotters.grid_search_heatmap
"""

import csv
import os
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from experiments.mib.visualization.utils import save_figure


PATHS = ("q", "k", "v", "gate", "up")
WEIGHT_VALUES = ("0.0", "0.05", "0.1", "0.2", "0.5", "1.0", "1.5", "3.0")
ORDER = (
    ("gpt2", "ioi"),
    ("qwen2.5", "ioi"),
    ("qwen2.5", "mcqa"),
    ("gemma2", "ioi"),
    ("gemma2", "mcqa"),
)

GRID_DIR_NAME = os.environ.get("GRID_DIR_NAME", "frozen_grid_search")
GRID_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    GRID_DIR_NAME,
)
GRID_REPORT_CSV = os.path.join(GRID_DIR, "report.csv")
PLOT_OUT = os.path.join(GRID_DIR, "plots", "param_sensitivity_heatmap.png")


def _load_blocks() -> dict[tuple[str, str], list[dict]]:
    rows = list(csv.DictReader(open(GRID_REPORT_CSV)))
    blocks: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["cpr"] in ("-", ""):
            continue
        blocks[(r["model"], r["task"])].append(r)
    return blocks


def _axis_sweep(block: list[dict], axis: str) -> list[Optional[float]]:
    """Return CPR for each value in WEIGHT_VALUES (others held at 1.0).
    Missing entries are None."""
    by_value: dict[str, float] = {}
    for r in block:
        if all(r[p] == "1.0" for p in PATHS if p != axis):
            by_value[r[axis]] = float(r["cpr"])
    return [by_value.get(v) for v in WEIGHT_VALUES]


def _build_sensitivity_matrix(blocks) -> np.ndarray:
    """Return (n_blocks, n_axes) array of max-min CPR per (block, axis)."""
    mat = np.full((len(ORDER), len(PATHS)), np.nan)
    for i, mt in enumerate(ORDER):
        block = blocks.get(mt, [])
        for j, ax in enumerate(PATHS):
            sweep = _axis_sweep(block, ax)
            vals = [v for v in sweep if v is not None]
            if not vals:
                continue
            mat[i, j] = max(vals) - min(vals)
    return mat


def _build_landscape_matrix(blocks, mt) -> np.ndarray:
    """Return (n_axes, n_values) array of CPR values for one (model, task)."""
    mat = np.full((len(PATHS), len(WEIGHT_VALUES)), np.nan)
    block = blocks.get(mt, [])
    for i, ax in enumerate(PATHS):
        sweep = _axis_sweep(block, ax)
        for j, v in enumerate(sweep):
            if v is not None:
                mat[i, j] = v
    return mat


def _annotate(ax: plt.Axes, mat: np.ndarray, fmt: str, contrast_threshold: float):
    """Write each cell value into the heatmap, choosing black/white text by cell luminance."""
    im = ax.images[0]
    rgba = im.cmap(im.norm(mat))
    for (i, j), value in np.ndenumerate(mat):
        if np.isnan(value):
            ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="0.5")
            continue
        # luminance-based color choice for legibility on diverging colormaps
        lum = 0.299 * rgba[i, j, 0] + 0.587 * rgba[i, j, 1] + 0.114 * rgba[i, j, 2]
        text_color = "white" if lum < contrast_threshold else "black"
        ax.text(j, i, format(value, fmt), ha="center", va="center",
                fontsize=7, color=text_color)


def make_figure() -> Figure:
    blocks = _load_blocks()

    # Baselines (uniform 1.0) per block, used for landscape diverging colormap centering
    baselines: dict[tuple[str, str], Optional[float]] = {}
    for mt in ORDER:
        b = next(
            (r for r in blocks.get(mt, []) if all(r[p] == "1.0" for p in PATHS)),
            None,
        )
        baselines[mt] = float(b["cpr"]) if b is not None else None

    # ----- Layout: top sensitivity heatmap, bottom 5 landscape heatmaps -----
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.2, 1.0],
                          hspace=0.45, wspace=0.25)

    # === Top: sensitivity heatmap (spans all 5 columns) ===
    ax_top = fig.add_subplot(gs[0, :])
    sens = _build_sensitivity_matrix(blocks)
    vmax = np.nanmax(sens)
    im = ax_top.imshow(sens, cmap="viridis", aspect="auto",
                       vmin=0.0, vmax=vmax)
    ax_top.set_xticks(range(len(PATHS)))
    ax_top.set_xticklabels(PATHS, fontsize=11)
    ax_top.set_yticks(range(len(ORDER)))
    ax_top.set_yticklabels([f"{m}/{t}" for m, t in ORDER], fontsize=10)
    ax_top.set_title(
        "Sensitivity: max(CPR) − min(CPR) across the 8-point sweep of each axis\n"
        "(higher = the parameter has a stronger relationship with performance)",
        fontsize=11, pad=10,
    )
    cbar = fig.colorbar(im, ax=ax_top, fraction=0.022, pad=0.01)
    cbar.set_label("Δ CPR (max − min)", fontsize=9)
    _annotate(ax_top, sens, fmt=".3f", contrast_threshold=0.55)

    # === Bottom: 5 landscape heatmaps (axis × value, color = CPR vs baseline) ===
    for k, mt in enumerate(ORDER):
        ax = fig.add_subplot(gs[1, k])
        mat = _build_landscape_matrix(blocks, mt)
        base = baselines[mt]
        # Diverging colormap centered at baseline (red = worse, blue = better);
        # if baseline missing, fall back to vmin/vmax of the block.
        if base is not None:
            half = max(np.nanmax(mat) - base, base - np.nanmin(mat), 1e-6)
            vmin, vmax_b = base - half, base + half
        else:
            vmin, vmax_b = np.nanmin(mat), np.nanmax(mat)
        im2 = ax.imshow(mat, cmap="RdBu", aspect="auto", vmin=vmin, vmax=vmax_b)
        ax.set_xticks(range(len(WEIGHT_VALUES)))
        ax.set_xticklabels(WEIGHT_VALUES, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(len(PATHS)))
        ax.set_yticklabels(PATHS, fontsize=9)
        title = f"{mt[0]} / {mt[1]}"
        if base is not None:
            title += f"\nbaseline CPR = {base:.3f}"
        ax.set_title(title, fontsize=9)

        # Hatch the baseline column (value=1.0) so it's visually obvious
        baseline_col = WEIGHT_VALUES.index("1.0")
        for i in range(len(PATHS)):
            ax.add_patch(plt.Rectangle((baseline_col - 0.5, i - 0.5), 1, 1,
                                        fill=False, edgecolor="black",
                                        linewidth=0.6, linestyle=":"))
        _annotate(ax, mat, fmt=".2f", contrast_threshold=0.55)

    fig.suptitle(
        "EAP+FrozenNorm+Scaling — parameter ↔ CPR association",
        fontsize=13, y=0.995,
    )

    # x/y labels for the landscape row, shared once
    fig.text(0.5, 0.025,
             "Bottom row: axis × weight value → CPR.  Color: CPR vs. block baseline "
             "(red = worse, blue = better). Dotted column = uniform 1.0 baseline.",
             ha="center", fontsize=9, color="0.3")

    return fig


def main() -> None:
    fig = make_figure()
    save_figure(fig, PLOT_OUT)


if __name__ == "__main__":
    main()
