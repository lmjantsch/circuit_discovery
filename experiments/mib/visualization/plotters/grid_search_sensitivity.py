"""
Sensitivity analysis plot — directly mirrors Figure 5 of the DPA workshop
paper (`/home/dacslab/djk/icml2026workshop_DPA/example_paper.pdf`):
  x-axis = per-axis weight value, y-axis = CPR (area_under).
  One line per axis (q, k, v, gate, up). Each line is the per-axis sweep
  with the other four weights held at 1.0.

Layout matches grid_search_best_v2.py:
  Row 0 (IOI):   gpt2 | qwen2.5 | gemma2
  Row 1 (MCQA):           qwen2.5 | gemma2  (centered; gpt2/mcqa absent)

A single shared legend sits below the figure, expanded to fill width.
No CPR numbers in the legend; baseline is drawn as a horizontal dashed line
at the uniform-1.0 CPR for each block.

Run:
  /raid/conda/envs/dacslab_djk_mib/bin/python \
      -m experiments.mib.visualization.plotters.grid_search_sensitivity
"""

import csv
import os
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

from experiments.mib.visualization.utils import save_figure


PATHS = ("q", "k", "v", "gate", "up")
AXIS_COLORS = {
    "q":    "#1f77b4",
    "k":    "#ff7f0e",
    "v":    "#2ca02c",
    "gate": "#d62728",
    "up":   "#9467bd",
}
# The exact weight values we swept — used to override matplotlib's auto-ticks.
WEIGHT_VALUES = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 3.0)

LAYOUT_GS = {
    ("gpt2",    "ioi"):  (0, slice(0, 2)),
    ("qwen2.5", "ioi"):  (0, slice(2, 4)),
    ("gemma2",  "ioi"):  (0, slice(4, 6)),
    ("qwen2.5", "mcqa"): (1, slice(1, 3)),
    ("gemma2",  "mcqa"): (1, slice(3, 5)),
}

GRID_DIR_NAME = os.environ.get("GRID_DIR_NAME", "grid_search")
GRID_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    GRID_DIR_NAME,
)
GRID_REPORT_CSV = os.path.join(GRID_DIR, "report.csv")
PLOT_OUT = os.path.join(GRID_DIR, "plots", "sensitivity_analysis.png")

SWEEP_CONFIG = {
    "frozen_grid_search": {"sweep_title": "EAP+FrozenNorm+Scaling"},
    "grid_search":        {"sweep_title": "EAP+Scaling"},
}
SWEEP_TITLE = SWEEP_CONFIG.get(GRID_DIR_NAME, SWEEP_CONFIG["grid_search"])["sweep_title"]


def _collect_per_block() -> dict:
    """Return {(model, task): {axis: [(weight_value_float, cpr_float), ...]}, plus baseline_cpr}."""
    rows = list(csv.DictReader(open(GRID_REPORT_CSV)))
    raw = defaultdict(list)
    for r in rows:
        if r["cpr"] in ("-", ""):
            continue
        raw[(r["model"], r["task"])].append(r)

    out: dict[tuple[str, str], dict] = {}
    for mt, block in raw.items():
        baseline = next(
            (r for r in block if all(r[p] == "1.0" for p in PATHS)),
            None,
        )
        per_axis: dict[str, list[tuple[float, float]]] = {}
        for ax in PATHS:
            ax_rows = [r for r in block if all(r[p] == "1.0" for p in PATHS if p != ax)]
            ax_rows.sort(key=lambda r: float(r[ax]))
            per_axis[ax] = [(float(r[ax]), float(r["cpr"])) for r in ax_rows]
        out[mt] = {
            "per_axis":     per_axis,
            "baseline_cpr": float(baseline["cpr"]) if baseline is not None else None,
        }
    return out


def _plot_block(ax: plt.Axes, model: str, task: str, summary: dict) -> None:
    per_axis = summary["per_axis"]
    base_cpr = summary["baseline_cpr"]

    # Categorical x-axis: each swept value gets the same horizontal slot.
    val_to_idx = {v: i for i, v in enumerate(WEIGHT_VALUES)}
    baseline_idx = val_to_idx.get(1.0)

    if base_cpr is not None:
        ax.axhline(base_cpr, color="0.4", linestyle="--", linewidth=1.4, zorder=1)

    for axis_name in PATHS:
        pts = per_axis.get(axis_name, [])
        if not pts:
            continue
        # Project actual weight value -> categorical index
        xs = [val_to_idx.get(round(v, 4), None) for v, _ in pts]
        ys = [c for _, c in pts]
        # Drop any points whose weight isn't in our canonical list (shouldn't happen).
        xy = [(x, y) for x, y in zip(xs, ys) if x is not None]
        if not xy:
            continue
        xs_, ys_ = zip(*sorted(xy))
        max_idx = max(range(len(ys_)), key=lambda i: ys_[i])
        ax.plot(xs_, ys_, color=AXIS_COLORS[axis_name],
                linewidth=1.8, marker="o", markersize=5, zorder=2)
        ax.plot([xs_[max_idx]], [ys_[max_idx]],
                color=AXIS_COLORS[axis_name],
                marker="*", markersize=11, linestyle="None", zorder=3)

    ax.set_xlabel("Scale parameter value", fontsize=14)
    ax.set_ylabel("CPR Score", fontsize=14)
    ax.set_title(f"Model: {model.upper()}, Dataset: {task.upper()}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(range(len(WEIGHT_VALUES))))
    ax.set_xticklabels([str(v) for v in WEIGHT_VALUES], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    # Mark baseline x position
    if baseline_idx is not None:
        ax.axvline(baseline_idx, color="0.7", linewidth=0.6, linestyle=":", zorder=0)


def _build_global_legend_handles() -> list:
    handles = [
        Line2D([0], [0], color="0.4", linestyle="--", linewidth=1.4,
               label="baseline (uniform 1.0)"),
    ]
    for ax in PATHS:
        handles.append(
            Line2D([0], [0], color=AXIS_COLORS[ax], linewidth=1.8,
                   marker="o", markersize=5, label=f"`{ax}` axis sweep")
        )
    handles.append(
        Line2D([0], [0], color="black", linestyle="None",
               marker="*", markersize=11, label="best value (max CPR)")
    )
    return handles


def make_figure() -> Figure:
    summaries = _collect_per_block()

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 6,
                          hspace=0.32, wspace=0.55,
                          left=0.06, right=0.98,
                          top=0.96, bottom=0.18)

    for mt, (row, col_slice) in LAYOUT_GS.items():
        ax = fig.add_subplot(gs[row, col_slice])
        if mt not in summaries:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{mt[0]}/{mt[1]}\n(no data)",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        _plot_block(ax, mt[0], mt[1], summaries[mt])

    handles = _build_global_legend_handles()
    # 7 entries split across 2 rows (4 + 3); stretched to full figure width.
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=14,
        frameon=True,
        bbox_to_anchor=(0.03, 0.02, 0.94, 0.13),
        mode="expand",
        borderaxespad=0.3,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.5,
    )

    return fig


def main() -> None:
    fig = make_figure()
    save_figure(fig, PLOT_OUT)


if __name__ == "__main__":
    main()
