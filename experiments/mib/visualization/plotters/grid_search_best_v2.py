"""
v2 of grid_search_best: cleaner layout for slides / talks.

Differences from grid_search_best.py:
  - Subplot grid is 2 (rows) x 3 (cols). Top row = IOI (gpt2, qwen2.5, gemma2),
    bottom row = MCQA (gpt2 empty since the benchmark has no gpt2/mcqa,
    qwen2.5, gemma2). Cleaner task-row grouping.
  - No per-subplot legends. A single shared legend sits below the figure
    in its own box.
  - Titles show only "model / task" (no CPR / Δ numbers in legend or title).
  - Curves still drawn the same way; baseline + 5 best-per-axis lines + the
    overall-best line is starred. Phantom legend markers indicate dead axes
    or baseline-tied axes when applicable.

Run:
  /raid/conda/envs/dacslab_djk_mib/bin/python \
      -m experiments.mib.visualization.plotters.grid_search_best_v2
"""

import csv
import os
import pickle
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

from experiments.mib.visualization.utils import save_figure


PATHS = ("q", "k", "v", "gate", "up")
AXIS_COLORS = {
    "q": "#1f77b4",
    "k": "#ff7f0e",
    "v": "#2ca02c",
    "gate": "#d62728",
    "up": "#9467bd",
}

# Layout intent:
#   Row 0 (IOI):   gpt2  | qwen2.5 | gemma2
#   Row 1 (MCQA):           qwen2.5 | gemma2          (centered: gpt2/mcqa absent)
# Implemented via a 2 x 6 GridSpec so each subplot spans 2 columns,
# letting the bottom row's two plots sit at cols [1:3] and [3:5].
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
GRID_RESULTS_DIR = os.path.join(GRID_DIR, "results")
GRID_REPORT_CSV = os.path.join(GRID_DIR, "report.csv")
PLOT_OUT = os.path.join(GRID_DIR, "plots", "best_params_faithfulness_v2.png")

SWEEP_CONFIG = {
    "frozen_grid_search": {"method_prefix": "eap_frnorm",  "sweep_title": "EAP+FrozenNorm+Scaling"},
    "grid_search":        {"method_prefix": "eap_scaling", "sweep_title": "EAP+Scaling"},
}
_cfg = SWEEP_CONFIG.get(GRID_DIR_NAME, SWEEP_CONFIG["grid_search"])
METHOD_PREFIX = _cfg["method_prefix"]
SWEEP_TITLE = _cfg["sweep_title"]
ABLATION_LEVEL_SUFFIX = "_patching_edge"


def _method_name(q, k, v, gate, up):
    return f"{METHOD_PREFIX}_q{q}_k{k}_v{v}_g{gate}_u{up}"


def _load_curve(method, task, model):
    path = os.path.join(GRID_RESULTS_DIR, f"{method}{ABLATION_LEVEL_SUFFIX}",
                        f"{task}_{model}_test_abs-False.pkl")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return {
        "weighted_edge_counts": d["weighted_edge_counts"],
        "faithfulnesses":       d["faithfulnesses"],
        "area_under":           d["area_under"],
    }


def _collect_per_block():
    rows = list(csv.DictReader(open(GRID_REPORT_CSV)))
    blocks = defaultdict(list)
    for r in rows:
        if r["cpr"] in ("-", ""):
            continue
        blocks[(r["model"], r["task"])].append(r)
    out = {}
    for mt, block in blocks.items():
        baseline = next((r for r in block if all(r[p] == "1.0" for p in PATHS)), None)
        best_axes = {}
        for ax in PATHS:
            ax_rows = [r for r in block if all(r[p] == "1.0" for p in PATHS if p != ax)]
            if not ax_rows:
                continue
            cprs = [float(r["cpr"]) for r in ax_rows]
            flat = (max(cprs) - min(cprs)) < 1e-4
            best = max(ax_rows, key=lambda r: float(r["cpr"]))
            if flat:
                status = "flat"
            elif best[ax] == "1.0":
                status = "baseline_tied"
            else:
                status = "improving"
            best_axes[ax] = {
                "value":  best[ax],
                "cpr":    float(best["cpr"]),
                "method": _method_name(best["q"], best["k"], best["v"], best["gate"], best["up"]),
                "status": status,
            }
        out[mt] = {"baseline": baseline, "best_axes": best_axes}
    return out


def _plot_block(ax, model, task, summary):
    baseline_row = summary["baseline"]
    best_axes = summary["best_axes"]

    base_cpr = None
    if baseline_row is not None:
        base_method = _method_name(*[baseline_row[p] for p in PATHS])
        base_curve = _load_curve(base_method, task, model)
        if base_curve is not None:
            base_cpr = base_curve["area_under"]
            ax.plot(base_curve["weighted_edge_counts"], base_curve["faithfulnesses"],
                    color="0.4", linestyle="--", linewidth=2.2,
                    marker="o", markersize=4, zorder=2)

    overall_best_axis = None
    overall_best_cpr = -float("inf")
    for axis_name in PATHS:
        info = best_axes.get(axis_name)
        if info is None or info["status"] != "improving":
            continue
        if info["cpr"] > overall_best_cpr:
            overall_best_cpr = info["cpr"]
            overall_best_axis = axis_name

    for axis_name in PATHS:
        info = best_axes.get(axis_name)
        if info is None or info["status"] != "improving":
            continue
        curve = _load_curve(info["method"], task, model)
        if curve is None:
            continue
        is_best = (axis_name == overall_best_axis)
        ax.plot(curve["weighted_edge_counts"], curve["faithfulnesses"],
                color=AXIS_COLORS[axis_name],
                linewidth=2.4 if is_best else 1.4,
                marker="*" if is_best else "o",
                markersize=11 if is_best else 5,
                zorder=3 if is_best else 2)

    ax.set_xscale("log")
    ax.set_xlabel("Weighted edge count (log)", fontsize=9)
    ax.set_ylabel("Faithfulness", fontsize=9)
    ax.set_title(f"{model} / {task}", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="0.7", linewidth=0.8, linestyle=":", zorder=1)


def _build_global_legend_handles(any_block_summary):
    """Single legend for the whole figure. No CPR numbers."""
    handles = [
        Line2D([0], [0], color="0.4", linestyle="--", linewidth=2.2,
               marker="o", markersize=5, label="baseline (uniform 1.0)"),
    ]
    for ax in PATHS:
        handles.append(Line2D([0], [0], color=AXIS_COLORS[ax],
                              linewidth=1.6, marker="o", markersize=5,
                              label=f"best `{ax}` perturbation"))
    handles.append(Line2D([0], [0], color="black", linewidth=2.4,
                          marker="*", markersize=11,
                          label="overall best (per block)"))
    handles.append(Line2D([0], [0], color="gray", linestyle="None",
                          marker="x", markersize=7,
                          label="axis dead (no effect)"))
    handles.append(Line2D([0], [0], color="gray", linestyle="None",
                          marker="o", markersize=6, markerfacecolor="none",
                          label="best value 1.0 (= baseline)"))
    return handles


def make_figure() -> Figure:
    summaries = _collect_per_block()

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 6, hspace=0.45, wspace=0.55,
                          left=0.05, right=0.98, top=0.94, bottom=0.20)

    for mt, (row, col_slice) in LAYOUT_GS.items():
        ax = fig.add_subplot(gs[row, col_slice])
        if mt not in summaries:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{mt[0]}/{mt[1]}\n(no data)",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        _plot_block(ax, mt[0], mt[1], summaries[mt])

    handles = _build_global_legend_handles(None)
    # Wide legend, 2 rows so each entry has enough horizontal room.
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,                  # 5 + 4 entries → 2 rows
        fontsize=12,
        frameon=True,
        bbox_to_anchor=(0.04, 0.02, 0.92, 0.13),  # x, y, width, height
        mode="expand",
        borderaxespad=0.5,
        handletextpad=0.8,
        columnspacing=1.5,
    )

    fig.suptitle(
        f"{SWEEP_TITLE} — best per-axis weight perturbation vs. uniform baseline",
        fontsize=13, y=0.985,
    )

    return fig


def main() -> None:
    fig = make_figure()
    save_figure(fig, PLOT_OUT)


if __name__ == "__main__":
    main()
