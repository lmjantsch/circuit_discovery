"""
Plot the faithfulness curves that explain the best per-axis EAP-frnorm
sweep result for each (model, task) block.

CPR (`area_under` in each result pkl) is the area under the
faithfulness-vs-edge-count curve. So plotting these curves alongside the
uniform baseline is the most direct visual answer to "why does the best
weight perturbation outperform the baseline?".

Source root is configurable via the GRID_DIR_NAME env var; default is
`frozen_grid_search` (the EAP+FrozenNorm+Scaling sweep).

The plot consumes:
  - experiments/mib/<GRID_DIR_NAME>/report.csv          (which combo is best per axis)
  - experiments/mib/<GRID_DIR_NAME>/results/<method>_patching_edge/
      <task>_<model>_test_abs-False.pkl                 (faithfulness data)

Run:
  /raid/conda/envs/dacslab_djk_mib/bin/python -m experiments.mib.visualization.plotters.grid_search_best
"""

import csv
import os
import pickle
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from experiments.mib.visualization.utils import save_figure


PATHS = ("q", "k", "v", "gate", "up")
ORDER = (
    ("gpt2", "ioi"),
    ("qwen2.5", "ioi"),
    ("qwen2.5", "mcqa"),
    ("gemma2", "ioi"),
    ("gemma2", "mcqa"),
)
AXIS_COLORS = {
    "q": "#1f77b4",
    "k": "#ff7f0e",
    "v": "#2ca02c",
    "gate": "#d62728",
    "up": "#9467bd",
}

GRID_DIR_NAME = os.environ.get("GRID_DIR_NAME", "grid_search")
GRID_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    GRID_DIR_NAME,
)
GRID_RESULTS_DIR = os.path.join(GRID_DIR, "results")
GRID_REPORT_CSV = os.path.join(GRID_DIR, "report.csv")
PLOT_OUT = os.path.join(GRID_DIR, "plots", "best_params_faithfulness.png")

# Per-sweep config: each grid root uses a different attribution method prefix
# and human-readable title, so the same plotter renders both runs cleanly.
SWEEP_CONFIG = {
    "frozen_grid_search": {
        "method_prefix": "eap_frnorm",
        "sweep_title": "EAP+FrozenNorm+Scaling",
    },
    "grid_search": {
        "method_prefix": "eap_scaling",
        "sweep_title": "EAP+Scaling",
    },
}
_cfg = SWEEP_CONFIG.get(GRID_DIR_NAME, SWEEP_CONFIG["grid_search"])
METHOD_PREFIX = _cfg["method_prefix"]
SWEEP_TITLE = _cfg["sweep_title"]
ABLATION_LEVEL_SUFFIX = "_patching_edge"


def _method_name(q: str, k: str, v: str, gate: str, up: str) -> str:
    return f"{METHOD_PREFIX}_q{q}_k{k}_v{v}_g{gate}_u{up}"


def _load_curve(method: str, task: str, model: str) -> Optional[dict]:
    path = os.path.join(
        GRID_RESULTS_DIR,
        f"{method}{ABLATION_LEVEL_SUFFIX}",
        f"{task}_{model}_test_abs-False.pkl",
    )
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return {
        "weighted_edge_counts": d["weighted_edge_counts"],
        "faithfulnesses": d["faithfulnesses"],
        "area_under": d["area_under"],
    }


def _axis_of(row: dict) -> tuple[Optional[str], Optional[str]]:
    """Return (axis_name, value) if exactly one weight != 1.0; else (None, None)
    for multi-axis or ('baseline', '1.0') for the uniform combo."""
    off = [(p, row[p]) for p in PATHS if row[p] != "1.0"]
    if not off:
        return ("baseline", "1.0")
    if len(off) == 1:
        return off[0]
    return (None, None)


def _collect_per_block() -> dict[tuple[str, str], dict]:
    """For each (model, task), collect best-per-axis combo plus the baseline combo.

    Each axis entry is tagged:
      'flat'         -> sweep gave identical CPR for every value (axis dead, e.g.,
                        gpt2's `gate` since gpt2 has no gated MLP).
      'baseline_tied'-> sweep showed variation but the optimum was value=1.0
                        (= baseline; no perturbation on this axis helps).
      'improving'    -> a non-1.0 value beats the others.

    Returns:
        {(model, task): {
            'baseline':  {'q','k','v','gate','up','cpr','method'},
            'best_axes': {axis: {'value','cpr','method','status'}},
        }}
    """
    rows = list(csv.DictReader(open(GRID_REPORT_CSV)))
    blocks: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["cpr"] in ("-", ""):
            continue
        blocks[(r["model"], r["task"])].append(r)

    out: dict[tuple[str, str], dict] = {}
    for mt, block in blocks.items():
        baseline = next(
            (r for r in block if all(r[p] == "1.0" for p in PATHS)), None
        )
        best_axes: dict[str, dict] = {}
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
                "value": best[ax],
                "cpr": float(best["cpr"]),
                "method": _method_name(best["q"], best["k"], best["v"], best["gate"], best["up"]),
                "status": status,
            }
        out[mt] = {"baseline": baseline, "best_axes": best_axes}
    return out


def _plot_block(ax: plt.Axes, model: str, task: str, summary: dict) -> None:
    baseline_row = summary["baseline"]
    best_axes = summary["best_axes"]

    # baseline curve (gray dashed thick)
    base_label = "baseline (uniform 1.0)"
    base_cpr = None
    if baseline_row is not None:
        base_method = _method_name(*[baseline_row[p] for p in PATHS])
        base_curve = _load_curve(base_method, task, model)
        if base_curve is not None:
            base_cpr = base_curve["area_under"]
            ax.plot(
                base_curve["weighted_edge_counts"],
                base_curve["faithfulnesses"],
                color="0.4", linestyle="--", linewidth=2.2,
                marker="o", markersize=4,
                label=f"{base_label}  CPR={base_cpr:.3f}",
                zorder=2,
            )

    # Determine the overall best axis (only among axes whose perturbation
    # actually beats the baseline — flat or baseline-tied axes don't count).
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
        if info is None:
            continue

        # Phantom legend entries (no curve drawn) for axes that did not produce
        # a meaningful perturbation. They still occupy a slot in the legend so
        # the reader can see all 5 axes per panel.
        if info["status"] == "flat":
            ax.plot(
                [], [],
                color=AXIS_COLORS[axis_name], linestyle="None", marker="x",
                markersize=7, markeredgewidth=1.5,
                label=f"{axis_name}: no effect (axis dead)",
            )
            continue
        if info["status"] == "baseline_tied":
            ax.plot(
                [], [],
                color=AXIS_COLORS[axis_name], linestyle="None", marker="o",
                markersize=6, markerfacecolor="none",
                label=f"{axis_name}: best at 1.0 (= baseline)",
            )
            continue

        curve = _load_curve(info["method"], task, model)
        if curve is None:
            continue
        is_overall_best = (axis_name == overall_best_axis)
        delta = info["cpr"] - base_cpr if base_cpr is not None else 0.0
        sign = "+" if delta >= 0 else ""
        marker = "*" if is_overall_best else "o"
        markersize = 11 if is_overall_best else 5
        lw = 2.4 if is_overall_best else 1.4
        suffix = "  ← BEST" if is_overall_best else ""
        ax.plot(
            curve["weighted_edge_counts"],
            curve["faithfulnesses"],
            color=AXIS_COLORS[axis_name],
            linewidth=lw,
            marker=marker,
            markersize=markersize,
            label=f"{axis_name}={info['value']}  CPR={info['cpr']:.3f} ({sign}{delta:.3f}){suffix}",
            zorder=3 if is_overall_best else 2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Weighted edge count (log)")
    ax.set_ylabel("Faithfulness")
    title = f"{model} / {task}"
    if base_cpr is not None and overall_best_cpr > -float("inf"):
        title += f"\nbaseline CPR={base_cpr:.3f} → best CPR={overall_best_cpr:.3f}  (Δ{overall_best_cpr-base_cpr:+.3f})"
    elif overall_best_cpr > -float("inf"):
        title += f"\nbaseline missing (block in progress); best CPR={overall_best_cpr:.3f}"
    else:
        title += "\nno data"
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.axhline(1.0, color="0.7", linewidth=0.8, linestyle=":", zorder=1)


def make_figure() -> Figure:
    summaries = _collect_per_block()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for i, mt in enumerate(ORDER):
        ax = axes_flat[i]
        if mt not in summaries:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{mt[0]}/{mt[1]}\n(no data)", ha="center", va="center")
            continue
        _plot_block(ax, mt[0], mt[1], summaries[mt])

    # blank the unused 6th panel
    axes_flat[-1].set_axis_off()
    axes_flat[-1].text(
        0.0, 0.95,
        "Faithfulness curves explain CPR\n"
        "(CPR = area under faithfulness vs.\n"
        " weighted-edge-count curve).\n\n"
        "Each colored line = best perturbation\n"
        "for a single weight axis (others held\n"
        "at 1.0). The dashed gray line is the\n"
        "uniform-1.0 baseline. The starred line\n"
        "is the overall best perturbation in\n"
        "the block.\n\n"
        "Legend markers without a line:\n"
        "  x  axis dead (no effect on this model,\n"
        "     e.g. gpt2 has no gated MLP so its\n"
        "     `gate` axis is flat)\n"
        "  o  best value is 1.0 — no perturbation\n"
        "     on that axis beats the baseline.\n\n"
        "Higher curves → larger CPR.",
        fontsize=9, va="top", ha="left",
        family="monospace",
    )

    fig.suptitle(
        f"{SWEEP_TITLE} — best per-axis weight perturbation vs. uniform baseline",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    fig = make_figure()
    save_figure(fig, PLOT_OUT)


if __name__ == "__main__":
    main()
