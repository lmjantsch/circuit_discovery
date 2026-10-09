"""
Render circuit-attribution score heatmaps (in the style of
`circuit_heatmap.py`) for the best-per-block EAP-frnorm sweep result.

For every (model, task) block we emit three heatmaps that reuse
`CircuitHeatmap._create_plot` for layout and colormap:
  - baseline.png   :  scores.pt of the uniform-1.0 baseline circuit.
  - best.png       :  scores.pt of the highest-CPR perturbation.
  - diff.png       :  best − baseline (drawn with block grid lines, same as
                      the existing `--base_method` flow in CircuitHeatmap).

Output: experiments/mib/<GRID_DIR_NAME>/plots/circuit_heatmap/
       <task>_<model>__{baseline,best,diff}.png

Source root is configurable via the GRID_DIR_NAME env var (default
`frozen_grid_search`).

Run:
  /raid/conda/envs/dacslab_djk_mib/bin/python \
      -m experiments.mib.visualization.plotters.grid_search_circuit_heatmap
"""

import csv
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from experiments.mib.visualization.plotters.circuit_heatmap import CircuitHeatmap
from experiments.mib.visualization.utils import save_figure


PATHS = ("q", "k", "v", "gate", "up")
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
CIRCUITS_DIR = os.path.join(GRID_DIR, "circuits")
GRID_REPORT_CSV = os.path.join(GRID_DIR, "report.csv")
PLOT_DIR = os.path.join(GRID_DIR, "plots", "circuit_heatmap")


def _method_name(q: str, k: str, v: str, gate: str, up: str) -> str:
    return f"eap_frnorm_q{q}_k{k}_v{v}_g{gate}_u{up}"


def _load_scores(method: str, task: str, model: str) -> Optional[np.ndarray]:
    path = os.path.join(CIRCUITS_DIR, method, f"{task}_{model}", "scores.pt")
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True).float().numpy()


def _find_best_per_block(rows: list[dict]) -> dict[tuple[str, str], dict]:
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
        non_baseline = [r for r in block if not all(r[p] == "1.0" for p in PATHS)]
        if not non_baseline:
            continue
        best = max(non_baseline, key=lambda r: float(r["cpr"]))
        out[mt] = {"baseline": baseline, "best": best}
    return out


def _short_pert_label(row: dict) -> str:
    """e.g., 'up=0.5' for a per-axis sweep row."""
    off = [(p, row[p]) for p in PATHS if row[p] != "1.0"]
    if len(off) == 1:
        return f"{off[0][0]}={off[0][1]}"
    if not off:
        return "(baseline)"
    return ", ".join(f"{p}={v}" for p, v in off)


def main() -> None:
    rows = list(csv.DictReader(open(GRID_REPORT_CSV)))
    summaries = _find_best_per_block(rows)

    # Reuse the rendering helpers without invoking Plotter.__init__ (no CLI args needed).
    ch = CircuitHeatmap.__new__(CircuitHeatmap)

    for mt in ORDER:
        info = summaries.get(mt)
        if info is None:
            print(f"[skip] {mt}: no rows in CSV")
            continue
        baseline_row = info["baseline"]
        best_row = info["best"]
        if baseline_row is None:
            print(f"[skip] {mt}: baseline (uniform 1.0) not yet measured")
            continue

        model, task = mt
        base_method = _method_name(*[baseline_row[p] for p in PATHS])
        best_method = _method_name(*[best_row[p] for p in PATHS])

        arr_base = _load_scores(base_method, task, model)
        arr_best = _load_scores(best_method, task, model)
        if arr_base is None or arr_best is None:
            print(f"[skip] {mt}: scores.pt missing for baseline or best")
            continue

        base_cpr = float(baseline_row["cpr"])
        best_cpr = float(best_row["cpr"])
        delta = best_cpr - base_cpr
        pert = _short_pert_label(best_row)

        # baseline circuit
        fig = ch._create_plot(
            arr_base,
            title=f"baseline (uniform 1.0) — {task}_{model}   CPR={base_cpr:.3f}",
        )
        save_figure(fig, os.path.join(PLOT_DIR, f"{task}_{model}__baseline.png"))

        # best perturbation circuit
        fig = ch._create_plot(
            arr_best,
            title=f"best `{pert}` — {task}_{model}   CPR={best_cpr:.3f}",
        )
        save_figure(fig, os.path.join(PLOT_DIR, f"{task}_{model}__best.png"))

        # diff (best − baseline)
        fig = ch._create_plot(
            arr_best - arr_base,
            title=(f"best (`{pert}`)  −  baseline   {task}_{model}   "
                   f"ΔCPR = {delta:+.3f}"),
            show_grid=True,
        )
        save_figure(fig, os.path.join(PLOT_DIR, f"{task}_{model}__diff.png"))


if __name__ == "__main__":
    main()
