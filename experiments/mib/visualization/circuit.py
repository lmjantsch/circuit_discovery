"""
Two plot types:

1. Faithfulness curves — per (task, model), one line per method.
   Output: experiments/mib/results/plots/faithfulness/<task>_<model>.png

2. Circuit score heatmaps — one per scores.pt file, mirroring the circuits/ tree.
   Output: experiments/mib/results/plots/circuits/<method>/<task_model>/scores.png
"""

import os
import glob
import pickle
import re
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt


PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
CIRCUITS_DIR = os.path.join(os.path.dirname(__file__), "circuits")


def _infer_model_dims(n_forward: int, n_backward: int) -> tuple[int, int]:
    """Return (n_layers, n_heads) from score matrix shape.

    n_forward  = 1 + n_layers * (n_heads + 1)
    n_backward = n_layers * (3 * n_heads + 1) + 1
    """
    nf = n_forward - 1
    nb = n_backward - 1
    n_heads = (nb - nf) // (3 * nf - nb)
    n_layers = nf // (n_heads + 1)
    return n_layers, n_heads


def _build_padded(
    arr: np.ndarray,
    n_layers: int,
    n_heads: int,
    gap_large: float = 1,
    gap_small: float = 0.35,
    gap_tiny: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[str], list[float], list[str]]:
    """Build a pcolormesh-ready array with NaN gap cells at block boundaries.

    NaN cells are sized by gap_large / gap_small / gap_tiny in coordinate units (data cells = 1.0).
    Returns (padded, x_edges, y_edges, x_tick_pos, x_tick_labels, y_tick_pos, y_tick_labels).

    Padded row layout:
      border | emb | gap_large | [head0 | tiny | head1 | ... | tiny | head_n-1 | gap_small | mlp | gap_large?] * n_layers | border

    Padded col layout:
      border | [q_head0 | tiny | ... | q_head_n-1 | gap_small | k... | gap_small | v... | gap_small | mlp | gap_large?] * n_layers | gap_large | lm | border

    Tick index formulas (indices into padded array, accounting for all gaps):
      Y emb:     rows [1, 2)
      Y layer L: rows [s, e)  where s = 3 + L*(2*n_heads+2),  e = s + 2*n_heads + 1
      X layer L: cols [s, e)  where s = 1 + L*(6*n_heads+2),  e = s + 6*n_heads + 1
      X lm_head: cols [-3, -2) in x_edges
    """
    n_orig_rows, n_orig_cols = arr.shape

    # ── build row-padded array ────────────────────────────────────────────────
    row_slices: list[np.ndarray] = []
    row_heights: list[float] = []

    def _rslice(data: np.ndarray) -> None:
        row_slices.append(data)
        row_heights.extend([1.0] * data.shape[0])

    def _rgap(h: float) -> None:
        row_slices.append(np.full((1, n_orig_cols), np.nan))
        row_heights.append(h)

    _rgap(gap_large)         # border
    _rslice(arr[0:1])        # embedding
    _rgap(gap_large)         # emb | L0

    for L in range(n_layers):
        rb = 1 + L * (n_heads + 1)
        for h in range(n_heads):
            _rslice(arr[rb + h:rb + h + 1])   # attention head h (2D slice)
            if h < n_heads - 1:
                _rgap(gap_tiny)               # tiny gap between heads
        _rgap(gap_small)                       # attn block | mlp
        _rslice(arr[rb + n_heads:rb + n_heads + 1])  # mlp
        if L < n_layers - 1:
            _rgap(gap_large)                   # L | L+1

    _rgap(gap_large)         # border

    padded_rows = np.concatenate(row_slices, axis=0)
    y_edges = np.concatenate([[0.0], np.cumsum(row_heights)])

    # ── build column-padded array ─────────────────────────────────────────────
    col_slices: list[np.ndarray] = []
    col_widths: list[float] = []
    nrows = padded_rows.shape[0]

    def _cslice(data: np.ndarray) -> None:
        col_slices.append(data)
        col_widths.extend([1.0] * data.shape[1])

    def _cgap(w: float) -> None:
        col_slices.append(np.full((nrows, 1), np.nan))
        col_widths.append(w)

    _cgap(gap_large)  # border

    for L in range(n_layers):
        cb = L * (3 * n_heads + 1)
        for i in range(3):                          # q, k, v
            for h in range(n_heads):
                _cslice(padded_rows[:, cb + i * n_heads + h:cb + i * n_heads + h + 1])
                if h < n_heads - 1:
                    _cgap(gap_tiny)               # tiny gap between heads
            _cgap(gap_small)                       # q|k, k|v, v|mlp
        _cslice(padded_rows[:, cb + 3 * n_heads:cb + 3 * n_heads + 1])  # mlp
        if L < n_layers - 1:
            _cgap(gap_large)                       # L | L+1

    _cgap(gap_large)                               # last L | lm
    lm_col = n_layers * (3 * n_heads + 1)
    _cslice(padded_rows[:, lm_col:lm_col + 1])    # lm_head
    _cgap(gap_large)                               # border

    padded = np.concatenate(col_slices, axis=1)
    x_edges = np.concatenate([[0.0], np.cumsum(col_widths)])

    # ── tick positions ────────────────────────────────────────────────────────
    # Y: emb at padded row 1; layer L block starts at row s = 3 + L*(2*n_heads+2)
    y_tick_pos = [(y_edges[1] + y_edges[2]) / 2]  # embedding
    y_tick_labels = ["emb"]
    for L in range(n_layers):
        s = 3 + L * (2 * n_heads + 2)   # first head row of layer L
        e = s + 2 * n_heads + 1          # one past mlp row
        y_tick_pos.append((y_edges[s] + y_edges[e]) / 2)
        y_tick_labels.append(f"L{L}")

    # X: layer L block starts at col s = 1 + L*(6*n_heads+2)
    x_tick_pos: list[float] = []
    x_tick_labels: list[str] = []
    for L in range(n_layers):
        s = 1 + L * (6 * n_heads + 2)   # first q-head col of layer L
        e = s + 6 * n_heads + 1          # one past mlp col
        x_tick_pos.append((x_edges[s] + x_edges[e]) / 2)
        x_tick_labels.append(f"L{L}")
    x_tick_pos.append((x_edges[-3] + x_edges[-2]) / 2)  # lm_head ([-3]=start, [-2]=end, [-1]=border)
    x_tick_labels.append("lm")

    return padded, x_edges, y_edges, x_tick_pos, x_tick_labels, y_tick_pos, y_tick_labels


def plot_circuit_scores(scores_path: str, out_path: str) -> None:
    t = torch.load(scores_path, map_location="cpu", weights_only=True)
    arr = t.float().numpy()  # (n_forward, n_backward)
    n_forward, n_backward = arr.shape
    n_layers, n_heads = _infer_model_dims(n_forward, n_backward)

    padded, x_edges, y_edges, x_tick_pos, x_tick_labels, y_tick_pos, y_tick_labels = \
        _build_padded(arr, n_layers, n_heads)

    w = n_backward / 10.0
    h = n_forward / 10.0
    fig, ax = plt.subplots(figsize=(w, h))

    vmax = float(np.percentile(np.abs(arr), 99.9)) or 1.0
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("white")

    im = ax.pcolormesh(
        x_edges, y_edges,
        np.ma.masked_invalid(padded),
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )
    ax.invert_yaxis()

    ax.set_yticks(y_tick_pos)
    ax.set_yticklabels(y_tick_labels, fontsize=6)
    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(x_tick_labels, fontsize=6, rotation=90)

    ax.set_xlabel("Backward  [q · k · v · mlp  per layer,  lm_head]")
    ax.set_ylabel("Forward  [emb,  attn · mlp  per layer]")

    task_model = os.path.basename(os.path.dirname(scores_path))
    method = os.path.basename(os.path.dirname(os.path.dirname(scores_path)))
    ax.set_title(f"{method} — {task_model}  (L={n_layers}, H={n_heads})")

    fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01, label="score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pt_files = sorted(glob.glob(os.path.join(CIRCUITS_DIR, "**", "scores.pt"), recursive=True))
    if not pt_files:
        print(f"No scores.pt files found under {CIRCUITS_DIR}")
        return

    circuits_plots_dir = os.path.join(PLOTS_DIR, "circuits")
    for pt_path in pt_files:
        rel = os.path.relpath(pt_path, CIRCUITS_DIR)          # eap/ioi_gpt2/scores.pt
        out_path = os.path.join(circuits_plots_dir, os.path.split(rel)[0] + ".png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plot_circuit_scores(pt_path, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
