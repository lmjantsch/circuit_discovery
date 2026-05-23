import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from experiments.mib.visualization.visualization import Plotter
from experiments.mib.visualization.utils import load_sdf_data, save_figure, PLOTS_DIR

_POS_LABELS = ("in", "mid", "out")
_POS_LINESTYLES = {"in": "-", "mid": "--", "out": ":", "all": "-."}


def _pos_indices(n_layers: int, pos_type: str) -> list[int]:
    """Column indices in a (n_samples, n_pos) array for a given position subtype."""
    if pos_type == "all":
        return list(range(n_layers * 3))
    offset = _POS_LABELS.index(pos_type)
    return [L * 3 + offset for L in range(n_layers)]


class SDFCurve(Plotter):
    """Plot residual-state SDF metrics vs percentage of circuit edges or vs layer index.

    Two modes controlled by --x-axis:
      percentage  x = edge percentage (log scale); one line per residual position subtype
      layer       x = layer index; one line per edge percentage, coloured by log-scale cmap
    """

    def __init__(self, args):
        super().__init__(args)
        self.split  = args.split
        self.x_axis = args.x_axis

    def plot(self) -> None:
        data = load_sdf_data(self.methods, self.tasks, self.models, self.split, self.use_abs)
        for (task, model), method_data in data.items():
            for method, sdf_d in method_data.items():
                if self.x_axis == "percentage":
                    fig = self._plot_by_percentage(sdf_d, f"{method} — {task} — {model}")
                else:
                    fig = self._plot_by_layer(sdf_d, f"{method} — {task} — {model}")
                out_path = os.path.join(
                    PLOTS_DIR, "sdf_curve", method, f"{task}_{model}_{self.x_axis}.png"
                )
                save_figure(fig, out_path)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_arrays(sdf_d: dict, component: str) -> list[np.ndarray]:
        """Return list[n_pct] of (n_samples, n_pos) float32 arrays.

        component:
          "outside" — dist_outside  (≥ 0)
          "inside"  — |dist_inside| (≥ 0; raw values are ≤ 0)
          "total"   — SDF = dist_outside + dist_inside (positive outside, negative inside)
        """
        outside = [t.numpy().astype(np.float32) for t in sdf_d["sdf_outside"]]
        inside  = [t.numpy().astype(np.float32) for t in sdf_d["sdf_inside"]]
        if component == "outside":
            return outside
        if component == "inside":
            return [-arr for arr in inside]   # flip sign → depth inside the box (≥ 0)
        return [o + i for o, i in zip(outside, inside)]

    @staticmethod
    def _pct_stats(
        arrays: list[np.ndarray],
        n_layers: int,
        pos_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std across samples (averaged over positions of given subtype) per percentage.

        Returns: means (n_pct,), stds (n_pct,)
        """
        idxs = _pos_indices(n_layers, pos_type)
        means, stds = [], []
        for arr in arrays:                               # (n_samples, n_pos)
            per_sample = arr[:, idxs].mean(axis=1)      # (n_samples,)
            means.append(float(per_sample.mean()))
            stds.append(float(per_sample.std()))
        return np.array(means), np.array(stds)

    @staticmethod
    def _layer_stats(
        arrays: list[np.ndarray],
        n_layers: int,
        pos_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std across samples per (percentage, layer) for a given position subtype.

        Returns: means (n_pct, n_layers), stds (n_pct, n_layers)
        """
        if pos_type == "all":
            idxs_per_layer = [list(range(L * 3, L * 3 + 3)) for L in range(n_layers)]
        else:
            offset = _POS_LABELS.index(pos_type)
            idxs_per_layer = [[L * 3 + offset] for L in range(n_layers)]
        means = np.zeros((len(arrays), n_layers), dtype=np.float32)
        stds  = np.zeros((len(arrays), n_layers), dtype=np.float32)
        for p_id, arr in enumerate(arrays):
            for L in range(n_layers):
                per_sample = arr[:, idxs_per_layer[L]].mean(axis=1)  # (n_samples,)
                means[p_id, L] = per_sample.mean()
                stds[p_id, L]  = per_sample.std()
        return means, stds

    # ------------------------------------------------------------------
    # Plot: by percentage
    # ------------------------------------------------------------------

    def _plot_by_percentage(self, sdf_d: dict, title: str) -> Figure:
        percentages = np.array(sdf_d["percentages"])
        n_layers    = sdf_d["n_layers"]
        pos_types   = list(_POS_LABELS) + ["all"]

        comp_keys   = ("outside", "inside", "total")
        comp_labels = ("dist_outside", "|dist_inside|", "SDF (total)")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, comp_key, comp_label in zip(axes, comp_keys, comp_labels):
            arrays = self._get_arrays(sdf_d, comp_key)
            for pos_type in pos_types:
                means, stds = self._pct_stats(arrays, n_layers, pos_type)
                ls = _POS_LINESTYLES[pos_type]
                (line,) = ax.plot(
                    percentages, means,
                    ls=ls, marker="o", markersize=3,
                    label=pos_type,
                )
                if self.show_errorbars:
                    ax.fill_between(
                        percentages,
                        means - stds, means + stds,
                        alpha=0.15, color=line.get_color(),
                    )

            ax.set_xscale("log")
            ax.set_xlabel("Fraction of circuit edges")
            ax.set_ylabel(comp_label)
            ax.set_title(comp_label)
            ax.legend(fontsize=8, title="position", loc="best")
            ax.grid(True, alpha=0.3)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Plot: by layer
    # ------------------------------------------------------------------

    def _plot_by_layer(self, sdf_d: dict, title: str) -> Figure:
        percentages = list(sdf_d["percentages"])
        n_layers    = sdf_d["n_layers"]
        layers      = np.arange(n_layers)

        norm = LogNorm(vmin=min(percentages), vmax=max(percentages))
        cmap = cm.viridis

        comp_keys   = ("outside", "inside", "total")
        comp_labels = ("dist_outside", "|dist_inside|", "SDF (total)")
        # Separate columns for each position subtype
        pos_types   = list(_POS_LABELS) + ["all"]

        n_rows = len(pos_types)
        n_cols = len(comp_keys)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)

        for col, (comp_key, comp_label) in enumerate(zip(comp_keys, comp_labels)):
            arrays = self._get_arrays(sdf_d, comp_key)
            for row, pos_type in enumerate(pos_types):
                ax = axes[row][col]
                means, stds = self._layer_stats(arrays, n_layers, pos_type)  # (n_pct, n_layers)

                for p_id, pct in enumerate(percentages):
                    color = cmap(norm(pct))
                    ax.plot(layers, means[p_id], color=color, marker="o", markersize=2, linewidth=1)
                    if self.show_errorbars:
                        ax.fill_between(
                            layers,
                            means[p_id] - stds[p_id],
                            means[p_id] + stds[p_id],
                            alpha=0.08, color=color,
                        )

                ax.set_xlabel("Layer")
                ax.set_ylabel(comp_label)
                ax.set_title(f"{comp_label}  [{pos_type}]")
                ax.grid(True, alpha=0.3)

        # Single shared colorbar for the percentage axis
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, label="Fraction of circuit edges", fraction=0.02, pad=0.04, format="%.3g")

        fig.suptitle(title, y=1.01)
        fig.tight_layout()
        return fig
