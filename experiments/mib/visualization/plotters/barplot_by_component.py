import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from experiments.mib.visualization.visualization import Plotter
from experiments.mib.visualization.utils import infer_model_dims, load_metric, apply_top_percent_mask, save_figure, PLOTS_DIR

def _stats(a: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) of finite values; (nan, nan) if none exist."""
    flat = a[np.isfinite(a)].ravel()
    if flat.size == 0:
        return np.nan, np.nan
    return float(np.mean(flat)), float(np.std(flat))


_SOURCE_SUBTYPES = ["emb", "attn_out", "mlp_out"]
_DEST_SUBTYPES = ["q_in", "k_in", "v_in", "mlp_in", "logits"]

# (mean_array, std_array) per subtype
_AggResult = dict[str, tuple[np.ndarray, np.ndarray]]


class BarplotByComponent(Plotter):
    """Two bar plots per (method, task, model): one by source subcomponent, one by destination."""

    def plot(self) -> None:
        for method in self.methods:
            for task in self.tasks:
                for model in self.models:
                    metric = load_metric(method, task, model, self.data_type)
                    if metric is None:
                        print(f"No data for: {method}, {task}, {model}. Skipping...")
                        continue
                    if self.percent < 1.0:
                        metric = apply_top_percent_mask(metric, method, task, model, self.percent, self.use_abs)
                    n_layers, n_heads = infer_model_dims(*metric.shape)

                    src_labels, src_data = self._aggregate_by_source(metric, n_layers, n_heads)
                    dst_labels, dst_data = self._aggregate_by_dest(metric, n_layers, n_heads)

                    pct_suffix = (
                        f"_top{int(self.percent * 100)}pct{'_abs' if self.use_abs else ''}"
                        if self.percent < 1.0 else ""
                    )
                    title_base = f"{method} — {task}_{model}  [{self.data_type}{pct_suffix}]"
                    out_base = os.path.join(
                        PLOTS_DIR, "component_barplot", method, f"{task}_{model}_{self.data_type}{pct_suffix}"
                    )
                    save_figure(
                        self._plot_bars(src_data, src_labels, f"{title_base}  (by source)"),
                        f"{out_base}_source.png",
                    )
                    save_figure(
                        self._plot_bars(dst_data, dst_labels, f"{title_base}  (by destination)"),
                        f"{out_base}_dest.png",
                    )

    def _aggregate_by_source(
        self,
        metric: np.ndarray,
        n_layers: int,
        n_heads: int,
    ) -> tuple[list[str], _AggResult]:
        """Return (x_labels, {subtype: (means, stds)}) aggregated over all destination columns.

        x_labels = ["emb", "L0", ..., "L_{n_layers-1}"]
        emb only has a value at index 0; attn_out/mlp_out only at layer indices.
        """
        x_labels = ["emb"] + [f"L{L}" for L in range(n_layers)]
        n_x = len(x_labels)
        means = {s: np.full(n_x, np.nan) for s in _SOURCE_SUBTYPES}
        stds  = {s: np.full(n_x, np.nan) for s in _SOURCE_SUBTYPES}

        means["emb"][0], stds["emb"][0] = _stats(metric[0, :])

        for L in range(n_layers):
            rb = 1 + L * (n_heads + 1)
            attn_block = metric[rb : rb + n_heads, :]   # (n_heads, n_dest)
            means["attn_out"][1 + L], stds["attn_out"][1 + L] = _stats(attn_block)
            means["mlp_out"][1 + L],  stds["mlp_out"][1 + L]  = _stats(metric[rb + n_heads, :])

        return x_labels, {s: (means[s], stds[s]) for s in _SOURCE_SUBTYPES}

    def _aggregate_by_dest(
        self,
        metric: np.ndarray,
        n_layers: int,
        n_heads: int,
    ) -> tuple[list[str], _AggResult]:
        """Return (x_labels, {subtype: (means, stds)}) aggregated over all source rows.

        x_labels = ["L0", ..., "L_{n_layers-1}", "logits"]
        q_in/k_in/v_in/mlp_in only have values at layer indices; logits only at the last index.
        """
        x_labels = [f"L{L}" for L in range(n_layers)] + ["logits"]
        n_x = len(x_labels)
        means = {s: np.full(n_x, np.nan) for s in _DEST_SUBTYPES}
        stds  = {s: np.full(n_x, np.nan) for s in _DEST_SUBTYPES}

        for L in range(n_layers):
            cb = L * (3 * n_heads + 1)
            for key, sl in [
                ("q_in",   metric[:, cb : cb + n_heads]),
                ("k_in",   metric[:, cb + n_heads : cb + 2 * n_heads]),
                ("v_in",   metric[:, cb + 2 * n_heads : cb + 3 * n_heads]),
                ("mlp_in", metric[:, cb + 3 * n_heads : cb + 3 * n_heads + 1]),
            ]:
                means[key][L], stds[key][L] = _stats(sl)

        means["logits"][-1], stds["logits"][-1] = _stats(metric[:, -1])

        return x_labels, {s: (means[s], stds[s]) for s in _DEST_SUBTYPES}

    def _plot_bars(
        self,
        data: _AggResult,
        x_labels: list[str],
        title: str,
    ) -> Figure:
        subtypes = list(data.keys())
        n_subtypes = len(subtypes)
        n_x = len(x_labels)
        bar_width = 0.8 / n_subtypes
        x = np.arange(n_x)

        fig, ax = plt.subplots(figsize=(max(8, n_x * 0.7), 5))

        legend_handles = []
        for i, stype in enumerate(subtypes):
            means, stds = data[stype]
            valid = ~np.isnan(means)
            offset = (i - (n_subtypes - 1) / 2) * bar_width
            bars = ax.bar(
                x[valid] + offset,
                means[valid],
                bar_width,
                yerr=stds[valid] if self.show_errorbars else None,
                error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8, "alpha": 0.6},
                alpha=0.8,
            )
            legend_handles.append(Patch(facecolor=bars[0].get_facecolor(), alpha=0.8, label=stype))

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Layer")
        ax.set_ylabel(self.data_type)
        if self.data_type != 'scores':
            ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(handles=legend_handles, fontsize=8, loc="best")
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        return fig
