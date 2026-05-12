import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde
from scipy.stats import gaussian_kde

from experiments.mib.visualization.visualization import Plotter
from experiments.mib.visualization.utils import load_metric, apply_top_percent_mask, apply_layer_mask, save_figure, PLOTS_DIR


class CVHistplot(Plotter):

    def plot(self) -> None:
        pct_suffix = (
            f"_top{int(self.percent * 100)}pct{'_abs' if self.use_abs else ''}"
            if self.percent < 1.0 else ""
        )
        layer_suffix = (
            (f"_drop-first{self.ignore_first}" if self.ignore_first is not None else "")
            + (f"_drop-last{self.ignore_last}" if self.ignore_last is not None else "")
        )
        for task in self.tasks:
            for model in self.models:
                method_cvs: dict[str, np.ndarray] = {}
                for method in self.methods:
                    cv_mat = load_metric(method, task, model, self.data_type)
                    if cv_mat is None:
                        print(f"No data for: {method}, {task}, {model}. Skipping...")
                        continue
                    if self.ignore_first is not None or self.ignore_last is not None:
                        cv_mat = apply_layer_mask(cv_mat, self.ignore_first, self.ignore_last)
                    if self.percent < 1.0:
                        cv_mat = apply_top_percent_mask(cv_mat, method, task, model, self.percent, self.use_abs)
                    flat = cv_mat[np.isfinite(cv_mat)].ravel()
                    if flat.size > 0:
                        method_cvs[method] = flat

                if not method_cvs:
                    continue

                fig = self._create_plot(method_cvs, f"{model} — {task.upper()}", self.data_type)
                out_path = os.path.join(PLOTS_DIR, "cv_histplot", f"{task}_{model}_{self.data_type}{pct_suffix}{layer_suffix}.png")
                save_figure(fig, out_path)

    def _create_plot(self, method_cvs: dict[str, np.ndarray], title: str, data_type: str) -> Figure:
        """Render overlapping CV histograms for each method.

        Args:
            method_cvs: {method_name: 1-D array of CV values}.
            title: Plot title string.
            data_type: "cv_in_between" or "cv_within".
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        all_cv = np.concatenate(list(method_cvs.values()))
        all_cv = all_cv[all_cv > 0]
        p_lo, p_hi = np.percentile(all_cv, [0, 99])

        mean_handle = plt.Line2D([], [], color="gray", linestyle="--", linewidth=1.0, label="mean")
        median_handle = plt.Line2D([], [], color="gray", linestyle=":", linewidth=1.0, label="median")

        for method, cv in sorted(method_cvs.items()):
            cv_pos = cv[cv > 0]
            mean_cv = float(np.mean(cv_pos))
            median_cv = float(np.median(cv_pos))

            # # Histogram version
            # cv_in_range = cv_pos[(cv_pos >= p_lo) & (cv_pos <= p_hi)]
            # bins = np.linspace(p_lo, p_hi, 80)
            # _, _, patches = ax.hist(
            #     cv_in_range,
            #     bins=bins,
            #     density=True,
            #     alpha=0.45,
            #     label=method,
            #     histtype="stepfilled",
            #     linewidth=0.8,
            # )
            # color = patches[0].get_facecolor()

            x_grid = np.linspace(p_lo, p_hi, 500)
            kde = gaussian_kde(cv_pos, bw_method="scott")
            density = kde(x_grid)
            (line,) = ax.plot(x_grid, density, linewidth=1.5, label=method)
            color = line.get_color()
            # ax.fill_between(x_grid, density, alpha=0.2, color=color)

            # ax.axvline(mean_cv, color=color, linestyle="-", linewidth=1.2, alpha=0.9, zorder=5)
            ax.axvline(median_cv, color=color, linestyle=":", linewidth=1.2, alpha=0.8, zorder=5)

        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=15))
        ax.set_xlabel(r"$\sigma\,/\,|\mu|$")
        ax.set_ylabel("Density")
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [median_handle], labels + ["median"], fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
