import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from experiments.mib.visualization.visualization import Plotter
from experiments.mib.visualization.utils import load_metric, apply_top_percent_mask, save_figure, PLOTS_DIR


class CVHistplot(Plotter):

    def plot(self) -> None:
        pct_suffix = (
            f"_top{int(self.percent * 100)}pct{'_abs' if self.use_abs else ''}"
            if self.percent < 1.0 else ""
        )
        for task in self.tasks:
            for model in self.models:
                method_cvs: dict[str, np.ndarray] = {}
                for method in self.methods:
                    cv_mat = load_metric(method, task, model, self.data_type)
                    if cv_mat is None:
                        print(f"No data for: {method}, {task}, {model}. Skipping...")
                        continue
                    if self.percent < 1.0:
                        cv_mat = apply_top_percent_mask(cv_mat, method, task, model, self.percent, self.use_abs)
                    flat = cv_mat[np.isfinite(cv_mat)].ravel()
                    if flat.size > 0:
                        method_cvs[method] = flat

                if not method_cvs:
                    continue

                fig = self._create_plot(method_cvs, f"{task} — {model}{pct_suffix}", self.data_type)
                out_path = os.path.join(PLOTS_DIR, "cv_histplot", f"{task}_{model}_{self.data_type}{pct_suffix}.png")
                save_figure(fig, out_path)

    def _create_plot(self, method_cvs: dict[str, np.ndarray], title: str, data_type: str) -> Figure:
        """Render overlapping CV histograms for each method.

        Args:
            method_cvs: {method_name: 1-D array of CV values}.
            title: Plot title string.
            data_type: "cv_in_between" or "cv_within".
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        # Shared linearly-spaced bin edges across all methods for a fair comparison.
        all_cv = np.concatenate(list(method_cvs.values()))
        all_cv = all_cv[all_cv > 0]
        p_lo, p_hi = np.percentile(all_cv, [2.5, 97.5])
        bins = np.linspace(p_lo, p_hi, 80)

        for method, cv in sorted(method_cvs.items()):
            cv_pos = cv[cv > 0]
            cv_in_range = cv_pos[(cv_pos >= p_lo) & (cv_pos <= p_hi)]
            mean_cv = float(np.mean(cv_pos))
            _, _, patches = ax.hist(
                cv_in_range,
                bins=bins,
                density=True,
                alpha=0.45,
                label=method,
                histtype="stepfilled",
                linewidth=0.8,
            )
            ax.axvline(mean_cv, color=patches[0].get_facecolor(), linestyle="--", linewidth=1.2, alpha=0.9)

        variance_label = "variance_in_between" if data_type == "cv_in_between" else "variance_within"
        ax.set_xlabel(f"Coefficient of variation  ({variance_label} / |score|)")
        ax.set_ylabel("Density  (area = 1,  heights scaled by bin width)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
