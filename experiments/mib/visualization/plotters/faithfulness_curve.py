import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from experiments.mib.visualization.visualization import Plotter
from experiments.mib.visualization.utils import load_faithfulness_data, save_figure, PLOTS_DIR

class FaithfulnessCurve(Plotter):

    def plot(self):

        data = load_faithfulness_data(self.methods, self.tasks, self.models)

        for (task, model) in data:
            fig = self._plot_faithfulness(data[(task, model)], f"{task} — {model}")

            out_path = os.path.join(PLOTS_DIR, 'faithfulness_curve', f"{task}_{model}.png")
            save_figure(fig, out_path)

    def _plot_faithfulness(self, method_data: dict[str, dict], title: str) -> Figure:
        """Render faithfulness curves for one (task, model) pair.

        Args:
            title: Plot title string.
            method_data: {method_name: {weighted_edge_counts, faithfulnesses}}.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        for method, values in sorted(method_data.items()):
            ax.plot(
                values["weighted_edge_counts"],
                values["faithfulnesses"],
                marker="o",
                markersize=4,
                label=method,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Weighted edge count (log scale)")
        ax.set_ylabel("Faithfulness")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
