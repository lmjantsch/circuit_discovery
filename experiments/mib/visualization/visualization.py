import argparse
import os

from abc import ABC, abstractmethod

from experiments.mib.visualization.utils import CIRCUITS_DIR, RESULTS_DIR, PLOTS_DIR


class Plotter(ABC):

    def __init__(self, args: argparse.Namespace):
        self.plot_type = args.plot_type
        self.methods = args.methods
        self.tasks = args.tasks
        self.models = args.models
        self.data_type = args.data_type
        self.show_errorbars = args.show_errorbars
        self.percent = args.percent
        self.use_abs = args.use_abs
        self.base_method = args.base_method

    @abstractmethod
    def plot(self):
        pass


def main() -> None:
    # Imported here to avoid a circular import (plot modules import Plotter from this file).
    from experiments.mib.visualization.plotters.circuit_heatmap import CircuitHeatmap
    from experiments.mib.visualization.plotters.faithfulness_curve import FaithfulnessCurve
    from experiments.mib.visualization.plotters.cv_histplot import CVHistplot
    from experiments.mib.visualization.plotters.barplot_by_component import BarplotByComponent

    plot_mapping = {
        "circuit_heatmap": {"type": "circuit", "cls": CircuitHeatmap},
        "faithfulness_curve": {"type": "results", "cls": FaithfulnessCurve},
        "cv_histplot": {"type": "circuit", "cls": CVHistplot},
        "component_barplot": {"type": "circuit", "cls": BarplotByComponent},
    }

    parser = argparse.ArgumentParser(description="MIB visualization CLI")
    parser.add_argument(
        "plot_type",
        choices=plot_mapping.keys(),
        metavar="PLOT",
        help="Which plot to generate (circuit_heatmap, faithfulness_curve, cv_histplot, component_barplot)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Filter by method directory name; default: all",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["ioi", "mcqa"],
        default=["ioi", "mcqa"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["gpt2", "llama3", "qwen2.5", "gemma2"],
        default=["gpt2", "llama3", "qwen2.5", "gemma2"],
    )
    parser.add_argument(
        "--data_type",
        choices=["scores", "variance", "cv"],
        default="scores",
        help="Which metric to plot: scores, variance, or coefficient of variation (cv)",
    )
    parser.add_argument(
        "--show-errorbars",
        action="store_true",
        help="Show standard deviation error bars on bar plots",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=1.0,
        metavar="P",
        help="Only include the top P fraction of edges by score (0.0–1.0, default: 1.0 = all)",
    )
    parser.add_argument(
        "--abs",
        action="store_true",
        dest="use_abs",
        help="Rank edges by absolute score when applying --percent",
    )
    parser.add_argument(
        "--base_method",
        default=None,
        metavar="METHOD",
        help="Subtract this method's scores to plot the residual (method - base_method)",
    )
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    entry = plot_mapping[args.plot_type]
    if args.methods is None:
        src = CIRCUITS_DIR if entry["type"] == "circuit" else RESULTS_DIR
        args.methods = os.listdir(src)

    plotter: Plotter = entry["cls"](args)
    plotter.plot()


if __name__ == "__main__":
    main()
