import os
import glob
import pickle
import re
from collections import defaultdict

import matplotlib.pyplot as plt

par_dir = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(par_dir, "results")
PLOTS_DIR = os.path.join(par_dir, "plots")

_FNAME_RE = re.compile(r"^(.+?)_([\w.]+)_(test|validation)_abs-(True|False)\.pkl$")


def load_results() -> dict[tuple[str, str], dict[str, dict]]:
    """Return {(task, model): {method: {weighted_edge_counts, faithfulnesses}}}."""
    data: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)

    for pkl_path in glob.glob(os.path.join(RESULTS_DIR, "**", "*.pkl"), recursive=True):
        fname = os.path.basename(pkl_path)
        m = _FNAME_RE.match(fname)
        if m is None:
            continue

        task, model, _split, _abs = m.groups()
        method = os.path.basename(os.path.dirname(pkl_path))

        with open(pkl_path, "rb") as fh:
            result = pickle.load(fh)

        if "weighted_edge_counts" not in result or "faithfulnesses" not in result:
            continue

        data[(task, model)][method] = {
            "weighted_edge_counts": result["weighted_edge_counts"],
            "faithfulnesses": result["faithfulnesses"],
        }

    return data


def plot_faithfulness(
    task: str,
    model: str,
    method_data: dict[str, dict],
    out_path: str,
) -> None:
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
    ax.set_title(f"{task} — {model}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    FAITHFULNESS_DIR = os.path.join(PLOTS_DIR, 'faithfulness')
    os.makedirs(FAITHFULNESS_DIR, exist_ok=True)
    data = load_results()
    if not data:
        print(f"No results found under {RESULTS_DIR}")
        return

    for (task, model), method_data in sorted(data.items()):
        out_path = os.path.join(FAITHFULNESS_DIR, f"faithfulness_{task}_{model}.png")
        plot_faithfulness(task, model, method_data, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()