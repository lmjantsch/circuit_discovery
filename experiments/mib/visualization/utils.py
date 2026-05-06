import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

_PAR = os.path.dirname(os.path.dirname(__file__))
CIRCUITS_DIR = os.path.join(_PAR, "circuits")
RESULTS_DIR = os.path.join(_PAR, "results")
PLOTS_DIR = os.path.join(_PAR, "plots")


def infer_model_dims(n_forward: int, n_backward: int) -> tuple[int, int]:
    """Return (n_layers, n_heads) from score matrix shape.

    n_forward  = 1 + n_layers * (n_heads + 1)
    n_backward = n_layers * (3 * n_heads + 1) + 1
    """
    nf = n_forward - 1
    nb = n_backward - 1
    n_heads = (nb - nf) // (3 * nf - nb)
    n_layers = nf // (n_heads + 1)
    return n_layers, n_heads


def load_circuit_data(
    method: str,
    task: str,
    model: str,
    load_variance: bool = False,
) -> np.ndarray | None:
    file_name = "variance.pt" if load_variance else "scores.pt"
    path = os.path.join(CIRCUITS_DIR, method, f"{task}_{model}", file_name)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True).float().numpy()


def load_faithfulness_data(
    methods: list[str],
    tasks: list[str],
    models: list[str],
) -> dict[tuple[str, str], dict[str, dict]]:
    """Return {(task, model): {method: {weighted_edge_counts, faithfulnesses}}}."""
    data: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)

    for method in methods:
        for task in tasks:
            for model in models:
                path = os.path.join(
                    RESULTS_DIR,
                    f"{method}_patching_edge",
                    f"{task}_{model}_test_abs-False.pkl",
                )
                if not os.path.exists(path):
                    print(f"No result for: {method}, {task}, {model}. Skipping...")
                    continue
                with open(path, "rb") as fh:
                    result = pickle.load(fh)
                data[(task, model)][method] = {
                    "weighted_edge_counts": result["weighted_edge_counts"],
                    "faithfulnesses": result["faithfulnesses"],
                }
    return data


def load_metric(
    method: str,
    task: str,
    model: str,
    data_type: str,
    min_abs_score: float = 1e-6,
) -> np.ndarray | None:
    """Load scores, variance, or CV matrix for one (method, task, model).

    data_type: "scores" | "variance" | "cv"
    CV entries where |score| < min_abs_score are set to NaN.
    """
    scores = load_circuit_data(method, task, model, load_variance=False)
    if data_type == "scores":
        return scores
    variance = load_circuit_data(method, task, model, load_variance=True)
    if data_type == "variance":
        return variance
    if scores is None or variance is None:
        return None
    safe_denom = np.where(np.abs(scores) >= min_abs_score, np.abs(scores), 1.0)
    return np.where(np.abs(scores) >= min_abs_score, variance / safe_denom, np.nan)


def apply_top_percent_mask(
    metric: np.ndarray,
    method: str,
    task: str,
    model: str,
    percent: float,
    use_abs: bool,
) -> np.ndarray:
    """Return metric with entries outside the top `percent` of scores set to NaN.

    Ranking is by absolute score when use_abs=True, otherwise by raw score.
    """
    scores = load_circuit_data(method, task, model, load_variance=False)
    if scores is None:
        return metric
    ranking = np.abs(scores) if use_abs else scores
    threshold = np.nanpercentile(ranking, (1.0 - percent) * 100)
    result = metric.copy()
    result[ranking < threshold] = np.nan
    return result


def save_figure(fig: plt.Figure, out_path: str, dpi: int = 150) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
