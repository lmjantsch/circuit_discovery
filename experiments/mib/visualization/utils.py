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
    data_type: str = "scores",
) -> np.ndarray | None:
    path = os.path.join(CIRCUITS_DIR, method, f"{task}_{model}", f"{data_type}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True).float().numpy()


def load_faithfulness_data(
    methods: list[str],
    tasks: list[str],
    models: list[str],
    use_abs: bool = False,
) -> dict[tuple[str, str], dict[str, dict]]:
    """Return {(task, model): {method: {weighted_edge_counts, faithfulnesses}}}."""
    data: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    abs_flag = "True" if use_abs else "False"

    for method in methods:
        for task in tasks:
            for model in models:
                path = os.path.join(
                    RESULTS_DIR,
                    method,
                    f"{task}_{model}_test_abs-{abs_flag}.pkl",
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

    data_type: "scores" | "variance_in_between" | "variance_within" | "cv_in_between" | "cv_within"
    CV entries where |score| < min_abs_score are set to NaN.
    """
    scores = load_circuit_data(method, task, model, "scores")
    if data_type == "scores":
        return scores
    if data_type in ("variance_in_between", "variance_within"):
        return load_circuit_data(method, task, model, data_type)
    variance_key = "variance_in_between" if data_type == "cv_in_between" else "variance_within"
    variance = load_circuit_data(method, task, model, variance_key)
    if scores is None or variance is None:
        return None
    safe_denom = np.where(np.abs(scores) >= min_abs_score, np.abs(scores), 1.0)
    return np.where(np.abs(scores) >= min_abs_score, np.sqrt(variance) / safe_denom, np.nan)


def apply_layer_mask(
    metric: np.ndarray,
    ignore_first: int | None,
    ignore_last: int | None,
) -> np.ndarray:
    """Return metric with source rows / target cols for the first/last N layers set to NaN.

    Source layout:  row 0 = embedding; layer L occupies rows 1+L*(n_heads+1) .. 1+(L+1)*(n_heads+1)-1
    Target layout:  layer L occupies cols L*(3*n_heads+1) .. (L+1)*(3*n_heads+1)-1; last col = lm_head

    ignore_first=N: always drops the embedding row; also drops the first N source/target layer slices.
    ignore_last=N:  always drops the lm_head col;   also drops the last  N source/target layer slices.
    Pass None to skip either side entirely.
    """
    n_forward, n_backward = metric.shape
    n_layers, n_heads = infer_model_dims(n_forward, n_backward)
    result = metric.copy()

    src_per_layer = n_heads + 1
    tgt_per_layer = 3 * n_heads + 1

    if ignore_first is not None:
        result[0, :] = np.nan                                    # embedding (always)
        n = min(ignore_first, n_layers)
        if n > 0:
            result[1 : 1 + n * src_per_layer, :] = np.nan       # source rows for first n layers
            result[:, 0 : n * tgt_per_layer] = np.nan            # target cols for first n layers

    if ignore_last is not None:
        result[:, n_layers * tgt_per_layer] = np.nan             # lm_head (always)
        n = min(ignore_last, n_layers)
        if n > 0:
            start_src = 1 + (n_layers - n) * src_per_layer
            start_tgt = (n_layers - n) * tgt_per_layer
            result[start_src :, :] = np.nan                      # source rows for last n layers
            result[:, start_tgt : n_layers * tgt_per_layer] = np.nan  # target cols for last n layers

    return result


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
    scores = load_circuit_data(method, task, model, "scores")
    if scores is None:
        return metric
    ranking = np.abs(scores) if use_abs else scores
    threshold = np.nanpercentile(ranking, (1.0 - percent) * 100)
    result = metric.copy()
    result[ranking < threshold] = np.nan
    return result


def load_sdf_data(
    methods: list[str],
    tasks: list[str],
    models: list[str],
    split: str = "validation",
    use_abs: bool = False,
) -> dict[tuple[str, str], dict[str, dict]]:
    """Return {(task, model): {method: sdf_dict}} loaded from *_sdf.pkl files."""
    data: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    abs_flag = "True" if use_abs else "False"
    for method in methods:
        for task in tasks:
            for model in models:
                path = os.path.join(
                    RESULTS_DIR,
                    method,
                    f"{task}_{model}_{split}_abs-{abs_flag}_sdf.pkl",
                )
                if not os.path.exists(path):
                    print(f"No SDF data for: {method}, {task}, {model}. Skipping...")
                    continue
                with open(path, "rb") as fh:
                    data[(task, model)][method] = pickle.load(fh)
    return data


def save_figure(fig: plt.Figure, out_path: str, dpi: int = 150) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
