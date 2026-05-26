#!/usr/bin/env python3
"""Run new-patcher attribution/evaluation and split-stability metrics.

This reproduces the manual workflow:
1. export the new-patcher ref into a temporary source tree,
2. apply the temporary fixes used for the ioi_gemma2 CPR=1.357 run,
3. run that tree's experiments.mib.run_attribution for each requested method/combo/train slice,
4. run that tree's experiments.mib.run_evaluation on the full test split,
5. aggregate score mean/std and pruned pairwise Jaccard by patcher percentage.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


SUPPORTED_COMBOS = (
    ("ioi", "gpt2"),
    ("ioi", "qwen2.5"),
    ("mcqa", "qwen2.5"),
    # ("ioi", "llama3"),
    # ("mcqa", "llama3"),
    # ("arithmetic_addition", "llama3"),
    # ("arithmetic_subtraction", "llama3"),
    # ("arc_easy", "llama3"),
    # ("arc_challenge", "llama3"),
    ("ioi", "gemma2"),
    ("mcqa", "gemma2"),
    # ("arc_easy", "gemma2"),
)

DEFAULT_METHODS = ("eap_pure", "eap_frnorm_secantmlp_bilinear")
DEFAULT_TRAIN_RANGES = ((0, 100), (100, 200), (200, 300), (300, 400), (400, 500))
DEFAULT_MCQA_TRAIN_RANGES = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100))
PERCENTAGES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
KNOWN_FILTERED_TRAIN_LIMITS = {
    "mcqa": 110,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run new-patcher attribution, test CPR, and split-stability metrics."
    )
    parser.add_argument(
        "--new-patcher-ref",
        default="origin/patcher",
        help="Git ref for the new-patcher code. origin/patcher matches the ioi_gemma2 CPR=1.357 run.",
    )
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--report-csv", default="experiments/mib/method_comparison/report.csv")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--circuits-dir", default="experiments/mib/method_comparison/circuits")
    parser.add_argument("--results-dir", default="experiments/mib/method_comparison/results")
    parser.add_argument("--stability-dir", default="experiments/mib/method_comparison/stability")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument(
        "--combo",
        action="append",
        default=None,
        help="Specific task:model combo. Can be repeated, e.g. --combo ioi:qwen2.5.",
    )
    parser.add_argument("--no-counterfactual", action="store_true")
    parser.add_argument(
        "--train-range",
        action="append",
        default=None,
        help="Zero-based half-open train slice START:END. Defaults to 0:100,...,400:500.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-attribution", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True)
    return Path(out.strip())


def parse_train_ranges(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.train_range is None:
        return list(DEFAULT_TRAIN_RANGES)
    ranges = []
    for item in args.train_range:
        if ":" not in item:
            raise SystemExit(f"--train-range must be START:END, got {item!r}")
        start_s, end_s = item.split(":", 1)
        start, end = int(start_s), int(end_s)
        if start < 0 or end <= start:
            raise SystemExit(f"Invalid --train-range {item!r}")
        ranges.append((start, end))
    return ranges


def train_ranges_for_task(
    task: str,
    split: str,
    ranges: list[tuple[int, int]],
    using_default_ranges: bool,
) -> list[tuple[int, int]]:
    if using_default_ranges and split == "train" and task == "mcqa":
        return list(DEFAULT_MCQA_TRAIN_RANGES)
    return ranges


def range_tag(start: int, end: int) -> str:
    return f"train{start + 1:03d}-{end:03d}"


def split_method_name(method: str, start: int, end: int) -> str:
    return f"{method}_{range_tag(start, end)}"


def known_filtered_train_limit(task: str, split: str) -> int | None:
    if split != "train":
        return None
    return KNOWN_FILTERED_TRAIN_LIMITS.get(task)


def method_flags(method: str, model: str) -> list[str]:
    if method == "eap_pure":
        return []
    if method == "eap_frnorm_secantmlp_bilinear":
        mlp_act = "secant_silu" if model in {"qwen2.5", "llama3"} else "secant_gelu_tanh"
        return [
            "--norm-approx",
            "frozen",
            "--mlp-act-fn",
            mlp_act,
            "--matmul-fn",
            "bilinear_matmul",
            "--mul-fn",
            "bilinear_mul",
        ]
    raise SystemExit(f"Unsupported method preset: {method}")


def selected_combos(args: argparse.Namespace, root: Path) -> list[tuple[str, str]]:
    if args.combo:
        combos = []
        for item in args.combo:
            if ":" not in item:
                raise SystemExit(f"--combo must be task:model, got {item!r}")
            task, model = item.split(":", 1)
            combos.append((task, model))
    else:
        combos = combos_from_report(root / args.report_csv, args.methods)

    if args.tasks is not None:
        allowed = set(args.tasks)
        combos = [(task, model) for task, model in combos if task in allowed]
    if args.models is not None:
        allowed = set(args.models)
        combos = [(task, model) for task, model in combos if model in allowed]

    unsupported = [combo for combo in combos if combo not in SUPPORTED_COMBOS]
    if unsupported:
        raise SystemExit(f"Unsupported combos requested: {unsupported}")
    if not combos:
        raise SystemExit("No combos selected.")
    return combos


def combos_from_report(path: Path, methods: list[str]) -> list[tuple[str, str]]:
    if not path.exists():
        return list(SUPPORTED_COMBOS)
    wanted = set(methods)
    combos: list[tuple[str, str]] = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("method") not in wanted:
                continue
            combo = (row["task"], row["model"])
            if combo in SUPPORTED_COMBOS and combo not in seen:
                seen.add(combo)
                combos.append(combo)
    return combos


def export_ref(ref: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(["git", "archive", ref], stdout=subprocess.PIPE)
    try:
        subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout, check=True)
    finally:
        if archive.stdout is not None:
            archive.stdout.close()
        rc = archive.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, ["git", "archive", ref])


def patch_new_patcher_tree(tree: Path) -> None:
    """Apply the temporary fixes used in the successful manual Gemma run."""
    adapter_path = tree / "adapters" / "adapter.py"
    adapter_text = adapter_path.read_text(encoding="utf-8")
    old = """\
    def residual_out_hook(self, detached=True) -> torch.Tensor:
        if detached:
            return self.layer.output.detach()
        return self.layer.output
"""
    new = """\
    def residual_out_hook(self, detached=True) -> torch.Tensor:
        output = self.layer.output[0] if isinstance(self.layer.output, tuple) else self.layer.output
        if detached:
            return output.detach()
        return output
"""
    if old in adapter_text:
        adapter_path.write_text(adapter_text.replace(old, new), encoding="utf-8")

    patcher_path = tree / "patcher" / "patcher.py"
    patcher_text = patcher_path.read_text(encoding="utf-8")
    patcher_text = patcher_text.replace(
        "proj_path = '/home/dacslab/lasse_jantsch/circuit_discovery'",
        f"proj_path = '{tree}'",
    )
    patcher_path.write_text(patcher_text, encoding="utf-8")

    data_utils_path = tree / "experiments" / "mib" / "data_utils.py"
    data_utils_text = data_utils_path.read_text(encoding="utf-8")
    old_select = """\
        if num_examples and num_examples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_examples))
"""
    new_select = """\
        start_env = os.environ.get("MIB_EXAMPLE_START")
        end_env = os.environ.get("MIB_EXAMPLE_END")
        if start_env is not None or end_env is not None:
            start = int(start_env or 0)
            end = int(end_env) if end_env is not None else len(self.dataset)
            end = min(end, len(self.dataset))
            if start < 0 or end < start:
                raise ValueError(f"Invalid dataset slice: {start}:{end}")
            self.dataset = self.dataset.select(range(start, end))
        elif num_examples and num_examples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_examples))
"""
    if "import os\n" not in data_utils_text:
        data_utils_text = data_utils_text.replace("from torch.utils.data", "import os\n\nfrom torch.utils.data", 1)
    if old_select in data_utils_text:
        data_utils_text = data_utils_text.replace(old_select, new_select)
    data_utils_path.write_text(data_utils_text, encoding="utf-8")


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def infer_dims(scores: torch.Tensor) -> tuple[int, int]:
    src, tgt = scores.shape
    for n_layers in range(1, 128):
        rem = src - 1
        if rem <= 0 or rem % n_layers:
            continue
        n_heads = rem // n_layers - 1
        if n_heads > 0 and tgt == n_layers * (3 * n_heads + 1) + 1:
            return n_layers, n_heads
    raise ValueError(f"Could not infer model dims from score shape {tuple(scores.shape)}")


def src_attn(layer: int, n_heads: int) -> slice:
    start = 1 + layer * (n_heads + 1)
    return slice(start, start + n_heads)


def src_mlp(layer: int, n_heads: int) -> slice:
    idx = 1 + layer * (n_heads + 1) + n_heads
    return slice(idx, idx + 1)


def tgt_q(layer: int, n_heads: int) -> slice:
    start = layer * (3 * n_heads + 1)
    return slice(start, start + n_heads)


def tgt_k(layer: int, n_heads: int) -> slice:
    start = layer * (3 * n_heads + 1) + n_heads
    return slice(start, start + n_heads)


def tgt_v(layer: int, n_heads: int) -> slice:
    start = layer * (3 * n_heads + 1) + 2 * n_heads
    return slice(start, start + n_heads)


def tgt_mlp(layer: int, n_heads: int) -> slice:
    idx = layer * (3 * n_heads + 1) + 3 * n_heads
    return slice(idx, idx + 1)


def valid_edge_mask(shape: torch.Size, n_layers: int, n_heads: int) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[0] = True
    for layer in range(n_layers):
        mask[src_attn(layer, n_heads), tgt_v(layer, n_heads).stop :] = True
        mask[src_mlp(layer, n_heads), tgt_mlp(layer, n_heads).stop :] = True
    return mask


def forward_to_backward(shape: torch.Size, n_layers: int, n_heads: int) -> torch.Tensor:
    ftb = torch.zeros(shape, dtype=torch.bool)
    for layer in range(n_layers):
        attn_src = src_attn(layer, n_heads)
        for tgt in (tgt_q(layer, n_heads), tgt_k(layer, n_heads), tgt_v(layer, n_heads)):
            ftb[attn_src, tgt] = True
        ftb[src_mlp(layer, n_heads), tgt_mlp(layer, n_heads)] = True
    return ftb


def prune(in_graph: torch.Tensor, ftb: torch.Tensor) -> torch.Tensor:
    nodes_in_graph = in_graph.any(dim=1)
    changed = True
    while changed:
        nodes_with_outgoing = in_graph.any(dim=1)
        nodes_with_ingoing = (in_graph.any(dim=0).float() @ ftb.float().T) > 0
        nodes_with_ingoing[0] = True
        new_nodes = nodes_with_outgoing & nodes_with_ingoing
        changed = not torch.equal(new_nodes, nodes_in_graph)
        nodes_in_graph = new_nodes

        backward_alive = nodes_in_graph.float() @ ftb.float()
        backward_alive[-1] = 1.0
        edge_mask = nodes_in_graph[:, None] & (backward_alive > 0)[None, :]
        in_graph = in_graph & edge_mask
    return in_graph


def circuit_mask(scores: torch.Tensor, percentage: float) -> torch.Tensor:
    n_layers, n_heads = infer_dims(scores)
    valid = valid_edge_mask(scores.shape, n_layers, n_heads)
    ftb = forward_to_backward(scores.shape, n_layers, n_heads)
    ranked = scores.clone()
    ranked[~valid] = torch.finfo(ranked.dtype).min
    sorted_idx = torch.argsort(ranked.reshape(-1), descending=True)
    k = int(valid.sum().item() * percentage)
    mask = torch.zeros_like(valid)
    if k > 0:
        mask.reshape(-1)[sorted_idx[:k]] = True
    return prune(mask, ftb) & valid


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    union = (a | b).sum().item()
    if union == 0:
        return 1.0
    return float((a & b).sum().item() / union)


def aggregate_stability(
    root: Path,
    method: str,
    task: str,
    model: str,
    ranges: list[tuple[int, int]],
    circuits_dir: Path,
    results_dir: Path,
    stability_dir: Path,
) -> None:
    scores = []
    faithfulnesses = []
    for start, end in ranges:
        smethod = split_method_name(method, start, end)
        score_path = circuits_dir / smethod / f"{task}_{model}" / "scores.pt"
        result_path = results_dir / smethod / f"{task}_{model}_test_abs-False.pkl"
        if not score_path.exists() or not result_path.exists():
            print(f"[stability skip] missing {score_path} or {result_path}", flush=True)
            return
        scores.append(torch.load(score_path, map_location="cpu").float())
        with result_path.open("rb") as f:
            faithfulnesses.append(torch.tensor(pickle.load(f)["faithfulnesses"], dtype=torch.float32))

    stack = torch.stack(scores)
    score_mean = stack.mean(dim=0)
    score_std = stack.std(dim=0, unbiased=len(scores) > 1)
    faith_stack = torch.stack(faithfulnesses)
    faith_mean = faith_stack.mean(dim=0)
    faith_std = faith_stack.std(dim=0, unbiased=len(faithfulnesses) > 1)

    out_dir = stability_dir / method / f"{task}_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(score_mean, out_dir / "score_mean.pt")
    torch.save(score_std, out_dir / "score_std.pt")

    rows = []
    for p_idx, pct in enumerate(PERCENTAGES):
        masks = [circuit_mask(s, pct) for s in scores]
        pairwise = [jaccard(a, b) for a, b in itertools.combinations(masks, 2)]
        pairwise_std = 0.0 if len(pairwise) < 2 else float(torch.tensor(pairwise).std(unbiased=True).item())
        rows.append(
            {
                "method": method,
                "task": task,
                "model": model,
                "percentage": pct,
                "faithfulness_mean": float(faith_mean[p_idx].item()),
                "faithfulness_std": float(faith_std[p_idx].item()),
                "pairwise_jaccard_mean": float(sum(pairwise) / len(pairwise)),
                "pairwise_jaccard_std": pairwise_std,
                "n_pairwise": len(pairwise),
            }
        )

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[stability] wrote {out_dir / 'summary.csv'}", flush=True)


def main() -> None:
    args = parse_args()
    root = repo_root()
    combos = selected_combos(args, root)
    train_ranges = parse_train_ranges(args)
    using_default_ranges = args.train_range is None

    circuits_dir = (root / args.circuits_dir).resolve()
    results_dir = (root / args.results_dir).resolve()
    stability_dir = (root / args.stability_dir).resolve()

    temp_root = Path(tempfile.mkdtemp(prefix="cd_new_patcher_eap_pure_"))
    new_patcher_tree = temp_root / "src"

    try:
        if not args.dry_run:
            export_ref(args.new_patcher_ref, new_patcher_tree)
            patch_new_patcher_tree(new_patcher_tree)
        else:
            print(f"Would export {args.new_patcher_ref} to {new_patcher_tree}")
            print("Would apply new-patcher temporary Gemma/proj_path fixes")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

        for method in args.methods:
            for task, model in combos:
                if (task, model) not in combos_from_report(root / args.report_csv, [method]) and not args.combo:
                    continue
                combo_train_ranges = train_ranges_for_task(
                    task,
                    args.train_split,
                    train_ranges,
                    using_default_ranges,
                )
                print(f"\n== {method} / {task}_{model} ==", flush=True)
                for start, end in combo_train_ranges:
                    train_limit = known_filtered_train_limit(task, args.train_split)
                    if train_limit is not None and start >= train_limit:
                        print(
                            f"[slice skip] {task}_{model} {start}:{end} is empty after filtering "
                            f"(available train examples: {train_limit})",
                            flush=True,
                        )
                        continue
                    smethod = split_method_name(method, start, end)
                    slice_env = env.copy()
                    slice_env["MIB_EXAMPLE_START"] = str(start)
                    slice_env["MIB_EXAMPLE_END"] = str(end)
                    print(f"-- train slice {start}:{end} ({range_tag(start, end)})", flush=True)

                    if not args.skip_attribution:
                        attr_cmd = [
                            sys.executable,
                            "-m",
                            "experiments.mib.run_attribution",
                            "--models",
                            model,
                            "--tasks",
                            task,
                            "--split",
                            args.train_split,
                            "--num-examples",
                            str(end - start),
                            "--output-dir",
                            str(circuits_dir),
                            "--method-name",
                            smethod,
                            *method_flags(method, model),
                        ]
                        if args.batch_size is not None:
                            attr_cmd.extend(["--batch-size", str(args.batch_size)])
                        if not args.no_counterfactual:
                            attr_cmd.append("--use-counterfactual")
                        if args.force:
                            attr_cmd.append("--force")
                        run_cmd(attr_cmd, new_patcher_tree, slice_env, args.dry_run)

                    if not args.skip_evaluation:
                        result_path = results_dir / smethod / f"{task}_{model}_{args.eval_split}_abs-False.pkl"
                        if result_path.exists() and not args.force:
                            print(f"[evaluation skip] {result_path}", flush=True)
                        else:
                            eval_cmd = [
                                sys.executable,
                                "-m",
                                "experiments.mib.run_evaluation",
                                "--models",
                                model,
                                "--tasks",
                                task,
                                "--methods",
                                smethod,
                                "--split",
                                args.eval_split,
                                "--circuit-dir",
                                str(circuits_dir),
                                "--output-dir",
                                str(results_dir),
                            ]
                            run_cmd(eval_cmd, new_patcher_tree, env, args.dry_run)

                if not args.skip_stability and not args.dry_run:
                    completed_ranges = []
                    for start, end in combo_train_ranges:
                        smethod = split_method_name(method, start, end)
                        score_path = circuits_dir / smethod / f"{task}_{model}" / "scores.pt"
                        result_path = results_dir / smethod / f"{task}_{model}_{args.eval_split}_abs-False.pkl"
                        if score_path.exists() and result_path.exists():
                            completed_ranges.append((start, end))
                    if len(completed_ranges) < 2:
                        print(
                            f"[stability skip] need at least 2 completed slices for {method} / {task}_{model}; "
                            f"found {len(completed_ranges)}",
                            flush=True,
                        )
                        continue
                    aggregate_stability(
                        root,
                        method,
                        task,
                        model,
                        completed_ranges,
                        circuits_dir,
                        results_dir,
                        stability_dir,
                    )

    finally:
        if args.keep_temp:
            print(f"Kept temp tree: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
