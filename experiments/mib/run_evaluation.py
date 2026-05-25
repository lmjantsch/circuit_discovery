import os
import sys
import math
import pickle
import argparse

proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if proj_path not in sys.path:
    sys.path.insert(0, proj_path)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from modular_transformer import patch_model_for_lvp
from experiments.mib.data_utils import MIBDataset
from patcher.patcher import (
    EdgeCircuitPatcher,
    MIB_MODEL_TO_HF_ID,
    MIB_MODEL_TO_ADAPTER_CLS,
    MIB_MODEL_TO_ARC,
    PERCENTAGES,
)

DEFAULT_CIRCUIT_DIR = os.path.join(proj_path, 'experiments/mib/circuits')
DEFAULT_OUTPUT_DIR  = os.path.join(proj_path, 'experiments/mib/results')


def compute_metrics(faithfulnesses: list[float], percentages: tuple) -> dict:
    area_under = 0.
    area_from_1 = 0.
    log_area_under = 0.
    log_area_from_1 = 0.
    for i in range(len(faithfulnesses) - 1):
        x1, x2 = percentages[i], percentages[i + 1]
        y1, y2 = faithfulnesses[i], faithfulnesses[i + 1]
        w = x2 - x1
        area_under  += w * (y1 + y2) / 2
        area_from_1 += w * (abs(1. - y1) + abs(1. - y2)) / 2
        if x1 > 0 and x2 > 0:
            log_w = math.log(x2) - math.log(x1)
            log_area_under  += log_w * (y1 + y2) / 2
            log_area_from_1 += log_w * (abs(1. - y1) + abs(1. - y2)) / 2
    return {
        "area_under":      area_under,
        "area_from_1":     area_from_1,
        "log_area_under":  log_area_under,
        "log_area_from_1": log_area_from_1,
        "average":         sum(faithfulnesses) / len(faithfulnesses),
        "faithfulnesses":  faithfulnesses,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",       type=str, nargs='+', required=True)
    parser.add_argument("--tasks",        type=str, nargs='+', required=True)
    parser.add_argument("--methods",      type=str, nargs='+', required=True,
                        help="Subdirectory names under circuit-dir (e.g. eap_bilin_frnorm)")
    parser.add_argument("--split",        type=str, choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument("--absolute",      action="store_true")
    parser.add_argument("--norm-matching", action="store_true")
    parser.add_argument("--batch-size",   type=int, default=16)
    parser.add_argument("--head",         type=int, default=None,
                        help="Limit dataset to this many examples (default: all)")
    parser.add_argument("--circuit-dir",  type=str, default=DEFAULT_CIRCUIT_DIR)
    parser.add_argument("--output-dir",   type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    for model_name in args.models:
        model_id = MIB_MODEL_TO_HF_ID[model_name]

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        if model_name == 'gpt2':
            tokenizer.padding_side = "right"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float32 if model_name == 'gpt2' else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="eager", device_map="auto",
        ).eval()
        model = patch_model_for_lvp(model, norm_approx='frozen')

        adapter = MIB_MODEL_TO_ADAPTER_CLS[model_name](model, MIB_MODEL_TO_ARC[model_name], frozen_norm=False)
        patcher = EdgeCircuitPatcher(adapter, tokenizer, norm_matching=args.norm_matching)

        for method in args.methods:
            for task in args.tasks:
                circuit_path = os.path.join(args.circuit_dir, method, f"{task}_{model_name}", 'scores.pt')
                if not os.path.exists(circuit_path):
                    print(f"Circuit not found, skipping: {circuit_path}")
                    continue

                print(f"[{method}/{model_name}/{task}] Loading circuit from {circuit_path}")
                circuit_scores = torch.load(circuit_path, map_location='cpu')
                if args.absolute:
                    circuit_scores.abs_()

                dataset = MIBDataset(task, tokenizer, model_name, split=args.split, num_examples=args.head)
                if args.head is not None and len(dataset) < args.head:
                    print(f"  Warning: dataset has only {len(dataset)} examples, head={args.head} ignored.")
                dataloader = dataset.dataloader(args.batch_size)

                faithfulnesses, percentages, weighted_edge_counts = patcher.patch_circuit(dataloader, circuit_scores)
                d = compute_metrics(faithfulnesses, percentages)
                d["weighted_edge_counts"] = weighted_edge_counts

                nm_prefix = "nm_" if args.norm_matching else ""
                output_dir = os.path.join(args.output_dir, f"{nm_prefix}{method}")
                os.makedirs(output_dir, exist_ok=True)
                out_file = os.path.join(output_dir, f"{task}_{model_name}_{args.split}_abs-{args.absolute}.pkl")
                with open(out_file, 'wb') as f:
                    pickle.dump(d, f)

                print(
                    f"  -> {out_file}\n"
                    f"     area_under={d['area_under']:.4f}  "
                    f"area_from_1={d['area_from_1']:.4f}  "
                    f"average={d['average']:.4f}\n"
                    f"     log_area_under={d['log_area_under']:.4f}  "
                    f"log_area_from_1={d['log_area_from_1']:.4f}"
                )
