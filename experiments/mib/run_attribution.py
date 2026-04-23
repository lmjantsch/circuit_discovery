"""
DPEA (Dual Path Edge Attribution) for MIB benchmark.

Uses FastEdgeTracer (vectorized, streaming backward) for attribution.
Only processes tasks that don't already have an importances.json file.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m experiments.mib.run_fast_attribution
"""

import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

from nnsight import LanguageModel
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset

from tracer.backend import BACKEND_MAPPING
from tracer.tracer import FastEdgeTracer
from experiments.mib.create_mib_circuite import build_graph_json

CIRCUIT_DIR = "experiments/mib/MIB-circuit-track/circuits"
SPLIT = "train"
NUM_EXAMPLES = 100

MIB_TO_DPA_MODEL = {
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "llama3": "meta-llama/Llama-3.1-8B",
}

TASKS_TO_HF_NAMES = {
    'ioi': 'ioi',
    'mcqa': 'copycolors_mcqa',
    'arithmetic_addition': 'arithmetic_addition',
    'arithmetic_subtraction': 'arithmetic_subtraction',
    'arc_easy': 'arc_easy',
    'arc_challenge': 'arc_challenge',
}


def collate_fn(xs):
    clean, corrupted, labels = zip(*xs)
    return list(clean), list(corrupted), labels


class MIBDataset(Dataset):
    """Minimal MIB dataset loader (no transformer_lens dependency)."""

    def __init__(self, task, tokenizer, model_name, split='train', num_examples=None):
        self.task = task
        self.tokenizer = tokenizer
        self.model_name = model_name

        hf_url = f"mib-bench/{TASKS_TO_HF_NAMES[task]}"
        if task == 'mcqa':
            self.dataset = load_dataset(hf_url, '4_answer_choices', split=split)
            self.counterfactual_type = "symbol_counterfactual"
        elif task.startswith('arc'):
            self.dataset = load_dataset(hf_url, split=split)
            self.counterfactual_type = "symbol_counterfactual"
        elif task.startswith('arithmetic'):
            self.dataset = load_dataset(hf_url, split=split)
            self.operator = "-" if "subtraction" in task else "+"
        else:
            self.dataset = load_dataset(hf_url, split=split)

        self.dataset = self._filter()
        if num_examples and num_examples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_examples))

    def _filter(self):
        tok = self.tokenizer
        if self.task == 'ioi':
            return self.dataset.filter(
                lambda x: (
                    len(tok(f" {x['metadata']['indirect_object']}", add_special_tokens=False).input_ids) ==
                    len(tok(f" {x['metadata']['subject']}", add_special_tokens=False).input_ids) and
                    len(tok(f" {x['metadata']['indirect_object']}", add_special_tokens=False).input_ids) ==
                    len(tok(f" {x['metadata']['random_c']}", add_special_tokens=False).input_ids)
                )
            )
        elif self.task == 'mcqa' or self.task.startswith('arc'):
            ct = self.counterfactual_type
            return self.dataset.filter(
                lambda x: (
                    len(tok(x["choices"]["label"][x["answerKey"]], add_special_tokens=False).input_ids) ==
                    len(tok(str(x[ct]["choices"]["label"][x[ct]["answerKey"]]), add_special_tokens=False).input_ids)
                )
            )
        elif self.task.startswith('arithmetic'):
            op = self.operator
            return self.dataset.filter(
                lambda x: (
                    len(tok(str(x["label"]), add_special_tokens=False).input_ids) == 1 and
                    x["random_counterfactual"] is not None and
                    x["random_counterfactual"]["prompt"] is not None and
                    x["operator"] == op and
                    len(tok(str(x["random_counterfactual"]["label"]), add_special_tokens=False).input_ids) == 1
                )
            )
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        tok = self.tokenizer

        if self.task == 'ioi':
            correct_idx = tok(f" {row['metadata']['indirect_object']}", add_special_tokens=False).input_ids[0]
            incorrect_idx = tok(f" {row['metadata']['subject']}", add_special_tokens=False).input_ids[0]
            cf = row.get("s2_io_flip_counterfactual", row.get("counterfactual", {}))
            return row["prompt"], cf.get("prompt", row["prompt"]), [correct_idx, incorrect_idx]

        elif self.task == 'mcqa' or self.task.startswith('arc'):
            ct = self.counterfactual_type
            correct_idx = tok(row["choices"]["label"][row["answerKey"]], add_special_tokens=False).input_ids[0]
            cf = row[ct]
            incorrect_idx = tok(str(cf["choices"]["label"][cf["answerKey"]]), add_special_tokens=False).input_ids[0]
            return row["prompt"], cf["prompt"], [correct_idx, incorrect_idx]

        elif self.task.startswith('arithmetic'):
            correct_idx = tok(str(row["label"]), add_special_tokens=False).input_ids[0]
            cf = row["random_counterfactual"]
            incorrect_idx = tok(str(cf["label"]), add_special_tokens=False).input_ids[0]
            return row["prompt"], cf["prompt"], [correct_idx, incorrect_idx]

# All valid combos. Will skip ones already done.
ALL_COMBOS = [
    ("ioi", "qwen2.5"),
    ("mcqa", "qwen2.5"),
    ("ioi", "llama3"),
    ("mcqa", "llama3"),
    ("arithmetic_addition", "llama3"),
    ("arithmetic_subtraction", "llama3"),
    ("arc_easy", "llama3"),
    ("arc_challenge", "llama3"),
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run():
    log("=" * 60)
    log("FAST DPA Edge Attribution (FastEdgeTracer)")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log("=" * 60)

    current_model_name = None
    model = None
    tracer = None
    tokenizer = None
    model_config = None

    total_start = time.time()

    for i, (task, model_name) in enumerate(ALL_COMBOS):
        task_col = f"{task.replace('_', '-')}_{model_name}"
        model_id = MIB_TO_DPA_MODEL[model_name]

        method_name = "dpa_patching_edge"
        circuit_path = os.path.join(CIRCUIT_DIR, method_name, task_col)
        output_path = os.path.join(circuit_path, "importances.json")
        if os.path.exists(output_path):
            log(f"[{i+1}/{len(ALL_COMBOS)}] {task_col} already done, skipping")
            continue

        if model_name != current_model_name:
            if model is not None:
                del model, tracer
                torch.cuda.empty_cache()

            log(f"\nLoading model: {model_name} ({model_id})")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.padding_side = 'left'
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

            if model_name == "qwen2.5":
                batch_size = 32
            else:
                batch_size = 4

            model = LanguageModel(
                model_id,
                attn_implementation='eager',
                device_map='auto',
                dispatch=True,
                dtype=torch.bfloat16,
            )

            backend_cls = BACKEND_MAPPING[model_id]
            backend = backend_cls(model)
            tracer = FastEdgeTracer(backend, tokenizer)

            cfg = AutoConfig.from_pretrained(model_id)
            model_config = {
                "n_layers": cfg.num_hidden_layers,
                "n_heads": cfg.num_attention_heads,
                "d_model": cfg.hidden_size,
                "parallel_attn_mlp": False,
            }
            current_model_name = model_name

        n_fwd = 1 + model_config['n_layers'] * (model_config['n_heads'] + 1)
        n_bwd = model_config['n_layers'] * (3 * model_config['n_heads'] + 1) + 1

        log(f"\n[{i+1}/{len(ALL_COMBOS)}] {task} / {model_name}")

        t0 = time.time()
        dataset = MIBDataset(
            task, tokenizer, model_name,
            split=SPLIT, num_examples=NUM_EXAMPLES,
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size,
            collate_fn=collate_fn, shuffle=False,
        )
        log(f"  Dataset: {len(dataset)} examples, {len(dataloader)} batches (bs={batch_size})")

        all_scores = torch.zeros(n_fwd, n_bwd)
        total_items = 0

        for batch_idx, (clean_strs, corrupt_strs, labels) in enumerate(dataloader):
            bs = len(clean_strs)
            total_items += bs

            clean_enc = tokenizer(list(clean_strs), return_tensors='pt', padding=True)
            corrupt_enc = tokenizer(list(corrupt_strs), return_tensors='pt', padding=True)

            if isinstance(labels[0], (list, tuple)):
                target_ids = torch.tensor([l[0] for l in labels])
            else:
                target_ids = torch.tensor(list(labels))

            clean_batch = {
                'input_ids': clean_enc['input_ids'],
                'attention_mask': clean_enc['attention_mask'],
                'targets': target_ids,
            }
            corrupt_batch = {
                'input_ids': corrupt_enc['input_ids'],
                'attention_mask': corrupt_enc['attention_mask'],
            }

            bt0 = time.time()
            batch_scores = tracer.trace(clean_batch, corrupt_batch, target_ids)
            bt_elapsed = time.time() - bt0
            all_scores += batch_scores * bs

            if batch_idx == 0 or (batch_idx + 1) % 5 == 0:
                log(f"  Batch {batch_idx+1}/{len(dataloader)} ({bt_elapsed:.1f}s), items: {total_items}")

        all_scores /= total_items

        os.makedirs(circuit_path, exist_ok=True)
        build_graph_json(all_scores, model_config, output_path)

        elapsed = time.time() - t0
        n_nonzero = (all_scores != 0).sum().item()
        log(f"  Done in {elapsed:.1f}s — non-zero: {n_nonzero}, saved: {output_path}")

    total_elapsed = time.time() - total_start
    log(f"\n{'='*60}")
    log(f"All attribution done! Total: {total_elapsed/60:.1f} min")
    log(f"{'='*60}")


if __name__ == "__main__":
    run()