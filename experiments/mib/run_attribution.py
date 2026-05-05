"""
Circuit attribution via EdgeCircuitTracer (LVP / DPA method).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m experiments.mib.run_attribution \\
        --models qwen2.5 gemma2 \\
        --tasks ioi mcqa \\
        --split train \\
        --num-examples 100 \\
        --output-dir experiments/mib/MIB-circuit-track/circuits
"""

import argparse
import json
import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import NNsight

from experiments.mib.data_utils import MIBDataset, create_mib_circuit
from linear_transformer import patch_model_for_lvp
from tracer.model_adapters import Llama2ModelAdapter, Gemma2ModelAdapter, GPT2ModelAdapter, ModelAdapter
from tracer.tracer import EdgeCircuitTracer

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MIB_MODEL_TO_HF_ID: dict[str, str] = {
    "gpt2": "gpt2",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "llama3": "meta-llama/Llama-3.1-8B",
    "gemma2": "google/gemma-2-2b",
}

MIB_MODEL_TO_ADAPTER_CLS: dict[str, type | None] = {
    "gpt2": GPT2ModelAdapter,
    "qwen2.5": Llama2ModelAdapter,
    "llama3": Llama2ModelAdapter,
    "gemma2": Gemma2ModelAdapter,
}

DEFAULT_BATCH_SIZES: dict[str, int] = {
    "gpt2": 64,
    "qwen2.5": 32,
    "llama3": 4,
    "gemma2": 16,
}

ALL_COMBOS = [
    ("ioi", "gpt2"),
    ("ioi", "qwen2.5"),
    ("mcqa", "qwen2.5"),
    ("ioi", "llama3"),
    ("mcqa", "llama3"),
    ("arithmetic_addition", "llama3"),
    ("arithmetic_subtraction", "llama3"),
    ("arc_easy", "llama3"),
    ("arc_challenge", "llama3"),
    ("ioi", "gemma2"),
    ("mcqa", "gemma2"),
    ("arc_easy", "gemma2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPA circuit attribution via EdgeCircuitTracer.")
    parser.add_argument(
        "--models", nargs="+",
        default=list(MIB_MODEL_TO_HF_ID.keys()),
        choices=list(MIB_MODEL_TO_HF_ID.keys()),
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=["ioi", "mcqa", "arithmetic_addition", "arithmetic_subtraction", "arc_easy", "arc_challenge"],
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None, help="Overrides per-model defaults.")
    parser.add_argument("--output-dir", default="experiments/mib/MIB-circuit-track/circuits")
    parser.add_argument("--method-name", default="dpa_patching_edge",
                        help="Subdirectory name under output-dir identifying this run's method/config.")
    parser.add_argument("--use-counterfactual", action="store_true")
    parser.add_argument("--integration-steps", type=int, default=1,
                        help="Number of integration steps for IG-style attribution (1 = plain gradient).")

    # patch_model_for_lvp kwargs
    parser.add_argument(
        "--attn-act-fn", default="softmax",
        help="Attention softmax rule (Rule 3). Key into ACT_FN, e.g. dtd_softmax, sec_jac_softmax.",
    )
    parser.add_argument(
        "--matmul-fn", default="matmul",
        help="QK and AV matmul rule (Rule 4). Key into BILINEAR_FN.",
    )
    parser.add_argument(
        "--mul-fn", default="mul",
        help="Gate x up product rule (Rule 4). Key into BILINEAR_FN.",
    )
    parser.add_argument(
        "--mlp-act-fn", default=None,
        help="MLP activation rule (Rule 2). Key into ACT_FN. Defaults to model-specific LVP fn.",
    )
    parser.add_argument(
        "--frozen-norm", dest="frozen_norm", action="store_true", default=False,
        help="Disable detaching the normalisation factor in RMSNorm/LayerNorm (Rule 1).",
    )
    parser.add_argument(
        "--ignore-norm", dest="ignore_norm", action="store_true", default=False,
        help="Ignores norms on backward pass",
    )
    parser.add_argument(
        "--center-writing-weights", dest="center_writing_weights", action="store_true", default=False,
    )
    parser.add_argument(
        "--attn-softcap-fn", default="tanh",
        help="Gemma2 logit softcap rule (Rule 2). Key into ACT_FN.",
    )
    parser.add_argument(
        "--force", dest="force", action="store_true", default=False,
    )
    return parser.parse_args()


def _load_model_components(
    model_name: str, lvp_kwargs: dict, args
) -> tuple[AutoTokenizer, ModelAdapter, EdgeCircuitTracer]:
    adapter_cls: type[ModelAdapter] = MIB_MODEL_TO_ADAPTER_CLS[model_name]
    if adapter_cls is None:
        raise NotImplementedError(f"No ModelAdapter implemented for '{model_name}'")

    model_id = MIB_MODEL_TO_HF_ID[model_name]
    logger.info("Loading %s (%s)", model_name, model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if model_name == 'gpt2':
        tokenizer.padding_side = "right" # positional embeddings dont like left embedding
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if model_id != 'gpt2' else torch.float

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="eager", device_map="auto",
    ).eval()
    model = patch_model_for_lvp(model, **lvp_kwargs)

    adapter = adapter_cls(model, frozen_norm = lvp_kwargs['frozen_norm'], ignore_norm=args.ignore_norm)
    tracer = EdgeCircuitTracer(adapter, tokenizer)
    return tokenizer, adapter, tracer


def run() -> None:
    args = parse_args()

    lvp_kwargs = {
        "attn_act_fn": args.attn_act_fn,
        "matmul_fn": args.matmul_fn,
        "mul_fn": args.mul_fn,
        "frozen_norm": args.frozen_norm,
        "attn_softcap_fn": args.attn_softcap_fn,
        "center_writing_weights": args.center_writing_weights,
    }
    if args.mlp_act_fn is not None:
        lvp_kwargs["mlp_act_fn"] = args.mlp_act_fn

    logger.info("=" * 60)
    logger.info("DPA Circuit Attribution (EdgeCircuitTracer / LVP)")
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("models=%s  tasks=%s  split=%s  n=%d", args.models, args.tasks, args.split, args.num_examples)
    logger.info("method=%s  integration_steps=%d  counterfactual=%s", args.method_name, args.integration_steps, args.use_counterfactual)
    logger.info("lvp: %s", lvp_kwargs)
    logger.info("=" * 60)

    total_start = time.time()
    all_requested = [(task, model) for model in args.models for task in args.tasks]
    combos = [c for c in all_requested if c in ALL_COMBOS]
    skipped = [c for c in all_requested if c not in ALL_COMBOS]
    if skipped:
        logger.warning("Skipping unsupported (task, model) combos: %s", skipped)

    current_model_name = None
    tokenizer = adapter = tracer = None

    for task, model_name in combos:
        tag = f"{task.replace('_', '-')}_{model_name}"
        circuit_dir = os.path.join(args.output_dir, args.method_name, tag)
        mib_output_path = os.path.join(circuit_dir, "importances.json")
        score_output_path = os.path.join(circuit_dir, "scores.pt")

        if os.path.exists(mib_output_path) and args.force == False:
            logger.info("Skip %s (exists)", tag)
            continue

        if model_name != current_model_name:
            if tracer is not None:
                del tokenizer, adapter, tracer
                torch.cuda.empty_cache()
            tokenizer, adapter, tracer = _load_model_components(model_name, lvp_kwargs, args)
            current_model_name = model_name

        batch_size = args.batch_size or DEFAULT_BATCH_SIZES[model_name]
        logger.info("[%s] bs=%d", tag, batch_size)

        t0 = time.time()
        dataset = MIBDataset(task, tokenizer, model_name, split=args.split, num_examples=args.num_examples)
        dataloader = dataset.dataloader(batch_size)
        logger.info("  %d examples, %d batches", len(dataset), len(dataloader))

        scores = tracer.build_circuit(dataloader, use_counterfactual=args.use_counterfactual, integration_steps=args.integration_steps)
        circuit = create_mib_circuit(scores, adapter.n_layers, adapter.n_heads, adapter.model_dim)

        os.makedirs(circuit_dir, exist_ok=True)
        with open(mib_output_path, "w") as f:
            json.dump(circuit, f, indent=2)
        torch.save(scores, score_output_path)
        
        logger.info("  Done in %.1fs — saved %s", time.time() - t0, circuit_dir)


    logger.info("=" * 60)
    logger.info("Total: %.1f min", (time.time() - total_start) / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
