# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **mechanistic interpretability research project** implementing and extending the [MIB benchmark](https://arxiv.org/abs/2504.13151) for circuit discovery in transformer models. The goal is to discover and evaluate neural circuits — subgraphs of important components (edges/nodes) — across 4 models and 6 tasks.

## Repository Structure

The repo has two layers:

- **`experiments/mib/`** — custom experiment code, primarily a faster DPA (Dual Path Attribution) method
- **`experiments/mib/MIB-circuit-track/`** — git submodule containing the MIB benchmark framework (EAP, EAP-IG, OA, etc.)

The MIB-circuit-track submodule itself wraps a local `eap` package (EAP-IG library, installed via `pip install .`) and uses **TransformerLens** for model hooking.

## Setup

```bash
cd experiments/mib/MIB-circuit-track
git submodule update --init --recursive
pip install .
# For optimal ablation:
pip install nnsight==0.2.15
# For circuit visualization:
pip install EAP-IG[viz]
```

## Common Commands

### Attribution (circuit discovery)

```bash
# EAP / EAP-IG methods (from MIB-circuit-track)
cd experiments/mib/MIB-circuit-track
python run_attribution.py \
  --models qwen2.5 gemma2 \
  --tasks ioi mcqa \
  --method EAP-IG-inputs \
  --level edge \
  --ablation patching \
  --batch-size 20

# DPA (custom fast method, from repo root)
CUDA_VISIBLE_DEVICES=0 python -m experiments.mib.run_attribution
```

Methods: `EAP`, `EAP-IG-inputs`, `EAP-IG-activations`, `IFR`, `UGS`, `OA`.  
Levels: `edge` (default), `node`, `neuron`.  
Ablations: `patching` (default), `zero`, `mean`, `mean-positional`, `optimal`.

### Evaluation

```bash
cd experiments/mib/MIB-circuit-track
python run_evaluation.py \
  --models qwen2.5 gemma2 \
  --tasks ioi mcqa \
  --method EAP-IG-inputs \
  --ablation patching \
  --level edge \
  --split validation   # or: test
```

### Results

```bash
python print_results.py --output-dir results --metric cpr --split validation
# metrics: cpr, cmd, auroc

# Generate markdown report from DPA results
python experiments/mib/create_report.py \
  "experiments/mib/MIB-circuit-track/results" \
  "experiments/mib/experiment_results.md"
```

### Optimal Ablation (two-step)

```bash
python oa.py --models qwen2.5 llama3 --tasks ioi mcqa
python run_attribution.py --method EAP --ablation optimal \
  --optimal_ablation_path <path>
```

## Architecture

### Computation Graph

Circuits are represented as directed graphs where:
- **Nodes**: `input`, `logits`, `a{L}.h{H}` (attention head), `m{L}` (MLP at layer L)
- **Edges**: `source->target<qkv>` — attention edges carry separate q/k/v scores (e.g., `input->a0.h0<q>`)
- **Output format**: JSON with `cfg`, `nodes`, `edges` keys; edges carry `score` and `in_graph` fields

### Key Modules in MIB-circuit-track

| File | Purpose |
|------|---------|
| `MIB_circuit_track/dataset.py` | `HFEAPDataset` — loads MIB tasks from HuggingFace |
| `MIB_circuit_track/metrics.py` | Task-specific loss functions (`logit_diff` is default) |
| `MIB_circuit_track/evaluation.py` | `evaluate_area_under_curve()` — CPR/CMD at 10 percentile points |
| `MIB_circuit_track/utils.py` | Model name registry, task→column mappings |
| `experiments/mib/create_mib_circuite.py` | Converts DPA score dict → MIB-compatible JSON (no TransformerLens dependency) |

### Supported Models & Tasks

**Models**: `gpt2`, `qwen2.5` (Qwen2.5-0.5B), `gemma2` (gemma-2-2b), `llama3` (Llama-3.1-8B), `interpbench`  
**Tasks**: `ioi`, `mcqa`, `arithmetic_addition`, `arithmetic_subtraction`, `arc_easy`, `arc_challenge`

### Evaluation Metrics

- **CPR** (Circuit Performance Ratio): `area_under` in result PKLs — faithfulness relative to full model
- **CMD** (Circuit Magnitude Difference): `area_from_1` — how close circuit performance is to 1.0
- **AUROC**: Used for InterpBench task only

Results are stored as `.pkl` files in `MIB-circuit-track/results/` and circuits as `.json` in `MIB-circuit-track/circuits/`.

### DPA Method (custom)

The custom DPA implementation in `experiments/mib/run_attribution.py` uses backward passes to trace dual-path edge attributions without running the full EAP-IG pipeline. It produces score dicts consumed by `create_mib_circuite.py` to build the graph JSON independently of TransformerLens.

### CircuitTracer (`tracer/`)

A standalone nnsight-based circuit tracer. Entry point is `CircuitTracer.__call__(batch)` where `batch = (clean_prompt, _, clean_targets, corrupt_targets)`.

| File | Purpose |
|------|---------|
| `tracer/tracer.py` | `CircuitTracer` — forward cache + backward scoring loop |
| `tracer/backend.py` | `compute_rmsnorm_input_gradients`, `compute_headwise_input_gradients` — gradient backprop through RMSNorm and linear projections (with optional inverse RoPE) |
| `tracer/modeling_utils.py` | `per_head_attn_out` — splits attention output by head via `o_proj`; `apply_inverse_rope` — undoes RoPE for k/q grad backprop |

**Score matrix layout** (`circuit_scores` shape `[source_dims, grad_dims]`):

- `source_dims = 1 + n_layers * (n_heads + 1)` — embedding (1) + per-layer: heads (n_heads) then MLP (1)
- `grad_dims = n_layers * (3 * n_heads + 1) + 1` — per-layer: q/k/v heads (3×n_heads) then MLP (1), plus lm_head (1)

**Assumptions**: Qwen2.5-style architecture — `model.model.embed_tokens`, `model.model.rotary_emb`, `model.model.layers[i]` with `.self_attn`, `.mlp`, `.input_layernorm`, `.post_attention_layernorm`. GQA is handled in `compute_headwise_input_gradients` via `repeat_interleave`.

## Testing

The root `test.ipynb` and `importances.json` / `test_importance.json` files are used for interactive prototyping. There is no automated test suite.
