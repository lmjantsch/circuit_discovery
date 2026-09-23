# Circuit Discovery

Research code for **mechanistic interpretability**: finding the small subnetworks ("circuits") inside large language models that are responsible for a specific behaviour, and measuring how faithfully those circuits reproduce the full model.

The project builds on the [MIB benchmark](https://arxiv.org/abs/2504.13151) (Mechanistic Interpretability Benchmark) and develops a new attribution method that scores every edge of a transformer's computation graph from a single forward/backward pass, instead of the many passes that integrated-gradient baselines such as EAP-IG require.

## The problem

A transformer can be viewed as a graph: token embeddings, every attention head and every MLP write into a shared residual stream, and later components read from it. A *circuit* is the subset of edges between these components that actually carries a behaviour — e.g. how a model resolves "When Mary and John went to the store, John gave a drink to ___".

Circuit discovery methods assign an importance score to each edge. The benchmark then keeps the top-k% of edges, ablates the rest, and checks how much of the model's behaviour survives. Good methods recover most of the behaviour with very few edges.

## Approach

1. **Linearised propagation.** The model is patched (via the companion `modular_transformer` package) so that non-linear operations — softmax, activations, norms, bilinear products in attention and gated MLPs — follow explicit decomposition rules (secant rules, Deep Taylor Decomposition, uniform splitting). The derivation is in [`ctx/math.md`](ctx/math.md). This makes a component's forward contribution and a target's backward signal symmetric, so an edge score is simply a dot product between them.
2. **Edge tracing** ([`src/tracer.py`](src/tracer.py)). One forward pass caches each component's output, and one backward pass collects the gradient signal at each reader (per-head Q/K/V inputs, MLP inputs, the unembedding). All edges are scored at once as a single `[sources × targets]` matrix. The tracer supports counterfactual baselines, IG-style integration steps, norm matching between clean and corrupted runs, and streaming variance estimates (Welford/Chan) for measuring how stable a score is across samples.
3. **Faithfulness evaluation** ([`src/patcher.py`](src/patcher.py)). Given a score matrix, the patcher builds circuits at 10 sparsity levels (0.1%–100% of edges), prunes dangling nodes, and patches every non-circuit edge with its counterfactual activation. Results are reported as normalised logit-difference curves and area-under-curve metrics (CPR/CMD, plus log-scale variants).
4. **Model adapters** ([`src/adapters/`](src/adapters)). A small abstraction layer that exposes per-head attention and MLP hooks, so the same tracer and patcher code runs on several architectures (GPT-2, Llama/Qwen with GQA and rotary embeddings, Gemma 2 with soft-capping).

The tracing and patching run on [nnsight](https://nnsight.net/) and Hugging Face `transformers`, without TransformerLens.

## Scope

| | |
|---|---|
| **Models** | GPT-2, Qwen2.5-0.5B, Gemma-2-2B, Llama-3.1-8B |
| **Tasks** | IOI, MCQA, arithmetic (addition / subtraction), ARC-Easy, ARC-Challenge |
| **Granularity** | edge level (embeddings, individual attention heads with separate Q/K/V inputs, MLPs, logits) |
| **Baselines** | EAP, EAP-IG, and others via the official MIB circuit track (git submodule) |

## Repository layout

```
src/
  tracer.py              EdgeCircuitTracer: single-pass edge attribution
  patcher.py             EdgeCircuitPatcher: faithfulness evaluation by edge patching
  adapters/              architecture-specific hooks (GPT-2, Llama/Qwen, Gemma 2)
experiments/mib/
  run_attribution.py     compute circuits for model × task combinations
  run_evaluation.py      evaluate circuits, write CPR/CMD metrics
  data_utils.py          MIB dataset loading, conversion to MIB circuit JSON
  visualization/         faithfulness curves, circuit heatmaps, variance plots
  data_exploration/      notebooks on residual-stream statistics and norm approximation
  MIB-circuit-track/     official MIB benchmark (submodule) for baseline methods
ctx/
  math.md                derivation of the propagation rules
  legacy_tracer_v1|v2/   earlier tracer implementations, kept for reference
  modeling_files/        reference HF modeling code for supported architectures
```

## Usage

```bash
git submodule update --init --recursive
# requires torch, transformers, nnsight and the modular_transformer package

# 1. Attribute: score all edges and save circuits
CUDA_VISIBLE_DEVICES=0 python -m experiments.mib.run_attribution \
    --models qwen2.5 gemma2 --tasks ioi mcqa \
    --use-counterfactual --method-name my_method

# 2. Evaluate: patch circuits at each sparsity level
python experiments/mib/run_evaluation.py \
    --models qwen2.5 gemma2 --tasks ioi mcqa \
    --methods my_method --split validation

# 3. Plot
python -m experiments.mib.visualization.visualization --help
```

## Engineering highlights

- **All edges in one pass.** Edge scores are computed as a batched `einsum` over cached source activations and target gradients, so cost grows with model depth rather than with the number of edges.
- **Architecture-agnostic core.** New models are added by writing a thin adapter; tracer and patcher logic is shared.
- **Memory-conscious statistics.** Means and variances are accumulated online across batches, so no per-sample score tensors are stored.
- **Vectorised circuit patching.** Patch corrections for all Q/K/V inputs of a layer are computed in a single masked `einsum`, and graph pruning is done with boolean matrix operations.

## Status

This is active research code. Interfaces change as experiments evolve; earlier implementations are kept in `ctx/` for comparison.
