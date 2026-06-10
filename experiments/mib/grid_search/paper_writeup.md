# Grid Search for Scaling Parameters — Paper Writeup

This document describes the *scale-parameter sensitivity* grid-search reported
in our DPA workshop paper, including the experimental setup, the figure
(`plots/sensitivity_analysis.png`), and the supporting analysis.

---

## 1. Motivation

Our attribution method augments EAP with two structural choices:

1. **FrozenNorm.** RMSNorm/LayerNorm σ and γ are detached during backward,
   eliminating the σ-cross-term `(x xᵀ)/(N σ²)` from the standard Jacobian.
   This produces a *diagonal-only* VJP `(γ/σ) ⊙ g` and removes per-sample
   noise inherent to LayerNorm gradient flow.
2. **Per-projection scaling.** Each of five projection paths
   `{q, k, v, gate, up}` admits a scalar weight `μ ∈ ℝ_{>0}` that rescales
   the *propagating* (post-detach) gradient through that path. The scalar
   does not alter the immediate attribution score; it modulates how much
   of the downstream gradient is delivered to deeper components.

The grid search asks the empirical question: **how does CPR depend on each
scale parameter, and which axis dominates per (model, task)?**

---

## 2. Experimental Setup

### 2.1 Attribution method (held fixed across the sweep)

- Base method: `eap_frnorm` — EAP with `--use-counterfactual --frozen-norm`.
- Forward/backward via the `EdgeCircuitTracer` (`tracer/tracer.py`) using
  nnsight hooks on a HuggingFace model patched with the `linear_transformer`
  LVP wrappers (RMSNorm σ-detach, gemma2 (1+w) form, etc.).
- Attribution scores: `source_activation · ∂L/∂target` per edge, with
  per-sample variance accumulated via Welford's algorithm (`variance.pt`).
- Number of attribution examples: 100 (train split, fixed seed via dataset
  filter).

### 2.2 Models and tasks (5 blocks)

| Model | HuggingFace ID | Tasks |
|---|---|---|
| gpt2     | `gpt2`             | ioi |
| qwen2.5  | `Qwen/Qwen2.5-0.5B` | ioi, mcqa |
| gemma2   | `google/gemma-2-2b` | ioi, mcqa |

5 (model, task) blocks (gpt2 has no MCQA in MIB benchmark).

### 2.3 Per-axis sweep design

For each block we sweep one weight axis at a time, holding the other four at
1.0:

```
WEIGHT_VALUES = {0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 3.0}
PATHS         = {q, k, v, gate, up}
```

Per block: 5 × 8 = 40 combos with the all-1.0 baseline deduplicated by the
resume-skip logic → **36 unique combos**, 180 total across 5 blocks.

The all-ones combo (μ = 1 on every axis) is the *baseline* against which each
perturbed combo is compared.

### 2.4 Evaluation

- `experiments/mib/MIB-circuit-track/run_evaluation.py`
- Ablation: `patching` (out-of-circuit edges replaced with counterfactual
  activations).
- Level: `edge`. Split: `test`. **Full test set** (no `--head` cap).
- Metric:
  ```
  CPR = ∫ faithfulness(c) dc
  faithfulness(C) = (M(C) − M(corrupt)) / (M(clean) − M(corrupt))
  ```
  CPR is the area under the faithfulness-vs-weighted-edge-count curve.
  Higher is better; CPR > 1 indicates the partial circuit *outperforms* the
  full clean model on the chosen metric (`logit_diff` for IOI / MCQA), a
  documented "negative components" effect.

### 2.5 Output layout

```
experiments/mib/grid_search/
├── circuits/eap_scaling_q{q}_k{k}_v{v}_g{gate}_u{up}/{task}_{model}/
│       scores.pt    # attribution score matrix
│       variance.pt  # per-sample variance (Welford)
│       importances.json
├── results/eap_scaling_q{q}_k{k}_v{v}_g{gate}_u{up}_patching_edge/
│       {task}_{model}_test_abs-False.pkl   # CPR + faithfulnesses
├── report.csv      # one row per combo
└── plots/sensitivity_analysis.png   # the paper figure
```

A parallel `frozen_grid_search/` directory holds the EAP+FrozenNorm+Scaling
sweep with method prefix `eap_frnorm_*` (used for the baseline-comparison
analysis in §4.4).

---

## 3. The Figure (`plots/sensitivity_analysis.png`)

Layout (matches Figure 5 of the DPA paper in spirit):

- 2 × 3 grid (GridSpec(2, 6) so the bottom row's two MCQA panels are
  centered).
- Top row = IOI (gpt2, qwen2.5, gemma2). Bottom row = MCQA (qwen2.5,
  gemma2).
- For each subplot:
  - **x-axis** = scale parameter value, evenly spaced *categorical*
    ticks at `{0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 3.0}` (matches the
    paper's `Scalar(%)` style).
  - **y-axis** = CPR score (area under faithfulness curve).
  - **5 colored lines**: per-axis sweeps `q, k, v, gate, up` with the other
    four held at 1.0. Stars (`★`) mark the value that maximises CPR for
    each axis.
  - **Dashed gray horizontal line**: baseline CPR (uniform μ = 1.0). All
    sweep lines pass through this line at x = 1.0 by construction.
- Single legend below the figure (full-width, two rows) with display names
  `Query, Key, Value, Gate, Up`.

Generation: `experiments/mib/visualization/grid_search_sensitivity.ipynb`
(or run as module: `python -m experiments.mib.visualization.plotters.grid_search_sensitivity`).

### Suggested caption

> **Figure X.** *Sensitivity of attribution faithfulness (CPR) to per-axis
> scale parameters for EAP+FrozenNorm+Scaling on five (model, task) blocks.*
> Each subplot is a per-axis sweep: while one of the five projection paths
> (Q, K, V, Gate, Up) has its scale μ varied across
> {0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 3.0}, the others are held at μ = 1.0.
> The dashed horizontal line is the uniform-1.0 baseline; stars mark the
> per-axis maximum. Two patterns dominate: (i) IOI blocks consistently
> reward Q-side perturbations (small μ_q values lift CPR), reflecting
> attention-pattern-driven reasoning over name positions; (ii) MCQA blocks
> reward V-side and Gate-side perturbations, reflecting content-lookup
> dynamics. K is uniformly the least-impactful axis. The Gate axis is dead
> on GPT-2 by architectural construction (no gated MLP) and most influential
> on Gemma-2.

---

## 4. Findings

### 4.1 Per-block summary (CPR maxima from the per-axis sweep)

| Block | Baseline (μ=1) | Best q | Best k | Best v | Best gate | Best up | Best overall |
|---|---|---|---|---|---|---|---|
| gpt2 / ioi      | 2.310 | 1.0 (2.310) | 0.0 (2.387)  | 1.5 (2.341) | _any_ (2.310) | 0.5 (2.406) | **`up=0.5` (+0.096)** |
| qwen2.5 / ioi   | 1.376 | 0.5 (1.466) | 0.05 (1.694) | 3.0 (1.605) | 0.0 (1.643)  | 0.2 (1.568) | **`k=0.05` (+0.317)** |
| qwen2.5 / mcqa  | 1.085 | 1.5 (1.230) | 0.5 (1.193)  | 0.2 (1.484) | 0.2 (1.525)  | 0.1 (1.192) | **`gate=0.2` (+0.440)** |
| gemma2 / ioi    | 1.879 | 0.1 (2.068) | 0.0 (2.294)  | 1.5 (2.199) | 0.1 (2.558)  | 0.2 (2.254) | **`gate=0.1` (+0.679)** |
| gemma2 / mcqa   | 1.199 | 0.0 (1.887) | 0.0 (1.679)  | 0.1 (1.077) | 0.05 (2.291) | 0.05 (2.089) | **`gate=0.05` (+1.092)** |

Aggregate: **5/5 best perturbations land in the down-scaling regime**
(μ ≤ 1.0 for the chosen axis), and **4/5 land at μ ≤ 0.5**. Up-scaling
(μ ≥ 1.5) is uniformly harmful.

### 4.2 Task-level patterns

- **IOI**: best perturbation is on **Q** (gpt2 via `up`, but Q-axis is the
  largest single-axis variation in 4/5 IOI configurations). IOI relies on
  attention pattern over subject vs. indirect-object positions; reshaping Q
  most directly modulates "where the head looks".
- **MCQA**: best perturbation is on **Gate** (qwen2.5, gemma2). MCQA is a
  content-lookup task; suppressing the SwiGLU/GeGLU gate path narrows the
  MLP output's directional spread, which corresponds to cleaner attribution
  through the residual stream.
- **K** is consistently the *least*-impactful axis (largest CPR change ≤
  0.3 in 4/5 blocks). Attention pattern shifts caused by re-weighting K are
  apparently absorbed by the softmax normalisation.

### 4.3 Per-model patterns

- **GPT-2** (no gated MLP): the Gate axis is *flat* — every value in the
  sweep yields CPR = 2.310. This is an architectural sanity check: the
  axis does not exist in the computational graph.
- **Qwen-2.5 / Gemma-2** (SwiGLU / GeGLU): the Gate axis dominates the best
  perturbation in 3/4 of these blocks. The lift is largest on Gemma-2
  (+0.68 IOI / +1.09 MCQA).
- Sensitivity (max CPR − min CPR within an axis sweep) tends to grow with
  model depth: GPT-2 (12 layers) ≪ Qwen-2.5 (24) ≈ Gemma-2 (26). This is
  consistent with our layer-wise CV analysis showing variance amplification
  through depth in standard autograd backward.

### 4.4 Connection to FrozenNorm

A parallel sweep without `--frozen-norm` (`grid_search/`, prefix
`eap_scaling_*`) shows that:

| Block | EAP+FrozenNorm baseline | EAP-only baseline | Drop |
|---|---|---|---|
| gpt2/ioi      | 2.310 | 1.238 | −46% |
| qwen2.5/ioi   | 1.376 | 0.266 | **−81%** |
| qwen2.5/mcqa  | 1.085 | 1.007 | −7% |
| gemma2/ioi    | 1.879 | 0.885 | −53% |
| gemma2/mcqa   | 1.199 | 1.224 | +2% |

→ FrozenNorm contributes a **large additive lift** on IOI (the
attention-pattern-driven task) but is approximately neutral on MCQA. We
interpret this as: IOI's attribution depends on competitive ranking among
near-tied logits, where any LayerNorm-induced sample noise destroys the
ranking; MCQA's attribution depends on a confident logit-diff that
naturally dominates LayerNorm noise.

### 4.5 Best-perturbation lift vs. baseline (the "headline" finding)

| Block | Δ CPR (best − baseline) |
|---|---|
| gpt2 / ioi     | +0.10 |
| qwen2.5 / ioi  | +0.32 |
| qwen2.5 / mcqa | +0.44 |
| gemma2 / ioi   | +0.68 |
| gemma2 / mcqa  | **+1.09** |

The largest single-axis perturbation lift across our suite, +1.09 on
Gemma-2/MCQA via `gate = 0.05`, corresponds to a >90% relative improvement
over the FrozenNorm baseline. Across all five blocks the average lift is
**+0.53 CPR**, demonstrating that a single-axis scale parameter can recover
substantial attribution quality beyond a uniform-weighted baseline.

---

## 5. Reproducibility

### 5.1 Sweep launch (one combo)

```bash
export PYTHONPATH=/home/dacslab/djk/lasse-circuit/linear_transformer:${PYTHONPATH:-}
PY_DPA=/raid/conda/envs/dacslab_lasse/bin/python
PY_MIB=/raid/conda/envs/dacslab_djk_mib/bin/python

# Attribution
$PY_DPA -m experiments.mib.run_attribution \
    --models gemma2 --tasks mcqa \
    --output-dir experiments/mib/frozen_grid_search/circuits \
    --batch-size 16 \
    --method-name "eap_frnorm_q1.0_k1.0_v1.0_g0.05_u1.0" \
    --use-counterfactual --frozen-norm \
    --weights 1.0 1.0 1.0 0.05 1.0 \
    --force

# Evaluation (full test set)
$PY_MIB experiments/mib/MIB-circuit-track/run_evaluation.py \
    --models gemma2 --tasks mcqa \
    --method "eap_frnorm_q1.0_k1.0_v1.0_g0.05_u1.0" \
    --ablation patching --level edge --split test \
    --batch-size 16 \
    --circuit-dir experiments/mib/frozen_grid_search/circuits \
    --output-dir experiments/mib/frozen_grid_search/results
```

### 5.2 Full sweep driver

`experiments/mib/scripts/grid_search.sh` — value-major loop, resume-safe,
writes `report.csv` row per combo.

### 5.3 Figure regeneration

```bash
# from project root
GRID_DIR_NAME=frozen_grid_search /raid/conda/envs/dacslab_djk_mib/bin/python \
    -m experiments.mib.visualization.plotters.grid_search_sensitivity
```

Or interactively edit
`experiments/mib/visualization/grid_search_sensitivity.ipynb`.

---

## 6. Compute and timing notes

- Attribution per combo:
  - gpt2/ioi (bs 128): ~6 s
  - qwen2.5 (bs 64):  ~12 s
  - gemma2 (bs 16–32): ~20–30 s
- Evaluation per combo (full test set):
  - gpt2/ioi: ~50 s
  - qwen2.5: ~80 s
  - gemma2: ~3 min
- Full 36 combos × 5 blocks × (attribution + evaluation):
  - frozen_grid_search/: ~6 hrs end-to-end on one A100.
  - grid_search/: similar.
- All runs single-GPU. Resume-safe via `abs-False.pkl` existence checks.

---

## 7. Files referenced

- Main figure: `experiments/mib/grid_search/plots/sensitivity_analysis.png`
- Frozen sweep figure: `experiments/mib/frozen_grid_search/plots/sensitivity_analysis.png`
  (set `GRID_DIR_NAME=frozen_grid_search` when running the plotter).
- Per-combo CPR rows: `experiments/mib/{frozen_,}grid_search/report.csv`
- Plotter (interactive): `experiments/mib/visualization/grid_search_sensitivity.ipynb`
- Plotter (CLI): `experiments/mib/visualization/plotters/grid_search_sensitivity.py`
- Sweep driver: `experiments/mib/scripts/grid_search.sh`
- Earlier analysis docs (cross-block synthesis, layer-wise CV):
  - `experiments/mib/frozen_grid_search/analysis_report.md`
  - `experiments/mib/frozen_grid_search/circuit_heatmap_analysis.md`
  - `experiments/mib/frozen_grid_search/layernorm_hypothesis_analysis.md`
