# Method comparison — implementation writeup

This document explains **how every row in `report.csv` is produced**, from
the high-level pipeline driver (`scripts/method_comparison.sh`) down to
the autograd-level building blocks. The goal is to make explicit
(a) how we replicate the standard EAP algorithm on top of **nnsight**
instead of TransformerLens, and (b) how each ablation
(`FrozenNorm`, `MLPsecant`, `Bilinear`, `Scaling`, `IG(k=5)`) plugs into
that base.

`report.csv` columns are `timestamp, method, model, task, cpr`. Each row
is the `area_under` field of the corresponding
`results/<method>_patching_edge/<task>_<model>_test_abs-False.pkl` file
produced by `MIB-circuit-track/run_evaluation.py`.

---

## 1. The seven methods, and what each flag changes

The driver script defines a fixed sweep:

```
METHODS = [
    eap_pure,
    eap_ig_k5,
    eap_frnorm,
    eap_frnorm_secantmlp,
    eap_frnorm_secantmlp_bilinear,
    eap_frnorm_secantmlp_scale_k02g02,
]
# + one extra variant, run as its own script:
#   eap_frnorm_secantmlp_bilinear_ig5
```

The mapping from method-name to `run_attribution.py` flags lives in
`build_attr_flags()` inside `scripts/method_comparison.sh` (and in
`scripts/method_comparison_ig5.sh` for the 7th variant). The cumulative
ablation order is:

| Method                                        | Flags (in addition to `--use-counterfactual`)                                                            |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `eap_pure`                                    | —                                                                                                        |
| `eap_ig_k5`                                   | `--integration-steps 5`                                                                                  |
| `eap_frnorm`                                  | `--frozen-norm`                                                                                          |
| `eap_frnorm_secantmlp`                        | `--frozen-norm`, `--mlp-act-fn {secant_gelu | secant_silu | secant_gelu_tanh}`                           |
| `eap_frnorm_secantmlp_bilinear`               | + `--matmul-fn bilinear_matmul`, `--mul-fn bilinear_mul`                                                 |
| `eap_frnorm_secantmlp_scale_k02g02`           | + `--weights 1.0 0.2 1.0 0.2 1.0` (i.e. `k=g=0.2`)                                                       |
| `eap_frnorm_secantmlp_bilinear_ig5`           | bilinear variant + `--integration-steps 5`                                                               |

Per-model MLP secant choice (matches each model's native activation):
`gpt2 → secant_gelu`, `qwen2.5 → secant_silu`, `gemma2 → secant_gelu_tanh`.

`--use-counterfactual` is always on so the metric is the counterfactual
logit-diff `logit(clean) − logit(corrupt)` summed over the batch — the
standard EAP target. This makes the gradient match the EAP definition
even when `--integration-steps == 1`.

---

## 2. nnsight-based EAP base (`tracer/`)

Standard EAP scores each edge `s → t` as

```
score(s, t) = ⟨ activation_s ,  ∂metric / ∂t ⟩
```

where `activation_s` is the (clean − corrupt) activation of the source
node and `∂metric/∂t` is the gradient of the loss flowing into the
target node. The MIB reference implementation uses TransformerLens hooks
plus `eap-ig` for caching. Our implementation reproduces the same dot
product using **nnsight** trace contexts and a custom backward pass
encoded inside `tracer/`.

### 2.1 Entry point — `experiments/mib/run_attribution.py`

`run_attribution.py` loads a HuggingFace model with
`attn_implementation="eager"`, wraps every supported submodule via
`linear_transformer.patch_model_for_lvp`, and then wraps the whole model
in `NNsight(...)`. The wrapper is what enables `with model.trace(...)`
and `with model.session()` contexts. The patched submodules (described
in §3) leave forward outputs numerically identical to the reference
model — they only replace the autograd `backward`s used by
`metric.backward()`.

The script then builds:
- a `ModelAdapter` (Llama2 / Gemma2 / GPT2 specific) that knows where
  every tensor lives in the wrapped model and how to compute the
  pre-norm gradient analytically;
- an `EdgeCircuitTracer(adapter, tokenizer, …)` that holds the running
  score matrix and orchestrates the forward/backward sweep.

### 2.2 `EdgeCircuitTracer.build_circuit` — main loop

The score matrix is shaped
`[source_dims, grad_dims]` with
- `source_dims = 1 + n_layers·(n_heads + 1)` — embedding (1) + per-layer
  `n_heads` attention sources + 1 MLP source;
- `grad_dims = n_layers·(3·n_heads + 1) + 1` — per-layer q/k/v
  (`3·n_heads`) + 1 MLP + lm_head.

For each batch (`tracer/tracer.py:80`), the loop is:

1. **Open an nnsight session.** Everything below runs inside
   `model.session()` so nnsight intercepts forward outputs and backward
   gradients without us re-implementing the forward graph.

2. **Corrupt forward (only if counterfactual / IG).** A
   `model.trace(corrupt_inputs)` populates `self.source_cache` with the
   corrupt-side residuals for every source node and stores
   `corrupt_embeds` for later interpolation.
   ```python
   if use_counterfactual or integration_steps > 1:
       with self.model.trace(**corrupt_inputs):
           self._forward_pass_and_cache()
       self.corrupt_embeds = self.cache['emb'].detach().clone()
       self.baseline_cache = self.source_cache
       self.source_cache  = self._init_source_cache()
   ```

3. **Clean forward.** A second `model.trace(clean_inputs)` re-populates
   `self.source_cache` with clean-side residuals. After both passes,
   `source_cache ← clean − corrupt`. This is the EAP source term.
   ```python
   with self.model.trace(**clean_inputs):
       self._forward_pass_and_cache()
       metric = self._get_metric(...)        # logit-diff
       if use_counterfactual or integration_steps > 1:
           self.source_cache -= self.baseline_cache
       self._backward_pass_and_scoring(metric)
   ```

4. **Backward.** Inside the same trace context, `metric.backward()` runs
   *under the patched autograd graph* (§3). The tracer reads
   `tensor.grad` at every cached node and combines it with the matching
   source slice via an `einsum`:
   ```python
   per_sample_scores = torch.einsum('ibsd,jbsd->bij', sources, grad)
   per_token_scores  = per_sample_scores / n_real_tokens
   # Welford running-mean update of self.circuit_scores
   ```
   Welford gives us per-sample variance for free if `--variance-type` is
   set.

5. **IG loop (only if `integration_steps > 1`).** Repeat the
   clean-forward + backward at `k − 1` interpolated embeddings:
   ```python
   for alpha in torch.linspace(0, 1, k+1)[1:-1]:
       with self.model.trace(**clean_inputs):
           integrated_embeds = (1 - alpha) * self.clean_embeds \
                             + alpha       * self.corrupt_embeds
           self._forward_pass_and_cache(integrated_embeds)
           metric = self._get_metric(...)
           self._backward_pass_and_scoring(metric)
   ```
   `source_cache` is frozen during this loop
   (`disable_source_caching = True`), only the gradient varies — exactly
   IG-inputs as defined in the MIB paper.

### 2.3 Forward caching — `_forward_pass_and_cache`

The forward sweep saves, per layer, all the tensors that appear on
either side of an EAP edge:

| Cache key                              | What it is (nnsight handle in `tracer/model_adapters.py`)                   |
|----------------------------------------|------------------------------------------------------------------------------|
| `cache['emb']`                         | `model.embed_tokens.output` — embedding source                              |
| `cache[-1]['out']`                     | residual at layer-0 input                                                   |
| `cache[L]['query_out'/'key_out'/'value_out']` | `layer.self_attn.q_proj.output` and the post-`repeat_kv` outputs        |
| `cache[L]['mid']`                      | input to `post_attention_layernorm`                                         |
| `cache[L]['gate_out']/['up_out']`      | `layer.mlp.gate_proj.output` / `layer.mlp.up_proj.output`                   |
| `cache[L]['out']`                      | `layer.output`                                                              |
| `cache['logits']`                      | `model.lm_head.output`                                                      |

At the same time, `source_cache` (shape `[source_dims, B, S, d]`) is
filled with the actual *source vectors* (residual at layer 0 = embedding
source, `per_head_attn_out` for attention heads, `mlp_out` for MLP). For
Gemma2, `per_head_attn_out` manually re-applies the post-attention
RMSNorm so the source matches the residual contribution
(`tracer/model_adapters.py:468`).

### 2.4 Backward + scoring — `_backward_pass_and_scoring`

After `metric.backward()` (executed inside the still-open trace
context via `with metric.backward(): …`), the tracer walks layers
backwards and, for each target node, pulls `tensor.grad` from the
forward-cached output and contracts it with the source cache.

For attention/MLP targets the *post-projection* gradient is propagated
through the projection matrix and through the pre-norm analytically by
the adapter's `_compute_norm_input_gradient` and
`_compute_headwise_attn_gradient` helpers
(`tracer/model_adapters.py:396–453`). This is the same back-substitution
that EAP-IG does inside TransformerLens, but expressed directly via
`einsum`s instead of TL hooks. The frozen-norm branch (§3.1) is exactly
the diagonal-only part of that closed form.

The `lm_head` gradient is read off the residual at the final layer:
`cache[n_layers−1]['out'].grad`.

The end result, after the full dataloader is consumed, is the score
tensor `self.circuit_scores` of shape `[source_dims, grad_dims]`. This
tensor is saved as `circuits/<method>/<task>_<model>/scores.pt`, and
`experiments/mib/data_utils.create_mib_circuit` converts it into the
MIB-compatible `importances.json` (nodes/edges schema) without ever
touching TransformerLens.

---

## 3. Per-ablation implementation

All of `linear_transformer/patching/registry.py` is class-level
dispatch: when the patcher walks the model tree it replaces
`Gemma2RMSNorm → LinearGemma2RMSNorm`, `Gemma2MLP → LinearGemma2GeGLU`,
etc. The kwargs of `patch_model_for_lvp` (forwarded from the CLI flags)
are threaded into each replacement's `from_module`, so a single
`run_attribution.py` invocation can toggle every ablation independently.

### 3.1 FrozenNorm (`--frozen-norm`)

Standard RMSNorm backward includes a "σ-cross-term" that recomputes
itself for every token (the `u·x · x / (N·σ²)` term in
`_compute_norm_input_gradient`). It introduces position-dependent noise
that EAP scores are sensitive to.

`--frozen-norm` does two complementary things:

1. **Patched forward** (`linear_transformer/models/gemma2.py:18`,
   analogous for Llama2/Qwen2/GPT2):
   ```python
   def _norm(self, x):
       if self.is_linear:
           return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps).detach()
       return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
   ```
   `.detach()` on both σ and γ removes them from the autograd graph, so
   `metric.backward()` propagates only the diagonal part — equivalent
   to treating the norm as a constant linear adjoint.

2. **Adapter branch** (`tracer/model_adapters.py:419`): the analytic
   pre-norm gradient used for q/k/v/MLP targets takes the same
   diagonal-only shortcut.
   ```python
   if self.frozen_norm:
       grad_pre_norm = u / sigma
       return grad_pre_norm
   ```

Mathematically, this matches the legacy `eap_compat=False` default
exactly on all four Gemma2 norms (input, post-attn, pre-FFN, post-FFN);
that equivalence is what gave us the empirically observed
`+0.98` average CPR jump over `eap_pure`.

### 3.2 MLP secant (`--mlp-act-fn secant_*`)

The standard backward of GELU/SiLU multiplies the upstream gradient by
the *derivative* `f′(x)`. "Secant" replaces that with the *secant*
`f(x)/x` — the average slope from 0 to `x` rather than the local slope.
For mass-conserving attribution this avoids the sign-flip and
near-zero-derivative pathologies in the original activations.

Per-activation implementation lives in
`linear_transformer/modules/activations.py`:

```python
class SecantGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return F.gelu(x.float()).to(x.dtype)            # forward unchanged

    @staticmethod
    def backward(ctx, grad_t):
        (x,) = ctx.saved_tensors
        c = 0.5 * (1.0 + torch.erf(x.float() / math.sqrt(2.0)))   # Φ(x)
        return (grad_t.float() * c).to(ctx._orig_dtype)
```

`SecantSiLU` uses `c = sigmoid(x)`; `SecantGELUTanh` uses the tanh-approx
form. These are wired into the patched MLP via
`act_fn = ACT_FN[kwargs['mlp_act_fn']]`, so the forward output is byte
identical to PyTorch's `F.gelu`/`F.silu` and only the gradient changes.

### 3.3 Bilinear (`--matmul-fn bilinear_matmul --mul-fn bilinear_mul`)

Three bilinear ops appear in every transformer block:
- `Q @ Kᵀ` (attention scores)
- `A @ V`  (attention output)
- `gate * up` (SwiGLU / GeGLU)

The standard autograd treats each operand as receiving the *full*
gradient with respect to the product. LVP redistributes the gradient
between operands with a weight split. Implementation
(`linear_transformer/modules/bilinear.py`):

```python
class BilinearMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, x_weight=0.5, y_weight=0.5):
        ctx.save_for_backward(x, y)
        ctx._x_weight = x_weight
        ctx._y_weight = y_weight
        return torch.matmul(x.float(), y.float()).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_t):
        x, y = ctx.saved_tensors
        grad_x = (ctx._x_weight * grad_t @ y.float().mT)
        grad_y = (ctx._y_weight * x.float().mT @ grad_t)
        return grad_x, grad_y, None, None
```

(One subtle fix during development: `save_for_backward` only accepts
tensors, so float weights had to be stashed as `ctx._x_weight` /
`ctx._y_weight` attributes.) `bilinear_mul` is the elementwise sibling.
Forward outputs are identical to `torch.matmul` / `torch.mul`; only the
backward changes the q/k and gate/up gradient routing.

### 3.4 Scaling (`--weights q k v gate up`)

Per-projection gradient scaling is applied *outside* the autograd graph,
inside `EdgeCircuitTracer._scale_and_detach_grad`
(`tracer/tracer.py:289`):

```python
def _scale_and_detach_grad(self, grad_tensor, weight, scale_loc='post'):
    if scale_loc == 'pre':
        grad_tensor.grad = weight * grad_tensor.grad
    detached_grad = grad_tensor.grad.detach()
    if scale_loc == 'post':
        grad_tensor.grad = weight * grad_tensor.grad
    return detached_grad
```

`scale_loc='pre'` rescales the projection's contribution to the score
*itself*. `scale_loc='post'` (default) leaves the current score alone
but rescales the gradient that *propagates further downstream* —
useful when you want to attenuate the influence of one projection on
the rest of the graph without zeroing its own attribution. The flag is
ordered `q k v gate up`; the `k=g=0.2` variant of `report.csv` therefore
maps to `--weights 1.0 0.2 1.0 0.2 1.0`.

### 3.5 IG-inputs k=5 (`--integration-steps 5`)

The IG path is the third loop in `_build_circuit_inner`
(`tracer/tracer.py:116`):

```python
for alpha in torch.linspace(0, 1, k+1)[1:-1]:  # 1/k, 2/k, …, (k-1)/k
    with self.model.trace(**clean_inputs):
        integrated_embeds = (1 - alpha) * clean_embeds + alpha * corrupt_embeds
        self._forward_pass_and_cache(integrated_embeds)
        metric = self._get_metric(...)
        self._backward_pass_and_scoring(metric)
```

The interpolated embedding is injected with `self.adapter.emb.output =
integrated_embeds` inside the trace context, which is the nnsight idiom
for replacing a node's forward value. `disable_source_caching = True`
fixes the source term to the original (clean − corrupt) difference, so
only the gradient varies across α — matching the IG-inputs definition
(`g(α) · (x_clean − x_corrupt)` integrated over α). The running-mean
update in `_update_scores` already handles the `1/k` averaging
implicitly via `num_processed_samples`.

---

## 4. Pipeline (`scripts/method_comparison.sh`)

The driver runs three phases over a fixed (model, task) list:

```
ORDER = [gpt2:ioi:128, qwen2.5:ioi:64, qwen2.5:mcqa:64,
         gemma2:ioi:32, gemma2:mcqa:16]
```

- **Phase 1 — attribution** (lasse env, `dacslab_lasse`): for every
  (method, model, task) it invokes `run_attribution.py` with the flags
  from `build_attr_flags`. Skips combos where
  `circuits/<method>/<task>_<model>/scores.pt` already exists, so the
  script is resume-safe.

- **Phase 2 — evaluation** (mib env, `dacslab_djk_mib`): runs
  `MIB-circuit-track/run_evaluation.py --split test` to compute the CPR.
  Output goes to
  `results/<method>_patching_edge/<task>_<model>_test_abs-False.pkl`. A
  small inline Python block opens the pickle, reads `area_under`, and
  appends one row to `report.csv`. Skips combos whose `abs-False.pkl`
  already exists.

- **Phase 3 — report build**: a deduplicating Python block reads
  `report.csv`, keeps the latest row per `(method, model, task)`, and
  emits `report.md` with the main matrix, Δ-vs-pure-EAP, and per-block
  best.

`PYTHONPATH` is exported to the local `linear_transformer/` checkout so
that the lasse env's editable install does **not** shadow our copy
(which is the one carrying `IdentityTanh`, `DiagSoftmax`, and the
bilinear fixes).

The crash-resume script (`method_comparison_resume.sh`) hardcodes the 9
combos that were missing at the time of the server restart; the IG5
extension (`method_comparison_ig5.sh`) appends the 7th variant. Both
append to the same `report.csv`, and Phase 3's latest-wins dedupe makes
duplicate rows from re-runs benign.

---

## 5. Output structure

```
experiments/mib/method_comparison/
├── circuits/<method>/<task>_<model>/
│       ├── importances.json   # MIB-format graph (nodes, edges, scores)
│       ├── scores.pt          # raw [source_dims, grad_dims] tensor
│       └── variance_*.pt      # optional Welford variance
├── results/<method>_patching_edge/
│       └── <task>_<model>_test_abs-False.pkl   # MIB evaluation result
├── logs/<model>__<task>__<method>.log
├── report.csv                  # one row per (method, model, task)
├── report.md                   # human-readable matrix + deltas
└── progress.txt                # append-only ATTR/EVAL marker log
```

`report.csv` is the canonical source of truth for everything in the
sweep; `report.md` is regenerated from it on every Phase-3 run.
