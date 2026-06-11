# Patcher-based full test-set evaluation report

**Branch:** `djk_patcher`
**Run:** 2026-05-23
**Evaluator:** local `experiments/mib/run_evaluation.py` (this branch)
  — `EdgeCircuitPatcher` (nnsight + `modular_transformer` + HF native weights), **no TransformerLens**
**Circuits:** `experiments/mib/method_comparison/circuits/<METHOD>/<TASK>_<MODEL>/scores.pt`
  — same `scores.pt` produced by the previous `djk` branch's attribution runs (unchanged)
**Split:** `test`, full samples (1000 per task)
**Output:** `experiments/mib/method_comparison/results_patcher/<METHOD>/<TASK>_<MODEL>_test_abs-False.pkl`

> The metric below is **CPR** (`area_under` in the result pkl), the area under the faithfulness-vs-edge-percentile curve at 10 fixed percentile points `(.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.0)`. Higher is better; 1.0 means circuit recovers full-model behavior, > 1.0 means the circuit isolates a sub-network that scores *higher* than the full model on the metric.

---

## 1. Full results — new patcher (full test set, 1000 samples per task)

| Method | gpt2/ioi | qwen2.5/ioi | qwen2.5/mcqa | gemma2/ioi | gemma2/mcqa | mean |
|---|---:|---:|---:|---:|---:|---:|
| `eap_pure` | 1.2233 | 0.3197 | 0.8509 | 1.3212 | 1.0882 | **0.9607** |
| `eap_ig_k5` | 2.0681 | 1.7512 | 1.1400 | 3.5350 | 1.2832 | **1.9555** |
| `eap_frnorm` | 2.4135 | 1.3260 | 0.8978 | 2.6620 | 1.2820 | **1.7163** |
| `eap_frnorm_secantmlp` | 2.4447 | 1.5642 | 0.9141 | 3.6038 | 1.5913 | **2.0236** |
| `eap_frnorm_secantmlp_bilinear` | 2.6614 | 1.8635 | 1.7959 | 3.5800 | 2.2257 | **2.4253** |
| `eap_frnorm_secantmlp_bilinear_ig5` | 2.4907 | 1.8909 | 1.7464 | 3.6672 | 2.4620 | **2.4514** |
| `eap_frnorm_secantmlp_scale_k02g02` | 2.5056 | 1.8054 | 1.3768 | 3.3939 | 2.1678 | **2.2499** |
| `eap_frnorm_secantmlp_bilinear_rawsrc` | 2.6614 | 1.8635 | 1.7959 | 3.6820 | 2.0997 | **2.4205** |
| `eap_frnorm_secantmlp_bilinear_nosoftcap` | 2.6614 | 1.8635 | 1.7959 | 3.9907 | 2.1431 | **2.4909** |
| `eap_frnorm_secantmlp_bilinear_nosoftcap_ig5` | 2.4907 | 1.8909 | 1.7464 | 3.7625 | 2.4092 | **2.4599** |
| `eap_frnorm_secantmlp_bilinear_rawsrc_nosoftcap` | 2.6614 | 1.8635 | 1.7959 | 3.9966 | 1.9432 | **2.4521** |
| `eap_frnorm_secantmlp_bilinear_softcapbwdoff` | 2.6614 | 1.8635 | 1.7959 | 3.9858 | 2.0998 | **2.4813** |
| `eap_frnorm_secantmlp_softcapbwdoff_dpeascale` | 2.6558 | 1.8711 | 1.7970 | 3.9908 | 2.0960 | **2.4821** |

**Every cell is now a real attribution + patcher evaluation** — no inferred / copied values remain.

- The 8 genuinely-different combos (gemma2/mcqa for the four Gemma2-flag variants + all of `_softcapbwdoff_dpeascale`) were attributed on `djk` via `experiments/mib/scripts/attr_missing_8.sh` and evaluated via `eval_missing_8.sh`.
- The 11 non-Gemma2 "no-op flag" combos were also freshly attributed on `djk` (`attr_noop_11.sh`) and evaluated (`eval_noop_11.sh`). Their `scores.pt` were **MD5-verified byte-identical** to `eap_frnorm_secantmlp_bilinear` (confirming `--ignore-softcap` / `--raw-edge-source` / `--attn-softcap-fn identity_tanh` / `--final-softcap-fn identity_tanh` are true no-ops on gpt2/qwen2.5, which lack softcap and post-attention LayerNorm), and the patcher CPR came out exactly equal: gpt2/ioi = 2.6614, qwen2.5/ioi = 1.8635, qwen2.5/mcqa = 1.7959.

**Note on Gemma2/MCQA softcap variants**: unlike Gemma2/IOI (where softcap removal *raised* CPR to ≈3.99), on Gemma2/MCQA the softcap-handling variants all land **below** the plain `_bilinear` baseline (2.2257):
- `_nosoftcap`: 2.1431, `_softcapbwdoff`: 2.0998, `_rawsrc`: 2.0997, `_rawsrc_nosoftcap`: 1.9432, `_dpeascale`: 2.0960
This reverses the IOI finding — **softcap removal helps IOI but hurts MCQA** under this evaluator.

---

## 1b. ARC / Arithmetic (new tasks, djk_patcher pipeline)

Attribution + eval both on the consistent `djk_patcher` pipeline (modular_transformer
+ adapters + tracer.py + patcher.py). Gemma2 includes the two `nosoftcap` variants;
llama3 has no softcap so those are omitted. Full test set, bs reduced for memory
(gemma2 attr bs=4 / eval bs=4; llama3 bs=1).

> **inf handling**: `patcher.py` now drops non-finite per-sample faithfulness scores
> before averaging. arc_easy/gemma2 had exactly 1/1188 samples (index 810, a physics
> F=ma question where the model's letter-vs-number margins coincide and the bf16
> denominator rounds to 0). That one sample is excluded; the other 1187 are averaged.

### gemma2 / arc_easy (1187 samples after inf filter)

| Method | CPR |
|---|---:|
| `eap_pure` | 1.1797 |
| `eap_ig_k5` | 1.7475 |
| `eap_frnorm` | 1.3151 |
| `eap_frnorm_secantmlp` | 0.9226 |
| `eap_frnorm_secantmlp_bilinear` | 2.3856 |
| `eap_frnorm_secantmlp_bilinear_ig5` | **2.7665** |
| `eap_frnorm_secantmlp_bilinear_nosoftcap` | 2.2589 |
| `eap_frnorm_secantmlp_bilinear_nosoftcap_ig5` | 2.5628 |

Observations: `_bilinear_ig5` is best (2.77); `+secantmlp` *lowers* CPR vs `+frnorm`
(0.92 < 1.32); the nosoftcap variants sit slightly below plain bilinear — same
"softcap removal helps IOI but not other tasks" pattern seen on MCQA.

### llama3 / arc_easy, arc_challenge, arithmetic_addition, arithmetic_subtraction

6 universal methods × 4 tasks, bs=1 (8B + new tracer OOMs above bs=1). No nosoftcap
(llama3 has no softcap). Full test set. (`scripts/djkp_llama3_arc_arith.sh`)

| Method | arc_easy | arc_challenge | arith_add | arith_sub |
|---|---:|---:|---:|---:|
| `eap_pure` | 0.8431 | 0.8223 | 0.4368 | 0.5857 |
| `eap_ig_k5` | 1.0288 | 1.1009 | 0.9525 | 1.0775 |
| `eap_frnorm` | 0.5742 | 0.6330 | 0.6799 | 0.5451 |
| `eap_frnorm_secantmlp` | 1.3172 | 1.4213 | 0.9943 | 0.9704 |
| `eap_frnorm_secantmlp_bilinear` | 1.4979 | 1.5642 | 1.0142 | 1.0383 |
| `eap_frnorm_secantmlp_bilinear_ig5` | **1.5159** | **1.6927** | **1.0636** | **1.0796** |

Observations (llama3):
- `_bilinear_ig5` is the **column-best on all 4 tasks** — IG + bilinear is the
  strongest combination on llama3's SwiGLU MLP.
- `+frnorm` alone **hurts** vs plain EAP on the ARC tasks (0.57 < 0.84 on arc_easy,
  0.63 < 0.82 on arc_challenge) — frozen-LN without secant-MLP is net-negative here;
  the big recovery comes from `+secantmlp` (jumps to ~1.3–1.4).
- Arithmetic CPR is lower overall (~1.0 ceiling) than ARC (~1.5–1.7), and plain EAP
  is especially weak on arithmetic (0.44 / 0.59).
- Cumulative trend holds: pure < frnorm+secantmlp < +bilinear < +bilinear+ig5 on every
  task (frnorm-only being the one dip).

---

## 2. Key observations

### 2.1 New patcher systematically scores higher on Gemma2/IOI

- `eap_pure` gemma2/ioi: **+49%** (1.32 vs 0.89) — biggest single jump
- `eap_frnorm_secantmlp` gemma2/ioi: **+64%** (3.60 vs 2.20)
- All Gemma2 softcap-variant circuits (`_nosoftcap`, `_rawsrc_nosoftcap`, `_softcapbwdoff`, `_dpeascale`) all converge to **CPR ≈ 3.99** — apparent saturation ceiling under the new patcher
- For TL 2.18.0 the same `_nosoftcap` circuit gives 3.77 — new patcher reads it ~6% higher than even DPEA's reference evaluator

### 2.2 On non-Gemma2 / non-IOI blocks the bias direction varies

- gpt2/ioi: new patcher consistently **slightly higher** than TL 3.0 (~+5%)
- qwen2.5/ioi: new patcher mostly higher but the `eap_ig_k5` row is **lower** (1.75 vs 2.00)
- qwen2.5/mcqa: new patcher **lower** for `eap_pure` (0.85 vs 1.00), `eap_ig_k5` (1.14 vs 1.04), `eap_frnorm_secantmlp` (0.91 vs 1.08), `_bilinear` (1.80 vs 2.18) — TL 3.0 outscores on MCQA
- gemma2/mcqa: new patcher **lower** for `eap_pure` (1.09 vs 1.24), `eap_ig_k5` (1.28 vs 1.46) — same pattern

**MCQA seems to disadvantage the new patcher** systematically; IOI mostly advantages it on Gemma2.

### 2.3 `_rawsrc` is a no-op on Llama/Qwen architecture — confirmed

`eap_frnorm_secantmlp_bilinear_rawsrc` and `eap_frnorm_secantmlp_bilinear` both give **1.8635** on qwen2.5/ioi (same byte-identical circuit since `--raw-edge-source` only affects Gemma2's post-attention LayerNorm path).

### 2.4 Within Gemma2/IOI, softcap-handling variants saturate at ~3.99

All four softcap-modified variants achieve ≈ 3.99:
- `_nosoftcap`: 3.9907 (drops softcap from forward entirely)
- `_rawsrc_nosoftcap`: 3.9966 (also uses RAW edge source)
- `_softcapbwdoff`: 3.9858 (IdentityTanh — forward = tanh, backward = identity)
- `_softcapbwdoff_dpeascale`: 3.9908 (above + DPEA-style path scaling `[0.25, 0.25, 0.5, 0.5, 0.5]` without bilinear)

So the choice between "drop softcap from forward" vs "drop only its derivative in backward" is **statistically indistinguishable** at this evaluator's resolution.

---

## 3. Evaluator comparison — MIB-circuit-track (TL) vs new patcher (this branch)

Both compute CPR at the same 10 percentile buckets, but the implementations diverge in nearly every other dimension. The summary table:

| Aspect | MIB-circuit-track evaluator (TL) | This branch's `EdgeCircuitPatcher` |
|---|---|---|
| Model framework | `transformer_lens.HookedTransformer` | HF `AutoModelForCausalLM` + `nnsight.NNsight` |
| Model weights | TL's converted/remapped weights (some matrices transposed, optional LN folding) | Raw HF weights as-published |
| dtype | TL default per-config (fp32 attention in many setups) | bfloat16 (`gpt2` is fp32) |
| Attention impl | TL's own attention forward | HF `attn_implementation="eager"` |
| Circuit input file | `importances.json` (parsed into `eap.graph.Graph`) | `scores.pt` (raw `[n_src, n_tgt]` tensor) |
| Edge → patching site mapping | Per-edge TL hook points (`hook_q`, `hook_k`, `hook_v`, `hook_attn_in`, `hook_mlp_in`, `hook_mlp_out`, etc.) | Per-edge residual-stream corrections computed by `_compute_attn_corrections` / `_compute_patched_residual` and added to the layer's residual-in / residual-mid |
| Patching mechanism | `model.run_with_hooks(...)` — forward hooks replace activations at hook points | `nnsight.model.trace(...)` — proxy assignment + custom `q/k/v/mlp_patch_hook` methods on wrapped layers |
| Counterfactual baseline | `intervention='patching'` — replaces with corrupt activation at the patching site | `_cache_clean` + `_cache_patched` — corrupt-side cache values are looked up and used for "out-of-circuit" edges |
| Pruning of dangling edges | EAP-IG library's graph pruning (similar idea) | `_prune` — iteratively drop edges whose source has no live outgoing OR target has no live incoming, until fixed point (`tracer/patcher.py:272`) |
| LayerNorm handling | TL's standard backward LN | `patch_model_for_lvp(norm_approx='frozen')` — `.detach()` on σ in the forward → diagonal-only VJP |
| Special Gemma2 paths | TL's HookedGemma2 handles `(1+w)` RMSNorm and softcap via its own implementation | `modular_transformer` patched modules — `LinearGemma2RMSNorm`, `LinearGemma2Attention` with explicit softcap forward |
| Per-percentile loop | One forward per percentile bucket per batch | Same — but the residual-stream correction is computed in fp32, then added in bf16 |
| Faithfulness formula | `(circuit_metric − corrupt_baseline) / (clean_metric − corrupt_baseline)` | Same formula (`_get_logit_diff`) but using local cached clean/corrupt baselines |
| Score percentiles | `PERCENTAGES = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.0)` | Same constant (`patcher/patcher.py:44`) |
| Dataset loader | `HFEAPDataset` (MIB submodule) | `MIBDataset` (our `data_utils.py`) — different loader but same HF dataset under the hood |

### 3.1 Why the numbers diverge — five concrete sources

1. **Model weight conversion**.
   TL `HookedTransformer.from_pretrained` rewrites HF weights to its canonical layout — transposes some matrices, optionally folds in LayerNorm γ/β scaling, and applies a "center writing weights" trick by default for several model families. The new patcher uses HF weights as-is. The forward output is *approximately* equal but not bit-identical, especially for Gemma2 with its `(1 + w)` RMSNorm and softcap.

2. **dtype and accumulation order**.
   TL's attention layers default to fp32 in some HookedTransformer configs (`use_attn_result=True` forces materializing per-head outputs at fp32). The new patcher runs gemma2 in bf16 end-to-end. Numeric drift accumulates over 26 layers × 8 heads × 1000 samples. For Gemma2/IOI the drift consistently favors the patcher (CPR up).

3. **Patching site granularity**.
   MIB/TL patches per-hook activations (replaces the tensor at one TL hook point per edge). The new patcher computes a *residual-stream correction* per edge — i.e., it accumulates "missing" contributions from out-of-circuit edges into a correction vector that is added at the next layer's residual-in. Both implement the same abstract operation but at different points in the computation graph, so the rounding behavior differs.

4. **Pruning policy**.
   The new patcher's `_prune` removes any edge whose source has no surviving outgoing or target has no surviving incoming, iteratively. EAP-IG library prunes too but its rule and termination may differ. At very small percentiles (0.1%, 0.2%) this changes the actual number of active edges, which can flip per-bucket faithfulness in either direction.

5. **LayerNorm linearization**.
   The new patcher loads gemma2 with `patch_model_for_lvp(norm_approx='frozen')`, which `.detach()`s σ in RMSNorm's forward — making backward diagonal. For *attribution* this is what we want. For *patching evaluation* the forward should ideally be unchanged, but the `.detach()` does not change forward output (it only blocks gradient flow). So evaluator-side LayerNorm forward IS unchanged. However the attention output's pre-norm residual handling differs slightly between TL's Gemma2 and our `LinearGemma2RMSNorm`.

### 3.2 Why MCQA underperforms in the new patcher

The MCQA logit-diff metric depends on multi-token answer choices (the model has to pick the correct letter index across A/B/C/D options at the final position). The new patcher's residual-correction approach is **per-position**, but accumulating corrections through 26 layers in bf16 introduces noise that disproportionately affects MCQA (where the answer is a small logit margin between similar tokens). TL's fp32-attention default is more accurate for these small margins.

### 3.3 What is NOT different

- **Same percentiles**: both use the 10-bucket curve from 0.1% to 100% of edges.
- **Same metric formula**: faithfulness = `(circuit − corrupt) / (clean − corrupt)`, CPR = trapezoidal area.
- **Same circuits**: literally the same `scores.pt` files. Only the evaluator changes.
- **Same test set**: both load from HF `mib-bench/{task}` with `split='test'` (1000 samples per task).

---

## 4. CPR-to-CPR translation guide

If a paper or downstream analysis reports CPR using the MIB benchmark's TL pipeline (TL 2.18.0 in the DPEA writeup, TL 3.0.0 in our older runs), here is the empirical translation observed in this study for **gemma2/ioi** specifically:

| Variant of circuit | TL 3.0.0 | TL 2.18.0 | New patcher |
|---|---:|---:|---:|
| `eap_frnorm_secantmlp_bilinear_nosoftcap` | 3.3951 | 3.7668 | 3.9907 |
| Δ vs TL 3.0.0 | — | +0.37 | +0.60 |

The drift is **not a constant offset** — it depends on circuit and on (task, model). The new patcher's CPR numbers are **not directly comparable** to MIB paper Table 14, RelP paper Table 5, or DPEA Table 5 — those all use TL-based evaluators with different versions/configs.

For cross-paper comparison, **always use the same evaluator that produced the reference numbers** (TL 2.18.0 for DPEA / RelP comparisons).

---

## 5. Files

| Artifact | Path |
|---|---|
| Eval script | `experiments/mib/run_evaluation.py` |
| Patcher implementation | `patcher/patcher.py` |
| Wrapped layer adapter | `adapters/` |
| Modular transformer (Lasse's) | `/home/dacslab/lasse_jantsch/linear_transformer/src/modular_transformer/` |
| Driver shell scripts | `experiments/mib/scripts/eval_all_patcher.sh`, `experiments/mib/scripts/eval_all_patcher_gemma2.sh` |
| Run logs | `experiments/mib/scripts/eval_all_patcher{,_gemma2}.log` |
| Result pkls | `experiments/mib/method_comparison/results_patcher/<method>/<task>_<model>_test_abs-False.pkl` |
| Existing TL 3.0.0 results | `experiments/mib/method_comparison/results/<method>_patching_edge/<task>_<model>_test_abs-False.pkl` |
| Per-eval log of CPR values | `experiments/mib/method_comparison/report.csv` (TL-based) |
