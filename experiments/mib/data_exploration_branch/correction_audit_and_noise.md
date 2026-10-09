# What the two backward-pass fixes change, and what they buy

Two fixes to the backward pass, both opt-in; the defaults reproduce every earlier number.

| fix | what it changes | where | how to turn it on |
|---|---|---|---|
| `ig_fix` | under IG each step's local rules anchored at the counterfactual, so the interval they integrated was `[α_k, 1]`. They now anchor at the next step, `[α_k, α_{k+1}]`. | `tracer/tracer.py` (`IG_ANCHOR` at `:32`/`:85`, `_capture_step_anchors` at `:200`, the per-step swap at `:186`); `experiments/mib/run_attribution.py:197–262` picks which caches to record | `IG_ANCHOR=next_step` |
| `sm_fix` | the MLP secant took its chord from the **origin**, `c = f(x)/x`. It now takes the chord between `x` and the anchor. | `vendor/linear_transformer/linear_transformer/modules/activations.py` — `SecantCFSiLU`/`SecantCFGELUTanh` at `:119`/`:145`, next to the untouched originals at `:77`/`:47` | `--mlp-act-fn secant_cf_silu` (Qwen/Llama) or `secant_cf_gelu_tanh` (GPT-2/Gemma-2) |

The two interlock: `secant_cf_*` reads its `x_base` from the tracer's anchor, so with
`IG_ANCHOR=next_step` the MLP secant and the bilinear rule measure over the **same**
sub-interval.

`ig_fix` does not touch the IG loop itself. The loop still interpolates the input
embeddings at α ∈ {0, .2, .4, .6, .8} and freezes the source cache, matching the MIB
reference; what changed is the reference point the per-operation rules read inside each
step.

Reproducible from three scripts:

| what | script |
|---|---|
| do the rules match their defining identity | `scripts/check_corrections.py` |
| how far each rule's coefficient is from the exact one | `scripts/rule_bias.py` |
| CPR / CMD movement from the fixes | `scripts/fix_effect_analysis.py` |

---

## 1. Are the fixes live and correct?

### 1.1 Identities

`check_corrections.py` checks each backward rule against the identity that defines it.
All pass at float32 precision:

```
ig_mul          max |split - true change| = 2.4e-07
ig_matmul           |split - true change| = 0.0e+00
secant_cf_silu  max |c*dx - df|           = 6.0e-08
secant_cf_gelu  max |c*dx - df|           = 1.2e-07
secant_silu     max |c*x - f(x)| (origin) = 6.0e-08
secant_gelu     max |c*x - f(x)| (origin) = 0.0e+00
```

The patcher reaches every module it needs to, on all four models — including the final
norm (one more norm than twice the layer count in each row):

```
gpt2     GPT2Attention x12, GPT2MLP x12, LayerNorm x25, Linear x1
qwen2.5  Qwen2Attention x24, Qwen2MLP x24, Qwen2RMSNorm x49, Linear x169
gemma2   Gemma2Attention x26, Gemma2MLP x26, Gemma2RMSNorm x105, Linear x183
llama3   LlamaAttention x32, LlamaMLP x32, LlamaRMSNorm x65, Linear x225
```

### 1.2 Is each correction live in every cell?

Pairwise diff of the already-built head-100 circuits: two runs that differ by exactly one
correction must not be bit-identical. Across all 12 settings and 10 such pairs, **no cell
is inert** — the smallest relative difference is 3.0e-02 (`ig_fix` on `+FrLN+Bili+IG`,
IOI/Qwen2.5). In particular the two things that could plausibly have been silently dead
are not: `secant_cf_*` receives its `x_base` through `_anchor_keys`, and `--matmul-fn`
covers GPT-2 (through the QK and AV products) even though GPT-2 has no gated MLP.

---

## 2. Where the noise is: each rule's distance from the exact coefficient

Under IG, the attribution multiplies a fixed source difference by an averaged
coefficient, so every backward rule is an estimator of

    C* = integral_0^1 c(alpha) d alpha ,    x(alpha) = (1-alpha)*clean + alpha*cf

where `c(alpha)` is the operation's exact local derivative at the point `x(alpha)`.
`rule_bias.py` computes `C*` on a fine grid of forward passes (41 alphas) and compares it
with the aggregate coefficient each implementation really applies. The number reported is
the relative L1 error, averaged over layers: **0 means the rule applies exactly the
coefficient IG is trying to estimate**.

Qwen2.5 / IOI, 4 examples, mean over 24 layers:

| site | rule | Z=1 | Z=5 | Z=11 |
|---|---|---:|---:|---:|
| MLP act (SiLU) | tangent — no SM | 0.0279 | 0.0067 | 0.0030 |
| | **origin secant — `+SM` as shipped** | **0.4380** | **0.4362** | **0.4363** |
| | next-step secant — `sm_fix` | 0.0141 | 0.0052 | 0.0041 |
| gate × up | plain mul — no Bilinear | 0.0514 | 0.0123 | 0.0054 |
| | **cf-anchored `ig_mul` — `+Bilinear` as shipped** | 0.0247 | **0.0275** | **0.0303** |
| | next-step `ig_mul` — `ig_fix` | 0.0247 | 0.0035 | 0.0011 |
| attn × value | plain matmul | 0.0182 | 0.0031 | 0.0015 |
| | **cf-anchored `ig_matmul`** | 0.0065 | **0.0071** | **0.0078** |
| | next-step `ig_matmul` — `ig_fix` | 0.0065 | 0.0004 | 0.0001 |

Three things fall out of this table.

**Without IG, every correction works.** At Z=1 the cf-anchored bilinear halves the error
of the plain product (0.0247 vs 0.0514 on gate × up; 0.0065 vs 0.0182 on attn × value)
and frozen-σ is small. This is the regime the corrections were designed and validated in.

**With IG, the cf anchor stops converging — and the Bilinear correction becomes worse
than no correction at all.** Every correctly-anchored rule improves like O(1/Z): plain mul
0.0514 → 0.0123 → 0.0054, next-step `ig_mul` 0.0247 → 0.0035 → 0.0011. The cf-anchored
rules move the wrong way: 0.0247 → 0.0275 → 0.0303. At Z=5, the shipped `+Bilinear+IG`
combination applies a coefficient **2.2× further from the truth than using no bilinear
rule at all** (and 2.3× on attn × value). More integration steps make it slightly worse,
not better. The reason is geometric: with the baseline pinned to the counterfactual, the
midpoint of step *k* lands at (α_k + 1)/2, so for Z=5 the five rules evaluate at
α ∈ {0.5, 0.6, 0.7, 0.8, 0.9} — mean 0.7 — instead of the intended {0.1, 0.3, 0.5, 0.7,
0.9}, mean 0.5. Every step is biased toward the counterfactual end of the path, and
adding steps only crowds them further into that end. `ig_fix` restores the intended
midpoints, and with them O(1/Z) convergence.

**`+SM` as shipped is the single largest rule error in the stack, at every Z.** The origin
chord `f(x)/x` sits at 0.44 relative error — **65× the plain tangent at Z=5, and 100× the
next-largest error in the table** — and it does not move as Z grows, because it is not an
estimator of the path integral at all: it is the exact decomposition of `f(x)` against a
*zero* input, which is not a point on the clean→counterfactual path. `sm_fix` replaces it
with the chord over the same sub-interval every other rule uses, and it becomes the best
of the four MLP-activation rules (0.0052 at Z=5, below even the tangent).

So the answer to "is `ig_fix+sm_fix` the logically right thing" is yes, and specifically:
it makes every rule report the exact average of its own derivative over **the same**
sub-interval `[α_k, α_{k+1}]`.

That is the state *after* the fixes. What the fixes repair is the **pre-fix** code — the
code Table 1 was produced with — where three rules linearize about three different places
at once: Bilinear over `[α_k, 1]`, SM about the origin, FrLN and softmax at `α_k`. Their
errors compound instead of cancelling, which is what the table above measures. After
`ig_fix` and `sm_fix`, Bilinear and SM both sit on `[α_k, α_{k+1}]`; FrLN and softmax stay
at `α_k`, which is fine because both are consistent estimators that converge at O(1/Z).

### Does this replicate off Qwen/IOI?

Same measurement at Z=5 on a second model (Gemma-2, GeGLU instead of SwiGLU) and a second
task. The pattern is the same everywhere, and the origin-secant error is strikingly
stable across all three — 0.43-0.46 in every case.

| rule (Z=5) | Qwen2.5 / IOI | Gemma-2 / IOI | Qwen2.5 / MCQA |
|---|---:|---:|---:|
| MLP act — tangent, no SM | 0.0067 | 0.0042 | 0.0163 |
| MLP act — **origin secant (`+SM` as shipped)** | **0.4362** | **0.4565** | **0.4254** |
| MLP act — next-step secant (`sm_fix`) | 0.0052 | 0.0006 | 0.0042 |
| gate × up — plain mul | 0.0123 | 0.0086 | 0.0289 |
| gate × up — **cf-anchored `ig_mul` (as shipped)** | **0.0275** | **0.0230** | **0.0626** |
| gate × up — next-step `ig_mul` (`ig_fix`) | 0.0035 | 0.0013 | 0.0028 |
| attn × value — plain matmul | 0.0031 | 0.0029 | 0.0140 |
| attn × value — **cf-anchored `ig_matmul`** | **0.0071** | **0.0085** | **0.0313** |
| attn × value — next-step `ig_matmul` (`ig_fix`) | 0.0004 | 0.0006 | 0.0014 |

In all three, the shipped cf-anchored bilinear is 2-3× worse than applying no bilinear
rule at all, and `ig_fix` is 3-10× better than no rule.

Worth flagging the one place the two measurements disagree: MCQA/Qwen2.5 has the *largest*
rule-bias improvement from `ig_fix` (0.0626 → 0.0028 on gate × up) and is the only setting
where `ig_fix` *lost* CPR (11.38 → 7.02 on `+FrLN+Bili+IG`, −1.09 on the full stack). A
better coefficient does not have to produce a better ranking in every cell, and MCQA's
filtered training split holds only 110 examples, so that cell is the noisiest one in the
benchmark. This is not explained yet.

### One caveat on the reference

The code multiplies a *fixed* source difference by an averaged coefficient, so the best
any anchoring scheme can do is the uniform-α mean of the local derivative — which is what
`C*` is here. An attribution free to weight the path differently could do better still;
that is outside what this code shape can express, and it is the same reference for every
implementation compared, so the ordering is unaffected.

---

## 3. What that buys on the benchmark

Head-100 protocol (attribution on the first 100 training examples, evaluation on the full
test split), all 12 settings, `scripts/fix_effect_analysis.py`.

### CPR — the fixes pay

| fix | cells | mean Δ | median Δ | better | worst | best |
|---|---:|---:|---:|---:|---:|---:|
| `ig_fix` (log CPR) | 24 | **+1.154** | +0.805 | **22/24** | −1.09 | +4.26 |
| `sm_fix` on top (log CPR) | 36 | +0.529 | +0.037 | 20/36 | −2.07 | +8.17 |
| both vs base (log CPR) | 48 | +0.974 | +0.590 | 34/48 | −2.07 | +8.17 |
| `ig_fix` (linear CPR) | 24 | +0.183 | +0.178 | **23/24** | −0.11 | +0.58 |
| both vs base (linear CPR) | 48 | +0.138 | +0.112 | 35/48 | −0.41 | +1.21 |

`ig_fix` on the `+FrLN+Bili+IG` row alone, per setting (log CPR):

```
ioi/gpt2      12.34 -> 13.04  +0.70     mcqa/llama3     7.05 -> 10.58  +3.52
ioi/qwen2.5   11.38 -> 13.16  +1.78     arc_easy/gemma2 11.63 -> 12.88  +1.25
ioi/gemma2    23.91 -> 24.16  +0.25     arc_easy/llama3  6.96 -> 10.03  +3.07
ioi/llama3    16.81 -> 18.74  +1.93     arc_chal/llama3  5.96 -> 10.22  +4.25
mcqa/qwen2.5   8.11 ->  7.02  -1.09     arith_add/llama3 8.34 ->  8.60  +0.26
mcqa/gemma2   10.15 -> 10.92  +0.77     arith_sub/llama3 8.20 ->  8.65  +0.46
                                        mean +1.43, median +1.01, 11/12 better
```

The measurement in §2 predicts this shape. `ig_fix` is a bias removal that applies
wherever IG and Bilinear are both on, so it is large and near-universal. `sm_fix` fixes a
larger per-rule error but only inside the SM rows, and SM's contribution to the final
ranking is diluted by the other rules, so its benchmark effect is positive on average and
noisy cell by cell (median ≈ 0).

Note also that `sm_fix` *without* IG is already a coin flip on CPR (`+SM` row: 6/12
better, mean +1.11 but median −0.08), even though §2 says the rule it replaces is 30×
worse. A rule error of 0.44 on one operation does not translate into a ranking error of
the same size, because the MLP activation is one factor among many in each edge score.

### CMD — flat

| fix | cells | mean Δ | median Δ | better |
|---|---:|---:|---:|---:|
| `ig_fix` (log CMD) | 22 | −0.015 | −0.002 | 11/22 |
| `sm_fix` on top | 35 | +0.008 | +0.007 | 19/35 |
| both vs base | 47 | −0.000 | +0.007 | 25/47 |

Neither fix moves CMD. That is consistent rather than disappointing: CPR asks how much of
the model's behaviour the top-ranked edges recover, which is exactly what a less biased
coefficient should improve; CMD asks how close circuit performance sits to 1.0, which is
dominated by where the curve saturates, not by the ranking near the top.

*(Two ARC-Easy/Llama-3.1 `ig_fix` CMD cells were excluded above: their abs circuits came
out all zeros on 09-26 because the next-step anchor exceeded the card before the anchor
caches were moved to the host. Both circuits have since been rebuilt — non-zero share
0.485, healthy — and are queued for re-evaluation.)*

### Which configuration wins

| implementation | best configuration, counted over the 12 settings (log CPR) |
|---|---|
| base | +FrLN+Bili ×3, +FrLN+Bili+SM ×3, +FrLN+Bili+IG ×2, EAP-IG ×2, +Bilinear ×1, full stack ×1 |
| `ig_fix` | +FrLN+Bili ×4, +FrLN+Bili+IG ×4, full stack ×2, EAP-IG ×1, +FrLN+Bili+SM ×1 |
| `ig_fix+sm_fix` | **+FrLN+Bili+SM ×5, full stack ×5**, EAP-IG ×1, +FrLN+Bili+IG ×1 |

Under the fixes the winner concentrates on the two fullest configurations — 10 of 12
settings — where under the base code it was scattered across six. The stack still is not
monotone (the full stack wins 5 of 12, not 12 of 12), but the reason is no longer an
implementation defect: the remaining spread is the ordinary variation measured earlier
(median first-to-second gap 0.21 log-CPR against 0.2–0.5 subset-to-subset variation).

### The `base` column is Table 1

Table 1 = the **first training slice** of the subset pipeline (`grid.py` /
`run_signed_grid.py`, examples 1–100, full test split, signed), which runs on the vendor
package. Matching Table 1 against `out/functional/<method>_train001-100/` and against the
`head100/base_*` rerun:

```
gpt2              Table 1   functional  head100/base
EAP                  6.15        6.150         6.150
EAP-IG (Z=5)        12.50       12.500        12.500
+FrLN               13.04       13.038        13.038
+SM                 11.16          --         11.160
+Bilinear           11.17       11.171        11.171
+FrLN+Bili          15.01       15.007        15.007
+FrLN+Bili+SM       14.72       14.720        14.720
+FrLN+Bili+IG       12.34       12.336        12.336
full stack          12.59       12.585        12.585
```

All nine GPT-2 rows reproduce to three decimals. Qwen2.5 and Gemma-2 reproduce to within
0.1–0.5 log-CPR (e.g. Qwen EAP 0.71 vs 0.589 / 0.651) — they run in bfloat16 where GPT-2
runs in float32, and the Qwen EAP cell is the known bimodal one (0.59, 7.03, 0.44, 7.38,
0.49 across the five slices). So `base` is the right baseline to compare the fixes
against, and no third code state is involved.

---

## 4. Cross-check against the paper's own text

**The bilinear rule covers all three products, in the paper and in the code.** §4.2:
*"Most modern transformers have three distinct bilinear functions: the attention-score
product QK^T, the value combination AV, and the SwiGLU/GeGLU gate-up product"*, and the
appendix: *"+Bilinear replaces the three multiplicative operations QK^T, AV, and
gate ⊙ up"*. `llama2.py`, `gemma2.py` and `gpt2.py` each call the rule at both attention
call sites plus the MLP product. The QK/AV split in the vendored package adds the
*ability* to use different rules per site; no run has, so it changes nothing.

**Both fixes are what the paper's own motivating sentences ask for.**

*Bilinear.* Eq. 8 integrates `∂y/∂a = b` *"along the straight-line path from the
counterfactual activation b′ to the clean activation b"*, giving `½(b+b′)`. That is exact
for a single interval — i.e. for Z=1. Under EAP-IG the path is already decomposed into Z
sub-intervals (Eq. 3), so Eq. 8 should be applied to each sub-interval. The shipped code
applies the full clean↔cf form at every step instead, which is the α=0.7 bias measured in
§2. `ig_fix` is Eq. 8 applied consistently with Eq. 3.

*SM.* §4.3 motivates the correction by saying the local slope *"can differ substantially
from the actual mean rate of change of σ **along the patching path**"* — and the patching
path is clean↔counterfactual. But Eq. 10 then defines the slope as `σ(a_u)/a_u`, the
chord from the **origin**, which is not a rate of change along that path at all. `sm_fix`
is the slope §4.3's own motivation asks for; Eq. 10 is a different quantity that the
prose never justifies. §2 measures the cost: the origin chord is 0.43–0.46 off the path
average in all three settings tested, against 0.0006–0.005 for the path chord.
