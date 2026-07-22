# Op-level Taylor Gap Experiments

Tests the hypothesis that the gap between EAP (a first-order Taylor approximation) and the golden AP
is **the 2nd-order-and-higher term of each nonlinear op**, by checking whether our modules
(Bilinear / SM / FrLN) fill that gap.

Setup: GPT-2 small, gemma-2-2b · task = IOI · clean/corrupt = the same 100 train CF pairs used for attribution.

---

## 1. Motivation

In the earlier experiments (single-edge ε, joint-AP), **the modules did not reduce the distance (ε) to
the golden AP**. Root-cause analysis:

1. **Single-edge AP is EAP's definitional target** — the EAP score is exactly the first-order linearization
   of the single-edge AP. So using single-edge AP as the reference structurally favors EAP, and no module
   can beat it (the evaluation frame is rigged toward EAP).
2. The logit-level gap **mixes two distinct things**:
   - **Self higher-order term**: the self-nonlinearity a single edge accumulates as it propagates downstream
     (= self-repair). ~95% of the gap on top edges.
   - **Cross / curvature 2nd-order term**: appears only when two inputs of an op move together, or from a
     nonlinearity's curvature. This is what the modules target.
   Self-repair swamps the 2nd-order term the modules target, so it looked like "the gap is not filled."

→ To **isolate only the 2nd-order term** the modules target, we must measure it **at the op output where the
term is born**, **before** downstream nonlinearity enters — not at the logit. That is the purpose of op-gap.

> Core claim: "our modules correct the op's 2nd-order term that first-order EAP misses." op-gap verifies this
> directly, per op.

---

## 2. Method

### 2.1 op-gap definition

For a single nonlinear op `f` (input → output `z=f(·)`), feed the **actual activation tensors from the two real
forwards (clean, corrupt)** as the op inputs:

- True change: `Δz_true = f(corrupt input) − f(clean input)`
- Estimator `Ê`: predicts `Δz` from the input change (EAP first-order, or a module)

```
op-gap(Ê) = sqrt( Σ ‖Ê − Δz_true‖²_F  /  Σ ‖Δz_true‖²_F )
```

= the **relative RMS error** the estimator leaves on the op's clean→corrupt output change. Sums run over all
examples · layers · heads · positions · hidden-dims.

**Conventions (important):**
- op input = **real clean·corrupt activations** (everything upstream is also corrupt = full-transition; NOT a
  single-edge partial patch).
- The op is evaluated **in isolation** (its output is not re-propagated through the network) → downstream
  nonlinearity / self-repair is **excluded**. Only the op's own nonlinearity is measured.
- No edge sampling — the measured object is not the attribution graph but the **op's full activation tensor**.

### 2.1b Relationship between op-gap's `Ê` and the actual EAP score (forward/backward equivalence)

Note: the `Ê` measured here is **NOT EAP's scalar edge score.** It is the **op's clean-point linearization
(Jacobian)** that EAP's backward relies on, applied in the forward direction instead.

- **Actual EAP (backward, VJP)**: for edge `u→(input t of op f)`, the score is
  `s = (a_u^clean − a_u^corrupt) · g_t`, where `g_t = ∂m/∂t`.
  This gradient `g_t` flows backward from the metric and, when passing through op `f`, uses the **transpose of
  `f`'s clean-point Jacobian `J_f(clean)`**. → plain EAP linearizes only at clean.
- **op-gap's `Ê` (forward, JVP)**: the first-order Taylor of the op output, `Ê = J_f(clean)·Δ_input`
  (e.g. `Δq·kᵀ + q·Δkᵀ`, all expanded at clean).

→ **Both use the same `J_f(clean)`. Only the direction differs (VJP vs JVP).**
- EAP (backward): `Δa · J_f(clean)ᵀ · g_out`
- op-gap (forward): `J_f(clean) · Δa` compared to the true output change `Δz_true`

"How badly the clean linearization mismatches the op's actual change" is direction-independent, so measuring it
forward is cleaner (it compares directly to the true change, without going through the scalar score or the
metric gradient). Hence **op-gap = the error the op's clean linearization leaves = the per-op source of EAP's
higher-order error.** The module replaces `J_f(clean)` with a secant/midpoint Jacobian so that `J·Δ = Δz_true`
(op-gap → 0); since the same Jacobian is used in EAP's VJP, this is exactly the correction applied to the EAP
score (midpoint/secant = the path-average Jacobian = that op's closed-form integrated gradients).

**Caveat**: op-gap is the linearization error of the op's output tensor (full-transition, all upstream sources
at once), **not the per-edge scalar score error.** The individual edge-score error was measured directly in the
single-edge ε / joint-AP experiments (where, due to the single-edge regime mismatch, the modules did not reduce
it); op-gap is a separate view that isolates "per op, where does EAP's first-order break and does the module fix
that linearization." → op-gap ≈ 0 is a **mechanism proof**, not a guarantee of attribution fidelity.

### 2.2 Estimators per op

| op | `Δz_true` | EAP `Ê` (1st order) | module `Ê` |
|---|---|---|---|
| **Q@K** | `q*k*ᵀ − qkᵀ` | `Δq·kᵀ + q·Δkᵀ` | midpoint: `Δq·(k+k*)/2ᵀ + (q+q*)/2·Δkᵀ` |
| **A@V** | `A*V* − AV` | `ΔA·V + A·ΔV` | midpoint: `ΔA·(V+V*)/2 + (A+A*)/2·ΔV` |
| **GeGLU gate·up** | `x*y* − xy` (x=act, y=up) | `Δx·y + x·Δy` | midpoint: `Δx·(y+y*)/2 + (x+x*)/2·Δy` |
| **GELU** | `φ(g*) − φ(g)` | `φ'(g)·Δg` (tangent) | secant: `[φ(g*)−φ(g)]/Δg · Δg` |
| **Norm** (LN/RMS) | `N(x*) − N(x)` | `J_full(x)·Δx` (exact analytic Jacobian) | FrLN: `J_frozen(x)·Δx` (denominator detached) |

(`Δq=q*−q`, etc. Q@K is ×1/√d_head. A = softmax(QK(+softcap)+causal mask).)

### 2.3 Implementation

- Run two real forwards (clean, corrupt) and capture op input/output tensors via **forward hooks** (vanilla GPT-2,
  gemma2 eager/bf16).
- GPT-2: Q@K, A@V (from c_attn split), GELU (c_fc), LayerNorm. gemma2: Q@K, A@V (rotary+GQA+softcap
  reconstruction), GeGLU gate·up, gelu_tanh, RMSNorm.
- Model runs in bf16 but the gap is computed in fp32. Examples whose clean/corrupt token lengths differ are
  skipped (alignment required).
- Scripts: `experiments/mib/scripts/oplevel_cross_gap_gpt2_ioi.py` (Q@K, A@V),
  `oplevel_sm_gap_gpt2_ioi.py` (GELU), `oplevel_fn_gap_gpt2_ioi.py` (LayerNorm),
  `oplevel_gap_gemma2_ioi.py` (GeGLU·gelu·RMSNorm), `oplevel_attn_gap_gemma2_ioi.py` (gemma2 Q@K·A@V).

---

## 3. Results

### 3.1 GPT-2 small / IOI (100 examples)

| op | module | EAP op-gap | module op-gap |
|---|---|---|---|
| Q@K | Bilinear(midpoint) | 0.6591 | 1.25e-6 |
| A@V | Bilinear(midpoint) | 0.2656 | 2.20e-7 |
| GELU | SM(secant) | 0.7182 | 2.33e-8 |
| LayerNorm | FrLN | 0.2482 | **0.6555** |

### 3.2 gemma-2-2b / IOI (100 examples)

| op | module | EAP op-gap | module op-gap |
|---|---|---|---|
| Q@K | Bilinear(midpoint) | 0.8260 | 1.62e-6 |
| A@V | Bilinear(midpoint) | 0.1480 | 2.13e-7 |
| GeGLU gate·up | Bilinear(midpoint) | 0.6843 | 1.11e-7 |
| gelu_tanh | SM(secant) | 0.5947 | 0.00e+0 |
| RMSNorm | FrLN | 0.3110 | **0.1082** |

(gemma2 attention: rotary + GQA(8Q/4KV) + softcap(50) reconstruction. Q@K is on the raw matmul.)
(module op-gaps of 1e-6~1e-8 are the float32 roundoff floor — midpoint/secant are algebraically exact, so the
residual is only floating-point error.)

### 3.3 Combined (EAP op-gap → module op-gap)

| op type | module | gpt2 | gemma2 |
|---|---|---|---|
| Q@K | Bilinear | 0.6591 → 1.25e-6 | 0.8260 → 1.62e-6 |
| A@V | Bilinear | 0.2656 → 2.20e-7 | 0.1480 → 2.13e-7 |
| GeGLU gate·up | Bilinear | (no GeGLU in gpt2) | 0.6843 → 1.11e-7 |
| GELU | SM | 0.7182 → 2.33e-8 | 0.5947 → 0.00e+0 |
| Norm | FrLN | 0.2482 → 0.6555 (worse) | 0.3110 → 0.1082 (better) |

How to read: **EAP op-gap = relative size of the 2nd-order term first-order misses. module op-gap = residual
error after the module.** module op-gap ≈ 0 = the module fills that op's 2nd-order term exactly.

### 3.4 Complete noise budget — including the uncorrected nonlinearities (softmax, softcap)

To answer "why were these three locations chosen," we must also quantify the nonlinearities the modules do NOT
address. Adding softmax and softcap:

| nonlinear op | gpt2 EAP op-gap | gemma2 EAP op-gap | corrector | after correction |
|---|---|---|---|---|
| **softmax** | **1.4592** | **1.1198** | IG (Z-step) | Z=5: 0.0106 / 0.0074 |
| Q@K (bilinear) | 0.6591 | 0.8260 | midpoint (1×) | ~1e-6 |
| GELU (MLP act) | 0.7182 | 0.5947 | secant (1×) | ~1e-7 |
| GeGLU gate·up (bilinear) | — | 0.6843 | midpoint (1×) | ~1e-7 |
| A@V (bilinear) | 0.2656 | 0.1480 | midpoint (1×) | ~1e-7 |
| Norm (LN/RMS) | 0.2482 | 0.3110 | freeze | LN 0.656 / RMS 0.108 |
| softcap (gemma tanh) | — | 0.0057 | secant (1×) | 7.2e-10 |

**softmax IG convergence** (gpt2 / gemma2):

| | EAP tangent | IG Z=1 | Z=2 | Z=5 | Z=10 | Z=20 |
|---|---|---|---|---|---|---|
| gpt2 | 1.4592 | 0.4080 | 0.1232 | 0.0106 | 0.0022 | 0.0005 |
| gemma2 | 1.1198 | 0.3757 | 0.0829 | 0.0074 | 0.0017 | 0.0004 |

**Key — the noise sources split into two classes:**
- **closed-form exact (1× cost)**: bilinear (Q@K·A@V·GeGLU) = midpoint, MLP-act (GELU) = secant, softcap =
  secant → the modules remove these exactly (≈1e-7) in a single step. **This is why these locations were chosen:
  they admit an exact closed-form correction at 1× cost.**
- **integration-only (Z× cost)**: softmax = the **largest source (1.1~1.5) but has no closed form** (multi-D
  coupled) → requires Z-step IG (1.46→0.011 @Z=5). = handled by `--integration-steps` (the eap_ig_5 variant).
- **norm**: freeze, architecture-dependent.
- **negligible**: softcap (0.006) — nearly linear, effectively ignorable.

→ The strongest CPR method `eap_ig_5_igbilin_frnorm_secmlp` is not a coincidence: it **covers the four
noise-source classes with the right tool for each**: ig_5→softmax, igbilin→bilinear, secmlp→GELU, frnorm→norm.

---

## 4. Analysis

### 4.1 Bilinear · SM — fill the 2nd-order term exactly, architecture-independent (robust)

- First-order EAP misses **15~83% of the output change** at each op (gpt2 Q@K 0.66 / A@V 0.27 / GELU 0.72;
  gemma2 Q@K 0.83 / A@V 0.15 / GeGLU 0.68 / gelu 0.59).
- Bilinear(midpoint) and SM(secant) fill that gap **down to the float32 roundoff floor (1e-6~1e-8) = effectively
  exact.**
  - secant: 1-D element-wise, so `[φ(g*)−φ(g)]/Δg·Δg = Δz_true` identically → 0.
  - midpoint: bilinear, so it **algebraically equals** `Δz_true` exactly.
- Holds across 5 op types × 2 models → **the claim "the modules correct EAP's 2nd-order term" is confirmed.**
- In particular gemma2 **Q@K (0.826→1.6e-6)** and **GeGLU gate·up (0.684→1.1e-7)**: EAP misses 83% and 68%
  respectively, and midpoint captures it fully. GeGLU is an op GPT-2 does not have — it sits exactly where
  gemma2's large bilinear CPR gain (6→22) comes from, and midpoint captures the biggest 2nd-order term there →
  consistent with the CPR gain.

### 4.2 FrLN — architecture-dependent, partial

- **GPT-2 LayerNorm**: worse (0.248 → 0.656). FrLN detaches a real term (the variance/denominator coupling) of
  the LN gradient, and for LayerNorm that widens the gap → it is a ranking heuristic, not a Taylor gap-filler.
- **gemma2 RMSNorm**: better (0.311 → 0.108). The frozen denominator is closer to the true finite change than the
  tangent (J_full). But not ~0 (the 0.108 residual is RMSNorm higher-order curvature; FrLN is not an exact
  corrector).
- Reason: LayerNorm/RMSNorm are neither element-wise nor bilinear but multi-D normalizations → an exact
  corrector like secant/midpoint **does not exist in principle.** FrLN does not close the gap; it detaches the
  denominator term, a different correction whose effect depends on the norm type.

### 4.3 Overall interpretation

- At the **op level (full-transition)**, Bilinear·SM close the 2nd-order term exactly, and FrLN closes it
  partially on RMSNorm.
- On **modern RMSNorm architectures (gemma/qwen/llama), all three modules reduce their op-gap** → consistent
  with why gemma2's CPR gain is much larger than GPT-2's (an extra GeGLU bilinear op + FrLN working on RMSNorm).

### 4.4b Why these noise sources were chosen (from the §3.4 budget)

- **softmax is the largest source** (1.1~1.5, exceeding 1.0 = error larger than the true change, extreme
  saturation), but it is neither element-wise nor bilinear, so **it has no closed-form exact corrector.** → it is
  correctable only via Z-step IG (1.46→0.011 @Z=5), i.e. at Z× cost.
- The bilinear·MLP-act·softcap ops the three modules target admit **exact correction at 1× cost** (midpoint/secant
  are algebraically exact). → **the answer to "why these three": size (top-ranked) + closed-form tractability.**
  softmax is handled complementarily by IG (integration-steps).
- The full method combination is explained by the budget: ig_5(softmax) + igbilin(bilinear) + secmlp(GELU) +
  frnorm(norm).

### 4.4 Limitations (stated honestly)

- op-gap **isolates a single op's nonlinearity**. The additional distortion the op's output receives as it passes
  through downstream nonlinearity (self-repair) is **excluded**. So **op-gap ≈ 0 does NOT guarantee final
  logit-level / single-edge AP fidelity** (in the earlier experiments the modules did not reduce the logit-level
  gap, and a ~95% self-repair error dominates on top edges).
- i.e. the modules "model each op's 2nd-order term exactly but do not handle self-repair." op-gap = 0 is a
  mechanism check, not an attribution-fidelity guarantee.
- op-level full-transition ≈ the circuit-ablation (CPR) regime → the modules' real utility is verified by CPR
  (model behavior).

---

## 5. Files

| script | measures |
|---|---|
| `scripts/oplevel_cross_gap_gpt2_ioi.py` | GPT-2 Q@K, A@V |
| `scripts/oplevel_sm_gap_gpt2_ioi.py` | GPT-2 GELU curvature |
| `scripts/oplevel_fn_gap_gpt2_ioi.py` | GPT-2 LayerNorm |
| `scripts/oplevel_gap_gemma2_ioi.py` | gemma2 GeGLU·gelu·RMSNorm |
| `scripts/oplevel_attn_gap_gemma2_ioi.py` | gemma2 Q@K·A@V (rotary+GQA+softcap) |
| `scripts/oplevel_softmax_gpt2_ioi.py` | GPT-2 softmax (EAP tangent + IG Z=1..20) |
| `scripts/oplevel_softmax_gemma2_ioi.py` | gemma2 softmax + softcap |

(Note: single-edge ε / joint-AP / SM-verify are separate experiments — edge-sampling based, distinct from op-gap.)
