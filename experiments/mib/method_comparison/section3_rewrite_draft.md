# Section 3 Rewrite Draft — "Where Does the Approximation Noise Come From?"

Addresses reviewer comment: (1) Ch.3 starts abruptly with no motivation, (2) no justification for choosing
these three nonlinearity locations over others, (3) no direct quantification of each noise source before the
solution, (4) the title promises a noise-origin analysis.

Fix: lead with a **noise budget** that quantifies every nonlinear op's first-order approximation error
(op-gap), split the sources into correctability classes, and derive the module choices from that budget.

---

## 3. Where Does the Approximation Noise Come From?

### 3.1 Setup — measuring per-op approximation noise (op-gap)

EAP replaces the true activation-patching effect with a first-order Taylor term evaluated at the clean run.
Its error therefore accumulates at every nonlinear operation `f` that lies between an edge and the metric.
To localize this error we measure, for each nonlinear op in isolation, how badly its first-order (clean-point)
linearization predicts the op's true clean→corrupt output change:

> Given the real clean and corrupt activations as op inputs, let `Δz_true = f(x*) − f(x)` and let
> `Ê = J_f(x)·Δx` be the first-order (EAP) estimate. We report the relative RMS error
> `op-gap = ‖Ê − Δz_true‖ / ‖Δz_true‖`, summed over all examples, layers, heads, positions and hidden units.

This is direction-independent: `J_f(x)` is exactly the clean-point Jacobian that EAP's backward propagates
through `f`, so `op-gap` is the per-op contribution to EAP's total approximation noise. We evaluate on GPT-2
small and Gemma-2-2B over 100 IOI counterfactual pairs. (Full definition and forward/backward equivalence in
App. X.)

### 3.2 The noise budget — where the error actually lives

Table 3.1 quantifies `op-gap` for every nonlinear op, *before any correction*. Three observations drive the
rest of the section:

**Table 3.1** First-order approximation noise (EAP op-gap) per nonlinear op.

| nonlinear op | GPT-2 | Gemma-2-2B |
|---|---|---|
| softmax (attention) | **1.46** | **1.12** |
| Q@K (bilinear) | 0.66 | 0.83 |
| MLP activation (GELU) | 0.72 | 0.59 |
| GeGLU gate·up (bilinear) | — | 0.68 |
| A@V (bilinear) | 0.27 | 0.15 |
| LayerNorm / RMSNorm | 0.25 | 0.31 |
| logit softcap (tanh) | — | 0.006 |

1. **The noise is dominated by a few ops, not spread uniformly.** softmax leaves a relative error above 1.0
   (its first-order estimate is worse than doing nothing), the bilinear ops and the MLP activation leave
   0.15–0.83, LayerNorm/RMSNorm 0.25–0.31, and the logit softcap is negligible (0.006). The linear operations
   (projections, rotary, residual, scaling) contribute no first-order error and need no treatment.
2. **Attention softmax is the single largest source.** This matches its strong saturation: near a peaked
   attention pattern the tangent overshoots badly.
3. **The sources differ not only in size but in *correctability*** (Sec. 3.3), and this — not size alone —
   determines which admit a cheap fix.

### 3.3 Two classes of noise source ⇒ two kinds of correction

The key structural fact is that a nonlinear op admits an *exact, single-pass* linear correction iff it is
element-wise or bilinear. This partitions Table 3.1:

**(a) Closed-form-exact sources (1× cost).**
For an element-wise op `φ`, the secant slope reproduces the finite change exactly:
`[φ(g*)−φ(g)]/(g*−g) · (g*−g) = φ(g*)−φ(g)`. For a bilinear op `z=x·y`, the midpoint rule does:
`Δx·(y+y*)/2 + (x+x*)/2·Δy = x*y* − xy`. Both are the path-average Jacobian in closed form (per-op integrated
gradients) and cost a single extra activation. This class covers the bilinear ops (Q@K, A@V, GeGLU gate·up),
the MLP activation, and the softcap. Replacing the clean-point Jacobian with the secant/midpoint one drives
their op-gap from 0.15–0.83 down to floating-point zero (~1e-7; Table 3.2).

**(b) Integration-only source (Z× cost).**
softmax is neither element-wise nor bilinear — its Jacobian couples all keys — so no closed-form secant exists.
It is still correctable, but only by integrating the Jacobian along the clean→corrupt path with `Z` steps,
which costs `Z×` forward evaluations. Fig. 3.1 shows the convergence: op-gap 1.46 → 0.41 (Z=1) → 0.011 (Z=5) →
0.0005 (Z=20) on GPT-2 (1.12 → 0.007 @Z=5 on Gemma-2). This is exactly the role of the IG variant
(`integration-steps`).

**(c) Normalization.**
LayerNorm/RMSNorm is a multi-dimensional normalization with no exact secant either. Freezing the normalizer
denominator (FrLN) is a cheap surrogate whose effect is architecture-dependent: it *widens* the LayerNorm gap
(0.25→0.66, GPT-2) but *reduces* the RMSNorm gap (0.31→0.11, Gemma-2). We adopt it for the RMSNorm family that
modern models use, and discuss the LayerNorm case in Sec. X.

### 3.4 The correction modules

Each module targets one class above with the matching tool:

**Table 3.2** op-gap before → after each module.

| op | module | GPT-2 | Gemma-2-2B |
|---|---|---|---|
| Q@K / A@V / GeGLU | **Bilinear** (midpoint) | 0.27–0.66 → ~1e-6 | 0.15–0.83 → ~1e-6 |
| MLP activation | **SM** (secant) | 0.72 → ~1e-7 | 0.59 → ~1e-7 |
| softmax | **IG** (Z-step) | 1.46 → 0.011 (Z=5) | 1.12 → 0.007 (Z=5) |
| LayerNorm / RMSNorm | **FrLN** (freeze) | 0.25 → 0.66 | 0.31 → 0.11 |

Thus the full method is a per-source decomposition: **IG for softmax, Bilinear for the bilinear ops, SM for the
MLP activation, FrLN for the normalizer.** This directly explains our best configuration,
`IG(Z=5)+Bilinear+SM+FrLN`, and the ablations in Table 1 — each component removes one identified noise source.

> Scope note: op-gap isolates a single op and excludes the downstream nonlinearity (self-repair) that a
> correction must still pass through; it is a proof that each module cancels its target term, not a claim about
> end-to-end fidelity. The downstream effect is measured separately by circuit faithfulness (CPR, Sec. X), where
> softmax's large residual is the dominant untreated term.

---

### Notes for integration into the paper

- **Fig 3.1**: bar chart of Table 3.1 (EAP op-gap per op, GPT-2 + Gemma-2 grouped), softmax bar visibly tallest,
  with a horizontal line at op-gap=1. Inset: softmax IG-convergence curve (op-gap vs Z, log-y).
- **Table 3.2**: before→after per module (the "we fix exactly what we target" table).
- The scope note pre-empts the "but does op-gap translate to fidelity?" follow-up by pointing to CPR.
- If space is tight, Table 3.1 + one sentence per class (3.3) + Table 3.2 is the minimal version that still
  answers all four reviewer points.
- Numbers are from `experiments/mib/method_comparison/opgap_report.md` §3 (100 IOI CF pairs).
