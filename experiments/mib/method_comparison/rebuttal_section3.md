# Response to Reviewer — Section 3

We agree and have restructured Section 3 to open with a **noise budget** that quantifies each nonlinear op's
contribution to EAP's error *before* proposing any fix, and derives the correction sites from it.

**What / why / how (new experiment, §3.1–3.2).**
- *Why:* to answer the section title, we must localize where EAP's first-order approximation actually breaks and
  measure each source before proposing a solution.
- *What:* we define a per-operation diagnostic, the **op-gap** — the relative error of EAP's first-order
  (clean-point) linearization of a single nonlinear op's clean→corrupt output change.
- *How:* for each nonlinear op `f`, we run the real clean and corrupt forwards, capture `f`'s input/output
  tensors, and compute
  `op-gap = ‖J_f(x)·Δx − (f(x*)−f(x))‖ / ‖f(x*)−f(x)‖`,
  where `J_f(x)` is exactly the clean-point Jacobian EAP back-propagates through `f` (so op-gap = per-op share
  of EAP's noise; forward/backward equivalence in App. X). Averaged over GPT-2 small and Gemma-2-2B, 100 IOI
  counterfactual pairs. No edge sampling — the whole activation tensor is used.

**Table 3.1 (new) — EAP op-gap per nonlinear op (before any correction).**

| op | GPT-2 | Gemma-2 | corrector |
|---|---|---|---|
| softmax | **1.46** | **1.12** | IG (Z-step) |
| Q@K / A@V / GeGLU (bilinear) | 0.27–0.66 | 0.15–0.83 | midpoint (1×) |
| MLP act (GELU) | 0.72 | 0.59 | secant (1×) |
| Norm (LN/RMS) | 0.25 | 0.31 | freeze |
| logit softcap | — | 0.006 | (negligible) |

**Why these sites (not others).** The budget shows the noise concentrates in a few ops — linear operations
(projections, rotary, residual) contribute none, and the softcap is negligible (0.006). It further shows a
nonlinear op admits an *exact single-pass (1×) correction* iff it is element-wise or bilinear (the secant slope
reproduces an element-wise change exactly, the midpoint rule a bilinear one — both are the closed-form
path-average Jacobian). Our three modules target exactly this class, driving those op-gaps to ~1e-7 (§3.4).
**softmax** is the largest source but is neither element-wise nor bilinear (its Jacobian couples all keys), so
it has no closed form and is corrected only by Z-step path integration — op-gap 1.46 → 0.011 at Z=5 (Fig 3.1b);
this is the role of the IG (`integration-steps`) component. The full method is thus a per-source decomposition:
IG = softmax, Bilinear = bilinear ops, SM = GELU, FrLN = norm — which also explains the Table 1 ablations.

**Scope.** op-gap isolates one op (excludes downstream self-repair), so it proves each module cancels its target
term, not end-to-end fidelity; the latter is measured by circuit faithfulness (CPR, §X).

**Manuscript changes:** new §3.1 (op-gap def.), §3.2 + Table 3.1 (budget, before any fix), §3.3 (closed-form vs
integration argument), Fig 3.1 (per-op bars + softmax IG curve); §3.4 reports op-gap after each module.
