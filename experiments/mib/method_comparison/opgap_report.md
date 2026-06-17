# Op-level Taylor Gap 실험 정리

EAP(1차 테일러 근사)와 golden AP 사이의 갭 = **각 비선형 op 의 2차항 이상**이라는 가설을, 우리 모듈
(Bilinear / SM / FrLN)이 그 갭을 메우는지로 검증한 실험.

대상: GPT-2 small, gemma-2-2b · task = IOI · clean/corrupt = attribution 과 동일한 train CF 쌍 100개.

---

## 1. 왜 했나 (motivation)

직전 실험(single-edge ε, joint-AP)에서 **모듈이 golden AP 와의 거리(ε)를 못 줄였다**. 원인 분석:

1. **single-edge AP 는 EAP 의 정의상 정답**이다 — EAP score = single-edge AP 의 1차 선형화. 그러니
   single-edge 를 reference 로 쓰면 EAP 가 구조적으로 1등이고, 어떤 모듈도 못 이긴다 (검증틀이 EAP 에 유리).
2. logit-level 에서 잰 갭은 **두 종류가 섞여 있다**:
   - **self 고차항** (single edge 가 아래로 전파되며 겪는 자기 비선형 = self-repair). top edge 에서 갭의 ~95%.
   - **cross/곡률 2차항** (op 에서 두 입력이 동시에 움직일 때 / 비선형 곡률). 모듈이 겨누는 것.
   self-repair 가 모듈이 겨누는 2차항을 덮어버려 "갭 못 메움"으로 나왔다.

→ 모듈이 겨누는 **2차항만 isolate** 하려면, logit 이 아니라 **그 항이 태어나는 op 출력**에서, downstream
비선형이 들어오기 **전에** 재야 한다. 그게 op-gap 실험의 목적이다.

> 핵심 주장: "우리 모듈은 EAP 1차가 놓치는 op 의 2차항을 보정한다." op-gap 은 이 주장을 op 단위로 직접 검증.

---

## 2. 어떻게 했나 (method)

### 2.1 op-gap 정의

비선형 op `f` (입력 → 출력 `z=f(·)`) 하나에 대해, **clean·corrupt 두 실제 forward 의 활성 텐서**를
op 입력으로 넣어:

- 참 변화: `Δz_true = f(corrupt 입력) − f(clean 입력)`
- 추정량 `Ê`: 입력 변화로부터 `Δz` 예측 (EAP 1차, 또는 모듈)

```
op-gap(Ê) = sqrt( Σ ‖Ê − Δz_true‖²_F  /  Σ ‖Δz_true‖²_F )
```

= op 의 clean→corrupt 출력 변화를 추정량이 남기는 **상대 RMS 오차**. 합산은 전 example·layer·head·position·hidden-dim.

**규약 (중요):**
- op 입력 = **실제 clean·corrupt 활성값** (상류 전부 corrupt = full-transition; single-edge 부분 patch 아님).
- op 을 **고립**해 평가 (출력을 망에 재전파 안 함) → downstream 비선형/self-repair 가 **배제됨**. op 고유 비선형만 측정.
- edge 샘플링 없음 — 측정 대상이 attribution 그래프가 아니라 **op 활성 텐서 전체**.

### 2.1b op-gap 의 `Ê` 와 실제 EAP score 의 관계 (forward/backward 등가성)

주의: 여기서 측정하는 `Ê` 는 **EAP 의 스칼라 edge score 가 아니다.** EAP 의 backward 가
의존하는 **op 의 clean-point 선형화(Jacobian)** 를, 방향만 바꿔 forward 로 적용한 것이다.

- **실제 EAP (backward, VJP)**: edge `u→(op f 의 입력 t)` 의 score
  `s = (a_u^clean − a_u^corrupt) · g_t`, `g_t = ∂m/∂t`.
  이 기울기 `g_t` 는 metric 에서 backward 로 흐르며 op `f` 를 통과할 때 **`f` 의 clean 지점 Jacobian
  `J_f(clean)` 의 전치** 를 쓴다. → plain EAP 는 clean 에서만 선형화.
- **op-gap 의 `Ê` (forward, JVP)**: op 출력의 1차 테일러 `Ê = J_f(clean)·Δ_input`
  (예: `Δq·kᵀ + q·Δkᵀ`, 전부 clean 에서 전개).

→ **둘 다 동일한 `J_f(clean)` 을 쓴다. 방향만 반대(VJP vs JVP).**
- EAP(backward): `Δa · J_f(clean)ᵀ · g_out`
- op-gap(forward): `J_f(clean) · Δa` 를 참 출력변화 `Δz_true` 와 비교

"clean 선형화가 op 의 실제 변화를 얼마나 못 맞히나"는 방향 무관이라, forward 로 재는 게 참 변화와
직접 비교돼 더 깨끗하다(스칼라 score·metric 기울기를 안 거침). 따라서 **op-gap = op 의 clean 선형화가
남기는 오차 = EAP 의 per-op 고차오차 원천.** 모듈은 `J_f(clean)` → secant/midpoint Jacobian 으로
교체해 `J·Δ = Δz_true` 가 되게 하고(op-gap→0), 같은 Jacobian 이 EAP 의 VJP 에도 쓰이므로 이것이 곧
EAP score 에 가해지는 보정이다 (midpoint/secant = 경로 평균 Jacobian = 그 op 의 closed-form IG).

**단서**: op-gap 은 op 출력 텐서의 선형화 오차(full-transition, 모든 상류 source 동시)이지 **per-edge
스칼라 score 오차가 아니다.** 개별 edge score 오차는 single-edge ε / joint-AP 실험에서 직접 쟀고
(거기선 single-edge regime mismatch 로 모듈이 못 줄임), op-gap 은 "op 단위로 EAP 1차가 어디서 깨지고
모듈이 그 선형화를 고치나"를 isolate 한 별개 관점이다. → op-gap≈0 은 **메커니즘 증명**이지 attribution
충실도 보장이 아니다.

### 2.2 op 별 추정량

| op | `Δz_true` | EAP `Ê` (1차) | 모듈 `Ê` |
|---|---|---|---|
| **Q@K** | `q*k*ᵀ − qkᵀ` | `Δq·kᵀ + q·Δkᵀ` | midpoint: `Δq·(k+k*)/2ᵀ + (q+q*)/2·Δkᵀ` |
| **A@V** | `A*V* − AV` | `ΔA·V + A·ΔV` | midpoint: `ΔA·(V+V*)/2 + (A+A*)/2·ΔV` |
| **GeGLU gate·up** | `x*y* − xy` (x=act, y=up) | `Δx·y + x·Δy` | midpoint: `Δx·(y+y*)/2 + (x+x*)/2·Δy` |
| **GELU** | `φ(g*) − φ(g)` | `φ'(g)·Δg` (tangent) | secant: `[φ(g*)−φ(g)]/Δg · Δg` |
| **Norm** (LN/RMS) | `N(x*) − N(x)` | `J_full(x)·Δx` (정확 해석 Jacobian) | FrLN: `J_frozen(x)·Δx` (분모 detach) |

(`Δq=q*−q` 등. Q@K 는 ×1/√d_head. A=softmax(QK(+softcap)+causal mask).)

### 2.3 구현

- 실제 forward 두 번(clean, corrupt) 돌리고 **forward hook** 으로 op 입출력 텐서 캡처 (GPT-2 vanilla, gemma2 eager/bf16).
- GPT-2: Q@K, A@V (c_attn 분해), GELU(c_fc), LayerNorm. gemma2: GeGLU gate·up, gelu_tanh, RMSNorm
  (Q@K/A@V 는 rotary+GQA+softcap 처리 필요 → 미실행).
- bf16 모델도 gap 계산은 fp32. clean/corrupt 토큰 길이 다른 예시는 제외(정렬 필요).
- 스크립트: `experiments/mib/scripts/oplevel_cross_gap_gpt2_ioi.py` (Q@K, A@V),
  `oplevel_sm_gap_gpt2_ioi.py` (GELU), `oplevel_fn_gap_gpt2_ioi.py` (LayerNorm),
  `oplevel_gap_gemma2_ioi.py` (GeGLU·gelu·RMSNorm).

---

## 3. 결과 테이블

### 3.1 GPT-2 small / IOI (100 examples)

| op | 모듈 | EAP op-gap | 모듈 op-gap |
|---|---|---|---|
| Q@K | Bilinear(midpoint) | 0.6591 | 1.25e-6 |
| A@V | Bilinear(midpoint) | 0.2656 | 2.20e-7 |
| GELU | SM(secant) | 0.7182 | 2.33e-8 |
| LayerNorm | FrLN | 0.2482 | **0.6555** |

### 3.2 gemma-2-2b / IOI (100 examples)

| op | 모듈 | EAP op-gap | 모듈 op-gap |
|---|---|---|---|
| Q@K | Bilinear(midpoint) | 0.8260 | 1.62e-6 |
| A@V | Bilinear(midpoint) | 0.1480 | 2.13e-7 |
| GeGLU gate·up | Bilinear(midpoint) | 0.6843 | 1.11e-7 |
| gelu_tanh | SM(secant) | 0.5947 | 0.00e+0 |
| RMSNorm | FrLN | 0.3110 | **0.1082** |

(gemma2 attn: rotary + GQA(8Q/4KV) + softcap(50) 재구성. Q@K 는 raw matmul 기준.)
(모듈 op-gap 1e-6~1e-8 = float32 roundoff floor — midpoint/secant 는 대수적으로 정확해 잔차가 부동소수점 오차뿐.)

### 3.3 통합 (EAP op-gap → 모듈 op-gap)

| op type | 모듈 | gpt2 | gemma2 |
|---|---|---|---|
| Q@K | Bilinear | 0.6591 → 1.25e-6 | 0.8260 → 1.62e-6 |
| A@V | Bilinear | 0.2656 → 2.20e-7 | 0.1480 → 2.13e-7 |
| GeGLU gate·up | Bilinear | (gpt2 없음) | 0.6843 → 1.11e-7 |
| GELU | SM | 0.7182 → 2.33e-8 | 0.5947 → 0.00e+0 |
| Norm | FrLN | 0.2482 → 0.6555 (악화) | 0.3110 → 0.1082 (개선) |

읽는 법: **EAP op-gap = 1차가 놓치는 2차항의 상대 크기. 모듈 op-gap = 모듈 적용 후 남는 오차.**
모듈 op-gap ≈ 0 = 모듈이 그 op 의 2차항을 정확히 메움.

---

## 4. 결과 분석

### 4.1 Bilinear · SM — 아키텍처 무관하게 2차항을 정확히 닫는다 (robust)

- EAP 1차는 각 op 에서 출력 변화의 **15~83% 를 놓친다** (gpt2 Q@K 0.66 / A@V 0.27 / GELU 0.72;
  gemma2 Q@K 0.83 / A@V 0.15 / GeGLU 0.68 / gelu 0.59).
- Bilinear(midpoint), SM(secant)는 그 갭을 **float32 roundoff floor(1e-6~1e-8)까지 메운다 = 사실상 정확.**
  - secant: 1-D element-wise 라 `[φ(g*)−φ(g)]/Δg·Δg = Δz_true` 항등적 0.
  - midpoint: bilinear 라 `Δz_true` 와 **대수적으로 정확히 일치**.
- 5개 op type × 2 모델 전부 성립 → **"모듈이 EAP 의 2차항을 보정한다"는 주장 확정.**
- 특히 gemma2 **Q@K (0.826→1.6e-6)** 와 **GeGLU gate·up (0.684→1.1e-7)**: EAP 가 각각 83%, 68% 를 놓치는데
  midpoint 가 완전히 잡는다. GeGLU 는 GPT-2 엔 없던 op — gemma2 의 큰 bilinear CPR gain(6→22)이
  나오는 자리에서 가장 큰 2차항을 정확히 잡는다 → CPR gain 과 일관.

### 4.2 FrLN — 아키텍처 의존, 부분적

- **GPT-2 LayerNorm**: 악화 (0.248 → 0.656). FrLN 이 LN gradient 의 실항(분산/분모 결합)을 detach 하는데,
  LayerNorm 에선 그게 갭을 키운다 → Taylor gap-filler 가 아니라 ranking heuristic.
- **gemma2 RMSNorm**: 개선 (0.311 → 0.108). frozen denominator 가 tangent(J_full)보다 true 유한변화에
  더 가까움. 단 ~0 은 아님(0.108 잔차 = RMSNorm 고차곡률, FrLN 이 정확 corrector 는 아님).
- 원인: LayerNorm/RMSNorm 은 element-wise 도 bilinear 도 아닌 multi-D 정규화 → secant/midpoint 같은
  exact corrector 가 **원리적으로 없다.** FrLN 은 갭을 닫는 게 아니라 분모항을 detach 하는 다른 보정이라,
  norm 종류에 따라 효과가 갈린다.

### 4.3 종합 해석

- **op-level (full-transition)** 에서는 Bilinear·SM 이 2차항을 정확히, FrLN 은 RMSNorm 에서 부분적으로 닫는다.
- **현대 RMSNorm 아키텍처(gemma/qwen/llama)에선 세 모듈 모두 자기 op-gap 을 줄인다** → gemma2 의 CPR gain 이
  GPT-2 보다 훨씬 큰 이유와 일관 (추가 GeGLU bilinear op + RMSNorm 에서 FrLN 작동).

### 4.4 한계 (정직하게)

- op-gap 은 **op 하나의 비선형을 isolate** 한 지표. op 출력이 downstream 비선형을 통과하며 받는 추가 왜곡
  (self-repair)은 **배제**된다. 따라서 **op-gap ≈ 0 이어도 최종 logit-level / single-edge AP fidelity 는 보장 안 됨**
  (앞 실험에서 모듈이 logit-level 갭은 못 줄였고, top edge 의 ~95% self-repair 오차가 지배).
- 즉 모듈은 "각 op 의 2차항은 정확히 모델링하지만 self-repair 는 못 한다." op-gap=0 은 메커니즘 검증이지
  attribution 충실도 보장이 아니다.
- op-level full-transition = circuit ablation(CPR) regime 과 가까움 → 모듈의 진짜 효용은 CPR(모델 행동)로 검증.

---

## 5. 파일

| 스크립트 | 측정 |
|---|---|
| `scripts/oplevel_cross_gap_gpt2_ioi.py` | GPT-2 Q@K, A@V |
| `scripts/oplevel_sm_gap_gpt2_ioi.py` | GPT-2 GELU 곡률 |
| `scripts/oplevel_fn_gap_gpt2_ioi.py` | GPT-2 LayerNorm |
| `scripts/oplevel_gap_gemma2_ioi.py` | gemma2 GeGLU·gelu·RMSNorm |
| `scripts/oplevel_attn_gap_gemma2_ioi.py` | gemma2 Q@K·A@V (rotary+GQA+softcap) |

(참고: single-edge ε / joint-AP / SM-verify 는 별도 실험 — edge 샘플링 기반, op-gap 과 다름.)
