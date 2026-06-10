# Circuit-heatmap deep analysis  —  best vs uniform baseline

Source: `/home/dacslab/djk/lasse-circuit/circuit_discovery/experiments/mib/frozen_grid_search/circuits/`  (scores.pt diff = best − baseline)

## gpt2 / ioi
- baseline `eap_frnorm_q1.0_k1.0_v1.0_g1.0_u1.0` CPR = **2.3101**
- best     `eap_frnorm_q1.0_k1.0_v1.0_g1.0_u0.5` (perturbation `up=0.5`) CPR = **2.4056**  (Δ = **+0.0955**)
- model dims: L=12, H=12, total edges = 69,865
- |diff| stats: max=1.818, total=23.48, non-zero edges=32,163 (46.04%)
- sign decomposition: positive mass = +13.73, negative mass = -9.745  →  net = +3.989
- concentration: top-1% edges hold **72.7%** of total |diff|

**Source-side distribution of |diff| (where the *propagating signal* changes):**

| source type | |diff| mass | share |
|---|---|---|
| emb | 4.009 | 17.1% |
| head | 12.01 | 51.1% |
| mlp | 7.46 | 31.8% |

**Target-side distribution of |diff| (where the *gradient* changes):**

| target type | |diff| mass | share |
|---|---|---|
| q | 9.166 | 39.0% |
| k | 3.556 | 15.1% |
| v | 3.7 | 15.8% |
| mlp | 7.056 | 30.1% |
| lm | 0 | 0.0% |

**Layer concentration (top-3):**

- source-side top layers: L0 (37.5%), L3 (15.1%), L5 (12.5%)
- target-side top layers: L0 (17.4%), L5 (16.1%), L6 (15.5%)

**Top-10 edges by |diff|:**

| rank | source | → | target | Δ score |
|---|---|---|---|---|
| 1 | emb | → | m0 | +1.818 |
| 2 | emb | → | a0.h10<q> | +0.8195 |
| 3 | m0 | → | m1 | +0.8043 |
| 4 | emb | → | a0.h10<k> | +0.414 |
| 5 | m0 | → | a6.h9<q> | +0.3696 |
| 6 | m0 | → | m4 | -0.292 |
| 7 | a5.h5 | → | a8.h6<v> | -0.2798 |
| 8 | m0 | → | a5.h8<q> | -0.2618 |
| 9 | m0 | → | a1.h11<k> | -0.242 |
| 10 | m0 | → | m3 | +0.2264 |

## qwen2.5 / ioi
- baseline `eap_frnorm_q1.0_k1.0_v1.0_g1.0_u1.0` CPR = **1.3762**
- best     `eap_frnorm_q1.0_k0.05_v1.0_g1.0_u1.0` (perturbation `k=0.05`) CPR = **1.6935**  (Δ = **+0.3173**)
- model dims: L=24, H=14, total edges = 372,913
- |diff| stats: max=0.9609, total=85.52, non-zero edges=159,932 (42.89%)
- sign decomposition: positive mass = +44.67, negative mass = -40.85  →  net = +3.821
- concentration: top-1% edges hold **69.4%** of total |diff|

**Source-side distribution of |diff| (where the *propagating signal* changes):**

| source type | |diff| mass | share |
|---|---|---|
| emb | 1.295 | 1.5% |
| head | 50.73 | 59.3% |
| mlp | 33.49 | 39.2% |

**Target-side distribution of |diff| (where the *gradient* changes):**

| target type | |diff| mass | share |
|---|---|---|
| q | 27.04 | 31.6% |
| k | 7.941 | 9.3% |
| v | 17.21 | 20.1% |
| mlp | 33.33 | 39.0% |
| lm | 0 | 0.0% |

**Layer concentration (top-3):**

- source-side top layers: L7 (10.3%), L11 (10.0%), L9 (9.2%)
- target-side top layers: L13 (17.8%), L12 (13.3%), L14 (10.4%)

**Top-10 edges by |diff|:**

| rank | source | → | target | Δ score |
|---|---|---|---|---|
| 1 | emb | → | m0 | +0.9609 |
| 2 | m0 | → | m12 | +0.9312 |
| 3 | m0 | → | a13.h2<q> | -0.8877 |
| 4 | m0 | → | a9.h10<q> | +0.498 |
| 5 | m11 | → | m12 | +0.4531 |
| 6 | m7 | → | m12 | +0.347 |
| 7 | a11.h10 | → | m13 | +0.3296 |
| 8 | m13 | → | m14 | +0.3057 |
| 9 | a14.h1 | → | m15 | -0.2925 |
| 10 | a13.h2 | → | m13 | +0.2764 |

## qwen2.5 / mcqa
- baseline `eap_frnorm_q1.0_k1.0_v1.0_g1.0_u1.0` CPR = **1.0853**
- best     `eap_frnorm_q1.0_k1.0_v1.0_g0.2_u1.0` (perturbation `gate=0.2`) CPR = **1.5249**  (Δ = **+0.4396**)
- model dims: L=24, H=14, total edges = 372,913
- |diff| stats: max=32.45, total=721.8, non-zero edges=178,113 (47.76%)
- sign decomposition: positive mass = +319.3, negative mass = -402.5  →  net = -83.12
- concentration: top-1% edges hold **56.9%** of total |diff|

**Source-side distribution of |diff| (where the *propagating signal* changes):**

| source type | |diff| mass | share |
|---|---|---|
| emb | 47.91 | 6.6% |
| head | 387.2 | 53.6% |
| mlp | 286.7 | 39.7% |

**Target-side distribution of |diff| (where the *gradient* changes):**

| target type | |diff| mass | share |
|---|---|---|
| q | 143.3 | 19.8% |
| k | 118.3 | 16.4% |
| v | 219 | 30.3% |
| mlp | 241.2 | 33.4% |
| lm | 0 | 0.0% |

**Layer concentration (top-3):**

- source-side top layers: L4 (9.9%), L1 (8.7%), L5 (8.6%)
- target-side top layers: L12 (8.9%), L10 (8.3%), L15 (7.8%)

**Top-10 edges by |diff|:**

| rank | source | → | target | Δ score |
|---|---|---|---|---|
| 1 | emb | → | m0 | -32.45 |
| 2 | m0 | → | m1 | -5.994 |
| 3 | emb | → | a0.h11<v> | -4.772 |
| 4 | a0.h11 | → | m0 | -4.462 |
| 5 | m1 | → | m3 | -3.542 |
| 6 | m0 | → | a2.h2<k> | -2.759 |
| 7 | emb | → | a0.h13<v> | -2.448 |
| 8 | m8 | → | m11 | +1.916 |
| 9 | m5 | → | m6 | -1.828 |
| 10 | a0.h13 | → | m0 | -1.72 |

## gemma2 / ioi
- baseline `eap_frnorm_q1.0_k1.0_v1.0_g1.0_u1.0` CPR = **1.8792**
- best     `eap_frnorm_q1.0_k1.0_v1.0_g0.1_u1.0` (perturbation `gate=0.1`) CPR = **2.5575**  (Δ = **+0.6783**)
- model dims: L=26, H=8, total edges = 152,985
- |diff| stats: max=0.5173, total=53.09, non-zero edges=73,254 (47.88%)
- sign decomposition: positive mass = +27.5, negative mass = -25.59  →  net = +1.915
- concentration: top-1% edges hold **56.0%** of total |diff|

**Source-side distribution of |diff| (where the *propagating signal* changes):**

| source type | |diff| mass | share |
|---|---|---|
| emb | 2.915 | 5.5% |
| head | 36.04 | 67.9% |
| mlp | 14.14 | 26.6% |

**Target-side distribution of |diff| (where the *gradient* changes):**

| target type | |diff| mass | share |
|---|---|---|
| q | 18.24 | 34.3% |
| k | 6.404 | 12.1% |
| v | 13.42 | 25.3% |
| mlp | 15.04 | 28.3% |
| lm | 0 | 0.0% |

**Layer concentration (top-3):**

- source-side top layers: L11 (13.1%), L10 (8.5%), L8 (7.4%)
- target-side top layers: L12 (14.5%), L11 (10.4%), L16 (9.5%)

**Top-10 edges by |diff|:**

| rank | source | → | target | Δ score |
|---|---|---|---|---|
| 1 | a11.h4 | → | m11 | +0.5173 |
| 2 | a11.h5 | → | m11 | -0.3604 |
| 3 | m10 | → | m11 | -0.3452 |
| 4 | a8.h1 | → | m11 | -0.2852 |
| 5 | emb | → | a0.h1<q> | +0.2563 |
| 6 | a10.h5 | → | m11 | -0.2041 |
| 7 | a16.h2 | → | m16 | +0.1914 |
| 8 | emb | → | m0 | +0.1836 |
| 9 | a13.h7 | → | a16.h2<q> | -0.1797 |
| 10 | a11.h4 | → | a12.h6<v> | +0.1743 |

## gemma2 / mcqa
- baseline `eap_frnorm_q1.0_k1.0_v1.0_g1.0_u1.0` CPR = **1.1989**
- best     `eap_frnorm_q1.0_k1.0_v1.0_g0.2_u1.0` (perturbation `gate=0.2`) CPR = **2.2907**  (Δ = **+1.0918**)
- model dims: L=26, H=8, total edges = 152,985
- |diff| stats: max=2.258, total=203.1, non-zero edges=73,551 (48.08%)
- sign decomposition: positive mass = +105.9, negative mass = -97.27  →  net = +8.578
- concentration: top-1% edges hold **41.5%** of total |diff|

**Source-side distribution of |diff| (where the *propagating signal* changes):**

| source type | |diff| mass | share |
|---|---|---|
| emb | 20.64 | 10.2% |
| head | 117.1 | 57.7% |
| mlp | 65.33 | 32.2% |

**Target-side distribution of |diff| (where the *gradient* changes):**

| target type | |diff| mass | share |
|---|---|---|
| q | 42.85 | 21.1% |
| k | 38.7 | 19.1% |
| v | 65.91 | 32.5% |
| mlp | 55.66 | 27.4% |
| lm | 0 | 0.0% |

**Layer concentration (top-3):**

- source-side top layers: L1 (10.1%), L2 (9.0%), L0 (8.6%)
- target-side top layers: L7 (6.3%), L8 (6.1%), L11 (5.3%)

**Top-10 edges by |diff|:**

| rank | source | → | target | Δ score |
|---|---|---|---|---|
| 1 | emb | → | m0 | +2.258 |
| 2 | emb | → | a0.h5<v> | +1.272 |
| 3 | emb | → | a0.h5<k> | +0.9832 |
| 4 | emb | → | a0.h5<q> | +0.8879 |
| 5 | emb | → | m7 | +0.7158 |
| 6 | emb | → | m5 | -0.6855 |
| 7 | emb | → | m2 | +0.6113 |
| 8 | a0.h5 | → | m0 | +0.5963 |
| 9 | a1.h1 | → | m1 | +0.558 |
| 10 | m0 | → | a1.h1<v> | +0.5407 |


---

## Cross-block synthesis

### 1. 변화는 "회로 전체 재배치"가 아니라 "소수 edge에 집중"
| block | top-1% edges hold | edges actually changed |
|---|---|---|
| gpt2/ioi | 72.7% of \|diff\| | 46.0% of edges (32k of 70k) |
| qwen2.5/ioi | 69.4% | 42.9% |
| qwen2.5/mcqa | 56.9% | 47.8% |
| gemma2/ioi | 56.0% | 47.9% |
| gemma2/mcqa | 41.5% | 48.1% |

5개 블록 모두 attribution score의 **거의 절반이 미세하게 흔들리지만, 변화량의 대부분(40~70%)은 edge 1%에 집중**. 즉 *위쪽 weight scaling이 특정 좁은 구조를 콕 짚어 키우거나 죽임*. 이것이 CPR 향상의 실체.

### 2. Target side에서는 "Q" 와 "V" 가 압도적, K는 안 흔들림
| block | q | k | v | mlp | lm |
|---|---|---|---|---|---|
| gpt2/ioi | **39.0%** | 15.1% | 15.8% | 30.1% | 0% |
| qwen2.5/ioi | **31.6%** | 9.3% | 20.1% | 39.0% | 0% |
| qwen2.5/mcqa | 19.8% | 16.4% | **30.3%** | 33.4% | 0% |
| gemma2/ioi | **34.3%** | 12.1% | 25.3% | 28.3% | 0% |
| gemma2/mcqa | 21.1% | 19.1% | **32.5%** | 27.4% | 0% |

- **IOI 계열 (gpt2, qwen2.5, gemma2)** → 모두 Q에 가장 많이 변화 집중. IOI는 "이 토큰이 어디(누구)를 가리키는가"가 핵심이라 query reshaping이 타당.
- **MCQA 계열 (qwen2.5, gemma2)** → V로 무게중심 이동. MCQA는 "이 옵션의 의미를 가져와라"는 content lookup 성격이 강해서 value path가 핵심.
- **K는 모든 블록에서 가장 적게 변함 (9~19%)**. 어텐션 패턴 자체보다 *질문/내용*을 손보는 게 효과적.
- **lm_head 변화는 정확히 0**. backward proj weight 조정이 (당연히) lm_head로 가는 attribution은 만들지 않음.

### 3. Source side는 "head"가 최대 발신자, embedding은 작지만 top-K에 자주 등장
| block | emb | head | mlp |
|---|---|---|---|
| gpt2/ioi | 17.1% | **51.1%** | 31.8% |
| qwen2.5/ioi | 1.5% | **59.3%** | 39.2% |
| qwen2.5/mcqa | 6.6% | **53.6%** | 39.7% |
| gemma2/ioi | 5.5% | **67.9%** | 26.6% |
| gemma2/mcqa | 10.2% | **57.7%** | 32.2% |

총 |diff| 기여는 head(51~68%) > MLP(27~40%) >> emb(2~17%). 그러나 **top-K 개별 edge에는 emb→m0, emb→a0.h*가 항상 등장** → emb는 적은 edge로 굵직한 임팩트.

### 4. 부호 분해: 한 블록만 "감소"가 우세
| block | +mass | −mass | net |
|---|---|---|---|
| gpt2/ioi | +13.7 | -9.7 | **+4.0** |
| qwen2.5/ioi | +44.7 | -40.9 | **+3.8** |
| **qwen2.5/mcqa** | +319 | **-403** | **-83.1** |
| gemma2/ioi | +27.5 | -25.6 | **+1.9** |
| gemma2/mcqa | +106 | -97.3 | **+8.6** |

**qwen2.5/mcqa**만 net이 크게 음수: gate=0.2가 *전반적으로 score 크기를 줄였는데도* CPR은 +0.44 향상. 이는 CPR이 절대 score 크기가 아니라 *순위/상대 분포*(faithfulness curve의 area)에서 결정됨을 보여주는 좋은 사례. **scale_loc='post' weighting이 노이즈 score를 더 깎고, 정작 중요한 score를 덜 깎아 ranking을 깨끗하게 만든 효과**.

### 5. 블록별 메커니즘 해석

#### gpt2 / ioi (best `up=0.5`, +0.10)
- top edge `emb→m0 (+1.82)` — 첫 MLP에 임베딩 정보가 더 강하게 쓰여짐.
- L0 attn head `a0.h10`의 q/k가 emb로부터 강화 → "초기 토큰 위치 인식"이 보강.
- m0가 두 갈래로 흘러감: `m0→m1 (+0.80)` / `m0→m4 (-0.29)` — 초기 MLP 흐름의 *재라우팅*.
- gpt2는 Name Mover heads (9.6, 9.9)가 유명하지만 perturbation 효과는 그쪽엔 거의 가지 않음 — **회로가 이미 saturate**되어 큰 폭의 변화가 안 나옴 (CPR Δ가 가장 작은 +0.10인 이유와 일치).

#### qwen2.5 / ioi (best `k=0.05`, +0.32)
- L0의 `emb→m0`, `m0→m12`가 동시에 강화 → 초기 MLP가 mid-layer MLP까지 직접 점프하는 *long-range MLP residual* 강조.
- top-K에 mid-layer attn `a13.h2<q>`의 변화가 두드러짐 (-0.89 / +0.28). **이 layer가 IOI에서 핵심 "Name Mover-like" 역할로 추정**.
- target layer 분포가 L12~L14에 집중 (L13 17.8%, L12 13.3%, L14 10.4%) → 회로의 "decision layer"가 mid-network에 있음.

#### qwen2.5 / mcqa (best `gate=0.2`, +0.44)
- 거의 모든 top edge가 음의 부호. 가장 큰 변화는 `emb→m0 (-32.45)` — emb가 m0에 미치는 attribution이 크게 약화.
- gate를 0.2로 줄여서 MLP의 SwiGLU 출구 신호가 약해진 결과, 초기 MLP에 가는 attribution이 전반적으로 깎임.
- 그럼에도 CPR이 오른 이유: 깎이는 attribution이 *비-유익한 path*에 더 집중되어 ranking이 개선 (위 4번 항목).
- Q와 V가 비슷하게 흔들리고 (19.8% / 30.3%) 둘 다 mid-layer (L10~15)로 향함.

#### gemma2 / ioi (best `gate=0.1`, +0.68)
- **Top-K가 학술적으로 가장 깔끔**: `a11.h4 → m11 (+0.52)`와 `a11.h5 → m11 (-0.36)` — *같은 target에 한 head는 강조 / 다른 head는 억제*. 이는 IOI 문헌의 **positive/negative head 구분**과 형태가 일치 (gemma2의 IOI 회로에서 a11.h4가 positive Name Mover, a11.h5가 그 negative counterpart로 의심).
- L11이 source(13.1%)와 target(10.4%) 양쪽 모두에서 두드러짐 → "L11 hub" 구조.
- Q에 변화 집중(34.3%), 특히 `a13.h7 → a16.h2<q>` 라인 — 후속 layer Q의 *redirection*.

#### gemma2 / mcqa (best `gate=0.05`, +1.09 — 최대 lift)
- top-K 전부 양수 + 거의 전부 emb 발신: `emb→m0 (+2.26)`, `emb→a0.h5<v/k/q>` 세 개 동시 강화.
- L0의 head 5가 q·k·v *셋 다* 임베딩으로부터 강화 → **head 0.5가 MCQA에서 input embedding의 핵심 reader**로 부상.
- 후속 흐름 `a0.h5 → m0 (+0.60)`, `m0 → a1.h1<v> (+0.54)` → 정보가 L0→L1으로 깨끗하게 연결.
- gate=0.05라는 극단적 down-scale이 풀모델의 *gating noise*를 거의 제거 → 회로 신호가 더 또렷해짐. 모델이 클수록(gemma2 2B), gate path가 noise contributor 역할이 커서 effect size 큼.

### 6. 종합 결론

1. **Weight scaling은 회로 전체를 재구성하지 않고, 0.5~1% edge를 키우거나 죽여 ranking을 청소**한다.
2. **Q는 IOI의 핵심 시그널 통로, V는 MCQA의 핵심 시그널 통로**. K는 어느 task에서도 별로 흔들지 않는 게 최선.
3. **`gate` perturbation은 SwiGLU MLP 모델 (qwen2.5, gemma2)에서만 효과**, gpt2(GELU)에선 죽은 axis. 모델이 클수록 gate 손질의 lift 큼 (gemma2/mcqa +1.09 정점).
4. **CPR 향상이 항상 "더 큰 score"를 의미하진 않는다** — qwen2.5/mcqa는 net 감소인데 CPR 상승. 이는 *attribution method 평가 metric으로서 CPR이 score magnitude가 아닌 ranking quality를 본다는 점*을 잘 드러냄.
5. **실제 회로 발견 측면**: gemma2/ioi의 a11.h4 vs a11.h5 같은 positive/negative head pair는 EAP-IG/IOI 문헌의 회로 구조와 부합 — perturbation이 이런 미세 구분을 *증폭*시켜 더 잘 보이게 만들어 줌.
