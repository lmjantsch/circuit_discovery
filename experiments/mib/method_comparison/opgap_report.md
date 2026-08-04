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
   - **self 고차항** (single edge 가 아래로 전파되며 겪는 자기 비선형 = self-repair). 갭의 대부분을 차지
     (component ablation 56%, 2-edge joint 50%; single edge 는 섭동이 작아 20~31% — §1a-정정 참조).
   - **cross/곡률 2차항** (op 에서 두 입력이 동시에 움직일 때 / 비선형 곡률). 모듈이 겨누는 것.
   self-repair 가 모듈이 겨누는 2차항을 덮어버려 "갭 못 메움"으로 나왔다.

→ 모듈이 겨누는 **2차항만 isolate** 하려면, logit 이 아니라 **그 항이 태어나는 op 출력**에서, downstream
비선형이 들어오기 **전에** 재야 한다. 그게 op-gap 실험의 목적이다.

> 핵심 주장: "우리 모듈은 EAP 1차가 놓치는 op 의 2차항을 보정한다." op-gap 은 이 주장을 op 단위로 직접 검증.

---

## 1a. logit-level 오차는 무엇으로 이루어져 있나

`golden AP − EAP = H` (2차 이상 전부). 이 `H` 를 개념적으로 셋으로 나눌 수 있고, **왜 스칼라 하나에서
op 별로 역산이 안 되는지**가 여기서 나온다.

### ① 각 op 의 선형화 오차 (국소 발생)

섭동 `δ` 가 v 입력 → op₁ → op₂ → … → logit 으로 흐르며, **각 비선형 op 이 자기 오차를 새로 만든다**:

```
ε_i = [f_i(x_i + δ_i) − f_i(x_i)]  −  J_i(clean)·δ_i
        └──── 실제 응답 ────┘        └── EAP 가 쓴 1차 ──┘
```

**이 `ε_i` 의 성분은 op 종류에 따라 두 가지로 갈린다 — 곡률(curvature) 과 교차항(cross):**

| | **곡률 (curvature)** | **cross (교차항)** |
|---|---|---|
| 어떤 op | 입력이 **1개**인 비선형: GELU, softmax, norm | 활성 **2개**가 곱해지는 bilinear: Q@K, A@V, gate·up |
| 1차가 놓치는 항 | `f(x+δ) − f(x) − f'(x)δ = ½f''δ² + …` | `(x+Δx)(y+Δy) − xy − (Δx·y + x·Δy) = Δx·Δy` |
| 원인 | 함수가 **휘어서** 접선이 곡선에서 벗어남 | 두 입력이 **동시에** 움직여 생기는 상호작용 |
| 입력 하나만 움직이면 | **여전히 발생** | **0** (`z=x·y` 는 y 고정 시 x 에 완전 선형, `∂²z/∂x² = 0`) |
| 정확한 보정 | **secant** (현의 기울기) → SM | **midpoint** (상대 operand 중점) → Bilinear |

`z = x·y` 는 각 변수에 대한 2차 도함수가 0 이라 "곡률"이 없다 — 비선형성이 **순전히 상호작용**에만 있다.
이것이 single-edge patch 에서 cross ≡ 0 인 이유이고(operand 하나만 이동), 반면 곡률은 입력 하나만 흔들려도 남는다.
또한 같은 곡률이어도 **1-D element-wise(GELU)면 secant 가 정확**하지만 **multi-D(softmax·norm)면 원리적으로 불가** —
이 구분이 뒤에서 모듈(closed-form) vs IG(적분)의 분업으로 이어진다.

**op-gap 이 재는 게 정확히 이 `ε_i`** (고립 상태·full-transition 크기): softmax 1.46, GELU 0.72, Q@K 0.66, norm 0.25.

### ② downstream 혼합·증폭 (전파되며 변형)

①이 만든 `ε_i` 는 그 자리에 머물지 않고 남은 경로를 마저 통과한다:
- **증폭/감쇠**: 최종 logit 기여분은 `ε_i` 가 아니라 `(나머지 망의 Jacobian)·ε_i`
- **op 간 결합**: op₁ 의 오차가 op₂ 의 **입력을 바꿔 동작점을 이동** → op₂ 의 오차 자체가 달라짐.
  즉 오차들이 **독립적으로 더해지지 않는다.**

실측 — **같은 cross 항을 세 깊이에서**: 태어난 op 66% → logit(2-edge joint) 10.1% → top edge 총오차 안에선 불가시.
→ **op-gap 을 더해도 logit 오차가 안 나오고, 역으로 되돌릴 수도 없다.**

### ③ self-repair (모델이 스스로 보상)

섭동이 downstream 에 닿으면 **다른 component 가 동작을 바꿔 원래 효과를 부분 상쇄**한다:
- **norm 재정규화**: component 를 빼면 잔차 norm 이 줄고 → norm 이 나머지를 키워 변화 일부 흡수
- **backup head (Hydra effect)**: 주 head 를 끊으면 억제돼 있던 head 가 기능을 대신함

실측(component ablation): `|net_frozen| 1.226 → |net_real| 1.113`, **LN 매개 self-repair ≈ 0.156 (효과의 ~14%)**.
EAP 1차가 남기는 총 상대오차는 **component ablation 56%**, **2-edge joint 50%**, **single edge 20~31%**
— 섭동이 클수록 커진다(§1a-정정).
근사 오차가 아니라 **실제 모델 행동**이며, 여러 op 을 거친 기능적 대체라 **국소 Jacobian 보정으론 못 잡는다.**

### ⚠ 정정 — 이전 "상대오차 ~95%" 는 단위 아티팩트였다

`tracer.py:_update_scores` 는 `per_sample_scores / n_real` 로 **score 를 토큰 수로 나눈다**(per-token 정규화).
반면 golden AP 는 총효과다. IOI 평균 프롬프트 길이 **T = 16.35** (최소제곱 slope 17.28 로 확인) 만큼 단위가 어긋나
`ε = |score − I_AP| ≈ |I_AP|` 가 되어버렸다.

**결정적 증거**: 미보정 상대오차가 **표본·기법 무관하게 0.943~0.984 로 일정** — 근사 품질이 아니라 단위 격차를 잰 것.

| 표본 (EAP) | 미보정 | **×T 보정** |
|---|---|---|
| PILOT decile-9 (옛 "0.93" 출처) | 0.9478 | **0.2027** |
| GLOBAL random (무편향 모집단) | 0.9484 | **0.2897** |
| GLOBAL union top-200 | 0.9500 | **0.3140** |

→ 이 문서에서 **single-edge / joint-AP 의 "95%" 서술은 모두 정정**되었다.
**op-gap 은 영향 없음** — score 를 쓰지 않고 활성 텐서끼리의 무차원 비율이라 단위가 약분된다(7개 스크립트 모두 확인).

### 왜 스칼라에서 역산이 불가능한가

세 항은 **직교하지도 독립적이지도 않다**:
- ③은 사실 ①②를 **통해** 발현된다 (norm 비선형 = ①, 여러 layer 경유 = ②). 개념적 구분이지 수학적 분해가 아님.
- ②는 ①의 각 항에 **서로 다른 배율**을 곱하는데 그 배율이 섭동 크기·방향에 의존(비선형).
- logit 에서 관측되는 건 이 얽힌 합의 **스칼라 하나** → 다대일 비선형 합성이라 **역상이 존재하지 않는다.**

**구체 예 — edge `u→v⟨Q⟩` 하나 patch:**
1. v 의 `Q@K`: Q 만 이동 → **cross ≡ 0**, 여기선 EAP 정확
2. v 의 **softmax**: 점수가 바뀜 → **곡률 오차 발생 (①)**
3. 잔차로 진입 → 이후 layer 의 norm·GELU 에서 추가 ①, 게다가 섭동이 뒤쪽 head 의 **Q·K 를 동시에** 건드려 **cross 도 발생 (①)**
4. 이 국소 오차들이 서로 다른 배율로 증폭/감쇠되고 서로의 동작점을 바꿈 **(②)**
5. 동시에 norm 재정규화·backup head 가 전체를 부분 상쇄 **(③)**
6. **logit 에서 보이는 건 숫자 하나** — 위 전부의 얽힌 합

→ 그래서 잡음원 진단은 logit 역산이 아니라 **각 op 을 고립시켜 직접 재는 것**(op-gap)으로만 가능하다.

---

## 1b. 전체 서사 — single/joint AP → 한계 → op-gap → 3 모듈 + input-IG

### A. 선행 실험 ①: single-edge AP (`compute_iap_gpt2_ioi.py`)

- **왜**: attribution score 가 그 edge 의 **실제 인과효과**를 얼마나 맞히나 — 충실도 직접 평가.
- **무엇을**: `I_AP(e)` = edge 하나를 끊었을 때의 **logit-diff 변화** (참값).
- **어떻게**: `in_graph` 를 전부 False 로 두고 **그 edge 하나만 True** → source u 의 기여분만 corrupt 로 swap,
  나머지 전부 clean → 실제 forward. `I_AP = mean_100prompt(clean_logit_diff − patched_logit_diff)`.
  edge 표본 V = 유효 edge 32,491개를 EAP `|score|` **10 decile × 200 = 2,000개**. 메트릭 `ε = |score − I_AP|`.

**결과 (GPT-2/IOI, 100 CF쌍)**

| method | mean ε (미보정) | Spearman(score, I_AP) | win-rate vs EAP |
|---|---|---|---|
| **EAP** | **0.00601** | **0.9613** | — |
| +FrozenNorm | 0.00600 | 0.8491 | 0.490 |
| +SecantMLP | 0.00603 | 0.8368 | 0.355 |
| +Bilinear | 0.00613 | 0.7780 | 0.083 |

**분석**: 모든 slice(ε·ranking·component)에서 EAP 1위.
(주의: 위 ε 절대값은 단위 미보정이라 크기 자체는 무의미하다 — §1a-정정. 기법 간 비교는 모두 같은 관례를
공유하므로 **paired 로는 유효**하다.)

**A-2. 표본 편향 검증 + 무편향 재측정** (`compute_iap_global_gpt2_ioi.py`)

위 V 는 EAP `|score|` 로 층화했으므로 EAP 에 유리할 수 있다. 이를 없앤 표본으로 재측정:
**랜덤 1000개**(전 유효 edge 균등 = 모집단 무편향) + **4기법 top-200 union 285개**(어느 기법도 자기 top edge 를
뺏기지 않음). 단위는 ×T 보정.

| 표본 | method | mean ε | ε/\|I_AP\| | win vs EAP | Spearman |
|---|---|---|---|---|---|
| **랜덤(무편향)** | **EAP** | **0.00072** | **0.290** | — | **0.9713** |
| n=1000 | +FrLN | 0.00082 | 0.330 | 0.259 | 0.8110 |
| | +SM | 0.00109 | 0.439 | 0.248 | 0.8421 |
| | +Bilinear | 0.00186 | 0.746 | 0.127 | 0.7649 |
| **union top-200** | **EAP** | **0.07202** | **0.314** | — | **0.9332** |
| n=285 | +FrLN | 0.07848 | 0.342 | 0.460 | 0.9076 |
| | +SM | 0.07558 | 0.329 | 0.333 | 0.9200 |
| | +Bilinear | 0.14057 | 0.613 | 0.189 | 0.8105 |

→ **표본 편향은 원인이 아니었다** (무편향 랜덤에서도 순위 동일). 그리고 단위 보정 후 EAP 의 상대오차는
**0.29~0.31** — 즉 이 regime 에서 **1차가 이미 ~70% 를 설명**한다. 고칠 여지 자체가 작다.
부수 관찰: union 표본에서 FrLN 의 Pearson 0.9186 > EAP 0.8016 (mean ε 는 근소 열세이나 선형 대응은 우위).

### B. 선행 실험 ②: joint AP (2-edge) (`compute_joint_ap_gpt2_ioi.py`)

- **왜**: 가법적 score 가 원리적으로 놓치는 **두 edge 간 상호작용(cross)** 을 잡는지 보기 위해.
- **무엇을**: `I_joint` (둘 다 끊은 효과) 와 **golden cross** `C = I_joint − I_s1 − I_s2` (비가법 성분).
- **어떻게**: head 별 `|EAP score|` 최대 Q-edge·K-edge 로 pair 구성 — **coupled 120쌍**(같은 head → `Q@K` 공유),
  **control 120쌍**(다른 head → 공유 op 없음, 대조군). pair 당 forward 3번(`e1`, `e2`, `both`).

**결과**

(모든 score 는 ×T 단위 보정 적용.)

| | coupled (n=120) | control (n=120) |
|---|---|---|
| golden \|C\| | **0.00976 (joint 의 10.1%)** | 0.00112 (1.2%) |
| EAP \|sum·T−I_joint\| | **0.04790 (\|I_joint\| 의 49.6%)** | 0.04807 (50.0%) |
| midpoint \|sum·T−I_joint\| | 0.05131 | 0.05337 |
| win vs EAP — static / midpoint | 0.317 / 0.392 | 0.275 / 0.325 |
| cross 회수 r(Δ, C) — static / midpoint | **+0.538 / +0.454** | −0.108 / −0.074 |
| 보정 크기 mean\|Δ\|/mean\|C\| — static / midpoint | **7.01× / 3.07×** | (대상 없음) |

**분석**: (1) coupled 의 cross 가 control 의 **8.7배** → 설계 검증. (2) 모듈의 Δ 가 **coupled 에서만** C 와
상관(+0.45~0.54, control ≈ 0) → **메커니즘(무엇을 겨누는지)은 검증됨**. (3) 그러나 **크기가 과하다** —
midpoint 의 보정 크기가 golden cross 의 **3.07배**(static 은 7.01배). 방향은 맞고 크기가 넘쳐 win-rate < 0.5.
그리고 남은 미보정 잔차가 `|EAP sum·T − I_joint| / |I_joint| = 49.6%` 로 cross(10.1%)보다 훨씬 커서,
cross 를 완벽히 고쳐도 총오차 변화가 묻힌다.

> 정정: 이전 판의 "cross 를 1/3만 포착(slope 3.11)"·"95.4%" 는 단위 미보정에서 나온 값이었다.
> 또한 회귀 slope 은 방향에 따라 1.05~5.6배로 흔들려(`r=0.45` 감쇠) 크기 주장의 근거로 부적합하므로,
> 방향-무관 지표인 **스케일 비 `mean|Δ|/mean|C|`** 로 대체했다.

### C. 두 실험의 한계 — 왜 **잡음원 진단**이 안 되나

1. **총합 하나만 나온다.** logit 에서의 오차는 *경로 위 모든 op 의 선형화 오차 + downstream 혼합 +
   self-repair* 의 합. 이 스칼라를 다시 op 별로 쪼갤 방법이 없다 → **"어느 op 에서 얼마"를 못 얻음**.
2. **지배항이 대상을 덮는다.** 겨누는 cross 는 joint 의 10.1%, 미보정 잔차는 49.6%. 대상을 완벽히 고쳐도 총오차 변화가
   측정 노이즈에 묻힌다 (실측: midpoint 가 방향은 맞혔는데 절대거리는 오히려 미세 악화).
3. **single-edge 엔 그 항이 아예 없다.** `Δ(Q@Kᵀ) = ΔQ·Kᵀ + Q·ΔKᵀ + ΔQ·ΔKᵀ` 에서 operand 하나만 움직이므로
   `ΔK = 0` → **cross ≡ 0**. 일어나지 않는 현상은 진단할 수 없다.
4. **스칼라 ε 는 두 가지를 구분 못 한다** — (a) 고차항 포착, (b) 1차 정확도 훼손. 예: FrLN 은 LN gradient 의
   *실재하는 1차 항*을 제거하므로 ε 증가가 "고차 못 잡음"이 아니라 "1차 훼손"이다. ε 만으론 판별 불가.


### D. op-gap 이 한계를 어떻게 보완하나

| 한계 | op-gap 의 해법 |
|---|---|
| ① 분해 불가 | **op 하나만 고립** → 남는 오차가 그 op 것 하나 → op 별 귀속 가능 |
| ② 지배항에 묻힘 | **downstream 진입 전** 측정 → self-repair 배제 → cross 가 27~83% 로 선명 |
| ③ 항이 존재하지 않음 | **full clean→corrupt** regime → 두 operand 다 이동 → cross 실재 |
| ④ 판별력 부족 | 참값이 **그 op 의 실제 출력 변화** (EAP 와 무관하게 정의) → 1차 이탈을 EAP 편향 없이 직접 측정 |

### E. 그래서 3 모듈이 도출된다

op-gap 으로 만든 noise budget(§3.4)이 두 가지를 동시에 알려준다 — **크기**와 **보정 가능성**:

- 비선형 op 은 **element-wise 이거나 bilinear 일 때만** 1-step exact corrector 를 갖는다
  (secant 는 element-wise 변화를, midpoint 는 bilinear 변화를 **대수적으로 정확히** 재현; 둘 다 closed-form
  경로평균 Jacobian).
- 따라서 큰 잡음원 중 이 조건을 만족하는 것들이 곧 모듈이 된다:
  **Bilinear**(Q@K·A@V·GeGLU, 0.15~0.83) / **SM**(GELU, 0.59~0.72) / **FrLN**(norm, 0.25~0.31; 정확
  corrector 가 없어 분모 detach 라는 대체 보정 — 그래서 아키텍처 의존).
- softcap(0.006)은 무시가능, 선형 연산(projection·rotary·residual)은 1차 오차가 0 이라 대상 아님.

### E-2. 각 도구가 **어떤 오차**를 해결하나 (§1a 분류와의 대응)

| 도구 | 겨누는 오차 (§1a) | 어느 op | 왜 그게 정확한가 | **못 잡는 것** |
|---|---|---|---|---|
| **Bilinear** (midpoint) | ①의 **cross** `Δx·Δy` | Q@K, A@V, GeGLU gate·up | `Δx·(y+y*)/2 + (x+x*)/2·Δy = x*y*−xy` — **대수적 항등** | 곡률, ②, ③ |
| **SM** (secant) | ①의 **곡률** (1-D element-wise) | GELU / SiLU | `[φ(g*)−φ(g)]/Δg · Δg = φ(g*)−φ(g)` — **정의상 정확** | cross, multi-D 곡률, ②, ③ |
| **FrLN** (freeze) | norm 의 ① (**근사**) | LayerNorm / RMSNorm | multi-D 라 정확 corrector **없음** — 분모 `∂σ/∂x` 항 detach 라는 대체 보정 | 정확성 미보장(아키텍처 의존: RMS 0.31→0.11 개선 / LN 0.25→0.66 악화), ②, ③ |
| **input-IG** (Z-step) | ①의 **모든 성분**, 특히 closed-form 이 없는 **multi-D 곡률(softmax)** | 경로 위 **전부** | 경로평균 Jacobian (Z→∞ 수렴) | 비용 Z×, 유한 Z 는 근사; ③은 부분적으로만 |

**핵심 대응 관계**
- **cross 는 bilinear op 에서만** 생기고 (`z=x·y` 는 각 변수엔 2차 도함수 0), **midpoint 가 대수적으로 정확히** 제거 → Bilinear.
- **곡률은 입력 1개짜리 비선형**에서 생기는데, **1-D element-wise 면 secant 가 정확** → SM(GELU).
  같은 곡률이어도 **multi-D(softmax·norm)면 secant 가 원리적으로 불가** → 적분(IG) 또는 대체 보정(FrLN).
- 그래서 **모듈 3개 = "closed-form 이 존재하는 ① 성분"**, **IG = "closed-form 이 없는 ① 성분"** 이라는 분업이다.

**어떤 도구도 안 잡는 것 (정직하게)**
- **② downstream 혼합·증폭**: 국소 보정은 그 op 의 오차만 없앨 뿐, 그 뒤 경로에서의 증폭·op 간 결합은 그대로.
- **③ self-repair**: 여러 op 을 거친 **기능적 대체**(backup head 등)라 국소 Jacobian 교체로는 접근 불가.
  (FrLN 이 norm 매개 self-repair 와 같은 항을 건드리지만, 실측상 실제 모델 대비 오차를 줄이지 못함 — §4.2.)
- 이 ②③ 잔차가 곧 **logit-level 오차의 대부분**(component ablation 56% / 2-edge joint 49.6% 중 cross 10.1% 를
  뺀 나머지)이고, op-gap≈0 이 end-to-end 충실도를 보장하지 못하는 이유이자
  남은 future work 이다.

### F. input-IG 의 역할 — closed-form 이 못 닿는 곳

- **무엇**: 입력 임베딩을 보간해 `(1−α)·clean + α·corrupt` 의 Z 지점에서 forward+backward 하고 gradient 를
  평균 (`tracer.py`: `integrated_embeds = (1-alpha)*clean_embeds + alpha*corrupt_embeds`). = **input-level IG**.
- **왜 필요**: softmax 는 element-wise 도 bilinear 도 아니라(Jacobian 이 모든 key 를 결합) **closed-form exact
  corrector 가 원리적으로 없다.** 그런데 **최대 잡음원**(op-gap 1.46/1.12). 경로 적분만이 해법이다.
- **효과 (측정)**: softmax op-gap 이 Z 에 따라 수렴 — 1.46 → 0.41(Z=1) → 0.011(Z=5) → 0.0005(Z=20) (GPT-2).
- **역할 분담**:

| | closed-form 모듈 (Bilinear/SM) | input-IG |
|---|---|---|
| 비용 | **1×** | **Z×** |
| 정확도 | **정확** (≈1e-7) | 근사 (Z↑ 로 수렴) |
| 적용 범위 | element-wise·bilinear op **만** | **모든** op (경로 위 전부) |

→ 둘은 **경쟁이 아니라 상보**다. closed-form 모듈이 자기 op 을 1× 에 정확히 처리해두면, IG 는 남은
비-closed-form 잔차(주로 softmax)만 감당하면 되므로 **작은 Z 로 충분**해진다. 이것이 최강 조합
`eap_ig_5 + igbilin + secmlp + frnorm` 의 설계 근거이고, 4개 잡음원 클래스에 4개 도구가 1:1 대응한다.

> 단서: §3.4 의 softmax IG 수렴은 **op 자체 입력을 직선 보간**한 이상화 측정이다. 실제 input-IG 는 임베딩에서
> 보간하므로 중간 활성이 모델 고유의(비직선) 경로를 따른다 — **정신은 같은 경로평균 Jacobian 이지만 수치가
> 동일하진 않다.** 즉 "softmax 는 적분으로 보정 가능하다"는 성질의 증명이지, input-IG 의 Z 별 성능 예측치는 아니다.

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

### 3.4 완전 noise budget — 미보정 비선형(softmax, softcap) 포함

"왜 이 세 곳을 골랐나"에 답하려면 **모듈이 안 푸는 비선형까지** 정량화해야 한다. softmax·softcap 추가:

| 비선형 op | gpt2 EAP op-gap | gemma2 EAP op-gap | corrector | 보정 후 |
|---|---|---|---|---|
| **softmax** | **1.4592** | **1.1198** | IG (Z-step) | Z=5: 0.0106 / 0.0074 |
| Q@K (bilinear) | 0.6591 | 0.8260 | midpoint (1×) | ~1e-6 |
| GELU (MLP act) | 0.7182 | 0.5947 | secant (1×) | ~1e-7 |
| GeGLU gate·up (bilinear) | — | 0.6843 | midpoint (1×) | ~1e-7 |
| A@V (bilinear) | 0.2656 | 0.1480 | midpoint (1×) | ~1e-7 |
| Norm (LN/RMS) | 0.2482 | 0.3110 | freeze | LN 0.656 / RMS 0.108 |
| softcap (gemma tanh) | — | 0.0057 | secant (1×) | 7.2e-10 |

**softmax IG 수렴** (gpt2 / gemma2):

| | EAP tangent | IG Z=1 | Z=2 | Z=5 | Z=10 | Z=20 |
|---|---|---|---|---|---|---|
| gpt2 | 1.4592 | 0.4080 | 0.1232 | 0.0106 | 0.0022 | 0.0005 |
| gemma2 | 1.1198 | 0.3757 | 0.0829 | 0.0074 | 0.0017 | 0.0004 |

**핵심 — 잡음원이 두 부류로 갈린다:**
- **closed-form exact (1× 비용)**: bilinear(Q@K·A@V·GeGLU)=midpoint, MLP-act(GELU)=secant, softcap=secant → 모듈이 1-step 에 정확히(≈1e-7) 제거. **이 셋을 고른 이유 = closed-form 으로 공짜 정확 보정 가능.**
- **integration-only (Z× 비용)**: softmax = **최대 잡음원(1.1~1.5)인데 closed-form 없음** (multi-D 결합) → Z-step IG 필요 (1.46→0.011 @Z=5). = `--integration-steps`(eap_ig_5) 로 처리.
- **norm**: freeze, 아키텍처 의존.
- **negligible**: softcap(0.006) — 거의 선형, 사실상 무시가능.

→ CPR 최강 method `eap_ig_5_igbilin_frnorm_secmlp` 는 우연이 아니라 **4개 잡음원 클래스를 각 올바른 도구로 커버**: ig_5→softmax, igbilin→bilinear, secmlp→GELU, frnorm→norm.

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

### 4.4b 잡음원 선택의 원리 (§3.4 budget 기반)

- **softmax 가 최대 잡음원**(1.1~1.5, 1.0 초과 = 오차 > 참변화, 극도로 포화)이지만 element-wise/bilinear 가
  아니라 **closed-form exact corrector 가 없다.** → Z-step IG 로만 보정(1.46→0.011 @Z=5), 즉 Z× 비용.
- 우리 세 모듈이 겨눈 bilinear·MLP-act·softcap 은 **1× 비용에 정확 보정** 가능 (midpoint/secant 가 대수적 exact).
  → **"왜 이 셋"의 답: 크기(상위) + closed-form tractability.** softmax 는 IG(integration-steps)로 상보적으로 처리.
- 전체 method 조합이 budget 으로 설명됨: ig_5(softmax) + igbilin(bilinear) + secmlp(GELU) + frnorm(norm).

### 4.4 한계 (정직하게)

- op-gap 은 **op 하나의 비선형을 isolate** 한 지표. op 출력이 downstream 비선형을 통과하며 받는 추가 왜곡
  (self-repair)은 **배제**된다. 따라서 **op-gap ≈ 0 이어도 최종 logit-level / single-edge AP fidelity 는 보장 안 됨**
  (앞 실험에서 모듈이 logit-level 갭은 못 줄였고, 미보정 self-repair 잔차가 지배 — component 56% / joint 49.6%).
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
| `scripts/oplevel_softmax_gpt2_ioi.py` | GPT-2 softmax (EAP tangent + IG Z=1..20) |
| `scripts/oplevel_softmax_gemma2_ioi.py` | gemma2 softmax + softcap |

(참고: single-edge ε / joint-AP / SM-verify 는 별도 실험 — edge 샘플링 기반, op-gap 과 다름.)
