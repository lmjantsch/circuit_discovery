# LAG (Local Approximation Gap) — 4 models × 2 tasks

`opgap_report.md` 의 op-gap 지표를 **LAG (Local Approximation Gap)** 로 명명하고
GPT-2 small · Qwen2.5-0.5B · Gemma-2-2B · Llama-3.1-8B × IOI · MCQA 로 확장한 결과 (§2, §3.1–3.8), 이어서 ARC-easy · ARC-challenge · arithmetic_addition · arithmetic_subtraction 을 추가한 결과 (§2b, §3.9). MIB circuit track 의 12 cell 을 모두 포함하고, 비교를 위해 비-MIB cell 12개를 참고용(†)으로 함께 쟀다.
정의·동기·한계는 `opgap_report.md` §1–§2 를 따르고, 여기서는 정의 요약 + 4×2 결과 + 아키텍처/태스크 비교만 다룬다.

---

## 1. 정의 (요약)

비선형 op `f` 하나를 고립시켜, clean·corrupt 두 실제 forward 의 입력 활성을 그대로 넣었을 때

```
LAG(Ê) = sqrt( Σ ‖Ê − Δz_true‖²_F / Σ ‖Δz_true‖²_F ),   Δz_true = f(x_corrupt) − f(x_clean)
```

- **EAP LAG**: `Ê = J_f(x_clean)·Δx` — EAP 가 backward 에서 쓰는 clean-point Jacobian 을 forward(JVP)로 적용한 1차 근사가 남기는 상대 RMS 오차. **= 그 op 이 EAP 에 주입하는 고차 잡음의 크기.**
- **module LAG**: `Ê` 를 모듈 corrector(midpoint / secant / frozen / IG-Z)로 바꾼 뒤 남는 오차.
- **pert** (신규 컬럼): `‖Δz_true‖ / ‖z_clean‖` — 그 op 출력이 clean→corrupt 에서 상대적으로 얼마나 움직였나. 2차항은 섭동의 제곱이므로 상대 LAG 는 pert 와 함께 커진다. 태스크 간 차이를 해석하는 데 쓴다.
- 합산은 정렬된 100 example × 전 layer × head × position × hidden dim. 예시당 forward 2회, 예시 간 정렬은 clean/corrupt 토큰 길이 동일 조건 (8 cell 모두 100/100).
- 규약은 이전과 동일: full clean→corrupt 전이, op 고립(downstream 재전파 없음 → self-repair 배제), edge 샘플링 없음.

### 1.1 아키텍처별 op 매핑

| op | 종류 | corrector | gpt2 | qwen2.5 | gemma2 | llama3 |
|---|---|---|---|---|---|---|
| Q@K | bilinear | midpoint | c_attn 분해, 12 head | rotary, GQA 14Q/2KV, qkv bias | rotary, GQA 8Q/4KV | rotary(llama3 scaling), GQA 32Q/8KV |
| Q@K row-centered (Q@K-c) | bilinear | midpoint | causal key 위에서 row 평균 제거 — softmax 가 실제로 보는 부분 (§3.4) | 〃 | 〃 | 〃 |
| softmax | multi-D | IG(Z) | scale 1/8 | 1/8 | `query_pre_attn_scalar^-½`, softcap 뒤 | 1/√128 |
| softcap | element-wise | secant | — | — | 50·tanh(·/50) | — |
| A@V | bilinear | midpoint | ○ | ○ | ○ | ○ |
| MLP act | element-wise | secant | gelu_new | SiLU | gelu_tanh | SiLU |
| gate·up | bilinear | midpoint | — | ○ | ○ | ○ |
| norm | multi-D | freeze | LayerNorm ×25 | RMSNorm `w·x·r` ×49 | RMSNorm `(1+w)·x·r` ×105 | RMSNorm `w·x·r` ×65 |

Q@K 는 rotary 적용·GQA repeat 뒤의 head 별 raw matmul (scale 은 비율에서 약분). qwen2.5-0.5B 는 sliding window 비활성이라 causal mask 만 적용.
dtype: gpt2·qwen2.5 fp32, gemma2·llama3 bf16 forward → gap 계산은 전부 fp32 (이전 관례). MIB 에 없는 cell(§1.2 의 †)은 CPR 대응값이 없는 **참고용**.

**회귀 검증**: 통합 스크립트(`scripts/lag_opgap.py`)로 gpt2/ioi, gemma2/ioi 를 재실행해 `opgap_report.md` §3 의 전 수치(소수 4자리, IG Z=1..20 포함)를 그대로 재현함.

### 1.2 커버리지 — MIB circuit track 대비

| task | MIB cell (모델) | LAG | 비-MIB 참고 cell († 표기, CPR 대응값 없음) |
|---|---|---|---|
| ioi | gpt2 · qwen2.5 · gemma2 · llama3 | 4/4 | — |
| mcqa | qwen2.5 · gemma2 · llama3 | 3/3 | gpt2 |
| arc_easy | gemma2 · llama3 | 2/2 | gpt2 · qwen2.5 |
| arc_challenge | llama3 | 1/1 | gpt2 · qwen2.5 · gemma2 |
| arithmetic_addition | llama3 | 1/1 | gpt2 · qwen2.5 · gemma2 |
| arithmetic_subtraction | llama3 | 1/1 | gpt2 · qwen2.5 · gemma2 |
| ioi_interpbench | interpbench | 미측정 (합성 모델, HookedTransformer 전용) | — |

총 24 cell = MIB 12 + 참고 12. 전 cell 100/100 정렬.

---

## 2. 결과 테이블

요약: T1 pooled EAP LAG · T1b layer-median · T2 모듈 잔차 · T3 섭동 크기 · T4 softmax IG · T5 norm · T6 태스크 비 · T7 메타데이터(‖Δq‖/‖q‖, ‖Δk‖/‖k‖ 포함).

<!-- LAG-TABLES-START -->
† = MIB circuit track 에 없는 참고 cell (CPR 대응값 없음). 나머지는 MIB cell.

### T1. EAP LAG — noise budget (first-order relative error per op)

행 = op (큰 잡음원 순), 열 = model/task. 값 = `‖J_clean·Δ − Δz_true‖ / ‖Δz_true‖` (전 example·layer·head·pos·dim RMS). `—` = 그 아키텍처에 없는 op.

| op | corrector | gpt2/ioi | qwen2.5/ioi | gemma2/ioi | llama3/ioi | gpt2/mcqa† | qwen2.5/mcqa | gemma2/mcqa | llama3/mcqa |
|---|---|---|---|---|---|---|---|---|---|
| **softmax** | IG (Z×) | 1.4592 | 1.4578 | 1.1198 | 1.1745 | 0.5021 | 0.5339 | 0.5363 | 0.6264 |
| **Q@K** | midpoint | 0.6591 | 0.0826 | 0.8260 | 0.7341 | 0.9281 | 0.0402 | 0.5707 | 0.3715 |
| **Q@K row-centered** | midpoint | 0.8428 | 0.5657 | 0.9436 | 0.8281 | 1.0144 | 0.6268 | 0.6394 | 0.4059 |
| **MLP act** | secant_cf (sm_fix) | 0.7182 | 0.4217 | 0.5947 | 0.2919 | 0.7793 | 0.5122 | 0.5206 | 0.3610 |
| **MLP act — shipped +SM** | origin f(x)/x | 0.7182 | 0.4217 | 0.5947 | 0.2919 | 0.7793 | 0.5122 | 0.5206 | 0.3610 |
| **gate·up** | midpoint | — | 0.4774 | 0.6843 | 0.4268 | — | 0.5615 | 0.6476 | 0.5353 |
| **A@V** | midpoint | 0.2656 | 0.2398 | 0.1480 | 0.2701 | 0.4768 | 0.4567 | 0.4273 | 0.5609 |
| **norm** | freeze | 0.2482 | 0.3084 | 0.3110 | 0.2634 | 0.2742 | 0.4228 | 0.3212 | 0.4174 |
| **softcap** | secant | — | — | 0.0057 | — | — | — | 0.0086 | — |

### T1b. EAP LAG — per-layer median (robust to norm-dominant layers)

T1 은 전 layer 를 norm 으로 합산(pooled)하므로 `‖Δz_true‖` 가 큰 소수 layer 가 지배할 수 있다. 여기서는 layer 별 LAG 의 중앙값 (norm 은 final norm 포함).

| op | gpt2/ioi | qwen2.5/ioi | gemma2/ioi | llama3/ioi | gpt2/mcqa† | qwen2.5/mcqa | gemma2/mcqa | llama3/mcqa |
|---|---|---|---|---|---|---|---|---|
| **softmax** | 0.8176 | 0.4449 | 0.6865 | 0.7130 | 0.5059 | 0.5054 | 0.3982 | 0.5897 |
| **Q@K** | 0.4970 | 0.2821 | 0.6677 | 0.6469 | 0.7563 | 0.2395 | 0.3634 | 0.2839 |
| **Q@K row-centered** | 0.4710 | 0.3572 | 0.8667 | 0.8358 | 0.9340 | 0.3203 | 0.4130 | 0.3290 |
| **MLP act** | 0.6982 | 0.4069 | 0.4225 | 0.2670 | 0.7826 | 0.4748 | 0.3740 | 0.3027 |
| **MLP act — shipped +SM** | 0.6982 | 0.4069 | 0.4225 | 0.2670 | 0.7826 | 0.4748 | 0.3740 | 0.3027 |
| **gate·up** | — | 0.5777 | 0.6542 | 0.6364 | — | 0.5896 | 0.5762 | 0.6062 |
| **A@V** | 0.2331 | 0.2096 | 0.1302 | 0.2202 | 0.3370 | 0.3685 | 0.3098 | 0.5246 |
| **norm** | 0.3078 | 0.3285 | 0.2068 | 0.2595 | 0.3273 | 0.4060 | 0.2664 | 0.4114 |
| **softcap** | — | — | 0.0021 | — | — | — | 0.0019 | — |

### T2. module LAG — residual after the corrector (softmax: IG Z=5)

| op | corrector | gpt2/ioi | qwen2.5/ioi | gemma2/ioi | llama3/ioi | gpt2/mcqa† | qwen2.5/mcqa | gemma2/mcqa | llama3/mcqa |
|---|---|---|---|---|---|---|---|---|---|
| **softmax** | IG (Z×) | 0.0106 | 0.0215 | 0.0074 | 0.0084 | 0.0027 | 0.0032 | 0.0059 | 0.0032 |
| **Q@K** | midpoint | 1.2e-06 | 7.6e-07 | 1.6e-06 | 1.9e-06 | 1.3e-06 | 5.5e-07 | 1.4e-06 | 1.1e-06 |
| **Q@K row-centered** | midpoint | 1.5e-06 | 6.1e-06 | 1.7e-06 | 2.0e-06 | 1.8e-06 | 9.6e-06 | 1.6e-06 | 1.3e-06 |
| **MLP act** | secant_cf (sm_fix) | 1.0e-09 | 6.3e-10 | 6.8e-15 | 6.4e-15 | 3.8e-10 | 1.7e-10 | 1.7e-15 | 1.3e-15 |
| **MLP act — shipped +SM** | origin f(x)/x | 0.7997 | 0.5010 | 0.6829 | 0.3438 | 0.8178 | 0.5068 | 0.6073 | 0.3379 |
| **gate·up** | midpoint | — | 1.2e-07 | 1.1e-07 | 1.3e-07 | — | 7.8e-08 | 7.7e-08 | 7.3e-08 |
| **A@V** | midpoint | 2.2e-07 | 1.9e-07 | 2.1e-07 | 1.9e-07 | 2.0e-07 | 2.0e-07 | 2.1e-07 | 1.5e-07 |
| **norm** | freeze | 0.6555 | 0.1387 | 0.1082 | 0.0907 | 0.5284 | 0.2537 | 0.2503 | 0.1656 |
| **softcap** | secant | — | — | 7.2e-10 | — | — | — | 1.6e-10 | — |

### T3. perturbation size at the op — `‖Δz_true‖ / ‖z_clean‖`

clean→corrupt 전이가 각 op 출력을 상대적으로 얼마나 움직이는가. 2차항은 섭동의 제곱이라 상대 LAG 는 이 값과 함께 커진다.

| op | gpt2/ioi | qwen2.5/ioi | gemma2/ioi | llama3/ioi | gpt2/mcqa† | qwen2.5/mcqa | gemma2/mcqa | llama3/mcqa |
|---|---|---|---|---|---|---|---|---|
| **softmax** | 0.0933 | 0.0990 | 0.0944 | 0.0532 | 0.1527 | 0.1933 | 0.1696 | 0.1339 |
| **Q@K** | 0.1113 | 0.0556 | 0.1220 | 0.0846 | 0.1495 | 0.1672 | 0.2068 | 0.1898 |
| **Q@K row-centered** | 0.1685 | 0.2108 | 0.1612 | 0.1329 | 0.2185 | 0.3195 | 0.2662 | 0.2936 |
| **MLP act** | 0.1706 | 0.1340 | 0.1372 | 0.0913 | 0.3758 | 0.3438 | 0.3137 | 0.2949 |
| **MLP act — shipped +SM** | 0.1706 | 0.1340 | 0.1372 | 0.0913 | 0.3758 | 0.3438 | 0.3137 | 0.2949 |
| **gate·up** | — | 0.0336 | 0.2041 | 0.0957 | — | 0.1233 | 0.4605 | 0.3340 |
| **A@V** | 0.2148 | 0.2611 | 0.2184 | 0.2690 | 0.3987 | 0.4189 | 0.3586 | 0.5255 |
| **norm** | 0.0527 | 0.1730 | 0.1622 | 0.1898 | 0.1889 | 0.4193 | 0.3603 | 0.4873 |
| **softcap** | — | — | 0.1218 | — | — | — | 0.2065 | — |

### T4. softmax IG convergence (LAG vs Z)

| cell | EAP tangent | Z=1 | Z=2 | Z=5 | Z=10 | Z=20 |
|---|---|---|---|---|---|---|
| gpt2/ioi | 1.4592 | 0.4080 | 0.1232 | 0.0106 | 0.0022 | 0.0005 |
| qwen2.5/ioi | 1.4578 | 0.4638 | 0.1550 | 0.0215 | 0.0044 | 0.0010 |
| gemma2/ioi | 1.1198 | 0.3757 | 0.0829 | 0.0074 | 0.0017 | 0.0004 |
| llama3/ioi | 1.1745 | 0.2688 | 0.0819 | 0.0084 | 0.0019 | 0.0005 |
| gpt2/mcqa† | 0.5021 | 0.1410 | 0.0212 | 0.0027 | 0.0007 | 0.0002 |
| qwen2.5/mcqa | 0.5339 | 0.1006 | 0.0233 | 0.0032 | 0.0008 | 0.0002 |
| gemma2/mcqa | 0.5363 | 0.1285 | 0.0255 | 0.0059 | 0.0017 | 0.0005 |
| llama3/mcqa | 0.6264 | 0.1120 | 0.0213 | 0.0032 | 0.0008 | 0.0002 |

### T5. norm — EAP (J_full) vs FrLN (J_frozen)

| cell | norm type | #norms | EAP LAG | FrLN LAG | FrLN/EAP | verdict |
|---|---|---|---|---|---|---|
| gpt2/ioi | LN | 25 | 0.2482 | 0.6555 | 2.64× | 악화 |
| qwen2.5/ioi | RMS(w) | 49 | 0.3084 | 0.1387 | 0.45× | 개선 |
| gemma2/ioi | RMS(1+w) | 105 | 0.3110 | 0.1082 | 0.35× | 개선 |
| llama3/ioi | RMS(w) | 65 | 0.2634 | 0.0907 | 0.34× | 개선 |
| gpt2/mcqa† | LN | 25 | 0.2742 | 0.5284 | 1.93× | 악화 |
| qwen2.5/mcqa | RMS(w) | 49 | 0.4228 | 0.2537 | 0.60× | 개선 |
| gemma2/mcqa | RMS(1+w) | 105 | 0.3212 | 0.2503 | 0.78× | 개선 |
| llama3/mcqa | RMS(w) | 65 | 0.4174 | 0.1656 | 0.40× | 개선 |

### T6. task effect — mcqa / ioi ratio of EAP LAG and of perturbation size

| op | gpt2 LAG | qwen2.5 LAG | gemma2 LAG | llama3 LAG | gpt2 pert | qwen2.5 pert | gemma2 pert | llama3 pert |
|---|---|---|---|---|---|---|---|---|
| **softmax** | 0.34× | 0.37× | 0.48× | 0.53× | 1.64× | 1.95× | 1.80× | 2.52× |
| **Q@K** | 1.41× | 0.49× | 0.69× | 0.51× | 1.34× | 3.01× | 1.70× | 2.24× |
| **Q@K row-centered** | 1.20× | 1.11× | 0.68× | 0.49× | 1.30× | 1.52× | 1.65× | 2.21× |
| **MLP act** | 1.09× | 1.21× | 0.88× | 1.24× | 2.20× | 2.57× | 2.29× | 3.23× |
| **MLP act — shipped +SM** | 1.09× | 1.21× | 0.88× | 1.24× | 2.20× | 2.57× | 2.29× | 3.23× |
| **gate·up** | — | 1.18× | 0.95× | 1.25× | — | 3.67× | 2.26× | 3.49× |
| **A@V** | 1.80× | 1.90× | 2.89× | 2.08× | 1.86× | 1.60× | 1.64× | 1.95× |
| **norm** | 1.10× | 1.37× | 1.03× | 1.58× | 3.58× | 2.42× | 2.22× | 2.57× |
| **softcap** | — | — | 1.51× | — | — | — | 1.70× | — |

### T7. run metadata

| cell | model id | dtype | layers | heads (Q/KV) | head_dim | act | norm | aligned ex. | time | ‖Δq‖/‖q‖ | ‖Δk‖/‖k‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt2/ioi | `openai-community/gpt2` | float32 | 12 | 12/12 | 64 | gelu_new | LN | 100/100 | 12s | 0.1644 | 0.0969 |
| qwen2.5/ioi | `Qwen/Qwen2.5-0.5B` | float32 | 24 | 14/2 | 64 | SiLU | RMS(w) | 100/100 | 26s | 0.0717 | 0.0258 |
| gemma2/ioi | `google/gemma-2-2b` | bfloat16 | 26 | 8/4 | 256 | gelu_tanh | RMS(1+w) | 100/100 | 37s | 0.1581 | 0.1474 |
| llama3/ioi | `meta-llama/Llama-3.1-8B` | bfloat16 | 32 | 32/8 | 128 | SiLU | RMS(w) | 100/100 | 37s | 0.1102 | 0.1165 |
| gpt2/mcqa† | `openai-community/gpt2` | float32 | 12 | 12/12 | 64 | gelu_new | LN | 100/100 | 12s | 0.2951 | 0.1754 |
| qwen2.5/mcqa | `Qwen/Qwen2.5-0.5B` | float32 | 24 | 14/2 | 64 | SiLU | RMS(w) | 100/100 | 27s | 0.1408 | 0.0529 |
| gemma2/mcqa | `google/gemma-2-2b` | bfloat16 | 26 | 8/4 | 256 | gelu_tanh | RMS(1+w) | 100/100 | 38s | 0.2722 | 0.2424 |
| llama3/mcqa | `meta-llama/Llama-3.1-8B` | bfloat16 | 32 | 32/8 | 128 | SiLU | RMS(w) | 100/100 | 37s | 0.2343 | 0.2305 |

<!-- LAG-TABLES-END -->

---

## 2b. 결과 테이블 — ARC-easy · ARC-challenge · arithmetic_addition · arithmetic_subtraction

같은 스크립트·규약. arithmetic 은 `random_counterfactual`(피연산자 교환), ARC 는 `symbol_counterfactual`. 열 약어: arcE / arcC / arith+ / arith−. † = 비-MIB 참고 cell.

<!-- LAG-TABLES-ARC-START -->
† = MIB circuit track 에 없는 참고 cell (CPR 대응값 없음). 나머지는 MIB cell.

### T1. EAP LAG — noise budget (first-order relative error per op)

행 = op (큰 잡음원 순), 열 = model/task. 값 = `‖J_clean·Δ − Δz_true‖ / ‖Δz_true‖` (전 example·layer·head·pos·dim RMS). `—` = 그 아키텍처에 없는 op.

| op | corrector | gpt2/arcE† | qwen2.5/arcE† | gemma2/arcE | llama3/arcE | gpt2/arcC† | qwen2.5/arcC† | gemma2/arcC† | llama3/arcC | gpt2/arith+† | qwen2.5/arith+† | gemma2/arith+† | llama3/arith+ | gpt2/arith−† | qwen2.5/arith−† | gemma2/arith−† | llama3/arith− |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **softmax** | IG (Z×) | 0.5464 | 0.5141 | 0.5206 | 0.5684 | 0.5454 | 0.5182 | 0.5217 | 0.5679 | 0.3704 | 0.4874 | 0.3899 | 0.5139 | 0.3802 | 0.5395 | 0.4550 | 0.5568 |
| **Q@K** | midpoint | 0.5778 | 0.0228 | 0.4092 | 0.2645 | 0.5314 | 0.0216 | 0.3797 | 0.2487 | 1.1290 | 0.0720 | 0.8793 | 0.5950 | 1.1235 | 0.1226 | 1.0348 | 0.6025 |
| **Q@K row-centered** | midpoint | 0.6687 | 0.4070 | 0.4730 | 0.2960 | 0.6164 | 0.3791 | 0.4374 | 0.2782 | 1.4036 | 0.8463 | 0.9424 | 0.6314 | 1.4696 | 1.0553 | 1.0568 | 0.6562 |
| **MLP act** | secant_cf (sm_fix) | 0.7483 | 0.4802 | 0.5151 | 0.3299 | 0.7470 | 0.4782 | 0.5096 | 0.3297 | 0.5463 | 0.4480 | 0.4670 | 0.3224 | 0.5388 | 0.4452 | 0.5204 | 0.3077 |
| **MLP act — shipped +SM** | origin f(x)/x | 0.7483 | 0.4802 | 0.5151 | 0.3299 | 0.7470 | 0.4782 | 0.5096 | 0.3297 | 0.5463 | 0.4480 | 0.4670 | 0.3224 | 0.5388 | 0.4452 | 0.5204 | 0.3077 |
| **gate·up** | midpoint | — | 0.5250 | 0.6155 | 0.5168 | — | 0.5239 | 0.6178 | 0.5188 | — | 0.5098 | 0.7062 | 0.5852 | — | 0.5398 | 0.7050 | 0.5225 |
| **A@V** | midpoint | 0.4623 | 0.4149 | 0.4273 | 0.5106 | 0.4646 | 0.4210 | 0.4247 | 0.5028 | 0.2874 | 0.3179 | 0.3531 | 0.5430 | 0.2444 | 0.2956 | 0.3412 | 0.5000 |
| **norm** | freeze | 0.3533 | 0.3883 | 0.2865 | 0.3784 | 0.3267 | 0.3871 | 0.2923 | 0.3813 | 0.2012 | 0.2977 | 0.2626 | 0.2878 | 0.2169 | 0.2850 | 0.3194 | 0.2649 |
| **softcap** | secant | — | — | 0.0060 | — | — | — | 0.0060 | — | — | — | 0.0040 | — | — | — | 0.0053 | — |

### T1b. EAP LAG — per-layer median (robust to norm-dominant layers)

T1 은 전 layer 를 norm 으로 합산(pooled)하므로 `‖Δz_true‖` 가 큰 소수 layer 가 지배할 수 있다. 여기서는 layer 별 LAG 의 중앙값 (norm 은 final norm 포함).

| op | gpt2/arcE† | qwen2.5/arcE† | gemma2/arcE | llama3/arcE | gpt2/arcC† | qwen2.5/arcC† | gemma2/arcC† | llama3/arcC | gpt2/arith+† | qwen2.5/arith+† | gemma2/arith+† | llama3/arith+ | gpt2/arith−† | qwen2.5/arith−† | gemma2/arith−† | llama3/arith− |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **softmax** | 0.5386 | 0.5029 | 0.3877 | 0.5701 | 0.5506 | 0.4942 | 0.3964 | 0.5621 | 0.3551 | 0.3382 | 0.3680 | 0.3699 | 0.3107 | 0.2656 | 0.3796 | 0.3310 |
| **Q@K** | 0.4546 | 0.1587 | 0.2439 | 0.1690 | 0.4219 | 0.1537 | 0.2274 | 0.1647 | 0.8260 | 0.1904 | 0.5511 | 0.2615 | 0.7573 | 0.1639 | 0.5631 | 0.2634 |
| **Q@K row-centered** | 0.6123 | 0.2089 | 0.2719 | 0.2143 | 0.5572 | 0.2004 | 0.2529 | 0.2070 | 0.7536 | 0.2720 | 0.5940 | 0.3445 | 0.6433 | 0.2455 | 0.5808 | 0.3047 |
| **MLP act** | 0.7455 | 0.4606 | 0.3415 | 0.2565 | 0.7453 | 0.4556 | 0.3416 | 0.2560 | 0.4879 | 0.4212 | 0.3978 | 0.2822 | 0.4516 | 0.4052 | 0.4113 | 0.2774 |
| **MLP act — shipped +SM** | 0.7455 | 0.4606 | 0.3415 | 0.2565 | 0.7453 | 0.4556 | 0.3416 | 0.2560 | 0.4879 | 0.4212 | 0.3978 | 0.2822 | 0.4516 | 0.4052 | 0.4113 | 0.2774 |
| **gate·up** | — | 0.5666 | 0.5320 | 0.5144 | — | 0.5642 | 0.5275 | 0.5087 | — | 0.5372 | 0.6384 | 0.6886 | — | 0.6091 | 0.6120 | 0.6670 |
| **A@V** | 0.3608 | 0.3544 | 0.2900 | 0.4728 | 0.3646 | 0.3577 | 0.2903 | 0.4641 | 0.2574 | 0.2926 | 0.3200 | 0.3387 | 0.2183 | 0.2108 | 0.3071 | 0.2702 |
| **norm** | 0.3293 | 0.3865 | 0.2447 | 0.3646 | 0.3267 | 0.3812 | 0.2513 | 0.3662 | 0.2001 | 0.3554 | 0.2877 | 0.2883 | 0.1992 | 0.3550 | 0.3003 | 0.2785 |
| **softcap** | — | — | 0.0019 | — | — | — | 0.0020 | — | — | — | 0.0019 | — | — | — | 0.0018 | — |

### T2. module LAG — residual after the corrector (softmax: IG Z=5)

| op | corrector | gpt2/arcE† | qwen2.5/arcE† | gemma2/arcE | llama3/arcE | gpt2/arcC† | qwen2.5/arcC† | gemma2/arcC† | llama3/arcC | gpt2/arith+† | qwen2.5/arith+† | gemma2/arith+† | llama3/arith+ | gpt2/arith−† | qwen2.5/arith−† | gemma2/arith−† | llama3/arith− |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **softmax** | IG (Z×) | 0.0033 | 0.0030 | 0.0053 | 0.0030 | 0.0035 | 0.0031 | 0.0050 | 0.0030 | 0.0017 | 0.0040 | 0.0019 | 0.0027 | 0.0017 | 0.0054 | 0.0027 | 0.0030 |
| **Q@K** | midpoint | 2.2e-06 | 5.0e-07 | 2.2e-06 | 2.0e-06 | 2.5e-06 | 5.1e-07 | 2.4e-06 | 2.1e-06 | 1.1e-06 | 4.4e-07 | 1.1e-06 | 1.1e-06 | 1.2e-06 | 6.9e-07 | 1.9e-06 | 1.3e-06 |
| **Q@K row-centered** | midpoint | 3.0e-06 | 1.0e-05 | 2.6e-06 | 2.4e-06 | 3.3e-06 | 1.0e-05 | 2.8e-06 | 2.5e-06 | 1.2e-06 | 6.8e-06 | 1.3e-06 | 1.2e-06 | 1.3e-06 | 7.0e-06 | 2.2e-06 | 1.4e-06 |
| **MLP act** | secant_cf (sm_fix) | 8.5e-10 | 4.7e-10 | 4.2e-15 | 5.3e-15 | 9.2e-10 | 4.6e-10 | 1.3e-14 | 6.6e-15 | 9.5e-10 | 4.3e-10 | 8.7e-15 | 0.0e+00 | 1.3e-09 | 7.3e-10 | 2.8e-15 | 0.0e+00 |
| **MLP act — shipped +SM** | origin f(x)/x | 0.7872 | 0.4932 | 0.6268 | 0.3339 | 0.7880 | 0.4923 | 0.6213 | 0.3338 | 0.7124 | 0.5045 | 0.5163 | 0.3401 | 0.7133 | 0.5288 | 0.5932 | 0.3388 |
| **gate·up** | midpoint | — | 9.3e-08 | 9.2e-08 | 9.1e-08 | — | 9.6e-08 | 9.6e-08 | 9.6e-08 | — | 2.6e-07 | 8.4e-08 | 1.2e-07 | — | 1.2e-07 | 8.2e-08 | 1.2e-07 |
| **A@V** | midpoint | 3.1e-07 | 3.5e-07 | 3.7e-07 | 2.6e-07 | 3.6e-07 | 3.9e-07 | 4.1e-07 | 2.9e-07 | 2.2e-07 | 1.5e-07 | 1.6e-07 | 1.5e-07 | 2.2e-07 | 2.0e-07 | 2.1e-07 | 1.6e-07 |
| **norm** | freeze | 0.4938 | 0.2406 | 0.2084 | 0.1850 | 0.5062 | 0.2425 | 0.2139 | 0.1907 | 0.5581 | 0.1282 | 0.1325 | 0.0951 | 0.5790 | 0.1280 | 0.1678 | 0.1090 |
| **softcap** | secant | — | — | 1.1e-09 | — | — | — | 1.3e-09 | — | — | — | 7.4e-10 | — | — | — | 9.4e-10 | — |

### T3. perturbation size at the op — `‖Δz_true‖ / ‖z_clean‖`

clean→corrupt 전이가 각 op 출력을 상대적으로 얼마나 움직이는가. 2차항은 섭동의 제곱이라 상대 LAG 는 이 값과 함께 커진다.

| op | gpt2/arcE† | qwen2.5/arcE† | gemma2/arcE | llama3/arcE | gpt2/arcC† | qwen2.5/arcC† | gemma2/arcC† | llama3/arcC | gpt2/arith+† | qwen2.5/arith+† | gemma2/arith+† | llama3/arith+ | gpt2/arith−† | qwen2.5/arith−† | gemma2/arith−† | llama3/arith− |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **softmax** | 0.1199 | 0.1421 | 0.1183 | 0.0878 | 0.1163 | 0.1411 | 0.1168 | 0.0865 | 0.0692 | 0.0894 | 0.1110 | 0.0541 | 0.0625 | 0.0754 | 0.1026 | 0.0463 |
| **Q@K** | 0.0788 | 0.1458 | 0.1166 | 0.0985 | 0.0717 | 0.1390 | 0.1117 | 0.0957 | 0.1134 | 0.0856 | 0.1686 | 0.1008 | 0.1031 | 0.0620 | 0.1553 | 0.0892 |
| **Q@K row-centered** | 0.1318 | 0.2219 | 0.1659 | 0.1849 | 0.1213 | 0.2147 | 0.1627 | 0.1862 | 0.1565 | 0.2104 | 0.1952 | 0.1343 | 0.1411 | 0.2257 | 0.1921 | 0.1197 |
| **MLP act** | 0.2958 | 0.2327 | 0.2065 | 0.1840 | 0.2834 | 0.2252 | 0.1992 | 0.1771 | 0.1925 | 0.2258 | 0.3064 | 0.1759 | 0.1764 | 0.1676 | 0.2632 | 0.1582 |
| **MLP act — shipped +SM** | 0.2958 | 0.2327 | 0.2065 | 0.1840 | 0.2834 | 0.2252 | 0.1992 | 0.1771 | 0.1925 | 0.2258 | 0.3064 | 0.1759 | 0.1764 | 0.1676 | 0.2632 | 0.1582 |
| **gate·up** | — | 0.1147 | 0.3332 | 0.2562 | — | 0.1170 | 0.3196 | 0.2540 | — | 0.0571 | 0.3805 | 0.1508 | — | 0.0462 | 0.3851 | 0.1371 |
| **A@V** | 0.3171 | 0.2979 | 0.2418 | 0.3521 | 0.2957 | 0.2896 | 0.2346 | 0.3374 | 0.2057 | 0.3677 | 0.3313 | 0.3696 | 0.1975 | 0.2774 | 0.2985 | 0.3488 |
| **norm** | 0.1560 | 0.3003 | 0.2329 | 0.3143 | 0.1385 | 0.2912 | 0.2277 | 0.3034 | 0.0544 | 0.2501 | 0.3244 | 0.2921 | 0.0466 | 0.2038 | 0.3080 | 0.2682 |
| **softcap** | — | — | 0.1165 | — | — | — | 0.1116 | — | — | — | 0.1685 | — | — | — | 0.1551 | — |

### T4. softmax IG convergence (LAG vs Z)

| cell | EAP tangent | Z=1 | Z=2 | Z=5 | Z=10 | Z=20 |
|---|---|---|---|---|---|---|
| gpt2/arcE† | 0.5464 | 0.1241 | 0.0280 | 0.0033 | 0.0008 | 0.0002 |
| qwen2.5/arcE† | 0.5141 | 0.0991 | 0.0218 | 0.0030 | 0.0007 | 0.0002 |
| gemma2/arcE | 0.5206 | 0.1238 | 0.0286 | 0.0053 | 0.0014 | 0.0004 |
| llama3/arcE | 0.5684 | 0.1021 | 0.0199 | 0.0030 | 0.0007 | 0.0002 |
| gpt2/arcC† | 0.5454 | 0.1212 | 0.0274 | 0.0035 | 0.0008 | 0.0002 |
| qwen2.5/arcC† | 0.5182 | 0.0998 | 0.0224 | 0.0031 | 0.0008 | 0.0002 |
| gemma2/arcC† | 0.5217 | 0.1251 | 0.0286 | 0.0050 | 0.0013 | 0.0003 |
| llama3/arcC | 0.5679 | 0.0998 | 0.0200 | 0.0030 | 0.0007 | 0.0002 |
| gpt2/arith+† | 0.3704 | 0.0732 | 0.0123 | 0.0017 | 0.0004 | 0.0001 |
| qwen2.5/arith+† | 0.4874 | 0.1003 | 0.0283 | 0.0040 | 0.0009 | 0.0002 |
| gemma2/arith+† | 0.3899 | 0.0649 | 0.0132 | 0.0019 | 0.0005 | 0.0001 |
| llama3/arith+ | 0.5139 | 0.0907 | 0.0209 | 0.0027 | 0.0007 | 0.0002 |
| gpt2/arith−† | 0.3802 | 0.0771 | 0.0138 | 0.0017 | 0.0004 | 0.0001 |
| qwen2.5/arith−† | 0.5395 | 0.1268 | 0.0406 | 0.0054 | 0.0013 | 0.0003 |
| gemma2/arith−† | 0.4550 | 0.1713 | 0.0207 | 0.0027 | 0.0007 | 0.0002 |
| llama3/arith− | 0.5568 | 0.0957 | 0.0219 | 0.0030 | 0.0007 | 0.0002 |

### T5. norm — EAP (J_full) vs FrLN (J_frozen)

| cell | norm type | #norms | EAP LAG | FrLN LAG | FrLN/EAP | verdict |
|---|---|---|---|---|---|---|
| gpt2/arcE† | LN | 25 | 0.3533 | 0.4938 | 1.40× | 악화 |
| qwen2.5/arcE† | RMS(w) | 49 | 0.3883 | 0.2406 | 0.62× | 개선 |
| gemma2/arcE | RMS(1+w) | 105 | 0.2865 | 0.2084 | 0.73× | 개선 |
| llama3/arcE | RMS(w) | 65 | 0.3784 | 0.1850 | 0.49× | 개선 |
| gpt2/arcC† | LN | 25 | 0.3267 | 0.5062 | 1.55× | 악화 |
| qwen2.5/arcC† | RMS(w) | 49 | 0.3871 | 0.2425 | 0.63× | 개선 |
| gemma2/arcC† | RMS(1+w) | 105 | 0.2923 | 0.2139 | 0.73× | 개선 |
| llama3/arcC | RMS(w) | 65 | 0.3813 | 0.1907 | 0.50× | 개선 |
| gpt2/arith+† | LN | 25 | 0.2012 | 0.5581 | 2.77× | 악화 |
| qwen2.5/arith+† | RMS(w) | 49 | 0.2977 | 0.1282 | 0.43× | 개선 |
| gemma2/arith+† | RMS(1+w) | 105 | 0.2626 | 0.1325 | 0.50× | 개선 |
| llama3/arith+ | RMS(w) | 65 | 0.2878 | 0.0951 | 0.33× | 개선 |
| gpt2/arith−† | LN | 25 | 0.2169 | 0.5790 | 2.67× | 악화 |
| qwen2.5/arith−† | RMS(w) | 49 | 0.2850 | 0.1280 | 0.45× | 개선 |
| gemma2/arith−† | RMS(1+w) | 105 | 0.3194 | 0.1678 | 0.53× | 개선 |
| llama3/arith− | RMS(w) | 65 | 0.2649 | 0.1090 | 0.41× | 개선 |

### T6. task effect — arith− / arcE ratio of EAP LAG and of perturbation size

| op | gpt2 LAG | qwen2.5 LAG | gemma2 LAG | llama3 LAG | gpt2 pert | qwen2.5 pert | gemma2 pert | llama3 pert |
|---|---|---|---|---|---|---|---|---|
| **softmax** | 0.70× | 1.05× | 0.87× | 0.98× | 0.52× | 0.53× | 0.87× | 0.53× |
| **Q@K** | 1.94× | 5.38× | 2.53× | 2.28× | 1.31× | 0.43× | 1.33× | 0.90× |
| **Q@K row-centered** | 2.20× | 2.59× | 2.23× | 2.22× | 1.07× | 1.02× | 1.16× | 0.65× |
| **MLP act** | 0.72× | 0.93× | 1.01× | 0.93× | 0.60× | 0.72× | 1.27× | 0.86× |
| **MLP act — shipped +SM** | 0.72× | 0.93× | 1.01× | 0.93× | 0.60× | 0.72× | 1.27× | 0.86× |
| **gate·up** | — | 1.03× | 1.15× | 1.01× | — | 0.40× | 1.16× | 0.54× |
| **A@V** | 0.53× | 0.71× | 0.80× | 0.98× | 0.62× | 0.93× | 1.23× | 0.99× |
| **norm** | 0.61× | 0.73× | 1.11× | 0.70× | 0.30× | 0.68× | 1.32× | 0.85× |
| **softcap** | — | — | 0.89× | — | — | — | 1.33× | — |

### T7. run metadata

| cell | model id | dtype | layers | heads (Q/KV) | head_dim | act | norm | aligned ex. | time | ‖Δq‖/‖q‖ | ‖Δk‖/‖k‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt2/arcE† | `openai-community/gpt2` | float32 | 12 | 12/12 | 64 | gelu_new | LN | 100/100 | 12s | 0.2166 | 0.1285 |
| qwen2.5/arcE† | `Qwen/Qwen2.5-0.5B` | float32 | 24 | 14/2 | 64 | SiLU | RMS(w) | 100/100 | 26s | 0.1002 | 0.0376 |
| gemma2/arcE | `google/gemma-2-2b` | bfloat16 | 26 | 8/4 | 256 | gelu_tanh | RMS(1+w) | 100/100 | 39s | 0.1769 | 0.1604 |
| llama3/arcE | `meta-llama/Llama-3.1-8B` | bfloat16 | 32 | 32/8 | 128 | SiLU | RMS(w) | 100/100 | 38s | 0.1449 | 0.1438 |
| gpt2/arcC† | `openai-community/gpt2` | float32 | 12 | 12/12 | 64 | gelu_new | LN | 100/100 | 13s | 0.2053 | 0.1217 |
| qwen2.5/arcC† | `Qwen/Qwen2.5-0.5B` | float32 | 24 | 14/2 | 64 | SiLU | RMS(w) | 100/100 | 27s | 0.0969 | 0.0362 |
| gemma2/arcC† | `google/gemma-2-2b` | bfloat16 | 26 | 8/4 | 256 | gelu_tanh | RMS(1+w) | 100/100 | 37s | 0.1706 | 0.1542 |
| llama3/arcC | `meta-llama/Llama-3.1-8B` | bfloat16 | 32 | 32/8 | 128 | SiLU | RMS(w) | 100/100 | 37s | 0.1403 | 0.1390 |
| gpt2/arith+† | `openai-community/gpt2` | float32 | 12 | 12/12 | 64 | gelu_new | LN | 100/100 | 12s | 0.1907 | 0.1185 |
| qwen2.5/arith+† | `Qwen/Qwen2.5-0.5B` | float32 | 24 | 14/2 | 64 | SiLU | RMS(w) | 100/100 | 26s | 0.0903 | 0.0356 |
| gemma2/arith+† | `google/gemma-2-2b` | bfloat16 | 26 | 8/4 | 256 | gelu_tanh | RMS(1+w) | 100/100 | 37s | 0.2570 | 0.2344 |
| llama3/arith+ | `meta-llama/Llama-3.1-8B` | bfloat16 | 32 | 32/8 | 128 | SiLU | RMS(w) | 100/100 | 37s | 0.1385 | 0.1417 |
| gpt2/arith−† | `openai-community/gpt2` | float32 | 12 | 12/12 | 64 | gelu_new | LN | 100/100 | 12s | 0.1785 | 0.1104 |
| qwen2.5/arith−† | `Qwen/Qwen2.5-0.5B` | float32 | 24 | 14/2 | 64 | SiLU | RMS(w) | 100/100 | 26s | 0.0807 | 0.0351 |
| gemma2/arith−† | `google/gemma-2-2b` | bfloat16 | 26 | 8/4 | 256 | gelu_tanh | RMS(1+w) | 100/100 | 37s | 0.2318 | 0.2120 |
| llama3/arith− | `meta-llama/Llama-3.1-8B` | bfloat16 | 32 | 32/8 | 128 | SiLU | RMS(w) | 100/100 | 36s | 0.1247 | 0.1266 |

<!-- LAG-TABLES-ARC-END -->

### 태스크 횡단 요약 (6 태스크)

<!-- LAG-SUMMARY-START -->
### T8. cross-task summary — pooled EAP LAG, min–max over the 4 models (softcap: gemma2 only)

| op | ioi | mcqa | arcE | arcC | arith+ | arith− |
|---|---|---|---|---|---|---|
| **softmax** | 1.12–1.46 | 0.50–0.63 | 0.51–0.57 | 0.52–0.57 | 0.37–0.51 | 0.38–0.56 |
| **Q@K** | 0.08–0.83 | 0.04–0.93 | 0.02–0.58 | 0.02–0.53 | 0.07–1.13 | 0.12–1.12 |
| **Q@K row-centered** | 0.57–0.94 | 0.41–1.01 | 0.30–0.67 | 0.28–0.62 | 0.63–1.40 | 0.66–1.47 |
| **MLP act** | 0.29–0.72 | 0.36–0.78 | 0.33–0.75 | 0.33–0.75 | 0.32–0.55 | 0.31–0.54 |
| **MLP act — shipped +SM** | 0.29–0.72 | 0.36–0.78 | 0.33–0.75 | 0.33–0.75 | 0.32–0.55 | 0.31–0.54 |
| **gate·up** | 0.43–0.68 | 0.54–0.65 | 0.52–0.62 | 0.52–0.62 | 0.51–0.71 | 0.52–0.71 |
| **A@V** | 0.15–0.27 | 0.43–0.56 | 0.41–0.51 | 0.42–0.50 | 0.29–0.54 | 0.24–0.50 |
| **norm** | 0.25–0.31 | 0.27–0.42 | 0.29–0.39 | 0.29–0.39 | 0.20–0.30 | 0.22–0.32 |
| **softcap** | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |

perturbation size `pert`, min–max over models:

| op | ioi | mcqa | arcE | arcC | arith+ | arith− |
|---|---|---|---|---|---|---|
| **softmax** | 0.05–0.10 | 0.13–0.19 | 0.09–0.14 | 0.09–0.14 | 0.05–0.11 | 0.05–0.10 |
| **Q@K** | 0.06–0.12 | 0.15–0.21 | 0.08–0.15 | 0.07–0.14 | 0.09–0.17 | 0.06–0.16 |
| **Q@K row-centered** | 0.13–0.21 | 0.22–0.32 | 0.13–0.22 | 0.12–0.21 | 0.13–0.21 | 0.12–0.23 |
| **MLP act** | 0.09–0.17 | 0.29–0.38 | 0.18–0.30 | 0.18–0.28 | 0.18–0.31 | 0.16–0.26 |
| **MLP act — shipped +SM** | 0.09–0.17 | 0.29–0.38 | 0.18–0.30 | 0.18–0.28 | 0.18–0.31 | 0.16–0.26 |
| **gate·up** | 0.03–0.20 | 0.12–0.46 | 0.11–0.33 | 0.12–0.32 | 0.06–0.38 | 0.05–0.39 |
| **A@V** | 0.21–0.27 | 0.36–0.53 | 0.24–0.35 | 0.23–0.34 | 0.21–0.37 | 0.20–0.35 |
| **norm** | 0.05–0.19 | 0.19–0.49 | 0.16–0.31 | 0.14–0.30 | 0.05–0.32 | 0.05–0.31 |
| **softcap** | 0.12 | 0.21 | 0.12 | 0.11 | 0.17 | 0.16 |

FrLN/EAP norm ratio per model (개선 < 1):

| model | ioi | mcqa | arcE | arcC | arith+ | arith− |
|---|---|---|---|---|---|---|
| gpt2 (LN) | 2.64× | 1.93× | 1.40× | 1.55× | 2.77× | 2.67× |
| qwen2.5 (RMS(w)) | 0.45× | 0.60× | 0.62× | 0.63× | 0.43× | 0.45× |
| gemma2 (RMS(1+w)) | 0.35× | 0.78× | 0.73× | 0.73× | 0.50× | 0.53× |
| llama3 (RMS(w)) | 0.34× | 0.40× | 0.49× | 0.50× | 0.33× | 0.41× |

softmax IG Z=2 residual, min–max over models:

| ioi | mcqa | arcE | arcC | arith+ | arith− |
|---|---|---|---|---|---|
| 0.082–0.155 | 0.021–0.025 | 0.020–0.029 | 0.020–0.029 | 0.012–0.028 | 0.014–0.041 |
<!-- LAG-SUMMARY-END -->

---

## 2c. MIB 12 cell 통합 표 — 세 모듈 기준

<!-- LAG-MIB-START -->
### T9. MIB 12 cell — EAP LAG → 모듈 LAG (모듈 = Bilinear · SM(sm_fix) · FrLN, softmax 는 IG Z=5)

각 칸 = `EAP 1차 LAG → 모듈 적용 후 LAG`. SM 은 clean↔cf chord(sm_fix, huisu `secant_cf_*`) 기준이며 출하판(원점 chord)은 `lag_smfix_report.md` 참조. Q@K 는 row-centered.

| cell | Q@K-c → Bilinear | A@V → Bilinear | gate·up → Bilinear | MLP act → SM | norm → FrLN | softmax → IG(Z=5) |
|---|---|---|---|---|---|---|
| gpt2/ioi | 0.84 → 1e-06 | 0.27 → 2e-07 | — | 0.72 → 1e-09 | 0.25 → 0.66 | 1.46 → 0.01 |
| qwen2.5/ioi | 0.57 → 6e-06 | 0.24 → 2e-07 | 0.48 → 1e-07 | 0.42 → 6e-10 | 0.31 → 0.14 | 1.46 → 0.02 |
| gemma2/ioi | 0.94 → 2e-06 | 0.15 → 2e-07 | 0.68 → 1e-07 | 0.59 → 7e-15 | 0.31 → 0.11 | 1.12 → 0.01 |
| llama3/ioi | 0.83 → 2e-06 | 0.27 → 2e-07 | 0.43 → 1e-07 | 0.29 → 6e-15 | 0.26 → 0.09 | 1.17 → 0.01 |
| qwen2.5/mcqa | 0.63 → 1e-05 | 0.46 → 2e-07 | 0.56 → 8e-08 | 0.51 → 2e-10 | 0.42 → 0.25 | 0.53 → 0.00 |
| gemma2/mcqa | 0.64 → 2e-06 | 0.43 → 2e-07 | 0.65 → 8e-08 | 0.52 → 2e-15 | 0.32 → 0.25 | 0.54 → 0.01 |
| llama3/mcqa | 0.41 → 1e-06 | 0.56 → 1e-07 | 0.54 → 7e-08 | 0.36 → 1e-15 | 0.42 → 0.17 | 0.63 → 0.00 |
| gemma2/arcE | 0.47 → 3e-06 | 0.43 → 4e-07 | 0.62 → 9e-08 | 0.52 → 4e-15 | 0.29 → 0.21 | 0.52 → 0.01 |
| llama3/arcE | 0.30 → 2e-06 | 0.51 → 3e-07 | 0.52 → 9e-08 | 0.33 → 5e-15 | 0.38 → 0.19 | 0.57 → 0.00 |
| llama3/arcC | 0.28 → 2e-06 | 0.50 → 3e-07 | 0.52 → 1e-07 | 0.33 → 7e-15 | 0.38 → 0.19 | 0.57 → 0.00 |
| llama3/arith+ | 0.63 → 1e-06 | 0.54 → 2e-07 | 0.59 → 1e-07 | 0.32 → 0e+00 | 0.29 → 0.10 | 0.51 → 0.00 |
| llama3/arith− | 0.66 → 1e-06 | 0.50 → 2e-07 | 0.52 → 1e-07 | 0.31 → 0e+00 | 0.26 → 0.11 | 0.56 → 0.00 |

읽는 법: Bilinear 와 SM(sm_fix) 은 closed-form 이라 전 cell fp32 floor; FrLN 은 RMSNorm(qwen·gemma·llama) 개선 / LayerNorm(gpt2) 악화; softmax 는 closed-form 이 없어 IG 로만 줄어든다.
<!-- LAG-MIB-END -->

---

## 3. 분석

그림: Fig 1 `fig_lag_budget.png` (a)(b) noise budget IOI/MCQA, (c) softmax IG 수렴, (d) 태스크 효과 산점도 · Fig 2 `fig_lag_layers.png` layer 별 EAP LAG 프로파일 · Fig 3 `fig_lag_budget_arc.png`, Fig 4 `fig_lag_layers_arc.png` ARC/arithmetic 판.

### 3.1 closed-form 모듈은 4 아키텍처 × 2 태스크 전부에서 정확하다 (T2)

- **bilinear (Q@K · Q@K-c · A@V · gate·up) → midpoint**: 30개 (op × cell) 전부 module LAG **1e-7 ~ 1e-5** (fp32 roundoff floor). 8B llama3 의 128-dim/32-head attention, qwen 의 14Q/2KV GQA + qkv bias, gemma 의 softcap 뒤 A@V 까지 예외 없음.
- **element-wise (gelu_new · gelu_tanh · SiLU · softcap) → secant**: 전부 **≤ 1e-9**. SiLU(qwen, llama)는 이번에 처음 검증 — GELU 와 동일하게 정의상 정확.
- 따라서 "Bilinear·SM 이 그 op 의 2차항을 대수적으로 닫는다"는 `opgap_report.md` §4.1 의 주장은 아키텍처·태스크에 무관하게 성립한다. 남는 것은 **softmax(IG)** 와 **norm(freeze, 근사)** 뿐.

### 3.2 잡음원 순위 — IOI 에서는 아키텍처 무관, 태스크가 바뀌면 선두가 바뀐다 (T1, Fig a·b)

**IOI** (4 모델 공통):

```
softmax (1.12–1.46, >1 = 오차가 참변화보다 큼)  ≫  Q@K-c (0.57–0.94) ≳ MLP act (0.29–0.72) ≈ gate·up (0.43–0.68)
   >  A@V (0.15–0.27) ≈ norm (0.25–0.31)  ≫  softcap (0.006)
```

- softmax 가 예외 없이 최대 잡음원. 4 모델 모두 layer 별로 LAG > 2 인 층이 있다 (gpt2 L0 3.05, gemma2 L1 2.56, qwen L9 2.71, llama3 L2 3.73; Fig 2 상단). 이름이 뒤바뀌면 attention 이 다른 토큰으로 **재라우팅**되는데, 그 점프는 포화된 softmax 의 접선으로는 잡히지 않는다.
- SiLU 모델(qwen 0.42, llama3 0.29)의 MLP-act LAG 가 GELU 모델(gpt2 0.72, gemma2 0.59)보다 낮다. llama3 는 초반 layer 에서 0.12 까지 내려간다 (Fig 2). SiLU 가 활성 분포 구간에서 더 선형에 가깝다는 뜻이지만, secant 가 어차피 정확히 닫으므로 모듈 설계에는 영향 없음.

**MCQA** (4 모델 공통):

```
모든 op 이 0.36–0.65 로 수렴.  softmax 0.50–0.63, Q@K-c 0.41–1.01, gate·up 0.54–0.65, A@V 0.43–0.56, act 0.36–0.78, norm 0.27–0.42
```

- softmax 가 더 이상 지배적이지 않다 (gpt2 에선 Q@K 0.93 이 최대, gemma2 에선 gate·up 0.65). layer 별로 LAG > 1 인 층이 gemma2 L0 하나뿐 (Fig 2 하단) — 보기 기호가 바뀌는 MCQA 에선 attention 패턴이 뒤집히지 않는다.
- 즉 **"softmax 가 최대 잡음원"은 IOI 형 태스크(attention 재라우팅)의 성질**이고, 기호 치환형 태스크에선 bilinear·act 와 같은 급이 된다.

### 3.3 태스크 효과 — 섭동은 MCQA 가 더 큰데 LAG 반응은 op 마다 다르다 (T3, T6, Fig d)

계획 단계의 가설("MCQA 는 기호 4 토큰만 바뀌니 섭동이 작고 LAG 도 작을 것")은 **틀렸다**. T3 의 pert 는 모든 op·모델에서 MCQA 가 **1.3–3.7×** 크다 (A–D → 1–4 는 이름 두 개를 서로 바꾸는 IOI 보다 임베딩 변화가 크고, 4 위치가 동시에 바뀐다). 그런데 LAG 는:

| op | MCQA/IOI LAG 비 | 해석 |
|---|---|---|
| softmax | **0.34–0.53×** (↓) | 재라우팅 없음 → 포화 점프 소멸. 섭동이 커도 LAG 는 반토막 |
| A@V | **1.8–2.9×** (↑) | A 와 V 가 **동시에** 크게 움직여 cross 항 ΔA·ΔV 증가 |
| norm | 1.0–1.6× (↑) | 잔차 섭동 자체가 커짐 (pert 2.2–3.6×) |
| MLP act, gate·up | 0.9–1.25× (≈) | 섭동은 2–3.7× 커졌는데 상대 LAG 는 그대로 — 이미 포화 근처 |
| Q@K-c | 0.49–1.2× (혼재) | gpt2·qwen ↑, gemma·llama ↓ |

→ 상대 LAG 는 섭동 크기만의 함수가 아니라 **op 이 어느 regime 에서 동작하는가**의 함수다. softmax 는 IOI 에서 포화 regime, MCQA 에서 선형 근처 regime.

CPR 과의 대응 (방향만): `eap_ig_5`(softmax 보정) 의 CPR 이득이 IOI 에서 MCQA 보다 훨씬 크다 — qwen2.5 0.71→11.29 (ioi) vs 4.81→7.53 (mcqa), gemma2 6.57→22.39 vs 7.16→9.49 (`logcpr_report.md`). softmax LAG 가 MCQA 에서 1/2–1/3 로 줄어드는 것과 방향이 일치한다. T4 에서도 MCQA 는 **Z=2 만으로 0.02** (IOI 는 0.08–0.16) 에 도달해 적분 step 이 덜 필요하다. 크기까지 예측한다는 주장은 하지 않는다 (CPR 은 baseline·self-repair 에 좌우됨).

### 3.4 qwen2.5 Q@K — raw LAG 0.08 은 k_proj bias 아티팩트, row-centered 0.57 이 실제 (T1, T7)

raw Q@K LAG 가 qwen2.5 만 0.08 (IOI) / 0.04 (MCQA) 로 다른 모델(0.57–0.93)보다 한 자릿수 작다. 진단(`lag_qk_sink.py`, `lag_sink_qwen2.5_ioi.log`):

- **attention sink(첫 key 열) 가설은 기각**: key 열 0 은 Σ‖Δ(qkᵀ)‖² 의 6% 뿐이고 ‖k₀‖ 도 다른 위치와 같다 (1.0×).
- **원인은 k_proj bias**: L0/L1/L2/L8 의 ‖b_k‖ = 367 / 209 / 135 / 284 vs ‖W_k‖_F = 11–28. key 가 프롬프트 불변 상수 `b_k` 에 지배되어 ‖Δk‖/‖k‖ = **0.026** (‖Δq‖/‖q‖ = 0.072; 다른 모델은 두 값이 비슷, T7). 항 크기 `Δq·kᵀ : q·Δkᵀ : Δq·Δkᵀ = 1 : 0.098 : 0.083` — 분모가 완전 선형인 `Δq·b_kᵀ` 로 부풀어 있다. layer 별로 보면 L8(73%) + L0(23%) 가 분모의 96% 를 차지하고 그 두 층의 LAG 는 0.00 (T1b 중앙값 0.28 과의 괴리가 여기서 나온다).
- `q_i·b_k` 는 (rotary 저주파 차원에서) key 위치 j 에 대해 상수이므로 **softmax 가 row 단위로 상쇄**한다. 즉 raw Q@K LAG 는 softmax-invariant 가 아니다. causal key 위에서 row-centering 한 **Q@K-c** 로 재면 qwen2.5 **0.57 / 0.63** 으로 다른 모델과 같은 급이 된다. 다른 모델도 centering 으로 0.66→0.84 (gpt2), 0.83→0.94 (gemma2), 0.73→0.83 (llama3) 로 소폭 오른다 (row 상수 성분은 원래 선형이라 "쉬운" 부분이었음).
- 부수 관찰: qwen2.5 의 clean attention 질량 77% 가 첫 토큰에 있고(sink), 그 열에서 softmax tangent LAG 가 **2.32** — sink 가 softmax 접선이 가장 나쁜 지점이다.

→ **Q@K 의 대표값은 row-centered 로 읽는다.** 이 교훈은 qkv bias 가 있는 모든 모델(Qwen 계열)에 해당한다.

### 3.5 FrLN — RMSNorm 3종 전부 개선, LayerNorm 은 악화 (T5)

| | LN (gpt2) | RMS `w` (qwen2.5) | RMS `(1+w)` (gemma2) | RMS `w` (llama3) |
|---|---|---|---|---|
| IOI FrLN/EAP | **2.64×** 악화 | 0.45× | 0.35× | 0.34× |
| MCQA FrLN/EAP | **1.93×** 악화 | 0.60× | 0.78× | 0.40× |

- RMSNorm 이면 weight 파라미터화(`w` vs `1+w`)·크기(0.5B–8B)·태스크 무관하게 frozen Jacobian 이 tangent 보다 참 유한변화에 가깝다. LayerNorm 은 반대. `opgap_report.md` §4.2 의 "아키텍처 의존" 결론이 2 모델 → 4 모델로 확장됐다.
- CPR 과의 대응: qwen2.5/ioi `eap_frnorm` 0.71→8.14, gemma2/ioi 6.57→14.54 는 LAG 개선과 일치. gpt2/ioi 는 LAG 가 악화되는데도 CPR 이 6.15→13.04 로 오른다 → LN 에서의 FrLN 은 Taylor gap-filler 가 아니라 ranking heuristic 이라는 기존 해석 유지. llama3 는 CPR 이 arc/arith 에만 있어(frnorm 이 악화) LAG(IOI/MCQA)와 직접 비교 불가 — LAG 개선은 CPR 개선의 필요조건 정도로만 읽는다.
- 깊이 프로파일 (Fig 2 우측): RMSNorm 모델은 norm LAG 가 초반 layer 0.58–0.73 → 후반 0.10–0.14 로 단조 감소 (IOI). 초반 잔차 norm 이 작아 상대 섭동이 크기 때문. MCQA 는 0.4 근처로 평탄.

### 3.6 pooled vs layer-median (T1 vs T1b)

pooled LAG 는 `‖Δz_true‖` 가 큰 층이 지배한다. 두 방향의 괴리가 있다:
- **pooled ≫ median**: IOI softmax (gpt2 1.46 vs 0.82, qwen 1.46 vs 0.44) — LAG > 2 인 소수 층이 참변화도 커서 pooled 를 끌어올린다.
- **pooled ≪ median**: qwen Q@K raw (0.08 vs 0.28) — §3.4 의 bias 층.
→ 두 값을 함께 보고, 괴리가 크면 layer 프로파일(Fig 2)로 내려간다.

### 3.7 방법론에 대한 함의

- **4 잡음원 클래스 ↔ 4 도구** 대응(`opgap_report.md` §1b-E)이 4 아키텍처 × 2 태스크에서 유지된다: bilinear→midpoint(정확), element-wise act/softcap→secant(정확), softmax→IG(Z), norm→freeze(RMSNorm 에서만 개선).
- softmax 만 적분이 필요하다는 점은 불변이지만, **필요한 Z 는 태스크 의존**: IOI 는 Z=5 에서 0.01, MCQA 는 Z=2 에서 0.02.
- FrLN 은 RMSNorm 계열에만 켜는 것이 LAG 근거상 옳다 (gpt2 에서의 CPR 이득은 다른 메커니즘).
- qkv bias 가 있는 모델에서 Q@K 관련 지표는 softmax-invariant 형태(row-centered)로 재야 한다.
- 어떤 op 이 최대 잡음원인지는 **counterfactual 의 섭동 유형**이 정한다 (§3.9): 재라우팅→softmax, 기호 치환→평탄, 내용 교환→Q@K. 모듈 조합의 기대 이득도 태스크별로 이 표로 읽을 수 있다.

### 3.8 한계 (정직하게)

- LAG 는 op 하나를 고립한 지표다. downstream 혼합·self-repair 는 배제되므로 **LAG ≈ 0 이 attribution 충실도를 뜻하지 않는다** (`opgap_report.md` §4.4). 실례: SiLU secant 의 LAG 는 0 인데 qwen2.5/ioi `eap_secmlp` CPR 은 EAP 보다 낮다 (0.56 vs 0.71) — LAG 로는 설명되지 않는 현상.
- gpt2/mcqa 는 MIB 에 없는 cell (gpt2 는 MCQA 를 못 푼다). 활성 텐서 성질로서만 의미 있고 CPR 대응값이 없다.
- gemma2·llama3 는 bf16 forward. 캡처된 텐서를 fp32 로 올려 gap 을 계산하므로 module LAG floor(1e-7) 는 유지되지만, EAP LAG 의 3–4째 자리는 bf16 forward 잡음을 포함한다.
- row-centering 은 per-row 상수만 제거한다. rotary 고주파 차원에 실린 bias 성분은 위치마다 회전되어 완전히 상수가 아니므로, Q@K-c 도 근사적으로만 softmax-invariant 다 (S ≤ 38 에선 저주파 차원이 지배).
- 100 example, train split, 1 seed. cell 간 차이가 0.05 이하인 경우는 해석하지 않았다.

### 3.9 ARC-easy · ARC-challenge · arithmetic 으로의 확장 — 섭동 유형이 noise budget 을 결정한다 (T8, §2b, Fig 3·4)

같은 4 모델로 arc_easy, arc_challenge, arithmetic_addition, arithmetic_subtraction 을 추가 측정했다 (16 cell, 전부 100/100 정렬; MIB cell 은 gemma2/llama3 arcE, llama3 arcC·arith+·arith−).
ARC 는 MCQA 와 같은 기호 치환 counterfactual(A–D → 1–4, 프롬프트 60–180 토큰), arithmetic 은 피연산자 자리 교환(`27 plus 64` → `72 plus 37`, 12 토큰)이다.

**ARC-easy ≈ ARC-challenge ≈ MCQA.** 두 ARC 는 모든 op·모델에서 ±0.03 이내로 같고(예: gpt2 softmax 0.546 / 0.545, llama3 Q@K-c 0.296 / 0.278), 프로파일도 MCQA 와 같다: softmax 0.51–0.57 (MCQA 0.50–0.63), A@V 0.41–0.51 (0.43–0.56), gate·up 0.52–0.62 (0.54–0.65), norm 0.29–0.39 (0.27–0.42). **문제 난이도(easy/challenge)나 질문 길이는 LAG 에 영향이 없고, counterfactual 이 무엇을 바꾸는가가 결정한다.** 차이는 Q@K-c 하나뿐 — ARC 0.28–0.67 < MCQA 0.41–1.01. 프롬프트가 2배 길어 바뀌는 4 위치가 key 전체에서 차지하는 비중이 작아지고, q·k 가 동시에 움직이는 쌍이 줄기 때문이다. layer 별로 softmax LAG > 1 인 층은 gemma2 L0 (1.37; MCQA 에서도 1.44) 하나뿐 — gemma2 layer-0 softmax 의 모델 고유 특성이다.

**arithmetic 은 세 번째 regime — Q@K 지배.** 피연산자 자리 교환은 두 숫자 토큰의 q 와 k 를 **동시에** 바꾼다. 그래서 bilinear cross 항 `Δq·Δkᵀ` 가 가장 크고 (Q@K-c 덧셈 0.63–1.40 / 뺄셈 0.66–1.47; gpt2 는 L0–L4 가 1.1–3.1, llama3·gemma2·qwen 도 4–6 개 층이 > 1), softmax 는 6 태스크 중 최저(덧셈 0.37–0.51, 뺄셈 0.38–0.56), norm 도 최저(0.20–0.32) 다. **덧셈과 뺄셈은 모든 op 에서 ±0.07 이내로 같다** (MIB cell llama3: Q@K-c 0.63 / 0.66, softmax 0.51 / 0.56, act 0.33 / 0.31) — ARC-easy ≈ ARC-challenge 와 같은 이유로, 연산자가 아니라 counterfactual 의 형태(피연산자 교환)가 LAG 를 정한다. ‖Δq‖/‖q‖ 와 ‖Δk‖/‖k‖ 가 비슷한 크기(gemma2 0.26/0.23, llama3 0.14/0.14)라는 점이 IOI(재라우팅: q 가 주로 이동)·MCQA(기호: v 경로가 주로 이동)와 다르다.

세 regime 정리:

| 섭동 유형 | 태스크 | 최대 잡음원 | 특징 |
|---|---|---|---|
| attention 재라우팅 | IOI (이름 교환) | **softmax** 1.1–1.5 | 포화 softmax 점프, 층별 LAG > 2 다수 |
| 기호 치환 | MCQA · ARC-easy · ARC-challenge | 없음 (모두 0.3–0.7) | softmax 반토막, A@V·norm 상승, 난이도·길이 무관 |
| 내용 교환 (q·k 동시) | arithmetic (+, −) | **Q@K** 0.6–1.5 | bilinear cross 항 최대, softmax 최저, 연산자 무관 |

**아키텍처 결론은 6 태스크에서 그대로.** closed-form 모듈은 16 cell 추가에서도 전부 floor (bilinear ≤ 1e-5, secant ≤ 1e-9). FrLN 은 24/24 cell 에서 LN 악화(1.40–2.77×) / RMSNorm 개선(0.33–0.78×). softmax IG 는 기호 치환·arithmetic 에서 Z=2 로 0.012–0.029 (IOI 만 0.08–0.16 로 Z=5 필요).

**CPR 과의 대응 (llama3 arc/arith, `logcpr_report.md`):** `eap_igbilin` 의 CPR 이득이 arithmetic 에서 크고(덧셈 1.86→8.64, 뺄셈 2.81→7.81) ARC-easy 에서 없다(5.53→5.57). LAG 도 llama3 Q@K-c 가 arithmetic 0.63–0.66 vs ARC 0.30 으로 2배 — bilinear 잡음이 큰 곳에서만 bilinear 보정이 CPR 로 이어진다는 방향과 일치. 반면 `eap_frnorm` 은 llama3 ARC-easy 에서 CPR 이 악화(5.53→2.48)되고 gemma2 ARC-easy 도 소폭 악화(9.20→8.90)되는데 LAG 는 둘 다 개선(0.49×, 0.73×) — **norm LAG 개선은 FrLN CPR 이득의 충분조건이 아니다** (§3.5 의 llama3 유보를 실측으로 확인).

---

## 4. 파일

| 파일 | 내용 |
|---|---|
| `scripts/lag_opgap.py` | 통합 측정 스크립트 (`--model`, `--task`), JSON 출력 |
| `scripts/lag_all.sh [model:task ...]` | cell 순차 실행 체인 (기본 8 cell; ARC/arith± 는 인자로) |
| `scripts/lag_{model}_{task}.log` | 실행 로그 |
| `method_comparison/lag/{model}_{task}.json` | op 별 합계 + layer 별 프로파일 (`per_layer`) |
| `scripts/build_lag_report.py` | JSON → 이 문서의 §2 · §2b 테이블 + T8 요약 (비-MIB cell 은 † 표기) |
| `scripts/fig_lag_budget.py [--tasks ... --out ...]` | Fig 1 (기본) / Fig 3 (`--tasks arc_easy arc_challenge arithmetic_addition arithmetic_subtraction --out fig_lag_budget_arc`): (a)(b) noise budget IOI/MCQA, (c) softmax IG 수렴, (d) 태스크 효과 산점도 |
| `scripts/fig_lag_layers.py [--tasks ... --out ...]` | Fig 2 / Fig 4 (`--out fig_lag_layers_arc`): layer 별 EAP LAG 프로파일 (softmax · Q@K-c · act · gate·up · norm) |
| `method_comparison/fig_lag_budget{,_arc}.{pdf,png}`, `fig_lag_layers{,_arc}.{pdf,png}` | 위 그림 |
| `scripts/lag_qk_sink.py`, `scripts/lag_sink_{qwen2.5,gpt2}_ioi.log`, `lag/sink_*.json` | §3.4 진단: key 열 0 분리, ‖Δq‖/‖Δk‖, 항 크기 비 |
