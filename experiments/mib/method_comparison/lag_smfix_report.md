# sm_fix 검증 — 출하된 +SM(원점 secant) vs sm_fix(cf secant) 의 MLP-activation LAG

측정일 2026-10-09. huisu 브랜치 머지 후, `③ sm_fix only (IG 없음)` 조건에 해당하는 op-level LAG. 본문은 **MIB circuit track 의 12 cell** (ioi×4, mcqa×3, arcE×2, arcC·arith+·arith− ×llama3). train split 100쌍, op 고립, full clean→corrupt 전이. 비-MIB 조합 12개는 부록.

## 1. 무엇을 비교했나

MLP activation op `z = φ(g)` 하나를 고립하고, clean→corrupt 입력 변화 `Δg = g* − g` 에 대한 출력 변화 추정량 `Ê = c · Δg` 를 세 가지 계수 `c` 로 만들어 참변화 `φ(g*) − φ(g)` 와 비교한다.

| 추정량 | 계수 `c` | 어디에 쓰였나 |
|---|---|---|
| tangent (EAP) | `φ'(g)` | 모듈 없는 EAP / EAP-IG 의 backward |
| **원점 chord — 출하된 +SM** | `φ(g) / g` (SiLU 면 `sigmoid(g)`) | `secant_silu`, `secant_gelu_tanh` — 지금까지의 모든 `*_secmlp` CPR |
| **cf chord — sm_fix** | `[φ(g*) − φ(g)] / (g* − g)` | huisu `secant_cf_silu`, `secant_cf_gelu_tanh` (DeepLIFT Rescale) |

```
LAG(c) = sqrt( Σ ‖c·Δg − (φ(g*) − φ(g))‖² / Σ ‖φ(g*) − φ(g)‖² )
```

원점 chord 는 `φ(g) = c·g` 를 만족하는 계수, 즉 입력을 0 으로 지울 때(zero ablation)만 정확한 분해다. counterfactual patching 에서 입력은 `g → g*` 로 움직이므로 필요한 계수는 그 구간의 chord 다. 이전 `lag_report.md` §3.1 의 "SM(secant) LAG ≈ 0" 은 후자(cf chord)를 가정한 값이었고, 출하 모듈의 LAG 가 아니었다.

## 2. 결과 — MIB 12 cell

| cell | act | pert | tangent (EAP) | **원점 chord (출하 +SM)** | 원점/tangent | **cf chord (sm_fix)** |
|---|---|---|---|---|---|---|
| gpt2/ioi | gelu_new | 0.171 | 0.7182 | **0.7997** | 1.11× | 1.0e-09 |
| qwen2.5/ioi | SiLU | 0.134 | 0.4217 | **0.5010** | 1.19× | 6.3e-10 |
| gemma2/ioi | gelu_tanh | 0.137 | 0.5947 | **0.6829** | 1.15× | 6.8e-15 |
| llama3/ioi | SiLU | 0.091 | 0.2919 | **0.3438** | 1.18× | 6.4e-15 |
| qwen2.5/mcqa | SiLU | 0.344 | 0.5122 | **0.5068** | 0.99× | 1.7e-10 |
| gemma2/mcqa | gelu_tanh | 0.314 | 0.5206 | **0.6073** | 1.17× | 1.7e-15 |
| llama3/mcqa | SiLU | 0.295 | 0.3610 | **0.3379** | 0.94× | 1.3e-15 |
| gemma2/arcE | gelu_tanh | 0.207 | 0.5151 | **0.6268** | 1.22× | 4.2e-15 |
| llama3/arcE | SiLU | 0.184 | 0.3299 | **0.3339** | 1.01× | 5.3e-15 |
| llama3/arcC | SiLU | 0.177 | 0.3297 | **0.3338** | 1.01× | 6.6e-15 |
| llama3/arith+ | SiLU | 0.176 | 0.3224 | **0.3401** | 1.05× | 0.0e+00 |
| llama3/arith− | SiLU | 0.158 | 0.3077 | **0.3388** | 1.10× | 0.0e+00 |


### 요약

| | tangent (EAP) | 원점 chord (출하 +SM) | cf chord (sm_fix) |
|---|---|---|---|
| MIB 12 cell 범위 | 0.29–0.72 | 0.33–0.80 | ≤ 1e-09 |
| 원점 chord 가 tangent 보다 나쁜 cell | — | **10 / 12** (부록 포함 22 / 24) | — |

예외 (원점 chord 가 tangent 보다 근소하게 나은 cell): qwen2.5/mcqa (0.99×), llama3/mcqa (0.94×). 두 cell 모두 cf chord 는 0 이라 우연한 부호 상쇄이지 메커니즘이 아니다.

## 3. 해석

1. **출하된 +SM 은 MLP-act 의 2차항을 메우는 모듈이 아니었다.** 원점 chord 는 clean→cf 경로 위 어떤 점의 기울기도 아니다. tangent 가 놓치는 곡률을 잡기는커녕 1차 항까지 틀리게 만들어, MIB 10/12 cell 에서 LAG 가 tangent 보다 **크다** (1.01~1.22×; 부록 포함 22/24, 최대 1.32×). huisu 의 rule-bias 측정(원점 chord 0.44 vs tangent 0.007, 상대 L1)과 같은 방향이며 지표가 달라 수치는 다르다.
2. **qwen2.5/ioi `eap_secmlp` CPR < EAP (0.56 vs 0.71) 가 설명된다.** 이전 리포트에서 "SM LAG 는 0 인데 CPR 이 떨어진다"를 미설명 사례로 들었으나, 실제 모듈의 LAG 는 0.50 으로 EAP 의 0.42 보다 나빴다.
3. **sm_fix(cf chord)는 정의상 정확하다** — 12/12 cell (부록 포함 24/24) ≤ 1.3e-9 (fp32 floor). "SM 이 op 의 2차항을 대수적으로 닫는다"는 주장은 sm_fix 에 대해서만 성립한다.
4. **지금까지의 `*_secmlp` CPR 열은 모듈 효과가 아니다.** 다른 모듈(Bilinear/FrLN/IG) 의 효과 위에 원점 chord 노이즈가 얹힌 값으로 읽어야 한다. huisu 문서 §3 의 CPR 재측정(sm_fix 단독 12 cell 중 6 승, ig_fix+sm_fix 로 full stack 이 12 중 5 승)과 함께 보면, op 수준 오차 0.5 를 0 으로 없애도 edge ranking 은 절반만 좋아진다 — LAG ≈ 0 이 충실도를 보장하지 않는다는 기존 결론 그대로.
5. 모델별: SiLU 모델(qwen, llama)에서 tangent LAG 자체가 낮고(0.29~0.51) 원점 chord 의 추가 손상도 작다(1.01~1.19×). GELU 모델(gpt2, gemma2)은 tangent 0.47~0.78 에 원점 chord 가 1.05~1.32× 를 더한다. MIB 안에서는 gemma2/arcE 가 최악(1.22×); 부록의 gpt2/arith± 는 1.30×, 1.32×.

## 4. 범위 밖

- ② ig_fix(`IG_ANCHOR=next_step`)는 IG 의 Z 단계별 sub-interval `[α_k, α_{k+1}]` 에서 chord 를 잡는 수정이다. 여기의 LAG 는 full clean→cf 단일 구간 정의라 ig_fix 효과를 재지 못한다. 재려면 Z 단계별 sub-interval LAG 를 따로 정의해야 한다.
- ④ ig_fix + sm_fix 도 같은 이유로 미측정.

## 5. 파일

| 파일 | 내용 |
|---|---|
| `scripts/lag_opgap.py` | `act` = cf chord(sm_fix), `act_origin` = 출하 +SM 원점 chord 추정량 추가 (2026-10-09) |
| `method_comparison/lag/{model}_{task}.json` | `ops.act`, `ops.act_origin` 의 `eap` / `mod` / `per_layer` |
| `method_comparison/lag_report.md` §3.10, T2 행 "MLP act — shipped +SM" | 본 결과의 요약본 |
| `data_exploration_branch/correction_audit_and_noise.md` (huisu) | 원점 chord 발견 경위, rule-bias, CPR 재측정 |
| `vendor/linear_transformer/linear_transformer/modules/activations.py` | `SecantSiLU`(원점) vs `SecantCFSiLU`(cf) 구현 |

## 부록. 비-MIB 참고 cell 12개 (CPR 대응값 없음)

같은 측정을 MIB 에 없는 (model, task) 조합에도 적용한 값. 아키텍처 간 비교용.

| cell | act | pert | tangent (EAP) | **원점 chord (출하 +SM)** | 원점/tangent | **cf chord (sm_fix)** |
|---|---|---|---|---|---|---|
| gpt2/mcqa† | gelu_new | 0.376 | 0.7793 | **0.8178** | 1.05× | 3.8e-10 |
| gpt2/arcE† | gelu_new | 0.296 | 0.7483 | **0.7872** | 1.05× | 8.5e-10 |
| qwen2.5/arcE† | SiLU | 0.233 | 0.4802 | **0.4932** | 1.03× | 4.7e-10 |
| gpt2/arcC† | gelu_new | 0.283 | 0.7470 | **0.7880** | 1.05× | 9.2e-10 |
| qwen2.5/arcC† | SiLU | 0.225 | 0.4782 | **0.4923** | 1.03× | 4.6e-10 |
| gemma2/arcC† | gelu_tanh | 0.199 | 0.5096 | **0.6213** | 1.22× | 1.3e-14 |
| gpt2/arith+† | gelu_new | 0.192 | 0.5463 | **0.7124** | 1.30× | 9.5e-10 |
| qwen2.5/arith+† | SiLU | 0.226 | 0.4480 | **0.5045** | 1.13× | 4.3e-10 |
| gemma2/arith+† | gelu_tanh | 0.306 | 0.4670 | **0.5163** | 1.11× | 8.7e-15 |
| gpt2/arith−† | gelu_new | 0.176 | 0.5388 | **0.7133** | 1.32× | 1.3e-09 |
| qwen2.5/arith−† | SiLU | 0.168 | 0.4452 | **0.5288** | 1.19× | 7.3e-10 |
| gemma2/arith−† | gelu_tanh | 0.263 | 0.5204 | **0.5932** | 1.14× | 2.8e-15 |
