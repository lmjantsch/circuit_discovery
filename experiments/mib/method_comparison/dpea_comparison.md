# DPEA(djk/DPA) vs lasse-circuit 구현 — 심층 비교

본 문서는 `/home/dacslab/djk/DPA/DPA_Tracing` (이하 **DPEA**) 와 본 저장소
(`circuit_discovery`, 이하 **lasse-circuit**) 두 attribution 파이프라인의
설계 차이를 코드 레벨로 정리하고, Gemma2/IOI 에서 관찰되는 **CPR gap
(3.77 vs 2.37, Δ=1.40)** 의 원인 후보를 좁힌다.

비교의 출발점: 두 파이프라인 모두 **nnsight + HuggingFace** 위에서 동작
하며, TransformerLens 는 attribution 단계에서 쓰지 않는다 (평가 시
`MIB-circuit-track/run_evaluation.py` 만 TL 사용). 따라서 모델 forward
자체는 거의 동일하고, 차이는 **backward 계산 방법과 source 시맨틱**
에서 발생한다.

---

## 1. 같은 부분 (수학적으로 동치)

| 항목 | 양쪽 모두 |
|---|---|
| 모델 forward | HF `AutoModelForCausalLM`, eager attn, bfloat16 (gpt2는 fp32) |
| Forward caches | clean + corrupt 두 패스 (IG variant 만 + k−1 보간) |
| Logit-diff target | `(lm_head[y+] − lm_head[y−])` |
| RMSNorm VJP (frozen 모드) | diagonal `(γ/σ)⊙g` |
| Softmax VJP (default) | 표준 Jacobian `A⊙(g − A·g)` |
| MLP secant — SiLU (Llama/Qwen) | `σ(z)·grad` (= `silu(z)/z`, analytically 동등) |
| MLP secant — GELU-erf | `Φ(x)·grad` (= `gelu(z)/z`) |
| MLP secant — GELU-tanh (Gemma2) | `0.5(1+tanh(...))·grad` |
| Gemma2 `(1+w)` RMSNorm weight | 양쪽 모두 정확히 처리 |
| Gemma2 4 norms 인지 | 4개 (input/post_attn/pre_ffn/post_ffn) 모두 인지 |
| Per-head attention 분해 | `einsum('BSHd, HdD -> HBSD', z, W_o per-head)` |

결론: **수학적 backward 규칙(VJP)은 거의 동등**. 차이는 *어떻게* 계산
하느냐와 *어디서* attribution source 를 읽느냐에 있다.

---

## 2. 결정적으로 다른 5 가지

### (1) Backward 구현 방식 — autograd vs hand-derived closed-form

| 측면 | lasse-circuit | DPEA |
|---|---|---|
| 진입점 | `metric.backward()` (nnsight) | `@torch.no_grad()` + 수동 |
| 그래디언트 그래프 | torch.autograd.Function patch | 없음 — 모든 VJP 명시 |
| 결과 | autograd 가 patched modules 의 backward 를 호출 | layer 단위로 수식 평가 |

DPEA 의 `FastEdgeTracer.trace` 전체에 `@torch.no_grad()`가 걸려 있고
([edge_tracer_fast.py:52](file:///home/dacslab/djk/DPA/DPA_Tracing/tracer/edge_tracer_fast.py#L52)),
`get_mlp_update_dual`, `_get_attn_update_per_head` 가 layer 별 backward
를 수동으로 계산한다. 우리는 `LinearGemma2RMSNorm`, `SecantGELUTanh`,
`BilinearMatmul` 등 `torch.autograd.Function` 으로 backward 를 정의해
autograd 가 자동 호출하게 했다.

수학적으로는 같지만 다음에서 갈릴 수 있다:
- bfloat16 누적 순서
- `.detach()` 위치
- nnsight session 내부에서 grad 가 어디까지 전파되는지

### (2) Path scaling 기본값

```python
# DPEA backend.py:43 — DEFAULT
self.scaling = scaling_config or {
    'up': 0.5, 'gate': 0.5, 'v': 0.5, 'q': 0.25, 'k': 0.25
}
```

```python
# lasse-circuit run_attribution.py:133 — DEFAULT
default=[1.0, 1.0, 1.0, 1.0, 1.0]      # (q, k, v, gate, up)
```

DPEA 는 **모든 run 에서** typed scaling `{q:0.25, k:0.25, v:0.5,
gate:0.5, up:0.5}` 가 켜져 있다. 이건 "Rule 4 typed forward
decomposition" 에 해당. 우리는 디폴트가 uniform 1.0 이고, `eap_frnorm_
secantmlp_scale_k02g02` 는 `[1, 0.2, 1, 0.2, 1]` 로 다른 패턴.

**DPEA 패턴은 단 한 번도 시도하지 않았음.**

### (3) Gemma2 softcap 도함수 — DPEA 기본 모드에서 DROPPED

Gemma2 는 두 곳에 `softcap·tanh(x/softcap)` 클리핑을 적용한다:
- attn logit softcap (정수 50.0)
- final logit softcap (정수 30.0)

DPEA 기본 모드 (`DPEA_EAP_COMPAT=0`):

```python
# DPEA edge_tracer_fast.py:327
attn_softcap = self.backend.attn_logit_softcap if eap_compat else None
if attn_softcap is not None:
    softcap_factor = 1.0 - torch.tanh(s_raw / attn_softcap) ** 2
    delta_ij = delta_ij * softcap_factor
```

`eap_compat=False` (default) → `attn_softcap = None` → softcap 도함수가
**적용되지 않음** (단, forward 는 그대로 softcap 적용). 결과적으로
backward 에서 `tanh` 를 identity 로 취급.

우리 (lasse-circuit) 는 patched `LinearGemma2Attention` 에서
`module.attn_softcap_fn` 호출. 기본값은 `tanh` 라서 표준 autograd 가
`(1 − tanh²)` 인수를 곱한다 (`s_raw/softcap` 가 큰 영역에서는 ~0). 이걸
끄려면 `--attn-softcap-fn identity_tanh --final-softcap-fn identity_tanh`
가 필요한데, `method_comparison.sh` 의 어느 변형도 이 플래그를 사용
하지 않음.

### (4) Bilinear (Q@Kᵀ, gate·up) gradient split

우리 `eap_frnorm_secantmlp_bilinear` 변형은 `BilinearMatmul` /
`BilinearMul` autograd Function 으로 Q@Kᵀ 와 gate·up 의 gradient 를
양쪽에 0.5×씩 분배:

```python
# linear_transformer/modules/bilinear.py:53
grad_x = (x_weight * grad_t @ y.float().mT)   # x_weight = 0.5
grad_y = (y_weight * x.float().mT @ grad_t)   # y_weight = 0.5
```

DPEA 는 표준 backward 를 그대로 쓴다 (gradient split 없음). 즉 우리
bilinear 변형의 q,k 점수는 DPEA 대비 implicit 으로 0.5× scaling 된
상태에서 추가로 `--weights` 가 곱해진다.

### (5) Edge source 시맨틱

| | lasse (기본) | lasse (`--raw-edge-source`) | DPEA |
|---|---|---|---|
| Qwen2/Llama2 attn source | `einsum(z, W_o)` (RAW) | 동일 (no-op) | `einsum(z, W_o)` (RAW) |
| Qwen2/Llama2 MLP source | `layer.mlp.output` (RAW) | 동일 (no-op) | `(gate·up)@W_down` (RAW) |
| Gemma2 attn source | RAW + `post_attention_layernorm` | RAW | RAW |
| Gemma2 MLP source | `post_feedforward_layernorm.output` | `layer.mlp.output` (RAW) | RAW |

방금 실험으로 **(5) 만 단독으로 바꿔도 Gemma2/ioi gap 은 거의 안 좁혀짐**
(2.3719 → 2.3563, Δ=−0.0156). 즉 RAW source 자체가 답은 아님.

---

## 3. 실험 결과 누적

| Method | gemma2/ioi CPR | 비고 |
|---|---:|---|
| `eap_pure` | 0.8867 | baseline |
| `eap_ig_k5` | 3.1410 | +IG(k=5) |
| `eap_frnorm` | 1.8670 | +diag RMSNorm VJP |
| `eap_frnorm_secantmlp` | 2.1989 | +GELU-tanh secant |
| `eap_frnorm_secantmlp_bilinear` | 2.3719 | +Bilinear |
| `eap_frnorm_secantmlp_scale_k02g02` | 2.3128 | +scale [1,0.2,1,0.2,1] |
| `eap_frnorm_secantmlp_bilinear_ig5` | 2.8923 | + IG(k=5) |
| `eap_frnorm_secantmlp_bilinear_rawsrc` | **2.3563** | + RAW src (Gemma2 post-norm bypass) |
| `eap_frnorm_secantmlp_bilinear_rawsrc_softcapoff` | TBD | + softcap deriv drop |
| **DPEA v1 NEW** | **3.7668** | DPEA full mode |
| **DPEA IG NEW (k=5)** | **3.7121** | DPEA + IG(k=5) |

Gemma2/ioi 에서 IG(k=5) 없이 우리가 도달한 최고점 = **2.3719**, IG 포함
도 **2.8923**. DPEA single-point 가 **3.77**.

---

## 4. 다음으로 좁혀진 가설

(1)–(5) 중 (5) 는 검증 완료 (영향 미미). 남은 후보:

### 후보 A — Softcap derivative drop (현재 진행 중)
- 변경: `--attn-softcap-fn identity_tanh --final-softcap-fn identity_tanh`
- 예상: Gemma2 only 큰 효과 (Qwen2엔 softcap 없으므로 우리도 이미 SOTA
  근처라는 정황과 합치). softcap 영역에서 `(1−tanh²)` 가 0 근처로
  내려가서 attribution 신호를 압축하는 걸 방지.
- 검증: `eap_frnorm_secantmlp_bilinear_rawsrc_softcapoff` (현재 GPU 4
  에서 평가 중)

### 후보 B — DPEA 패턴 scaling `[0.25, 0.25, 0.5, 0.5, 0.5]`
- 변경: `--weights 0.25 0.25 0.5 0.5 0.5`
- 예상: 우리의 `[1, 0.2, 1, 0.2, 1]` 와는 패턴이 완전히 달라 별도 효과
  가능. q,k 가 동일 비율인 것에 주목.
- 검증: 후보 A 결과 이후

### 후보 C — Bilinear 제거 + DPEA scaling
- 변경: 위 (B) 에서 `--matmul-fn` / `--mul-fn` 둘 다 제거
- 검증: 후보 B 결과 이후

후보 A 가 단독으로 Δ=1.4 를 메우면 다른 후보는 ablation 으로 진행. 결과
가 부분적이면 (예: Δ=0.5–0.8) 후보 B,C 를 cumulative 로 더한다.

---

## 5. 코드 위치 색인

| 항목 | lasse-circuit | DPEA |
|---|---|---|
| 메인 tracer | `tracer/tracer.py:13` (`EdgeCircuitTracer`) | `tracer/edge_tracer_fast.py:24` (`FastEdgeTracer`) |
| 모델 어댑터 base | `tracer/model_adapters.py:8` (`ModelAdapter`) | `tracer/backend.py:19` (`ModelBackend`) |
| Gemma2 어댑터 | `tracer/model_adapters.py:461` | `tracer/backend.py:610` |
| RMSNorm VJP (diag) | adapter 내 `_compute_norm_input_gradient` 의 `frozen_norm=True` 분기 (`model_adapters.py:419`, `:547`) | `backend.py:84` (`_rmsnorm_vjp_diag`) |
| Softmax VJP (standard) | autograd (`torch.softmax`) | `edge_tracer_fast.py:319-321` (수동) |
| MLP gate secant | `linear_transformer/modules/activations.py` (`SecantGELU`, `SecantSiLU`, `SecantGELUTanh`) | `backend.py:177` (`_gate_secant`), Gemma override `:634` |
| Bilinear (Q@K, gate·up) | `linear_transformer/modules/bilinear.py` (only with `--matmul-fn`) | 없음 (표준 backward) |
| Path scaling | `tracer/tracer.py:289` (`_scale_and_detach_grad`), `--weights` | `backend.py:42`, `self.scaling[*]` 적용 in VJP |
| Softcap fn | `linear_transformer/modules/activations.py` (`IdentityTanh`) + `--attn-softcap-fn` / `--final-softcap-fn` | `edge_tracer_fast.py:179, 327` (eap_compat 조건부) |
| Edge source (Gemma2) | `tracer/model_adapters.py:467-488` (raw_edge_source 분기) | `edge_tracer_fast.py:425-455` |
| Method 매핑 | `scripts/method_comparison.sh:72-100` (`build_attr_flags`) | `scripts/run_v1_logitdiff_all.sh`, `run_ig_inputs_logitdiff_all.sh` |
