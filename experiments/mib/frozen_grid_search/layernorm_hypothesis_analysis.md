# LayerNorm-distortion 가설 분석

**가설**: `--frozen-norm` 없이 LayerNorm/RMSNorm을 통과하는 backward pass가 σ-의존 항 때문에 sample-by-sample로 출렁여, attribution score의 noise (variance)를 *층 깊이에 따라 증폭*시킨다. Weight scaling perturbation은 그 distortion을 보정하기 위한 일종의 noise cleanup이다.

연관 plot:
- `frozen_grid_search/plots/layernorm_cv_analysis.png` — frozen vs scaling per-layer CV
- `frozen_grid_search/plots/layernorm_cv_decomposition.png` — `|score|`, variance, CV 분해

CV 정의 (project utils): `CV = variance / |score|` (행렬을 element-wise로). 행 단위 (소스 레이어) median으로 layer aggregate.

---

## 1. 측정: Layer-depth CV 추세

### 1-1. CV (median by source layer) — frozen baselines 5블록

| block | L0 CV | last-layer CV | last/L0 |
|---|---|---|---|
| gpt2/ioi | 0.0251 | 0.127 | **5.1×** |
| qwen2.5/ioi | 0.00183 | 0.679 | **371×** |
| qwen2.5/mcqa | 0.000124 | 0.0752 | **605×** |
| gemma2/ioi | 0.0109 | 0.186 | **17×** |
| gemma2/mcqa | 0.00213 | 0.0537 | **25×** |

**모든 블록에서 CV는 깊은 layer로 갈수록 커진다** — 5×~605× 범위. 가설의 핵심 예측인 "noise가 깊이 따라 증폭"은 frozen run에서도 분명히 관찰됨.

### 1-2. CV 분해: |score| vs variance

| block | last/L0 of \|score\| | last/L0 of variance | CV last/L0 |
|---|---|---|---|
| gpt2/ioi | 30× | ~150× | 5× |
| qwen2.5/ioi | 5,840× | ~2,170,000× | 371× |
| qwen2.5/mcqa | 596× | ~360,000× | 605× |
| gemma2/ioi | 340× | ~5,800× | 17× |
| gemma2/mcqa | 201× | ~5,000× | 25× |

**variance가 \|score\|보다 훨씬 빠르게 자란다**. 즉, 깊은 layer일수록 score의 *상대적* noise가 커짐. 이것이 CV 증가의 직접 원인이며, LayerNorm이 σ-의존 path를 통해 sample noise를 누적시킨다는 가설과 일치하는 정성적 패턴.

---

## 2. Frozen vs Scaling 직접 비교 (gpt2/ioi만 가능)

scaling sweep은 아직 진행 중이라 baseline (q=k=v=gate=up=1.0) 파일이 gpt2/ioi에만 있음. 다른 블록은 sweep이 value=1.0 라운드에 도달해야 비교 가능.

| metric | frozen | scaling | scaling/frozen |
|---|---|---|---|
| 전체 median CV | 0.0112 | 0.0101 | **0.90×** |
| L0 source CV | 0.025 | 0.024 | 0.96× |
| 마지막 source CV | 0.127 | 0.110 | 0.87× |

**예상 외**: gpt2/ioi에서 scaling이 frozen보다 *살짝 낮은* CV를 보임. 가설 (frozen이 noise를 줄여야 함)과 반대 방향.

가능한 해석:
1. **gpt2 특수성**: gpt2는 LayerNorm + GELU + 12 layer, GQA 없음. RMSNorm 모델 (qwen2.5, gemma2)과 gradient distortion 양상이 다를 수 있음.
2. **얕은 깊이**: 12 layer는 noise 누적이 충분히 일어나기 전에 끝남. 깊은 모델 (24, 26 layer)에서는 frozen-norm 효과가 더 클 가능성.
3. **CV 정의의 양면성**: `var/|s|`. frozen이 score를 더 보수적으로(작게) 만들면 분모가 작아져 CV가 큼. scaling이 score를 키우면서 CV를 낮출 수 있음 — 즉 "noise는 같지만 signal이 상대적으로 큰" 효과.

→ 가설을 본격 검증하려면 qwen2.5/gemma2 scaling baseline이 필요 (현재 sweep 진행 중).

---

## 3. Parameter sweep 결과와 layer-depth CV의 상관

### 3-1. CV 증가폭 vs CPR lift

| block | CV growth (last/L0) | best CPR lift |
|---|---|---|
| gpt2/ioi | 5.1× | +0.10 |
| gemma2/ioi | 17× | +0.68 |
| gemma2/mcqa | 25× | +1.09 |
| qwen2.5/ioi | 371× | +0.32 |
| qwen2.5/mcqa | 605× | +0.44 |

직관: "noise 누적이 클수록, perturbation이 청소할 여지가 크고 lift도 커야 함".

- **gpt2/ioi (가장 작은 growth, 가장 작은 lift)**: 일치.
- **gemma2 두 블록 (중간 growth, 큰 lift)**: 일관성 OK.
- **qwen2.5 (가장 큰 growth, 중간 lift)**: 어긋남. growth가 가장 큰데 lift는 gemma2보다 작음.

→ CV growth와 CPR lift는 **단조 관계가 아님**. "noise 청소"만으로는 lift를 다 설명 못 함. 다른 요인:
- 모델의 회로 구조 자체의 redundancy
- 모델 크기와 task 난이도
- 어느 path (q/k/v/gate/up)가 효과적인지가 architecture에 의존 (gpt2엔 gate 없음 등)

### 3-2. Best perturbation의 *공간적 위치* vs CV의 *깊이 분포*

가설 검증: "perturbation이 효과적인 곳이 high-CV 영역인가?"

이전 circuit-heatmap 분석에서 |diff|가 가장 큰 source layer:

| block | top source layers (|diff| share) | CV at those layers (median) |
|---|---|---|
| gpt2/ioi | L0 (37.5%), L3 (15.1%), L5 (12.5%) | low~mid (CV: 0.025, 0.029, 0.034) |
| qwen2.5/ioi | L7 (10.3%), L11 (10.0%), L9 (9.2%) | mid (CV ~0.005~0.01) |
| qwen2.5/mcqa | L4 (9.9%), L1 (8.7%), L5 (8.6%) | early-low (CV ~0.0003) |
| gemma2/ioi | L11 (13.1%), L10 (8.5%), L8 (7.4%) | mid (CV ~0.01) |
| gemma2/mcqa | L1 (10.1%), L2 (9.0%), L0 (8.6%) | early (CV ~0.003) |

**관찰**: Perturbation은 CV가 *가장 높은* (마지막 깊은) layer가 아니라, **early-mid layer**에 집중. 즉, **perturbation이 noise cleanup 그 자체를 한다기보다는, 회로의 'real work' 영역을 강조/억제**.

깊은 layer의 high CV는 readout에 가까운 "saturated noise" 영역으로 해석하는 게 자연스러움 (lm_head 직전이라 score와 variance 둘 다 큼). 이 영역은 회로 mechanism이 '결정되는' 곳이 아니라 '실현되는' 곳.

---

## 4. 가설 평가

| 가설 요소 | 데이터 지지 여부 |
|---|---|
| (a) noise가 layer 깊이에 따라 증폭 | **YES** — 모든 블록에서 CV가 5~605× 증가. variance가 \|score\|보다 빠르게 커짐. |
| (b) `--frozen-norm`이 noise를 줄임 | **부분적/불확실** — gpt2에서 거의 차이 없음 (오히려 살짝 반대). qwen/gemma 데이터 필요. |
| (c) perturbation이 noise 정화 메커니즘 | **부분적** — CV growth와 CPR lift는 거칠게 양의 상관이지만 단조 아님. perturbation 위치는 high-CV deep layer가 아닌 mid layer에 집중. |
| (d) `--frozen-norm`은 LayerNorm gradient 왜곡을 막는다 | **개념적으로 맞음** (수식상 σ-detach). 하지만 *측정 가능한 attribution 안정성* 차이는 모델/태스크에 따라 다를 수 있음. |

종합: **"layer 깊이 따라 noise 증폭" 부분은 강하게 지지**, 하지만 **"frozen-norm으로 단조 개선되고 그게 perturbation lift의 주 원인"이라는 인과 사슬은 데이터로 부분만 확인**. 특히 gpt2/ioi에서 frozen-norm 효과가 미미한 점은 가설의 *모델/depth 의존성*을 시사.

---

## 5. 다음 단계 제안

1. **scaling sweep의 5블록 baseline 모두 수집 후 재비교**: 진행 중. 끝나면 frozen vs scaling 5블록 ratio 비교 → 가설 (b)에 대한 결정적 증거.
2. **per-layer NORM-only diff**: scaling에서 frozen으로 갔을 때 layer별 score 자체가 어떻게 바뀌는지 (rank-correlation 같은 robust metric으로). CV는 noise 측정이지만 *순위 보존* 여부도 봐야 함 (CPR이 ranking-driven).
3. **gemma2의 `gate` perturbation 유의성**: 이미 가장 큰 lift (+1.09). gemma2 scaling baseline 받으면 그 perturbation으로 CV가 얼마나 떨어지는지 측정 → "perturbation = noise cleanup" 가설의 직접 검증.
4. **qwen2.5의 365×~600× CV growth**: 다른 모델보다 한 자릿수 더 큼. 모델 archi 차이 (RMSNorm + GQA + 24 layer)와 어떻게 연결되는지 별도 조사할 만함.
