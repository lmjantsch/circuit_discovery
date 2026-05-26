# Split Stability Experiment Report

## Scope

This report summarizes the split-stability experiment for two attribution
methods:

- `eap_pure`
- `eap_frnorm_secantmlp_bilinear`

The evaluated model/task combinations are:

- `ioi_gpt2`
- `ioi_qwen2.5`
- `ioi_gemma2`
- `mcqa_qwen2.5`
- `mcqa_gemma2`

The circuits were generated on train subsets and evaluated on the full test
split. IOI results use five 100-example train slices:

- `train001-100`
- `train101-200`
- `train201-300`
- `train301-400`
- `train401-500`

The available MCQA stability outputs use five 20-example train slices:

- `train001-020`
- `train021-040`
- `train041-060`
- `train061-080`
- `train081-100`

Circuit overlap is measured with pairwise Jaccard index across the five circuit
sets for each percentage. For each percentage, the circuit mask is built using
the same percentages as `patcher.py`, and the selected top-percentage circuit is
pruned before computing Jaccard.

For five splits, each Jaccard estimate is computed from `5 choose 2 = 10`
pairwise comparisons. The reported values are the mean and standard deviation
over those 10 pairwise Jaccard values.

## Output Files

Main generated summaries:

- `experiments/mib/method_comparison/stability/cpr_summary_clean.csv`
- `experiments/mib/method_comparison/stability/jaccard_all_summary.csv`
- `experiments/mib/method_comparison/stability/jaccard_selected_summary.csv`
- `experiments/mib/method_comparison/stability/score_stats_summary.csv`

Per-condition stability outputs:

- `experiments/mib/method_comparison/stability/{method}/{task}_{model}/summary.csv`
- `experiments/mib/method_comparison/stability/{method}/{task}_{model}/score_mean.pt`
- `experiments/mib/method_comparison/stability/{method}/{task}_{model}/score_std.pt`

## CPR Summary

CPR is `area_under` from the evaluation result pickle.

| Method | Combo | Splits | CPR mean +/- std | CPR min-max |
|---|---|---:|---:|---:|
| `eap_pure` | `ioi_gpt2` | 5 | 1.3226 +/- 0.1084 | 1.2233-1.4582 |
| `eap_pure` | `ioi_qwen2.5` | 5 | 0.7207 +/- 0.5822 | 0.2745-1.4229 |
| `eap_pure` | `ioi_gemma2` | 5 | 1.2889 +/- 0.1996 | 0.9946-1.4879 |
| `eap_pure` | `mcqa_qwen2.5` | 5 | 0.8787 +/- 0.0663 | 0.8133-0.9826 |
| `eap_pure` | `mcqa_gemma2` | 5 | 1.2664 +/- 0.0670 | 1.1618-1.3194 |
| `eap_frnorm_secantmlp_bilinear` | `ioi_gpt2` | 5 | 2.4899 +/- 0.1527 | 2.3411-2.6563 |
| `eap_frnorm_secantmlp_bilinear` | `ioi_qwen2.5` | 5 | 1.8060 +/- 0.0692 | 1.7077-1.8722 |
| `eap_frnorm_secantmlp_bilinear` | `ioi_gemma2` | 5 | 3.5309 +/- 0.0276 | 3.4837-3.5492 |
| `eap_frnorm_secantmlp_bilinear` | `mcqa_qwen2.5` | 5 | 1.8268 +/- 0.0592 | 1.7675-1.9160 |
| `eap_frnorm_secantmlp_bilinear` | `mcqa_gemma2` | 5 | 2.2210 +/- 0.0851 | 2.1423-2.3665 |

## Pairwise Jaccard Mean and Standard Deviation

The table entries are `mean +/- std`.

### eap_pure

| Combo | 0.001 | 0.002 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ioi_gpt2` | 0.3649 +/- 0.4729 | 0.5471 +/- 0.4710 | 0.8820 +/- 0.0259 | 0.8135 +/- 0.0344 | 0.7310 +/- 0.0275 | 0.6431 +/- 0.0320 | 0.5919 +/- 0.0307 | 0.5431 +/- 0.0277 | 0.5894 +/- 0.0228 |
| `ioi_qwen2.5` | 1.0000 +/- 0.0000 | 0.6000 +/- 0.5164 | 0.3761 +/- 0.4904 | 0.3245 +/- 0.2916 | 0.6106 +/- 0.0541 | 0.6095 +/- 0.0116 | 0.5674 +/- 0.0097 | 0.5166 +/- 0.0094 | 0.5381 +/- 0.0091 |
| `ioi_gemma2` | 0.9264 +/- 0.0584 | 0.6585 +/- 0.1976 | 0.6303 +/- 0.1308 | 0.8504 +/- 0.0186 | 0.7984 +/- 0.0186 | 0.7687 +/- 0.0132 | 0.7295 +/- 0.0191 | 0.6872 +/- 0.0221 | 0.6527 +/- 0.0213 |
| `mcqa_qwen2.5` | 0.7410 +/- 0.0582 | 0.7621 +/- 0.0567 | 0.7368 +/- 0.0515 | 0.7677 +/- 0.0475 | 0.7644 +/- 0.0481 | 0.7682 +/- 0.0468 | 0.7743 +/- 0.0426 | 0.7920 +/- 0.0381 | 0.8190 +/- 0.0291 |
| `mcqa_gemma2` | 0.5837 +/- 0.0924 | 0.7524 +/- 0.0756 | 0.7203 +/- 0.0657 | 0.7430 +/- 0.0387 | 0.7206 +/- 0.0464 | 0.7201 +/- 0.0426 | 0.7235 +/- 0.0446 | 0.7413 +/- 0.0371 | 0.7994 +/- 0.0278 |

### eap_frnorm_secantmlp_bilinear

| Combo | 0.001 | 0.002 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ioi_gpt2` | 0.9520 +/- 0.0413 | 0.9038 +/- 0.0474 | 0.8515 +/- 0.0375 | 0.8334 +/- 0.0219 | 0.7926 +/- 0.0266 | 0.7002 +/- 0.0335 | 0.6628 +/- 0.0385 | 0.6010 +/- 0.0306 | 0.6256 +/- 0.0272 |
| `ioi_qwen2.5` | 0.8280 +/- 0.0493 | 0.8320 +/- 0.0191 | 0.7896 +/- 0.0125 | 0.7623 +/- 0.0097 | 0.7159 +/- 0.0127 | 0.6609 +/- 0.0131 | 0.6067 +/- 0.0114 | 0.5582 +/- 0.0143 | 0.5589 +/- 0.0150 |
| `ioi_gemma2` | 0.8739 +/- 0.1148 | 0.9373 +/- 0.0179 | 0.8132 +/- 0.0597 | 0.8693 +/- 0.0105 | 0.8326 +/- 0.0233 | 0.8246 +/- 0.0131 | 0.7781 +/- 0.0174 | 0.7324 +/- 0.0224 | 0.6807 +/- 0.0253 |
| `mcqa_qwen2.5` | 0.8502 +/- 0.0578 | 0.8242 +/- 0.0461 | 0.8129 +/- 0.0436 | 0.8251 +/- 0.0451 | 0.8303 +/- 0.0439 | 0.8320 +/- 0.0392 | 0.8321 +/- 0.0384 | 0.8341 +/- 0.0361 | 0.8434 +/- 0.0280 |
| `mcqa_gemma2` | 0.7658 +/- 0.1273 | 0.7202 +/- 0.0781 | 0.8238 +/- 0.0546 | 0.8083 +/- 0.0437 | 0.8112 +/- 0.0410 | 0.8187 +/- 0.0388 | 0.8202 +/- 0.0366 | 0.8280 +/- 0.0316 | 0.8604 +/- 0.0227 |

`p=1.0` is omitted from the table because it is always 1.0: the full valid-edge
set is selected, so all split circuits are identical at that endpoint.

## Score Tensor Stability

The score tensors are aggregated edgewise across the five train splits. The
saved tensors are:

- `score_mean.pt`: edgewise mean attribution score
- `score_std.pt`: edgewise standard deviation of attribution score

Representative aggregate statistics:

| Method | Combo | mean(abs(score mean)) | mean(score std) | median(score std) |
|---|---|---:|---:|---:|
| `eap_pure` | `ioi_gpt2` | 7.7649e-05 | 1.6381e-05 | 0.0000e+00 |
| `eap_pure` | `ioi_qwen2.5` | 4.3319e-05 | 1.4539e-05 | 0.0000e+00 |
| `eap_pure` | `ioi_gemma2` | 7.5934e-05 | 1.3772e-05 | 0.0000e+00 |
| `eap_pure` | `mcqa_qwen2.5` | 6.5785e-05 | 1.4124e-05 | 0.0000e+00 |
| `eap_pure` | `mcqa_gemma2` | 5.8552e-05 | 1.4972e-05 | 0.0000e+00 |
| `eap_frnorm_secantmlp_bilinear` | `ioi_gpt2` | 3.1295e-05 | 4.4236e-06 | 0.0000e+00 |
| `eap_frnorm_secantmlp_bilinear` | `ioi_qwen2.5` | 1.2130e-05 | 2.5376e-06 | 0.0000e+00 |
| `eap_frnorm_secantmlp_bilinear` | `ioi_gemma2` | 3.0197e-05 | 3.8615e-06 | 0.0000e+00 |
| `eap_frnorm_secantmlp_bilinear` | `mcqa_qwen2.5` | 1.8039e-05 | 2.6876e-06 | 0.0000e+00 |
| `eap_frnorm_secantmlp_bilinear` | `mcqa_gemma2` | 2.3210e-05 | 3.9139e-06 | 0.0000e+00 |

The median score standard deviation is zero because most possible edge slots
have zero or invalid attribution under the sparse valid-edge structure. The
mean score standard deviation is more informative for global score-scale
stability.

## Interpretation

`eap_frnorm_secantmlp_bilinear` is more stable than `eap_pure` overall. It
improves CPR for every evaluated model/task combination and usually gives
higher pairwise Jaccard overlap at small and mid-range percentages.

The largest stability gap appears on `ioi_qwen2.5`. For `eap_pure`, the Jaccard
value drops to 0.3245 +/- 0.2916 at 1 percent, and CPR has high split variance
of 0.7207 +/- 0.5822. For `eap_frnorm_secantmlp_bilinear`, the same 1 percent
Jaccard is 0.7623 +/- 0.0097, and CPR is 1.8060 +/- 0.0692. This indicates that
the bilinear/frozen-norm/secant-MLP variant finds much more reproducible
circuits for this setting.

MCQA circuits are comparatively stable for both methods. Jaccard values mostly
stay between about 0.72 and 0.86 across percentages. The bilinear variant still
improves CPR substantially, but the overlap gap is smaller than in IOI.

For IOI, Jaccard often decreases as the percentage increases from very small
circuits to mid-size circuits. This is expected: the strongest top edges are
often repeatedly selected across train splits, while lower-ranked edges are more
split-sensitive. The 0.005, 0.01, and 0.05 percentages are the most useful
stability region to inspect. The 0.001 percentage can be unstable or artificially
high because pruning leaves very few edges, and the 1.0 endpoint is trivial.

Overall, the results support the conclusion that
`eap_frnorm_secantmlp_bilinear` gives both stronger circuit performance and more
reproducible circuit structure across train splits than `eap_pure`, especially
for IOI.
