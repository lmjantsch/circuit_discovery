# Log-scale CPR — gemma --no-softcap sweep

Pipeline: data_exploration (linear_transformer attribution + patcher eval). Split=test, signed scores (abs-False). `log_area_under` = log-scale CPR.

Scope: gpt2/qwen2.5/gemma2 × ioi/mcqa. softcap-ON columns are mandatory; `gemma2-nsc` = gemma2 with `--no-softcap` (additional). softcap-ON eap/eap_ig_5/eap_frnorm/eap_igbilin/eap_igbilin_frnorm copied from `/home/dacslab/lasse_jantsch/circuit_discovery`; rest computed here.

> llama3 + arc/arithmetic have circuits in Lasse's repo but were not evaluated (out of this sweep's scope).

### Log-scale CPR (`log_area_under`) — gpt2 / qwen2.5 / gemma2

| method | gpt2/ioi | qwen2.5/ioi | qwen2.5/mcqa | gemma2/ioi | gemma2/mcqa | gemma2/arc_easy | gemma2-nsc/ioi | gemma2-nsc/mcqa | gemma2-nsc/arc_easy |
|---|---|---|---|---|---|---|---|---|---|
| `eap` | 6.1503 | 0.7147 | 4.8105 | 6.5658 | 7.1556 | 9.2042 | 6.3196 | 7.3809 | 8.3840 |
| `eap_ig_5` | 12.4999 | 11.2918 | 7.5309 | 22.3901 | 9.4885 | 12.4905 | 23.6197 | 9.4645 | 12.2187 |
| `eap_frnorm` | 13.0379 | 8.1440 | 5.5808 | 14.5370 | 7.7842 | 8.9022 | 14.7317 | 5.0893 | 7.7342 |
| `eap_igbilin` | 11.1708 | 13.2341 | 7.5049 | 22.5214 | 9.7771 | 11.9918 | 23.0025 | 9.9485 | 11.6381 |
| `eap_secmlp` | 11.1602 | 0.5639 | 6.1767 | 8.5217 | 7.5320 | 8.2123 | 9.6824 | 6.0390 | 6.6372 |
| `eap_igbilin_frnorm` | 15.0065 | 12.9204 | 7.8522 | 22.1686 | 11.4538 | 12.6997 | 22.2350 | 11.0345 | 12.9961 |
| `eap_ig_5_igbilin_frnorm` | 12.3364 | 11.4097 | 8.1390 | 23.8295 | 10.1814 | 11.0885 | 23.7895 | 9.9866 | 10.9130 |
| `eap_igbilin_frnorm_secmlp` | 14.7203 | 12.8986 | 7.8445 | 23.1885 | 10.3605 | 12.2433 | 22.7041 | 11.0814 | 12.9088 |
| `eap_ig_5_igbilin_frnorm_secmlp` | 12.5850 | 11.3585 | 7.9616 | 23.9571 | 9.3088 | 11.6795 | 23.5787 | 9.4740 | 11.8582 |


llama3 (arc/arithmetic; no softcap — eval-only methods from copied circuits, secmlp methods computed here):

### Log-scale CPR (`log_area_under`) — llama3

| method | arc_easy | arc_challenge | arith_add | arith_sub |
|---|---|---|---|---|
| `eap` | 5.5289 | 5.1465 | 1.8601 | 2.8052 |
| `eap_ig_5` | 7.8436 | 8.9751 | 6.6932 | 7.8544 |
| `eap_frnorm` | 2.4762 | 3.0354 | 3.8841 | 2.6349 |
| `eap_igbilin` | 5.5747 | 4.9686 | 8.6383 | 7.8058 |
| `eap_secmlp` | 5.5070 | 5.2397 | 2.2441 | 2.8814 |
| `eap_igbilin_frnorm` | 8.3464 | 8.9443 | 8.7964 | 8.4361 |
| `eap_ig_5_igbilin_frnorm` | 6.9714 | 5.9959 | 8.3349 | 8.2595 |
| `eap_igbilin_frnorm_secmlp` | 9.4813 | 10.3352 | 8.8678 | 8.3943 |
| `eap_ig_5_igbilin_frnorm_secmlp` | 9.1796 | 8.5167 | 8.4993 | 8.5301 |


---

## Linear CPR (`area_under`) — reference

### Linear CPR (`area_under`) — gpt2 / qwen2.5 / gemma2

| method | gpt2/ioi | qwen2.5/ioi | qwen2.5/mcqa | gemma2/ioi | gemma2/mcqa | gemma2/arc_easy | gemma2-nsc/ioi | gemma2-nsc/mcqa | gemma2-nsc/arc_easy |
|---|---|---|---|---|---|---|---|---|---|
| `eap` | 1.2233 | 0.3364 | 0.8074 | 1.3367 | 1.0924 | 1.1966 | 1.3170 | 1.1591 | 1.1107 |
| `eap_ig_5` | 2.0681 | 1.7938 | 1.1347 | 3.5546 | 1.2819 | 1.7512 | 3.7857 | 1.3259 | 1.7099 |
| `eap_frnorm` | 2.4135 | 1.3724 | 0.9020 | 2.6623 | 1.2805 | 1.3319 | 2.7476 | 0.7717 | 1.0673 |
| `eap_igbilin` | 2.1201 | 1.9498 | 1.0746 | 3.4895 | 1.4405 | 1.6318 | 3.7298 | 1.4654 | 1.6300 |
| `eap_secmlp` | 2.1250 | 0.2990 | 1.0335 | 1.7502 | 1.3553 | 1.1337 | 2.0592 | 0.9717 | 0.9592 |
| `eap_igbilin_frnorm` | 2.6154 | 1.8848 | 1.1273 | 3.3882 | 1.7164 | 1.8124 | 3.5142 | 1.6638 | 1.8219 |
| `eap_ig_5_igbilin_frnorm` | 1.9924 | 1.6064 | 1.2163 | 3.5553 | 1.5102 | 1.5686 | 3.5772 | 1.3989 | 1.5338 |
| `eap_igbilin_frnorm_secmlp` | 2.6048 | 1.9073 | 1.0998 | 3.4564 | 1.5114 | 1.7152 | 3.4954 | 1.5897 | 1.8234 |
| `eap_ig_5_igbilin_frnorm_secmlp` | 2.0924 | 1.6073 | 1.1286 | 3.5308 | 1.3336 | 1.6070 | 3.4967 | 1.3327 | 1.6744 |


llama3 (arc/arithmetic):

### Linear CPR (`area_under`) — llama3

| method | arc_easy | arc_challenge | arith_add | arith_sub |
|---|---|---|---|---|
| `eap` | 0.7872 | 0.7811 | 0.4453 | 0.5806 |
| `eap_ig_5` | 1.0336 | 1.1017 | 0.9417 | 1.0765 |
| `eap_frnorm` | 0.5720 | 0.6302 | 0.6948 | 0.5466 |
| `eap_igbilin` | 0.7578 | 0.6695 | 1.1331 | 1.0261 |
| `eap_secmlp` | 0.8795 | 0.8294 | 0.4923 | 0.6010 |
| `eap_igbilin_frnorm` | 1.1772 | 1.2573 | 1.1720 | 1.1048 |
| `eap_ig_5_igbilin_frnorm` | 0.9671 | 0.7707 | 1.0999 | 1.0935 |
| `eap_igbilin_frnorm_secmlp` | 1.3749 | 1.4208 | 1.1786 | 1.0949 |
| `eap_ig_5_igbilin_frnorm_secmlp` | 1.2521 | 1.1267 | 1.1088 | 1.1244 |
