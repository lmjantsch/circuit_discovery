#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=3 nohup bash experiments/mib/scripts/run_attribution.sh > experiments/mib/scripts/run_attribution.log 2>&1 &

set -euo pipefail

MODELS="gpt2 qwen2.5 gemma2"
TASKS="ioi mcqa" # mcqa
OUT="experiments/mib/circuits"
BASE="--tasks $TASKS --output-dir $OUT --batch-size 16"

# (1) EAP — counterfactual, plain gradient
python -m experiments.mib.run_attribution $BASE \
    --models $MODELS \
    --method-name eap \
    --use-counterfactual \
    --force

# (2) EAP-IG-inputs — counterfactual, 5 integration steps
python -m experiments.mib.run_attribution $BASE \
    --models $MODELS \
    --method-name eap_ig_5 \
    --use-counterfactual \
    --integration-steps 5 \
    --force

python -m experiments.mib.run_attribution $BASE \
    --models $MODELS \
    --method-name eap_no_norm \
    --use-counterfactual \
    --ignore-norm \
    --force

python -m experiments.mib.run_attribution $BASE \
    --models $MODELS \
    --method-name eap_ig_5_no_norm \
    --use-counterfactual \
    --integration-steps 5 \
    --ignore-norm \
    --force

# python -m experiments.mib.run_attribution $BASE \
#     --method-name eap_bilin \
#     --use-counterfactual \
#     --matmul-fn bilinear_matmul \
#     --mul-fn bilinear_mul \

# python -m experiments.mib.run_attribution $BASE \
#     --method-name eap_norm \
#     --use-counterfactual \
#     --frozen-norm

python -m experiments.mib.run_attribution $BASE \
    --models $MODELS \
    --method-name eap_bilin_norm \
    --use-counterfactual \
    --matmul-fn bilinear_matmul \
    --mul-fn bilinear_mul \
    --frozen-norm \
    --force

# python -m experiments.mib.run_attribution $BASE \
#     --method-name eap_secmlp \
#     --use-counterfactual \
#     --mlp-act-fn secant_gelu_tanh

python -m experiments.mib.run_attribution $BASE \
    --models gpt2 gemma2\
    --method-name eap_bilin_norm_secmlp \
    --use-counterfactual \
    --matmul-fn bilinear_matmul \
    --mul-fn bilinear_mul \
    --mlp-act-fn secant_gelu_tanh \
    --frozen-norm \
    --force
    
python -m experiments.mib.run_attribution $BASE \
    --models qwen2.5 \
    --method-name eap_bilin_norm_secmlp \
    --use-counterfactual \
    --matmul-fn bilinear_matmul \
    --mul-fn bilinear_mul \
    --mlp-act-fn secant_silu \
    --frozen-norm \
    --force
# python -m experiments.mib.run_attribution $BASE \
#     --method-name eap_bilin_norm_secmlp_secsoftc \
#     --use-counterfactual \
#     --matmul-fn bilinear_matmul \
#     --mul-fn bilinear_mul \
#     --mlp-act-fn secant_gelu_tanh \
#     --attn-softcap-fn secant_tanh \
#     --frozen-norm