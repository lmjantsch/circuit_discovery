
#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=3 nohup bash experiments/mib/scripts/run_evaluation.sh > experiments/mib/scripts/run_evaluation2.log 2>&1 &
# python experiments/mib/MIB-circuit-track/print_results.py --output-dir experiments/mib/results --split test --metric cpr

for method in eap eap_ig_5 eap_no_norm eap_ig_5_no_norm eap_bilin_norm eap_bilin_norm_secmlp; do #eap eap_ig_5 eap_bilin_norm eap_bilin_norm_secmlp; do #eap_no_norm eap_ig_5_no_norm eap eap_ig_5 eap_bilin eap_norm eap_bilin_norm eap_secmlp eap_bilin_norm_secmlp eap_bilin_norm_secmlp_secsoftc # ap eap_ig_5 eap_bilin_norm eap_bilin_norm_secmlp
    python experiments/mib/MIB-circuit-track/run_evaluation.py \
        --models gpt2 qwen2.5 gemma2 --tasks ioi \
        --method $method --ablation patching --level edge --split test \
        --batch-size 16 --head 200 \
        --circuit-dir experiments/mib/circuits \
        --output-dir experiments/mib/results

done