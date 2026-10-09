#!/bin/bash
# LAG (Local Approximation Gap) sweep, sequential on one GPU. Usage: lag_all.sh [model:task ...]
cd /home/dacslab/djk/lasse-circuit/circuit_discovery
PY=/raid/conda/envs/dacslab_djk_mib/bin/python
export CUDA_VISIBLE_DEVICES=3
FILT='warning|resource_tracker|Traceback|File "|AttributeError|Exception ignored|Filter:|Loading weights'
CELLS="${@:-gpt2:ioi qwen2.5:ioi gemma2:ioi llama3:ioi gpt2:mcqa qwen2.5:mcqa gemma2:mcqa llama3:mcqa}"
for cell in $CELLS; do
  m=${cell%%:*}; t=${cell##*:}
  echo "===== START $m $t  $(date)" >> experiments/mib/scripts/lag_all.progress
  $PY experiments/mib/scripts/lag_opgap.py --model $m --task $t 2>&1 | grep -v -i -E "$FILT" > experiments/mib/scripts/lag_${m}_${t}.log
  echo "===== DONE  $m $t  $(date)  exit=${PIPESTATUS[0]}" >> experiments/mib/scripts/lag_all.progress
done
echo "ALL DONE $(date)" >> experiments/mib/scripts/lag_all.progress
