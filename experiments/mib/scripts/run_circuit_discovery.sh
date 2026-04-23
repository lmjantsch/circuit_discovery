#!/bin/bash
# Rerun Llama3 preliminary tasks on FULL test split (matches MIB paper).
# No --head flag — uses full test set samples.
set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="/home/dacslab/djk/DPA/DPA_Tracing"
MIB_DIR="$PROJECT_ROOT/experiments/mib/MIB-circuit-track"
RESULT_DIR="$MIB_DIR/results"
CIRCUIT_DIR="$RESULT_DIR/circuits"
PYTHON_MIB="/raid/conda/envs/dacslab_lmj_mib/bin/python"
PYTHON_DPA="/raid/conda/envs/dacslab_djk_dpa/bin/python"
GPU=1

# Preliminary tasks to rerun on test split
TASKS=("ioi" "arithmetic_addition" "arithmetic_subtraction")

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

log "=== Llama3 Preliminary Rerun on TEST split (full samples) ==="

cd "$MIB_DIR"

for task in "${TASKS[@]}"; do
    task_col="${task//_/-}_llama3"
    log "  Evaluating $task_col on test split (GPU $GPU)..."
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
        "$PYTHON_MIB" run_evaluation.py \
        --models llama3 \
        --tasks "$task" \
        --method dpa \
        --ablation patching \
        --level edge \
        --split test \
        --batch-size 4 \
        --circuit-dir "$CIRCUIT_DIR" \
        --output-dir "$RESULT_DIR" \
        >> "$PROJECT_ROOT/experiments/mib/scripts/eval_llama_test_log.txt" 2>&1 || {
        log "  ERROR evaluating $task_col (continuing)"
    }
done

log ""
log "=== Generate updated report ==="
cd "$PROJECT_ROOT"
"$PYTHON_DPA" experiments/mib/generate_report.py \
    "$RESULT_DIR" \
    "$RESULT_DIR/experiment_results.md"

log "=== DONE ==="
log "Updated report: $PROJECT_ROOT/experiments/mib/docs/experiment_results.md"