#!/bin/bash
# =============================================================================
# Auto Pipeline v3 — Fast DPA Attribution + Eval + Report (--head 200)
#
# Uses GPU 1 (GPU 2 is taken by another user).
# Uses --head 200 for Llama3 evaluations to keep runtime manageable.
#
# Flow:
# 1. Run FastEdgeTracer for 5 missing Llama3 tasks on GPU 1
# 2. Re-run ioi_llama3 eval with --head 200 (for consistency)
# 3. Run evals for all 6 Llama3 tasks with --head 200
# 4. Generate final report
# =============================================================================
set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="/home/dacslab/lasse_jantsch/circuit_discovery"
MIB_DIR="$PROJECT_ROOT/experiments/mib/MIB-circuit-track"
CIRCUIT_DIR="$MIB_DIR/circuits"
RESULT_DIR="$MIB_DIR/results"
DPA_DIR="$CIRCUIT_DIR/dpa_patching_edge"
PYTHON_MIB="/raid/conda/envs/dacslab_lmj_mib/bin/python"
PYTHON_DPA="/raid/conda/envs/dacslab_lasse/bin/python"
GPU=1

LLAMA_TASKS=("ioi" "mcqa" "arithmetic_addition" "arithmetic_subtraction" "arc_easy" "arc_challenge")

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# -------------------------------------------------------
# Phase 1: Fast attribution for missing tasks
# -------------------------------------------------------
log "=== Phase 1: FastEdgeTracer on GPU $GPU (missing tasks only) ==="

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
    "$PYTHON_DPA" -m experiments.mib.run_attribution \
    > "$PROJECT_ROOT/experiments/mib/scripts/fast_attribution_log.txt" 2>&1 || {
    log "ERROR: Fast attribution failed"
    exit 1
}

log "  Fast attribution complete"
n_files=$(find "$DPA_DIR" -name "importances.json" 2>/dev/null | wc -l)
log "  Total importances.json files: $n_files / 8"

# -------------------------------------------------------
# Phase 2: Run Llama3 evals with --head 200
# -------------------------------------------------------
log ""
log "=== Phase 2: Llama3 evaluations (--head 200) on GPU $GPU ==="

cd "$MIB_DIR"

for task in "${LLAMA_TASKS[@]}"; do
    task_col="${task//_/-}_llama3"
    circuit_file="$DPA_DIR/$task_col/importances.json"
    if [ ! -f "$circuit_file" ]; then
        log "  Skip $task_col (no circuit file)"
        continue
    fi

    log "  Evaluating $task_col on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
        "$PYTHON_MIB" run_evaluation.py \
        --models llama3 \
        --tasks "$task" \
        --method dpa \
        --ablation patching \
        --level edge \
        --split validation \
        --batch-size 4 \
        --head 200 \
        --circuit-dir "$CIRCUIT_DIR" \
        --output-dir "$RESULT_DIR" \
        >> "$PROJECT_ROOT/experiments/mib/scripts/eval_llama_all_log.txt" 2>&1 || {
        log "  ERROR evaluating $task_col (continuing)"
    }
done

# -------------------------------------------------------
# Phase 3: Generate final report
# -------------------------------------------------------
log ""
log "=== Phase 3: Generate experiment_results.md ==="

cd "$PROJECT_ROOT"
"$PYTHON_DPA" experiments/mib/create_report.py \
    "$RESULT_DIR" \
    "$RESULT_DIR/experiment_results.md"

log ""
log "=== AUTO PIPELINE V3 DONE ==="
log "Report: $RESULT_DIR/experiment_results.md"
cat "$RESULT_DIR/experiment_results.md"