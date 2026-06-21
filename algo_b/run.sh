#!/bin/bash
# ============================================================
# Algorithm B — Full-Parameter SFT Launch Script
# Run this from the training/ root directory:
#   bash algo_b/run.sh
# ============================================================

set -e   # exit on first error

# ── Configuration ──────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-14B}"
TRAIN_FILE="${TRAIN_FILE:-./data/processed/train.json}"
EVAL_FILE="${EVAL_FILE:-./data/processed/eval.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/algo_b_full_sft}"
DS_CONFIG="${DS_CONFIG:-./algo_b/ds_zero3.json}"
NPROC="${NPROC:-4}"                    # number of GPUs
MASTER_PORT="${MASTER_PORT:-29500}"
RESUME_FROM="${RESUME_FROM:-}"         # set to checkpoint path to resume

# ── Pre-flight checks ──────────────────────────────────────
echo "============================================================"
echo " Algorithm B — Full-Parameter SFT (DeepSpeed ZeRO-3)"
echo "============================================================"
echo " Model:      $MODEL_PATH"
echo " Train data: $TRAIN_FILE"
echo " Output:     $OUTPUT_DIR"
echo " GPUs:       $NPROC"
echo " DeepSpeed:  $DS_CONFIG"
echo "============================================================"

# Check training data exists
if [ ! -f "$TRAIN_FILE" ]; then
    echo "[ERROR] Training file not found: $TRAIN_FILE"
    echo "  Run first: python data/pipeline.py --sources_dir ./raw_docs"
    exit 1
fi

# Check DeepSpeed config exists
if [ ! -f "$DS_CONFIG" ]; then
    echo "[ERROR] DeepSpeed config not found: $DS_CONFIG"
    exit 1
fi

# Check GPU count
ACTUAL_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$ACTUAL_GPUS" -lt "$NPROC" ]; then
    echo "[WARNING] Requested $NPROC GPUs but only $ACTUAL_GPUS detected."
    echo "  Set NPROC=$ACTUAL_GPUS or add more GPUs."
    NPROC=$ACTUAL_GPUS
fi

echo ""
echo "[INFO] Starting training with $NPROC GPUs..."
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/algo_b

# ── Launch ─────────────────────────────────────────────────
RESUME_ARG=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_ARG="--resume_from $RESUME_FROM"
    echo "[INFO] Resuming from checkpoint: $RESUME_FROM"
fi

torchrun \
    --nproc_per_node "$NPROC" \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port "$MASTER_PORT" \
    algo_b/train_full_sft.py \
        --model_id         "$MODEL_PATH" \
        --train_file       "$TRAIN_FILE" \
        --eval_file        "$EVAL_FILE" \
        --output_dir       "$OUTPUT_DIR" \
        --deepspeed_config "$DS_CONFIG" \
        $RESUME_ARG \
    2>&1 | tee logs/algo_b/train_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "[DONE] Training log saved to logs/algo_b/"
