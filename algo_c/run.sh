#!/bin/bash
# ============================================================
# Algorithm C — DPO Training Launch Script
# Run from the training/ root directory:
#   bash algo_c/run.sh
# ============================================================

set -e   # exit on first error

# ── Configuration ──────────────────────────────────────────
SFT_MODEL="${SFT_MODEL:-./output/algo_a_merged}"
TRAIN_FILE="${TRAIN_FILE:-./data/dpo/dpo_train.json}"
EVAL_FILE="${EVAL_FILE:-./data/dpo/dpo_eval.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/algo_c_dpo}"
MERGED_OUTPUT="${MERGED_OUTPUT:-./output/algo_c_merged}"

# DPO hyperparameters (safe defaults — adjust if needed)
BETA="${BETA:-0.1}"
LR="${LR:-5e-7}"
EPOCHS="${EPOCHS:-1}"

echo "============================================================"
echo " Algorithm C — Direct Preference Optimization (DPO)"
echo "============================================================"
echo " SFT model:   $SFT_MODEL"
echo " Train data:  $TRAIN_FILE"
echo " Eval data:   $EVAL_FILE"
echo " Output:      $OUTPUT_DIR"
echo " Beta:        $BETA"
echo " LR:          $LR"
echo " Epochs:      $EPOCHS"
echo "============================================================"

# ── Pre-flight checks ──────────────────────────────────────

# Check SFT model exists
if [ ! -d "$SFT_MODEL" ]; then
    echo "[ERROR] SFT model not found: $SFT_MODEL"
    echo "  Run algo_a or algo_b first, then merge the adapter:"
    echo "    python algo_a/train_qlora.py --merge"
    exit 1
fi

# Check DPO dataset exists — build it first if missing
if [ ! -f "$TRAIN_FILE" ]; then
    echo "[INFO] DPO dataset not found. Building it now..."
    echo "       This requires ANTHROPIC_API_KEY and the SFT model."

    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "[ERROR] ANTHROPIC_API_KEY not set."
        echo "  Run: export \$(cat data/.env | grep -v '^#' | xargs)"
        exit 1
    fi

    mkdir -p ./data/dpo
    python algo_c/build_dpo_dataset.py \
        --sft_model    "$SFT_MODEL" \
        --questions    ./data/processed/train.json \
        --output_dir   ./data/dpo \
        --strategy     auto \
        --num_questions 1000

    echo "[INFO] DPO dataset built."
fi

# Check GPU
GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPUS" -lt 1 ]; then
    echo "[ERROR] No GPUs detected. Algorithm C requires at least 1× A100 80GB."
    exit 1
fi
echo "[INFO] Detected $GPUS GPU(s). Using 1 GPU for DPO training."

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/algo_c

echo ""
echo "[INFO] Starting DPO training..."
echo ""

# ── Train ─────────────────────────────────────────────────
python algo_c/train_dpo.py \
    --sft_model     "$SFT_MODEL" \
    --train_file    "$TRAIN_FILE" \
    --eval_file     "$EVAL_FILE" \
    --output_dir    "$OUTPUT_DIR" \
    --beta          "$BETA" \
    --learning_rate "$LR" \
    --num_epochs    "$EPOCHS" \
    2>&1 | tee logs/algo_c/train_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "[INFO] Training complete. Merging adapter..."
echo ""

# ── Merge ─────────────────────────────────────────────────
python algo_c/train_dpo.py \
    --merge \
    --sft_model      "$SFT_MODEL" \
    --adapter_path   "$OUTPUT_DIR/final_adapter" \
    --merged_output  "$MERGED_OUTPUT"

echo ""
echo "[INFO] Running inference test on merged model..."
python algo_c/train_dpo.py \
    --test \
    --merged_output "$MERGED_OUTPUT"

echo ""
echo "============================================================"
echo " Algorithm C DONE"
echo " Merged model: $MERGED_OUTPUT"
echo "============================================================"
echo ""
echo "NEXT STEPS:"
echo "  Run benchmark evaluation:"
echo "  python eval/evaluate.py --model $MERGED_OUTPUT"
echo ""
echo "  Log saved to: logs/algo_c/"
