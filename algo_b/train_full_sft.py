"""
Algorithm B — Full-Parameter SFT with DeepSpeed ZeRO-3
=======================================================
Project: RFP PSX-20260510 — Power Systems AI Model
Hardware: 4× A100 80GB (single node)
Base model: Qwen/Qwen2.5-14B (Apache 2.0)

Full-parameter supervised fine-tuning using BF16 precision and DeepSpeed ZeRO-3
to shard optimizer states, gradients, and model parameters across all 4 GPUs.
No LoRA — every parameter is updated, giving stronger domain fluency than QLoRA
at the cost of higher compute and memory requirements.

USAGE (single node, 4 GPUs):
    # Recommended — via torchrun:
    torchrun --nproc_per_node 4 algo_b/train_full_sft.py

    # With overrides:
    torchrun --nproc_per_node 4 algo_b/train_full_sft.py \\
        --train_file ./data/processed/train.json \\
        --eval_file  ./data/processed/eval.json \\
        --output_dir ./output/algo_b_full_sft \\
        --num_epochs 3

    # Resume from checkpoint:
    torchrun --nproc_per_node 4 algo_b/train_full_sft.py \\
        --resume_from ./output/algo_b_full_sft/checkpoint-500

REQUIREMENTS:
    pip install transformers datasets accelerate deepspeed flash-attn trl
    pip install --no-build-isolation flash-attn   # requires CUDA ≥ 11.6

DEEPSPEED CONFIG:
    Uses algo_b/ds_zero3.json (ZeRO stage 3, BF16, overlap_comm=True)
    This is automatically picked up by the Accelerate/DeepSpeed integration.

OUTPUT:
    ./output/algo_b_full_sft/          — full model weights (safetensors shards)
    ./output/algo_b_full_sft/tokenizer — tokenizer files
    ./logs/algo_b/                     — training logs
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model
    model_id: str = "Qwen/Qwen2.5-14B"
    trust_remote_code: bool = True
    use_flash_attention: bool = True    # requires flash-attn package + A100/H100

    # Data
    train_file: str = "./data/processed/train.json"
    eval_file:  str = "./data/processed/eval.json"
    max_seq_length: int = 4096

    # Training hyperparameters
    # Lower LR than QLoRA (2e-4) because we're updating ALL parameters
    output_dir: str = "./output/algo_b_full_sft"
    num_epochs: int = 3
    per_device_batch_size: int = 2      # 2 per GPU × 4 GPUs = 8 total
    gradient_accumulation: int = 4      # effective batch = 8 × 4 = 32
    learning_rate: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    # DeepSpeed
    deepspeed_config: str = "./algo_b/ds_zero3.json"

    # System prompt
    system_prompt: str = (
        "You are an expert power systems engineer with deep knowledge of "
        "electrical grid operation, protection systems, stability analysis, "
        "load flow, fault calculations, energy markets, and power electronics. "
        "Provide accurate, technically precise, and practical answers."
    )

    # W&B (set to None to disable)
    wandb_project: Optional[str] = "psx-20260510-algo-b"
    wandb_run_name: Optional[str] = "full-sft-qwen2.5-14b"


CFG = Config()


# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Algorithm B — Full-parameter SFT with DeepSpeed ZeRO-3"
    )
    parser.add_argument("--model_id",           default=CFG.model_id)
    parser.add_argument("--train_file",         default=CFG.train_file)
    parser.add_argument("--eval_file",          default=CFG.eval_file)
    parser.add_argument("--output_dir",         default=CFG.output_dir)
    parser.add_argument("--num_epochs",         type=int,   default=CFG.num_epochs)
    parser.add_argument("--learning_rate",      type=float, default=CFG.learning_rate)
    parser.add_argument("--max_seq_length",     type=int,   default=CFG.max_seq_length)
    parser.add_argument("--per_device_batch_size", type=int, default=CFG.per_device_batch_size)
    parser.add_argument("--gradient_accumulation", type=int, default=CFG.gradient_accumulation)
    parser.add_argument("--deepspeed_config",   default=CFG.deepspeed_config)
    parser.add_argument("--resume_from",        default=None,
                        help="Checkpoint directory to resume from")
    parser.add_argument("--no_flash_attn",      action="store_true",
                        help="Disable FlashAttention-2 (use if not on A100/H100)")
    parser.add_argument("--no_wandb",           action="store_true",
                        help="Disable W&B logging")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

ALPACA_PROMPT = """\
Below is an instruction from a power systems engineering context. \
Write a technically accurate and detailed response.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def load_and_format_dataset(train_file: str, eval_file: str, tokenizer, system_prompt: str):
    """Load Alpaca JSON and format into chat-style prompts."""
    from datasets import Dataset

    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    train_data = load_json(train_file)
    eval_data  = load_json(eval_file) if Path(eval_file).exists() else []

    eos = tokenizer.eos_token or ""

    def format_examples(examples):
        texts = []
        instructions = examples["instruction"]
        inputs   = examples.get("input",  [""] * len(instructions))
        outputs  = examples["output"]

        for inst, inp, out in zip(instructions, inputs, outputs):
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": inst if not inp else f"{inst}\n\n{inp}"},
                    {"role": "assistant", "content": out},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False) + eos
            else:
                text = ALPACA_PROMPT.format(inst, inp, out) + eos
            texts.append(text)
        return {"text": texts}

    train_ds = Dataset.from_list(train_data)
    train_ds = train_ds.map(format_examples, batched=True,
                             remove_columns=train_ds.column_names)

    eval_ds = None
    if eval_data:
        eval_ds = Dataset.from_list(eval_data)
        eval_ds = eval_ds.map(format_examples, batched=True,
                               remove_columns=eval_ds.column_names)

    return train_ds, eval_ds


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Apply overrides
    CFG.model_id              = args.model_id
    CFG.train_file            = args.train_file
    CFG.eval_file             = args.eval_file
    CFG.output_dir            = args.output_dir
    CFG.num_epochs            = args.num_epochs
    CFG.learning_rate         = args.learning_rate
    CFG.max_seq_length        = args.max_seq_length
    CFG.per_device_batch_size = args.per_device_batch_size
    CFG.gradient_accumulation = args.gradient_accumulation
    CFG.deepspeed_config      = args.deepspeed_config
    CFG.use_flash_attention   = not args.no_flash_attn

    if args.no_wandb:
        CFG.wandb_project = None

    # ── Logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── W&B ──
    if CFG.wandb_project:
        os.environ["WANDB_PROJECT"] = CFG.wandb_project
        if CFG.wandb_run_name:
            os.environ["WANDB_NAME"] = CFG.wandb_run_name
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # ── Validate inputs ──
    if not Path(CFG.train_file).exists():
        logger.error(f"Training file not found: {CFG.train_file}")
        logger.error("  Run: python data/pipeline.py --sources_dir ./raw_docs")
        sys.exit(1)

    if not Path(CFG.deepspeed_config).exists():
        logger.error(f"DeepSpeed config not found: {CFG.deepspeed_config}")
        sys.exit(1)

    # ── Imports ──
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from trl import SFTTrainer

    # ── GPU check ──
    if not torch.cuda.is_available():
        logger.error("No GPU detected. Algorithm B requires 4× A100 80GB GPUs.")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    logger.info(f"GPUs detected: {num_gpus}")
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
        logger.info(f"  GPU {i}: {name} ({mem:.0f} GB)")

    if num_gpus < 2:
        logger.warning(
            "Algorithm B is designed for 4× A100 80GB. "
            "Running on fewer GPUs will likely OOM on a 14B model in full BF16. "
            "Consider Algorithm A (QLoRA) for single-GPU training."
        )

    # ── Load tokenizer ──
    logger.info(f"\n[STEP 1] Loading tokenizer: {CFG.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        CFG.model_id,
        trust_remote_code=CFG.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load model ──
    logger.info(f"\n[STEP 2] Loading model in BF16: {CFG.model_id}")

    attn_impl = "flash_attention_2" if CFG.use_flash_attention else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=CFG.trust_remote_code,
        attn_implementation=attn_impl,
        # ZeRO-3 handles device placement — do NOT use device_map="auto" here
    )

    # Disable model parallelism flags that conflict with DeepSpeed
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    logger.info(f"  Total parameters:     {total_params:.1f}B")
    logger.info(f"  Trainable parameters: {trainable:.1f}B  (100% — full SFT)")

    # ── Load dataset ──
    logger.info(f"\n[STEP 3] Loading dataset")
    train_ds, eval_ds = load_and_format_dataset(
        CFG.train_file, CFG.eval_file, tokenizer, CFG.system_prompt
    )
    logger.info(f"  Train: {len(train_ds):,} examples")
    if eval_ds:
        logger.info(f"  Eval:  {len(eval_ds):,} examples")

    # ── Training arguments ──
    logger.info(f"\n[STEP 4] Configuring trainer")
    os.makedirs(CFG.output_dir, exist_ok=True)

    effective_batch = CFG.per_device_batch_size * CFG.gradient_accumulation * num_gpus
    logger.info(f"  Effective batch size: {effective_batch}")
    logger.info(f"  Learning rate:        {CFG.learning_rate}")
    logger.info(f"  Epochs:               {CFG.num_epochs}")

    training_args = TrainingArguments(
        output_dir=CFG.output_dir,

        # Batch & accumulation
        per_device_train_batch_size=CFG.per_device_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation,

        # Training schedule
        num_train_epochs=CFG.num_epochs,
        learning_rate=CFG.learning_rate,
        lr_scheduler_type=CFG.lr_scheduler,
        warmup_ratio=CFG.warmup_ratio,
        weight_decay=CFG.weight_decay,
        max_grad_norm=CFG.max_grad_norm,

        # Precision — BF16, no quantization (full precision)
        bf16=True,
        fp16=False,

        # Optimizer — standard AdamW (DeepSpeed handles memory offloading)
        optim="adamw_torch",

        # Gradient checkpointing — saves ~30% memory at cost of ~20% speed
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging & checkpointing
        logging_steps=CFG.logging_steps,
        save_steps=CFG.save_steps,
        save_total_limit=CFG.save_total_limit,
        evaluation_strategy="steps" if eval_ds else "no",
        eval_steps=CFG.eval_steps if eval_ds else None,
        load_best_model_at_end=bool(eval_ds),

        # DeepSpeed ZeRO-3
        deepspeed=CFG.deepspeed_config,

        # Misc
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=True,
        report_to="wandb" if CFG.wandb_project else "none",

        # Required for ZeRO-3 multi-GPU
        ddp_find_unused_parameters=False,
        ddp_timeout=180000,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=CFG.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=training_args,
    )

    # ── Train ──
    steps_per_epoch = len(train_ds) // effective_batch
    total_steps     = steps_per_epoch * CFG.num_epochs
    logger.info(f"\n[STEP 5] Starting training")
    logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
    logger.info(f"  Total steps:     {total_steps:,}")
    logger.info(f"  Checkpoints every {CFG.save_steps} steps → {CFG.output_dir}")

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from)

    runtime_hours = trainer_stats.metrics["train_runtime"] / 3600
    logger.info(f"\n[DONE] Training complete!")
    logger.info(f"  Total time: {runtime_hours:.1f} hours")
    logger.info(f"  Final loss: {trainer_stats.metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"  GPU-hours:  {runtime_hours * num_gpus:.1f}")

    # ── Save model ──
    # ZeRO-3 gathers all shards on rank 0 before saving
    # (controlled by stage3_gather_16bit_weights_on_model_save in ds_zero3.json)
    logger.info(f"\n[STEP 6] Saving model to {CFG.output_dir}")
    trainer.save_model(CFG.output_dir)
    tokenizer.save_pretrained(CFG.output_dir)
    logger.info("  Model saved.")

    # ── Quick inference test (rank 0 only) ──
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        logger.info(f"\n[STEP 7] Quick inference test...")
        import torch
        model.eval()

        test_question = "Explain the equal area criterion for transient stability analysis."
        messages = [
            {"role": "system",  "content": CFG.system_prompt},
            {"role": "user",    "content": test_question},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.shape[1]:], skip_special_tokens=True
        )
        logger.info(f"\n  Q: {test_question}")
        logger.info(f"\n  A: {response[:500]}...")

        logger.info(f"\n{'='*60}")
        logger.info("ALGORITHM B COMPLETE")
        logger.info(f"  Model saved to: {CFG.output_dir}")
        logger.info(f"  Next: Run eval/evaluate.py to benchmark the model")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
