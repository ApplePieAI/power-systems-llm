"""
train_dpo.py — Algorithm C: Direct Preference Optimization
===========================================================
Project: RFP PSX-20260510 — Power Systems AI Model

DPO aligns the SFT model to prefer accurate engineering answers
over hallucinated or incomplete ones, WITHOUT needing a reward model.

The training objective is:
    L_DPO = -log σ(β · log[π(chosen)/π_ref(chosen)] - β · log[π(rejected)/π_ref(rejected)])

where β controls how much we allow deviation from the SFT policy.
Lower β (0.1) → stay close to the SFT model, safer but less aggressive.

REQUIREMENTS:
    - A merged SFT model (algo_a or algo_b output)
    - A DPO preference dataset (triplets from build_dpo_dataset.py)
    - 1× A100 80GB GPU (or 2× for headroom)

USAGE:
    python algo_c/train_dpo.py \\
        --sft_model    ./output/algo_a_merged \\
        --train_file   ./data/dpo/dpo_train.json \\
        --eval_file    ./data/dpo/dpo_eval.json \\
        --output_dir   ./output/algo_c_dpo

OUTPUT:
    output/algo_c_dpo/
      final_adapter/     LoRA adapter weights (merge later with --merge flag)
      logs/              Training logs
      trainer_state.json Loss curves and metrics

KEY METRICS TO WATCH (in W&B or logs):
    rewards/margins   → should increase over training
    rewards/chosen    → should increase
    rewards/rejected  → should decrease
    eval/loss         → should decrease then plateau
    If rewards/margins stays near zero: β is too high, lower it.
    If rewards/margins collapses: overfitting, lower lr or epochs.
"""

import os
import sys
import json
import logging
import argparse
import dataclasses
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import DPOTrainer, DPOConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class DPOTrainConfig:
    # Model
    sft_model:       str   = "./output/algo_a_merged"
    output_dir:      str   = "./output/algo_c_dpo"

    # Data
    train_file:      str   = "./data/dpo/dpo_train.json"
    eval_file:       str   = "./data/dpo/dpo_eval.json"

    # DPO hyperparameters
    beta:            float = 0.1     # KL penalty; lower = closer to SFT policy
    max_prompt_length: int = 512     # truncate prompts longer than this
    max_length:      int   = 1024    # max total sequence length

    # LoRA on top of SFT (keeps memory low, adds flexibility)
    lora_r:          int   = 32
    lora_alpha:      int   = 64
    lora_dropout:    float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj")

    # Training
    num_epochs:      int   = 1        # DPO only needs 1–3 epochs
    learning_rate:   float = 5e-7     # ~100× smaller than SFT — critical
    per_device_batch_size: int = 1    # DPO stores 2 responses per sample
    gradient_accumulation_steps: int = 8   # effective batch = 8
    warmup_ratio:    float = 0.1
    lr_scheduler:    str   = "cosine"
    weight_decay:    float = 0.01
    max_grad_norm:   float = 0.3

    # Evaluation
    eval_steps:      int   = 50
    save_steps:      int   = 100
    logging_steps:   int   = 10

    # System
    seed:            int   = 42
    bf16:            bool  = True    # A100 supports BF16
    gradient_checkpointing: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_dpo_dataset(path: str) -> Dataset:
    """
    Load DPO triplets from JSON file.
    Expected format: [{"prompt": "...", "chosen": "...", "rejected": "..."}]
    TRL DPOTrainer expects exactly these three fields.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Validate and normalize
    valid = []
    for i, item in enumerate(data):
        if not all(k in item for k in ("prompt", "chosen", "rejected")):
            logger.warning(f"Skipping item {i}: missing required fields")
            continue
        if len(item["prompt"].strip()) < 10:
            logger.warning(f"Skipping item {i}: prompt too short")
            continue
        if item["chosen"].strip() == item["rejected"].strip():
            logger.warning(f"Skipping item {i}: chosen == rejected (no signal)")
            continue
        valid.append({
            "prompt":   item["prompt"].strip(),
            "chosen":   item["chosen"].strip(),
            "rejected": item["rejected"].strip(),
        })

    logger.info(f"Loaded {len(valid):,} valid triplets from {path} "
                f"({len(data) - len(valid)} skipped)")
    return Dataset.from_list(valid)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: DPOTrainConfig):
    """
    Load the SFT model with LoRA adapters for DPO.
    The reference model is automatically handled by TRL DPOTrainer
    (it keeps a frozen copy of the base model internally).
    """
    logger.info(f"Loading tokenizer from: {cfg.sft_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.sft_model, trust_remote_code=True
    )
    tokenizer.padding_side = "left"    # DPO requires left-padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from: {cfg.sft_model}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.sft_model,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()

    # Add LoRA adapters for DPO
    logger.info("Adding LoRA adapters (r=%d, α=%d)...", cfg.lora_r, cfg.lora_alpha)
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: DPOTrainConfig) -> None:
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_dataset = load_dpo_dataset(cfg.train_file)
    eval_dataset  = load_dpo_dataset(cfg.eval_file)
    logger.info(f"Train: {len(train_dataset):,}  |  Eval: {len(eval_dataset):,}")

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # DPO training config
    dpo_config = DPOConfig(
        output_dir=cfg.output_dir,
        beta=cfg.beta,
        max_prompt_length=cfg.max_prompt_length,
        max_length=cfg.max_length,

        # Training loop
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        # Optimisation
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        optim="paged_adamw_32bit",

        # Precision
        bf16=cfg.bf16,
        fp16=not cfg.bf16,

        # Gradient checkpointing — saves ~30% memory
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Evaluation and checkpointing
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_steps=cfg.logging_steps,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"],
        run_name="algo_c_dpo",

        # Misc
        dataloader_num_workers=2,
        seed=cfg.seed,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ── Train ──
    logger.info("Starting DPO training...")
    logger.info(f"  β = {cfg.beta}")
    logger.info(f"  LR = {cfg.learning_rate}")
    logger.info(f"  Epochs = {cfg.num_epochs}")
    logger.info(f"  Effective batch size = "
                f"{cfg.per_device_batch_size * cfg.gradient_accumulation_steps}")

    # Check for resume checkpoint
    resume = False
    checkpoints = sorted(output_dir.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]))
    if checkpoints:
        latest = checkpoints[-1]
        logger.info(f"Resuming from checkpoint: {latest}")
        resume = str(latest)

    result = trainer.train(resume_from_checkpoint=resume if resume else None)

    logger.info(f"\nTraining complete.")
    logger.info(f"  Steps:       {result.global_step}")
    logger.info(f"  Train loss:  {result.training_loss:.4f}")

    # ── Save final adapter ──
    adapter_path = output_dir / "final_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info(f"\nAdapter saved: {adapter_path}")

    # ── Save training metrics ──
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    logger.info("\n" + "=" * 60)
    logger.info("Algorithm C (DPO) Training Complete")
    logger.info("=" * 60)
    logger.info(f"  Adapter:    {adapter_path}")
    logger.info(f"  Eval loss:  {metrics.get('eval_loss', 'N/A'):.4f}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("  1. Merge adapter into base model:")
    logger.info(f"     python algo_c/train_dpo.py --merge \\")
    logger.info(f"         --sft_model {cfg.sft_model} \\")
    logger.info(f"         --adapter_path {adapter_path} \\")
    logger.info(f"         --merged_output ./output/algo_c_merged")
    logger.info("  2. Run evaluation:")
    logger.info(f"     python eval/evaluate.py --model ./output/algo_c_merged")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# MERGE UTILITY  (apply LoRA adapter → full merged weights)
# ──────────────────────────────────────────────────────────────────────────────

def merge_adapter(sft_model: str, adapter_path: str, merged_output: str) -> None:
    """
    Merge the DPO LoRA adapter back into the base model for inference.
    Run this after training to produce a standalone model.
    """
    logger.info(f"Loading base model from: {sft_model}")
    tokenizer = AutoTokenizer.from_pretrained(sft_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        sft_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",    # merge on CPU to avoid OOM
        trust_remote_code=True,
    )

    logger.info(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    logger.info("Merging adapter weights...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {merged_output}")
    model.save_pretrained(merged_output, safe_serialization=True)
    tokenizer.save_pretrained(merged_output)

    logger.info(f"Merged model saved: {merged_output}")
    logger.info("You can now use this model for inference or evaluation.")


# ──────────────────────────────────────────────────────────────────────────────
# QUICK INFERENCE TEST  (verify the adapter is working before full eval)
# ──────────────────────────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    "What is the per-unit system and why is it used in power systems analysis?",
    "Explain the difference between synchronous and induction generators in wind farms.",
    "How do you calculate the fault current for a three-phase bolted fault at a 132 kV bus?",
]


def inference_test(model_path: str) -> None:
    logger.info("\n── Inference Test ──")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    for q in TEST_QUESTIONS:
        messages = [
            {"role": "system", "content":
             "You are an expert power systems engineer. Provide precise technical answers."},
            {"role": "user", "content": q},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids=inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()

        logger.info(f"\nQ: {q}")
        logger.info(f"A: {response[:400]}")
        logger.info("-" * 40)


# ──────────────────────────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Algorithm C: DPO fine-tuning for power systems AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Main mode
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA adapter into base model (run after training)")
    parser.add_argument("--test",  action="store_true",
                        help="Run inference test on the merged model")

    # Paths
    parser.add_argument("--sft_model",     default="./output/algo_a_merged",
                        help="Path to merged SFT model (from algo_a or algo_b)")
    parser.add_argument("--train_file",    default="./data/dpo/dpo_train.json")
    parser.add_argument("--eval_file",     default="./data/dpo/dpo_eval.json")
    parser.add_argument("--output_dir",    default="./output/algo_c_dpo")
    parser.add_argument("--adapter_path",  default=None,
                        help="Adapter path for --merge mode")
    parser.add_argument("--merged_output", default="./output/algo_c_merged",
                        help="Output path for merged model")

    # DPO hyperparameters
    parser.add_argument("--beta",          type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--num_epochs",    type=int,   default=1)
    parser.add_argument("--batch_size",    type=int,   default=1)
    parser.add_argument("--grad_accum",    type=int,   default=8)

    # LoRA
    parser.add_argument("--lora_r",        type=int,   default=32)
    parser.add_argument("--lora_alpha",    type=int,   default=64)

    parser.add_argument("--seed",          type=int,   default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.merge:
        adapter = args.adapter_path or str(Path(args.output_dir) / "final_adapter")
        merge_adapter(args.sft_model, adapter, args.merged_output)

    elif args.test:
        model_path = args.merged_output
        if not Path(model_path).exists():
            logger.error(f"Model not found: {model_path}")
            logger.error("Run training first, then --merge, then --test")
            sys.exit(1)
        inference_test(model_path)

    else:
        cfg = DPOTrainConfig(
            sft_model=args.sft_model,
            output_dir=args.output_dir,
            train_file=args.train_file,
            eval_file=args.eval_file,
            beta=args.beta,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            seed=args.seed,
        )
        train(cfg)


if __name__ == "__main__":
    main()
