"""
Algorithm A — QLoRA Fine-Tuning with Unsloth
=============================================
Project: RFP PSX-20260510 — Power Systems AI Model
Hardware: 1× A100 80GB (or 1× H100 80GB)
Base model: Qwen/Qwen2.5-14B (Apache 2.0)

USAGE:
    python train_qlora.py
    python train_qlora.py --model_id Qwen/Qwen2.5-7B  (smaller/faster)
    python train_qlora.py --resume_from ./checkpoints/checkpoint-500

CONFIG:
    Edit the CONFIG block below before running.
    No other file edits needed.

OUTPUT:
    ./output/algo_a_adapter/   — LoRA adapter weights (~200MB)
    ./output/algo_a_merged/    — merged full model (~28GB, optional)
    ./logs/                    — training logs + W&B (if configured)
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model
    model_id: str = "Qwen/Qwen2.5-14B"       # base model (HuggingFace ID or local path)
    load_in_4bit: bool = True                  # 4-bit NF4 quantization (saves ~70% VRAM)
    max_seq_length: int = 4096                 # max tokens per example

    # LoRA (Proposal: r=32, alpha=64)
    lora_r: int = 32                           # LoRA rank
    lora_alpha: int = 64                       # LoRA alpha (usually 2×r)
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (             # Qwen2.5 attention + FFN layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Data
    train_file: str = "./data/train.json"
    eval_file: str = "./data/eval.json"

    # Training
    output_dir: str = "./output/algo_a_adapter"
    num_epochs: int = 3
    per_device_batch_size: int = 2             # keep at 2 for 14B on A100 80GB
    gradient_accumulation: int = 8             # effective batch = 2×8 = 16
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 250
    save_total_limit: int = 3

    # System prompt
    system_prompt: str = (
        "You are an expert power systems engineer with deep knowledge of "
        "electrical grid operation, protection systems, stability analysis, "
        "load flow, fault calculations, energy markets, and power electronics. "
        "Provide accurate, technically precise, and practical answers."
    )

    # W&B logging (set to None to disable)
    wandb_project: Optional[str] = "psx-20260510-algo-a"
    wandb_run_name: Optional[str] = "qlora-qwen2.5-14b"

    # Merge adapter into full model at the end (takes ~20 min, needs ~80GB RAM)
    merge_at_end: bool = False
    merged_output_dir: str = "./output/algo_a_merged"


CFG = Config()

# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT OVERRIDES (command-line flags override Config above)
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",           default=CFG.model_id)
    parser.add_argument("--train_file",         default=CFG.train_file)
    parser.add_argument("--eval_file",          default=CFG.eval_file)
    parser.add_argument("--output_dir",         default=CFG.output_dir)
    parser.add_argument("--num_epochs",         type=int,   default=CFG.num_epochs)
    parser.add_argument("--learning_rate",      type=float, default=CFG.learning_rate)
    parser.add_argument("--lora_r",             type=int,   default=CFG.lora_r)
    parser.add_argument("--lora_alpha",         type=int,   default=CFG.lora_alpha)
    parser.add_argument("--max_seq_length",     type=int,   default=CFG.max_seq_length)
    parser.add_argument("--merge_at_end",       action="store_true", default=CFG.merge_at_end)
    parser.add_argument("--resume_from",        default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no_wandb",           action="store_true", help="Disable W&B logging")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# DATASET LOADING & FORMATTING
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
    from datasets import load_dataset, Dataset

    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    train_data = load_json(train_file)
    eval_data  = load_json(eval_file) if Path(eval_file).exists() else []

    eos = tokenizer.eos_token

    def format_examples(examples):
        texts = []
        for inst, inp, out in zip(
            examples["instruction"],
            examples.get("input", [""] * len(examples["instruction"])),
            examples["output"],
        ):
            # Use Qwen chat template if available, else Alpaca template
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
    train_ds = train_ds.map(format_examples, batched=True, remove_columns=train_ds.column_names)

    eval_ds = None
    if eval_data:
        eval_ds = Dataset.from_list(eval_data)
        eval_ds = eval_ds.map(format_examples, batched=True, remove_columns=eval_ds.column_names)

    return train_ds, eval_ds


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Apply CLI overrides to config
    CFG.model_id          = args.model_id
    CFG.train_file        = args.train_file
    CFG.eval_file         = args.eval_file
    CFG.output_dir        = args.output_dir
    CFG.num_epochs        = args.num_epochs
    CFG.learning_rate     = args.learning_rate
    CFG.lora_r            = args.lora_r
    CFG.lora_alpha        = args.lora_alpha
    CFG.max_seq_length    = args.max_seq_length
    CFG.merge_at_end      = args.merge_at_end

    if args.no_wandb:
        CFG.wandb_project = None

    # ── W&B setup ──
    if CFG.wandb_project:
        os.environ["WANDB_PROJECT"] = CFG.wandb_project
        if CFG.wandb_run_name:
            os.environ["WANDB_NAME"] = CFG.wandb_run_name
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # ── Check GPU ──
    if not torch.cuda.is_available():
        print("[ERROR] No GPU detected. This script requires a CUDA GPU (A100 recommended).")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[OK] GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    # ── Import Unsloth ──
    try:
        from unsloth import FastLanguageModel
        print("[OK] Unsloth loaded")
    except ImportError:
        print("[ERROR] Unsloth not installed.")
        print("  Install with: pip install unsloth")
        sys.exit(1)

    # ── Load model ──
    print(f"\n[STEP 1] Loading base model: {CFG.model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG.model_id,
        max_seq_length=CFG.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=CFG.load_in_4bit,
    )

    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after model load: {vram_used:.1f} GB / {gpu_mem:.0f} GB")

    # ── Inject LoRA adapters ──
    print(f"\n[STEP 2] Injecting LoRA adapters (r={CFG.lora_r}, alpha={CFG.lora_alpha})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        target_modules=list(CFG.lora_target_modules),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()

    # ── Load dataset ──
    print(f"\n[STEP 3] Loading dataset from {CFG.train_file}")
    if not Path(CFG.train_file).exists():
        print(f"[ERROR] Training file not found: {CFG.train_file}")
        print("  Run: python data/build_dataset.py --input_dir ./raw_docs")
        sys.exit(1)

    train_ds, eval_ds = load_and_format_dataset(
        CFG.train_file, CFG.eval_file, tokenizer, CFG.system_prompt
    )
    print(f"  Train: {len(train_ds):,} examples")
    if eval_ds:
        print(f"  Eval:  {len(eval_ds):,} examples")

    # ── Configure trainer ──
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"\n[STEP 4] Configuring SFT trainer")
    os.makedirs(CFG.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        per_device_train_batch_size=CFG.per_device_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation,
        num_train_epochs=CFG.num_epochs,
        learning_rate=CFG.learning_rate,
        lr_scheduler_type=CFG.lr_scheduler,
        warmup_ratio=CFG.warmup_ratio,
        weight_decay=CFG.weight_decay,
        max_grad_norm=CFG.max_grad_norm,
        bf16=True,
        fp16=False,
        logging_steps=CFG.logging_steps,
        save_steps=CFG.save_steps,
        save_total_limit=CFG.save_total_limit,
        evaluation_strategy="steps" if eval_ds else "no",
        eval_steps=CFG.eval_steps if eval_ds else None,
        load_best_model_at_end=True if eval_ds else False,
        optim="adamw_8bit",
        seed=42,
        report_to="wandb" if CFG.wandb_project else "none",
        dataloader_num_workers=4,
        remove_unused_columns=True,
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

    # Print estimated training time
    steps_per_epoch = len(train_ds) // (CFG.per_device_batch_size * CFG.gradient_accumulation)
    total_steps = steps_per_epoch * CFG.num_epochs
    print(f"\n  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps:     {total_steps:,}")
    print(f"  Effective batch: {CFG.per_device_batch_size * CFG.gradient_accumulation}")

    # ── Train ──
    print(f"\n[STEP 5] Starting training...")
    print(f"  Checkpoints saved every {CFG.save_steps} steps to {CFG.output_dir}")

    trainer_stats = trainer.train(
        resume_from_checkpoint=args.resume_from
    )

    # ── Print stats ──
    runtime_hours = trainer_stats.metrics["train_runtime"] / 3600
    print(f"\n[DONE] Training complete!")
    print(f"  Total time:   {runtime_hours:.1f} hours")
    print(f"  Final loss:   {trainer_stats.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  GPU-hours:    {runtime_hours:.1f}")

    # ── Save adapter ──
    print(f"\n[STEP 6] Saving LoRA adapter to {CFG.output_dir}")
    trainer.model.save_pretrained(CFG.output_dir)
    tokenizer.save_pretrained(CFG.output_dir)
    print(f"  Adapter saved. Size: ~{sum(p.numel() * 2 for p in model.parameters() if p.requires_grad) / 1e6:.0f} MB")

    # ── Optional: Merge adapter into full model ──
    if CFG.merge_at_end:
        print(f"\n[STEP 7] Merging adapter into full model (takes ~20 min)...")
        os.makedirs(CFG.merged_output_dir, exist_ok=True)
        model.save_pretrained_merged(
            CFG.merged_output_dir,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"  Merged model saved to {CFG.merged_output_dir}")

    # ── Quick test inference ──
    print(f"\n[STEP 8] Quick inference test...")
    FastLanguageModel.for_inference(model)

    test_question = "Explain the Newton-Raphson method for load flow analysis in power systems."
    messages = [
        {"role": "system",  "content": CFG.system_prompt},
        {"role": "user",    "content": test_question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"\n  Q: {test_question}")
    print(f"\n  A: {response[:500]}...")

    print(f"\n{'='*60}")
    print(f"ALGORITHM A COMPLETE")
    print(f"  Adapter: {CFG.output_dir}")
    if CFG.merge_at_end:
        print(f"  Merged:  {CFG.merged_output_dir}")
    print(f"  Next: Run eval/evaluate.py to benchmark the model")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
