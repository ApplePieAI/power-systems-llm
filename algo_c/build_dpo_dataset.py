"""
build_dpo_dataset.py — DPO Preference Dataset Builder
======================================================
Project: RFP PSX-20260510 — Power Systems AI Model

DPO requires triplets: (prompt, chosen_response, rejected_response)
where chosen > rejected according to a domain expert or an AI judge.

This script builds the DPO dataset using two strategies:

Strategy 1 — Model vs Model (automated, no SME needed)
    Take questions from the SFT training set.
    Generate two responses from the SFT-trained model at different temperatures.
    Use Claude as a judge to pick chosen vs rejected.

Strategy 2 — SME Rating (highest quality, requires expert time)
    Generate two responses per question (same as above).
    Export a CSV for a power systems SME to manually rate each pair.
    Import the SME ratings to produce the final dataset.

USAGE:
    # Strategy 1 — automated (recommended first pass):
    python algo_c/build_dpo_dataset.py \\
        --sft_model    ./output/algo_a_merged \\
        --questions    ./data/processed/train.json \\
        --output_dir   ./data/dpo \\
        --strategy     auto \\
        --num_questions 1000

    # Strategy 2 — export for SME rating:
    python algo_c/build_dpo_dataset.py \\
        --sft_model    ./output/algo_a_merged \\
        --questions    ./data/processed/train.json \\
        --output_dir   ./data/dpo \\
        --strategy     sme \\
        --num_questions 1000

    # Import SME ratings after review:
    python algo_c/build_dpo_dataset.py \\
        --import_sme   ./data/dpo/sme_rating.csv \\
        --output_dir   ./data/dpo

OUTPUT:
    data/dpo/
      dpo_train.json     DPO training triplets (90%)
      dpo_eval.json      DPO eval triplets (10%)
      sme_rating.csv     Export for SME manual rating (strategy=sme)
      build.log
"""

import os
import re
import sys
import csv
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# RESPONSE GENERATOR  (uses the fine-tuned SFT model)
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert power systems engineer with deep knowledge of "
    "electrical grid operation, protection systems, stability analysis, "
    "load flow, fault calculations, energy markets, and power electronics. "
    "Provide accurate, technically precise, and practical answers."
)


class SFTModel:
    """Loads the SFT-trained model and generates responses at variable temperatures."""

    def __init__(self, model_path: str):
        logger.info(f"Loading SFT model from {model_path}...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Run: pip install transformers torch")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.torch = torch
        logger.info("SFT model loaded.")

    def generate(self, question: str, temperature: float = 0.7, max_new_tokens: int = 512) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with self.torch.no_grad():
            out = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()


# ──────────────────────────────────────────────────────────────────────────────
# CLAUDE JUDGE  (picks chosen vs rejected automatically)
# ──────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a senior power systems engineer evaluating AI-generated answers "
    "for technical accuracy, completeness, and practical usefulness. "
    "Be strict — prefer answers that are precise, correctly reference engineering "
    "principles, include correct units, and avoid vague generalisations."
)

JUDGE_PROMPT = """\
A power systems AI model generated two responses to the same question.
Select which response is better for training a domain-specialized power systems AI.

QUESTION:
{question}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Evaluate on:
1. Technical accuracy (correct formulas, values, and principles)
2. Completeness (covers all relevant aspects of the question)
3. Practical usefulness (actionable for a practising engineer)
4. Clarity (well-structured, correct units, no unnecessary vagueness)

Reply with ONLY one of:
  A  — if Response A is clearly better
  B  — if Response B is clearly better
  TIE — if both are roughly equal quality

Then on the next line, one sentence explaining why."""


class ClaudeJudge:
    """Uses Claude to automatically select chosen vs rejected responses."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY or pass api_key=")

        self.client = __import__("anthropic").Anthropic(api_key=key)
        self.model  = model
        self.calls  = 0
        self.ties   = 0
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    def judge(
        self,
        question: str,
        response_a: str,
        response_b: str,
    ) -> tuple[Optional[str], Optional[str], str]:
        """
        Returns (chosen, rejected, explanation).
        Returns (None, None, explanation) on TIE or error.
        """
        prompt = JUDGE_PROMPT.format(
            question=question[:800],
            response_a=response_a[:600],
            response_b=response_b[:600],
        )

        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=150,
                    system=JUDGE_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.calls += 1
                self.total_input_tokens  += resp.usage.input_tokens
                self.total_output_tokens += resp.usage.output_tokens

                raw = resp.content[0].text.strip()
                first_line = raw.split("\n")[0].strip().upper()
                explanation = raw.split("\n", 1)[1].strip() if "\n" in raw else ""

                if first_line == "A":
                    return response_a, response_b, explanation
                elif first_line == "B":
                    return response_b, response_a, explanation
                elif "TIE" in first_line:
                    self.ties += 1
                    return None, None, "TIE"
                else:
                    logger.warning(f"Judge gave unexpected response: {raw[:60]}")
                    return None, None, "parse_error"

            except Exception as e:
                logger.warning(f"Judge error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt * 2)

        return None, None, "error"

    def cost_summary(self) -> str:
        cost = (self.total_input_tokens  / 1_000_000 * 3.0 +
                self.total_output_tokens / 1_000_000 * 15.0)
        return (
            f"Judge calls: {self.calls}  |  Ties: {self.ties}  |  "
            f"Tokens: {self.total_input_tokens:,} in + {self.total_output_tokens:,} out  |  "
            f"Cost: ${cost:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# TRIPLET BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_triplets_auto(
    questions: list[dict],
    sft_model: SFTModel,
    judge: ClaudeJudge,
    num_questions: int,
    save_every: int = 50,
    checkpoint_path: Optional[Path] = None,
) -> list[dict]:
    """
    Generate response pairs from the SFT model at different temperatures,
    then use Claude judge to assign chosen/rejected labels automatically.

    Two temperatures create natural variation:
    - Low temp (0.3)  → deterministic, likely correct but possibly dry
    - High temp (0.9) → creative, richer but more likely to hallucinate
    The judge picks which is better per question.
    """
    # Load checkpoint if resuming
    triplets = []
    done_ids = set()
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)
        triplets  = data.get("triplets", [])
        done_ids  = set(data.get("done_ids", []))
        logger.info(f"Resuming: {len(triplets)} triplets already built")

    questions = random.sample(questions, min(num_questions, len(questions)))

    for i, q in enumerate(questions):
        qid = q.get("id", str(i))
        if qid in done_ids:
            continue

        prompt = q["instruction"]
        if q.get("input"):
            prompt += f"\n\n{q['input']}"

        logger.info(f"  [{i+1}/{len(questions)}] Generating responses...")

        # Generate two responses at different temperatures
        resp_low  = sft_model.generate(prompt, temperature=0.3)
        resp_high = sft_model.generate(prompt, temperature=0.9)

        # Skip if responses are identical
        if resp_low.strip() == resp_high.strip():
            logger.debug(f"  Skipping (identical responses): {prompt[:60]}")
            done_ids.add(qid)
            continue

        # Judge picks chosen vs rejected
        chosen, rejected, explanation = judge.judge(prompt, resp_low, resp_high)

        if chosen is None:
            logger.debug(f"  Skipping (TIE or error): {prompt[:60]}")
            done_ids.add(qid)
            continue

        triplets.append({
            "prompt":    prompt,
            "chosen":    chosen,
            "rejected":  rejected,
            "judge_note": explanation,
        })
        done_ids.add(qid)

        logger.info(f"  → Triplet built ({len(triplets)} total)")

        # Checkpoint every N
        if checkpoint_path and len(triplets) % save_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"triplets": triplets, "done_ids": list(done_ids)}, f)

    return triplets


def build_triplets_sme_export(
    questions: list[dict],
    sft_model: SFTModel,
    num_questions: int,
    csv_path: str,
) -> None:
    """
    Generate two responses per question and export to CSV for SME rating.
    SME fills in the 'better' column (A or B) for each row.
    """
    questions = random.sample(questions, min(num_questions, len(questions)))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "prompt", "response_a", "response_b",
            "better",       # SME fills: A or B
            "reason",       # SME fills: brief explanation
        ])
        writer.writeheader()

        for i, q in enumerate(questions):
            prompt = q["instruction"]
            if q.get("input"):
                prompt += f"\n\n{q['input']}"

            logger.info(f"  [{i+1}/{len(questions)}] Generating pair...")
            resp_a = sft_model.generate(prompt, temperature=0.3)
            resp_b = sft_model.generate(prompt, temperature=0.9)

            writer.writerow({
                "id":         i + 1,
                "prompt":     prompt,
                "response_a": resp_a,
                "response_b": resp_b,
                "better":     "",   # SME fills
                "reason":     "",   # SME fills
            })

    logger.info(f"SME rating file saved: {csv_path} ({len(questions)} pairs)")
    logger.info("Instructions for SME:")
    logger.info("  - Open the CSV in Excel or Google Sheets")
    logger.info("  - For each row, fill 'better' with A or B")
    logger.info("  - Optionally fill 'reason' with a brief explanation")
    logger.info("  - Save and return the file")
    logger.info(f"  - Then run: python algo_c/build_dpo_dataset.py --import_sme {csv_path}")


def import_sme_ratings(csv_path: str) -> list[dict]:
    """Convert SME-rated CSV back into DPO triplets."""
    triplets = []
    skipped  = 0

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            better = row.get("better", "").strip().upper()
            if better == "A":
                chosen, rejected = row["response_a"], row["response_b"]
            elif better == "B":
                chosen, rejected = row["response_b"], row["response_a"]
            else:
                skipped += 1
                continue

            triplets.append({
                "prompt":    row["prompt"],
                "chosen":    chosen,
                "rejected":  rejected,
                "judge_note": row.get("reason", "SME rated"),
            })

    logger.info(f"Imported {len(triplets)} triplets from SME ratings ({skipped} skipped/blank)")
    return triplets


# ──────────────────────────────────────────────────────────────────────────────
# EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def save_dpo_splits(
    triplets: list[dict],
    output_dir: Path,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> None:
    """Shuffle and split triplets into train/eval, strip internal metadata."""
    random.seed(seed)
    random.shuffle(triplets)
    cut = int(len(triplets) * train_ratio)

    train = triplets[:cut]
    eval_ = triplets[cut:]

    # Strip judge_note from final output — not needed for training
    def clean(pairs):
        return [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
                for p in pairs]

    train_path = output_dir / "dpo_train.json"
    eval_path  = output_dir / "dpo_eval.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(clean(train), f, indent=2, ensure_ascii=False)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(clean(eval_), f, indent=2, ensure_ascii=False)

    logger.info(f"DPO train: {len(train):,} triplets → {train_path}")
    logger.info(f"DPO eval:  {len(eval_):,} triplets → {eval_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DPO preference dataset from SFT model outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sft_model",      default=None,
                        help="Path to fine-tuned SFT model (merged weights)")
    parser.add_argument("--questions",      default="./data/processed/train.json",
                        help="SFT training pairs to use as question source")
    parser.add_argument("--output_dir",     default="./data/dpo")
    parser.add_argument("--strategy",       default="auto",
                        choices=["auto", "sme"],
                        help="auto=Claude judge  sme=export CSV for human rating")
    parser.add_argument("--num_questions",  type=int, default=1000,
                        help="Number of questions to generate pairs for")
    parser.add_argument("--anthropic_key",  default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--import_sme",     default=None,
                        help="Import SME-rated CSV and convert to DPO triplets")
    parser.add_argument("--train_ratio",    type=float, default=0.9)
    parser.add_argument("--seed",           type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(output_dir / "build.log"), mode="a"),
        ],
    )

    logger.info("=" * 60)
    logger.info("DPO Preference Dataset Builder")
    logger.info("=" * 60)

    # ── Import SME ratings (standalone mode) ──
    if args.import_sme:
        triplets = import_sme_ratings(args.import_sme)
        save_dpo_splits(triplets, output_dir, args.train_ratio, args.seed)
        return

    # ── Load questions ──
    if not args.sft_model:
        logger.error("--sft_model required (path to merged SFT model weights)")
        sys.exit(1)

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    logger.info(f"Loaded {len(questions):,} questions from {args.questions}")

    # ── Load SFT model ──
    sft_model = SFTModel(args.sft_model)

    if args.strategy == "sme":
        # Export for SME manual rating
        csv_path = str(output_dir / "sme_rating.csv")
        build_triplets_sme_export(questions, sft_model, args.num_questions, csv_path)

    else:
        # Automated: Claude judge
        api_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set. Use --anthropic_key or export it.")
            sys.exit(1)

        judge = ClaudeJudge(api_key=api_key)
        checkpoint = output_dir / ".checkpoint.json"

        triplets = build_triplets_auto(
            questions=questions,
            sft_model=sft_model,
            judge=judge,
            num_questions=args.num_questions,
            checkpoint_path=checkpoint,
        )

        logger.info(f"\n{judge.cost_summary()}")

        if not triplets:
            logger.error("No triplets built — check model output and judge responses.")
            sys.exit(1)

        save_dpo_splits(triplets, output_dir, args.train_ratio, args.seed)

        # Clean up checkpoint on success
        if checkpoint.exists():
            checkpoint.unlink()

    logger.info("\nNEXT STEPS:")
    logger.info("  Run DPO training:")
    logger.info("  python algo_c/train_dpo.py \\")
    logger.info(f"      --sft_model {args.sft_model} \\")
    logger.info(f"      --train_file {output_dir}/dpo_train.json \\")
    logger.info(f"      --eval_file  {output_dir}/dpo_eval.json")


if __name__ == "__main__":
    main()
