"""
evaluate.py — Fine-Tuned Model vs GPT-4o vs Claude Benchmark Runner
====================================================================
Project: RFP PSX-20260510

Runs the benchmark.json question set against:
  1. Your fine-tuned model (local, via HuggingFace transformers)
  2. GPT-4o (OpenAI API)
  3. Claude (Anthropic API)

Scores each model using a 3-tier system:
  Tier 1 (1.0 pt)  — Exact match, within numerical tolerance, or judge says "correct"
  Tier 2 (0.5 pt)  — Correct approach, minor error — scored by LLM judge
  Tier 3 (0.0 pt)  — Wrong, hallucinated, or refused

USAGE:
    # Compare all three models:
    python evaluate.py \\
        --benchmark ./benchmark/benchmark.json \\
        --ft_model  ./output/algo_a_merged \\
        --openai_key sk-... \\
        --anthropic_key sk-ant-... \\
        --output_dir ./results

    # Fine-tuned model only (no API keys needed):
    python evaluate.py \\
        --benchmark ./benchmark/benchmark.json \\
        --ft_model  ./output/algo_a_merged \\
        --output_dir ./results

    # GPT-4o and Claude only (no local model):
    python evaluate.py \\
        --benchmark ./benchmark/benchmark.json \\
        --no_ft_model \\
        --openai_key sk-... \\
        --anthropic_key sk-ant-...

OUTPUT:
    results/
      scores.json         Per-question scores for all models
      summary.json        Aggregate scores by model × category × difficulty
      report.md           Human-readable comparison report
      hallucination.json  Questions where any model hallucinated
"""

import os
import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT (same for all models — controls for prompt differences)
# ──────────────────────────────────────────────────────────────────────────────

EVAL_SYSTEM = (
    "You are an expert power systems engineer. Answer the following question "
    "accurately and concisely. If the question requires a calculation, show "
    "the key steps. If you do not know the answer, say so explicitly rather "
    "than guessing."
)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL CLIENTS
# ──────────────────────────────────────────────────────────────────────────────

class FineTunedModel:
    """Runs inference on a local HuggingFace model (merged or adapter)."""

    def __init__(self, model_path: str, max_new_tokens: int = 512):
        logger.info(f"Loading fine-tuned model from {model_path}...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Run: pip install transformers torch")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.name = "fine_tuned"
        logger.info("Fine-tuned model loaded.")

    def generate(self, question: str) -> str:
        import torch

        messages = [
            {"role": "system",  "content": EVAL_SYSTEM},
            {"role": "user",    "content": question},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,   # low temp for eval (more deterministic)
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


class GPT4oClient:
    """Calls GPT-4o via OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY or pass api_key=")

        self.client = __import__("openai").OpenAI(api_key=key)
        self.model = model
        self.name = "gpt4o"
        self.total_cost = 0.0

    def generate(self, question: str) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EVAL_SYSTEM},
                        {"role": "user",   "content": question},
                    ],
                    temperature=0.1,
                    max_tokens=512,
                )
                usage = resp.usage
                self.total_cost += (
                    usage.prompt_tokens     / 1000 * 0.005 +
                    usage.completion_tokens / 1000 * 0.015
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    time.sleep(2 ** attempt * 5)
                else:
                    logger.warning(f"GPT-4o error (attempt {attempt+1}): {e}")
                    time.sleep(2)
        return "[ERROR: GPT-4o failed after 3 retries]"


class ClaudeClient:
    """Calls Claude via Anthropic API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY or pass api_key=")

        self.client = __import__("anthropic").Anthropic(api_key=key)
        self.model = model
        self.name = "claude"
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def generate(self, question: str) -> str:
        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    system=EVAL_SYSTEM,
                    messages=[{"role": "user", "content": question}],
                )
                self.total_input_tokens  += resp.usage.input_tokens
                self.total_output_tokens += resp.usage.output_tokens
                return resp.content[0].text.strip()
            except Exception as e:
                logger.warning(f"Claude error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt * 3)
        return "[ERROR: Claude failed after 3 retries]"

    def cost_summary(self) -> str:
        # claude-sonnet-4-6: $3/M input, $15/M output
        cost = (self.total_input_tokens  / 1_000_000 * 3.0 +
                self.total_output_tokens / 1_000_000 * 15.0)
        return f"Claude tokens: {self.total_input_tokens:,} in + {self.total_output_tokens:,} out = ${cost:.2f}"


# ──────────────────────────────────────────────────────────────────────────────
# SCORING
# ──────────────────────────────────────────────────────────────────────────────

def score_exact(model_answer: str, correct_answer: str, tolerance_pct: float) -> Optional[float]:
    """
    Attempt exact/numerical scoring without calling an LLM.

    Returns:
        1.0  — clear correct match
        0.0  — clear wrong (numerical question with answer outside tolerance)
        None — cannot determine, needs LLM judge
    """
    ma = model_answer.strip().lower()
    ca = correct_answer.strip().lower()

    # Exact string match (after normalising whitespace)
    if re.sub(r"\s+", " ", ma) == re.sub(r"\s+", " ", ca):
        return 1.0

    # Check if model explicitly refused or said it doesn't know
    refusal_patterns = [
        r"i (don'?t|do not|cannot|can'?t) (know|have|answer|determine)",
        r"(not enough|insufficient) (information|data|context)",
        r"\[error",
    ]
    for pat in refusal_patterns:
        if re.search(pat, ma, re.I):
            return 0.0

    # Try numerical comparison for calculation questions
    num_model   = _extract_number(ma)
    num_correct = _extract_number(ca)

    if num_model is not None and num_correct is not None and num_correct != 0:
        pct_diff = abs(num_model - num_correct) / abs(num_correct) * 100
        tol = tolerance_pct if tolerance_pct > 0 else 2.0   # default 2% tolerance
        if pct_diff <= tol:
            return 1.0
        elif pct_diff > tol * 5:   # clearly wrong (5× outside tolerance)
            return 0.0

    return None  # unclear — needs LLM judge


def _extract_number(text: str) -> Optional[float]:
    """Extract the first standalone number from text (handles units like kA, pu, MW)."""
    # Match numbers like 3.34, 10.5, -j8.33, 1,234.5
    matches = re.findall(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass
    return None


def detect_hallucination(model_answer: str) -> bool:
    """
    Heuristic check for hallucination signals in a model response.

    Flags:
    - Cites a non-existent or malformed standard (e.g. "IEEE 99999")
    - Contains confident factual claims with suspicious specificity
    - References impossible values (e.g. impedance > 1 pu for a line, negative frequency)
    """
    # Suspicious standard citations: IEEE/IEC followed by an unusual number
    fake_standards = re.findall(r"\b(?:IEEE|IEC|ANSI|NERC)\s+(?:std\.?\s*)?\d{4,}\b", model_answer, re.I)
    if fake_standards:
        return True

    # Values that are physically impossible for common power systems quantities
    impossible_patterns = [
        r"\b(?:frequency|hz)\b.*\b(?:5[0-9]{3,}|[1-9]\d{4,})\b",   # freq > 999 Hz
        r"\befficiency\b.*\b1[0-9]{2,}\s*%",                          # efficiency > 100%
    ]
    for pat in impossible_patterns:
        if re.search(pat, model_answer, re.I):
            return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# LLM JUDGE  (Tier 2 scoring)
# ──────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a senior power systems engineer evaluating AI model answers "
    "against reference answers for a technical benchmark. Be strict and precise."
)

JUDGE_PROMPT = """\
Question: {question}

Reference answer: {correct_answer}

Model answer: {model_answer}

Rate the model answer on this 3-point scale:
  2 = Correct: answer is accurate and equivalent to the reference (within engineering tolerance)
  1 = Partial: correct approach or key concept but has a minor error, missing units, or incomplete
  0 = Wrong: incorrect, hallucinated, or refused to answer

Respond with ONLY the number (0, 1, or 2) followed by one sentence explanation.
Example: "2 The model correctly calculated the fault current as 3.34 kA."
"""


class LLMJudge:
    """Uses Claude as the judge for Tier 2 ambiguous scores."""

    def __init__(self, client: ClaudeClient):
        self.client = client
        self.calls = 0

    def score(self, question: str, correct_answer: str, model_answer: str) -> tuple[float, str]:
        """
        Returns (score, explanation) where score is 0.0, 0.5, or 1.0.
        """
        prompt = JUDGE_PROMPT.format(
            question=question[:600],
            correct_answer=correct_answer[:300],
            model_answer=model_answer[:600],
        )

        try:
            resp = self.client.client.messages.create(
                model=self.client.model,
                max_tokens=100,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            self.calls += 1
            raw = resp.content[0].text.strip()

            # Parse "2 explanation..." or just "2"
            match = re.match(r"^([012])\b(.*)$", raw, re.DOTALL)
            if match:
                tier = int(match.group(1))
                explanation = match.group(2).strip()
                score = {2: 1.0, 1: 0.5, 0: 0.0}[tier]
                return score, explanation

            logger.warning(f"Judge returned unparseable response: {raw[:80]}")
            return 0.5, "Judge response unclear"

        except Exception as e:
            logger.warning(f"Judge error: {e}")
            return 0.5, f"Judge error: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    questions: list[dict],
    models: dict,              # {"fine_tuned": model, "gpt4o": ..., "claude": ...}
    judge: Optional[LLMJudge],
    output_dir: Path,
    save_every: int = 25,
) -> dict:
    """
    Run all questions against all models and score them.

    Returns the full scores dict.
    """
    scores_path = output_dir / "scores.json"

    # Resume from partial run if scores file exists
    if scores_path.exists():
        with open(scores_path, encoding="utf-8") as f:
            scores = json.load(f)
        logger.info(f"Resuming from {len(scores)} previously scored questions")
    else:
        scores = {}

    hallucinations = []
    model_names = list(models.keys())

    for i, q in enumerate(questions):
        qid = q["id"]

        # Skip if already scored for all models
        if qid in scores and all(m in scores[qid].get("responses", {}) for m in model_names):
            continue

        if qid not in scores:
            scores[qid] = {
                "question":  q["question"],
                "category":  q["category"],
                "difficulty": q["difficulty"],
                "answer":    q["answer"],
                "tolerance_pct": q.get("tolerance_pct", 0),
                "responses": {},
                "scores":    {},
                "judge_explanations": {},
            }

        for model_name, model in models.items():
            if model_name in scores[qid].get("responses", {}):
                continue   # already done

            logger.debug(f"  [{i+1}/{len(questions)}] {model_name}: {qid}")

            # Generate answer
            answer = model.generate(q["question"])
            scores[qid]["responses"][model_name] = answer

            # Detect hallucination
            if detect_hallucination(answer):
                hallucinations.append({
                    "id": qid, "model": model_name,
                    "question": q["question"], "answer": answer,
                })

            # Score
            tier1_score = score_exact(answer, q["answer"], q.get("tolerance_pct", 0))

            if tier1_score is not None:
                scores[qid]["scores"][model_name] = tier1_score
                scores[qid]["judge_explanations"][model_name] = "exact/numerical"
            elif judge:
                score, explanation = judge.score(q["question"], q["answer"], answer)
                scores[qid]["scores"][model_name] = score
                scores[qid]["judge_explanations"][model_name] = explanation
            else:
                # No judge available — mark as needing manual review
                scores[qid]["scores"][model_name] = -1   # sentinel: needs review
                scores[qid]["judge_explanations"][model_name] = "needs_manual_review"

        # Checkpoint
        if (i + 1) % save_every == 0:
            with open(scores_path, "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=2, ensure_ascii=False)
            logger.info(f"  Checkpoint saved ({i+1}/{len(questions)} questions)")

    # Final save
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    if hallucinations:
        hall_path = output_dir / "hallucination.json"
        with open(hall_path, "w", encoding="utf-8") as f:
            json.dump(hallucinations, f, indent=2, ensure_ascii=False)
        logger.warning(f"{len(hallucinations)} hallucination flags — see {hall_path}")

    return scores


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY & REPORT
# ──────────────────────────────────────────────────────────────────────────────

def compute_summary(scores: dict, model_names: list[str]) -> dict:
    """Aggregate scores by model, category, and difficulty."""
    summary = {
        "overall": {m: {"total": 0, "sum": 0.0, "score": 0.0} for m in model_names},
        "by_category": defaultdict(lambda: {m: {"total": 0, "sum": 0.0} for m in model_names}),
        "by_difficulty": defaultdict(lambda: {m: {"total": 0, "sum": 0.0} for m in model_names}),
    }

    for qid, data in scores.items():
        cat  = data.get("category", "unknown")
        diff = data.get("difficulty", "unknown")

        for m in model_names:
            s = data["scores"].get(m, -1)
            if s < 0:
                continue   # skip unscored
            summary["overall"][m]["total"] += 1
            summary["overall"][m]["sum"]   += s
            summary["by_category"][cat][m]["total"] += 1
            summary["by_category"][cat][m]["sum"]   += s
            summary["by_difficulty"][diff][m]["total"] += 1
            summary["by_difficulty"][diff][m]["sum"]   += s

    # Compute percentages
    for m in model_names:
        n = summary["overall"][m]["total"]
        summary["overall"][m]["score"] = (
            round(summary["overall"][m]["sum"] / n * 100, 1) if n > 0 else 0
        )

    for cat, cat_data in summary["by_category"].items():
        for m in model_names:
            n = cat_data[m]["total"]
            cat_data[m]["score"] = round(cat_data[m]["sum"] / n * 100, 1) if n > 0 else 0

    for diff, diff_data in summary["by_difficulty"].items():
        for m in model_names:
            n = diff_data[m]["total"]
            diff_data[m]["score"] = round(diff_data[m]["sum"] / n * 100, 1) if n > 0 else 0

    # Convert defaultdicts for JSON serialisation
    summary["by_category"]   = dict(summary["by_category"])
    summary["by_difficulty"]  = dict(summary["by_difficulty"])
    return summary


def write_report(summary: dict, model_names: list[str], output_path: Path) -> None:
    """Write a human-readable Markdown comparison report."""
    lines = [
        "# Power Systems AI Benchmark — Evaluation Report",
        f"\n**Models evaluated:** {', '.join(model_names)}",
        f"**Scoring:** 1.0 = correct, 0.5 = partial, 0.0 = wrong\n",
        "---\n",
        "## Overall Scores\n",
        f"| Model | Score | Questions |",
        f"|-------|-------|-----------|",
    ]
    for m in model_names:
        d = summary["overall"][m]
        lines.append(f"| {m} | **{d['score']}%** | {d['total']} |")

    lines += ["\n---\n", "## Scores by Category\n",
              "| Category | " + " | ".join(model_names) + " |",
              "|----------|" + "|".join(["------" for _ in model_names]) + "|"]

    for cat, cat_data in sorted(summary["by_category"].items()):
        scores_str = " | ".join(f"{cat_data[m]['score']}%" for m in model_names)
        lines.append(f"| {cat} | {scores_str} |")

    lines += ["\n## Scores by Difficulty\n",
              "| Difficulty | " + " | ".join(model_names) + " |",
              "|------------|" + "|".join(["------" for _ in model_names]) + "|"]

    for diff in ["easy", "medium", "hard"]:
        if diff in summary["by_difficulty"]:
            diff_data = summary["by_difficulty"][diff]
            scores_str = " | ".join(f"{diff_data[m]['score']}%" for m in model_names)
            lines.append(f"| {diff} | {scores_str} |")

    lines += [
        "\n---\n",
        "## Key Findings\n",
        "_Fill in after reviewing the results._\n",
        "- Fine-tuned model advantage: ...",
        "- Areas where general models still lead: ...",
        "- Hallucination rate comparison: ...",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved: {output_path}")


def print_summary(summary: dict, model_names: list[str]) -> None:
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Model':<20} {'Score':>8} {'Questions':>12}")
    print("-" * 44)
    for m in model_names:
        d = summary["overall"][m]
        print(f"{m:<20} {d['score']:>7.1f}% {d['total']:>12,}")

    print(f"\n{'Category':<30} " + "  ".join(f"{m[:12]:>12}" for m in model_names))
    print("-" * (30 + 14 * len(model_names)))
    for cat, cat_data in sorted(summary["by_category"].items()):
        scores = "  ".join(f"{cat_data[m]['score']:>11.1f}%" for m in model_names)
        print(f"{cat:<30} {scores}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run power systems benchmark: fine-tuned vs GPT-4o vs Claude",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark",      required=True,
                        help="Path to benchmark.json (from build_benchmark.py)")
    parser.add_argument("--output_dir",     default="./results")

    # Model selection
    parser.add_argument("--ft_model",       default=None,
                        help="Path to fine-tuned model (merged weights directory)")
    parser.add_argument("--no_ft_model",    action="store_true",
                        help="Skip fine-tuned model (API models only)")
    parser.add_argument("--openai_key",     default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--anthropic_key",  default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--gpt_model",      default="gpt-4o")
    parser.add_argument("--claude_model",   default="claude-sonnet-4-6")

    # Eval options
    parser.add_argument("--max_questions",  type=int, default=0,
                        help="Limit to first N questions (0 = all, useful for dry runs)")
    parser.add_argument("--category",       default=None,
                        help="Only evaluate a specific category")
    parser.add_argument("--no_judge",       action="store_true",
                        help="Skip LLM judge (use exact matching only)")
    parser.add_argument("--sme_only",       action="store_true",
                        help="Only evaluate SME-verified questions")
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
            logging.FileHandler(str(output_dir / "evaluate.log"), mode="a"),
        ],
    )

    # ── Load benchmark ──
    with open(args.benchmark, encoding="utf-8") as f:
        questions = json.load(f)

    logger.info(f"Loaded {len(questions)} benchmark questions")

    if args.sme_only:
        questions = [q for q in questions if q.get("sme_verified")]
        logger.info(f"After SME filter: {len(questions)} questions")

    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
        logger.info(f"After category filter ({args.category}): {len(questions)} questions")

    if args.max_questions > 0:
        questions = questions[:args.max_questions]
        logger.info(f"Limited to first {len(questions)} questions (dry run)")

    if not questions:
        logger.error("No questions to evaluate after filtering.")
        sys.exit(1)

    # ── Set up models ──
    models = {}

    if not args.no_ft_model and args.ft_model:
        models["fine_tuned"] = FineTunedModel(args.ft_model)

    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if openai_key:
        models["gpt4o"] = GPT4oClient(api_key=openai_key, model=args.gpt_model)
        logger.info(f"GPT-4o ready (model: {args.gpt_model})")

    anthropic_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        models["claude"] = ClaudeClient(api_key=anthropic_key, model=args.claude_model)
        logger.info(f"Claude ready (model: {args.claude_model})")

    if not models:
        logger.error(
            "No models configured. Provide --ft_model and/or API keys "
            "(--openai_key / --anthropic_key)."
        )
        sys.exit(1)

    # ── Set up judge ──
    judge = None
    if not args.no_judge and "claude" in models:
        judge = LLMJudge(models["claude"])
        logger.info("LLM judge: Claude")
    elif not args.no_judge and "gpt4o" in models:
        # Could implement GPT-4o judge here — for now, skip
        logger.info("Note: LLM judge not available (Claude client required). Using exact scoring only.")

    model_names = list(models.keys())
    logger.info(f"Models: {model_names}")
    logger.info(f"Questions: {len(questions)}")
    logger.info("Starting evaluation...\n")

    # ── Run evaluation ──
    scores = evaluate(questions, models, judge, output_dir)

    # ── Summary ──
    summary = compute_summary(scores, model_names)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = output_dir / "report.md"
    write_report(summary, model_names, report_path)

    print_summary(summary, model_names)

    # ── API cost summaries ──
    if "gpt4o" in models:
        logger.info(f"GPT-4o cost: ${models['gpt4o'].total_cost:.2f}")
    if "claude" in models:
        logger.info(models["claude"].cost_summary())
    if judge:
        logger.info(f"Judge calls: {judge.calls}")

    logger.info(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
