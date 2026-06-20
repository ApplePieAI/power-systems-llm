"""
build_benchmark.py — Power Systems Benchmark Dataset Builder
=============================================================
Project: RFP PSX-20260510

Builds the held-out evaluation benchmark used to compare the fine-tuned model
against GPT-4o and Claude. This is separate from the training pipeline —
source documents used here must NOT overlap with training sources.

Benchmark structure (target ~2,500 questions):
  - numerical_calculation   (600)  exact math with tolerance window
  - standards_codes         (400)  IEEE/IEC/NERC clause accuracy
  - equipment_specs         (300)  relay settings, transformer specs
  - operational_procedure   (300)  switching sequences, isolation
  - fault_diagnosis         (500)  scenario-based troubleshooting
  - terminology             (200)  IEC vs IEEE, regional differences
  - adversarial             (200)  misconceptions, trick questions

USAGE:
    # Generate template questions (free):
    python build_benchmark.py --sources_dir ./benchmark_docs --output_dir ./benchmark

    # Generate with Claude (recommended for quality):
    export ANTHROPIC_API_KEY=sk-ant-...
    python build_benchmark.py --sources_dir ./benchmark_docs --output_dir ./benchmark --use_claude

    # Merge manually written questions:
    python build_benchmark.py --merge ./my_questions.json --output_dir ./benchmark

IMPORTANT: benchmark_docs/ must contain DIFFERENT documents than those used
for training. Keep a strict train/test source split.

OUTPUT:
    benchmark/
      benchmark.json        Full benchmark with metadata
      benchmark_summary.csv For human review before running eval
      stats.json            Category/difficulty distribution
"""

import os
import re
import sys
import json
import uuid
import random
import logging
import argparse
from pathlib import Path
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARK QUESTION SCHEMA
# ──────────────────────────────────────────────────────────────────────────────

CATEGORIES = [
    "numerical_calculation",
    "standards_codes",
    "equipment_specs",
    "operational_procedure",
    "fault_diagnosis",
    "terminology",
    "adversarial",
]

DIFFICULTIES = ["easy", "medium", "hard"]

# Category targets for a balanced 2,500-question benchmark
CATEGORY_TARGETS = {
    "numerical_calculation":  600,
    "standards_codes":        400,
    "equipment_specs":        300,
    "operational_procedure":  300,
    "fault_diagnosis":        500,
    "terminology":            200,
    "adversarial":            200,
}


def make_question(
    question: str,
    answer: str,
    category: str,
    difficulty: str = "medium",
    answer_working: str = "",
    tolerance_pct: float = 0.0,
    source_doc: str = "",
    subcategory: str = "",
) -> dict:
    """Create a benchmark question record with full metadata."""
    if category not in CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of: {CATEGORIES}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Invalid difficulty '{difficulty}'. Must be one of: {DIFFICULTIES}")

    return {
        "id": f"ps-{category[:4]}-{uuid.uuid4().hex[:6]}",
        "category": category,
        "subcategory": subcategory,
        "difficulty": difficulty,
        "question": question.strip(),
        "answer": answer.strip(),
        "answer_working": answer_working.strip(),
        "tolerance_pct": tolerance_pct,   # 0 = exact match, 5 = ±5% accepted
        "source_doc": source_doc,
        "sme_verified": False,            # SME must flip this to True before final run
    }


# ──────────────────────────────────────────────────────────────────────────────
# TEMPLATE-BASED QUESTION SEEDS
# Seeds are structured prompts sent to Claude, or used directly as templates.
# ──────────────────────────────────────────────────────────────────────────────

CLAUDE_SYSTEM = (
    "You are a power systems engineering expert creating benchmark evaluation "
    "questions to test a domain-specialized AI model. Questions must be technically "
    "precise, unambiguous, and answerable from engineering principles or standards. "
    "Do not invent made-up standards or incorrect facts."
)

CLAUDE_CATEGORY_PROMPTS = {
    "numerical_calculation": """\
Generate {n} power systems engineering calculation questions from this passage.

Each question must:
- Require a specific numerical answer (not a range or description)
- Include all parameters needed to solve it in the question text
- Have a worked solution showing the calculation steps
- Be at {difficulty} difficulty level

Passage:
{passage}

Return a JSON array ONLY:
[{{
  "question": "...",
  "answer": "numerical answer with units",
  "answer_working": "step-by-step solution",
  "tolerance_pct": 3,
  "subcategory": "e.g. fault_current / load_flow / per_unit"
}}]""",

    "standards_codes": """\
Generate {n} questions about IEEE, IEC, NERC, or ANSI standards from this passage.

Each question must:
- Ask about a specific clause, threshold, or requirement from a named standard
- Have a precise, verifiable answer (not an opinion)
- Be at {difficulty} difficulty level

Passage:
{passage}

Return a JSON array ONLY:
[{{
  "question": "...",
  "answer": "specific standard requirement or value",
  "answer_working": "explanation with standard reference",
  "tolerance_pct": 0,
  "subcategory": "standard name e.g. IEEE_C37 / IEC_61850 / NERC_FAC"
}}]""",

    "fault_diagnosis": """\
Generate {n} scenario-based fault diagnosis questions from this passage.

Each question must:
- Describe a real grid scenario with specific measurements, relay events, or alarms
- Ask the engineer to identify the fault type, location, or recommended action
- Have a single best answer (not open-ended)
- Be at {difficulty} difficulty level

Passage:
{passage}

Return a JSON array ONLY:
[{{
  "question": "...",
  "answer": "specific fault type, location, or action",
  "answer_working": "reasoning chain from symptoms to conclusion",
  "tolerance_pct": 0,
  "subcategory": "e.g. protection / stability / equipment"
}}]""",

    "equipment_specs": """\
Generate {n} questions about equipment specifications, ratings, or settings from this passage.

Each question must:
- Ask about a specific parameter, rating, or setting with a precise answer
- Reference real equipment or industry-standard configurations
- Be at {difficulty} difficulty level

Passage:
{passage}

Return a JSON array ONLY:
[{{
  "question": "...",
  "answer": "specific value, setting, or specification",
  "answer_working": "explanation of why this value applies",
  "tolerance_pct": 0,
  "subcategory": "e.g. transformer / relay / cable / generator"
}}]""",

    "operational_procedure": """\
Generate {n} questions about operational procedures, switching sequences, or safety protocols
from this passage.

Each question must:
- Ask about a specific step, sequence, or requirement in a named procedure
- Have a precise, verifiable answer
- Be at {difficulty} difficulty level

Passage:
{passage}

Return a JSON array ONLY:
[{{
  "question": "...",
  "answer": "specific procedural requirement",
  "answer_working": "rationale for this procedure",
  "tolerance_pct": 0,
  "subcategory": "e.g. switching / LOTO / commissioning / protection_testing"
}}]""",

    "terminology": """\
Generate {n} terminology and definition questions from this passage.

Each question must:
- Ask about a specific technical term, acronym, or the difference between two similar terms
- Have a precise, concise answer (1-3 sentences)
- Prefer cases where IEC and IEEE use different terminology for the same concept
- Be at {difficulty} difficulty level

Passage:
{passage}

Return a JSON array ONLY:
[{{
  "question": "...",
  "answer": "precise definition or clarification",
  "answer_working": "",
  "tolerance_pct": 0,
  "subcategory": "e.g. IEC_vs_IEEE / acronym / concept"
}}]""",

    "adversarial": """\
Generate {n} adversarial questions that test common misconceptions in power systems engineering.

Each question must:
- Present a scenario where a naive or partially-informed engineer would make a mistake
- Have an answer that corrects the misconception with the right explanation
- Be at {difficulty} difficulty level

Passage (use as context):
{passage}

Return a JSON array ONLY:
[{{
  "question": "...",
  "answer": "correct answer that addresses the misconception",
  "answer_working": "explanation of why the common mistake is wrong",
  "tolerance_pct": 0,
  "subcategory": "misconception type"
}}]""",
}


# ──────────────────────────────────────────────────────────────────────────────
# CLAUDE-BASED QUESTION GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class ClaudeGenerator:
    """Uses Claude to generate high-quality benchmark questions from passages."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Export it or pass api_key= to ClaudeGenerator()."
            )

        self.client = __import__("anthropic").Anthropic(api_key=key)
        self.model = model
        self.total_calls = 0
        self.failed_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def generate(
        self,
        chunk: str,
        category: str,
        difficulty: str = "medium",
        num_questions: int = 2,
    ) -> list[dict]:
        """Generate benchmark questions from a text chunk."""
        prompt_template = CLAUDE_CATEGORY_PROMPTS.get(category)
        if not prompt_template:
            logger.warning(f"No prompt template for category: {category}")
            return []

        prompt = prompt_template.format(
            n=num_questions,
            difficulty=difficulty,
            passage=chunk[:3000],
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=CLAUDE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

            self.total_calls += 1
            self.total_input_tokens  += resp.usage.input_tokens
            self.total_output_tokens += resp.usage.output_tokens

            raw = resp.content[0].text.strip()
            return self._parse(raw, category, difficulty)

        except Exception as e:
            logger.warning(f"Claude error (category={category}): {e}")
            self.failed_calls += 1
            return []

    def _parse(self, raw: str, category: str, difficulty: str) -> list[dict]:
        """Parse Claude's JSON response into benchmark records."""
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group())
                except Exception:
                    return []
            else:
                return []

        if not isinstance(items, list):
            return []

        results = []
        for item in items:
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if len(q.split()) < 8 or len(a.split()) < 2:
                continue
            try:
                results.append(make_question(
                    question=q,
                    answer=a,
                    category=category,
                    difficulty=difficulty,
                    answer_working=str(item.get("answer_working", "")),
                    tolerance_pct=float(item.get("tolerance_pct", 0)),
                    subcategory=str(item.get("subcategory", "")),
                ))
            except ValueError as e:
                logger.debug(f"Skipping invalid question: {e}")

        return results

    def cost_summary(self) -> str:
        # claude-sonnet-4-6 pricing (input $3/M, output $15/M)
        cost = (self.total_input_tokens / 1_000_000 * 3.0 +
                self.total_output_tokens / 1_000_000 * 15.0)
        return (
            f"Claude calls: {self.total_calls} ok / {self.failed_calls} failed  |  "
            f"Tokens: {self.total_input_tokens:,} in + {self.total_output_tokens:,} out  |  "
            f"Estimated cost: ${cost:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# HAND-AUTHORED QUESTION LOADER
# For the numerical_calculation and standards_codes categories, hand-authored
# questions from the SME are higher quality than generated ones.
# ──────────────────────────────────────────────────────────────────────────────

def load_manual_questions(json_path: str) -> list[dict]:
    """
    Load manually authored benchmark questions.

    Expected format (list of dicts with these fields):
    {
      "category": "numerical_calculation",
      "difficulty": "hard",
      "question": "...",
      "answer": "...",
      "answer_working": "...",
      "tolerance_pct": 3,
      "source_doc": "IEEE_141_1993",
      "subcategory": "fault_current"
    }
    """
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    questions = []
    errors = 0
    for item in raw:
        try:
            q = make_question(
                question=item["question"],
                answer=item["answer"],
                category=item["category"],
                difficulty=item.get("difficulty", "medium"),
                answer_working=item.get("answer_working", ""),
                tolerance_pct=float(item.get("tolerance_pct", 0)),
                source_doc=item.get("source_doc", "manual"),
                subcategory=item.get("subcategory", ""),
            )
            q["sme_verified"] = True  # manual questions are assumed SME-verified
            questions.append(q)
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping malformed manual question: {e}")
            errors += 1

    logger.info(f"Loaded {len(questions)} manual questions ({errors} skipped)")
    return questions


# ──────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ──────────────────────────────────────────────────────────────────────────────

def dedup_benchmark(questions: list[dict]) -> tuple[list[dict], int]:
    """Remove duplicate questions by normalised question text."""
    seen = set()
    unique = []
    for q in questions:
        key = re.sub(r"\s+", " ", q["question"].lower().strip())
        key = re.sub(r"[^\w\s]", "", key)
        fp = hash(key)
        if fp not in seen:
            seen.add(fp)
            unique.append(q)

    removed = len(questions) - len(unique)
    return unique, removed


# ──────────────────────────────────────────────────────────────────────────────
# STATS & EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(questions: list[dict]) -> dict:
    """Compute distribution stats for the benchmark."""
    by_category  = Counter(q["category"]   for q in questions)
    by_difficulty = Counter(q["difficulty"] for q in questions)
    sme_verified = sum(1 for q in questions if q.get("sme_verified"))

    return {
        "total": len(questions),
        "sme_verified": sme_verified,
        "sme_pending": len(questions) - sme_verified,
        "by_category": dict(by_category),
        "by_difficulty": dict(by_difficulty),
        "category_targets": CATEGORY_TARGETS,
        "gaps": {
            cat: max(0, CATEGORY_TARGETS[cat] - by_category.get(cat, 0))
            for cat in CATEGORY_TARGETS
        },
    }


def print_stats(stats: dict) -> None:
    print(f"\n  Total questions: {stats['total']:,}")
    print(f"  SME verified:   {stats['sme_verified']} / {stats['total']}")
    print(f"\n  {'Category':<28} {'Have':>6} {'Target':>8} {'Gap':>6}")
    print(f"  {'-'*52}")
    for cat in CATEGORY_TARGETS:
        have   = stats["by_category"].get(cat, 0)
        target = CATEGORY_TARGETS[cat]
        gap    = stats["gaps"][cat]
        flag   = "  ←" if gap > 0 else ""
        print(f"  {cat:<28} {have:>6} {target:>8} {gap:>6}{flag}")
    print(f"\n  Difficulty: {dict(stats['by_difficulty'])}")


def export_summary_csv(questions: list[dict], output_path: str) -> None:
    """Export a CSV for SME verification review."""
    import csv

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "category", "subcategory", "difficulty",
            "question", "answer", "answer_working",
            "tolerance_pct", "source_doc",
            "sme_verified", "sme_notes", "corrected_answer",
        ])
        writer.writeheader()
        for q in questions:
            writer.writerow({
                "id":              q["id"],
                "category":        q["category"],
                "subcategory":     q.get("subcategory", ""),
                "difficulty":      q["difficulty"],
                "question":        q["question"],
                "answer":          q["answer"],
                "answer_working":  q.get("answer_working", ""),
                "tolerance_pct":   q.get("tolerance_pct", 0),
                "source_doc":      q.get("source_doc", ""),
                "sme_verified":    q.get("sme_verified", False),
                "sme_notes":       "",          # SME fills this
                "corrected_answer": "",         # SME fills if answer is wrong
            })

    logger.info(f"Summary CSV saved: {output_path} ({len(questions)} questions)")


def import_sme_corrections(questions: list[dict], csv_path: str) -> list[dict]:
    """
    Apply SME corrections from the review CSV back to the benchmark.

    SME workflow:
    - Set sme_verified = True if the question+answer is correct
    - Fill corrected_answer if the answer needs fixing
    - Add sme_notes for any issues
    - Leave sme_verified = False to flag for removal
    """
    import csv

    corrections = {}
    removals = set()

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row["id"]
            verified = str(row.get("sme_verified", "")).strip().lower()
            corrected = str(row.get("corrected_answer", "")).strip()

            if verified == "false" or verified == "":
                removals.add(qid)
            elif corrected:
                corrections[qid] = corrected

    result = []
    for q in questions:
        qid = q["id"]
        if qid in removals:
            continue
        if qid in corrections:
            q = dict(q)
            q["answer"] = corrections[qid]
        q["sme_verified"] = qid not in removals
        result.append(q)

    logger.info(
        f"SME corrections applied: {len(corrections)} fixed, "
        f"{len(removals)} removed, {len(result)} remaining"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
    from extract import extract_and_chunk

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(output_dir / "benchmark_build.log"), mode="a"),
        ],
    )

    logger.info("=" * 60)
    logger.info("Power Systems Benchmark Builder")
    logger.info("=" * 60)

    all_questions: list[dict] = []

    # ── Load any existing benchmark to append to ──
    existing_path = output_dir / "benchmark.json"
    if existing_path.exists() and not args.rebuild:
        with open(existing_path, encoding="utf-8") as f:
            all_questions = json.load(f)
        logger.info(f"Loaded {len(all_questions)} existing questions")

    # ── Merge manual questions ──
    if args.merge:
        manual = load_manual_questions(args.merge)
        all_questions.extend(manual)
        logger.info(f"After merge: {len(all_questions)} total")

    # ── Apply SME corrections ──
    if args.apply_sme:
        all_questions = import_sme_corrections(all_questions, args.apply_sme)

    # ── Generate from source documents ──
    if args.sources_dir:
        claude = None
        if args.use_claude:
            claude = ClaudeGenerator(api_key=args.api_key)
            logger.info(f"Claude generator ready (model: {claude.model})")

        sources_dir = Path(args.sources_dir)
        if not sources_dir.exists():
            logger.error(f"Sources directory not found: {sources_dir}")
            sys.exit(1)

        files = []
        for ext in ["*.pdf", "*.txt", "*.md"]:
            files.extend(sorted(sources_dir.rglob(ext)))

        if not files:
            logger.error(f"No PDF/TXT/MD files found in {sources_dir}")
            sys.exit(1)

        logger.info(f"Found {len(files)} source files")

        # Current category counts to decide where to focus
        current_counts = Counter(q["category"] for q in all_questions)

        for filepath in files:
            logger.info(f"\nProcessing: {filepath.name}")
            chunks = extract_and_chunk(str(filepath), chunk_words=600, overlap_words=80)

            if not chunks:
                logger.warning(f"  No chunks extracted — skipping")
                continue

            for chunk in chunks:
                # Determine which categories still need questions
                needed_categories = [
                    cat for cat in CATEGORIES
                    if current_counts.get(cat, 0) < CATEGORY_TARGETS[cat]
                ]
                if not needed_categories:
                    logger.info("  All category targets met — stopping generation")
                    break

                # Pick the category most behind target
                cat = min(needed_categories,
                          key=lambda c: current_counts.get(c, 0) / CATEGORY_TARGETS[c])

                difficulty = random.choice(["medium", "medium", "hard"])  # skew harder

                if claude:
                    questions = claude.generate(
                        chunk, category=cat, difficulty=difficulty,
                        num_questions=args.questions_per_chunk,
                    )
                else:
                    # Template fallback: create simple questions from chunk
                    questions = _template_fallback(chunk, cat, difficulty)

                for q in questions:
                    q["source_doc"] = filepath.name
                    all_questions.append(q)
                    current_counts[cat] = current_counts.get(cat, 0) + 1

        if claude:
            logger.info(f"\n{claude.cost_summary()}")

    # ── Dedup ──
    all_questions, n_dupes = dedup_benchmark(all_questions)
    if n_dupes:
        logger.info(f"Removed {n_dupes} duplicate questions")

    # ── Save ──
    benchmark_path = output_dir / "benchmark.json"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "benchmark_summary.csv"
    export_summary_csv(all_questions, str(csv_path))

    stats = compute_stats(all_questions)
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK BUILD COMPLETE")
    logger.info("=" * 60)
    print_stats(stats)
    logger.info(f"\n  benchmark.json       → {benchmark_path}")
    logger.info(f"  benchmark_summary.csv → {csv_path}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("  1. Open benchmark_summary.csv and have SME set sme_verified=True")
    logger.info("     for each correct question, or fill corrected_answer for fixes")
    logger.info("  2. Apply corrections:")
    logger.info("     python build_benchmark.py --apply_sme benchmark_summary.csv \\")
    logger.info("            --output_dir ./benchmark")
    logger.info("  3. Run evaluation:")
    logger.info("     python eval/evaluate.py --benchmark ./benchmark/benchmark.json \\")
    logger.info("            --model ./output/algo_a_merged")
    logger.info("=" * 60)


def _template_fallback(chunk: str, category: str, difficulty: str) -> list[dict]:
    """Minimal template-based question when Claude is not available."""
    words = chunk.split()
    if len(words) < 60:
        return []

    q = make_question(
        question=f"Based on the following passage, what are the key technical requirements "
                 f"related to this power systems topic?\n\n{chunk[:500]}",
        answer=" ".join(words[:150]),
        category=category,
        difficulty=difficulty,
        answer_working="Extracted from source document passage.",
        source_doc="template",
    )
    return [q]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build power systems evaluation benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sources_dir",        default=None,
                        help="Directory with benchmark-only source PDFs/TXT/MD")
    parser.add_argument("--output_dir",         default="./benchmark")
    parser.add_argument("--merge",              default=None,
                        help="JSON file with manually authored questions to merge in")
    parser.add_argument("--apply_sme",          default=None,
                        help="Apply SME corrections from a reviewed benchmark_summary.csv")
    parser.add_argument("--use_claude",         action="store_true",
                        help="Use Claude API to generate questions (recommended)")
    parser.add_argument("--api_key",            default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--questions_per_chunk", type=int, default=2)
    parser.add_argument("--rebuild",            action="store_true",
                        help="Ignore existing benchmark.json and rebuild from scratch")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
