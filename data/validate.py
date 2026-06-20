"""
validate.py — Dataset Deduplication, Filtering & Stats
=======================================================
Cleans a list of raw Q&A pairs and exports them for SME review.

Steps:
  1. Deduplication  — remove exact/near-duplicate instructions
  2. Quality filter — length, placeholder, coherence checks
  3. Stats report   — length distributions, topic coverage
  4. SME export     — saves a random sample as a review spreadsheet

Usage (standalone):
    python validate.py --input ./processed/raw_pairs.json --output ./processed/
"""

import re
import json
import random
import logging
import hashlib
from pathlib import Path
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ──────────────────────────────────────────────────────────────────────────────

def _fingerprint(text: str) -> str:
    """Normalise and hash an instruction for dedup comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return hashlib.md5(text.encode()).hexdigest()


def dedup(pairs: list[dict]) -> tuple[list[dict], int]:
    """
    Remove duplicate pairs based on instruction fingerprint.

    Returns:
        (deduplicated list, number removed)
    """
    seen = set()
    unique = []
    for pair in pairs:
        fp = _fingerprint(pair.get("instruction", ""))
        if fp not in seen:
            seen.add(fp)
            unique.append(pair)

    removed = len(pairs) - len(unique)
    return unique, removed


# ──────────────────────────────────────────────────────────────────────────────
# QUALITY FILTERS
# ──────────────────────────────────────────────────────────────────────────────

# Unfilled template placeholders
PLACEHOLDER_PATTERN = re.compile(r"\{(topic|passage|question|input|output)\}")

# Patterns that indicate low-quality outputs
NOISE_PATTERNS = [
    re.compile(r"^\s*N/A\s*$", re.I),
    re.compile(r"^\s*None\s*$", re.I),
    re.compile(r"I (don't|cannot|do not) (know|have)", re.I),
    re.compile(r"(cannot|can't) (answer|determine|tell)", re.I),
]


def is_valid(
    pair: dict,
    min_instruction_words: int = 8,
    min_output_words: int = 40,
    max_output_words: int = 700,
) -> tuple[bool, str]:
    """
    Check if a Q&A pair meets quality requirements.

    Returns:
        (is_valid: bool, reason: str)
    """
    inst = pair.get("instruction", "")
    out  = pair.get("output", "")

    if not inst or not out:
        return False, "empty field"

    inst_words = len(inst.split())
    out_words  = len(out.split())

    if inst_words < min_instruction_words:
        return False, f"instruction too short ({inst_words} words)"

    if out_words < min_output_words:
        return False, f"output too short ({out_words} words)"

    if out_words > max_output_words:
        return False, f"output too long ({out_words} words)"

    if PLACEHOLDER_PATTERN.search(inst) or PLACEHOLDER_PATTERN.search(out):
        return False, "unfilled template placeholder"

    if inst.strip() == out.strip():
        return False, "instruction equals output"

    for pattern in NOISE_PATTERNS:
        if pattern.search(out):
            return False, "low-quality output pattern"

    return True, "ok"


def filter_pairs(
    pairs: list[dict],
    min_instruction_words: int = 8,
    min_output_words: int = 40,
    max_output_words: int = 700,
) -> tuple[list[dict], dict]:
    """
    Filter pairs by quality. Returns (valid_pairs, rejection_stats).
    """
    valid = []
    rejection_counts: Counter = Counter()

    for pair in pairs:
        ok, reason = is_valid(
            pair,
            min_instruction_words=min_instruction_words,
            min_output_words=min_output_words,
            max_output_words=max_output_words,
        )
        if ok:
            valid.append(pair)
        else:
            rejection_counts[reason] += 1

    return valid, dict(rejection_counts)


# ──────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(pairs: list[dict]) -> dict:
    """Compute descriptive statistics for the dataset."""
    if not pairs:
        return {}

    inst_lens = [len(p["instruction"].split()) for p in pairs]
    out_lens  = [len(p["output"].split())      for p in pairs]

    def percentile(data, pct):
        data = sorted(data)
        idx  = int(len(data) * pct / 100)
        return data[min(idx, len(data)-1)]

    # Source distribution
    sources = Counter(p.get("source", "unknown") for p in pairs)

    return {
        "total_pairs":   len(pairs),
        "instruction_words": {
            "min":    min(inst_lens),
            "median": percentile(inst_lens, 50),
            "p95":    percentile(inst_lens, 95),
            "max":    max(inst_lens),
        },
        "output_words": {
            "min":    min(out_lens),
            "median": percentile(out_lens, 50),
            "p95":    percentile(out_lens, 95),
            "max":    max(out_lens),
        },
        "top_sources": dict(sources.most_common(10)),
    }


def print_stats(stats: dict) -> None:
    """Print a human-readable stats summary."""
    if not stats:
        print("  (no stats)")
        return
    print(f"  Total pairs:        {stats['total_pairs']:,}")
    iw = stats["instruction_words"]
    print(f"  Instruction words:  min={iw['min']}  median={iw['median']}  p95={iw['p95']}  max={iw['max']}")
    ow = stats["output_words"]
    print(f"  Output words:       min={ow['min']}  median={ow['median']}  p95={ow['p95']}  max={ow['max']}")
    if stats.get("top_sources"):
        print(f"  Top sources:")
        for src, cnt in list(stats["top_sources"].items())[:5]:
            print(f"    {cnt:>5}  {src}")


# ──────────────────────────────────────────────────────────────────────────────
# SME REVIEW EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def export_sme_review(
    pairs: list[dict],
    output_path: str,
    n_samples: int = 100,
    seed: int = 42,
) -> None:
    """
    Export a random sample as a CSV spreadsheet for SME review.

    The SME adds a 'rating' column (Good / Bad / Edit) and an
    'corrected_output' column for any fixes.

    Saves to output_path (should end in .csv).
    """
    try:
        import csv
    except ImportError:
        pass  # csv is stdlib, always available

    random.seed(seed)
    samples = random.sample(pairs, min(n_samples, len(pairs)))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "instruction", "input", "output",
            "source", "rating", "corrected_output", "notes"
        ])
        writer.writeheader()
        for i, pair in enumerate(samples, 1):
            writer.writerow({
                "id":               i,
                "instruction":      pair.get("instruction", ""),
                "input":            pair.get("input", ""),
                "output":           pair.get("output", ""),
                "source":           pair.get("source", ""),
                "rating":           "",           # SME fills: Good / Bad / Edit
                "corrected_output": "",           # SME fills if rating=Edit
                "notes":            "",
            })

    logger.info(f"SME review file saved: {output_path} ({len(samples)} samples)")


def import_sme_corrections(
    pairs: list[dict],
    review_csv_path: str,
) -> list[dict]:
    """
    Apply SME corrections back to the dataset.

    - Rating 'Good' or '' → keep as-is
    - Rating 'Edit' → replace output with corrected_output
    - Rating 'Bad'  → remove from dataset

    Returns the corrected dataset.
    """
    import csv

    corrections = {}
    removals    = set()

    try:
        with open(review_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx    = int(row["id"]) - 1
                rating = row.get("rating", "").strip().lower()
                if rating == "bad":
                    removals.add(idx)
                elif rating == "edit":
                    corrected = row.get("corrected_output", "").strip()
                    if corrected:
                        corrections[idx] = corrected
    except FileNotFoundError:
        logger.warning(f"Review CSV not found: {review_csv_path}")
        return pairs

    result = []
    for i, pair in enumerate(pairs):
        if i in removals:
            continue
        if i in corrections:
            pair = dict(pair)
            pair["output"] = corrections[i]
            pair["sme_reviewed"] = True
        result.append(pair)

    removed    = len(pairs) - len(result) - len(corrections)
    corrected  = len(corrections)
    logger.info(f"SME corrections applied: {corrected} edits, {removed} removals")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(
    pairs: list[dict],
    train_ratio: float = 0.95,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split into train / eval sets."""
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]


def strip_metadata(pairs: list[dict]) -> list[dict]:
    """Remove pipeline-internal fields before saving final output."""
    keep = {"instruction", "input", "output"}
    return [{k: v for k, v in p.items() if k in keep} for p in pairs]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Validate and clean a raw pairs JSON file")
    parser.add_argument("--input",  required=True, help="Path to raw_pairs.json")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--sme_samples", type=int, default=100)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, encoding="utf-8") as f:
        raw = json.load(f)

    print(f"Loaded {len(raw):,} raw pairs")

    # Dedup
    pairs, n_removed = dedup(raw)
    print(f"After dedup:   {len(pairs):,} pairs  ({n_removed} removed)")

    # Filter
    pairs, rejections = filter_pairs(pairs)
    print(f"After filter:  {len(pairs):,} pairs")
    for reason, cnt in sorted(rejections.items(), key=lambda x: -x[1]):
        print(f"  Rejected ({cnt}): {reason}")

    # Stats
    stats = compute_stats(pairs)
    print("\nDataset stats:")
    print_stats(stats)

    # SME export
    sme_path = output_dir / "sme_review.csv"
    export_sme_review(pairs, str(sme_path), n_samples=args.sme_samples)
    print(f"\nSME review file: {sme_path}")

    # Split and save
    train, eval_ = split_dataset(pairs, train_ratio=args.train_ratio)
    train_path = output_dir / "train.json"
    eval_path  = output_dir / "eval.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(strip_metadata(train), f, indent=2, ensure_ascii=False)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(strip_metadata(eval_), f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {len(train):,} train → {train_path}")
    print(f"Saved: {len(eval_):,} eval  → {eval_path}")
