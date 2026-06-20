"""
pipeline.py — Main Data Pipeline Orchestrator
=============================================
Runs the full dataset-build workflow end-to-end:

  Stage 1  Extract text from all sources (PDF / TXT / MD / URLs)
  Stage 2  Chunk into ~800-word passages
  Stage 3  Generate Q&A pairs (template + GPT-4o)
  Stage 4  Deduplicate and filter
  Stage 5  Export SME review CSV + train/eval splits

HOW TO RUN:
    # First time — template only (free):
    python pipeline.py --sources_dir ./raw_docs --output_dir ./processed

    # With GPT-4o (recommended for best quality):
    export OPENAI_API_KEY=sk-...
    python pipeline.py --sources_dir ./raw_docs --output_dir ./processed --strategy both

    # Add specific URLs:
    python pipeline.py --sources_dir ./raw_docs --urls urls.txt --output_dir ./processed

    # Resume interrupted run (skips already-processed sources):
    python pipeline.py --sources_dir ./raw_docs --output_dir ./processed --resume

INPUTS:
    --sources_dir   Folder with PDF / TXT / MD files
    --urls          Text file with one URL per line (optional)
    --output_dir    Where to save all outputs

OUTPUTS:
    processed/
      raw_pairs.json     All generated pairs (before cleaning)
      train.json         95% split — used for training
      eval.json          5% split — used to monitor training loss
      sme_review.csv     100 random samples for SME quality check
      pipeline.log       Full log of the run
      stats.json         Dataset statistics
"""

import os
import sys
import json
import time
import logging
import argparse
import random
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Local modules
from extract  import extract_and_chunk
from generate import generate_pairs, GPT4oGenerator
from validate import (
    dedup, filter_pairs, compute_stats, print_stats,
    export_sme_review, split_dataset, strip_metadata,
)


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path) -> None:
    log_path = output_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path), mode="a", encoding="utf-8"),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE DISCOVERY
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".rst"}


def discover_sources(sources_dir: str, urls_file: Optional[str] = None) -> list[str]:
    """Find all file paths and URLs to process."""
    sources = []

    # Files in sources_dir
    if sources_dir:
        base = Path(sources_dir)
        if not base.exists():
            logging.warning(f"Sources directory not found: {sources_dir}")
        else:
            for f in sorted(base.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    sources.append(str(f))

    # URLs from text file
    if urls_file:
        url_path = Path(urls_file)
        if url_path.exists():
            with open(url_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and (line.startswith("http://") or line.startswith("https://")):
                        sources.append(line)
        else:
            logging.warning(f"URLs file not found: {urls_file}")

    return sources


# ──────────────────────────────────────────────────────────────────────────────
# PROGRESS CHECKPOINT  (enables --resume)
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: Path) -> tuple[set, list]:
    """Load previously processed sources and collected pairs."""
    if not checkpoint_path.exists():
        return set(), []
    try:
        with open(checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("done", [])), data.get("pairs", [])
    except Exception as e:
        logging.warning(f"Could not load checkpoint: {e}")
        return set(), []


def save_checkpoint(checkpoint_path: Path, done: set, pairs: list) -> None:
    """Save progress so the run can be resumed."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump({"done": list(done), "pairs": pairs}, f, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    log = logging.getLogger(__name__)
    log.info("=" * 60)
    log.info("Power Systems Dataset Pipeline — Starting")
    log.info(f"Strategy:     {args.strategy}")
    log.info(f"Sources dir:  {args.sources_dir}")
    log.info(f"Output dir:   {output_dir}")
    log.info("=" * 60)

    # ── GPT-4o client ──
    gpt = None
    if args.strategy in ("gpt4o", "both"):
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.error("OPENAI_API_KEY not set. Use --openai_api_key or export it.")
            sys.exit(1)
        gpt = GPT4oGenerator(api_key=api_key)
        log.info("GPT-4o client initialised")

    # ── Discover sources ──
    sources = discover_sources(args.sources_dir, args.urls)
    if not sources:
        log.error(f"No sources found in {args.sources_dir}. "
                  f"Add PDF/TXT/MD files or a --urls file.")
        sys.exit(1)
    log.info(f"Found {len(sources)} sources to process")

    # ── Load checkpoint (if resuming) ──
    checkpoint_path = output_dir / ".checkpoint.json"
    done_set, all_pairs = (load_checkpoint(checkpoint_path)
                           if args.resume else (set(), []))
    if args.resume and done_set:
        log.info(f"Resuming: {len(done_set)} sources already done, "
                 f"{len(all_pairs)} pairs collected")

    # ── Stage 1–3: Extract → Chunk → Generate ──
    start_time = time.time()
    pending = [s for s in sources if s not in done_set]
    log.info(f"Processing {len(pending)} remaining sources...")

    for source in tqdm(pending, desc="Sources", unit="src"):
        src_label = Path(source).name if not source.startswith("http") else source[:60]

        # Stage 1+2: Extract and chunk
        chunks = extract_and_chunk(
            source,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            url_delay=args.url_delay,
        )

        if not chunks:
            log.warning(f"  SKIP (no chunks): {src_label}")
            done_set.add(source)
            continue

        # Stage 3: Generate pairs from each chunk
        src_pairs = []
        for chunk in chunks:
            pairs = generate_pairs(
                chunk,
                strategy=args.strategy,
                num_pairs=args.pairs_per_chunk,
                gpt_generator=gpt,
            )
            for p in pairs:
                p["source"] = src_label
            src_pairs.extend(pairs)

        all_pairs.extend(src_pairs)
        done_set.add(source)

        tqdm.write(f"  {src_label}: {len(chunks)} chunks → {len(src_pairs)} pairs  "
                   f"(total: {len(all_pairs):,})")

        # Save checkpoint every 10 sources
        if len(done_set) % 10 == 0:
            save_checkpoint(checkpoint_path, done_set, all_pairs)

    # Final checkpoint save
    save_checkpoint(checkpoint_path, done_set, all_pairs)

    elapsed = time.time() - start_time
    log.info(f"\nExtraction complete in {elapsed/60:.1f} min")
    log.info(f"Raw pairs collected: {len(all_pairs):,}")

    if gpt:
        log.info(f"GPT-4o cost:  {gpt.cost_summary()}")

    if not all_pairs:
        log.error("No pairs generated — check your source documents.")
        sys.exit(1)

    # ── Save raw pairs ──
    raw_path = output_dir / "raw_pairs.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
    log.info(f"Raw pairs saved: {raw_path}")

    # ── Stage 4: Dedup + Filter ──
    log.info("\n── Stage 4: Dedup & Filter ──")

    pairs, n_dupes = dedup(all_pairs)
    log.info(f"After dedup:   {len(pairs):,}  ({n_dupes} duplicates removed)")

    pairs, rejections = filter_pairs(
        pairs,
        min_instruction_words=args.min_instruction_words,
        min_output_words=args.min_output_words,
        max_output_words=args.max_output_words,
    )
    log.info(f"After filter:  {len(pairs):,} pairs")
    for reason, cnt in sorted(rejections.items(), key=lambda x: -x[1]):
        log.info(f"  Rejected ({cnt:>4}): {reason}")

    # ── Stats ──
    stats = compute_stats(pairs)
    log.info("\nDataset statistics:")
    print_stats(stats)

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # ── Stage 5: Export ──
    log.info("\n── Stage 5: Export ──")

    # SME review sample
    sme_path = output_dir / "sme_review.csv"
    export_sme_review(pairs, str(sme_path), n_samples=args.sme_samples)

    # Train / eval split
    train, eval_ = split_dataset(pairs, train_ratio=args.train_ratio, seed=args.seed)

    train_path = output_dir / "train.json"
    eval_path  = output_dir / "eval.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(strip_metadata(train), f, indent=2, ensure_ascii=False)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(strip_metadata(eval_), f, indent=2, ensure_ascii=False)

    # ── Summary ──
    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info(f"  Train:         {len(train):,} pairs  →  {train_path}")
    log.info(f"  Eval:          {len(eval_):,} pairs  →  {eval_path}")
    log.info(f"  SME review:    {min(args.sme_samples, len(pairs))} samples  →  {sme_path}")
    log.info(f"  Stats:         {stats_path}")
    log.info(f"  Total time:    {elapsed/60:.1f} min")
    if gpt:
        log.info(f"  GPT-4o cost:   {gpt.cost_summary()}")
    log.info("")
    log.info("NEXT STEPS:")
    log.info("  1. Open sme_review.csv and have a power systems engineer")
    log.info("     rate each pair: Good / Bad / Edit")
    log.info("  2. When done, run:")
    log.info("     python validate.py --review sme_review.csv --pairs raw_pairs.json")
    log.info("  3. Upload train.json and eval.json to your GPU pod")
    log.info("  4. Run: python algo_a/train_qlora.py")
    log.info("=" * 60)

    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info("(Checkpoint cleaned up)")


# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build power systems LLM training dataset from raw documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Sources
    parser.add_argument("--sources_dir",    default="./raw_docs",
                        help="Folder containing PDF/TXT/MD source files")
    parser.add_argument("--urls",           default=None,
                        help="Text file with one URL per line")

    # Strategy
    parser.add_argument("--strategy",       default="both",
                        choices=["template", "gpt4o", "both"],
                        help="template=fast/free  gpt4o=best quality  both=recommended")
    parser.add_argument("--openai_api_key", default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")

    # Chunking
    parser.add_argument("--chunk_words",    type=int, default=800,
                        help="Target words per chunk")
    parser.add_argument("--overlap_words",  type=int, default=100,
                        help="Word overlap between consecutive chunks")

    # Generation
    parser.add_argument("--pairs_per_chunk", type=int, default=2,
                        help="Q&A pairs to generate per chunk")

    # Filtering
    parser.add_argument("--min_instruction_words", type=int, default=8)
    parser.add_argument("--min_output_words",       type=int, default=40)
    parser.add_argument("--max_output_words",       type=int, default=700)

    # Output
    parser.add_argument("--output_dir",    default="./processed",
                        help="Directory for all output files")
    parser.add_argument("--train_ratio",   type=float, default=0.95)
    parser.add_argument("--sme_samples",   type=int,   default=100,
                        help="Number of samples to export for SME review")
    parser.add_argument("--seed",          type=int,   default=42)

    # URL fetching
    parser.add_argument("--url_delay",     type=float, default=1.5,
                        help="Seconds to wait between URL fetches (be polite)")

    # Resume
    parser.add_argument("--resume",        action="store_true",
                        help="Resume from last checkpoint (skip already-processed sources)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
