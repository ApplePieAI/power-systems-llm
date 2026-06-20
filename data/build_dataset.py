"""
Data Pipeline: Raw Documents → Alpaca Training Dataset
========================================================
Project: RFP PSX-20260510 — Power Systems AI Model

This script converts your raw power systems documents (PDFs, DOCX, TXT)
into instruction-response pairs in Alpaca JSON format ready for QLoRA / Full SFT.

USAGE:
    python build_dataset.py --input_dir ./raw_docs --output_dir ./processed

REQUIREMENTS:
    pip install pymupdf python-docx tqdm openai tiktoken scikit-learn

WHAT IT DOES:
    1. Reads all PDFs, DOCX, TXT files from --input_dir
    2. Extracts and cleans text
    3. Chunks text into passages
    4. Generates instruction-response pairs using one of two strategies:
       a) Template-based (fast, no LLM needed) — for structured content
       b) GPT-4o-assisted (slower, higher quality) — for complex passages
    5. Deduplicates and validates pairs
    6. Saves train.json / eval.json splits

RECOMMENDED FLOW:
    Step 1: Run with --strategy template (fast, free)
    Step 2: Review samples in ./processed/samples.json
    Step 3: If quality is good, submit for SME review
    Step 4: Optionally re-run with --strategy gpt4o for a subset
"""

import os
import re
import json
import hashlib
import argparse
import random
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert power systems engineer with deep knowledge of "
    "electrical grid operation, protection, stability analysis, load flow, "
    "fault calculations, energy markets, and power electronics. "
    "Provide accurate, technically precise, and practical answers."
)

# Instruction templates for template-based generation
# Each tuple: (instruction_template, requires_passage)
INSTRUCTION_TEMPLATES = [
    ("Explain the following concept in power systems engineering: {topic}", False),
    ("What does the following passage mean for power systems practice?\n\n{passage}", True),
    ("Summarize the key technical points in the following text:\n\n{passage}", True),
    ("A power systems engineer asks: based on the following reference material, {question}\n\nReference:\n{passage}", True),
    ("Define and explain: {topic}", False),
    ("What are the main considerations when dealing with: {topic}?", False),
    ("Describe the procedure for {topic} in power systems.", False),
    ("What are the typical values and standards for {topic}?", False),
    ("How does {topic} affect power system stability?", False),
    ("Explain the relationship between {topic} and power quality.", False),
]

# Seed questions to use when topic extraction is not possible
FALLBACK_QUESTIONS = [
    "How is this concept applied in real power system operations?",
    "What are the key parameters an engineer needs to consider here?",
    "What standards or codes govern this area of power systems?",
    "What are common failure modes or challenges in this area?",
    "How would you calculate or measure this in practice?",
]

# ──────────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(filepath: str) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except ImportError:
        print("  [!] PyMuPDF not installed. Install with: pip install pymupdf")
        return ""
    except Exception as e:
        print(f"  [!] Error reading PDF {filepath}: {e}")
        return ""


def extract_text_from_docx(filepath: str) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Also extract table content
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        return "\n".join(paragraphs)
    except ImportError:
        print("  [!] python-docx not installed. Install with: pip install python-docx")
        return ""
    except Exception as e:
        print(f"  [!] Error reading DOCX {filepath}: {e}")
        return ""


def extract_text_from_txt(filepath: str) -> str:
    """Extract text from plain text file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"  [!] Error reading TXT {filepath}: {e}")
        return ""


def extract_text(filepath: str) -> str:
    """Route to correct extractor based on file extension."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".docx":
        return extract_text_from_docx(filepath)
    elif ext in [".txt", ".md", ".rst"]:
        return extract_text_from_txt(filepath)
    else:
        print(f"  [!] Unsupported file type: {ext}, skipping {filepath}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean extracted text: fix encoding, remove boilerplate, normalize whitespace."""
    # Remove common PDF artifacts
    text = re.sub(r'\x00', '', text)
    text = re.sub(r'[\x01-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)

    # Fix common OCR/PDF extraction issues
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)       # rejoin hyphenated line breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)        # single newline → space
    text = re.sub(r'\n{3,}', '\n\n', text)              # 3+ newlines → 2
    text = re.sub(r' {2,}', ' ', text)                  # multiple spaces → one
    text = re.sub(r'Page \d+ of \d+', '', text)         # remove page markers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # standalone page numbers

    # Remove very short lines (likely headers/footers)
    lines = text.split('\n')
    lines = [l for l in lines if len(l.strip()) > 10 or l.strip() == '']
    text = '\n'.join(lines)

    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# TEXT CHUNKING
# ──────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks by words.
    chunk_size: target words per chunk
    overlap: words shared between consecutive chunks
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        # Only keep chunks with enough content
        if len(chunk.split()) >= 50:
            chunks.append(chunk)

        if end >= len(words):
            break
        start = end - overlap

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# TOPIC EXTRACTION (for template-based generation)
# ──────────────────────────────────────────────────────────────────────────────

# Common power systems topics to detect in text
POWER_SYSTEMS_TOPICS = [
    "load flow", "power flow", "Newton-Raphson", "Gauss-Seidel", "fast decoupled",
    "fault analysis", "short circuit", "symmetrical components", "sequence network",
    "protection relay", "distance relay", "overcurrent relay", "differential relay",
    "transient stability", "voltage stability", "frequency stability", "small signal",
    "power factor", "reactive power", "VAR compensation", "STATCOM", "SVC",
    "transformer", "per-unit system", "base MVA", "bus admittance matrix", "Ybus",
    "economic dispatch", "optimal power flow", "unit commitment", "energy market",
    "SCADA", "EMS", "DMS", "smart grid", "AMI", "DERMS",
    "transmission line", "cable", "overhead line", "bundled conductor",
    "generator", "synchronous machine", "excitation system", "AVR", "PSS",
    "HVDC", "VSC", "LCC", "multi-terminal", "FACTS",
    "harmonic", "power quality", "THD", "voltage flicker", "sag", "swell",
    "grounding", "neutral", "earthing", "lightning arrester", "surge protection",
    "capacitor bank", "reactor", "shunt compensation", "series compensation",
    "frequency regulation", "AGC", "primary response", "inertia", "droop",
    "renewable energy", "wind turbine", "solar PV", "grid integration",
    "battery storage", "microgrid", "islanding", "grid-forming inverter",
]


def extract_topic_from_chunk(chunk: str) -> Optional[str]:
    """Try to identify a power systems topic from a text chunk."""
    chunk_lower = chunk.lower()
    found_topics = []
    for topic in POWER_SYSTEMS_TOPICS:
        if topic.lower() in chunk_lower:
            found_topics.append(topic)

    if found_topics:
        # Return the most specific (longest) matching topic
        return max(found_topics, key=len)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# PAIR GENERATION — TEMPLATE STRATEGY
# ──────────────────────────────────────────────────────────────────────────────

def generate_pairs_template(chunk: str, num_pairs: int = 2) -> list[dict]:
    """
    Generate instruction-response pairs from a chunk using templates.
    Fast and free — no LLM needed. Good starting point.
    """
    pairs = []
    topic = extract_topic_from_chunk(chunk)

    # Truncate chunk for use as response (first 600 words)
    response = " ".join(chunk.split()[:600])

    templates_used = random.sample(INSTRUCTION_TEMPLATES, min(num_pairs, len(INSTRUCTION_TEMPLATES)))

    for template, requires_passage in templates_used:
        if requires_passage:
            question = random.choice(FALLBACK_QUESTIONS)
            instruction = template.format(
                passage=chunk[:1500],
                question=question,
            )
        elif topic:
            instruction = template.format(topic=topic)
        else:
            continue  # skip if no topic found and template needs one

        if instruction and response:
            pairs.append({
                "instruction": instruction.strip(),
                "input": "",
                "output": response.strip(),
            })

    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# PAIR GENERATION — GPT-4o STRATEGY (optional, higher quality)
# ──────────────────────────────────────────────────────────────────────────────

def generate_pairs_gpt4o(chunk: str, client, num_pairs: int = 3) -> list[dict]:
    """
    Use GPT-4o to generate high-quality Q&A pairs from a chunk.
    Requires OPENAI_API_KEY in environment. ~$0.01–0.05 per chunk.
    """
    prompt = f"""You are creating training data for a domain-specialized power systems AI model.

Given the following text passage from a power systems document, generate {num_pairs} high-quality instruction-response pairs.

Rules:
- Instructions should be realistic questions a power systems engineer would ask
- Responses should be technically accurate, detailed, and grounded in the passage
- Vary the question types: definition, calculation approach, practical application, comparison
- Format as JSON array: [{{"instruction": "...", "input": "", "output": "..."}}]
- Do NOT include questions that cannot be answered from the passage alone
- Keep responses 150-400 words

PASSAGE:
{chunk[:2000]}

Return ONLY the JSON array, no other text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON (handle markdown code blocks)
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        pairs = json.loads(raw)

        # Validate structure
        valid_pairs = []
        for p in pairs:
            if all(k in p for k in ["instruction", "input", "output"]):
                if len(p["instruction"]) > 20 and len(p["output"]) > 50:
                    valid_pairs.append(p)
        return valid_pairs

    except json.JSONDecodeError as e:
        print(f"  [!] JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"  [!] GPT-4o error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ──────────────────────────────────────────────────────────────────────────────

def dedup_pairs(pairs: list[dict]) -> list[dict]:
    """Remove duplicate pairs based on instruction hash."""
    seen = set()
    unique = []
    for pair in pairs:
        key = hashlib.md5(pair["instruction"].lower().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate_pair(pair: dict) -> bool:
    """Check if a pair meets minimum quality requirements."""
    inst = pair.get("instruction", "")
    out = pair.get("output", "")

    if len(inst.split()) < 8:   return False   # too short
    if len(out.split()) < 30:   return False   # response too short
    if len(out.split()) > 800:  return False   # response too long
    if inst == out:             return False   # instruction = response (bug)
    if "{passage}" in inst:     return False   # unfilled template

    return True


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build power systems training dataset")
    parser.add_argument("--input_dir",   default="./raw_docs",   help="Folder with PDFs/DOCX/TXT")
    parser.add_argument("--output_dir",  default="./processed",  help="Where to save output")
    parser.add_argument("--strategy",    default="template",     choices=["template", "gpt4o"],
                        help="Pair generation strategy (template=fast/free, gpt4o=best quality)")
    parser.add_argument("--chunk_size",  type=int, default=800,  help="Words per text chunk")
    parser.add_argument("--pairs_per_chunk", type=int, default=2, help="Pairs to generate per chunk")
    parser.add_argument("--train_split", type=float, default=0.95, help="Train fraction (rest=eval)")
    parser.add_argument("--max_files",   type=int, default=0,    help="Limit files (0=all)")
    parser.add_argument("--seed",        type=int, default=42,   help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Setup GPT-4o client if needed ---
    gpt_client = None
    if args.strategy == "gpt4o":
        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("[ERROR] OPENAI_API_KEY not set. Export it or use --strategy template")
                return
            gpt_client = OpenAI(api_key=api_key)
            print("[OK] GPT-4o client initialized")
        except ImportError:
            print("[ERROR] openai not installed: pip install openai")
            return

    # --- Find all source files ---
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"[ERROR] Input directory not found: {args.input_dir}")
        print(f"  Create it and add your PDF/DOCX/TXT files there, then re-run.")
        return

    all_files = []
    for ext in ["*.pdf", "*.docx", "*.txt", "*.md"]:
        all_files.extend(input_path.rglob(ext))

    if args.max_files > 0:
        all_files = all_files[:args.max_files]

    if not all_files:
        print(f"[ERROR] No PDF/DOCX/TXT files found in {args.input_dir}")
        return

    print(f"\n[INFO] Found {len(all_files)} files to process")
    print(f"[INFO] Strategy: {args.strategy}")
    print(f"[INFO] Chunk size: {args.chunk_size} words | Pairs per chunk: {args.pairs_per_chunk}")
    print("-" * 60)

    # --- Process files ---
    all_pairs = []
    for filepath in tqdm(all_files, desc="Processing files"):
        filepath = str(filepath)
        filename = Path(filepath).name

        # Extract text
        text = extract_text(filepath)
        if not text or len(text.split()) < 100:
            print(f"  [SKIP] {filename}: too little text ({len(text.split())} words)")
            continue

        # Clean
        text = clean_text(text)

        # Chunk
        chunks = chunk_text(text, chunk_size=args.chunk_size)
        if not chunks:
            continue

        # Generate pairs
        file_pairs = []
        for chunk in chunks:
            if args.strategy == "gpt4o":
                pairs = generate_pairs_gpt4o(chunk, gpt_client, num_pairs=args.pairs_per_chunk)
            else:
                pairs = generate_pairs_template(chunk, num_pairs=args.pairs_per_chunk)
            file_pairs.extend(pairs)

        # Add source metadata
        for pair in file_pairs:
            pair["source"] = filename

        all_pairs.extend(file_pairs)
        tqdm.write(f"  {filename}: {len(chunks)} chunks → {len(file_pairs)} pairs")

    if not all_pairs:
        print("[ERROR] No pairs generated. Check your input files.")
        return

    # --- Deduplicate and validate ---
    print(f"\n[INFO] Before dedup/validation: {len(all_pairs)} pairs")
    all_pairs = dedup_pairs(all_pairs)
    all_pairs = [p for p in all_pairs if validate_pair(p)]
    print(f"[INFO] After dedup/validation:  {len(all_pairs)} pairs")

    # --- Shuffle and split ---
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * args.train_split)
    train_pairs = all_pairs[:split_idx]
    eval_pairs  = all_pairs[split_idx:]

    # Remove 'source' field from final output (not needed for training)
    def strip_source(pairs):
        return [{"instruction": p["instruction"], "input": p.get("input",""), "output": p["output"]}
                for p in pairs]

    # --- Save outputs ---
    train_path = Path(args.output_dir) / "train.json"
    eval_path  = Path(args.output_dir) / "eval.json"
    sample_path = Path(args.output_dir) / "samples.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(strip_source(train_pairs), f, indent=2, ensure_ascii=False)

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(strip_source(eval_pairs), f, indent=2, ensure_ascii=False)

    # Save 20 samples for human review
    samples = random.sample(all_pairs, min(20, len(all_pairs)))
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"  Train:   {len(train_pairs):,} pairs  →  {train_path}")
    print(f"  Eval:    {len(eval_pairs):,} pairs  →  {eval_path}")
    print(f"  Samples: {len(samples)} pairs  →  {sample_path}")
    print()
    print("NEXT STEPS:")
    print("  1. Review samples.json to check quality")
    print("  2. Have the SME review and correct any bad pairs")
    print("  3. Copy train.json and eval.json to ./data/ in your training directory")
    print("  4. Run: python algo_a/train_qlora.py  (1× A100)")
    print("  5. Run: bash algo_b/run_algo_b.sh     (4× A100)")


if __name__ == "__main__":
    main()
