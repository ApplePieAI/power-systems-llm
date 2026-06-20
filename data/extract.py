"""
extract.py — Source Document Text Extractor
============================================
Handles three source types:
  • PDF files       (via PyMuPDF / fitz)
  • TXT / Markdown  (plain read)
  • Web URLs        (requests + BeautifulSoup)

Usage (standalone test):
    python extract.py --source ./raw_docs/standard.pdf
    python extract.py --source https://www.nerc.com/pa/Stand/Pages/default.aspx
"""

import re
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

def extract_pdf(filepath: str) -> str:
    """
    Extract text from a PDF using PyMuPDF.

    Strategy:
    - Page-by-page extraction preserving paragraph structure
    - Detects and skips image-only pages (scanned without OCR layer)
    - Joins hyphenated line-breaks back into words
    - Returns empty string with a warning if the file is unreadable
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    try:
        doc = fitz.open(filepath)
    except Exception as e:
        logger.warning(f"Cannot open PDF {filepath}: {e}")
        return ""

    pages_text = []
    skipped = 0

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")  # plain text mode

        # Skip nearly-empty pages (covers, blank pages, image-only)
        if len(text.strip()) < 80:
            skipped += 1
            continue

        pages_text.append(text)

    doc.close()

    if skipped:
        logger.debug(f"  Skipped {skipped} near-empty pages in {Path(filepath).name}")

    return "\n\n".join(pages_text)


# ──────────────────────────────────────────────────────────────────────────────
# TEXT / MARKDOWN EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_file(filepath: str) -> str:
    """
    Read a plain-text or Markdown file.

    For Markdown: strips common syntax tokens (##, **, `, ---) so the model
    trains on clean prose rather than markup characters.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception as e:
        logger.warning(f"Cannot read {filepath}: {e}")
        return ""

    ext = Path(filepath).suffix.lower()
    if ext in (".md", ".markdown"):
        raw = _strip_markdown(raw)

    return raw


def _strip_markdown(text: str) -> str:
    """Remove Markdown syntax, leaving readable prose."""
    # Code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    # Headings
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold / italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Links and images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Table pipes
    text = re.sub(r"\|", " ", text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# WEB URL EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

def extract_url(
    url: str,
    delay: float = 1.5,
    timeout: int = 20,
    max_chars: int = 50_000,
) -> str:
    """
    Fetch a web page and extract its main readable text.

    - Removes nav bars, footers, scripts, and style blocks
    - Targets <main>, <article>, or <body> in that priority order
    - Respects delay between requests (be polite to servers)
    - Returns empty string on HTTP errors or timeouts

    Args:
        url:       Page URL to fetch
        delay:     Seconds to sleep before fetching (rate limiting)
        timeout:   HTTP request timeout in seconds
        max_chars: Truncate output to this many characters
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "Missing dependencies. Run: pip install requests beautifulsoup4"
        )

    time.sleep(delay)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; PowerSystems-DataBot/1.0; "
            "research purposes only)"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "header", "footer",
                      "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    # Try to find main content area
    content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content|main|body", re.I))
        or soup.find("body")
    )

    if not content:
        return ""

    # Extract text with newlines between block elements
    text = content.get_text(separator="\n")
    return text[:max_chars]


# ──────────────────────────────────────────────────────────────────────────────
# TEXT CLEANER  (shared by all extractors)
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise extracted text from any source.

    Steps:
    1. Remove null bytes and control characters
    2. Rejoin words split by hyphen + newline (common in PDFs)
    3. Collapse single newlines (mid-paragraph line wraps) into spaces
    4. Collapse 3+ blank lines into 2
    5. Remove standalone page numbers
    6. Remove lines that are pure noise (< 4 chars after strip)
    7. Collapse multiple spaces
    """
    if not text:
        return ""

    # 1. Remove null bytes and control chars (keep tab + newline)
    text = re.sub(r"\x00", "", text)
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 2. Rejoin hyphenated line breaks: "trans-\nmission" → "transmission"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 3. Single newline (mid-paragraph wrap) → space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 4. 3+ consecutive blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Standalone page numbers (line with only digits, maybe spaces)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # 6. Remove very short lines (likely headers, footers, artefacts)
    lines = text.split("\n")
    lines = [l for l in lines if len(l.strip()) >= 4 or l.strip() == ""]
    text = "\n".join(lines)

    # 7. Multiple spaces → single space
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# CHUNKER
# ──────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_words: int = 800,
    overlap_words: int = 100,
    min_words: int = 60,
) -> list[str]:
    """
    Split cleaned text into overlapping word-based chunks.

    Args:
        text:         Cleaned text to split
        chunk_words:  Target words per chunk
        overlap_words: Words shared between consecutive chunks
        min_words:    Discard chunks shorter than this

    Returns:
        List of text chunks (strings)
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])

        if len(chunk.split()) >= min_words:
            chunks.append(chunk)

        if end >= len(words):
            break
        start = end - overlap_words

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# ROUTER  (used by pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

def extract_and_chunk(
    source: str,
    chunk_words: int = 800,
    overlap_words: int = 100,
    url_delay: float = 1.5,
) -> list[str]:
    """
    Auto-detect source type, extract, clean, and chunk.

    Args:
        source:       File path (PDF/TXT/MD) or URL (http/https)
        chunk_words:  Words per chunk
        overlap_words: Overlap between chunks
        url_delay:    Delay before fetching URLs (seconds)

    Returns:
        List of text chunks ready for Q&A generation
    """
    source = source.strip()

    # Detect type
    if source.startswith("http://") or source.startswith("https://"):
        logger.debug(f"Fetching URL: {source}")
        raw = extract_url(source, delay=url_delay)
        source_label = source
    else:
        path = Path(source)
        if not path.exists():
            logger.warning(f"File not found: {source}")
            return []

        ext = path.suffix.lower()
        if ext == ".pdf":
            logger.debug(f"Reading PDF: {path.name}")
            raw = extract_pdf(str(path))
        elif ext in (".txt", ".md", ".markdown", ".rst"):
            logger.debug(f"Reading text file: {path.name}")
            raw = extract_text_file(str(path))
        else:
            logger.warning(f"Unsupported file type '{ext}': {source}")
            return []

        source_label = path.name

    if not raw:
        logger.warning(f"No text extracted from: {source_label}")
        return []

    cleaned = clean_text(raw)
    word_count = len(cleaned.split())

    if word_count < 100:
        logger.warning(f"Too little text ({word_count} words): {source_label}")
        return []

    chunks = chunk_text(cleaned, chunk_words=chunk_words, overlap_words=overlap_words)
    logger.info(f"  {source_label}: {word_count:,} words → {len(chunks)} chunks")
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# CLI (for quick standalone testing)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Test extractor on a single source")
    parser.add_argument("--source", required=True, help="PDF path, TXT path, or URL")
    parser.add_argument("--chunk_words",   type=int, default=800)
    parser.add_argument("--overlap_words", type=int, default=100)
    args = parser.parse_args()

    chunks = extract_and_chunk(
        args.source,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )

    print(f"\n✓ Extracted {len(chunks)} chunks\n")
    for i, c in enumerate(chunks[:3], 1):
        print(f"── Chunk {i} ({len(c.split())} words) ──")
        print(c[:400])
        print("...")
        print()
