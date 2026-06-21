"""
generate.py — Q&A Pair Generator
==================================
Turns text chunks into instruction-response pairs using two strategies:

  Strategy A — Template-based (free, fast)
      Detects power systems topics in the chunk and fills instruction templates.
      Good for volume. Produces ~2 pairs per chunk.

  Strategy B — GPT-4o-assisted (costs ~$0.02/chunk, best quality)
      Sends each chunk to GPT-4o with a structured prompt.
      GPT-4o generates 3 realistic engineer Q&A pairs grounded in the passage.
      Includes retry logic, cost tracking, and partial-save on failure.

Usage (standalone test):
    python generate.py --text "The Newton-Raphson method is used..." --strategy template
    python generate.py --text "..." --strategy gpt4o --api_key sk-...
"""

import os
import re
import json
import time
import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# POWER SYSTEMS TOPIC DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

TOPICS = [
    # Load flow & network
    "load flow", "power flow", "Newton-Raphson", "Gauss-Seidel", "fast decoupled",
    "bus admittance matrix", "Y-bus", "Ybus", "Z-bus", "power balance",
    # Fault analysis
    "fault analysis", "short circuit", "three-phase fault", "line-to-ground fault",
    "symmetrical components", "positive sequence", "negative sequence", "zero sequence",
    "sequence network", "fault current", "fault impedance",
    # Protection
    "protection relay", "distance relay", "impedance relay", "overcurrent relay",
    "differential relay", "directional relay", "pilot protection", "POTT", "DCUB",
    "protection coordination", "relay setting", "pickup current", "reach setting",
    "zone 1", "zone 2", "zone 3", "time overcurrent", "TOC",
    # Stability
    "transient stability", "voltage stability", "frequency stability", "small signal stability",
    "equal area criterion", "swing equation", "rotor angle", "power angle",
    "critical clearing time", "CCT", "voltage collapse", "P-V curve", "Q-V curve",
    # Reactive power & voltage
    "reactive power", "power factor", "VAR compensation", "STATCOM", "SVC",
    "shunt reactor", "shunt capacitor", "tap changer", "voltage regulation",
    # Generation & machines
    "synchronous generator", "excitation system", "AVR", "governor", "PSS",
    "inertia constant", "droop control", "primary frequency response",
    "economic dispatch", "optimal power flow", "OPF", "unit commitment",
    # Transmission
    "transmission line", "overhead line", "cable", "surge impedance",
    "per-unit system", "base MVA", "base kV", "transformer", "OLTC",
    "bundled conductor", "corona", "skin effect",
    # HVDC & FACTS
    "HVDC", "VSC-HVDC", "LCC-HVDC", "FACTS", "UPFC", "SSSC", "TCSC",
    # Power quality
    "harmonic", "power quality", "THD", "voltage sag", "voltage swell",
    "flicker", "unbalance", "IEEE 519", "IEC 61000",
    # Grid operation & control
    "SCADA", "EMS", "energy management", "state estimation", "contingency analysis",
    "AGC", "automatic generation control", "frequency regulation", "inertia response",
    # Renewables & microgrids
    "wind turbine", "solar PV", "photovoltaic", "grid integration", "curtailment",
    "battery storage", "BESS", "microgrid", "islanding", "grid-forming inverter",
    # Protection standards
    "IEC 60909", "IEEE C37", "NERC", "ANSI", "IEC 61850", "GOOSE", "sampled values",
    # Grounding
    "grounding", "earthing", "neutral", "ground fault", "earth fault",
    "surge arrester", "lightning protection",
]

# Sort by length descending so more specific phrases match first
TOPICS.sort(key=len, reverse=True)


def detect_topics(text: str, max_topics: int = 3) -> list[str]:
    """Return up to max_topics power systems topics found in the text."""
    text_lower = text.lower()
    found = []
    for topic in TOPICS:
        if topic.lower() in text_lower:
            found.append(topic)
            if len(found) >= max_topics:
                break
    return found


# ──────────────────────────────────────────────────────────────────────────────
# INSTRUCTION TEMPLATES
# ──────────────────────────────────────────────────────────────────────────────

# Each template is (instruction_template, needs_topic, needs_passage)
TEMPLATES = [
    # Passage-based (response = chunk text)
    ("What are the key technical points covered in the following power systems passage?\n\n{passage}",
     False, True),
    ("Summarise the following passage for a power systems engineer:\n\n{passage}",
     False, True),
    ("A junior engineer asks: based on the reference below, {question}\n\nReference material:\n{passage}",
     False, True),
    ("Explain what the following technical content means in practice:\n\n{passage}",
     False, True),

    # Topic-based (response = chunk text as the answer context)
    ("Explain the concept of {topic} in power systems engineering.",
     True, False),
    ("What are the key considerations when analysing {topic}?",
     True, False),
    ("Describe the standard procedure for {topic} in power system studies.",
     True, False),
    ("What are typical parameter values and industry standards for {topic}?",
     True, False),
    ("How does {topic} affect power system stability and reliability?",
     True, False),
    ("What are the common failure modes and engineering challenges related to {topic}?",
     True, False),
    ("A power systems engineer needs to evaluate {topic}. What are the steps involved?",
     True, False),
]

FALLBACK_QUESTIONS = [
    "what are the main technical considerations described here?",
    "how would this be applied in real grid operations?",
    "what standards or codes govern this aspect?",
    "what are the key parameters an engineer must know?",
    "how would you calculate or verify this in practice?",
]


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY A — TEMPLATE-BASED GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def _is_equation_heavy(text: str) -> bool:
    """
    Return True if the chunk is dominated by equations/symbols rather than prose.
    These chunks produce meaningless template pairs and should be skipped.
    """
    words = text.split()
    if not words:
        return True
    # Count tokens that look like math: contain digits mixed with symbols,
    # brackets, or are very short non-words
    math_tokens = sum(
        1 for w in words
        if re.search(r"[=\[\]{}<>\\]", w)
        or re.search(r"\d", w) and re.search(r"[^0-9\s.,;:()%]", w)
        or (len(w) <= 2 and not w.isalpha())
    )
    return math_tokens / len(words) > 0.25   # >25% math tokens → skip


def generate_template(chunk: str, num_pairs: int = 2) -> list[dict]:
    """
    Generate instruction-response pairs from a chunk using templates.
    Fast, free, no LLM needed.

    Only uses passage-based templates so the answer is always grounded in the
    same text as the question. Topic-based templates are excluded because the
    chunk used as a generic answer is frequently unrelated to the topic keyword
    that was detected.
    """
    # Skip chunks that are mostly equations — they produce garbage pairs
    if _is_equation_heavy(chunk):
        return []

    # Cap response at 300 words for readability
    response = " ".join(chunk.split()[:300])
    pairs = []

    # Only passage-based templates: question and answer both reference the chunk
    passage_templates = [
        (tmpl, needs_topic, needs_passage)
        for tmpl, needs_topic, needs_passage in TEMPLATES
        if needs_passage
    ]

    if not passage_templates:
        return []

    selected = random.sample(passage_templates, min(num_pairs, len(passage_templates)))

    for template, _, _ in selected:
        question = random.choice(FALLBACK_QUESTIONS)
        instruction = template.format(
            passage=chunk[:1000],
            question=question,
        )

        if "{topic}" in instruction or "{passage}" in instruction:
            continue

        pairs.append({
            "instruction": instruction.strip(),
            "input": "",
            "output": response.strip(),
        })

    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY B — GPT-4o ASSISTED GENERATION
# ──────────────────────────────────────────────────────────────────────────────

GPT4O_SYSTEM = (
    "You are a technical content specialist creating training data for a "
    "domain-specialized power systems AI model. You generate realistic, "
    "accurate engineering Q&A pairs from source document passages."
)

GPT4O_USER_TEMPLATE = """\
Generate {n} high-quality instruction-response pairs from the following power systems \
engineering passage.

Rules:
- Instructions must be realistic questions a practising power systems engineer would ask
- Vary question types: definition, calculation approach, practical application, \
standards/codes, troubleshooting
- Responses must be technically accurate and grounded entirely in the passage
- Keep responses between 120 and 400 words
- Do NOT invent facts not present in the passage
- Format output as a JSON array ONLY — no other text:
  [{{"instruction": "...", "input": "", "output": "..."}}]

PASSAGE:
{passage}"""

# Cost tracking (GPT-4o pricing, June 2026)
GPT4O_COST_PER_1K_INPUT  = 0.005   # $0.005 per 1K input tokens
GPT4O_COST_PER_1K_OUTPUT = 0.015   # $0.015 per 1K output tokens


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY C — CLAUDE-ASSISTED GENERATION
# ──────────────────────────────────────────────────────────────────────────────

CLAUDE_SYSTEM = (
    "You are a technical content specialist creating training data for a "
    "domain-specialized power systems AI model. You generate realistic, "
    "accurate engineering Q&A pairs from source document passages."
)

CLAUDE_USER_TEMPLATE = """\
Generate {n} high-quality instruction-response pairs from the following power systems \
engineering passage.

Rules:
- Instructions must be realistic questions a practising power systems engineer would ask
- Vary question types: definition, calculation approach, practical application, \
standards/codes, troubleshooting
- Responses must be technically accurate and grounded entirely in the passage
- Keep responses between 120 and 400 words
- Do NOT invent facts not present in the passage
- Format output as a JSON array ONLY — no other text:
  [{{"instruction": "...", "input": "", "output": "..."}}]

PASSAGE:
{passage}"""


# ──────────────────────────────────────────────────────────────────────────────
# CHAIN-OF-THOUGHT PROMPTS
# ──────────────────────────────────────────────────────────────────────────────

CLAUDE_COT_SYSTEM = (
    "You are a senior power systems engineer and technical trainer creating "
    "chain-of-thought training data for a domain-specialized AI model. "
    "Your answers must show explicit, numbered reasoning steps before reaching "
    "a conclusion — teaching the model HOW to think, not just what the answer is."
)

# CoT prompt variant per question type — each enforces a different reasoning structure
CLAUDE_COT_TEMPLATES = {

    "calculation": """\
From the following power systems passage, generate {n} calculation-based \
instruction-response pairs.

Each response MUST follow this exact structure:
1. Identify given values and what is being solved
2. State the relevant formula or principle
3. Substitute values and show each arithmetic step
4. State the final answer with correct units
5. Sanity-check: does the magnitude make physical sense?

Rules:
- Only generate questions where the passage contains enough data to solve numerically
- Show ALL intermediate steps — do not skip algebra
- Include units at every step
- If the passage lacks a specific number needed, use a clearly labelled assumption
- Format output as JSON array ONLY:
  [{{"instruction": "...", "input": "", "output": "Step 1: ...\\n\\nStep 2: ...\\n\\n...\\n\\nAnswer: ..."}}]

PASSAGE:
{passage}""",

    "diagnosis": """\
From the following power systems passage, generate {n} fault diagnosis or \
troubleshooting instruction-response pairs.

Each response MUST follow this reasoning structure:
1. List the observable symptoms or given conditions
2. Identify possible causes (differential diagnosis)
3. Apply elimination reasoning — rule out each cause with evidence
4. State the most likely root cause and why
5. Recommend the corrective action

Rules:
- Questions must describe a realistic grid scenario (relay operation, abnormal readings, alarms)
- Do not jump to conclusions — show the elimination process
- Ground all reasoning in the passage content
- Format output as JSON array ONLY:
  [{{"instruction": "...", "input": "", "output": "Step 1: ...\\n\\nStep 2: ...\\n\\n...\\n\\nConclusion: ..."}}]

PASSAGE:
{passage}""",

    "conceptual": """\
From the following power systems passage, generate {n} conceptual explanation \
instruction-response pairs.

Each response MUST follow this structure:
1. State the core principle or definition in one sentence
2. Explain the underlying physical or mathematical mechanism
3. Give a concrete example or analogy that makes it intuitive
4. State the practical engineering implication (when does this matter?)
5. Note any common misconceptions or edge cases

Rules:
- Questions should ask "why", "how", or "what happens when" — not just "what is"
- Responses should build understanding progressively, not just recite facts
- Keep each step concise (2–4 sentences)
- Format output as JSON array ONLY:
  [{{"instruction": "...", "input": "", "output": "Step 1: ...\\n\\nStep 2: ...\\n\\n...\\n\\nKey takeaway: ..."}}]

PASSAGE:
{passage}""",

    "standards": """\
From the following power systems passage, generate {n} standards and compliance \
instruction-response pairs.

Each response MUST follow this structure:
1. Identify the applicable standard(s) and relevant clause or section
2. State the specific requirement or threshold verbatim or closely paraphrased
3. Explain the engineering rationale behind this requirement
4. Describe how compliance is verified in practice
5. Note any exceptions, special conditions, or regional variations

Rules:
- Only generate questions where the passage references a named standard (IEEE, IEC, NERC, ANSI)
- Do NOT invent clause numbers or thresholds not present in the passage
- If the passage is ambiguous about the standard version, say so explicitly
- Format output as JSON array ONLY:
  [{{"instruction": "...", "input": "", "output": "Step 1: ...\\n\\nStep 2: ...\\n\\n...\\n\\nCompliance check: ..."}}]

PASSAGE:
{passage}""",
}

# Maps chunk content to the most appropriate CoT question type
COT_TYPE_KEYWORDS = {
    "calculation": [
        "calculate", "compute", "formula", "equation", "pu", "per unit",
        "impedance", "current", "voltage", "power", "mva", "kv", "mw",
        "fault", "load flow", "newton", "gauss", "matrix",
    ],
    "diagnosis": [
        "fault", "failure", "trip", "relay", "alarm", "abnormal", "instability",
        "oscillation", "collapse", "outage", "contingency", "emergency",
        "protection", "detection", "isolation",
    ],
    "standards": [
        "ieee", "iec", "nerc", "ansi", "standard", "requirement", "clause",
        "compliance", "code", "specification", "limit", "threshold", "regulation",
    ],
}


def detect_cot_type(chunk: str) -> str:
    """Detect the most appropriate CoT question type for a chunk."""
    chunk_lower = chunk.lower()
    scores = {qtype: 0 for qtype in COT_TYPE_KEYWORDS}

    for qtype, keywords in COT_TYPE_KEYWORDS.items():
        scores[qtype] = sum(1 for kw in keywords if kw in chunk_lower)

    best = max(scores, key=scores.get)
    # Fall back to conceptual if no strong signal
    return best if scores[best] > 0 else "conceptual"


class ClaudeGenerator:
    """
    Wrapper around the Anthropic API for Claude Q&A generation.
    Tracks cumulative cost and implements retry on errors.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY or pass api_key=."
            )

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=key)
        self.model  = model
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.total_calls         = 0
        self.failed_calls        = 0

    def generate(self, chunk: str, num_pairs: int = 3, max_retries: int = 3) -> list[dict]:
        prompt = CLAUDE_USER_TEMPLATE.format(n=num_pairs, passage=chunk[:2500])

        for attempt in range(max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    system=CLAUDE_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )

                self.total_input_tokens  += resp.usage.input_tokens
                self.total_output_tokens += resp.usage.output_tokens
                self.total_calls         += 1

                raw   = resp.content[0].text.strip()
                pairs = self._parse_response(raw)

                if pairs:
                    return pairs
                logger.warning(f"Claude returned unparseable response (attempt {attempt+1})")

            except Exception as e:
                logger.warning(f"Claude error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt * 2)

        self.failed_calls += 1
        return []

    def _parse_response(self, raw: str) -> list[dict]:
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return []
            else:
                return []

        if not isinstance(data, list):
            return []

        valid = []
        for item in data:
            if not isinstance(item, dict):
                continue
            inst = str(item.get("instruction", "")).strip()
            out  = str(item.get("output", "")).strip()
            if len(inst.split()) >= 8 and len(out.split()) >= 30:
                valid.append({
                    "instruction": inst,
                    "input":       str(item.get("input", "")).strip(),
                    "output":      out,
                })
        return valid

    def generate_cot(self, chunk: str, num_pairs: int = 2, max_retries: int = 3) -> list[dict]:
        """
        Generate Chain-of-Thought instruction-response pairs from a chunk.

        Automatically detects the best CoT question type (calculation, diagnosis,
        conceptual, or standards) based on chunk content, then uses a
        type-specific prompt that enforces explicit step-by-step reasoning.
        """
        cot_type  = detect_cot_type(chunk)
        prompt    = CLAUDE_COT_TEMPLATES[cot_type].format(
            n=num_pairs,
            passage=chunk[:2500],
        )

        for attempt in range(max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=2500,   # CoT answers are longer
                    system=CLAUDE_COT_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )

                self.total_input_tokens  += resp.usage.input_tokens
                self.total_output_tokens += resp.usage.output_tokens
                self.total_calls         += 1

                raw   = resp.content[0].text.strip()
                pairs = self._parse_response(raw)

                # Tag each pair with its CoT type for traceability
                for p in pairs:
                    p["cot_type"] = cot_type

                if pairs:
                    return pairs
                logger.warning(f"Claude CoT returned unparseable response (attempt {attempt+1})")

            except Exception as e:
                logger.warning(f"Claude CoT error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt * 2)

        self.failed_calls += 1
        return []

    def cost_summary(self) -> str:
        # claude-sonnet-4-6: $3/M input, $15/M output
        cost = (self.total_input_tokens  / 1_000_000 * 3.0 +
                self.total_output_tokens / 1_000_000 * 15.0)
        return (
            f"Claude calls: {self.total_calls} ok / {self.failed_calls} failed  |  "
            f"Tokens: {self.total_input_tokens:,} in + {self.total_output_tokens:,} out  |  "
            f"Estimated cost: ${cost:.4f}"
        )


class GPT4oGenerator:
    """
    Wrapper around the OpenAI API for GPT-4o Q&A generation.

    Tracks cumulative cost and implements exponential backoff on rate-limit errors.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key= to GPT4oGenerator()."
            )

        self.client = OpenAI(api_key=key)
        self.model  = model
        self.total_cost       = 0.0
        self.total_input_tok  = 0
        self.total_output_tok = 0
        self.total_calls      = 0
        self.failed_calls     = 0

    def generate(self, chunk: str, num_pairs: int = 3, max_retries: int = 3) -> list[dict]:
        """
        Generate Q&A pairs from a single chunk using GPT-4o.

        Returns a list of dicts with keys: instruction, input, output.
        Returns [] on failure (after retries).
        """
        prompt = GPT4O_USER_TEMPLATE.format(
            n=num_pairs,
            passage=chunk[:2500],  # keep within context budget
        )

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system",  "content": GPT4O_SYSTEM},
                        {"role": "user",    "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=1500,
                )

                # Track usage
                usage = resp.usage
                cost = (
                    usage.prompt_tokens     / 1000 * GPT4O_COST_PER_1K_INPUT +
                    usage.completion_tokens / 1000 * GPT4O_COST_PER_1K_OUTPUT
                )
                self.total_cost       += cost
                self.total_input_tok  += usage.prompt_tokens
                self.total_output_tok += usage.completion_tokens
                self.total_calls      += 1

                raw = resp.choices[0].message.content.strip()
                pairs = self._parse_response(raw)

                if pairs:
                    return pairs
                else:
                    logger.warning(f"GPT-4o returned unparseable response (attempt {attempt+1})")

            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() or "429" in err:
                    wait = 2 ** attempt * 5   # 5s, 10s, 20s
                    logger.warning(f"Rate limited — waiting {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                else:
                    logger.warning(f"GPT-4o error (attempt {attempt+1}): {e}")
                    if attempt == max_retries - 1:
                        self.failed_calls += 1
                        return []
                    time.sleep(2)

        self.failed_calls += 1
        return []

    def _parse_response(self, raw: str) -> list[dict]:
        """Parse GPT-4o JSON response into list of pair dicts."""
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to salvage partial JSON by finding the array
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return []
            else:
                return []

        if not isinstance(data, list):
            return []

        valid = []
        for item in data:
            if not isinstance(item, dict):
                continue
            inst = str(item.get("instruction", "")).strip()
            out  = str(item.get("output", "")).strip()
            if len(inst.split()) >= 8 and len(out.split()) >= 30:
                valid.append({
                    "instruction": inst,
                    "input":       str(item.get("input", "")).strip(),
                    "output":      out,
                })

        return valid

    def cost_summary(self) -> str:
        return (
            f"GPT-4o calls: {self.total_calls} success / {self.failed_calls} failed  |  "
            f"Tokens: {self.total_input_tok:,} in + {self.total_output_tok:,} out  |  "
            f"Estimated cost: ${self.total_cost:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# UNIFIED INTERFACE  (used by pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

def generate_pairs(
    chunk: str,
    strategy: str = "template",
    num_pairs: int = 2,
    gpt_generator: Optional["GPT4oGenerator"] = None,
    claude_generator: Optional["ClaudeGenerator"] = None,
) -> list[dict]:
    """
    Generate Q&A pairs from a single chunk.

    Args:
        chunk:             Text passage to generate from
        strategy:          "template", "gpt4o", "claude", or "both"
        num_pairs:         Number of pairs to request
        gpt_generator:     GPT4oGenerator instance (required for gpt4o/both)
        claude_generator:  ClaudeGenerator instance (required for claude)

    Returns:
        List of {"instruction": ..., "input": ..., "output": ...} dicts
    """
    if strategy == "template":
        return generate_template(chunk, num_pairs=num_pairs)

    if strategy == "gpt4o":
        if not gpt_generator:
            raise ValueError("gpt_generator required for strategy='gpt4o'")
        return gpt_generator.generate(chunk, num_pairs=num_pairs)

    if strategy == "claude":
        if not claude_generator:
            raise ValueError("claude_generator required for strategy='claude'")
        return claude_generator.generate(chunk, num_pairs=num_pairs)

    if strategy == "cot":
        if not claude_generator:
            raise ValueError("claude_generator required for strategy='cot'")
        return claude_generator.generate_cot(chunk, num_pairs=num_pairs)

    if strategy == "both":
        pairs = generate_template(chunk, num_pairs=1)
        if claude_generator:
            pairs += claude_generator.generate(chunk, num_pairs=num_pairs)
        elif gpt_generator:
            pairs += gpt_generator.generate(chunk, num_pairs=num_pairs)
        return pairs

    if strategy == "cot+claude":
        # Best quality: standard pairs for knowledge + CoT pairs for reasoning
        if not claude_generator:
            raise ValueError("claude_generator required for strategy='cot+claude'")
        pairs  = claude_generator.generate(chunk, num_pairs=1)
        pairs += claude_generator.generate_cot(chunk, num_pairs=num_pairs)
        return pairs

    raise ValueError(
        f"Unknown strategy: {strategy!r}. "
        f"Choose 'template', 'gpt4o', 'claude', 'cot', 'both', or 'cot+claude'."
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Test Q&A generator on a text chunk")
    parser.add_argument("--text",     required=True, help="Text chunk to generate from")
    parser.add_argument("--strategy", default="template", choices=["template","gpt4o","both"])
    parser.add_argument("--num_pairs",type=int, default=2)
    parser.add_argument("--api_key",  default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    args = parser.parse_args()

    gpt = None
    if args.strategy in ("gpt4o", "both"):
        gpt = GPT4oGenerator(api_key=args.api_key)

    pairs = generate_pairs(args.text, strategy=args.strategy,
                           num_pairs=args.num_pairs, gpt_generator=gpt)

    print(f"\n✓ Generated {len(pairs)} pairs\n")
    for i, pair in enumerate(pairs, 1):
        print(f"── Pair {i} ──")
        print(f"Q: {pair['instruction'][:200]}")
        print(f"A: {pair['output'][:200]}...")
        print()

    if gpt:
        print(gpt.cost_summary())
