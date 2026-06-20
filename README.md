# Power Systems LLM — Training & Evaluation Pipeline

Domain-specialized AI model for power systems engineering, built in response to **RFP PSX-20260510**.

Fine-tunes **Qwen2.5-14B** on power systems engineering knowledge to outperform general-purpose models (GPT-4o, Claude) on domain-specific tasks including fault calculations, protection relay settings, standards compliance, and operational procedures.

---

## Project Structure

```
training/
├── data/               # Training data pipeline
│   ├── pipeline.py         # Main orchestrator (run this first)
│   ├── extract.py          # PDF / TXT / Markdown / URL extractor + chunker
│   ├── generate.py         # Q&A pair generator (template + GPT-4o)
│   ├── validate.py         # Dedup, filter, SME review export
│   ├── build_dataset.py    # Simplified single-script alternative
│   └── requirements.txt
│
├── algo_a/             # Algorithm A — QLoRA (1× A100 80GB)
│   └── train_qlora.py
│
├── algo_b/             # Algorithm B — Full SFT with DeepSpeed ZeRO-3 (4× A100)
│   └── ds_zero3.json
│
└── eval/               # Evaluation — benchmark builder + model comparison
    ├── build_benchmark.py  # Builds the held-out test set
    ├── evaluate.py         # Runs fine-tuned vs GPT-4o vs Claude, scores results
    └── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
# Training data pipeline
pip install -r data/requirements.txt

# Evaluation pipeline
pip install -r eval/requirements.txt

# Algorithm A (QLoRA)
pip install unsloth transformers trl peft bitsandbytes accelerate
```

### 2. Build training data

Add your power systems source documents (PDFs, TXT, Markdown) to `./raw_docs/`, then run:

```bash
# Template-based generation (free, fast — good starting point)
python data/pipeline.py --sources_dir ./raw_docs --output_dir ./processed

# GPT-4o assisted (recommended for best quality, ~$0.02/chunk)
export OPENAI_API_KEY=sk-...
python data/pipeline.py --sources_dir ./raw_docs --output_dir ./processed --strategy both
```

Output: `processed/train.json`, `processed/eval.json`, `processed/sme_review.csv`

Have a power systems SME review `sme_review.csv` before training.

### 3. Train — Algorithm A (QLoRA, 1× A100)

```bash
python algo_a/train_qlora.py \
    --train_file ./processed/train.json \
    --eval_file  ./processed/eval.json \
    --output_dir ./output/algo_a_adapter
```

### 4. Train — Algorithm B (Full SFT, 4× A100)

```bash
# Using LLaMA-Factory with DeepSpeed ZeRO-3
llamafactory-cli train sft_qwen14b_full.yaml
# or
torchrun --nproc_per_node 4 LLaMA-Factory/src/train.py --deepspeed algo_b/ds_zero3.json ...
```

See [`Fine-Tuning_Algorithms_Step-by-Step_Guide.md`](../Fine-Tuning_Algorithms_Step-by-Step_Guide.md) for the full training runbook.

### 5. Build evaluation benchmark

**Important:** use source documents that were NOT used for training.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python eval/build_benchmark.py \
    --sources_dir ./benchmark_docs \
    --output_dir  ./benchmark \
    --use_claude
```

Have the SME verify `benchmark/benchmark_summary.csv`, then apply corrections:

```bash
python eval/build_benchmark.py \
    --apply_sme ./benchmark/benchmark_summary.csv \
    --output_dir ./benchmark
```

### 6. Compare against GPT-4o and Claude

```bash
python eval/evaluate.py \
    --benchmark    ./benchmark/benchmark.json \
    --ft_model     ./output/algo_a_merged \
    --openai_key   $OPENAI_API_KEY \
    --anthropic_key $ANTHROPIC_API_KEY \
    --output_dir   ./results
```

Output: `results/report.md` (score table), `results/summary.json`, `results/hallucination.json`

---

## Fine-Tuning Algorithms

| # | Algorithm | Method | Hardware | Est. Runtime |
|---|-----------|--------|----------|-------------|
| A | QLoRA | 4-bit NF4 + LoRA (r=32, α=64) via Unsloth | 1× A100 80GB | 25–50 GPU-hours |
| B | Full SFT | BF16 + DeepSpeed ZeRO-3 via LLaMA-Factory | 4× A100 80GB | 80–120 GPU-hours |
| C | DPO *(stretch)* | Preference optimization on top of A or B | 1× A100 80GB | 2–8 GPU-hours |

**Base model:** [`Qwen/Qwen2.5-14B`](https://huggingface.co/Qwen/Qwen2.5-14B) (Apache 2.0)

**Fallback base:** [`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)

---

## Evaluation Benchmark

The benchmark targets areas where the fine-tuned model is expected to outperform general models:

| Category | Target | Why fine-tuned wins |
|----------|--------|---------------------|
| Numerical calculation | 600 Q | Exact multi-step formulas (fault current, load flow, per-unit) |
| Fault diagnosis | 500 Q | Scenario pattern recognition trained on real grid data |
| Standards & codes | 400 Q | GPT-4o hallucinates IEEE/IEC clause numbers; fine-tuned memorises them |
| Equipment specs | 300 Q | Relay settings, transformer ratings from proprietary source docs |
| Operational procedure | 300 Q | Switching sequences, LOTO — procedural precision |
| Terminology | 200 Q | IEC vs IEEE naming differences, regional grid terminology |
| Adversarial | 200 Q | Common misconceptions that trip up general models |

**Scoring (3-tier):**
- `1.0` — Correct or within numerical tolerance
- `0.5` — Right approach, minor error (scored by Claude as judge)
- `0.0` — Wrong or hallucinated

---

## Data Pipeline Architecture

```
Raw documents (PDF / TXT / MD / URLs)
         │
         ▼
    extract.py          ← PyMuPDF, BeautifulSoup
    (extract + clean + chunk ~800 words)
         │
         ▼
    generate.py         ← Template-based (free) or GPT-4o (~$0.02/chunk)
    (instruction-response pairs, Alpaca format)
         │
         ▼
    validate.py         ← dedup, length filter, SME review CSV export
         │
         ▼
    train.json / eval.json   (95% / 5% split)
```

Target dataset size: **30,000–70,000 instruction-response pairs**

---

## Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Data pipeline | Any CPU | 8-core CPU, 32 GB RAM |
| Algorithm A (QLoRA) | 1× A100 40GB | 1× A100 80GB |
| Algorithm B (Full SFT) | 2× A100 80GB | 4× A100 80GB |
| Inference (merged model) | 1× RTX 4090 (24GB) | 1× A100 40GB |

Cloud GPU options: [Lambda Labs](https://lambdalabs.com), [RunPod](https://runpod.io), [Vast.ai](https://vast.ai)

---

## Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `data/pipeline.py`, `eval/evaluate.py` | GPT-4o for data generation and evaluation |
| `ANTHROPIC_API_KEY` | `eval/build_benchmark.py`, `eval/evaluate.py` | Claude for benchmark generation and LLM judge |
| `WANDB_API_KEY` | `algo_a/train_qlora.py` | Weights & Biases training logs (optional) |

---

## What Is NOT in This Repo

The following are excluded from version control (see `.gitignore`):

- `models/` — base model weights (download from HuggingFace)
- `output/` — fine-tuned model checkpoints
- `raw_docs/` — source PDF/TXT documents (client-provided)
- `processed/` — generated training data
- `benchmark/` — evaluation benchmark questions
- `results/` — evaluation results
- `.env` — API keys

---

## References

- [Qwen2.5-14B on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-14B)
- [Unsloth — QLoRA fine-tuning](https://github.com/unslothai/unsloth)
- [LLaMA-Factory — full SFT + DeepSpeed](https://github.com/hiyouga/LLaMA-Factory)
- [TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [QLoRA paper (arXiv:2305.14314)](https://arxiv.org/abs/2305.14314)
- [DPO paper (arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
