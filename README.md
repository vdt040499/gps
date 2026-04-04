# GPS-VI: Genetic Prompt Search for Vietnamese

Vietnamese implementation of **“GPS: Genetic Prompt Search for Efficient Few-shot Learning”** (EMNLP 2022) — Evolutionary Algorithms course project.

---

## Overview

GPS uses a **genetic algorithm** to search for strong NLP prompts automatically, instead of hand-writing templates. Core ideas:

- **Population** = set of prompt templates  
- **Fitness** = accuracy on a small development set (32 examples)  
- **Mutation** = new prompts from a language model  
- **Selection** = keep the top-K prompts each generation  

This project applies GPS to **Vietnamese sentiment analysis** on UIT-VSFC.

---

## Expected results (English, original paper)

| Strategy | Accuracy |
|---|---|
| Hand-crafted prompt (baseline) | 57.52% |
| Back Translation | 60.65% |
| Cloze (T5) | 57.65% |
| Sentence Continuation | **61.72%** |

---

## Requirements

- Python 3.10+
- RAM: 8GB minimum, **16GB recommended**
- Disk: ~5GB (models + data)
- GPU: optional (Apple MPS and CPU supported)

---

## Setup

**1 — Clone and environment:**
```bash
git clone https://github.com/<your-org>/gps_research.git
cd gps_research
python -m venv venv
source venv/bin/activate       # macOS/Linux
# or: venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

**2 — Download models locally (~4GB, once):**
```bash
python scripts/download_models.py
```

**3 — Prepare the dataset:**
```bash
python data/prepare_data.py
```

**4 — Run the demo:**
```bash
python app.py
# Open http://127.0.0.1:7860
```

---

## Project layout

```
gps_research/
├── CLAUDE.md               ← context for AI assistants
├── README.md               ← this file
├── requirements.txt
├── app.py                  ← Gradio web demo
│
├── core/
│   ├── ga_engine.py        ← main GA loop
│   └── scorer.py           ← prompt scoring
│
├── mutation/
│   ├── __init__.py         ← PromptMutator
│   ├── back_translation.py
│   ├── cloze.py
│   └── sentence_cont.py
│
├── data/
│   ├── prepare_data.py
│   ├── vi_sentiment_dev.json
│   └── seed_prompts.json
│
├── models/                 ← not in git (.gitignore)
├── results/
├── logs/
├── docs/
└── scripts/
    └── download_models.py
```

---

## Team modules

| Module | Area | Guide |
|---|---|---|
| Core GA engine | `core/ga_engine.py`, `core/scorer.py` | [docs/core_ga_engine_guide.md](docs/core_ga_engine_guide.md) |
| Mutation strategies | `mutation/*.py` | [docs/mutation_strategies_guide.md](docs/mutation_strategies_guide.md) |
| Data + web demo | `data/prepare_data.py`, `app.py` | [docs/data_and_web_demo_guide.md](docs/data_and_web_demo_guide.md) |

---

## Documentation

- [System architecture](docs/architecture.md)
- [API reference](docs/api_reference.md)
- [Core GA engine & scorer](docs/core_ga_engine_guide.md)
- [Mutation strategies](docs/mutation_strategies_guide.md)
- [Data pipeline & Gradio demo](docs/data_and_web_demo_guide.md)

---

## References

- Paper: [GPS: Genetic Prompt Search for Efficient Few-shot Learning](https://arxiv.org/abs/2210.17041)
- Original repo: [hwxu20/GPS](https://github.com/hwxu20/GPS)
- Dataset (UIT-VSFC CSV mirror): [ura-hcmut/UIT-VSFC](https://huggingface.co/datasets/ura-hcmut/UIT-VSFC) (same corpus as the official UIT-VSFC paper; `prepare_data.py` maps labels to Vietnamese)
- Scoring model: [google/flan-t5-large](https://huggingface.co/google/flan-t5-large)
