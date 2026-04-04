# CLAUDE.md — GPS-VI (Vietnamese GPS)

This file gives Claude (or any AI assistant) immediate project context without reading the entire codebase.

---

## What is this project?

**GPS-VI** is a Vietnamese implementation of the paper  
["GPS: Genetic Prompt Search for Efficient Few-shot Learning"](https://arxiv.org/abs/2210.17041) (EMNLP 2022).

Instead of English datasets (SuperGLUE, ANLI) like the original paper, this project:
- Uses a **genetic algorithm** to **automatically search for strong prompts** for Vietnamese NLP
- Main task: **sentiment analysis** on UIT-VSFC (positive / negative / neutral)
- Scoring model: `bigscience/mt0-large` (~1.2B params, multilingual, runs on CPU/MPS)
- Prompt generator for mutations: `vinai/bartpho-syllable` (~140M params, Vietnamese-native)
- Demo: local **Gradio** web app

This is an **Evolutionary Algorithms** course project (team of three).

---

## Repository layout

```
gps_research/
├── CLAUDE.md                  ← this file
├── README.md                  ← how to run
├── app.py                     ← Gradio web demo (Data & UI)
├── requirements.txt
│
├── core/                      ← Core GA + scoring
│   ├── __init__.py
│   ├── ga_engine.py           ← main GA loop (GeneticPromptSearch)
│   └── scorer.py              ← prompt scoring (PromptScorer)
│
├── mutation/                  ← Mutation strategies
│   ├── __init__.py            ← PromptMutator (unified interface)
│   ├── back_translation.py    ← back_translate()
│   ├── cloze.py               ← ClozeGenerator
│   └── sentence_cont.py       ← SentenceContinuation
│
├── data/
│   ├── prepare_data.py        ← download + sample UIT-VSFC
│   ├── vi_sentiment_dev.json  ← 32-example dev set (from prepare_data.py)
│   └── seed_prompts.json      ← 5 Vietnamese seed prompts G₀
│
├── models/                    ← not committed (download locally)
│   ├── mt0-large/
│   └── bartpho-syllable/
│
├── results/                   ← GPS outputs (auto-generated)
│   └── *.json
│
├── logs/                      ← per-generation logs
│
└── docs/
    ├── architecture.md
    ├── core_ga_engine_guide.md
    ├── mutation_strategies_guide.md
    ├── data_and_web_demo_guide.md
    └── api_reference.md
```

---

## System architecture

```
app.py (Gradio UI)
    │
    ├── GeneticPromptSearch.run()          ← core/ga_engine.py
    │       │
    │       ├── PromptScorer.score_all()   ← core/scorer.py
    │       │       └── mt0-large (accuracy on dev set)
    │       │
    │       └── PromptMutator.mutate()       ← mutation/__init__.py
    │               ├── back_translate()       (deep-translator, no GPU)
    │               ├── SentenceContinuation     (bartpho-syllable)
    │               └── ClozeGenerator           (bartpho-syllable)
    │
    └── data/vi_sentiment_dev.json         ← 32 examples, 3 balanced labels
```

**Data flow for one GA iteration:**
```
seed_prompts (list[str])
    → score_all()  → {prompt: accuracy}
    → top-K selection
    → mutate()     → new_candidates (list[str])
    → population = top_K + new_candidates
    → [repeat T times]
    → merge all top-K → rescore → best_prompts
```

---

## Module interfaces (important for integration)

### Core calls mutation via:
```python
# mutation/__init__.py
class PromptMutator:
    def mutate(self,
               prompts: list[str],
               strategy: str,        # "bt" | "sc" | "cloze" | "all"
               n_candidates: int = 10
               ) -> list[str]:       # new prompts, deduplicated
```

### Core exposes to the UI via:
```python
# core/ga_engine.py
class GeneticPromptSearch:
    def run(self,
            seed_prompts: list[str],
            dev_set: list[dict],     # [{"text": ..., "label": ...}]
            n_iter: int = 6,
            strategy: str = "sc",
            callback = None          # fn(gen, prompts, scores) — UI hook
            ) -> tuple[list[str], list[dict], dict]:
            # returns: best_prompts, history, final_scores
```

### `dev_set` format (from Data module):
```json
[
  {"text": "Sản phẩm rất tốt!", "label": "tích cực"},
  {"text": "Hàng kém chất lượng.", "label": "tiêu cực"},
  {"text": "Bình thường thôi.", "label": "trung lập"}
]
```

### `seed_prompts` format:
```json
[
  "Đánh giá sau đây: {{văn_bản}}\nCảm xúc của đánh giá này là gì? Trả lời: tích cực, tiêu cực, hoặc trung lập.",
  "Hãy phân tích cảm xúc của câu sau: {{văn_bản}}\nĐây là đánh giá tích cực, tiêu cực hay trung lập?"
]
```

**Required placeholder:** every template must contain `{{văn_bản}}` or `{{text}}`.  
The GA engine filters out prompts without a placeholder.

---

## Key design choices

| Decision | Rationale |
|---|---|
| `mt0-large` as scorer | Multilingual T0 (~1.2B), understands Vietnamese, fits ~16GB MacBook (~5GB RAM) |
| `bartpho-syllable` as generator | Vietnamese-native BART (~140M), no translation needed for SC/Cloze |
| `strategy="sc"` as default | Sentence Continuation reached 61.72% in the paper (best of three) |
| `top_p=0.9` when sampling | Matches the original paper |
| `n_iter=4` in demo | Enough to see convergence; ~15–20 min on CPU |
| UIT-VSFC | Standard Vietnamese sentiment dataset on Hugging Face |
| 32-example dev set | Matches “true few-shot” setup in the paper |
| `device="mps"` on Mac | Apple Silicon MPS is often ~3× faster than CPU |

---

## Common commands

```bash
pip install -r requirements.txt

# Download models (run once)
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
AutoTokenizer.from_pretrained('bigscience/mt0-large').save_pretrained('./models/mt0-large')
AutoModelForSeq2SeqLM.from_pretrained('bigscience/mt0-large').save_pretrained('./models/mt0-large')
AutoTokenizer.from_pretrained('vinai/bartpho-syllable').save_pretrained('./models/bartpho-syllable')
AutoModelForSeq2SeqLM.from_pretrained('vinai/bartpho-syllable').save_pretrained('./models/bartpho-syllable')
"

python data/prepare_data.py

python -m pytest tests/ -v

python app.py
```

---

## Out of scope for this project

- **No** fine-tuning of model weights (GPS is gradient-free)
- **No** soft prompts (hard/discrete text prompts only)
- **No** GPU required (CPU/MPS are fine)
- **No** datasets other than UIT-VSFC for the main task (consistent comparisons)
- **Do not** commit `models/` (~6GB)

---

## Team roles & modules

| Role | Module | Main files |
|---|---|---|
| Core GA | GA engine + scorer | `core/ga_engine.py`, `core/scorer.py` |
| Mutation | Mutation strategies | `mutation/*.py` |
| Data & UI | Data + web demo | `data/prepare_data.py`, `app.py` |

**Paper:** https://arxiv.org/abs/2210.17041  
**Original code:** https://github.com/hwxu20/GPS  
**Dataset (UIT-VSFC):** loaded via https://huggingface.co/datasets/ura-hcmut/UIT-VSFC (CSV mirror; same UIT-VSFC corpus)
