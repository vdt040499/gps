# data/prepare_data.py — run once to build the dev split and seed prompts
from pathlib import Path

from datasets import load_dataset
import json
import random

# CSV mirror of UIT-VSFC (same corpus as uitnlp/vietnamese_students_feedback).
# Use this id: `uitnlp/UIT-VSFC` does not exist; `uitnlp/vietnamese_students_feedback`
# relies on a deprecated dataset script rejected by recent `datasets` versions.
HF_DATASET_ID = "ura-hcmut/UIT-VSFC"

_LABEL_EN_TO_VI = {
    "negative": "tiêu cực",
    "neutral": "trung lập",
    "positive": "tích cực",
}


def prepare_vsfc(n=32, seed=42):
    """UIT-VSFC: Vietnamese Students' Feedback Corpus (sentiment)."""
    random.seed(seed)
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(HF_DATASET_ID)

    examples = []
    for split in ["train", "validation"]:
        for ex in ds[split]:
            text = (ex["text"] or "").strip()
            if not text:
                continue
            examples.append({
                "text": text,
                "label": _LABEL_EN_TO_VI[ex["label"]],
            })

    per_class = n // 3
    result = []
    for label in ["tích cực", "tiêu cực", "trung lập"]:
        pool = [e for e in examples if e["label"] == label]
        k = min(per_class, len(pool))
        result.extend(random.sample(pool, k))

    random.shuffle(result)
    out = data_dir / "vi_sentiment_dev.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(result)} examples to {out}")
    return result

SEED_PROMPTS = [
    "Đánh giá sau đây: {{văn_bản}}\nCảm xúc của đánh giá này là gì? Trả lời: tích cực, tiêu cực, hoặc trung lập.",
    "Hãy phân tích cảm xúc của câu sau: {{văn_bản}}\nĐây là đánh giá tích cực, tiêu cực hay trung lập?",
    "Cho đoạn văn: {{văn_bản}}\nNgười viết có cảm xúc như thế nào? Chọn: tích cực / tiêu cực / trung lập",
    "Đọc đánh giá sau và xác định cảm xúc: {{văn_bản}}\nCảm xúc:",
    "{{văn_bản}}\nDựa vào nội dung trên, đây là nhận xét tích cực, tiêu cực hay trung lập?",
]

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    prepare_vsfc()
    with open(data_dir / "seed_prompts.json", "w", encoding="utf-8") as f:
        json.dump(SEED_PROMPTS, f, ensure_ascii=False, indent=2)
    print("Seed prompts saved!")