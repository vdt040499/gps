# core/scorer.py
import core.warnings_config  # noqa: F401 — load before torch/transformers

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from core.model_utils import resolve_model_id


class PromptScorer:
    """Ranking-based prompt scoring with flan-t5 (negative log-loss per label)."""

    def __init__(self, model_path=None):
        model_id = resolve_model_id(
            model_path,
            local_relative="models/flan-t5-large",
            hub_id="google/flan-t5-large",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        self.model.eval()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Scorer loaded on {self.device}")

    def _render(self, template: str, text: str) -> str:
        out = template.replace("{{văn_bản}}", text)
        return out.replace("{{text}}", text)

    def _label_scores(self, prompt_template: str, text: str, labels: list[str]) -> dict[str, float]:
        """Return higher-is-better scores (negative CE loss) for each label string."""
        rendered = self._render(prompt_template, text)
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        scores: dict[str, float] = {}
        for label in labels:
            label_ids = self.tokenizer(label, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                loss = self.model(**inputs, labels=label_ids).loss
            scores[label] = -loss.item()
        return scores

    def predict_label(self, prompt_template: str, text: str, labels: list[str]) -> str:
        """Pick the label with highest model score (same rule as classification)."""
        scores = self._label_scores(prompt_template, text, labels)
        return max(scores, key=scores.get)

    def score_one(self, prompt_template, examples, labels):
        """Accuracy of one prompt over a list of labeled examples."""
        correct = 0
        for ex in examples:
            scores = self._label_scores(prompt_template, ex["text"], labels)
            predicted = max(scores, key=scores.get)
            if predicted == ex["label"]:
                correct += 1
        return correct / len(examples)

    def score_all(self, prompts, dev_set, desc="Scoring prompts"):
        """Score every prompt in the population on the dev set."""
        from tqdm import tqdm

        labels = list({ex["label"] for ex in dev_set})
        results: dict[str, float] = {}
        for p in tqdm(prompts, desc=desc, leave=False):
            results[p] = self.score_one(p, dev_set, labels)
        return results
