# mutation/sentence_cont.py
import core.warnings_config  # noqa: F401 — before torch/transformers

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from core.model_utils import resolve_model_id


class SentenceContinuation:
    """Paraphrase prompts with flan-t5-base (paper-style sentence continuation)."""

    def __init__(self, model_path=None):
        model_id = resolve_model_id(
            model_path,
            local_relative="models/flan-t5-base",
            hub_id="google/flan-t5-base",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        self.model.eval()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, prompt: str, n_candidates: int = 5) -> list[str]:
        # English instruction — flan-t5 is stronger on English than Vietnamese
        template = (
            f"Write a sentence that means the same as the following instruction "
            f"but with different wording:\n{prompt}\nRewrite:"
        )
        inputs = self.tokenizer(
            template,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=n_candidates,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )
        results: list[str] = []
        for out in outputs:
            text = self.tokenizer.decode(out, skip_special_tokens=True).strip()
            if text and text != prompt:
                results.append(text)
        return results
