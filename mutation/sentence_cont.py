# mutation/sentence_cont.py
import core.warnings_config  # noqa: F401 — before torch/transformers

import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from core.device_utils import get_torch_device
from core.model_utils import resolve_model_id


class SentenceContinuation:
    """Paraphrase prompts by masking multiple words and letting bartpho reconstruct.

    bartpho-syllable is a Vietnamese BART model. It outputs the full
    reconstructed sentence (not just fills), so parsing is straightforward.
    """

    def __init__(self, model_path=None):
        model_id = resolve_model_id(
            model_path,
            local_relative="models/bartpho-syllable",
            hub_id="vinai/bartpho-syllable",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        self.model.eval()
        self.device = get_torch_device()
        self.model.to(self.device)

    @staticmethod
    def _split_placeholder(prompt: str) -> tuple[str, str]:
        """Separate instruction from placeholder + tail."""
        for ph in ("{{văn_bản}}", "{{text}}"):
            if ph in prompt:
                parts = prompt.split(ph, 1)
                return parts[0].strip(), ph + parts[1]
        return prompt.strip(), ""

    def _mask_words(self, words: list[str], n_masks: int = 2) -> str | None:
        """Replace n_masks random interior words with <mask>."""
        if len(words) < 4:
            return None
        interior = list(range(1, len(words) - 1))
        random.shuffle(interior)
        mask_indices = set(interior[:n_masks])

        result = []
        for i, w in enumerate(words):
            result.append(self.tokenizer.mask_token if i in mask_indices else w)
        return " ".join(result)

    @staticmethod
    def _truncate(text: str, max_words: int) -> str:
        """Truncate to roughly the same word count as original instruction."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])

    def generate(self, prompt: str, n_candidates: int = 5) -> list[str]:
        instruction, suffix = self._split_placeholder(prompt)
        if not instruction:
            return []

        words = instruction.split()
        if len(words) < 4:
            return []

        max_words = len(words) + 3  # allow slight expansion

        results: list[str] = []
        max_attempts = n_candidates * 3

        for _ in range(max_attempts):
            n_masks = random.randint(2, min(3, len(words) - 2))
            masked_text = self._mask_words(words, n_masks)
            if not masked_text:
                continue

            inputs = self.tokenizer(
                masked_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                )

            vi_text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            if not vi_text or len(vi_text) < 10:
                continue

            vi_text = self._truncate(vi_text, max_words)

            new_prompt = (vi_text.strip() + " " + suffix).strip()
            if (
                new_prompt != prompt
                and "{{văn_bản}}" in new_prompt
                and new_prompt not in results
            ):
                results.append(new_prompt)

            if len(results) >= n_candidates:
                break

        return results
