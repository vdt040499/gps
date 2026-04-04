# mutation/cloze.py
import core.warnings_config  # noqa: F401 — before torch/transformers

import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from core.device_utils import get_torch_device
from core.model_utils import resolve_model_id


class ClozeGenerator:
    """Mask one word in the instruction and let bartpho-syllable fill the blank.

    bartpho outputs the full reconstructed sentence, so we just take
    the output directly (truncated to original length).
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

    @staticmethod
    def _truncate(text: str, max_words: int) -> str:
        """Truncate to roughly the same word count as original."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])

    def generate(self, prompt: str, n_candidates: int = 5) -> list[str]:
        instruction, suffix = self._split_placeholder(prompt)
        if not instruction:
            return []

        vi_words = instruction.split()
        if len(vi_words) < 3:
            return []

        max_words = len(vi_words) + 2  # allow slight expansion

        results: list[str] = []
        max_attempts = n_candidates * 3

        for _ in range(max_attempts):
            idx = random.randint(1, len(vi_words) - 2)
            masked_words = vi_words[:idx] + [self.tokenizer.mask_token] + vi_words[idx + 1:]
            masked_text = " ".join(masked_words)

            inputs = self.tokenizer(
                masked_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=40, num_beams=4)

            new_vi = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

            if not new_vi or new_vi == instruction:
                continue

            new_vi = self._truncate(new_vi, max_words)

            new_prompt = (new_vi.strip() + " " + suffix).strip()

            if (
                new_prompt != prompt
                and "{{văn_bản}}" in new_prompt
                and new_prompt not in results
            ):
                results.append(new_prompt)

            if len(results) >= n_candidates:
                break

        return results
