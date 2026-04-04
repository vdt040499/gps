# mutation/cloze.py
import core.warnings_config  # noqa: F401 — before torch/transformers

import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from core.model_utils import resolve_model_id


class ClozeGenerator:
    """Mask one token in the instruction and let T5 fill the blank."""

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
        # Split instruction (before "{{") from the placeholder tail
        if "{{" in prompt:
            instruction, rest = prompt.split("{{", 1)
        else:
            instruction, rest = prompt, ""

        words = instruction.strip().split()
        if len(words) < 3:
            return []

        results: list[str] = []
        for _ in range(n_candidates * 2):
            # Mask one random interior word
            idx = random.randint(1, len(words) - 2)
            masked = words[:idx] + ["<extra_id_0>"] + words[idx + 1 :]
            masked_text = " ".join(masked)
            if rest:
                masked_text += " {{" + rest

            inputs = self.tokenizer(
                masked_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=10, num_beams=4)
            filled = self.tokenizer.decode(out[0], skip_special_tokens=True)
            new_prompt = masked_text.replace("<extra_id_0>", filled.strip())
            if new_prompt != prompt and "{{" in new_prompt:
                results.append(new_prompt)
            if len(results) >= n_candidates:
                break
        return results
