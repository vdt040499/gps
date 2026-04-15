# core/naturalness.py
"""Naturalness scoring via perplexity (PPL).

Lower perplexity means the text reads more naturally to the model.
We convert PPL into a 0–1 score: score = 1 / (1 + log(PPL)).
"""
from __future__ import annotations

import math
import torch


class NaturalnessScorer:
    """Evaluate how natural / fluent a prompt reads.

    Reuses an already-loaded seq2seq model (e.g. mt0-large) so there is
    **no additional GPU/RAM cost**.

    The score is derived from perplexity:
        PPL  = exp(cross-entropy loss on the instruction text)
        score = 1 / (1 + ln(PPL))          ∈ (0, 1]
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @staticmethod
    def _extract_instruction(prompt: str) -> str:
        """Return the instruction part (before the placeholder)."""
        for ph in ("{{văn_bản}}", "{{text}}"):
            if ph in prompt:
                return prompt.split(ph, 1)[0].strip()
        return prompt.strip()

    def score(self, prompt: str) -> float:
        """Return a naturalness score ∈ (0, 1].

        Higher is better (more natural).
        """
        instruction = self._extract_instruction(prompt)
        if not instruction or len(instruction) < 5:
            return 0.0

        # Encode instruction as both input and target (teacher-forcing)
        inputs = self.tokenizer(
            instruction,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        labels = self.tokenizer(
            instruction,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).input_ids.to(self.device)

        with torch.no_grad():
            loss = self.model(**inputs, labels=labels).loss.item()

        ppl = math.exp(min(loss, 20.0))  # clamp to avoid overflow
        score = 1.0 / (1.0 + math.log(max(ppl, 1.0)))
        return round(score, 4)

    def score_batch(self, prompts: list[str]) -> dict[str, float]:
        """Score every prompt in a list."""
        return {p: self.score(p) for p in prompts}
