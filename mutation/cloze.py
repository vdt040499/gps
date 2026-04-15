# mutation/cloze.py
import core.warnings_config  # noqa: F401 — before torch/transformers

import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from core.device_utils import get_torch_device
from core.model_utils import resolve_model_id


# Words that must appear in generated output (whitelist — on-topic check)
_REQUIRED_WORDS = [
    "cảm xúc", "tích cực", "tiêu cực", "trung lập",
    "đánh giá", "nhận xét", "phân tích", "phân loại",
    "bình luận", "cảm nhận", "ý kiến", "phản hồi",
]

# Words never chosen as mask targets
_PROTECTED = {
    "cảm", "xúc", "tích", "cực", "tiêu", "trung", "lập",
    "đánh", "giá", "xác", "định", "phân", "tích", "loại",
    "nhận", "xét", "đọc", "chọn", "trả", "lời",
    "giảng", "viên", "sinh", "môn", "học",
}


class ClozeGenerator:
    """Multi-span masking: mask 2-4 separate spans (each 1-3 words) and
    let bartpho-syllable reconstruct the full sentence.

    Compared to the previous version (single word mask + beam search),
    this version:
    - Masks multiple spans for more structural changes
    - Uses sampling (top_p) instead of beam search for diversity
    - Produces prompts that differ meaningfully from the input
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

    def _multi_span_mask(self, words: list[str]) -> str | None:
        """Mask 1-2 spans of 1-2 words each, skipping protected words.

        Reduced from 2-4 spans to 1-2 to prevent excessive topic drift.
        Protected words (sentiment labels, task verbs) are never masked.
        """
        if len(words) < 4:
            return None

        # Only non-protected interior positions are maskable
        candidates = [
            i for i in range(1, len(words) - 1)
            if words[i].lower().rstrip(".,?!:/") not in _PROTECTED
        ]

        if len(candidates) < 1:
            return None

        n_spans = random.randint(1, min(2, len(candidates)))
        random.shuffle(candidates)
        span_starts = sorted(candidates[:n_spans * 2])

        masked_indices: set[int] = set()
        used = []

        for start in span_starts:
            if len(used) >= n_spans:
                break
            span_len = random.randint(1, min(2, len(words) - start - 1))
            span_range = set(range(start, min(start + span_len, len(words) - 1)))
            # skip if overlapping already masked or hits protected word
            if not span_range & masked_indices and not any(
                words[j].lower().rstrip(".,?!:/") in _PROTECTED for j in span_range
            ):
                masked_indices |= span_range
                used.append(start)

        if not masked_indices:
            return None

        return " ".join(
            self.tokenizer.mask_token if i in masked_indices else w
            for i, w in enumerate(words)
        )

    def generate(self, prompt: str, n_candidates: int = 5) -> list[str]:
        instruction, suffix = self._split_placeholder(prompt)
        if not instruction:
            return []

        vi_words = instruction.split()
        if len(vi_words) < 5:
            return []

        max_words = len(vi_words) + 4  # allow moderate expansion

        results: list[str] = []
        max_attempts = n_candidates * 4

        for _ in range(max_attempts):
            masked_text = self._multi_span_mask(vi_words)
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
                    max_new_tokens=60,
                    do_sample=True,       # sampling instead of beam search
                    top_p=0.9,
                    temperature=0.9,
                )

            new_vi = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

            if not new_vi or new_vi == instruction:
                continue

            new_vi = self._truncate(new_vi, max_words)

            new_prompt = (new_vi.strip() + " " + suffix).strip()

            lower_vi = new_vi.lower()
            is_relevant = any(w in lower_vi for w in _REQUIRED_WORDS)
            if (
                new_prompt != prompt
                and "{{văn_bản}}" in new_prompt
                and new_prompt not in results
                and is_relevant
            ):
                results.append(new_prompt)

            if len(results) >= n_candidates:
                break

        return results
