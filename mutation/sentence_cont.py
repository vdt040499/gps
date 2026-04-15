# mutation/sentence_cont.py
import core.warnings_config  # noqa: F401 — before torch/transformers

import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from core.device_utils import get_torch_device
from core.model_utils import resolve_model_id


# ─────────────────────────────────────────────────────────────────────────────
# Domain anchors — words NEVER masked (keep sentiment/task context intact)
# ─────────────────────────────────────────────────────────────────────────────
_PROTECTED = {
    # Task verbs
    "đánh", "giá", "xác", "định", "phân", "tích", "phân", "loại",
    "nhận", "xét", "đọc", "câu", "chọn", "trả", "lời",
    # Sentiment keywords — critical anchors
    "cảm", "xúc", "tích", "cực", "tiêu", "trung", "lập",
    "tốt", "xấu", "hài", "lòng", "hài lòng",
    # Education domain (when present in seed)
    "giảng", "viên", "sinh", "môn", "học", "lớp",
    "giáo", "thầy", "cô", "bài", "khóa",
}

# ─────────────────────────────────────────────────────────────────────────────
# Whitelist — generated instruction MUST contain ≥1 of these to be kept.
# This is much more robust than a blacklist approach.
# ─────────────────────────────────────────────────────────────────────────────
_REQUIRED_WORDS = [
    "cảm xúc", "tích cực", "tiêu cực", "trung lập",
    "đánh giá", "nhận xét", "phân tích", "phân loại",
    "bình luận", "cảm nhận", "ý kiến", "phản hồi",
]


class SentenceContinuation:
    """Paraphrase prompts by masking ~20-35% of non-protected words.

    Key design:
    - Protected words (sentiment labels, task verbs) are never masked so
      bartpho stays on-topic.
    - Generated prompts must contain ≥1 whitelist word (domain-relevance
      whitelist check) to be accepted — much more robust than a blacklist.
    - Mask ratio kept moderate (20-35%) so structure is preserved.
    """

    def __init__(self, model_path=None):
        model_id = resolve_model_id(
            model_path,
            local_relative="models/bartpho-syllable",
            hub_id="vinai/bartpho-syllable",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, torch_dtype=torch.float32,
        )
        self.model.eval()
        self.device = get_torch_device()
        self.model.to(self.device)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _split_placeholder(prompt: str) -> tuple[str, str]:
        for ph in ("{{văn_bản}}", "{{text}}"):
            if ph in prompt:
                parts = prompt.split(ph, 1)
                return parts[0].strip(), ph + parts[1]
        return prompt.strip(), ""

    def _mask_words(self, words: list[str], mask_ratio: float) -> str | None:
        """Mask mask_ratio of words that are NOT in _PROTECTED."""
        if len(words) < 4:
            return None

        # Candidate positions: interior, word not protected
        candidates = [
            i for i in range(1, len(words) - 1)
            if words[i].lower().rstrip(".,?!:/") not in _PROTECTED
        ]

        if len(candidates) < 1:
            return None  # nothing safe to mask → skip

        n_masks = max(1, int(len(candidates) * mask_ratio))
        n_masks = min(n_masks, len(candidates))

        random.shuffle(candidates)
        mask_set = set(candidates[:n_masks])

        return " ".join(
            self.tokenizer.mask_token if i in mask_set else w
            for i, w in enumerate(words)
        )

    @staticmethod
    def _is_relevant(text: str) -> bool:
        """Return True only if text contains ≥1 required domain/task word."""
        lower = text.lower()
        return any(w in lower for w in _REQUIRED_WORDS)

    @staticmethod
    def _truncate(text: str, max_words: int) -> str:
        words = text.split()
        return " ".join(words[:max_words]) if len(words) > max_words else text

    # ── main ─────────────────────────────────────────────────────────────────

    def generate(self, prompt: str, n_candidates: int = 5) -> list[str]:
        instruction, suffix = self._split_placeholder(prompt)
        if not instruction:
            return []

        words = instruction.split()
        if len(words) < 4:
            return []

        max_words = len(words) + 4
        results: list[str] = []
        max_attempts = n_candidates * 6  # extra budget for whitelist rejections

        for _ in range(max_attempts):
            # Moderate masking: 20-35% — keeps enough structure for on-topic output
            mask_ratio = random.uniform(0.20, 0.35)

            masked_text = self._mask_words(words, mask_ratio)
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

            if not vi_text or len(vi_text) < 8:
                continue

            vi_text = self._truncate(vi_text, max_words)
            new_prompt = (vi_text.strip() + " " + suffix).strip()

            # ── Validity checks ───────────────────────────────────────────
            if new_prompt == prompt:
                continue
            if "{{văn_bản}}" not in new_prompt:
                continue
            if new_prompt in results:
                continue
            # WHITELIST: must contain at least one domain-relevant word
            if not self._is_relevant(vi_text):
                continue
            # ─────────────────────────────────────────────────────────────

            results.append(new_prompt)
            if len(results) >= n_candidates:
                break

        return results
