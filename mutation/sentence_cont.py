# mutation/sentence_cont.py
import core.warnings_config  # noqa: F401 — before torch/transformers

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from core.device_utils import get_torch_device
from core.model_utils import resolve_model_id


# ─────────────────────────────────────────────────────────────────────────────
# Paraphrase templates — inspired by DINO (Schick & Schütze, 2021)
# as described in GPS paper: "Write two sentences that mean the same thing."
# Adapted to Vietnamese for this project.
# ─────────────────────────────────────────────────────────────────────────────
_PARAPHRASE_TEMPLATES = [
    "Viết hai câu có cùng ý nghĩa.\nCâu 1: {prompt}\nCâu 2:",
    "Write two sentences that mean the same thing.\nSentence 1: {prompt}\nSentence 2:",
    "Diễn đạt lại câu sau với ý nghĩa tương tự.\nCâu gốc: {prompt}\nCâu mới:",
]

# ─────────────────────────────────────────────────────────────────────────────
# Whitelist — generated instruction MUST contain ≥1 of these to be kept.
# This ensures the generated paraphrase stays on-topic for sentiment analysis.
# ─────────────────────────────────────────────────────────────────────────────
_REQUIRED_WORDS = [
    "cảm xúc", "tích cực", "tiêu cực", "trung lập",
    "đánh giá", "nhận xét", "phân tích", "phân loại",
    "bình luận", "cảm nhận", "ý kiến", "phản hồi",
    "sentiment", "positive", "negative", "neutral",
]


class SentenceContinuation:
    """Generate prompt paraphrases via Sentence Continuation (SC).

    Follows the GPS paper (Xu et al., EMNLP 2022):
        Use a pretrained language model with the template
        "Write two sentences that mean the same thing.
         Sentence 1: <original prompt>. Sentence 2:"
        to generate continuations as new prompt candidates.

    This project uses mt0-large (already loaded for scoring) as the
    generation model, since it understands Vietnamese instructions well
    (multilingual T0, 1.2B params). The paper originally used GPT2-XL
    or T5LM-XXL.

    Key differences from the masking/cloze approach:
        - Model generates a COMPLETELY NEW sentence (free generation)
        - Template explicitly asks for semantic equivalence
        - Much higher diversity than infilling masked spans
    """

    def __init__(self, model_path=None, *, model=None, tokenizer=None, device=None):
        """Initialise with a fresh model load, or reuse an existing one.

        Args:
            model_path: Optional explicit model path / Hub id.
            model, tokenizer, device: If all three are provided, reuse them
                instead of loading a new copy (saves ~5 GB when the scorer's
                mt0-large is shared).
        """
        if model is not None and tokenizer is not None and device is not None:
            # Reuse pre-loaded model (e.g. from PromptScorer)
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
        else:
            model_id = resolve_model_id(
                model_path,
                local_relative="models/mt0-large",
                hub_id="bigscience/mt0-large",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, torch_dtype=torch.float32,
            )
            self.model.eval()
            self.device = get_torch_device()
            self.model.to(self.device)

    @classmethod
    def from_scorer(cls, scorer) -> "SentenceContinuation":
        """Create an SC instance that shares the scorer's mt0-large model.

        This avoids loading the ~5 GB model a second time.

        Args:
            scorer: A PromptScorer instance (core.scorer.PromptScorer).
        """
        return cls(model=scorer.model, tokenizer=scorer.tokenizer, device=scorer.device)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _split_placeholder(prompt: str) -> tuple[str, str]:
        """Separate instruction from placeholder + tail.

        Example:
            "Đánh giá cảm xúc: {{văn_bản}}\\nTrả lời ..."
            → ("Đánh giá cảm xúc:", "{{văn_bản}}\\nTrả lời ...")
        """
        for ph in ("{{văn_bản}}", "{{text}}"):
            if ph in prompt:
                parts = prompt.split(ph, 1)
                return parts[0].strip(), ph + parts[1]
        return prompt.strip(), ""

    @staticmethod
    def _is_relevant(text: str) -> bool:
        """Return True only if text contains ≥1 required domain/task word."""
        lower = text.lower()
        return any(w in lower for w in _REQUIRED_WORDS)

    @staticmethod
    def _truncate(text: str, max_words: int) -> str:
        words = text.split()
        return " ".join(words[:max_words]) if len(words) > max_words else text

    @staticmethod
    def _clean_generated(text: str) -> str:
        """Post-process generated text: strip quotes, numbering, etc."""
        text = text.strip()
        # Remove leading "Câu 2:" or "Sentence 2:" artifacts
        for prefix in ("Câu 2:", "Câu 2 :", "Sentence 2:", "Sentence 2 :"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        # Remove surrounding quotes if model added them
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        return text

    # ── main ─────────────────────────────────────────────────────────────────

    def generate(self, prompt: str, n_candidates: int = 5) -> list[str]:
        """Generate n_candidates paraphrases of the instruction part of prompt.

        Uses multiple paraphrase templates and sampling to produce diverse
        candidates. Each candidate is validated for domain relevance and
        placeholder presence.
        """
        instruction, suffix = self._split_placeholder(prompt)
        if not instruction:
            return []

        words = instruction.split()
        if len(words) < 3:
            return []

        max_words = len(words) + 6  # allow slightly longer paraphrases
        results: list[str] = []
        max_attempts = n_candidates * 6  # extra budget for relevance rejections

        template_idx = 0
        for _ in range(max_attempts):
            # Rotate through templates for diversity
            template = _PARAPHRASE_TEMPLATES[template_idx % len(_PARAPHRASE_TEMPLATES)]
            template_idx += 1

            # Build the SC input: template with original instruction
            sc_input = template.format(prompt=instruction)

            inputs = self.tokenizer(
                sc_input,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                )

            generated = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            generated = self._clean_generated(generated)

            if not generated or len(generated) < 6:
                continue

            generated = self._truncate(generated, max_words)
            new_prompt = (generated.strip() + " " + suffix).strip()

            # ── Validity checks ───────────────────────────────────────────
            if new_prompt == prompt:
                continue
            if "{{văn_bản}}" not in new_prompt and "{{text}}" not in new_prompt:
                # The generated text won't have placeholder — append suffix
                # which already contains the placeholder
                if not suffix:
                    continue
            if new_prompt in results:
                continue
            # WHITELIST: must contain at least one domain-relevant word
            if not self._is_relevant(generated):
                continue
            # ─────────────────────────────────────────────────────────────

            results.append(new_prompt)
            if len(results) >= n_candidates:
                break

        return results
