from __future__ import annotations

# mutation/__init__.py — single entry point for the GA engine
from mutation.back_translation import back_translate
from mutation.sentence_cont import SentenceContinuation
from mutation.cloze import ClozeGenerator


class PromptMutator:
    """Dispatches mutation strategies for GPS."""

    def __init__(self) -> None:
        self._sc: SentenceContinuation | None = None
        self._cloze: ClozeGenerator | None = None

    def mutate(
        self,
        prompts: list[str],
        strategy: str,
        n_candidates: int = 10,
    ) -> list[str]:
        if not prompts:
            return []

        per_prompt = max(1, n_candidates // len(prompts))
        results: list[str] = []

        for p in prompts:
            if strategy == "bt":
                n_var = min(per_prompt, 6)
                results.extend(back_translate(p, n_variants=n_var))
            elif strategy == "sc":
                if self._sc is None:
                    self._sc = SentenceContinuation()
                results.extend(self._sc.generate(p, n_candidates=per_prompt))
            elif strategy == "cloze":
                if self._cloze is None:
                    self._cloze = ClozeGenerator()
                results.extend(self._cloze.generate(p, n_candidates=per_prompt))
            elif strategy == "all":
                k = max(1, per_prompt // 2)
                results.extend(back_translate(p, n_variants=k))
                if self._sc is None:
                    self._sc = SentenceContinuation()
                results.extend(self._sc.generate(p, n_candidates=k))
            else:
                raise ValueError(f"Unknown strategy: {strategy!r}")

        return list(dict.fromkeys(results))
