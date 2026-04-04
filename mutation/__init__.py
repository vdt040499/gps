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

    def _get_sc(self) -> SentenceContinuation:
        if self._sc is None:
            self._sc = SentenceContinuation()
        return self._sc

    def _get_cloze(self) -> ClozeGenerator:
        if self._cloze is None:
            self._cloze = ClozeGenerator()
        return self._cloze

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
                results.extend(back_translate(p, n_variants=min(per_prompt, 6)))

            elif strategy == "sc":
                results.extend(self._get_sc().generate(p, n_candidates=per_prompt))

            elif strategy == "cloze":
                results.extend(self._get_cloze().generate(p, n_candidates=per_prompt))

            elif strategy == "all":
                # Distribute across all three strategies
                k = max(1, per_prompt // 3)
                results.extend(back_translate(p, n_variants=max(k, 1)))
                results.extend(self._get_sc().generate(p, n_candidates=max(k, 1)))
                results.extend(self._get_cloze().generate(p, n_candidates=max(k, 1)))

            else:
                raise ValueError(f"Unknown strategy: {strategy!r}")

        # Deduplicate, excluding original prompts
        seen = set(prompts)
        deduped: list[str] = []
        for r in results:
            if r and r not in seen:
                seen.add(r)
                deduped.append(r)
        return deduped
