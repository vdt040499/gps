# core/ga_engine.py
from __future__ import annotations

import logging
from typing import Any, Callable

from core.scorer import PromptScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

Callback = Callable[[int, list[str], dict[str, float], dict[str, float]], None]


class GeneticPromptSearch:
    """Genetic Prompt Search loop: score → top-k → mutate → repeat."""

    def __init__(self, scorer: PromptScorer, mutator: Any, top_k: int = 5):
        self.scorer = scorer
        self.mutator = mutator
        self.top_k = top_k

    def run(
        self,
        seed_prompts: list[str],
        dev_set: list[dict],
        n_iter: int = 6,
        strategy: str = "sc",
        callback: Callback | None = None,
    ):
        """
        Args:
            seed_prompts: Initial population G0.
            dev_set: Labeled examples, e.g. [{"text": str, "label": str}, ...].
            n_iter: Number of generations.
            strategy: Mutation strategy key passed to the mutator (bt / sc / cloze / all).
            callback: Optional ``fn(gen, prompts, scores, all_scores)`` after each generation.
        """
        from tqdm import tqdm

        population = list(seed_prompts)
        history: list[dict] = []

        for t in tqdm(range(n_iter), desc="GPS — tổng tiến trình", unit="gen"):
            gen_label = f"Gen {t+1}/{n_iter}"
            log.info(f"=== Generation {t} — {len(population)} prompts ===")

            scores = self.scorer.score_all(
                population, dev_set,
                desc=f"[{gen_label}] Scoring {len(population)} prompts",
            )

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_k_prompts = [p for p, _ in ranked[: self.top_k]]
            top_k_scores = {p: scores[p] for p in top_k_prompts}
            history.append({"gen": t, "prompts": top_k_prompts, "scores": top_k_scores})

            log.info(
                "Top-1 gen %s: %s... score=%.3f",
                t,
                ranked[0][0][:60],
                ranked[0][1],
            )

            if callback:
                callback(gen=t, prompts=top_k_prompts, scores=top_k_scores, all_scores=scores)

            if t < n_iter - 1:
                new_candidates = self.mutator.mutate(
                    top_k_prompts,
                    strategy=strategy,
                    n_candidates=20,
                )
                new_candidates = self._filter_candidates(new_candidates, population)
                population = top_k_prompts + new_candidates

        all_candidates: list[str] = []
        for h in history:
            all_candidates.extend(h["prompts"])
        all_candidates = list(dict.fromkeys(all_candidates))

        final_scores = self.scorer.score_all(
            all_candidates, dev_set,
            desc=f"[Final] Rescoring {len(all_candidates)} prompts",
        )

        final_ranked = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        best_prompts = [p for p, _ in final_ranked[: self.top_k]]

        return best_prompts, history, final_scores

    def _filter_candidates(self, candidates: list[str], existing: list[str]) -> list[str]:
        """Keep prompts with a text placeholder and drop duplicates."""
        seen = set(existing)
        out: list[str] = []
        for c in candidates:
            has_ph = "{{văn_bản}}" in c or "{{text}}" in c
            if has_ph and c not in seen:
                seen.add(c)
                out.append(c)
        return out
