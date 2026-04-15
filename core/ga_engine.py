# core/ga_engine.py
from __future__ import annotations

import logging
from typing import Any, Callable

from core.scorer import PromptScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

Callback = Callable[[int, list[str], dict[str, dict], dict[str, dict]], None]


class GeneticPromptSearch:
    """Genetic Prompt Search loop: score → top-k → mutate → repeat.

    Selection is based on the **combined** score (α × accuracy + (1−α) × naturalness)
    so that the GA favours prompts that are both accurate AND natural-sounding.
    """

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
        alpha: float = 0.7,
        callback: Callback | None = None,
    ):
        """
        Args:
            seed_prompts: Initial population G0.
            dev_set: Labeled examples, e.g. [{"text": str, "label": str}, ...].
            n_iter: Number of generations.
            strategy: Mutation strategy key passed to the mutator (bt / sc / cloze / all).
            alpha: Weight for accuracy in combined score (0–1).
                   combined = α * accuracy + (1 − α) * naturalness
            callback: Optional ``fn(gen, prompts, scores, all_scores)`` after each generation.
                      scores and all_scores are dicts of
                      {prompt: {"accuracy": float, "naturalness": float, "combined": float}}.
        """
        from tqdm import tqdm

        population = list(seed_prompts)
        history: list[dict] = []

        for t in tqdm(range(n_iter), desc="GPS — tổng tiến trình", unit="gen"):
            gen_label = f"Gen {t+1}/{n_iter}"
            log.info(f"=== Generation {t} — {len(population)} prompts ===")

            scores = self.scorer.score_all(
                population, dev_set,
                alpha=alpha,
                desc=f"[{gen_label}] Scoring {len(population)} prompts",
            )

            # Rank by combined score (accuracy + naturalness)
            ranked = sorted(
                scores.items(),
                key=lambda x: x[1]["combined"],
                reverse=True,
            )
            top_k_prompts = [p for p, _ in ranked[: self.top_k]]
            top_k_scores = {p: scores[p] for p in top_k_prompts}
            history.append({"gen": t, "prompts": top_k_prompts, "scores": top_k_scores})

            best = ranked[0]
            log.info(
                "Top-1 gen %s: %s... acc=%.3f nat=%.3f combined=%.3f",
                t,
                best[0][:60],
                best[1]["accuracy"],
                best[1]["naturalness"],
                best[1]["combined"],
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
            alpha=alpha,
            desc=f"[Final] Rescoring {len(all_candidates)} prompts",
        )

        final_ranked = sorted(
            final_scores.items(),
            key=lambda x: x[1]["combined"],
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
