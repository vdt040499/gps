# mutation/back_translation.py
import time

from deep_translator import GoogleTranslator

PIVOT_LANGS = ["en", "fr", "de", "zh-CN", "ja", "ko"]


def back_translate(prompt: str, src: str = "vi", n_variants: int = 3) -> list[str]:
    """
    Paraphrase via pivot languages: Vietnamese → pivot → Vietnamese.

    Args:
        prompt: Source text (typically a Vietnamese prompt template).
        src: Source language code for the translator.
        n_variants: How many pivot languages to use (capped by ``PIVOT_LANGS``).
    """
    results: list[str] = []
    for pivot in PIVOT_LANGS[:n_variants]:
        try:
            mid = GoogleTranslator(source=src, target=pivot).translate(prompt)
            time.sleep(0.3)  # reduce rate-limit issues
            back = GoogleTranslator(source=pivot, target=src).translate(mid)
            if back and back != prompt:
                results.append(back)
        except Exception as e:
            print(f"BT error ({pivot}): {e}")
    return results
