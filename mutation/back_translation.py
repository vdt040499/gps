# mutation/back_translation.py
import time

from deep_translator import GoogleTranslator

PIVOT_LANGS = ["en", "fr", "de", "zh-CN", "ja", "ko"]

_SENTINEL = "XXXPLACEHOLDERXXX"


def back_translate(prompt: str, src: str = "vi", n_variants: int = 3) -> list[str]:
    """
    Paraphrase via pivot languages: Vietnamese → pivot → Vietnamese.

    Protects ``{{văn_bản}}`` / ``{{text}}`` by replacing with a sentinel
    before translation, then restoring after.
    """
    # Protect placeholders
    safe = prompt.replace("{{văn_bản}}", _SENTINEL).replace("{{text}}", _SENTINEL)

    results: list[str] = []
    for pivot in PIVOT_LANGS[:n_variants]:
        try:
            mid = GoogleTranslator(source=src, target=pivot).translate(safe)
            time.sleep(0.5)
            back = GoogleTranslator(source=pivot, target=src).translate(mid)
            time.sleep(0.5)

            if not back:
                continue

            # Restore placeholder
            back = back.replace(_SENTINEL, "{{văn_bản}}")

            if back != prompt and "{{văn_bản}}" in back:
                results.append(back)

        except Exception as e:
            print(f"[BT] Error with pivot {pivot}: {e}")
    return results
