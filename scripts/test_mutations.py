"""Quick smoke test for SC and Cloze mutations with vit5-base."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TEST_PROMPT = "Đánh giá sau đây: {{văn_bản}}\nCảm xúc là gì?"


def test_sc():
    print("\n" + "=" * 50)
    print("Testing SentenceContinuation")
    print("=" * 50)
    from mutation.sentence_cont import SentenceContinuation

    sc = SentenceContinuation()
    results = sc.generate(TEST_PROMPT, 3)
    print(f"Input:  {TEST_PROMPT!r}")
    print(f"Output ({len(results)} candidates):")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r}")
    return results


def test_cloze():
    print("\n" + "=" * 50)
    print("Testing ClozeGenerator")
    print("=" * 50)
    from mutation.cloze import ClozeGenerator

    cg = ClozeGenerator()
    results = cg.generate(TEST_PROMPT, 3)
    print(f"Input:  {TEST_PROMPT!r}")
    print(f"Output ({len(results)} candidates):")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r}")
    return results


if __name__ == "__main__":
    sc_results = test_sc()
    cloze_results = test_cloze()

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"SC:    {len(sc_results)} candidates")
    print(f"Cloze: {len(cloze_results)} candidates")

    ok = len(sc_results) > 0 and len(cloze_results) > 0
    print(f"\nResult: {'PASS' if ok else 'FAIL'}")
