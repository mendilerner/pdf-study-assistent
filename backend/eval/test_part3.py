"""Part 3 verification: Embedding generation with correct e5 prefixes.

The key test: a relevant passage must score higher than an unrelated one.
This fails if "passage: " / "query: " prefixes are missing.

Usage:
    uv run python backend/eval/test_part3.py
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.embeddings import embed_query, embed_passages


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def main():
    passed = True

    question = "מהם ארבעת גורמי החוסן?"
    relevant = "גורמי החוסן הנפשי כוללים אופטימיות, חוש הומור, דבקות במשימה ואמונה דתית."
    unrelated = "תורת הקוונטים עוסקת בתיאור התנהגותם של חלקיקים תת-אטומיים ואנרגיה."

    print("Loading model and encoding (first call may take 10-20s)...\n")

    q_vec = embed_query(question)
    p_vecs = embed_passages([relevant, unrelated])

    # Test 1: Vector dimensions
    print("--- Test 1: Vector dimensions (1024) ---")
    if len(q_vec) == 1024:
        print(f"  query vector: {len(q_vec)} dims — PASS")
    else:
        print(f"  FAIL: query vector has {len(q_vec)} dims, expected 1024")
        passed = False

    for i, v in enumerate(p_vecs):
        if len(v) == 1024:
            print(f"  passage {i} vector: {len(v)} dims — PASS")
        else:
            print(f"  FAIL: passage {i} vector has {len(v)} dims, expected 1024")
            passed = False

    # Test 2: Normalization (L2 norm ≈ 1.0)
    print("\n--- Test 2: L2 normalization ---")
    for label, vec in [("query", q_vec), ("passage_0", p_vecs[0]), ("passage_1", p_vecs[1])]:
        norm = math.sqrt(sum(x * x for x in vec))
        ok = abs(norm - 1.0) < 0.01
        print(f"  {label} L2 norm: {norm:.6f} — {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    # Test 3: Relevant passage scores higher (the real test)
    print("\n--- Test 3: Relevant passage outranks irrelevant ---")
    sim_relevant = cosine_sim(q_vec, p_vecs[0])
    sim_unrelated = cosine_sim(q_vec, p_vecs[1])
    print(f"  query <-> relevant:  {sim_relevant:.4f}")
    print(f"  query <-> unrelated: {sim_unrelated:.4f}")
    margin = sim_relevant - sim_unrelated
    print(f"  margin: {margin:.4f}")

    if sim_relevant > sim_unrelated:
        print("  PASS")
    else:
        print("  FAIL: relevant passage did not score higher")
        passed = False

    # Test 4: Margin is meaningful (not a near-tie)
    print("\n--- Test 4: Margin is meaningful (> 0.05) ---")
    if margin > 0.05:
        print(f"  margin {margin:.4f} > 0.05 — PASS")
    else:
        print(f"  FAIL: margin {margin:.4f} is too small — prefixes may be wrong")
        passed = False

    # Test 5: Batch consistency (single vs batch should match)
    print("\n--- Test 5: Batch consistency ---")
    single_vec = embed_passages([relevant])[0]
    diff = max(abs(a - b) for a, b in zip(single_vec, p_vecs[0]))
    ok = diff < 1e-5
    print(f"  max element diff between single and batch: {diff:.2e} — {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    print(f"\n{'='*40}")
    if passed:
        print("Part 3 PASSED — embeddings are correct.")
    else:
        print("Part 3 FAILED — see errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
