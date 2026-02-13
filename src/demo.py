# src/demo.py
import os
import joblib

from rag import baseline_llm_like_answer, enhanced_llm_like_answer


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    index_path = os.path.join(project_root, "carecontext_index.joblib")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Cannot find index file: {index_path}\n"
            "Run: python src/build_index.py"
        )

    rag = joblib.load(index_path)

    situation = (
        "Resident is standing with a walking frame in the sitting room at 19:00. "
        "Staff leaves to the equipment room to get a wheelchair, leaving the resident standing."
    )

    # Retrieval query (can be edited for other test prompts)
    query = "sitting room evening left standing transfer wheelchair walking frame fall"

    retrieved = rag.retrieve(query, k=3)

    print("=" * 90)
    print("SCENARIO")
    print("=" * 90)
    print(situation)

    print("\n" + "=" * 90)
    print(baseline_llm_like_answer(situation))

    print("\n" + "=" * 90)
    print(enhanced_llm_like_answer(situation, retrieved))

    print("\n" + "=" * 90)
    print("RETRIEVED EVIDENCE (raw snapshots)")
    print("=" * 90)
    for i, item in enumerate(retrieved, 1):
        print(f"\n[{i}] score={item.score:.3f}")
        print(item.text)


if __name__ == "__main__":
    main()
