import os
import pandas as pd
import joblib

from rag import CareContextRAG


def main():
    # Make paths work no matter where you run the script from
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "events.csv")
    out_path = os.path.join(project_root, "carecontext_index.joblib")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Cannot find events.csv at: {data_path}\n"
            "Make sure the file exists in: data/events.csv"
        )

    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("events.csv loaded but it is empty. Add at least 1 row of data.")

    # Build the retrieval index
    rag = CareContextRAG(df)

    # Save it for demo.py to use
    joblib.dump(rag, out_path)

    print(f"Loaded rows: {len(df)}")
    print(f"Built index and saved to: {out_path}")
    print("Next run: python src/demo.py")


if __name__ == "__main__":
    main()
