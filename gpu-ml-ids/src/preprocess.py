from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "CSE-CIC-IDS2018"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000


def preprocess_file(file_path):
    output_file = OUTPUT_DIR / f"clean_{file_path.stem}.csv"

    print(f"\nProcessing: {file_path.name}")

    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip()

        label_col = next(
            (c for c in chunk.columns if c.lower() == "label"),
            None
        )

        if label_col is None:
            print("Skipped: Label column not found")
            return

        labels = chunk[label_col].astype(str).str.strip().str.upper()

        # Remove original label before cleaning features
        chunk.drop(columns=[label_col], inplace=True)

        # Remove identifier columns
        remove_cols = [
            c for c in chunk.columns
            if any(term in c.lower() for term in
                   ["ip", "timestamp", "flow id", "flowid"])
        ]
        chunk.drop(columns=remove_cols, errors="ignore", inplace=True)

        # Convert features to numeric
        for col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # Handle infinite and missing values
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_indices = chunk.dropna().index
        chunk = chunk.loc[valid_indices].copy()

        # Binary labels: BENIGN=0, all attacks=1
        chunk["Label"] = (labels.loc[valid_indices] != "BENIGN").astype("int8")

        # Save cleaned chunk
        chunk.to_csv(
            output_file,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk
        )

        first_chunk = False
        total_rows += len(chunk)

        print(f"Valid rows processed: {total_rows:,}")

    print(f"Saved: {output_file}")
    print(f"Total valid rows: {total_rows:,}")


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in:", DATA_DIR)
        return

    print(f"Found {len(csv_files)} CSV files")

    for file_path in csv_files:
        preprocess_file(file_path)

    print("\nPREPROCESSING COMPLETE!")


if __name__ == "__main__":
    main()