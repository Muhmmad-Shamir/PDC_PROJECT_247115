from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

files = sorted(DATA_DIR.glob("clean_*.csv"))

print("Processed files:", len(files))

total_rows = 0

for file in files:
    print("\nFile:", file.name)

    # Read column information
    df = pd.read_csv(file, nrows=5)

    columns = len(df.columns)
    features = columns - 1

    # Count all rows without loading entire file into RAM
    rows = 0

    for chunk in pd.read_csv(file, chunksize=100_000):
        rows += len(chunk)

    total_rows += rows

    print("Rows:", f"{rows:,}")
    print("Columns:", columns)
    print("Features:", features)
    print("Label column:", df.columns[-1])

print("\n========== DATASET SUMMARY ==========")
print("Total CSV files:", len(files))
print("Total rows:", f"{total_rows:,}")
print("Total columns:", columns if files else 0)
print("Dataset check completed.")