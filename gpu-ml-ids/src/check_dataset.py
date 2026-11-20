from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

files = list(DATA_DIR.glob("clean_*.csv"))

print("Processed files:", len(files))

for file in files:
    print("\nFile:", file.name)

    df = pd.read_csv(file, nrows=5)

    print("Columns:", len(df.columns))
    print("Features:", list(df.columns[:-1]))
    print("Label column:", df.columns[-1])
    print("Sample rows:")
    print(df.head(2))

print("\nDataset check completed.")