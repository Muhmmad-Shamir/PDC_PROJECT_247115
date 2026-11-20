from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

files = sorted(DATA_DIR.glob("clean_*.csv"))

reference_columns = None

for file in files:
    df = pd.read_csv(file, nrows=5)
    columns = list(df.columns)

    print(f"\nFile: {file.name}")
    print("Columns:", len(columns))
    print("Label:", columns[-1])

    if reference_columns is None:
        reference_columns = columns
        print("Reference file established.")
    elif columns == reference_columns:
        print("Columns match.")
    else:
        print("WARNING: Columns do not match!")

print("\nDataset verification complete.")