from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

files = sorted(DATA_DIR.glob("clean_*.csv"))

for file in files:
    print("\nChecking:", file.name)

    for chunk in pd.read_csv(file, chunksize=100_000):
        features = chunk.drop(columns=["Label"])

        numeric = features.apply(pd.to_numeric, errors="coerce")

        print("NaN values:", numeric.isna().sum().sum())
        print("Infinite values:", np.isinf(numeric.to_numpy()).sum())
        print("Largest feature value:", numeric.max().max())

        break  # Check only the first chunk of each file

print("\nCheck completed.")
