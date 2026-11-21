"""Prepare real CSE-CIC-IDS2018 data for CPU-versus-GPU training.

Reads project1/data/processed/clean_*.csv from ALL available days.
Run once before src/train.py, e.g.:
    python src/preprocess.py --samples 2000000

Output: data/prepared/{X_train,y_train,X_test,y_test}.npy plus metadata.
No neural-network training happens in this file.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "prepared"
MODEL_DIR = ROOT / "models"
SEED = 42


def count_rows(path):
    """Count rows quickly for these cleaned, one-record-per-line CSV files."""
    newline_count = 0
    last_byte = b"\n"
    with open(path, "rb") as file:
        while True:
            block = file.read(8 * 1024 * 1024)
            if not block:
                break
            newline_count += block.count(b"\n")
            last_byte = block[-1:]
    if last_byte != b"\n":
        newline_count += 1
    return max(0, newline_count - 1)


def clean_header(name):
    return str(name).replace("\ufeff", "").strip()


def load_sample(max_samples, chunk_size=50_000):
    """Uniform random rows without replacement from every cleaned CSV."""
    files = sorted(INPUT_DIR.glob("clean_*.csv"))
    if not files:
        raise FileNotFoundError(f"No clean_*.csv found in {INPUT_DIR}")

    print(f"Found {len(files)} cleaned CSV files.", flush=True)
    file_counts = []
    headers = {}
    common = None
    first_header = None

    for path in files:
        names = [clean_header(n) for n in pd.read_csv(path, nrows=0).columns]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate columns after trimming: {path.name}")
        if "Label" not in names:
            raise ValueError(f"Missing Label column: {path.name}")
        features = [n for n in names if n != "Label"]
        headers[path.name] = features
        if common is None:
            common = set(features)
            first_header = features
        else:
            common &= set(features)
        count = count_rows(path)
        file_counts.append(count)
        print(f"  {path.name}: {count:,} rows, {len(features)} features", flush=True)

    features = [name for name in first_header if name in common]
    if len(features) < 5:
        raise ValueError("Fewer than 5 features shared by every file. Check CSV headers.")
    print(f"Common real feature columns: {len(features)}", flush=True)
    total_available = sum(file_counts)
    sample_count = min(max_samples, total_available)
    if sample_count < 10:
        raise ValueError("Not enough rows to train a model")
    print(f"Available: {total_available:,} | Sampling: {sample_count:,}", flush=True)

    # Allocate only requested rows, not the entire 16M-record dataset.
    X = np.empty((sample_count, len(features)), dtype=np.float32)
    y = np.empty(sample_count, dtype=np.float32)
    rng = np.random.default_rng(SEED)
    remaining_rows = total_available
    remaining_to_sample = sample_count
    offset = 0

    for path, expected_count in zip(files, file_counts):
        selected_from_file = 0
        seen = 0
        for chunk in pd.read_csv(path, chunksize=chunk_size, dtype=np.float32):
            chunk.columns = [clean_header(n) for n in chunk.columns]
            n = len(chunk)
            seen += n
            if n > remaining_rows:
                raise ValueError("CSV files changed during sampling")
            if remaining_to_sample:
                k = int(rng.hypergeometric(n, remaining_rows - n, remaining_to_sample))
                if k:
                    chosen = rng.choice(n, size=k, replace=False)
                    rows = chunk.iloc[chosen]
                    X[offset:offset + k] = rows[features].to_numpy(dtype=np.float32)
                    y[offset:offset + k] = rows["Label"].to_numpy(dtype=np.float32)
                    offset += k
                    remaining_to_sample -= k
                    selected_from_file += k
            remaining_rows -= n
        if seen != expected_count:
            raise ValueError(f"CSV row count changed: {path.name}")
        print(f"  Selected {selected_from_file:,} from {path.name}", flush=True)

    if offset != sample_count:
        raise RuntimeError(f"Expected {sample_count:,} rows; got {offset:,}")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Label must contain binary 0/1 values")

    print(f"Total rows: {offset:,} | Benign: {(y == 0).sum():,} | Attacks: {(y == 1).sum():,}")
    return X, y, features, total_available


def prepare(samples):
    start = time.perf_counter()
    X, y, feature_names, available_rows = load_sample(samples)
    total_sampled = len(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    del X, y

    # Learn clipping and scaling from TRAINING data only.
    np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # A fixed training-data subset keeps percentile memory manageable for 16 GB RAM.
    rng = np.random.default_rng(SEED)
    count = min(100_000, len(X_train))
    subset_ids = rng.choice(len(X_train), size=count, replace=False)
    lower, upper = np.percentile(X_train[subset_ids], [1, 99], axis=0)
    np.clip(X_train, lower, upper, out=X_train)
    np.clip(X_test, lower, upper, out=X_test)

    scaler = StandardScaler(copy=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    if not (np.isfinite(X_train).all() and np.isfinite(X_test).all()):
        raise ValueError("Features contain invalid values after preprocessing")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train.astype(np.float32))
    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_test.npy", y_test.astype(np.float32))
    joblib.dump(scaler, MODEL_DIR / "comparison_scaler.pkl")
    np.savez(MODEL_DIR / "comparison_clip_bounds.npz", lower=lower, upper=upper)

    info = {
        "dataset": "CSE-CIC-IDS2018 cleaned CSVs",
        "available_rows": available_rows,
        "sampled_rows": total_sampled,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "features": feature_names,
        "feature_count": len(feature_names),
        "seed": SEED,
    }
    with open(OUTPUT_DIR / "dataset_info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)

    print("\nPREPROCESSING COMPLETE", flush=True)
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | Features: {len(feature_names)}")
    print(f"Preparation time: {(time.perf_counter() - start)/60:.2f} minutes")
    print(f"Saved prepared arrays in: {OUTPUT_DIR}")
    print("Next run: python src/train.py --epochs 25", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare CIC-IDS2018 samples for CPU/GPU training")
    parser.add_argument("--samples", type=int, default=2_000_000)
    args = parser.parse_args()
    if args.samples < 10:
        parser.error("--samples must be at least 10")
    prepare(args.samples)


if __name__ == "__main__":
    main()
