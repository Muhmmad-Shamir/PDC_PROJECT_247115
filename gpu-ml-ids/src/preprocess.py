"""CPU vs GPU training benchmark using cleaned CIC-IDS2018 CSV files.

Save as src/preprocess.py. Uses ../data/processed/clean_*.csv.
Run with --samples 2000000 --epochs 25.
Requires: numpy, pandas, torch, scikit-learn, matplotlib.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'processed'
RESULTS_DIR = ROOT / 'results'
SEED = 42


class IDSNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Bigger than the earlier 128 -> 64 model: more substantial real computation.
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x)


def count_csv_rows(path):
    """Count data rows in cleaned, single-line-per-record CSV efficiently."""
    lines = 0
    last_byte = b'\n'
    with open(path, 'rb') as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            lines += block.count(b'\n')
            last_byte = block[-1:]
    if last_byte != b'\n':
        lines += 1
    return max(0, lines - 1)  # One header line


def normalize_column(name):
    """Normalize incidental whitespace/BOM in cleaned CSV headers."""
    return str(name).replace('\ufeff', '').strip()


def load_uniform_sample(max_samples, chunk_size=50_000):
    """Sample real rows across all CSVs using columns common to every file.

    Column orders and occasional missing/extra columns may differ among days.
    We train on the shared numeric feature names, in one consistent order.
    The train and test data use the same selected feature set.
    """
    files = sorted(DATA_DIR.glob('clean_*.csv'))
    if not files:
        raise FileNotFoundError(f'No clean_*.csv files found in {DATA_DIR}')

    print(f'Found {len(files)} cleaned CSV files. Counting available rows...', flush=True)
    counts = []
    headers = {}
    shared_features = None
    first_features = None
    for path in files:
        raw_header = list(pd.read_csv(path, nrows=0).columns)
        header = [normalize_column(name) for name in raw_header]
        if len(header) != len(set(header)):
            raise ValueError(f'Duplicate column names after cleaning in {path.name}')
        if 'Label' not in header:
            raise ValueError(f'Missing Label column in {path.name}')
        features = [name for name in header if name != 'Label']
        headers[path] = features
        if first_features is None:
            first_features = features
            shared_features = set(features)
        else:
            shared_features.intersection_update(features)
        rows = count_csv_rows(path)
        counts.append(rows)
        print(f'  {path.name}: {rows:,} rows | {len(features)} features', flush=True)

    # Preserve the column order of the first file across all datasets.
    feature_columns = [name for name in first_features if name in shared_features]
    if len(feature_columns) < 5:
        raise ValueError('Fewer than 5 common features found; inspect CSV headers')
    print(f'Common features used for both devices: {len(feature_columns)}', flush=True)
    for path in files:
        excluded = sorted(set(headers[path]) - shared_features)
        if excluded:
            print(f'  {path.name}: excluding {len(excluded)} unmatched feature(s): '
                  f'{excluded[:5]}', flush=True)

    total_rows = sum(counts)
    target = min(max_samples, total_rows)
    if target < 10:
        raise ValueError('Need at least 10 real rows to train/test')
    print(f'Available: {total_rows:,}  |  Sampling: {target:,} rows', flush=True)
    print('Only original shared network-traffic features are used; none are invented.', flush=True)

    # Preallocate float32 arrays to avoid retaining all CSV chunks in RAM.
    X = np.empty((target, len(feature_columns)), dtype=np.float32)
    y = np.empty(target, dtype=np.float32)
    rng = np.random.default_rng(SEED)
    remaining_population = total_rows
    needed = target
    filled = 0

    for path, expected_rows in zip(files, counts):
        from_file = 0
        observed = 0
        for chunk in pd.read_csv(path, chunksize=chunk_size, dtype=np.float32):
            chunk.columns = [normalize_column(name) for name in chunk.columns]
            n = len(chunk)
            observed += n
            if remaining_population < n:
                raise ValueError('Row counts changed while reading the CSV files')
            if needed == 0:
                remaining_population -= n
                continue

            # Uniform sampling without replacement across all files/chunks.
            k = int(rng.hypergeometric(n, remaining_population - n, needed))
            if k:
                positions = rng.choice(n, size=k, replace=False)
                selected = chunk.iloc[positions]
                X[filled:filled + k] = selected[feature_columns].to_numpy(dtype=np.float32)
                y[filled:filled + k] = selected['Label'].to_numpy(dtype=np.float32)
                filled += k
                needed -= k
                from_file += k
            remaining_population -= n
        if observed != expected_rows:
            raise ValueError(f'Row count changed in {path.name}: expected {expected_rows}, got {observed}')
        print(f'  Sampled {from_file:,} from {path.name} | cumulative {filled:,}', flush=True)

    if filled != target:
        raise RuntimeError(f'Sampling mismatch: got {filled:,}; expected {target:,}')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Label column must contain binary 0/1 labels')
    print(f'Final sample: {filled:,}, benign {(y == 0).sum():,}, attack {(y == 1).sum():,}', flush=True)
    return X, y


def preprocess(X, y):
    # Exact same train/test split and fitted transform for both devices.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    del X, y

    # Clean invalid values before computing percentiles. Some CSVs may have
    # unusual float values even after the original cleaning stage.
    np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Learn clipping/scaling from TRAINING data only (avoid test leakage).
    print('Clipping extreme values and scaling real features...', flush=True)
    lower, upper = np.percentile(X_train, [1, 99], axis=0)
    np.clip(X_train, lower, upper, out=X_train)
    np.clip(X_test, lower, upper, out=X_test)

    scaler = StandardScaler(copy=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError('Invalid numerical values remain after preprocessing')
    return (torch.from_numpy(X_train), torch.from_numpy(y_train).view(-1, 1),
            torch.from_numpy(X_test), torch.from_numpy(y_test).view(-1, 1))


def train_and_test(device, initial_weights, train_set, test_set, epochs, batch_size, input_size):
    torch.manual_seed(SEED)
    model = IDSNet(input_size).to(device)
    model.load_state_dict(initial_weights)
    positives = float(train_set.tensors[1].sum().item())
    negatives = len(train_set) - positives
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1)], device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
        pin_memory=(device.type == 'cuda'),
    )

    # Time training including batch transfers for the GPU, excluding CSV/preprocessing.
    epoch_seconds = []
    print(f'\n--- TRAINING ON {device.type.upper()} ---', flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        if device.type == 'cuda':
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=(device.type == 'cuda'))
            y_batch = y_batch.to(device, non_blocking=(device.type == 'cuda'))
            optimizer.zero_grad(set_to_none=True)
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss on {device.type}')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter() - epoch_start
        epoch_seconds.append(seconds)
        print(f'{device.type.upper()} | epoch {epoch + 1:2d}/{epochs} | '
              f'loss {loss_sum / len(loader):.5f} | {seconds:.2f} seconds', flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    training_time = time.perf_counter() - total_start

    model.eval()
    all_predictions = []
    with torch.no_grad():
        for x_batch, _ in DataLoader(test_set, batch_size=batch_size * 2):
            x_batch = x_batch.to(device)
            scores = model(x_batch)
            all_predictions.append((torch.sigmoid(scores) >= 0.5).cpu().numpy().flatten())
    predicted = np.concatenate(all_predictions)
    actual = test_set.tensors[1].numpy().flatten()
    scores = {
        'device': device.type.upper(),
        'seconds': round(training_time, 3),
        'minutes': round(training_time / 60, 3),
        'accuracy': round(float(accuracy_score(actual, predicted)), 5),
        'precision': round(float(precision_score(actual, predicted, zero_division=0)), 5),
        'recall': round(float(recall_score(actual, predicted, zero_division=0)), 5),
        'f1': round(float(f1_score(actual, predicted, zero_division=0)), 5),
        'epoch_seconds': epoch_seconds,
    }
    print(f"{device.type.upper()} training: {training_time:.1f} seconds "
          f"({training_time / 60:.2f} minutes), F1: {scores['f1']:.4f}", flush=True)
    return scores, model


def main():
    parser = argparse.ArgumentParser(description='CPU vs GPU real IDS training benchmark')
    parser.add_argument('--samples', type=int, default=2_000_000,
                        help='maximum number of real rows across all cleaned CSVs')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=4096)
    args = parser.parse_args()
    if args.samples < 10 or args.epochs < 1 or args.batch_size < 1:
        parser.error('samples must be >= 10; epochs and batch-size must be >= 1')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable. Both CPU and RTX 3070 are required for comparison.')

    RESULTS_DIR.mkdir(exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(6, os.cpu_count() or 4))
    print('GPU:', torch.cuda.get_device_name(0))
    print('Requested:', f'{args.samples:,} real rows;', f'{args.epochs} epochs;',
          f'batch size {args.batch_size}', flush=True)

    load_start = time.perf_counter()
    X, y = load_uniform_sample(args.samples)
    sample_count, feature_count = X.shape
    x_train, y_train, x_test, y_test = preprocess(X, y)
    del X, y
    print(f'Data preparation: {(time.perf_counter() - load_start) / 60:.2f} minutes '
          f'(excluded from CPU/GPU training comparison)', flush=True)
    train_set = TensorDataset(x_train, y_train)
    test_set = TensorDataset(x_test, y_test)

    # Both devices start with EXACTLY the same model weights, random seeds, and data.
    initial = IDSNet(feature_count).state_dict()
    initial = {key: val.clone() for key, val in initial.items()}

    cpu, cpu_model = train_and_test(torch.device('cpu'), initial, train_set, test_set,
                                    args.epochs, args.batch_size, feature_count)
    del cpu_model
    gpu, gpu_model = train_and_test(torch.device('cuda'), initial, train_set, test_set,
                                    args.epochs, args.batch_size, feature_count)

    speedup = cpu['seconds'] / gpu['seconds']
    print('\n========= TRAINING COMPARISON =========', flush=True)
    print(f'Rows: {sample_count:,} | Features: {feature_count} | Epochs: {args.epochs}')
    print(f"CPU training: {cpu['minutes']:.2f} minutes")
    print(f"GPU training: {gpu['minutes']:.2f} minutes")
    print(f'GPU speedup:  {speedup:.2f}x')
    print(f"CPU F1: {cpu['f1'] * 100:.2f}% | GPU F1: {gpu['f1'] * 100:.2f}%")

    row_data = [
        {'device': score['device'], 'rows': sample_count, 'features': feature_count,
         'epochs': args.epochs, 'batch_size': args.batch_size,
         'training_seconds': score['seconds'], 'training_minutes': score['minutes'],
         'accuracy': score['accuracy'], 'precision': score['precision'],
         'recall': score['recall'], 'f1': score['f1']}
        for score in (cpu, gpu)
    ]
    csv_path = RESULTS_DIR / 'training_comparison.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(row_data[0].keys()))
        writer.writeheader()
        writer.writerows(row_data)

    result_json = RESULTS_DIR / 'training_comparison.json'
    with open(result_json, 'w', encoding='utf-8') as f:
        json.dump({'rows': sample_count, 'features': feature_count,
                   'epochs': args.epochs, 'batch_size': args.batch_size,
                   'speedup': speedup, 'cpu': cpu, 'gpu': gpu}, f, indent=2)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(['CPU', 'RTX 3070 GPU'], [cpu['minutes'], gpu['minutes']])
    plt.ylabel('Actual training time (minutes)')
    plt.title(f'IDS Training: {sample_count:,} Real Rows, {args.epochs} Epochs')
    for bar, seconds in zip(bars, [cpu['seconds'], gpu['seconds']]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{seconds / 60:.2f} min', ha='center', va='bottom')
    plt.ylim(0, max(cpu['minutes'], gpu['minutes']) * 1.18)
    plt.tight_layout()
    plot_path = RESULTS_DIR / 'training_comparison.png'
    plt.savefig(plot_path, dpi=180)
    plt.close()
    print('\nSaved:', csv_path, result_json, plot_path, sep='\n', flush=True)
    # Keep new benchmark model separate from your original 128/64 IDS model.
    torch.save(gpu_model.cpu().state_dict(), RESULTS_DIR / 'comparison_gpu_model.pth')


if __name__ == '__main__':
    main()
