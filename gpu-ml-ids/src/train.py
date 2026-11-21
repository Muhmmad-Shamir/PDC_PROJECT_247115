"""Train the SAME IDS network on CPU and GPU and compare actual training times.

First run: python src/preprocess.py --samples 2000000
Then run:  python src/train.py --epochs 25

Uses cached data/prepared/*.npy. Does not load or clean CSV files.
Results: results/training_comparison.{csv,json,png}.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
PREPARED_DIR = ROOT / "data" / "prepared"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
SEED = 42


class IDSNet(nn.Module):
    """Larger network for a meaningful training workload."""
    def __init__(self, input_size):
        super().__init__()
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


def load_prepared_data():
    required = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy", "dataset_info.json"]
    for name in required:
        if not (PREPARED_DIR / name).exists():
            raise FileNotFoundError(
                f"Missing {PREPARED_DIR / name}. Run src/preprocess.py first."
            )
    # Memory mapping avoids eagerly copying prepared millions of rows into RAM.
    x_train = np.load(PREPARED_DIR / "X_train.npy", mmap_mode="r+")
    y_train = np.load(PREPARED_DIR / "y_train.npy", mmap_mode="r+")
    x_test = np.load(PREPARED_DIR / "X_test.npy", mmap_mode="r+")
    y_test = np.load(PREPARED_DIR / "y_test.npy", mmap_mode="r+")
    with open(PREPARED_DIR / "dataset_info.json", encoding="utf-8") as file:
        info = json.load(file)
    if x_train.shape[1] != x_test.shape[1] or len(x_train) != len(y_train):
        raise ValueError("Prepared data shapes do not match")
    train_set = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).view(-1, 1))
    test_set = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test).view(-1, 1))
    return train_set, test_set, info


def train_on_device(device, initial_state, train_set, test_set, epochs, batch_size, input_size):
    torch.manual_seed(SEED)
    model = IDSNet(input_size)
    model.load_state_dict(initial_state)
    model = model.to(device)

    positives = float(train_set.tensors[1].sum().item())
    negatives = len(train_set) - positives
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
        pin_memory=(device.type == "cuda")
    )

    print(f"\n--- {device.type.upper()} TRAINING ---", flush=True)
    epoch_times = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        loss_total = torch.zeros((), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_total += loss.detach()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - epoch_start
        avg_loss = (loss_total / len(loader)).item()
        if not np.isfinite(avg_loss):
            raise RuntimeError(f"Non-finite loss during {device.type} training")
        epoch_times.append(elapsed)
        print(f"{device.type.upper()} | Epoch {epoch+1}/{epochs} | "
              f"Loss {avg_loss:.5f} | {elapsed:.2f} sec", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    # Training timer excludes test-time evaluation and CSV preprocessing.
    total_seconds = time.perf_counter() - total_start
    model.eval()
    outputs = []
    with torch.no_grad():
        for xb, _ in DataLoader(test_set, batch_size=batch_size):
            predicted = (torch.sigmoid(model(xb.to(device))) >= 0.5)
            outputs.append(predicted.cpu().numpy().flatten())
    prediction = np.concatenate(outputs)
    actual = test_set.tensors[1].numpy().flatten()
    result = {
        "device": device.type.upper(),
        "seconds": round(total_seconds, 3),
        "minutes": round(total_seconds / 60, 3),
        "accuracy": round(float(accuracy_score(actual, prediction)), 5),
        "precision": round(float(precision_score(actual, prediction, zero_division=0)), 5),
        "recall": round(float(recall_score(actual, prediction, zero_division=0)), 5),
        "f1": round(float(f1_score(actual, prediction, zero_division=0)), 5),
        "epoch_seconds": epoch_times,
    }
    print(f"{result['device']} training took {total_seconds/60:.2f} minutes | "
          f"Accuracy {result['accuracy']*100:.2f}% | F1 {result['f1']*100:.2f}%", flush=True)
    return result, model


def save_comparison(cpu, gpu, info, epochs, batch_size):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cpu_time = cpu["seconds"]
    gpu_time = gpu["seconds"] if gpu else None
    speedup = cpu_time / gpu_time if gpu_time and gpu_time > 0 else None
    results = [cpu] + ([gpu] if gpu else [])

    rows = [{
        "device": r["device"],
        "samples": info["sampled_rows"],
        "training_rows": info["training_rows"],
        "features": info["feature_count"],
        "epochs": epochs,
        "batch_size": batch_size,
        "training_seconds": r["seconds"],
        "training_minutes": r["minutes"],
        "accuracy": r["accuracy"],
        "precision": r["precision"],
        "recall": r["recall"],
        "f1": r["f1"],
    } for r in results]
    with open(RESULTS_DIR / "training_comparison.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(RESULTS_DIR / "training_comparison.json", "w", encoding="utf-8") as file:
        json.dump({"dataset": info, "epochs": epochs, "batch_size": batch_size,
                   "cpu": cpu, "gpu": gpu, "gpu_speedup": speedup}, file, indent=2)

    labels = [r["device"] for r in results]
    seconds = [r["seconds"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, [t / 60 for t in seconds])
    ax.set_title(f"IDS Training: {info['sampled_rows']:,} rows, {epochs} epochs")
    ax.set_ylabel("Actual training time (minutes)")
    ax.set_ylim(0, max(seconds) / 60 * 1.20 if max(seconds) > 0 else 1)
    for bar, sec in zip(bars, seconds):
        ax.text(bar.get_x() + bar.get_width()/2, sec/60, f"{sec/60:.2f} min",
                ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_comparison.png", dpi=180)
    plt.close(fig)

    print("\n=========== TRAINING COMPARISON ===========")
    print(f"Total sampled: {info['sampled_rows']:,} rows | Training: {info['training_rows']:,} "
          f"| Features: {info['feature_count']} | Epochs: {epochs}")
    print(f"CPU training time: {cpu_time:.2f} seconds ({cpu_time/60:.2f} min)")
    if gpu:
        print(f"GPU training time: {gpu_time:.2f} seconds ({gpu_time/60:.2f} min)")
        print(f"GPU speedup: {speedup:.2f}x")
    else:
        print("CUDA unavailable: CPU-only result, no speedup")
    print("Saved: results/training_comparison.csv, .json and .png")


def main():
    parser = argparse.ArgumentParser(description="Compare real CPU and GPU IDS training times")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    torch.manual_seed(SEED)
    torch.set_num_threads(min(6, os.cpu_count() or 4))
    train_set, test_set, info = load_prepared_data()
    print(f"Prepared dataset: {info['sampled_rows']:,} rows "
          f"({info['training_rows']:,} train, {info['testing_rows']:,} test)")
    print(f"Real features: {info['feature_count']} | Batch size: {args.batch_size}")
    print("Data preparation and testing are excluded from training times.")
    initial = IDSNet(info["feature_count"]).state_dict()
    initial = {key: val.detach().clone() for key, val in initial.items()}

    cpu, cpu_model = train_on_device(
        torch.device("cpu"), initial, train_set, test_set,
        args.epochs, args.batch_size, info["feature_count"]
    )
    del cpu_model
    gpu = None
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        gpu, gpu_model = train_on_device(
            torch.device("cuda"), initial, train_set, test_set,
            args.epochs, args.batch_size, info["feature_count"]
        )
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in gpu_model.state_dict().items()},
                   MODELS_DIR / "ids_comparison_gpu.pth")
    save_comparison(cpu, gpu, info, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
