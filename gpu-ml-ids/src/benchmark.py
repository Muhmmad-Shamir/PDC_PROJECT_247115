import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

BATCH_SIZE = 2048
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10


class IDSModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


def prepare_test_data():
    data = np.load(RESULTS_DIR / "test_data.npz")
    X_test = data["X_test"]

    clip_data = np.load(MODEL_DIR / "clip_bounds.npz")
    X_test = np.clip(X_test, clip_data["lower"], clip_data["upper"])

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    X_test = scaler.transform(X_test)

    X_test = np.nan_to_num(
        X_test, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    return torch.tensor(X_test, dtype=torch.float32)


def benchmark_device(model, data, device):
    model = model.to(device)
    model.eval()

    # Prepare batches on the selected device
    batches = [
        batch.to(device)
        for batch in data.split(BATCH_SIZE)
    ]

    # Warm-up runs
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            for batch in batches:
                model(batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(BENCHMARK_RUNS):
            for batch in batches:
                model(batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    average_time = elapsed / BENCHMARK_RUNS
    milliseconds = average_time * 1000
    per_sample = milliseconds / len(data)

    return milliseconds, per_sample


def main():
    print("Preparing test data...")
    X_test = prepare_test_data()

    print("Test samples:", len(X_test))
    print("Batch size:", BATCH_SIZE)

    # Load saved model
    model = IDSModel(X_test.shape[1])
    model.load_state_dict(
        torch.load(
            MODEL_DIR / "ids_model.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    print("\n--- CPU Benchmark ---")
    cpu_ms, cpu_per_sample = benchmark_device(
        model, X_test, torch.device("cpu")
    )

    print(f"Average inference time: {cpu_ms:.2f} ms")
    print(f"Per sample: {cpu_per_sample:.4f} ms")

    if torch.cuda.is_available():
        print("\n--- GPU Benchmark ---")
        gpu_ms, gpu_per_sample = benchmark_device(
            model, X_test, torch.device("cuda")
        )

        print(f"Average inference time: {gpu_ms:.2f} ms")
        print(f"Per sample: {gpu_per_sample:.4f} ms")

        speedup = cpu_ms / gpu_ms
        print(f"\nGPU Speedup: {speedup:.2f}x")
    else:
        print("\nCUDA is not available. GPU benchmark skipped.")


if __name__ == "__main__":
    main()