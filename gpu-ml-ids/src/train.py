from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Settings
SAMPLE_PER_CHUNK = 500
MAX_SAMPLES = 200_000
BATCH_SIZE = 2048
EPOCHS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_sample():
    print("Loading dataset sample...")

    files = sorted(DATA_DIR.glob("clean_*.csv"))
    samples = []

    for file in files:
        print("Reading:", file.name)

        for chunk in pd.read_csv(file, chunksize=100_000):
            sample_size = min(SAMPLE_PER_CHUNK, len(chunk))

            samples.append(
                chunk.sample(n=sample_size, random_state=42)
            )

            combined = pd.concat(samples, ignore_index=True)

            if len(combined) > MAX_SAMPLES:
                combined = combined.sample(
                    n=MAX_SAMPLES, random_state=42
                )

            samples = [combined]

    data = pd.concat(samples, ignore_index=True)

    print("Sample size:", len(data))
    print("Label distribution:")
    print(data["Label"].value_counts())

    return data


def main():
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    data = load_sample()

    X = data.drop(columns=["Label"]).values.astype(np.float32)
    y = data["Label"].values.astype(np.float32)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train).unsqueeze(1)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test).unsqueeze(1)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )

    # Model
    model = IDSModel(X_train.shape[1]).to(DEVICE)

    # Handle class imbalance
    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)

    pos_weight = torch.tensor(
        [negatives / positives],
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    print("\nStarting training...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {avg_loss:.4f}"
        )

    # Evaluation
    model.eval()
    predictions = []
    actual = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)

            outputs = model(batch_X)
            preds = (torch.sigmoid(outputs) >= 0.5).int()

            predictions.extend(preds.cpu().numpy().flatten())
            actual.extend(batch_y.numpy().flatten())

    print("\n--- TEST RESULTS ---")
    print("Accuracy :", accuracy_score(actual, predictions))
    print("Precision:", precision_score(actual, predictions, zero_division=0))
    print("Recall   :", recall_score(actual, predictions, zero_division=0))
    print("F1 Score:", f1_score(actual, predictions, zero_division=0))

    # Save model and scaler
    torch.save(model.state_dict(), MODEL_DIR / "ids_model.pth")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    print("\nModel saved:", MODEL_DIR / "ids_model.pth")
    print("Scaler saved:", MODEL_DIR / "scaler.pkl")
    print("Training completed!")


if __name__ == "__main__":
    main()