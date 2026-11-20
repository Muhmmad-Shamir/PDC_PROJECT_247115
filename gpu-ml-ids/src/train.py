from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2048
EPOCHS = 5
MAX_SAMPLES = 200_000


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
    samples = []
    total = 0

    files = sorted(DATA_DIR.glob("clean_*.csv"))

    for file in files:
        print("Reading:", file.name)

        for chunk in pd.read_csv(file, chunksize=100_000):
            sample = chunk.sample(
                n=min(500, len(chunk)),
                random_state=42
            )

            samples.append(sample)
            total += len(sample)

            if total >= MAX_SAMPLES:
                break

        if total >= MAX_SAMPLES:
            break

    data = pd.concat(samples, ignore_index=True)
    print("\nSample size:", len(data))
    print("Labels:\n", data["Label"].value_counts())

    return data


def main():
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    data = load_sample()

    X = data.drop(columns="Label").to_numpy(dtype=np.float32)
    y = data["Label"].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    np.savez_compressed(
    RESULTS_DIR / "test_data.npz",
    X_test=X_test,
    y_test=y_test
)

    # Clip extreme values using training data percentiles
    lower = np.percentile(X_train, 1, axis=0)
    upper = np.percentile(X_train, 99, axis=0)

    np.savez(
    MODEL_DIR / "clip_bounds.npz",
    lower=lower,
    upper=upper
)

    X_train = np.clip(X_train, lower, upper)
    X_test = np.clip(X_test, lower, upper)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Replace any remaining invalid values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    print("Scaled values valid:", np.isfinite(X_train).all())

    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train).reshape(-1, 1)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test).reshape(-1, 1)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE
    )

    model = IDSModel(X_train.shape[1]).to(DEVICE)

    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)

    pos_weight = torch.tensor(
        [negatives / positives],
        device=DEVICE
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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

            if not torch.isfinite(loss):
                print("Invalid loss detected. Stopping.")
                return

            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.6f}")

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

        # Confusion matrix
        cm = confusion_matrix(actual, predictions, labels=[0, 1])

        print("\n--- CONFUSION MATRIX ---")
        print("                 Predicted")
        print("              Benign  Attack")
        print(f"Actual Benign   {cm[0][0]:>6}  {cm[0][1]:>6}")
        print(f"Actual Attack   {cm[1][0]:>6}  {cm[1][1]:>6}")

        # Plot confusion matrix
        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.title("GPU-ML-IDS Confusion Matrix")
        plt.xticks([0, 1], ["Benign", "Attack"])
        plt.yticks([0, 1], ["Benign", "Attack"])
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")

        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

        plt.tight_layout()

        cm_path = RESULTS_DIR / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300)
        plt.close()

        print("Confusion matrix saved:", cm_path)

        # Save model and scaler
        torch.save(model.state_dict(), MODEL_DIR / "ids_model.pth")
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

        print("\nModel saved successfully.")
        print("Training completed!")


if __name__ == "__main__":
    main()