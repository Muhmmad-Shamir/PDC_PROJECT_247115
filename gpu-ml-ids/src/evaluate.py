import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saved test data
test_data = np.load(RESULTS_DIR / "test_data.npz")
X_test = test_data["X_test"]
y_test = test_data["y_test"]

# Load clipping limits
clip_data = np.load(MODEL_DIR / "clip_bounds.npz")
lower = clip_data["lower"]
upper = clip_data["upper"]

# Apply the same clipping used during training
X_test = np.clip(X_test, lower, upper)

# Load scaler
import joblib

scaler = joblib.load(MODEL_DIR / "scaler.pkl")

X_test = scaler.transform(X_test)

# Define model architecture
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


model = IDSModel(X_test.shape[1]).to(device)

# Load trained weights
model.load_state_dict(torch.load(
    MODEL_DIR / "ids_model.pth",
    map_location=device,
    weights_only=True
))

model.eval()

# Convert test data to tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# Predictions
with torch.no_grad():
    outputs = model(X_test_tensor)
    predictions = (torch.sigmoid(outputs) >= 0.5).cpu().numpy().astype(int).flatten()

# Calculate metrics
print("\n--- Model Evaluation ---")
print(f"Accuracy:  {accuracy_score(y_test, predictions) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0) * 100:.2f}%")
print(f"Recall:    {recall_score(y_test, predictions, zero_division=0) * 100:.2f}%")
print(f"F1 Score:  {f1_score(y_test, predictions, zero_division=0) * 100:.2f}%")

print("\n--- Classification Report ---")
print(classification_report(
    y_test,
    predictions,
    target_names=["Benign", "Attack"],
    zero_division=0
))