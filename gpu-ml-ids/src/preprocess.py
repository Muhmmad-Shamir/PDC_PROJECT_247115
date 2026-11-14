"""
Preprocess network-traffic data for the intrusion detection system.

Reads a CSV, turns text columns into numbers, scales the features,
and splits the rows into train and test sets. The last column is the label
(normal vs attack, or a specific attack type).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch

def check_gpu():
    """Check if CUDA is available"""
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

def load_and_preprocess(csv_path):
    """
    Load dataset and do basic preprocessing.
    Update  once you download NSL-KDD or CICIDS.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Original shape:", df.shape)
    print("Columns:", df.columns.tolist())
    
    # Encode categorical columns if any
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Separate features and target (assume last column is label)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    check_gpu()
    # Uncomment after you put the dataset in the data/ folder
    # X_train, X_test, y_train, y_test, scaler = load_and_preprocess("data/your_dataset.csv")
    print("Starter file is ready. Add your dataset and continue.")