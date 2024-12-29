# src/data/split.py

import pandas as pd
from sklearn.model_selection import train_test_split

def split_data():
    processed_data_path = "data/processed/train_processed.csv"
    data = pd.read_csv(processed_data_path)

    X = data.drop("Survived", axis=1)
    y = data["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val
