# src/data/preprocess.py

import pandas as pd
import os

def preprocess_data():
    raw_data_path = os.path.join("data", "raw")
    processed_data_path = os.path.join("data", "processed")
    os.makedirs(processed_data_path, exist_ok=True)

    train_path = os.path.join(raw_data_path, "train.csv")
    test_path = os.path.join(raw_data_path, "test.csv")
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Example preprocessing: fill missing values and encode categorical variables
    for df in [train_data, test_data]:
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna('S', inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    train_data.to_csv(os.path.join(processed_data_path, "train_processed.csv"), index=False)
    test_data.to_csv(os.path.join(processed_data_path, "test_processed.csv"), index=False)
    print(f"Processed data saved to {processed_data_path}.")
