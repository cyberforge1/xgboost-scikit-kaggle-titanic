# src/data/preprocess.py

import pandas as pd
import os
import logging

def preprocess_data():
    # Configure logging for preprocessing
    logging.basicConfig(
        filename='preprocessing.log',  # Log to a file named preprocessing.log
        filemode='a',  # Append mode
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    
    logging.info("Starting data preprocessing...")
    
    raw_data_path = "data/raw"
    processed_data_path = "data/processed"
    os.makedirs(processed_data_path, exist_ok=True)
    
    train_path = os.path.join(raw_data_path, "train.csv")
    test_path = os.path.join(raw_data_path, "test.csv")
    
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info("Successfully loaded train and test data.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise e
    
    # Handle missing values and encode categorical variables
    for df in [train_data, test_data]:
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna('S', inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        
        # Drop unwanted columns
        df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        logging.info("Dropped columns: Name, Ticket, Cabin.")
    
    # Save the processed data
    train_processed_path = os.path.join(processed_data_path, "train_processed.csv")
    test_processed_path = os.path.join(processed_data_path, "test_processed.csv")
    
    try:
        train_data.to_csv(train_processed_path, index=False)
        test_data.to_csv(test_processed_path, index=False)
        logging.info("Successfully saved processed train and test data.")
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
        raise e
    
    logging.info("Data preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_data()
