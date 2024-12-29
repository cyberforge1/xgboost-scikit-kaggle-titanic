# src/models/train.py

import sys
import os
import logging
from src.data.split import split_data  # Absolute import
import xgboost as xgb

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def train_model():
    logging.info("Starting the data splitting process...")
    X_train, X_val, y_train, y_val = split_data()
    logging.info("Data successfully split into training and validation sets.")

    logging.info("Initializing the XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        objective="binary:logistic",
        eval_metric="logloss",  # Set eval_metric here
        use_label_encoder=False  # Suppress label encoder warnings
    )

    logging.info("Training the model with early stopping...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
        early_stopping_rounds=10
    )

    logging.info("Evaluating the model...")
    accuracy = model.score(X_val, y_val)
    logging.info(f"Validation Accuracy: {accuracy:.2f}")

    model_path = "xgboost_titanic_model.json"
    model.save_model(model_path)
    logging.info(f"Model saved to {model_path}.")

    return model

if __name__ == "__main__":
    logging.info("Starting the training script...")
    train_model()
    logging.info("Training script completed.")
