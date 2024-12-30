# src/models/train.py

import sys
import os
import logging
from typing import Dict

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from src.data.split import split_data  # Ensure src is in PYTHONPATH
import joblib

# Configuration for logging
def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Sets up a logger with specified name and log file."""
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console_handler)

    return logger

# Initialize logger
logger = setup_logger('training_logger', 'training.log')

def train_model(config: Dict) -> xgb.XGBClassifier:
    """
    Trains an XGBoost model based on the provided configuration.

    Args:
        config (dict): Configuration parameters for the model.

    Returns:
        xgb.XGBClassifier: Trained XGBoost model.
    """
    try:
        # Log the XGBoost version and file path
        logger.info(f"Using XGBoost version: {xgb.__version__}")
        logger.info(f"XGBoost is loaded from: {xgb.__file__}")
        
        # Start data splitting
        logger.info("Starting the data splitting process...")
        X_train, X_val, y_train, y_val = split_data()
        logger.info("Data successfully split into training and validation sets.")
    except Exception as e:
        logger.exception("Failed during data splitting.")
        raise e

    try:
        # Extract model parameters from config
        model_params = {k: v for k, v in config.items() if k in ["n_estimators", "learning_rate", "max_depth", "objective", "use_label_encoder"]}
        logger.info(f"Initializing the XGBoost model with parameters: {model_params}")
        model = xgb.XGBClassifier(**model_params)
    except Exception as e:
        logger.exception("Failed to initialize the XGBoost model.")
        raise e

    try:
        # Extract fit parameters from config
        fit_params = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": config.get("eval_metric", "logloss"),
            "early_stopping_rounds": config.get("early_stopping_rounds", 10),
            "verbose": False
        }
        logger.info(f"Training the model with fit parameters: {fit_params}")
        model.fit(
            X_train,
            y_train,
            **fit_params
        )
        logger.info("Model training completed successfully.")
    except Exception as e:
        logger.exception("Error during model training.")
        raise e

    try:
        # Evaluate the model
        logger.info("Evaluating the model...")
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
    except Exception as e:
        logger.exception("Error during model evaluation.")
        raise e

    try:
        # Save the model
        model_path = config.get("model_path", "models/xgboost_titanic_model.joblib")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}.")
    except Exception as e:
        logger.exception("Error saving the model.")
        raise e

    return model

def main():
    logger.info("Starting the training script...")
    
    # Configuration dictionary for the XGBoost model
    config = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "objective": "binary:logistic",
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "early_stopping_rounds": 10,
        "model_path": "models/xgboost_titanic_model.joblib"
    }

    try:
        model = train_model(config)
        logger.info("Training script completed successfully.")
    except Exception as e:
        logger.error("Training script failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
