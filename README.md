# xgboost-scikit-kaggle-titanic

```python
import sys
import os
from typing import Dict

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from src.data.split import split_data  # Ensure src is in PYTHONPATH
import joblib

def train_model(config: Dict) -> xgb.XGBClassifier:
    X_train, X_val, y_train, y_val = split_data()

    model_params = {k: v for k, v in config.items() if k in ["n_estimators", "learning_rate", "max_depth", "objective", "use_label_encoder"]}
    model = xgb.XGBClassifier(**model_params)

    fit_params = {
        "eval_set": [(X_val, y_val)],
        "eval_metric": config.get("eval_metric", "logloss"),
        "early_stopping_rounds": config.get("early_stopping_rounds", 10),
        "verbose": False
    }

    model.fit(X_train, y_train, **fit_params)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

    model_path = config.get("model_path", "models/xgboost_titanic_model.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}.")

    return model

def main():
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
        train_model(config)
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()

```