# test_xgboost.py

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_xgboost_fit():
    print(f"XGBoost version: {xgb.__version__}")
    print(f"XGBoost is loaded from: {xgb.__file__}")

    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False
    )

    # Train model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    test_xgboost_fit()
