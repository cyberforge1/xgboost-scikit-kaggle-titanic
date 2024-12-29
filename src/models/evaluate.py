# src/models/evaluate.py

import pandas as pd
import xgboost as xgb

def evaluate_model(input_features: dict):
    model = xgb.XGBClassifier()
    model.load_model("xgboost_titanic_model.json")

    input_df = pd.DataFrame([input_features])
    prediction = model.predict(input_df)
    return prediction[0]
