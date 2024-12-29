# tests/test_train.py

import unittest
import os
from src.models.train import train_model

class TestTrain(unittest.TestCase):
    def test_model_training(self):
        # Train the model
        model = train_model()
        
        # Check if the model is saved
        self.assertTrue(os.path.exists("xgboost_titanic_model.json"))
        
        # Check if the returned model is not None
        self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()
