# tests/test_pipeline.py

import unittest
import os
from src.pipeline import run_pipeline

class TestPipeline(unittest.TestCase):
    def test_pipeline_execution(self):
        # Run the entire pipeline
        run_pipeline()
        
        # Verify if all components produced their outputs
        train_processed = os.path.join("data", "processed", "train_processed.csv")
        test_processed = os.path.join("data", "processed", "test_processed.csv")
        model_file = "xgboost_titanic_model.json"
        
        self.assertTrue(os.path.exists(train_processed))
        self.assertTrue(os.path.exists(test_processed))
        self.assertTrue(os.path.exists(model_file))

if __name__ == "__main__":
    unittest.main()
