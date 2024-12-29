# tests/test_preprocess.py

import unittest
import os
import pandas as pd
from src.data.preprocess import preprocess_data

class TestPreprocess(unittest.TestCase):
    def test_preprocessed_files_exist(self):
        # Run the preprocessing
        preprocess_data()
        
        # Check if processed files exist
        train_path = os.path.join("data", "processed", "train_processed.csv")
        test_path = os.path.join("data", "processed", "test_processed.csv")
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(test_path))

    def test_no_missing_values(self):
        preprocess_data()
        train_data = pd.read_csv(os.path.join("data", "processed", "train_processed.csv"))
        
        # Ensure no missing values in the processed dataset
        self.assertFalse(train_data.isnull().any().any())

if __name__ == "__main__":
    unittest.main()
