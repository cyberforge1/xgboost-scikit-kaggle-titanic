# tests/test_helpers.py

import unittest
from src.utils.helpers import parse_input

class TestHelpers(unittest.TestCase):
    def test_parse_input(self):
        input_str = "age=22,sex=female,class=3"
        expected_output = {"age": 22.0, "sex": "female", "class": 3.0}
        parsed_input = parse_input(input_str)
        
        self.assertEqual(parsed_input, expected_output)

if __name__ == "__main__":
    unittest.main()
