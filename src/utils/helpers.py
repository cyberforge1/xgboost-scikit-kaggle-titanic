# src/utils/helpers.py

def parse_input(input_str):
    input_data = {}
    for pair in input_str.split(","):
        key, value = pair.split("=")
        input_data[key] = float(value) if value.replace('.', '', 1).isdigit() else value
    return input_data
