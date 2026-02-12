import pandas as pd
import json
import xmltodict
import os

def load_data(file_path):
    """
    Universal data loader for CSV, Excel, JSON, and XML files.
    Returns a Pandas DataFrame.
    """

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".csv":
        df = pd.read_csv(file_path)

    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)

    elif file_extension == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.json_normalize(data)

    elif file_extension == ".xml":
        with open(file_path, "r", encoding="utf-8") as f:
            data = xmltodict.parse(f.read())
        df = pd.json_normalize(data)

    else:
        raise ValueError("Unsupported file format")

    return df
