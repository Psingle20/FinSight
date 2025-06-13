# For loading .xlsx or .csv data
import pandas as pd
import os

def load_tabular_data():
    data_path = "data"
    dfs = []

    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_path, file))
            dfs.append(df)
        elif file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(data_path, file))
            dfs.append(df)

    return dfs