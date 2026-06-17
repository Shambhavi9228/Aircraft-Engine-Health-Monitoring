import pandas as pd

def load_data(path):
    df = pd.read_csv(path, sep=" ", header=None)
    return df