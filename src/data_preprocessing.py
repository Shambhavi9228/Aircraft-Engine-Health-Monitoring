import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, sep=" ", header=None)
    return df

def preprocess_data(df):
    df = df.dropna(axis=1, how='all')
    return df
