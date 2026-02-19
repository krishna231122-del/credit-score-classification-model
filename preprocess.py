# preprocess.py

import pandas as pd


def preprocess_data(path):
    df = pd.read_csv('/Users/krishnasoni/Downloads/cc.csv')
    df.columns = df.columns.str.strip()
    df["Credit Score"] = df["Credit Score"].str.strip().str.lower()
    df["Credit Score"] = df["Credit Score"].map({
        "low": 0,
        "average": 1,
        "high": 2
    })
    X = df.drop("Credit Score", axis=1)
    y = df["Credit Score"]
    X = pd.get_dummies(X, drop_first=True)
    return X, y
