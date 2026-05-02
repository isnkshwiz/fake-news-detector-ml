import pandas as pd

def load_data():
    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/True.csv")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true]).reset_index(drop=True)
    return df


def preprocess(df):
    # combine title + text
    df["content"] = df["title"] + " " + df["text"]

    # drop unused columns
    df = df[["content", "label"]]

    return df