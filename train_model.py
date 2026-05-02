import pandas as pd

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true])

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df["label"].value_counts())