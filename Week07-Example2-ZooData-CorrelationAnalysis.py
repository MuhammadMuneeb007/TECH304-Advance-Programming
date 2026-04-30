# -*- coding: utf-8 -*-
"""
Week 07 - Example 2: Correlation Analysis on Zoo Data
Loads Zoo.csv from GitHub and studies feature relationships with a heatmap.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/Zoo.csv"
CLASS_NAMES = {
    1: "mammal",
    2: "bird",
    3: "reptile",
    4: "fish",
    5: "amphibian",
    6: "insect",
    7: "invertebrate",
}


def load_zoo_data():
    df = pd.read_csv(DATA_URL)
    df.columns = [column.strip().lower() for column in df.columns]
    for column in df.columns:
        if column not in {"animal_name", "legs", "type"}:
            df[column] = df[column].replace({True: 1, False: 0, "TRUE": 1, "FALSE": 0}).astype(int)
    if pd.api.types.is_numeric_dtype(df["type"]):
        df["class_name"] = df["type"].map(CLASS_NAMES)
    else:
        df["class_name"] = df["type"].astype(str).str.strip().str.lower()
    df["class_code"] = pd.factorize(df["class_name"])[0] + 1
    return df


df = load_zoo_data()
numeric_cols = [column for column in df.columns if column not in {"animal_name", "type", "class_name", "class_code"}]
corr_matrix = df[numeric_cols + ["class_code"]].corr()

print("=== Correlation Analysis on Zoo Data ===\n")
print("Dataset shape:", df.shape)
print("\nCorrelation matrix:")
print(corr_matrix.round(2))

class_code_corr = corr_matrix["class_code"].drop("class_code").sort_values(key=lambda s: s.abs(), ascending=False)
print("\nStrongest absolute correlations with the numeric class code:")
print(class_code_corr.head(8).round(2))
print("\nNote: class_code is a derived label encoding, so treat those correlations as teaching signals rather than ordinal truth.")

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True, linewidths=0.4, cbar_kws={"label": "Correlation"})
plt.title("Zoo Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nCorrelation takeaway: features like hair, milk, feathers, eggs, and backbone separate animal groups more clearly than weaker traits such as domestic.")
