# -*- coding: utf-8 -*-
"""
Week 07 - Example 1: Exploratory Data Analysis on Zoo Data
Loads Zoo.csv directly from GitHub and explores the animal feature columns.
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
feature_cols = [column for column in df.columns if column not in {"animal_name", "type", "class_name", "class_code"}]
preview_cols = [
    "row_id", "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
    "predator", "toothed", "backbone", "breathes", "venomous", "fins", "legs",
    "tail", "domestic", "catsize", "class_name"
]
preview_df = df.reset_index().rename(columns={"index": "row_id"})

print("=== EDA on Zoo Data ===\n")
print("Dataset shape:", df.shape)
print("\nFirst two rows that match the requested style:")
print(preview_df.loc[preview_df["class_name"] == "mammal", preview_cols].head(2))
print("\nClass distribution:")
print(df["class_name"].value_counts().sort_index())

selected_traits = ["hair", "feathers", "eggs", "milk", "airborne", "aquatic", "backbone", "breathes", "tail", "catsize"]
trait_summary = df.groupby("class_name")[selected_traits].mean().round(2)
print("\nAverage trait values by class:")
print(trait_summary)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(data=df, x="class_name", order=df["class_name"].value_counts().sort_index().index, ax=axes[0], palette="Set2")
axes[0].set_title("Class Counts")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=30)

sns.heatmap(trait_summary, annot=True, cmap="YlGnBu", fmt=".2f", ax=axes[1], cbar_kws={"label": "Mean value"})
axes[1].set_title("Average Trait Profile by Class")
axes[1].set_xlabel("Trait")
axes[1].set_ylabel("Class")

sns.histplot(df["legs"], bins=8, discrete=True, ax=axes[2], color="steelblue")
axes[2].set_title("Leg Count Distribution")
axes[2].set_xlabel("Legs")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.show()

print("\nEDA takeaway: hair and milk are strong mammal indicators, feathers point to birds, and eggs appear in many non-mammal classes.")
