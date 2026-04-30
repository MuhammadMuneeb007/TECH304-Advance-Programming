# -*- coding: utf-8 -*-
"""
Week 07 - Example 7: Heatmap for Feature Selection on Zoo Data
Loads Zoo.csv and uses heatmaps to compare feature similarity and class profiles.
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
selected_cols = ["hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize"]

print("=== Heatmap for Feature Selection on Zoo Data ===\n")
print("Dataset shape:", df.shape)

correlation_matrix = df[feature_cols].corr()
class_profile = df.groupby("class_name")[selected_cols].mean().round(2)
feature_spread = class_profile.max() - class_profile.min()
important_features = feature_spread.sort_values(ascending=False).head(6)

print("\nTop traits that vary most across classes:")
print(important_features.round(2))

print("\nClass profile preview:")
print(class_profile.head())

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, square=True, linewidths=0.4, ax=axes[0], cbar_kws={"label": "Correlation"})
axes[0].set_title("Feature Correlation Heatmap")

sns.heatmap(class_profile, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.4, ax=axes[1], cbar_kws={"label": "Mean value"})
axes[1].set_title("Average Trait Heatmap by Class")
axes[1].set_xlabel("Trait")
axes[1].set_ylabel("Class")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()

print("\nHeatmap takeaway: traits with the largest spread across classes are the best candidates for cluster separation, especially hair, feathers, milk, eggs, and backbone.")
