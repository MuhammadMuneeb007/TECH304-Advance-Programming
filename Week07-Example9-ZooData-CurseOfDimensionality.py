# -*- coding: utf-8 -*-
"""
Week 07 - Example 9: Curse of Dimensionality on Zoo Data
Loads Zoo.csv and shows how pairwise distances become less informative as dimensions grow.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

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
    df["class_name"] = df["type"].map(CLASS_NAMES)
    return df


def distance_statistics(matrix):
    distances = pairwise_distances(matrix)
    upper = distances[np.triu_indices_from(distances, k=1)]
    nearest = np.partition(upper, 0)[0]
    farthest = upper.max()
    return {
        "mean_distance": upper.mean(),
        "std_distance": upper.std(),
        "cv_distance": upper.std() / upper.mean(),
        "nearest_to_farthest": nearest / farthest,
    }


df = load_zoo_data()
feature_cols = [column for column in df.columns if column not in {"animal_name", "type", "class_name"}]
base_matrix = StandardScaler().fit_transform(df[feature_cols])

rng = np.random.default_rng(42)
sample_size = min(60, len(df))
base_matrix = base_matrix[:sample_size]

feature_levels = [2, 4, 8, 12, 16, 24, 32]
results = []

print("=== Curse of Dimensionality on Zoo Data ===\n")
print("Base feature count:", len(feature_cols))
print("Sample size used for distance calculations:", sample_size)

for total_dimensions in feature_levels:
    if total_dimensions <= base_matrix.shape[1]:
        matrix = base_matrix[:, :total_dimensions]
    else:
        extra_dimensions = total_dimensions - base_matrix.shape[1]
        noise = rng.normal(size=(sample_size, extra_dimensions))
        matrix = np.hstack([base_matrix, noise])
    stats = distance_statistics(matrix)
    stats["dimensions"] = matrix.shape[1]
    results.append(stats)

results_df = pd.DataFrame(results)
print("\nDistance summary by dimension:")
print(results_df[["dimensions", "mean_distance", "cv_distance", "nearest_to_farthest"]].round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(results_df["dimensions"], results_df["cv_distance"], marker="o", color="teal")
axes[0].set_title("Distance Concentration")
axes[0].set_xlabel("Number of dimensions")
axes[0].set_ylabel("Coefficient of variation")
axes[0].grid(True, alpha=0.3)

axes[1].plot(results_df["dimensions"], results_df["nearest_to_farthest"], marker="o", color="darkorange")
axes[1].set_title("Nearest-to-Farthest Distance Ratio")
axes[1].set_xlabel("Number of dimensions")
axes[1].set_ylabel("Ratio")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nCurse of dimensionality takeaway: as dimensions increase, distances become more similar, so clustering needs careful feature selection and scaling.")
