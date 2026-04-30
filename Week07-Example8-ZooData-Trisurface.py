# -*- coding: utf-8 -*-
"""
Week 07 - Example 8: Trisurface Plot on Zoo Data
Loads Zoo.csv and draws a 3D trisurface using selected animal traits.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
rng = np.random.default_rng(42)
plot_df = df[["hair", "milk", "legs", "class_name"]].copy()
plot_df["hair_plot"] = plot_df["hair"] + rng.normal(0, 0.03, len(plot_df))
plot_df["milk_plot"] = plot_df["milk"] + rng.normal(0, 0.03, len(plot_df))
preview_df = df.reset_index().rename(columns={"index": "row_id"})

print("=== Trisurface Plot on Zoo Data ===\n")
print("Small sample rows:")
print(preview_df[["row_id", "hair", "milk", "legs", "class_name"]].head(2))
print("\nInterpretation note: the surface shows how hair and milk cluster animals near the mammal region while leg counts vary across classes.")

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
surface = ax.plot_trisurf(plot_df["hair_plot"], plot_df["milk_plot"], plot_df["legs"], cmap="viridis", linewidth=0.2, alpha=0.9)
ax.scatter(plot_df["hair_plot"], plot_df["milk_plot"], plot_df["legs"], c=df["class_code"], cmap="viridis", s=25, alpha=0.65)
ax.set_title("Trisurface of Hair, Milk, and Legs")
ax.set_xlabel("Hair")
ax.set_ylabel("Milk")
ax.set_zlabel("Legs")
fig.colorbar(surface, ax=ax, shrink=0.7, pad=0.1, label="Surface height")
plt.tight_layout()
plt.show()

print("\nTrisurface takeaway: mammals sit near the high-hair/high-milk area, while fish and birds separate through low milk and differing leg counts.")
