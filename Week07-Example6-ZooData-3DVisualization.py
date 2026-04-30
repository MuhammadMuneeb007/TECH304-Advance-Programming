# -*- coding: utf-8 -*-
"""
Week 07 - Example 6: 3D Visualisation on Zoo Data
Uses mpl_toolkits.mplot3d and matplotlib.pyplot to create 3D scatter plots.
"""

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

print("=== 3D Visualisation on Zoo Data ===\n")
print("This example uses fig, ax, and scatter in 3D to show how animal traits spread across classes.")
print("\nSmall sample rows:")
preview_df = df.reset_index().rename(columns={"index": "row_id"})
print(preview_df[["row_id", "hair", "feathers", "eggs", "milk", "legs", "class_name"]].head(2))

fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121, projection="3d")
scatter1 = ax1.scatter(df["hair"], df["milk"], df["legs"], c=df["class_code"], cmap="viridis", s=60, alpha=0.85)
ax1.set_title("Hair vs Milk vs Legs")
ax1.set_xlabel("Hair")
ax1.set_ylabel("Milk")
ax1.set_zlabel("Legs")
fig.colorbar(scatter1, ax=ax1, shrink=0.65, pad=0.1, label="Class code")

ax2 = fig.add_subplot(122, projection="3d")
scatter2 = ax2.scatter(df["feathers"], df["eggs"], df["backbone"], c=df["class_code"], cmap="plasma", s=60, alpha=0.85)
ax2.set_title("Feathers vs Eggs vs Backbone")
ax2.set_xlabel("Feathers")
ax2.set_ylabel("Eggs")
ax2.set_zlabel("Backbone")
fig.colorbar(scatter2, ax=ax2, shrink=0.65, pad=0.1, label="Class code")

plt.tight_layout()
plt.show()

print("\n3D takeaway: mammals cluster near high hair and milk values, while birds separate strongly through feathers and eggs.")
