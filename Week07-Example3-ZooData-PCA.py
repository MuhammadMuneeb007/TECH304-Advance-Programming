# -*- coding: utf-8 -*-
"""
Week 07 - Example 3: PCA on Zoo Data
Loads Zoo.csv directly and reduces the animal feature space to two principal components.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
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


df = load_zoo_data()
feature_cols = [column for column in df.columns if column not in {"animal_name", "type", "class_name"}]
X = df[feature_cols].to_numpy()
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["class_name"] = df["class_name"]

print("=== PCA on Zoo Data ===\n")
print("Dataset shape:", df.shape)
print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_.round(3))
print("Cumulative explained variance:", pca.explained_variance_ratio_.sum().round(3))

loadings = pd.DataFrame(pca.components_.T, index=feature_cols, columns=["PC1", "PC2"])
print("\nPCA loadings for the first two components:")
print(loadings.round(2))

plt.figure(figsize=(12, 7))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="class_name", palette="Set2", s=90, edgecolor="black")
plt.title("Zoo Data in PCA Space")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(loadings, annot=True, cmap="RdBu_r", center=0, fmt=".2f", cbar_kws={"label": "Loading"})
plt.title("PCA Loadings")
plt.tight_layout()
plt.show()

print("\nPCA takeaway: if the first two components separate mammals, birds, and fish clusters, the original features contain strong structure even before clustering.")
