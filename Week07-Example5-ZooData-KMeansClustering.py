# -*- coding: utf-8 -*-
"""
Week 07 - Example 5: K-means Clustering on Zoo Data
Loads Zoo.csv directly and shows the elbow method, silhouette scores, and cluster profiles.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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
X_scaled = StandardScaler().fit_transform(df[feature_cols])
preview_df = df.reset_index().rename(columns={"index": "row_id"})

print("=== K-means Clustering on Zoo Data ===\n")
print("Dataset shape:", df.shape)
print("\nSmall sample rows:")
print(preview_df[["row_id", "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize", "class_name"]].head(2))

k_values = range(2, 9)
inertias = []
silhouette_scores = []
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(k_values), inertias, marker="o", color="steelblue")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_values), silhouette_scores, marker="o", color="seagreen")
axes[1].set_title("Silhouette Score by k")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette score")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

optimal_k = 7
model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["kmeans_cluster"] = model.fit_predict(X_scaled)
print(f"Selected k: {optimal_k}")
print(f"Silhouette score: {silhouette_score(X_scaled, df['kmeans_cluster']):.3f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(12, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["kmeans_cluster"], cmap="viridis", s=75, edgecolors="black", alpha=0.85)
plt.title("K-means Clusters in PCA Space")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

cluster_profile = df.groupby("kmeans_cluster")[["hair", "feathers", "eggs", "milk", "airborne", "aquatic", "backbone", "breathes", "fins", "legs", "tail", "catsize"]].mean().round(2)
print("\nAverage feature profile by cluster:")
print(cluster_profile)

print("\nCluster vs class summary:")
print(pd.crosstab(df["class_name"], df["kmeans_cluster"], normalize="index").round(2))

print("\nK-means takeaway: binary animal traits plus legs are enough to separate many zoo groups when the features are scaled first.")
