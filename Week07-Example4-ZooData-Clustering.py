# -*- coding: utf-8 -*-
"""
Week 07 - Example 4: Clustering on Zoo Data
Loads Zoo.csv directly and demonstrates hierarchical clustering and DBSCAN.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

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
X_scaled = StandardScaler().fit_transform(df[feature_cols])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("=== Clustering on Zoo Data ===\n")
print("Dataset shape:", df.shape)
print("\nFeature columns:", ", ".join(feature_cols))

print("\nHierarchical clustering")
linkage_matrix = linkage(X_scaled, method="ward")
fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(linkage_matrix, ax=ax, no_labels=True)
ax.set_title("Zoo Data Dendrogram")
ax.set_xlabel("Sample index")
ax.set_ylabel("Distance")
plt.tight_layout()
plt.show()

hierarchical = AgglomerativeClustering(n_clusters=7, linkage="ward")
df["hierarchical_cluster"] = hierarchical.fit_predict(X_scaled)
print(f"Silhouette score: {silhouette_score(X_scaled, df['hierarchical_cluster']):.3f}")

print("\nDBSCAN clustering")
dbscan = DBSCAN(eps=2.2, min_samples=4)
df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)
cluster_count = len(set(df["dbscan_cluster"])) - (1 if -1 in df["dbscan_cluster"].values else 0)
noise_count = int((df["dbscan_cluster"] == -1).sum())
print(f"Number of clusters: {cluster_count}")
print(f"Number of noise points: {noise_count}")
if cluster_count > 1:
    mask = df["dbscan_cluster"] != -1
    print(f"Silhouette score: {silhouette_score(X_scaled[mask], df.loc[mask, 'dbscan_cluster']):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=df["hierarchical_cluster"], cmap="plasma", s=60, edgecolors="black", alpha=0.8)
axes[0].set_title("Hierarchical Clusters in PCA Space")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

mask = df["dbscan_cluster"] != -1
axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=df.loc[mask, "dbscan_cluster"], cmap="viridis", s=60, edgecolors="black", alpha=0.8, label="Clusters")
axes[1].scatter(X_pca[~mask, 0], X_pca[~mask, 1], c="red", marker="x", s=80, linewidths=2, label="Noise")
axes[1].set_title("DBSCAN Clusters in PCA Space")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[1].legend()

plt.tight_layout()
plt.show()

print("\nClustering takeaway: the animal traits naturally group similar creatures, but density-based methods can mark mixed or sparse animals as noise.")
