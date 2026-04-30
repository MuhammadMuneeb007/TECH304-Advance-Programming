# -*- coding: utf-8 -*-
"""
Week 07 - Example 8: Clustering Analysis
Demonstrates k-means, hierarchical clustering, and cluster evaluation techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

print("=== Clustering Analysis ===\n")

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)
X[:100] += [2, 2]
X[100:200] += [-2, -2]
X[200:] += [2, -2]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. K-Means Clustering
print("K-Means Clustering")
print("=" * 40)

# Find optimal number of clusters using Elbow Method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Visualize Elbow Method
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
axes[0].set_title('Elbow Method for Optimal k')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score for Different k')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Optimal k appears to be 3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels_km = kmeans.fit_predict(X_scaled)

print(f"Optimal number of clusters: {optimal_k}")
print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels_km):.3f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, cluster_labels_km):.3f}")

# Visualize K-Means Results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot with clusters
scatter = axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels_km, 
                         cmap='viridis', s=50, alpha=0.6, edgecolors='black')
centers_scaled = scaler.transform(kmeans.cluster_centers_)
axes[0].scatter(centers_scaled[:, 0], centers_scaled[:, 1], c='red', marker='X', 
               s=300, edgecolors='black', linewidths=2, label='Centroids')
axes[0].set_title(f'K-Means Clustering (k={optimal_k})')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend()
plt.colorbar(scatter, ax=axes[0], label='Cluster')

# Silhouette plot
silhouette_vals = silhouette_samples(X_scaled, cluster_labels_km)
y_lower = 10

for i in range(optimal_k):
    cluster_silhouette_vals = silhouette_vals[cluster_labels_km == i]
    cluster_silhouette_vals.sort()
    
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = plt.cm.viridis(i / optimal_k)
    axes[1].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)
    y_lower = y_upper + 10

axes[1].set_xlabel('Silhouette Coefficient')
axes[1].set_ylabel('Cluster Label')
axes[1].set_title('Silhouette Plot for K-Means')
axes[1].axvline(x=silhouette_score(X_scaled, cluster_labels_km), color="red", linestyle="--",
               label=f'Average Silhouette Score: {silhouette_score(X_scaled, cluster_labels_km):.3f}')
axes[1].legend()

plt.tight_layout()
plt.show()

# 2. Hierarchical Clustering
print("\n\nHierarchical Clustering")
print("=" * 40)

# Create linkage matrix
linkage_matrix = linkage(X_scaled, method='ward')

# Dendrogram
fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(linkage_matrix, ax=ax, no_labels=True)
ax.set_title('Hierarchical Clustering Dendrogram')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Distance')
ax.axhline(y=20, color='r', linestyle='--', label='Cut-off line')
ax.legend()
plt.tight_layout()
plt.show()

# Agglomerative Clustering
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
cluster_labels_hc = hierarchical.fit_predict(X_scaled)

print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels_hc):.3f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, cluster_labels_hc):.3f}")

# Visualize Hierarchical Clustering
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

scatter = axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels_hc, 
                         cmap='plasma', s=50, alpha=0.6, edgecolors='black')
axes[0].set_title('Hierarchical Clustering (Ward)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
plt.colorbar(scatter, ax=axes[0], label='Cluster')

# Compare K-Means vs Hierarchical
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels_km, 
               cmap='viridis', s=50, alpha=0.6, edgecolors='black', marker='o', label='K-Means')
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels_hc, 
               cmap='plasma', s=50, alpha=0.3, edgecolors='black', marker='s', label='Hierarchical')
axes[1].set_title('Comparison: K-Means (circles) vs Hierarchical (squares)')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()

plt.tight_layout()
plt.show()

# 3. DBSCAN (Density-Based Clustering)
print("\n\nDBSCAN Clustering")
print("=" * 40)

dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels_dbscan = dbscan.fit_predict(X_scaled)

n_clusters = len(set(cluster_labels_dbscan)) - (1 if -1 in cluster_labels_dbscan else 0)
n_noise = list(cluster_labels_dbscan).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
if n_clusters > 1:
    print(f"Silhouette Score: {silhouette_score(X_scaled[cluster_labels_dbscan != -1], cluster_labels_dbscan[cluster_labels_dbscan != -1]):.3f}")

# Visualize DBSCAN
fig, ax = plt.subplots(figsize=(10, 8))

# Plot normal points
mask = cluster_labels_dbscan != -1
scatter = ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=cluster_labels_dbscan[mask], 
                    cmap='rainbow', s=50, alpha=0.7, edgecolors='black', label='Clusters')

# Plot noise points
ax.scatter(X_scaled[~mask, 0], X_scaled[~mask, 1], c='red', marker='x', 
          s=100, label='Noise Points', linewidths=2)

ax.set_title(f'DBSCAN Clustering (eps=0.5, min_samples=5)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.tight_layout()
plt.show()

# 4. Comparison of Methods
print("\n\nComparison of Clustering Methods")
print("=" * 40)

methods = ['K-Means', 'Hierarchical', 'DBSCAN']
silhouette_scores_comp = [
    silhouette_score(X_scaled, cluster_labels_km),
    silhouette_score(X_scaled, cluster_labels_hc),
    silhouette_score(X_scaled[cluster_labels_dbscan != -1], cluster_labels_dbscan[cluster_labels_dbscan != -1]) if n_clusters > 1 else 0
]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(methods, silhouette_scores_comp, color=['skyblue', 'lightcoral', 'lightgreen'])
ax.set_ylabel('Silhouette Score')
ax.set_title('Comparison of Clustering Methods')
ax.set_ylim(0, 1)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.3f}', ha='center', va='bottom')

ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("\nAll clustering analysis completed successfully!")
