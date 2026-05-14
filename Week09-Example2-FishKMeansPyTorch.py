from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

# Load the Fish dataset
DATA_PATH = Path(__file__).parent / "Fish.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

fish_data = pd.read_csv(DATA_PATH)

# Use numeric features only (drop Species)
features = fish_data.drop(columns=["Species"])

# Convert data to PyTorch tensor
X = features.values
X_tensor = torch.tensor(X, dtype=torch.float)

# Number of iterations
num_iterations = 100

for K in [3, 4]:
    # Randomly initialize centroids
    centroids = X_tensor[torch.randperm(X_tensor.size(0))[:K]]

    # Perform K-means clustering
    for _ in range(num_iterations):
        # Compute distances from each data point to centroids
        distances = torch.sqrt(torch.sum((X_tensor[:, None] - centroids) ** 2, dim=2))

        # Assign each data point to the closest centroid
        _, cluster_assignment = torch.min(distances, dim=1)

        # Update centroids based on the mean of data points in each cluster
        for k in range(K):
            cluster_points = X_tensor[cluster_assignment == k]
            if cluster_points.numel() == 0:
                continue
            centroids[k] = torch.mean(cluster_points, dim=0)

    # Visualize the clusters (first two features)
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_assignment.numpy(), s=50, cmap="viridis")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200, alpha=0.75)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Clusters Identified by KMeans (PyTorch) | K={K}")
    plt.show()
