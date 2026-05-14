import torch
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Convert data to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float)

# Number of clusters
K = 4

# Randomly initialize centroids
centroids = X_tensor[torch.randperm(X_tensor.size(0))[:K]]

# Number of iterations
num_iterations = 100

# Perform K-means clustering
for _ in range(num_iterations):
    # Compute distances from each data point to centroids
    distances = torch.sqrt(torch.sum((X_tensor[:, None] - centroids) ** 2, dim=2))

    # Assign each data point to the closest centroid
    _, cluster_assignment = torch.min(distances, dim=1)

    # Update centroids based on the mean of data points in each cluster
    for k in range(K):
        cluster_points = X_tensor[cluster_assignment == k]
        centroids[k] = torch.mean(cluster_points, dim=0)

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=cluster_assignment.numpy(), s=50, cmap="viridis")
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200, alpha=0.75)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Clusters Identified by KMeans (PyTorch)")
plt.show()
