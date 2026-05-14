import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Convert data to TensorFlow tensor
X_tensor = tf.constant(X, dtype=tf.float32)

# Define the number of clusters
num_clusters = 4

# Define KMeansClustering model
kmeans = tf.compat.v1.estimator.experimental.KMeans(
    num_clusters=num_clusters, use_mini_batch=False
)

# Define input function
def input_fn():
    return tf.compat.v1.train.limit_epochs(X_tensor, num_epochs=1)


# Train the KMeansClustering model
num_iterations = 100
for _ in range(num_iterations):
    kmeans.train(input_fn)

# Get the cluster centroids
cluster_centers = kmeans.cluster_centers()

# Get the cluster assignments for each data point
cluster_indices = list(kmeans.predict_cluster_index(input_fn))

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=cluster_indices, s=50, cmap="viridis")
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c="red", s=200, alpha=0.75)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Clusters Identified by KMeans (TensorFlow)")
plt.show()
