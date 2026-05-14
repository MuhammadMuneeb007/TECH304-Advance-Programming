from pathlib import Path

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# Load the Fish dataset
DATA_PATH = Path(__file__).parent / "Fish.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

fish_data = pd.read_csv(DATA_PATH)

# Use numeric features only (drop Species)
features = fish_data.drop(columns=["Species"])
X = features.values

# Convert data to TensorFlow tensor
X_tensor = tf.constant(X, dtype=tf.float32)

# Define input function
def input_fn():
    return tf.compat.v1.train.limit_epochs(X_tensor, num_epochs=1)


# Run K-means for k=3 and k=4
for num_clusters in [3, 4]:
    kmeans = tf.compat.v1.estimator.experimental.KMeans(
        num_clusters=num_clusters, use_mini_batch=False
    )

    num_iterations = 100
    for _ in range(num_iterations):
        kmeans.train(input_fn)

    cluster_centers = kmeans.cluster_centers()
    cluster_indices = list(kmeans.predict_cluster_index(input_fn))

    # Visualize clusters using the first two features
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_indices, s=50, cmap="viridis")
    plt.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1], c="red", s=200, alpha=0.75
    )
    plt.xlabel("Weight")
    plt.ylabel("Length1")
    plt.title(f"Fish KMeans (TensorFlow) | k={num_clusters}")
    plt.show()
