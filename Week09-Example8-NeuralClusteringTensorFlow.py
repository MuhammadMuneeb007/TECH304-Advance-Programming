import tensorflow as tf
import matplotlib.pyplot as plt


tf.random.set_seed(42)

# Generate a synthetic dataset (four Gaussian blobs)
centers = tf.constant([[-4.0, -2.0], [0.0, 4.0], [4.0, 0.0], [3.5, 4.5]])
points = [center + 0.6 * tf.random.normal((100, 2)) for center in centers]
X = tf.concat(points, axis=0)

# Standardize the data
X = (X - tf.reduce_mean(X, axis=0)) / tf.math.reduce_std(X, axis=0)


class NeuralClusterer(tf.keras.Model):
    def __init__(self, num_clusters, dim):
        super().__init__()
        # Trainable centroids
        self.centroids = tf.Variable(
            tf.random.normal((num_clusters, dim)), trainable=True
        )

    def call(self, inputs, temperature=1.0):
        # Squared Euclidean distances to centroids
        distances = tf.reduce_sum(
            tf.square(tf.expand_dims(inputs, 1) - self.centroids), axis=2
        )
        # Soft assignments: higher weight for closer centroids
        weights = tf.nn.softmax(-distances / temperature, axis=1)
        return weights, distances


def train_clusterer(data, num_clusters, epochs=200):
    model = NeuralClusterer(num_clusters, data.shape[1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            weights, distances = model(data)
            loss = tf.reduce_mean(tf.reduce_sum(weights * distances, axis=1))

        grads = tape.gradient(loss, [model.centroids])
        optimizer.apply_gradients(zip(grads, [model.centroids]))

        if (epoch + 1) % 50 == 0:
            print(
                f"K={num_clusters} | Epoch [{epoch + 1}/{epochs}] | Loss: {loss.numpy():.4f}"
            )

    weights, _ = model(data)
    labels = tf.argmax(weights, axis=1)
    return labels.numpy(), model.centroids.numpy()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, k in zip(axes, [3, 4]):
    labels, centroids = train_clusterer(X, num_clusters=k)
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap="viridis")
    ax.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200, alpha=0.75)
    ax.set_title(f"Neural Clustering (TensorFlow) | k={k}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.tight_layout()
plt.show()
