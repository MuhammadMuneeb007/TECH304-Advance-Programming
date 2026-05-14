import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Fix seed for reproducible synthetic data and training
torch.manual_seed(42)

# Generate a synthetic dataset (four Gaussian blobs)
centers = torch.tensor([[-4.0, -2.0], [0.0, 4.0], [4.0, 0.0], [3.5, 4.5]])
points = [center + 0.6 * torch.randn(100, 2) for center in centers]
X = torch.cat(points, dim=0)

# Standardize the data so each feature has mean 0 and std 1
X = (X - X.mean(dim=0)) / X.std(dim=0)


class NeuralClusterer(nn.Module):
    def __init__(self, num_clusters, dim):
        super().__init__()
        # Learnable centroids (cluster centers)
        self.centroids = nn.Parameter(torch.randn(num_clusters, dim))

    def forward(self, x, temperature=1.0):
        # Squared Euclidean distances to centroids
        distances = torch.cdist(x, self.centroids) ** 2
        # Soft assignments: higher weight for closer centroids
        weights = torch.softmax(-distances / temperature, dim=1)
        return weights, distances


def train_clusterer(data, num_clusters, epochs=200):
    # Create the neural clustering model
    model = NeuralClusterer(num_clusters, data.size(1))
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    for epoch in range(epochs):
        # Forward pass: compute soft assignments and distances
        weights, distances = model(data)
        # Weighted distance loss (soft K-means objective)
        loss = (weights * distances).sum(dim=1).mean()

        # Backpropagation and parameter update (centroids move)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"K={num_clusters} | Epoch [{epoch + 1}/{epochs}] | Loss: {loss.item():.4f}")

    with torch.no_grad():
        # Convert soft assignments into hard cluster labels
        weights, _ = model(data)
        labels = torch.argmax(weights, dim=1)
        # Extract learned centroids for plotting
        centroids = model.centroids.detach()

    return labels, centroids


# Train and visualize clustering for k=3 and k=4
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, k in zip(axes, [3, 4]):
    # Train a separate clustering model per k
    labels, centroids = train_clusterer(X, num_clusters=k)
    # Plot clustered data and learned centroids
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap="viridis")
    ax.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200, alpha=0.75)
    ax.set_title(f"Neural Clustering (k={k})")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.tight_layout()
plt.show()
