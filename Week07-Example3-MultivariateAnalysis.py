# -*- coding: utf-8 -*-
"""
Week 07 - Example 3: Multivariate Analysis
Demonstrates analyzing relationships between multiple variables simultaneously using PCA and correlation matrices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Iris Dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print("=== Multivariate Analysis ===\n")
print("Dataset Shape:", df.shape)
print("\nFirst Few Rows:")
print(df.head())

# Correlation Matrix
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print("\n\nCorrelation Matrix:")
corr_matrix = df[numeric_cols].corr()
print(corr_matrix)

# Visualize Correlation Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1, ax=axes[0])
axes[0].set_title('Correlation Matrix Heatmap')

# Clustermap shows clustering of variables
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

# Alternative: Create clustered heatmap
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
            fmt='.2f', square=True, ax=axes[1], cbar_kws={'label': 'Correlation'})
axes[1].set_title('Correlation Matrix (Alternative View)')

plt.tight_layout()
plt.show()

# Principal Component Analysis
print("\n\n=== Principal Component Analysis (PCA) ===")

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained Variance
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"\nExplained Variance Ratio: {explained_var}")
print(f"Cumulative Explained Variance: {cumulative_var}")

# Visualize PCA Results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree Plot
axes[0].bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, label='Individual')
axes[0].plot(range(1, len(explained_var) + 1), cumulative_var, 'ro-', linewidth=2, label='Cumulative')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('PCA Scree Plot')
axes[0].set_xticks(range(1, len(explained_var) + 1))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# PC1 vs PC2 by Species
pca_df = pd.DataFrame(data=X_pca[:, :2], columns=['PC1', 'PC2'])
pca_df['species'] = df['species']

colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
for species in df['species'].unique():
    mask = pca_df['species'] == species
    axes[1].scatter(pca_df[mask]['PC1'], pca_df[mask]['PC2'], 
                   label=species, alpha=0.7, s=100, color=colors.get(species, 'black'))

axes[1].set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
axes[1].set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
axes[1].set_title('First Two Principal Components by Species')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Loadings (contribution of original variables to PCs)
print("\n\nPCA Loadings (Contributions):")
loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(numeric_cols))],
    index=numeric_cols
)
print(loadings_df)

# Visualize Loadings
plt.figure(figsize=(10, 6))
sns.heatmap(loadings_df.iloc[:, :2], annot=True, cmap='RdBu_r', center=0, 
            fmt='.2f', cbar_kws={'label': 'Loading'})
plt.title('PCA Loadings for PC1 and PC2')
plt.tight_layout()
plt.show()
