# -*- coding: utf-8 -*-
"""
Week 07 - Example 6: Seaborn Statistical Visualization
Demonstrates advanced statistical plotting with seaborn for exploratory data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=== Seaborn Statistical Visualization ===\n")

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Load sample dataset
df_iris = sns.load_dataset('iris')
df_tips = sns.load_dataset('tips')
df_flights = sns.load_dataset('flights')

print("Sample Dataset (Iris):")
print(df_iris.head())

# 1. Relational Plots (showing relationships)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter Plot with Hue
sns.scatterplot(data=df_iris, x='sepal_length', y='sepal_width', 
                hue='species', size='petal_length', ax=axes[0, 0])
axes[0, 0].set_title('Sepal Length vs Width (by Species)')

# Line Plot with Hue
df_tips_daily = df_tips.groupby(['day', 'sex']).agg({'total_bill': 'mean'}).reset_index()
sns.lineplot(data=df_tips_daily, x='day', y='total_bill', hue='sex', 
             marker='o', ax=axes[0, 1])
axes[0, 1].set_title('Average Bill by Day and Gender')

# Regression Plot
sns.regplot(data=df_iris, x='petal_length', y='petal_width', 
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[1, 0])
axes[1, 0].set_title('Petal Length vs Width with Regression Line')

# Scatter Plot with Style
sns.scatterplot(data=df_tips, x='total_bill', y='tip', 
                hue='day', style='sex', size='size', ax=axes[1, 1])
axes[1, 1].set_title('Bill vs Tip (by Day and Gender)')

plt.tight_layout()
plt.show()

# 2. Distribution Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram with KDE
sns.histplot(data=df_iris, x='sepal_length', kde=True, hue='species', ax=axes[0, 0])
axes[0, 0].set_title('Sepal Length Distribution')

# KDE Plot
sns.kdeplot(data=df_iris, x='sepal_length', y='sepal_width', 
            hue='species', fill=True, ax=axes[0, 1])
axes[0, 1].set_title('2D KDE Plot: Sepal Dimensions')

# Violin Plot
sns.violinplot(data=df_tips, x='day', y='total_bill', hue='sex', ax=axes[1, 0])
axes[1, 0].set_title('Bill Amount by Day and Gender')

# Box Plot
sns.boxplot(data=df_iris, x='species', y='sepal_length', ax=axes[1, 1])
axes[1, 1].set_title('Sepal Length by Species')

plt.tight_layout()
plt.show()

# 3. Categorical Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Strip Plot
sns.stripplot(data=df_tips, x='day', y='total_bill', hue='sex', 
              jitter=True, alpha=0.6, ax=axes[0, 0])
axes[0, 0].set_title('Bill Amount by Day (Strip Plot)')

# Swarm Plot
sns.swarmplot(data=df_tips, x='day', y='total_bill', hue='sex', 
              ax=axes[0, 1])
axes[0, 1].set_title('Bill Amount by Day (Swarm Plot)')

# Bar Plot
sns.barplot(data=df_tips, x='day', y='total_bill', hue='sex', ax=axes[1, 0])
axes[1, 0].set_title('Average Bill by Day and Gender')

# Count Plot
sns.countplot(data=df_tips, x='day', hue='sex', ax=axes[1, 1])
axes[1, 1].set_title('Count of Customers by Day and Gender')

plt.tight_layout()
plt.show()

# 4. Matrix Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap - Correlation
corr_matrix = df_iris.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, ax=axes[0])
axes[0].set_title('Iris Dataset Correlation Heatmap')

# Heatmap - Pivot Table
flights_pivot = df_flights.pivot_table(values='passengers', index='month', columns='year')
sns.heatmap(flights_pivot, cmap='YlGnBu', annot=True, fmt='d', ax=axes[1], 
            cbar_kws={'label': 'Passengers'})
axes[1].set_title('Airline Passengers Heatmap')

plt.tight_layout()
plt.show()

# 5. Figure-level Interface (FacetGrid)
print("\n\nCreating FacetGrid visualization...")

g = sns.FacetGrid(df_tips, col='time', row='sex', height=5)
g.map(sns.scatterplot, 'total_bill', 'tip', alpha=0.6)
g.set_axis_labels('Total Bill ($)', 'Tip ($)')
g.fig.suptitle('Bills and Tips by Time and Gender', y=1.01)
plt.tight_layout()
plt.show()

# 6. Pair Plot
print("\nCreating Pair Plot...")
pairplot = sns.pairplot(df_iris, hue='species', diag_kind='kde', height=2.5)
pairplot.fig.suptitle('Iris Dataset Pair Plot', y=1.01)
plt.tight_layout()
plt.show()

# 7. Joint Plot
print("\nCreating Joint Plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter with marginal distributions
g1 = sns.jointplot(data=df_iris, x='sepal_length', y='sepal_width', 
                   kind='scatter', height=5)
g1.plot_joint(sns.kdeplot, fill=True, cmap='Blues', thresh=0)
plt.suptitle('Sepal Dimensions Joint Plot', y=0.98)

# Regression joint plot
g2 = sns.jointplot(data=df_tips, x='total_bill', y='tip', kind='reg', height=5)
plt.suptitle('Bill vs Tip Joint Plot', y=0.98)

plt.tight_layout()
plt.show()

# 8. Cluster Map (Hierarchical Clustering)
print("\nCreating Cluster Map...")
g = sns.clustermap(corr_matrix, cmap='coolwarm', center=0, 
                   fmt='.2f', figsize=(8, 8))
g.ax_heatmap.set_title('Hierarchical Clustered Heatmap')
plt.show()

# 9. LM Plot (Regression with FacetGrid)
print("\nCreating LM Plots...")
g = sns.lmplot(data=df_tips, x='total_bill', y='tip', col='sex', row='time', height=4)
g.fig.suptitle('Tip Amount vs Bill (by Gender and Time)', y=1.01)
plt.tight_layout()
plt.show()

print("\nAll seaborn visualizations completed successfully!")
