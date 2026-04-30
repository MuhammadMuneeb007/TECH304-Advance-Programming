# -*- coding: utf-8 -*-
"""
Week 07 - Example 1: Univariate Analysis
Demonstrates analyzing a single variable at a time using descriptive statistics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample dataset
np.random.seed(42)
data = {
    'student_id': range(1, 101),
    'math_score': np.random.normal(75, 15, 100),
    'reading_score': np.random.normal(72, 12, 100),
    'gender': np.random.choice(['Male', 'Female'], 100),
    'grade': np.random.choice(['A', 'B', 'C', 'D'], 100)
}
df = pd.DataFrame(data)

print("=== Univariate Analysis ===\n")

# Descriptive Statistics
print("Descriptive Statistics for Math Score:")
print(df['math_score'].describe())
print(f"\nSkewness: {df['math_score'].skew():.2f}")
print(f"Kurtosis: {df['math_score'].kurtosis():.2f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram
axes[0, 0].hist(df['math_score'], bins=20, edgecolor='black', color='skyblue')
axes[0, 0].set_title('Math Score Distribution (Histogram)')
axes[0, 0].set_xlabel('Score')
axes[0, 0].set_ylabel('Frequency')

# Box Plot
axes[0, 1].boxplot(df['math_score'], vert=True)
axes[0, 1].set_title('Math Score Distribution (Box Plot)')
axes[0, 1].set_ylabel('Score')

# Density Plot
df['math_score'].plot(kind='density', ax=axes[1, 0], color='green')
axes[1, 0].set_title('Math Score Distribution (Density Plot)')
axes[1, 0].set_xlabel('Score')

# KDE Plot
sns.kdeplot(data=df, x='math_score', ax=axes[1, 1], fill=True, color='purple')
axes[1, 1].set_title('Math Score Distribution (KDE Plot)')
axes[1, 1].set_xlabel('Score')

plt.tight_layout()
plt.show()

# Categorical Variable Analysis
print("\n\nGrade Distribution:")
print(df['grade'].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar Chart
df['grade'].value_counts().plot(kind='bar', ax=axes[0], color='coral')
axes[0].set_title('Grade Distribution (Bar Chart)')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Grade')

# Pie Chart
df['grade'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
axes[1].set_title('Grade Distribution (Pie Chart)')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
