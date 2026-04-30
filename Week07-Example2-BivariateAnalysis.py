# -*- coding: utf-8 -*-
"""
Week 07 - Example 2: Bivariate Analysis
Demonstrates analyzing relationships between two variables using correlation and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample dataset
np.random.seed(42)
n = 100
study_hours = np.random.uniform(1, 8, n)
# Create correlated test score
test_score = 50 + 8 * study_hours + np.random.normal(0, 5, n)

data = {
    'student_id': range(1, n + 1),
    'study_hours': study_hours,
    'test_score': test_score,
    'gender': np.random.choice(['Male', 'Female'], n),
    'prior_gpa': np.random.uniform(2.0, 4.0, n)
}
df = pd.DataFrame(data)

print("=== Bivariate Analysis ===\n")

# Correlation Analysis
print("Correlation between Study Hours and Test Score:")
correlation = df['study_hours'].corr(df['test_score'])
print(f"Pearson Correlation: {correlation:.3f}")

# Spearman Correlation (rank-based)
spearman_corr = df['study_hours'].corr(df['test_score'], method='spearman')
print(f"Spearman Correlation: {spearman_corr:.3f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter Plot with Regression Line
axes[0, 0].scatter(df['study_hours'], df['test_score'], alpha=0.6)
z = np.polyfit(df['study_hours'], df['test_score'], 1)
p = np.poly1d(z)
axes[0, 0].plot(df['study_hours'], p(df['study_hours']), "r--", linewidth=2, label='Trend Line')
axes[0, 0].set_title(f'Study Hours vs Test Score (r={correlation:.3f})')
axes[0, 0].set_xlabel('Study Hours')
axes[0, 0].set_ylabel('Test Score')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Scatter Plot with Categories
for gender in df['gender'].unique():
    mask = df['gender'] == gender
    axes[0, 1].scatter(df[mask]['study_hours'], df[mask]['test_score'], 
                       label=gender, alpha=0.6, s=50)
axes[0, 1].set_title('Study Hours vs Test Score by Gender')
axes[0, 1].set_xlabel('Study Hours')
axes[0, 1].set_ylabel('Test Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Hexbin Plot (density)
hexbin = axes[1, 0].hexbin(df['study_hours'], df['test_score'], gridsize=15, cmap='YlOrRd')
axes[1, 0].set_title('Study Hours vs Test Score (Density)')
axes[1, 0].set_xlabel('Study Hours')
axes[1, 0].set_ylabel('Test Score')
plt.colorbar(hexbin, ax=axes[1, 0], label='Count')

# Seaborn Joint Plot alternative - Regression Plot
from scipy import stats as sp_stats
axes[1, 1].scatter(df['study_hours'], df['test_score'], alpha=0.6)
slope, intercept, r_value, p_value, std_err = sp_stats.linregress(df['study_hours'], df['test_score'])
line = slope * df['study_hours'] + intercept
axes[1, 1].plot(df['study_hours'], line, 'r-', linewidth=2)
axes[1, 1].set_title(f'Linear Regression (R²={r_value**2:.3f}, p<0.001)')
axes[1, 1].set_xlabel('Study Hours')
axes[1, 1].set_ylabel('Test Score')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Correlation with other variables
print("\n\nCorrelation Matrix:")
numeric_cols = ['study_hours', 'test_score', 'prior_gpa']
corr_matrix = df[numeric_cols].corr()
print(corr_matrix)

# Visualize correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()
