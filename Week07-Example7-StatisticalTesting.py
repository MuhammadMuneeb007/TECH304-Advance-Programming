# -*- coding: utf-8 -*-
"""
Week 07 - Example 7: Statistical Testing and Analysis
Demonstrates hypothesis testing, statistical measures, and distributions using scipy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("=== Statistical Testing and Analysis ===\n")

# Generate sample data
np.random.seed(42)
group1 = np.random.normal(100, 15, 100)  # Control group
group2 = np.random.normal(108, 15, 100)  # Treatment group
group3 = np.random.normal(95, 15, 100)   # Another group

# 1. Descriptive Statistics
print("Descriptive Statistics for Group 1:")
print(f"Mean: {np.mean(group1):.2f}")
print(f"Median: {np.median(group1):.2f}")
print(f"Std Dev: {np.std(group1):.2f}")
print(f"Skewness: {stats.skew(group1):.3f}")
print(f"Kurtosis: {stats.kurtosis(group1):.3f}")
print(f"95% CI: {stats.t.interval(0.95, len(group1)-1, loc=np.mean(group1), scale=stats.sem(group1))}")

# 2. Normality Tests
print("\n\n=== Normality Tests ===")
shapiro_stat, shapiro_p = stats.shapiro(group1)
print(f"Shapiro-Wilk Test:")
print(f"  Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
print(f"  Result: {'Data is normally distributed' if shapiro_p > 0.05 else 'Data is NOT normally distributed'}")

# Visualize normality
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Q-Q Plot
stats.probplot(group1, dist="norm", plot=axes[0])
axes[0].set_title('Q-Q Plot (Group 1)')

# Histogram with Normal Curve
axes[1].hist(group1, bins=20, density=True, alpha=0.7, edgecolor='black')
mu, sigma = np.mean(group1), np.std(group1)
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
axes[1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
axes[1].set_title('Distribution with Normal Curve')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
plt.show()

# 3. Comparison of Two Groups (t-test)
print("\n\n=== Two-Sample T-Test ===")
t_stat, t_pval = stats.ttest_ind(group1, group2)
print(f"Independent t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {t_pval:.4f}")
print(f"  Result: {'Statistically significant difference' if t_pval < 0.05 else 'No significant difference'}")

# Effect size (Cohen's d)
cohens_d = (np.mean(group2) - np.mean(group1)) / np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
print(f"  Cohen's d (effect size): {cohens_d:.3f}")

# Paired t-test (if comparing same subjects)
group1_before = np.random.normal(100, 10, 50)
group1_after = group1_before + np.random.normal(5, 5, 50)
t_stat_paired, t_pval_paired = stats.ttest_rel(group1_before, group1_after)
print(f"\nPaired t-test (before-after):")
print(f"  t-statistic: {t_stat_paired:.4f}")
print(f"  p-value: {t_pval_paired:.4f}")

# 4. Comparison of Multiple Groups (ANOVA)
print("\n\n=== One-Way ANOVA ===")
f_stat, anova_pval = stats.f_oneway(group1, group2, group3)
print(f"ANOVA Test:")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value: {anova_pval:.4f}")
print(f"  Result: {'Significant difference between groups' if anova_pval < 0.05 else 'No significant difference'}")

# Visualization of groups
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Box plot
data_groups = [group1, group2, group3]
axes[0].boxplot(data_groups, labels=['Group 1', 'Group 2', 'Group 3'])
axes[0].set_title('Comparison of Three Groups')
axes[0].set_ylabel('Value')
axes[0].grid(True, alpha=0.3, axis='y')

# Violin plot
parts = axes[1].violinplot(data_groups, positions=[1, 2, 3], showmeans=True, showmedians=True)
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(['Group 1', 'Group 2', 'Group 3'])
axes[1].set_title('Violin Plot Comparison')
axes[1].set_ylabel('Value')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 5. Non-parametric Tests
print("\n\n=== Non-parametric Tests ===")

# Mann-Whitney U test (alternative to t-test for non-normal data)
u_stat, u_pval = stats.mannwhitneyu(group1, group2)
print(f"Mann-Whitney U Test:")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {u_pval:.4f}")

# Kruskal-Wallis test (alternative to ANOVA)
h_stat, h_pval = stats.kruskal(group1, group2, group3)
print(f"\nKruskal-Wallis Test:")
print(f"  H-statistic: {h_stat:.4f}")
print(f"  p-value: {h_pval:.4f}")

# 6. Correlation Tests
print("\n\n=== Correlation Tests ===")
x = np.random.normal(100, 15, 100)
y = x * 0.7 + np.random.normal(0, 10, 100)  # Correlated with x

pearson_r, pearson_p = stats.pearsonr(x, y)
spearman_r, spearman_p = stats.spearmanr(x, y)

print(f"Pearson Correlation:")
print(f"  r: {pearson_r:.3f}, p-value: {pearson_p:.4f}")

print(f"\nSpearman Correlation:")
print(f"  rho: {spearman_r:.3f}, p-value: {spearman_p:.4f}")

# Visualize correlation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(x, y, alpha=0.6)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
axes[0].plot(x, p(x), "r--", linewidth=2)
axes[0].set_title(f'Pearson r = {pearson_r:.3f}, p < 0.001')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].grid(True, alpha=0.3)

# Rank correlation
x_ranked = stats.rankdata(x)
y_ranked = stats.rankdata(y)
axes[1].scatter(x_ranked, y_ranked, alpha=0.6, color='orange')
axes[1].set_title(f'Spearman rho = {spearman_r:.3f}, p < 0.001')
axes[1].set_xlabel('X Rank')
axes[1].set_ylabel('Y Rank')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. Chi-Square Test (for categorical data)
print("\n\n=== Chi-Square Test ===")
observed = np.array([30, 45, 25])
expected = np.array([33.33, 33.33, 33.33])
chi2_stat, chi2_p = stats.chisquare(observed, expected)
print(f"Chi-Square Test:")
print(f"  Chi-square statistic: {chi2_stat:.4f}")
print(f"  p-value: {chi2_p:.4f}")

print("\n\nAll statistical tests completed successfully!")
