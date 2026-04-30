# -*- coding: utf-8 -*-
"""
Week 07 - Example 10: Feature Engineering and Advanced Analytics
Demonstrates feature creation, selection, and comprehensive data pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

print("=== Feature Engineering and Advanced Analytics ===\n")

# 1. Create Raw Dataset
np.random.seed(42)
n_samples = 500

raw_data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.uniform(25000, 150000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'loan_amount': np.random.uniform(5000, 500000, n_samples),
    'employment_years': np.random.randint(0, 40, n_samples),
    'num_accounts': np.random.randint(0, 15, n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
})

# Create target variable (loan approval)
raw_data['approved'] = (
    (raw_data['credit_score'] > 650) & 
    (raw_data['income'] > 40000) & 
    (raw_data['age'] > 25)
).astype(int) + np.random.binomial(1, 0.1, n_samples)  # Add noise

print("Raw Dataset:")
print(raw_data.head(10))
print(f"\nDataset Info:")
print(raw_data.info())

# 2. Feature Scaling
print("\n\n=== Feature Scaling ===")

numeric_features = ['age', 'income', 'credit_score', 'loan_amount', 'employment_years', 'num_accounts']

# Standardization (Z-score)
scaler_std = StandardScaler()
data_standardized = raw_data.copy()
data_standardized[numeric_features] = scaler_std.fit_transform(raw_data[numeric_features])

# Normalization (Min-Max)
scaler_minmax = MinMaxScaler()
data_normalized = raw_data.copy()
data_normalized[numeric_features] = scaler_minmax.fit_transform(raw_data[numeric_features])

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(raw_data['income'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Original Income Distribution')
axes[0].set_xlabel('Income')

axes[1].hist(data_standardized['income'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].set_title('Standardized Income Distribution')
axes[1].set_xlabel('Standardized Income')

axes[2].hist(data_normalized['income'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
axes[2].set_title('Normalized Income Distribution')
axes[2].set_xlabel('Normalized Income')

plt.tight_layout()
plt.show()

# 3. Feature Creation
print("\n\n=== Feature Creation ===")

features = raw_data.copy()

# Ratio features
features['debt_to_income'] = features['loan_amount'] / (features['income'] + 1)
features['age_group'] = pd.cut(features['age'], bins=[0, 30, 45, 60, 100], 
                               labels=['Young', 'Middle', 'Senior', 'Elder'])

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(features[['age', 'credit_score']])
poly_df = pd.DataFrame(poly_features, columns=['age', 'credit_score', 'age_squared', 'age_credit', 'credit_squared'])
features = pd.concat([features, poly_df[['age_squared', 'age_credit', 'credit_squared']]], axis=1)

# Interaction features
features['age_income_interaction'] = features['age'] * (features['income'] / 10000)

# Binning features
features['income_bin'] = pd.cut(features['income'], bins=3, labels=['Low', 'Medium', 'High'])
features['credit_rating'] = pd.cut(features['credit_score'], 
                                   bins=[0, 580, 670, 740, 800, 850],
                                   labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])

print("\nNew Features Created:")
print(features[['debt_to_income', 'age_squared', 'age_income_interaction', 'income_bin']].head())

# 4. Categorical Encoding
print("\n\n=== Categorical Encoding ===")

# One-hot encoding
features_encoded = pd.get_dummies(features, columns=['gender', 'marital_status', 'age_group', 
                                                      'income_bin', 'credit_rating'], 
                                  drop_first=True)

print(f"Shape before encoding: {features.shape}")
print(f"Shape after encoding: {features_encoded.shape}")

# 5. Feature Selection
print("\n\n=== Feature Selection ===")

# Prepare data for feature selection
X = features_encoded.drop(['approved'], axis=1)
y = features_encoded['approved']

# SelectKBest with f_classif
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get feature scores
feature_scores = pd.DataFrame({
    'feature': X.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)

print("\nTop 15 Features by F-Score:")
print(feature_scores.head(15))

# Visualize feature importance
fig, ax = plt.subplots(figsize=(12, 6))
top_features = feature_scores.head(15)
ax.barh(range(len(top_features)), top_features['score'], color='steelblue', edgecolor='black')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'])
ax.set_xlabel('F-Score')
ax.set_title('Top 15 Features by F-Score')
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# 6. Correlation Analysis
print("\n\n=== Correlation Analysis ===")

correlation_matrix = features[numeric_features + ['approved']].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# 7. Dimensionality Reduction
print("\n\n=== Dimensionality Reduction with PCA ===")

# Standardize data for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_var = pca.explained_variance_ratio_
cumsum_var = np.cumsum(explained_var)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, len(explained_var[:20]) + 1), explained_var[:20], 
           alpha=0.7, color='steelblue', edgecolor='black', label='Individual')
axes[0].plot(range(1, len(explained_var[:20]) + 1), cumsum_var[:20], 
            'ro-', linewidth=2, label='Cumulative')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('PCA Scree Plot')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cumulative variance
n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
axes[1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-', linewidth=2)
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
axes[1].axvline(x=n_components_95, color='g', linestyle='--', 
               label=f'{n_components_95} components for 95%')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Explained Variance Ratio')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nNumber of components needed for 95% variance: {n_components_95}")
print(f"Original features: {X.shape[1]}")
print(f"Dimensionality reduction: {X.shape[1]} -> {n_components_95} ({n_components_95/X.shape[1]*100:.1f}%)")

# 8. Data Quality Assessment
print("\n\n=== Data Quality Assessment ===")

quality_report = pd.DataFrame({
    'Feature': X.columns,
    'Missing %': (X.isnull().sum() / len(X) * 100).values,
    'Unique Values': [X[col].nunique() for col in X.columns],
    'Data Type': [str(X[col].dtype) for col in X.columns],
    'Variance': [X[col].var() for col in X.columns]
})

print("\nData Quality Report:")
print(quality_report.head(15))

# Visualize missing data
fig, ax = plt.subplots(figsize=(12, 6))
missing_pct = (X.isnull().sum() / len(X) * 100).sort_values(ascending=False)
if missing_pct.sum() > 0:
    missing_pct.head(10).plot(kind='barh', ax=ax, color='coral', edgecolor='black')
    ax.set_xlabel('Missing Percentage (%)')
    ax.set_title('Top 10 Features with Missing Data')
else:
    ax.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

print("\n\nFeature Engineering and Advanced Analytics completed successfully!")
