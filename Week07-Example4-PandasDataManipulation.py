# -*- coding: utf-8 -*-
"""
Week 07 - Example 4: Pandas Data Manipulation and Analysis
Demonstrates data loading, cleaning, transformation, and basic analysis with pandas.
"""

import pandas as pd
import numpy as np

print("=== Pandas Data Manipulation and Analysis ===\n")

# 1. Create Sample Dataset
np.random.seed(42)
data = {
    'employee_id': range(1, 51),
    'name': [f'Employee_{i}' for i in range(1, 51)],
    'department': np.random.choice(['Sales', 'Engineering', 'HR', 'Finance'], 50),
    'salary': np.random.randint(40000, 120000, 50),
    'experience_years': np.random.randint(1, 20, 50),
    'performance_score': np.random.uniform(2.0, 5.0, 50),
    'hire_date': pd.date_range('2015-01-01', periods=50, freq='M')
}
df = pd.DataFrame(data)

print("Dataset Overview:")
print(df.head(10))
print(f"\nDataset Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")

# 2. Data Cleaning
print("\n\n=== Data Cleaning ===")
print(f"Missing Values:\n{df.isnull().sum()}")

# Remove duplicates
df_clean = df.drop_duplicates()
print(f"Duplicates Removed: {len(df) - len(df_clean)} rows")

# Handle outliers using IQR method
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['salary'] < Q1 - 1.5 * IQR) | (df['salary'] > Q3 + 1.5 * IQR)]
print(f"Outliers Found in Salary: {len(outliers)} records")

# 3. Data Transformation
print("\n\n=== Data Transformation ===")

# Add new columns
df['salary_category'] = pd.cut(df['salary'], 
                               bins=[0, 50000, 75000, 100000, 150000],
                               labels=['Low', 'Medium', 'High', 'Very High'])

df['experience_level'] = pd.cut(df['experience_years'],
                                bins=[0, 5, 10, 15, 20],
                                labels=['Junior', 'Mid', 'Senior', 'Expert'])

df['hire_year'] = df['hire_date'].dt.year
df['hire_month'] = df['hire_date'].dt.month

print("New Columns Added:")
print(df[['salary', 'salary_category', 'experience_years', 'experience_level']].head())

# 4. Descriptive Statistics
print("\n\n=== Descriptive Statistics ===")
print("\nNumerical Summary by Department:")
print(df.groupby('department')[['salary', 'experience_years', 'performance_score']].agg([
    'count', 'mean', 'std', 'min', 'max'
]).round(2))

print("\n\nSalary Statistics by Department:")
dept_stats = df.groupby('department')['salary'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
]).round(2)
print(dept_stats)

# 5. Grouping and Aggregation
print("\n\n=== Grouping and Aggregation ===")

# Multiple aggregations
agg_dict = {
    'salary': ['count', 'mean', 'sum'],
    'experience_years': ['mean', 'max'],
    'performance_score': ['mean', 'std']
}
print("\nAggregation by Department:")
print(df.groupby('department').agg(agg_dict).round(2))

# 6. Pivot Tables
print("\n\n=== Pivot Tables ===")
pivot_table = df.pivot_table(
    values='salary',
    index='department',
    columns='salary_category',
    aggfunc='count',
    fill_value=0
)
print("\nEmployee Count by Department and Salary Category:")
print(pivot_table)

# 7. Sorting and Filtering
print("\n\n=== Sorting and Filtering ===")
print("\nTop 5 Employees by Salary:")
print(df.nlargest(5, 'salary')[['name', 'department', 'salary', 'experience_years']])

print("\nEmployees with High Performance (>4.0) and High Experience (>10 years):")
high_performers = df[(df['performance_score'] > 4.0) & (df['experience_years'] > 10)]
print(high_performers[['name', 'department', 'performance_score', 'experience_years']])

# 8. String Operations
print("\n\n=== String Operations ===")
df['name_upper'] = df['name'].str.upper()
df['dept_abbrev'] = df['department'].str[:3].str.upper()
print("\nString Operations:")
print(df[['name', 'name_upper', 'department', 'dept_abbrev']].head())

# 9. Missing Value Imputation (simulating missing data)
print("\n\n=== Missing Value Handling ===")
df_missing = df.copy()
df_missing.loc[df_missing.sample(5, random_state=42).index, 'performance_score'] = np.nan

print(f"Missing Values: {df_missing.isnull().sum().sum()}")
print("Before Imputation:")
print(df_missing[['name', 'performance_score']].tail())

# Fill with mean
df_missing['performance_score'].fillna(df_missing['performance_score'].mean(), inplace=True)
print("\nAfter Imputation (with mean):")
print(df_missing[['name', 'performance_score']].tail())

# 10. Correlation and Relationships
print("\n\n=== Correlation Analysis ===")
numeric_data = df[['salary', 'experience_years', 'performance_score']]
print("\nCorrelation Matrix:")
print(numeric_data.corr().round(3))
