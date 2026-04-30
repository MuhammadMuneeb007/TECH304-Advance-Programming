# -*- coding: utf-8 -*-
"""
Week 07 - Example 9: Time Series Analysis
Demonstrates time series visualization, trend analysis, decomposition, and forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

print("=== Time Series Analysis ===\n")

# 1. Create Time Series Data
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
n_days = len(dates)

# Generate synthetic time series with trend, seasonality, and noise
trend = np.linspace(100, 150, n_days)
seasonality = 20 * np.sin(np.arange(n_days) * 2 * np.pi / 365)
noise = np.random.normal(0, 5, n_days)
values = trend + seasonality + noise

ts_data = pd.DataFrame({
    'date': dates,
    'value': values,
    'sales': 100 + 0.1 * np.arange(n_days) + 30 * np.sin(np.arange(n_days) * 2 * np.pi / 52) + np.random.normal(0, 3, n_days)
})
ts_data.set_index('date', inplace=True)

print("Time Series Data:")
print(ts_data.head(10))
print(f"\nData shape: {ts_data.shape}")

# 2. Basic Time Series Plots
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Full time series
axes[0].plot(ts_data.index, ts_data['value'], linewidth=1, color='steelblue')
axes[0].fill_between(ts_data.index, ts_data['value'], alpha=0.3, color='steelblue')
axes[0].set_title('Time Series: Value Over Time', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Value')
axes[0].grid(True, alpha=0.3)

# With moving averages
ma7 = ts_data['value'].rolling(window=7).mean()
ma30 = ts_data['value'].rolling(window=30).mean()
axes[1].plot(ts_data.index, ts_data['value'], label='Original', linewidth=1, alpha=0.7)
axes[1].plot(ts_data.index, ma7, label='7-day MA', linewidth=2, color='orange')
axes[1].plot(ts_data.index, ma30, label='30-day MA', linewidth=2, color='red')
axes[1].set_title('Time Series with Moving Averages', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Value')
axes[1].set_xlabel('Date')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. Seasonal Decomposition
print("\n\nSeasonal Decomposition")
print("=" * 40)

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts_data['value'], model='additive', period=365)
trend_comp = decomposition.trend
seasonal_comp = decomposition.seasonal
residual_comp = decomposition.resid

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

axes[0].plot(ts_data.index, ts_data['value'], linewidth=1, color='steelblue')
axes[0].set_ylabel('Original')
axes[0].set_title('Time Series Decomposition', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(trend_comp.index, trend_comp, linewidth=2, color='green')
axes[1].set_ylabel('Trend')
axes[1].grid(True, alpha=0.3)

axes[2].plot(seasonal_comp.index, seasonal_comp, linewidth=1, color='orange')
axes[2].set_ylabel('Seasonality')
axes[2].grid(True, alpha=0.3)

axes[3].plot(residual_comp.index, residual_comp, linewidth=0.5, color='red')
axes[3].set_ylabel('Residuals')
axes[3].set_xlabel('Date')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. Autocorrelation and Partial Autocorrelation
print("\n\nAutocorrelation Analysis")
print("=" * 40)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# ACF Plot
plot_acf(ts_data['value'].dropna(), lags=40, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')
axes[0].set_ylabel('ACF')

# PACF Plot
plot_pacf(ts_data['value'].dropna(), lags=40, ax=axes[1], method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF)')
axes[1].set_ylabel('PACF')

plt.tight_layout()
plt.show()

# 5. Weekly and Monthly Aggregation
print("\n\nTime Series Aggregation")
print("=" * 40)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Weekly aggregation
weekly_data = ts_data['value'].resample('W').mean()
axes[0].plot(weekly_data.index, weekly_data, marker='o', linewidth=2, markersize=5, color='steelblue')
axes[0].fill_between(weekly_data.index, weekly_data, alpha=0.3, color='steelblue')
axes[0].set_title('Weekly Aggregated Time Series')
axes[0].set_ylabel('Average Value')
axes[0].grid(True, alpha=0.3)

# Monthly aggregation
monthly_data = ts_data['value'].resample('M').agg(['mean', 'min', 'max'])
axes[1].plot(monthly_data.index, monthly_data['mean'], marker='o', linewidth=2, 
            label='Mean', color='steelblue')
axes[1].fill_between(monthly_data.index, monthly_data['min'], monthly_data['max'], 
                     alpha=0.3, color='steelblue', label='Min-Max Range')
axes[1].set_title('Monthly Aggregated Time Series with Range')
axes[1].set_ylabel('Value')
axes[1].set_xlabel('Date')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Year-over-Year Comparison
print("\n\nYear-over-Year Comparison")
print("=" * 40)

fig, ax = plt.subplots(figsize=(12, 6))

for year in [2022, 2023]:
    year_data = ts_data[ts_data.index.year == year]
    day_of_year = year_data.index.dayofyear
    ax.plot(day_of_year, year_data['value'], marker='o', markersize=3, 
           label=str(year), linewidth=2, alpha=0.7)

ax.set_title('Year-over-Year Comparison')
ax.set_xlabel('Day of Year')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 7. Distribution Analysis
print("\n\nDistribution Analysis")
print("=" * 40)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Histogram
axes[0].hist(ts_data['value'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Distribution of Values')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].axvline(ts_data['value'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0].legend()

# Box plot by month
monthly_ts = ts_data['value'].copy()
monthly_ts = monthly_ts.reset_index()
monthly_ts['month'] = monthly_ts['date'].dt.month
monthly_ts['month_name'] = monthly_ts['month'].map({
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
})

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_groups = [monthly_ts[monthly_ts['month_name'] == month]['value'].values for month in month_order]
bp = axes[1].boxplot(monthly_groups, labels=month_order)
axes[1].set_title('Distribution by Month')
axes[1].set_ylabel('Value')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

# Q-Q plot
from scipy import stats
stats.probplot(ts_data['value'], dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()

# 8. Change Point Detection
print("\n\nChange Point Analysis")
print("=" * 40)

# Simple trend change detection
window_size = 30
rolling_std = ts_data['value'].rolling(window=window_size).std()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ts_data.index, ts_data['value'], label='Original Series', linewidth=1, alpha=0.7)
ax.plot(rolling_std.index, rolling_std * 5, label='Rolling Std (x5)', linewidth=2, color='red')

# Find points where std increases
change_points = rolling_std[rolling_std > rolling_std.quantile(0.75)].index
for cp in change_points[::len(change_points)//5]:  # Show every 5th point for clarity
    ax.axvline(cp, color='orange', linestyle='--', alpha=0.3)

ax.set_title('Time Series with Rolling Standard Deviation')
ax.set_ylabel('Value')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nTime Series Analysis completed successfully!")
