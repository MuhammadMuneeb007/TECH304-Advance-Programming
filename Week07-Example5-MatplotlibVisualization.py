# -*- coding: utf-8 -*-
"""
Week 07 - Example 5: Matplotlib Visualization Techniques
Demonstrates various plotting techniques and customizations using matplotlib.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

print("=== Matplotlib Visualization Techniques ===\n")

# Create sample data
np.random.seed(42)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sales = np.random.randint(10000, 50000, 12)
expenses = np.random.randint(5000, 25000, 12)
profit = sales - expenses

# 1. Basic Line Plot with Multiple Lines
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(months, sales, marker='o', linewidth=2, label='Sales', color='green')
ax.plot(months, expenses, marker='s', linewidth=2, label='Expenses', color='red')
ax.plot(months, profit, marker='^', linewidth=2, label='Profit', color='blue')

ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Amount ($)', fontsize=12, fontweight='bold')
ax.set_title('Monthly Sales, Expenses, and Profit', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Subplots with Different Chart Types
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Bar Chart
axes[0, 0].bar(months, sales, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Sales by Month (Bar Chart)')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Stacked Bar Chart
axes[0, 1].bar(months, expenses, label='Expenses', color='coral', alpha=0.7)
axes[0, 1].bar(months, profit, bottom=expenses, label='Profit', color='lightgreen', alpha=0.7)
axes[0, 1].set_title('Revenue Breakdown (Stacked Bar)')
axes[0, 1].set_ylabel('Amount ($)')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)

# Area Chart
axes[1, 0].fill_between(range(len(months)), sales, alpha=0.5, color='purple', label='Sales')
axes[1, 0].fill_between(range(len(months)), expenses, alpha=0.5, color='orange', label='Expenses')
axes[1, 0].set_title('Sales Trends (Area Chart)')
axes[1, 0].set_ylabel('Amount ($)')
axes[1, 0].set_xticks(range(len(months)))
axes[1, 0].set_xticklabels(months, rotation=45)
axes[1, 0].legend()

# Scatter with Size and Color
categories = np.random.choice([0, 1, 2], 12)
sizes = profit / 100
colors_scatter = ['red' if p < 20000 else 'yellow' if p < 30000 else 'green' for p in profit]
scatter = axes[1, 1].scatter(range(len(months)), profit, s=sizes, c=profit, 
                            cmap='RdYlGn', alpha=0.6, edgecolors='black')
axes[1, 1].set_title('Profit Trend (Scatter Plot)')
axes[1, 1].set_ylabel('Profit ($)')
axes[1, 1].set_xticks(range(len(months)))
axes[1, 1].set_xticklabels(months, rotation=45)
plt.colorbar(scatter, ax=axes[1, 1], label='Profit ($)')

plt.tight_layout()
plt.show()

# 3. Pie Chart with Explode
categories = ['Product A', 'Product B', 'Product C', 'Product D']
sales_by_product = [15000, 25000, 18000, 22000]
colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
explode = (0.05, 0.05, 0, 0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie Chart
wedges, texts, autotexts = axes[0].pie(sales_by_product, labels=categories, autopct='%1.1f%%',
                                        colors=colors_pie, explode=explode, startangle=90,
                                        textprops={'fontsize': 10})
axes[0].set_title('Sales Distribution by Product', fontweight='bold', fontsize=12)

# Make percentage text bold and white
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# Donut Chart
axes[1].pie(sales_by_product, labels=categories, autopct='%1.1f%%',
            colors=colors_pie, startangle=90, textprops={'fontsize': 10})
circle = plt.Circle((0, 0), 0.70, fc='white')
axes[1].add_artist(circle)
axes[1].set_title('Sales Distribution (Donut Chart)', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.show()

# 4. Histograms and Box Plots
data_dist1 = np.random.normal(100, 15, 1000)
data_dist2 = np.random.normal(110, 20, 1000)
data_dist3 = np.random.normal(95, 10, 1000)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
axes[0, 0].hist(data_dist1, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution 1 (Histogram)')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(np.mean(data_dist1), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data_dist1):.1f}')
axes[0, 0].legend()

# Overlapping Histograms
axes[0, 1].hist(data_dist1, bins=20, alpha=0.5, label='Distribution 1', color='blue')
axes[0, 1].hist(data_dist2, bins=20, alpha=0.5, label='Distribution 2', color='red')
axes[0, 1].set_title('Overlapping Distributions')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

# Box Plot
box_data = [data_dist1, data_dist2, data_dist3]
bp = axes[1, 0].boxplot(box_data, labels=['Dist1', 'Dist2', 'Dist3'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
    patch.set_facecolor(color)
axes[1, 0].set_title('Comparison of Distributions (Box Plot)')
axes[1, 0].set_ylabel('Value')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Violin Plot
parts = axes[1, 1].violinplot(box_data, positions=[1, 2, 3], showmeans=True, showmedians=True)
axes[1, 1].set_title('Comparison of Distributions (Violin Plot)')
axes[1, 1].set_xticks([1, 2, 3])
axes[1, 1].set_xticklabels(['Dist1', 'Dist2', 'Dist3'])
axes[1, 1].set_ylabel('Value')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 5. Advanced: Heatmap
data_matrix = np.random.randint(10, 100, (5, 6))
fig, ax = plt.subplots(figsize=(10, 6))

im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')

ax.set_xticks(np.arange(6))
ax.set_yticks(np.arange(5))
ax.set_xticklabels([f'Week {i+1}' for i in range(6)])
ax.set_yticklabels([f'Product {i+1}' for i in range(5)])

# Add text annotations
for i in range(5):
    for j in range(6):
        text = ax.text(j, i, data_matrix[i, j], ha="center", va="center", color="black", fontweight='bold')

ax.set_title('Sales Heatmap (Product vs Week)')
plt.colorbar(im, ax=ax, label='Sales ($)')
plt.tight_layout()
plt.show()

print("\nAll visualization examples completed successfully!")
