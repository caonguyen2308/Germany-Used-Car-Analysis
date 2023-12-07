import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df3 = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/final.csv', encoding='utf-8')
top_10_models_with_prices = df3.groupby('model')['price_in_euro'].agg(['count', 'mean']).reset_index()
top_10_models_with_prices = top_10_models_with_prices.nlargest(10, 'count')

# Set the Seaborn theme
sns.set_theme()
# Set the size of the chart
fig, ax1 = plt.subplots(figsize=(12, 8))

# Bar chart for the number of car sales
sns.barplot(x='model', y='count', data=top_10_models_with_prices, color='skyblue', ax=ax1)
# Set labels and title for the bar chart
ax1.set_xlabel('Model')
ax1.set_ylabel('Car Sales', color='orange')
ax1.set_title('Top 10 Best-Selling Car Models with Average Prices')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.grid(visible=False)

# Create a second y-axis for the line chart
ax2 = ax1.twinx()

# Line chart for average prices
sns.lineplot(x='model', y='mean', data=top_10_models_with_prices, color='orange', marker='s', ax=ax2)
# Set labels and title for the line chart
ax2.set_ylabel('Average Price', color='orange')
ax2.grid(visible=False)
plt.show()