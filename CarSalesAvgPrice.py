import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

df3 = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/brandcode_final.csv', encoding='utf-8')
# Group by brand and calculate count and average price
brand_stats = df3.groupby('brand')['price_in_euro'].agg(['count', 'mean'])

# Sort the DataFrame by count in descending order
brand_stats = brand_stats.sort_values(by='count', ascending=False)

# Set the size of the chart
fig, ax1 = plt.subplots(figsize=(20, 8))

# Plot count on the primary y-axis (bar chart)
x_positions = range(len(brand_stats))
ax1.bar(x_positions, brand_stats['count'], color='skyblue', width=0.8, label='Unit')

# Create a twin Axes sharing the x-axis
ax2 = ax1.twinx()

# Plot average price on the secondary y-axis (line chart)
ax2.plot(x_positions, brand_stats['mean'], marker='o', color='green', label='Average Price')

# Set labels and title
ax1.set_xlabel('Car Brands')
ax1.set_ylabel('Car Sales')
ax2.set_ylabel('Average Price')
plt.title('Car sales by brand and Average Prices')

#set style
plt.style.use('default')

ax1.grid(False)
ax2.grid(False)

# Show legend
ax1.legend(loc='upper left', bbox_to_anchor=(0.85, 0.90))
ax2.legend(loc='upper left', bbox_to_anchor=(0.85, 0.95))

# Set x-axis ticks and labels
ax1.set_xticks(x_positions)
ax1.set_xticklabels(brand_stats.index, rotation=55, ha='right')

sns.set_palette("pastel")

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the chart
plt.show()