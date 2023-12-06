import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df3 = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/brandcode_final.csv', encoding='utf-8')
brand_counts = df3['brand'].value_counts()
# Set the size of the chart
plt.figure(figsize=(12, 8))
sns.set_palette("pastel")
# Create a bar chart
brand_counts.plot(kind='bar', color='skyblue',width=0.8 )
plt.xlabel('Car Brands')
plt.ylabel('Car Sales')
plt.title('Car Sales by Brands')
plt.xticks(rotation=55, ha='right')
# Display the chart
plt.show()