import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
from sklearn.linear_model import LinearRegression

df = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/final.csv', encoding='ISO-8859-1')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

#3
# Create a DataFrame with the percentages
fuel_type_counts = df['fuel_type'].value_counts()
threshold = 500 
# Filter values greater and smaller than the threshold
larger_values = fuel_type_counts[fuel_type_counts >= threshold]
smaller_values = fuel_type_counts[fuel_type_counts < threshold]

# Create subplots for larger and smaller values
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for larger values
sns.histplot(data=df[df['fuel_type'].isin(larger_values.index)], x='fuel_type',
             hue='fuel_type', multiple='stack', palette='viridis', ax=axes[0])
axes[0].set_title('Distribution of Larger Values')

# Plot for smaller values
sns.histplot(data=df[df['fuel_type'].isin(smaller_values.index)], x='fuel_type',
             hue='fuel_type', multiple='stack', palette='magma', ax=axes[1])
axes[1].set_title('Distribution of Smaller Values')

plt.tight_layout()
plt.show()