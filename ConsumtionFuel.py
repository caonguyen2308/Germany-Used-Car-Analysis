import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df3 = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/brandcode_final.csv', encoding='utf-8')        
fig8, axes8 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True)
cdata_normal_fuel = df3.loc[df3['fuel_type'] !='Electric'].copy()
mean_fuel = cdata_normal_fuel.groupby(['brand'])['fuel_consumption_l_100km'].mean().reset_index()
mean_fuel.sort_values('fuel_consumption_l_100km',ascending=False,inplace=True)
sns.barplot(data=mean_fuel.head(10), x='fuel_consumption_l_100km', y='brand', palette='viridis', ax=axes8[0, 0])
axes8[0, 0].set_title('Top 1-10 Brands')
axes8[0, 0].set_xlabel('Fuel Consumption (l/100km)')
axes8[0, 0].set_ylabel('Brand Name')
axes8[0, 0].grid(axis='x')


sns.barplot(data=mean_fuel.iloc[10:20], x='fuel_consumption_l_100km', y='brand', palette='rocket', ax=axes8[0, 1])
axes8[0, 1].set_title('Top 11-20 Brands')
axes8[0, 1].set_xlabel('Fuel Consumption (l/100km)')
axes8[0, 1].set_ylabel('Brand Name')
axes8[0, 1].grid(axis='x')


sns.barplot(data=mean_fuel.iloc[20:30], x='fuel_consumption_l_100km', y='brand', palette='mako', ax=axes8[1, 0])
axes8[1, 0].set_title('Top 21-30 Brands')
axes8[1, 0].set_xlabel('Fuel Consumption (l/100km)')
axes8[1, 0].set_ylabel('Brand Name')
axes8[1, 0].grid(axis='x')


sns.barplot(data=mean_fuel.iloc[30:], x='fuel_consumption_l_100km', y='brand', palette='magma', ax=axes8[1, 1])
axes8[1, 1].set_title('Top 31-45 Brands')
axes8[1, 1].set_xlabel('Fuel Consumption (l/100km)')
axes8[1, 1].set_ylabel('Brand Name')
axes8[1, 1].grid(axis='x')


fig8.suptitle('Mean Fuel Consumption per 100km Comparison of Brands', fontweight='bold', fontsize=16)

plt.tight_layout(pad=1.0)
fig8.patch.set_facecolor('#d3f0ff')
plt.show()