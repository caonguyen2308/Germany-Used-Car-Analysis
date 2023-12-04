import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df3 = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/brandcode_final.csv', encoding='utf-8') 
mean_charge = df3.groupby(['brand'])['charge_time_100km'].mean().reset_index()
mean_charge.sort_values('charge_time_100km',ascending=False,inplace=True)
mean_charge.reset_index(drop=True,inplace=True)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 7), sharex=True)

sns.barplot(data=mean_charge.head(7), x='charge_time_100km', y='brand', palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('Top 1-7 Brands')
axes[0, 0].set_xlabel('Electric Consumption (kW/100km)')
axes[0, 0].set_ylabel('Brand Name')
axes[0, 0].grid(axis='x')


sns.barplot(data=mean_charge.iloc[7:14], x='charge_time_100km', y='brand', palette='rocket', ax=axes[0, 1])
axes[0, 1].set_title('Top 8-14 Brands')
axes[0, 1].set_xlabel('Eletric Consumption (kW/100km)')
axes[0, 1].set_ylabel('Brand Name')
axes[0, 1].grid(axis='x')

sns.barplot(data=mean_charge.iloc[14:21], x='charge_time_100km', y='brand', palette='mako', ax=axes[1, 0])
axes[1, 0].set_title('Top 15-21 Brands')
axes[1, 0].set_xlabel('Electric Consumption (kW/100km)')
axes[1, 0].set_ylabel('Brand Name')
axes[1, 0].grid(axis='x')

sns.barplot(data=mean_charge.iloc[21:], x='charge_time_100km', y='brand', palette='magma', ax=axes[1, 1])
axes[1, 1].set_title('Top 22-27 Brands')
axes[1, 1].set_xlabel('Eletric Consumption (kW/100km)')
axes[1, 1].set_ylabel('Brand Name')
axes[1, 1].grid(axis='x')

fig.suptitle('Mean Electric Consumption per 100km Comparison of Brands', fontweight='bold', fontsize=16,color='yellow')
plt.tight_layout(pad=2.0)
fig.patch.set_facecolor('#87CEEB')
plt.savefig('Electric_consumption', bbox_inches='tight')
plt.show()