import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df3 = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/brandcode_final.csv', encoding='utf-8')

mean_power_ps_brand = df3.groupby('brand')['power_ps'].mean().reset_index()
mean_power_ps_brand.sort_values('power_ps',ascending=False,inplace=True)
mean_power_ps_brand.reset_index(drop=True,inplace=True)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True)

sns.barplot(data=mean_power_ps_brand.head(10), x='power_ps', y='brand', palette='viridis', ax=axes[0,0])
axes[0, 0].set_title('Top 1-10 Brands')
axes[0, 0].set_xlabel('power_ps')
axes[0, 0].set_ylabel('Brand Name')
axes[0, 0].grid(axis='x')


sns.barplot(data=mean_power_ps_brand.iloc[10:20], x='power_ps', y='brand', palette='rocket', ax=axes[0, 1])
axes[0, 1].set_title('Top 11-20 Brands')
axes[0, 1].set_xlabel('power_ps')
axes[0, 1].set_ylabel('Brand Name')
axes[0, 1].grid(axis='x')


sns.barplot(data=mean_power_ps_brand.iloc[20:30], x='power_ps', y='brand', palette='mako', ax=axes[1, 0])
axes[1, 0].set_title('Top 21-30 Brands')
axes[1, 0].set_xlabel('power_ps')
axes[1, 0].set_ylabel('Brand Name')
axes[1, 0].grid(axis='x')


sns.barplot(data=mean_power_ps_brand.iloc[30:], x='power_ps', y='brand', palette='magma', ax=axes[1, 1])
axes[1, 1].set_title('Top 31-45 Brands')
axes[1, 1].set_xlabel('power_ps')
axes[1, 1].set_ylabel('Brand Name')
axes[1, 1].grid(axis='x')


fig.suptitle('power_ps_comparision', fontweight='bold', fontsize=16)

plt.tight_layout(pad=1.0)
fig.patch.set_facecolor('#d3f0ff')
plt.show()