import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cdata = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/brandcode_final.csv', encoding='utf-8')
cdata_normal_fuel = cdata[cdata['fuel_type']!='electric']
cdata_normal_fuel['price_in_euro'].describe()
cdata_normal_fuel['fuel_consumption_l_100km'].describe()

mean_price_brand = cdata_normal_fuel.groupby('brand')['price_in_euro'].mean().reset_index()
mean_price_brand.sort_values('price_in_euro',ascending=False,inplace=True)
mean_price_brand.reset_index(drop=True,inplace=True)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), sharex=True)

sns.barplot(data=mean_price_brand.head(10), x='price_in_euro', y='brand', palette='viridis', ax=axes[0])
axes[0].set_title('Top 1-10 Brands')
axes[0].set_xlabel('Price')
axes[0].set_ylabel('Brand Name')
axes[0].grid(axis='x')

sns.barplot(data=mean_price_brand.iloc[10:20], x='price_in_euro', y='brand', palette='mako', ax=axes[1])
axes[1].set_title('Top 11-20 Brands')
axes[1].set_xlabel('Price')
axes[1].set_ylabel('Brand Name')
axes[1].grid(axis='x')


fig.suptitle('top 20 brands price comparision', fontweight='bold', fontsize=16)

plt.tight_layout(pad=1.0)
fig.patch.set_facecolor('#e6e6e6')
plt.savefig('price_top20_normal', bbox_inches='tight')
plt.show()