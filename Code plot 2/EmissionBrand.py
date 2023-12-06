import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cdata = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/Germany_Used_Cars.csv', encoding='utf-8')

g_data_fuel = cdata[cdata['fuel_type']!="Electric"]
g_data_fuel.reset_index(drop=True,inplace=True)

mean_g = g_data_fuel.groupby('brand')['fuel_consumption_g_km'].mean().reset_index()
mean_g.sort_values('fuel_consumption_g_km',ascending=False,inplace=True)
mean_g.reset_index(drop=True,inplace=True)

fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(12,9),sharex=True)

sns.barplot(data=mean_g.head(10),  x='fuel_consumption_g_km',y='brand', palette='viridis',
           ax=axes[0,0])
axes[0,0].set_title('top 10 Brands')
axes[0,0].set_xlabel("Mean Emission (g/km)")
axes[0,0].set_ylabel('Brand Name')
axes[0,0].grid(axis='x')

sns.barplot(data=mean_g.loc[10:20],x='fuel_consumption_g_km',y='brand', palette='rocket',
           ax=axes[0,1])
axes[0,1].set_title('top 11-20 Brands')
axes[0,1].set_xlabel("Mean Emission (g/km)")
axes[0,1].set_ylabel('Brand Name')
axes[0,1].grid(axis='x')

sns.barplot(data=mean_g.loc[20:30],x='fuel_consumption_g_km', y='brand', palette='mako',
           ax=axes[1,0])
axes[1,0].set_title('top 21-30 Brands')
axes[1,0].set_xlabel("Mean Emission (g/km)")
axes[1,0].set_ylabel('Brand Name')
axes[1,0].grid(axis='x')

sns.barplot(data=mean_g.loc[30:],x='fuel_consumption_g_km', y='brand', palette='plasma',
           ax=axes[1,1])
axes[1,1].set_title('top 31-45 Brands')
axes[1,1].set_xlabel("Mean Emission (g/km)")
axes[1,1].set_ylabel('Brand Name')
axes[1,1].grid(axis='x')

fig.suptitle('The Mean Emission Comparision of Normal Fuel Cars of Brands ',
             fontweight='bold', fontsize=16)

plt.tight_layout(pad=2.0)
fig.patch.set_facecolor('#e6ccb3')
plt.show()