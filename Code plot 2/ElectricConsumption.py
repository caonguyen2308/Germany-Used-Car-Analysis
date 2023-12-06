import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df3 = cdata = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/Germany_Used_Cars.csv', encoding='utf-8')  
cdata_electric = cdata.loc[cdata['fuel_type'] == 'Electric'].copy()
cdata_electric.rename(columns={'fuel_consumption_l_100km':'charge_time_100km'},inplace=True)
cdata_electric.reset_index(drop=True,inplace=True)
cdata_normal_fuel = cdata.loc[cdata['fuel_type'] != 'Electric'].copy()   
fig9, axes9 = plt.subplots(nrows=2, ncols=2, figsize=(11, 7), sharex=True)
cdata_electric = df3.loc[df3['fuel_type'] == 'Electric'].copy()
cdata_electric.reset_index(drop=True)
cdata_electric.rename(columns={'fuel_consumption_l_100km':'charge_time_100km'},inplace=True)
cdata_electric.reset_index(drop=True,inplace=True)
mean_charge = cdata_electric.groupby(['brand'])['charge_time_100km'].mean().reset_index()
mean_charge.sort_values('charge_time_100km',ascending=False,inplace=True)
mean_charge.reset_index(drop=True,inplace=True)
sns.barplot(data=mean_charge.head(7), x='charge_time_100km', y='brand', palette='viridis', ax=axes9[0, 0])
axes9[0, 0].set_title('Top 1-7 Brands')
axes9[0, 0].set_xlabel('Electric Consumption (charge/100km)')
axes9[0, 0].set_ylabel('Brand Name')
axes9[0, 0].grid(axis='x')


sns.barplot(data=mean_charge.iloc[7:14], x='charge_time_100km', y='brand', palette='rocket', ax=axes9[0, 1])
axes9[0, 1].set_title('Top 8-14 Brands')
axes9[0, 1].set_xlabel('Eletric Consumption (charges/100km)')
axes9[0, 1].set_ylabel('Brand Name')
axes9[0, 1].grid(axis='x')

sns.barplot(data=mean_charge.iloc[14:21], x='charge_time_100km', y='brand', palette='mako', ax=axes9[1, 0])
axes9[1, 0].set_title('Top 15-21 Brands')
axes9[1, 0].set_xlabel('Electric Consumption (charges/100km)')
axes9[1, 0].set_ylabel('Brand Name')
axes9[1, 0].grid(axis='x')

sns.barplot(data=mean_charge.iloc[21:], x='charge_time_100km', y='brand', palette='magma', ax=axes9[1, 1])
axes9[1, 1].set_title('Top 22-27 Brands')
axes9[1, 1].set_xlabel('Eletric Consumption (charges/100km)')
axes9[1, 1].set_ylabel('Brand Name')
axes9[1, 1].grid(axis='x')

fig9.suptitle('Mean Electric Consumption per 100km Comparison of Brands', fontweight='bold', fontsize=16,color='yellow')
plt.tight_layout(pad=2.0)
fig9.patch.set_facecolor('#87CEEB')
plt.show()