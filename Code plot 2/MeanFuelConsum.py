import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df3 = cdata = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/final.csv', encoding='utf-8') 
cdata_electric = cdata.loc[cdata['fuel_type'] == 'Electric'].copy()
cdata_electric.rename(columns={'fuel_consumption_l_100km':'charge_time_100km'},inplace=True)
cdata_electric.reset_index(drop=True,inplace=True)
cdata_normal_fuel = cdata.loc[cdata['fuel_type'] != 'Electric'].copy()
fuelt_fuelc = cdata_normal_fuel.groupby('fuel_type')['fuel_consumption_l_100km'].mean().reset_index()
fig, ax = plt.subplots(figsize=(11, 7))

sns.barplot(data=fuelt_fuelc, x='fuel_type', y='fuel_consumption_l_100km', ax=ax, palette= 'tab10')

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.title("Mean fuel consumption of each fuel type (litter per 100km)",fontweight='bold')
plt.xlabel("Name of Fuel Type")
plt.ylabel("Mean fuel consumption (litter/100km)")
plt.show()