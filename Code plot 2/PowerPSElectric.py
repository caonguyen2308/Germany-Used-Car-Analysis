import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cdata = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/final.csv', encoding='utf-8')
cdata_electric = cdata.loc[cdata['fuel_type'] == 'Electric'].copy()
cdata_electric.rename(columns={'fuel_consumption_l_100km':'charge_time_100km'},inplace=True)
cdata_electric.reset_index(drop=True,inplace=True)


cdata_normal_fuel = cdata.loc[cdata['fuel_type'] != 'Electric'].copy()
selected_columns = ['power_ps', 'charge_time_100km']
selected_data = cdata_electric[selected_columns]
correlation_matrix = selected_data.corr()
print(correlation_matrix)
sns.scatterplot(data=cdata_electric, x='power_ps', y='charge_time_100km')
plt.title('Scatter Plot of Power (PS) vs. Electric Consumption (kW/100km)')
plt.xlabel('Power (PS)')
plt.ylabel('Electric Consumption (kW/100km)')
plt.show()
plt.close('all')