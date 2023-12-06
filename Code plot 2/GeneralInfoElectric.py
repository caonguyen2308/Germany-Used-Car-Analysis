import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True, precision=True)

df3 = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/Germany_Used_Cars.csv', encoding='utf-8')
cdata_electric = df3.loc[df3['fuel_type'] == 'Electric'].copy()
cdata_electric.rename(columns={'fuel_consumption_l_100km':'charge_time_100km'},inplace=True)
cdata_electric.reset_index(drop=True,inplace=True)
cdata_normal_fuel = df3.loc[df3['fuel_type'] != 'Electric'].copy()
fig, ax =plt.subplots(figsize=(10,7),num=1)
ax.hist(cdata_electric['charge_time_100km'], bins = 100, color='y',edgecolor='black')

ax.axvline(344.555, color='blue',linestyle=':',linewidth=2)
ax.text(344.555 +10, 180, f"Mean:\n{344.555:.2f}", color="b")

ax.axvline(344,color='green', linestyle='--',linewidth=2)
ax.text(344-60,170 ,f"Median:\n{344:.2f}",color='green')

ax.axvline(201, color='violet',linestyle='--',linewidth=2)
ax.text(201-60,170,f"Mode:\n{201:0.2f}",color='violet')

ax.set_xlabel("Electric consumption (kW/100km)")

plt.title("General information of Electric consumption",fontweight='bold')
plt.grid()
plt.show()