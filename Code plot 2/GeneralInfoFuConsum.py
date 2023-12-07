import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True, precision=True)

df3 = cdata = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/final.csv', encoding='utf-8')
cdata_normal_fuel = cdata.loc[cdata['fuel_type'] != 'Electric'].copy()
fig, ax =plt.subplots(figsize=(10,7),num=1)
ax.hist(cdata_normal_fuel['fuel_consumption_l_100km'], bins = 100, color='r',edgecolor='black')

ax.axvline(6.055, color='blue',linestyle=':',linewidth=2)
ax.text(6.055+0.2, 13000, f"Mean:\n{6.055:.2f}", color="b")

ax.axvline(5.7,color='green', linestyle='--',linewidth=2)
ax.text(5.7-2.5,13000 ,f"Median:\n{5.7:.2f}",color='green')

ax.axvline(5.25, color='violet',linestyle='--',linewidth=2)
ax.text(5.35-4,13000,f"Mode:\n{5.25:0.2f}",color='violet')

ax.set_xlabel("Fuel consumption (litter/100km)")

plt.title("General information of Fuel Consumption",fontweight='bold')
plt.grid()
plt.show()