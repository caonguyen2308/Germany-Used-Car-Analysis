import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df3 = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/final.csv', encoding='utf-8')
fuel_type,count = np.unique(df3['fuel_type'],return_counts=True)
labels=list(fuel_type)
data = list(count)
fig,ax =plt.subplots(figsize=(8,8),num=1)
ax.pie(data,wedgeprops={'width':0.5},
      startangle=90,labels=labels
)
ax.legend(loc='center',fontsize=11)
plt.show()