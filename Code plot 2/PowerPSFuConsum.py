import ipywidgets as widgets
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
df3 = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/final.csv', encoding='utf-8')

# Calculate Correlation
selected_columns = ['power_ps', 'fuel_consumption_l_100km']
selected_data = df3[selected_columns]
correlation_matrix = selected_data.corr()
print(correlation_matrix)

@widgets.interact(aa=(0.01,0.1,0.005),bb=(0,5,0.5)) 
def Ve_Ham_So3(aa=0.001, bb=1, cc='r', show_line=False, text='Linear Regression'):
    xx = np.array([50,700])
    yy = xx*aa + bb
    sns.scatterplot(data=df3.sample(frac=0.3), x='power_ps', y='fuel_consumption_l_100km')
    plt.title('Scatter Plot of Power (PS) vs. Fuel Consumption (l/100km)')
    plt.xlabel('Power (PS)')
    plt.ylabel('Fuel Consumption (l/100km)')
    
    if show_line:
        xx = np.array([50,700])
        yy = xx*0.02 + 2.5
        sns.scatterplot(df3.sample(frac=0.3), x='power_ps', y='fuel_consumption_l_100km')
        plt.title('Scatter Plot of Power (PS) vs. Fuel Consumption (l/100km)\nLinear Regression: 0.02*x + 2.5')
        plt.xlabel('Power (PS)')
        plt.ylabel('Fuel Consumption (l/100km)')
            
        sns.lineplot(x=xx, y=yy, color='red')
        plt.show()
