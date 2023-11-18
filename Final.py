import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import ipywidgets as widgets
import os
import warnings
import re

df = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/data.csv', encoding='ISO-8859-1')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# Clean data
# fillna values by mean (type int only)
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Convert registration_date to DateTime format
df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
df['registration_date'] = df['registration_date'].dt.strftime('%Y-%m-%d')

# Delete Duplicate 
df = df.drop_duplicates()


# Clean column with has (l/km) and (g/km)
def clean_and_convert(value):
    if isinstance(value, str) and not value.isdigit():
        numeric_value = ''.join(filter(str.isdigit, value))
        return int(numeric_value) if numeric_value else None
    elif isinstance(value, str) and value.isdigit():
        return int(value)
    else:
        return value
def clean_and_convert(value):
    if isinstance(value, str):
        numeric_part = ''.join(filter(lambda x: x.isdigit() or x in ['.', ','], value))
        numeric_part = numeric_part.replace(',', '.')  
        try:
            float_value = float(numeric_part)
            rounded_value = round(float_value, 1)  
            return rounded_value
        except ValueError:
            return None
    else:
        return value

# Apply the cleaning function to the column
df['fuel_consumption_g_km'] = df['fuel_consumption_g_km'].apply(clean_and_convert)
df['fuel_consumption_l_100km'] = df['fuel_consumption_l_100km'].apply(clean_and_convert)

# Fill NaN value by the mean of column fuel_consumption_g_km
df['fuel_consumption_g_km'] = pd.to_numeric(df['fuel_consumption_g_km'], errors='coerce') 
mean_fuel_consumption_g_km = df['fuel_consumption_g_km'].mean()  
df['fuel_consumption_g_km'].fillna(mean_fuel_consumption_g_km, inplace=True)

# Fill NaN value by the mean of column fuel_consumption_l/km
df['fuel_consumption_l_100km'] = pd.to_numeric(df['fuel_consumption_l_100km'], errors='coerce') 
mean_fuel_consumption_l_km = df['fuel_consumption_l_100km'].mean()  
df['fuel_consumption_l_100km'].fillna(mean_fuel_consumption_l_km, inplace=True)
df = df[df['fuel_consumption_l_100km'] <= 20]

# Print column
print(df['fuel_consumption_g_km'])
print(df['fuel_consumption_l_100km'])

# Save new file csv
new_cleaned_data = df.to_csv('D:/Cybersoft/Germany-Used-Car-Analysis/final_cleaned_data.csv', index = True) 
print('\nnew_cleaned_data\n', new_cleaned_data) 





