import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import ipywidgets as widgets
import re

df = pd.read_csv('D:/Cybersoft/Final-Test-Cybersoft/data.csv', encoding='ISO-8859-1')

# Clean data
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
# Convert 'registration_date' to DateTime format
df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
df['registration_date'] = df['registration_date'].dt.strftime('%Y-%m-%d')
a = df.isnull().sum()
#print(a)
df.to_csv('D:/Cybersoft/Final-Test-Cybersoft/final_cleaned_data.csv', index=False)
print (df)
