import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import ipywidgets as widgets
import os
import warnings

df = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/data.csv', encoding='ISO-8859-1')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
# Clean data
# fillna values by mean (type int only)
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Convert registration_date to DateTime format
df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
df['registration_date'] = df['registration_date'].dt.strftime('%Y-%m-%d')

# Save new file csv
output_directory = 'D:/Cybersoft/Final-Test-Cybersoft'
os.makedirs(output_directory, exist_ok=True)
output_file_path = os.path.join(output_directory, 'final_cleaned_data.csv')
df.to_csv(output_file_path, index=False)

print(df)




