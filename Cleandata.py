import pandas as pd
import numpy as np
import warnings
import re

# Read the CSV file
df = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/data.csv', encoding='ISO-8859-1')

# Suppress warnings temporarily
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# Clean data: fill NaN values by mean
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Convert 'registration_date' to DateTime format
df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
df['registration_date'] = pd.to_datetime(df['registration_date'], format='%m/%d/%Y%H:%M', errors='coerce')
    
    
# Delete duplicates
df = df.drop_duplicates()

# Define function to clean and convert fuel consumption values
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

# Apply cleaning function to fuel consumption columns
df['fuel_consumption_g_km'] = df['fuel_consumption_g_km'].apply(clean_and_convert)
df['fuel_consumption_l_100km'] = df['fuel_consumption_l_100km'].apply(clean_and_convert)

# Fill NaN values by the mean of respective columns
df['fuel_consumption_g_km'].fillna(df['fuel_consumption_g_km'].mean(), inplace=True)
df['fuel_consumption_l_100km'].fillna(df['fuel_consumption_l_100km'].mean(), inplace=True)
df = df[df['fuel_consumption_l_100km'] <= 20]
   
    
# Fill NaN values for other columns
df['price_in_euro'] = pd.to_numeric(df['price_in_euro'], errors='coerce')
df['price_in_euro'].fillna(df['price_in_euro'].mean(), inplace=True)
df['fuel_type'].fillna('', inplace=True)
df['color'].fillna('', inplace=True)
df['transmission_type'].fillna('', inplace=True)
df['power_kw'] = pd.to_numeric(df['power_kw'], errors='coerce')

# Fill NaN values in 'power_kw' column with the mean
df['power_kw'].fillna(df['power_kw'].mean(), inplace=True)

# Filter valid fuel types
valid_fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'LPG', 'CNG', 'Diesel Hybrid', 'Electric', 'Ethanol', 'Hydrogen', 'Unknown']
df = df[df['fuel_type'].isin(valid_fuel_types)]

# Separate electric cars into a new DataFrame and drop 'fuel_consumption_l_100km' only if NaN
df_electric = df[df['fuel_type'] == 'Electric']
df_electric = df_electric.dropna(subset=['fuel_consumption_l_100km'])

# Reset DataFrame indices
df.reset_index(drop=True, inplace=True)
df_electric.reset_index(drop=True, inplace=True)

# Filter the DataFrame to include only rows with valid fuel types
df_cleaned_fuel_types = df[df['fuel_type'].isin(valid_fuel_types)]

distinct_valid_fuel_types = df_cleaned_fuel_types['fuel_type'].unique()
non_fuel_type_count = df[~df['fuel_type'].isin(valid_fuel_types)].shape[0]
df = df[df['fuel_type'].isin(valid_fuel_types)]
df.reset_index(drop=True, inplace=True)
df_electric = df[df['fuel_type'] == 'Electric']

df.drop('offer_description', axis=1, inplace=True)
# Save cleaned DataFrame to a new CSV file
df.to_csv('D:/Cybersoft/Germany-Used-Car-Analysis/final.csv', index=True)
print('File replaced successfully!')
#df.info()








