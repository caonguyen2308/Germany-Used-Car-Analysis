import pandas as pd
import numpy as np
import warnings
import re

df = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/data.csv', encoding='ISO-8859-1')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
# Clean data
# fillna values by mean 
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
filtered_consumption_l_km_df = df[df['fuel_consumption_l_100km'] <= 20]
filtered_consumption_l_km_df.to_csv('D:/Cybersoft/Germany-Used-Car-Analysis/newdata.csv', index=True)
# Print column
#print(df['fuel_consumption_g_km'])
#print(df['fuel_consumption_l_100km'])

# Fill NaN values by the mean of column price_in_euro
df['price_in_euro'] = pd.to_numeric(df['price_in_euro'], errors='coerce') 
mean_price_euro = df['price_in_euro'].mean()  
df['price_in_euro'].fillna(mean_price_euro, inplace=True)

# Fill NaN values for fuel type
df['fuel_type'].fillna('', inplace=True)
valid_fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'LPG', 'CNG', 'Diesel Hybrid', 'Electric', 'Ethanol', 'Hydrogen', 'Unknown']

# Filter the DataFrame to include only rows with valid fuel types
df_cleaned_fuel_types = df[df['fuel_type'].isin(valid_fuel_types)]

distinct_valid_fuel_types = df_cleaned_fuel_types['fuel_type'].unique()
non_fuel_type_count = df[~df['fuel_type'].isin(valid_fuel_types)].shape[0]
df = df[df['fuel_type'].isin(valid_fuel_types)]
df.reset_index(drop=True, inplace=True)
df_electric = df[df['fuel_type'] == 'Electric']

# Reset the index of the new DataFrame
df_electric.reset_index(drop=True, inplace=True)

# Drop the 'fuel_consumption_l_100km' column from df_electric
df_electric = df_electric.drop(columns=['fuel_consumption_l_100km'])

# Display the first few rows of the new DataFrame
df_electric['distance_per_charge'] = df_electric['fuel_consumption_g_km'].str.extract('(\d+)').astype(float)

# Fill NaN values for color
df['color'].fillna('', inplace=True)

print ([df['fuel_type']], df['fuel_consumption_g_km'], df['fuel_consumption_l_100km'])

# Fill NaN values for transmission
df['transmission_type'].fillna('', inplace=True)
# Save new file csv
#df.to_csv('D:/Cybersoft/Germany-Used-Car-Analysis/brandcode_final.csv', index=True)
#print('File replaced successfully!')





