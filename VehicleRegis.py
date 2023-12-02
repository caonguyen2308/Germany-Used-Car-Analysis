import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df3 = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/brandcode.csv', encoding='utf-8')
df3['registration_date'] = pd.to_datetime(df3['registration_date'], format='%m/%d/%Y %H:%M', errors='coerce')
# Extract year from 'registration_date'
df3['year'] = df3['registration_date'].dt.year
# Count the number of registrations for each year
registrations_by_year = df3['year'].value_counts().sort_index()
# Plot the count of registrations per year as a line chart
plt.figure(figsize=(10, 6))
plt.plot(registrations_by_year.index, registrations_by_year.values, marker='o', linestyle='-', color='b')
plt.title('Vehicle Registrations by Year')
plt.xlabel('Year')
plt.ylabel('Number of Registrations')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()