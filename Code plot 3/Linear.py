import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
from sklearn.linear_model import LinearRegression

df = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/final.csv', encoding='ISO-8859-1')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")


#4 Linear regression
X = df[['year']] 
y = df['price_in_euro']  

# Calculate the mean prices in each year
average_prices = df.groupby('year')['price_in_euro'].mean().reset_index()
#print(average_prices)

# Calculate the mean percentages
average_prices['percentage_change'] = average_prices['price_in_euro'].pct_change()
mean_percentage_change = average_prices['percentage_change'].mean()
#print(mean_percentage_change)

model = LinearRegression()

# Fit the model
model.fit(X, y)
# Predict prices for the next 5 years
future_years = pd.DataFrame({'year': range(2023, 2028)})
predicted_prices = model.predict(future_years)
# Add the predicted prices to the DataFrame
future_years['predicted_price'] = predicted_prices
plt.figure(figsize=(8, 6))
plt.scatter(average_prices['year'], average_prices['price_in_euro'], color='blue', label='Historical Data')

# Plotting the regression line
plt.plot(X, model.predict(X), color='green', linestyle='-', label='Regression Line')

# Plotting the predicted prices for the next 5 years
plt.scatter(future_years['year'], predicted_prices, color='red', label='Predicted Prices')

plt.xlabel('Year')
plt.ylabel('Price in Euro')
plt.title('Historical and Predicted Prices')
plt.legend()
plt.grid(True)
plt.show()