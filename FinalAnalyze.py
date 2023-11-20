import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
import plotly.express as px
from sklearn.linear_model import LinearRegression

df = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/brandcode.csv', encoding='ISO-8859-1')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# Topic: Based on the mileage, power_km, fuel_type find out the prediction of prices in different types of cars

# 1: Draw a chart following brand and price
brands = df['code']
prices = df['price_in_euro']

# Create the bar chart
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
sns.set_style('darkgrid')
sns.barplot(x=prices, y=brands, color='skyblue')  # Using Seaborn's barplot function with x and y switched for horizontal view
plt.xlabel('Price')
plt.ylabel('Brand Codename')
plt.title('Price Distribution by Brand')
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()

# Price following the power_kw
prices = df['price_in_euro']
power_km = df['power_kw']
    
plt.figure(figsize=(8, 6))
plt.scatter(power_km, prices, alpha=0.5, color = 'salmon')
plt.xlabel('Power KW')
plt.ylabel('Price')
plt.title('Scatter Chart: Price vs Power KW')
plt.grid(True)
plt.show()


# Create a DataFrame with the percentages
fuel_type_counts = df['fuel_type'].value_counts()
threshold = 500 
# Filter values greater and smaller than the threshold
larger_values = fuel_type_counts[fuel_type_counts >= threshold]
smaller_values = fuel_type_counts[fuel_type_counts < threshold]

# Create subplots for larger and smaller values
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for larger values
sns.histplot(data=df[df['fuel_type'].isin(larger_values.index)], x='fuel_type',
             hue='fuel_type', multiple='stack', palette='viridis', ax=axes[0])
axes[0].set_title('Distribution of Larger Values')

# Plot for smaller values
sns.histplot(data=df[df['fuel_type'].isin(smaller_values.index)], x='fuel_type',
             hue='fuel_type', multiple='stack', palette='magma', ax=axes[1])
axes[1].set_title('Distribution of Smaller Values')

plt.tight_layout()
plt.show()


# Linear regression
X = df[['year']]  # Features (independent variable)
y = df['price_in_euro']  # Target variable (dependent variable)

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
future_years = pd.DataFrame({'year': range(2024, 2028)})
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


import streamlit as st
import plotly.graph_objs as go


st.title("Car Analysis Dashboard")

# Add a sidebar for selecting different plots
selected_plot = st.sidebar.radio("Select Plot", ('Brand vs Price', 'Power vs Price', 'Fuel Type Distribution', 'Historical and Predicted Prices'))

if selected_plot == 'Brand vs Price':
    # Create figure for Brand vs Price
    fig = go.Figure(data=[go.Bar(x=df['price_in_euro'], y=df['code'], orientation='h', marker=dict(color='skyblue'))])
    fig.update_layout(title='Price Distribution by Brand', xaxis={'title': 'Price'}, yaxis={'title': 'Brand Codename'})
    st.plotly_chart(fig)

elif selected_plot == 'Power vs Price':
    # Create figure for Power vs Price
    fig = go.Figure(data=[go.Scatter(x=df['power_kw'], y=df['price_in_euro'], mode='markers', marker=dict(color='salmon', opacity=0.5))])
    fig.update_layout(title='Scatter Chart: Price vs Power KW', xaxis={'title': 'Power KW'}, yaxis={'title': 'Price'})
    st.plotly_chart(fig)

elif selected_plot == 'Fuel Type Distribution':
    # Create a DataFrame with the percentages for Fuel Type Distribution
    fuel_type_counts = df['fuel_type'].value_counts()
    threshold = 500 
    larger_values = fuel_type_counts[fuel_type_counts >= threshold]
    smaller_values = fuel_type_counts[fuel_type_counts < threshold]

    # Create subplots for larger and smaller values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot for larger values
    sns.histplot(data=df[df['fuel_type'].isin(larger_values.index)], x='fuel_type',
                 hue='fuel_type', multiple='stack', palette='viridis', ax=axes[0])
    axes[0].set_title('Distribution of Larger Values')

    # Plot for smaller values
    sns.histplot(data=df[df['fuel_type'].isin(smaller_values.index)], x='fuel_type',
                 hue='fuel_type', multiple='stack', palette='magma', ax=axes[1])
    axes[1].set_title('Distribution of Smaller Values')

    plt.tight_layout()
    st.pyplot(fig)

elif selected_plot == 'Historical and Predicted Prices':
    X = df[['year']]  # Features (independent variable)
    y = df['price_in_euro']  # Target variable (dependent variable)

    # Fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict prices for the next 5 years
    future_years = pd.DataFrame({'year': range(2024, 2028)})  # Predicting 2024 to 2028
    predicted_prices = model.predict(future_years[['year']])

    # Calculate mean prices in each year for historical data
    average_prices = df.groupby('year')['price_in_euro'].mean().reset_index()

    # Plotting historical and predicted prices
    plt.figure(figsize=(8, 6))
    plt.scatter(average_prices['year'], average_prices['price_in_euro'], color='blue', label='Historical Data')
    plt.scatter(future_years['year'], predicted_prices, color='red', label='Predicted Prices')
  

    # Plotting the regression line for historical data
    plt.plot(X, model.predict(X), color='green', label='Regression Line')

    plt.xlabel('Year')
    plt.ylabel('Price in Euro')
    plt.title('Historical and Predicted Prices')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)












