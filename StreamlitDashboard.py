import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import streamlit as st
import random

# Reading data
df = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/brandcode_final.csv', encoding='ISO-8859-1')
df3 = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/brandcode_final.csv', encoding='utf-8')
# Ignoring warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
# Average prices value
average_prices = df.groupby('year')['price_in_euro'].mean().reset_index()
average_prices['percentage_change'] = average_prices['price_in_euro'].pct_change()
mean_percentage_change = average_prices['percentage_change'].mean()
percentage_value = mean_percentage_change * 100
# Configuring Streamlit page
st.set_page_config(page_title="Germany Used Car Analysis Dashboard", page_icon="rocket", layout="wide")

# Sidebar for selecting different plots

selected_plot = st.sidebar.radio("Select Plot", ('CSV Dataset','Fuel Type and Predicting Prices', 'Popular Brand Models', 'Consumption'))

if selected_plot == 'CSV Dataset':
    datafile = st.sidebar.file_uploader("Upload dataset", ["csv"])
    if datafile is None:
        st.info("""My dataset (.csv) in the sidebar to get started.""")
    else:
        dataset = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/brandcode_final.csv', encoding='ISO-8859-1')
        st.write(dataset)
elif selected_plot == 'Fuel Type and Predicting Prices':
    tab1, tab2 = st.tabs(["Fuel Type", "Predicting Prices"])

    with tab1:
        st.header("_Fuel Type_ :bar_chart:")
        fuel_type_counts = df['fuel_type'].value_counts()
        threshold = 500 
        larger_values = fuel_type_counts[fuel_type_counts >= threshold]
        smaller_values = fuel_type_counts[fuel_type_counts < threshold]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.histplot(data=df[df['fuel_type'].isin(larger_values.index)], x='fuel_type',
                     hue='fuel_type', multiple='stack', palette='viridis', ax=axes[0])
        axes[0].set_title('Distribution of Larger Values')

        sns.histplot(data=df[df['fuel_type'].isin(smaller_values.index)], x='fuel_type',
                     hue='fuel_type', multiple='stack', palette='magma', ax=axes[1])
        axes[1].set_title('Distribution of Smaller Values')

        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        col1, col2 = st.columns([2, 2])  # Divide the layout into two columns
        col_bottom = st.columns([1])     # Single column at the bottom

        with col1:
            st.header("Percentage Over Years:")
            def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
                fig = go.Figure()

                fig.add_trace(
                    go.Indicator(
                        value=value,
                        gauge={"axis": {"visible": False}},
                        number={
                            "prefix": prefix,
                            "suffix": suffix,
                            "font.size": 28 
                        },
                        title={
                            "text": label,
                            "font": {"size": 24},
                        },
                    )
                )

                if show_graph:
                    fig.add_trace(
                        go.Scatter(
                            y=random.sample(range(0, 101), 30),
                            hoverinfo="skip",
                            fill="tozeroy",
                            fillcolor=color_graph,
                            line={
                    "color": color_graph,
                            },
                        )
                    )

                fig.update_xaxes(visible=False, fixedrange=True)
                fig.update_yaxes(visible=False, fixedrange=True)
                fig.update_layout(
                    # paper_bgcolor="lightgrey",
                    margin=dict(t=30, b=0),
                    showlegend=False,
                    plot_bgcolor="white",
                    height=100,
                )

                st.plotly_chart(fig, use_container_width=True)
            plot_metric("Percentage Change Over Years", percentage_value, suffix=" %", show_graph=True, color_graph="rgba(0, 104, 201, 0.2)")
            

        with col2:
            fig, ax = plt.subplots()
            ax.bar(average_prices['year'], average_prices['percentage_change'])
            ax.set_xlabel('Year')
            ax.set_ylabel('Percentage Change')
            ax.set_title('Percentage Change Over Years')
            st.pyplot(fig)


        #with col_bottom:
            col_bottom[0].header("Predicting Data in next 5 years:")    
            X = df[['year']]  
            y = df['price_in_euro']  

            model = LinearRegression()
            model.fit(X, y)

            future_years = pd.DataFrame({'year': range(2023, 2028)})  
            predicted_prices = model.predict(future_years[['year']])

            plt.figure(figsize=(6, 4))
            plt.scatter(average_prices['year'], average_prices['price_in_euro'], color='blue', label='Historical Data')
            plt.scatter(future_years['year'], predicted_prices, color='red', label='Predicted Prices')
            plt.plot(X, model.predict(X), color='green', label='Regression Line')
            plt.xlabel('Year')
            plt.ylabel('Price in Euro')
            plt.title('Predicting Prices')
            plt.legend()
            plt.grid(True)
            col_bottom[0].pyplot(plt)

elif selected_plot == 'Popular Brand Models':
    tab3, tab4, tab5, tab6, tab7 = st.tabs(["Car Sales By Brand", "Car Sales By Brand And Average Prices", "Top 10 Best Selling", "Top List Car Best-Seller", "Registered Over Years"])
    with tab3:
        brand_counts = df3['brand'].value_counts()
        # Set the size of the chart
        plt.figure(figsize=(12, 8))
        sns.set_palette("pastel")
        # Create a bar chart
        brand_counts.plot(kind='bar', color='skyblue',width=0.8 )
        plt.xlabel('Car Brands')
        plt.ylabel('Car Sales')
        plt.title('Car Sales by Brands')
        plt.xticks(rotation=55, ha='right')
        # Display the 
        st.header ("Car Sales by Brand :car:")
        st.pyplot(plt)

    with tab4:
        brand_stats = df3.groupby('brand')['price_in_euro'].agg(['count', 'mean'])
        # Sort the DataFrame by count in descending order
        brand_stats = brand_stats.sort_values(by='count', ascending=False)
        # Set the size of the chart
        fig, ax1 = plt.subplots(figsize=(20, 8))
        # Plot count on the primary y-axis (bar chart)
        x_positions = range(len(brand_stats))
        ax1.bar(x_positions, brand_stats['count'], color='skyblue', width=0.8, label='Unit')
        # Create a twin Axes sharing the x-axis
        ax2 = ax1.twinx()
        # Plot average price on the secondary y-axis (line chart)
        ax2.plot(x_positions, brand_stats['mean'], marker='o', color='green', label='Average Price')
        # Set labels and title
        ax1.set_xlabel('Car Brands')
        ax1.set_ylabel('Car Sales')
        ax2.set_ylabel('Average Price')
        plt.title('Car sales by brand and Average Prices')
        #set style
        plt.style.use('default')
        ax1.grid(False)
        ax2.grid(False)
        # Show legend
        ax1.legend(loc='upper left', bbox_to_anchor=(0.85, 0.90))
        ax2.legend(loc='upper left', bbox_to_anchor=(0.85, 0.95))
        # Set x-axis ticks and labels
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(brand_stats.index, rotation=55, ha='right')
        sns.set_palette("pastel")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Display the chart
        st.header ("Car sales by brand and Average Prices")
        st.pyplot(fig)

    with tab5:
        top_10_models_with_prices = df3.groupby('model')['price_in_euro'].agg(['count', 'mean']).reset_index()
        top_10_models_with_prices = top_10_models_with_prices.nlargest(10, 'count')

        # Set the Seaborn theme
        sns.set_theme()
        # Set the size of the chart
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Bar chart for the number of car sales
        sns.barplot(x='model', y='count', data=top_10_models_with_prices, color='skyblue', ax=ax1)
        # Set labels and title for the bar chart
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Car Sales', color='orange')
        ax1.set_title('Top 10 Best-Selling Car Models with Average Prices')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.grid(visible=False)

        # Create a second y-axis for the line chart
        ax2 = ax1.twinx()

        # Line chart for average prices
        sns.lineplot(x='model', y='mean', data=top_10_models_with_prices, color='orange', marker='s', ax=ax2)
        # Set labels and title for the line chart
        ax2.set_ylabel('Average Price', color='orange')
        ax2.grid(visible=False)
        st.header ("Top 10 best-selling Car")
        st.pyplot(fig)

    with tab6:
        grouped_data = df3.groupby(['brand', 'model']).size().reset_index(name='count')
        # Find the index of the most common model for each brand
        idx_most_common = grouped_data.groupby('brand')['count'].idxmax()
        most_common_models = grouped_data.loc[idx_most_common]
        most_common_models = most_common_models.sort_values(by='count', ascending=False)
        # Create a dropdown list with unique brand values and an "all" option
        brand_options = ['all'] + most_common_models['brand'].unique().tolist()
        selected_brand = st.selectbox('Select Brand:', brand_options)
        # Function to filter data based on the selected brand
        def filter_data(selected_brand):
            if selected_brand == 'all':
                result = most_common_models
            else:
                result = most_common_models[most_common_models['brand'] == selected_brand]
            return result.rename(columns={'count': 'Car Sales'})
        # Display the filtered result
        result = filter_data(selected_brand)
        st.header ("Top Selling Car")
        st.write(result)
        

    with tab7:
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
        st.header ("Vehicle Registrations by Year")
        st.pyplot(plt)

elif selected_plot == 'Consumption':
    tab8, tab9 = st.tabs(["Fuel Consumption of Brands", "Electric Consumption of Brands"])
    with tab8:
        st.header("Mean Fuel Consumption per 100km Comparison of Brands")
        fig8, axes8 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True)
        cdata_normal_fuel = df3.loc[df3['fuel_type'] !='Electric'].copy()
        mean_fuel = cdata_normal_fuel.groupby(['brand'])['fuel_consumption_l_100km'].mean().reset_index()
        mean_fuel.sort_values('fuel_consumption_l_100km',ascending=False,inplace=True)
        sns.barplot(data=mean_fuel.head(10), x='fuel_consumption_l_100km', y='brand', palette='viridis', ax=axes8[0, 0])
        axes8[0, 0].set_title('Top 1-10 Brands')
        axes8[0, 0].set_xlabel('Fuel Consumption (l/100km)')
        axes8[0, 0].set_ylabel('Brand Name')
        axes8[0, 0].grid(axis='x')


        sns.barplot(data=mean_fuel.iloc[10:20], x='fuel_consumption_l_100km', y='brand', palette='rocket', ax=axes8[0, 1])
        axes8[0, 1].set_title('Top 11-20 Brands')
        axes8[0, 1].set_xlabel('Fuel Consumption (l/100km)')
        axes8[0, 1].set_ylabel('Brand Name')
        axes8[0, 1].grid(axis='x')


        sns.barplot(data=mean_fuel.iloc[20:30], x='fuel_consumption_l_100km', y='brand', palette='mako', ax=axes8[1, 0])
        axes8[1, 0].set_title('Top 21-30 Brands')
        axes8[1, 0].set_xlabel('Fuel Consumption (l/100km)')
        axes8[1, 0].set_ylabel('Brand Name')
        axes8[1, 0].grid(axis='x')


        sns.barplot(data=mean_fuel.iloc[30:], x='fuel_consumption_l_100km', y='brand', palette='magma', ax=axes8[1, 1])
        axes8[1, 1].set_title('Top 31-45 Brands')
        axes8[1, 1].set_xlabel('Fuel Consumption (l/100km)')
        axes8[1, 1].set_ylabel('Brand Name')
        axes8[1, 1].grid(axis='x')


        fig8.suptitle('Mean Fuel Consumption per 100km Comparison of Brands', fontweight='bold', fontsize=16)

        plt.tight_layout(pad=1.0)
        fig8.patch.set_facecolor('#d3f0ff')
        plt.savefig('Normal_Fuel_consumption', bbox_inches='tight')
        st.pyplot(fig8)
    with tab9:
        st.header("Mean Electric Consumption per 100km Comparison of Brands")
        fig9, axes9 = plt.subplots(nrows=2, ncols=2, figsize=(11, 7), sharex=True)
        cdata_electric = df3.loc[df3['fuel_type'] == 'Electric'].copy()
        cdata_electric.reset_index(drop=True)
        cdata_electric.rename(columns={'fuel_consumption_l_100km':'charge_time_100km'},inplace=True)
        cdata_electric.reset_index(drop=True,inplace=True)
        mean_charge = cdata_electric.groupby(['brand'])['charge_time_100km'].mean().reset_index()
        mean_charge.sort_values('charge_time_100km',ascending=False,inplace=True)
        mean_charge.reset_index(drop=True,inplace=True)
        sns.barplot(data=mean_charge.head(7), x='charge_time_100km', y='brand', palette='viridis', ax=axes9[0, 0])
        axes9[0, 0].set_title('Top 1-7 Brands')
        axes9[0, 0].set_xlabel('Electric Consumption (charge/100km)')
        axes9[0, 0].set_ylabel('Brand Name')
        axes9[0, 0].grid(axis='x')


        sns.barplot(data=mean_charge.iloc[7:14], x='charge_time_100km', y='brand', palette='rocket', ax=axes9[0, 1])
        axes9[0, 1].set_title('Top 8-14 Brands')
        axes9[0, 1].set_xlabel('Eletric Consumption (charges/100km)')
        axes9[0, 1].set_ylabel('Brand Name')
        axes9[0, 1].grid(axis='x')

        sns.barplot(data=mean_charge.iloc[14:21], x='charge_time_100km', y='brand', palette='mako', ax=axes9[1, 0])
        axes9[1, 0].set_title('Top 15-21 Brands')
        axes9[1, 0].set_xlabel('Electric Consumption (charges/100km)')
        axes9[1, 0].set_ylabel('Brand Name')
        axes9[1, 0].grid(axis='x')

        sns.barplot(data=mean_charge.iloc[21:], x='charge_time_100km', y='brand', palette='magma', ax=axes9[1, 1])
        axes9[1, 1].set_title('Top 22-27 Brands')
        axes9[1, 1].set_xlabel('Eletric Consumption (charges/100km)')
        axes9[1, 1].set_ylabel('Brand Name')
        axes9[1, 1].grid(axis='x')

        fig9.suptitle('Mean Electric Consumption per 100km Comparison of Brands', fontweight='bold', fontsize=16,color='yellow')
        plt.tight_layout(pad=2.0)
        fig9.patch.set_facecolor('#87CEEB')
        st.pyplot(fig9)

# To run streamlit in websit local host: Please run the code and it will return the link for you
# Then, copy it and run it in the terminal












