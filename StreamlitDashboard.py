import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import streamlit as st
import random
import numpy as np

# Reading data

df = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/final.csv', encoding='ISO-8859-1')
df3 = cdata = g_data = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/final.csv', encoding='utf-8')

# Ignoring warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
cdata_electric = df3.loc[df3['fuel_type'] == 'Electric'].copy()
cdata_electric.rename(columns={'fuel_consumption_l_100km':'charge_time_100km'},inplace=True)
cdata_electric.reset_index(drop=True,inplace=True)
cdata_normal_fuel = df3.loc[df3['fuel_type'] != 'Electric'].copy()
# Average prices value
average_prices = df.groupby('year')['price_in_euro'].mean().reset_index()
average_prices['percentage_change'] = average_prices['price_in_euro'].pct_change()
mean_percentage_change = average_prices['percentage_change'].mean()
percentage_value = mean_percentage_change * 100
# Configuring Streamlit page
st.set_page_config(page_title="Germany Used Car Analysis Dashboard", page_icon="rocket", layout="wide")

# Sidebar for selecting different plots

selected_plot = st.sidebar.radio("Select Plot", ('CSV Dataset', 'Popular Brand Models', 'Consumption','Analyze Prices'))

if selected_plot == 'CSV Dataset':
    datafile = st.sidebar.file_uploader("Upload dataset", ["csv"])
    if datafile is None:
        st.info("""My dataset (.csv) in the sidebar to get started.""")
    else:
        dataset = pd.read_csv('D:/Cybersoft/Germany-Used-Car-Analysis/final.csv', encoding='ISO-8859-1')
        st.write(dataset)

elif selected_plot == 'Popular Brand Models':
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Car Sales By Brand", "Car Sales By Brand And Average Prices", "Top 10 Best Selling", "Top List Car Best-Seller", "Registration Car"])
    with tab1:
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

    with tab2:
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

    with tab3:
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

    with tab4:
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

    with tab5:
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
        st.pyplot(plt)
        

elif selected_plot == 'Consumption':
    tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs(["Fuel Types", "Consumption by fuel type", "Fuel Consumption l/100km ", "PowerPS by Fuel Type", "Information of Fuel Consumption", "Emission By Brand", "Emission with Fuel Consumption"])
    with tab6:
        st.header("_Fuel Type_ :bar_chart:")
        columns = st.columns(2)
        top = columns[0]
        bottom = columns[1]

        # Content for the top section
        with top:
            fuel_type_counts = df['fuel_type'].value_counts()
            threshold = 500 
            larger_values = fuel_type_counts[fuel_type_counts >= threshold]
            smaller_values = fuel_type_counts[fuel_type_counts < threshold]

            fig_top, axes_top = plt.subplots(1, 2, figsize=(12, 5))

            sns.histplot(data=df[df['fuel_type'].isin(larger_values.index)], x='fuel_type',
                        hue='fuel_type', multiple='stack', palette='viridis', ax=axes_top[0])
            axes_top[0].set_title('Distribution of Larger Values')
            sns.histplot(data=df[df['fuel_type'].isin(smaller_values.index)], x='fuel_type',
                        hue='fuel_type', multiple='stack', palette='magma', ax=axes_top[1])
            axes_top[1].set_title('Distribution of Smaller Values')
            plt.tight_layout()

        # Content for the bottom section
        with bottom:
            fuel_type, count = np.unique(df3['fuel_type'], return_counts=True)
            labels = list(fuel_type)
            data = list(count)
            fig_bottom, ax_bottom = plt.subplots(figsize=(6, 6), num=2)
            ax_bottom.pie(data, wedgeprops={'width': 0.5},
                        startangle=90, labels=labels
                        )
            ax_bottom.legend(loc='center', fontsize=8)

        # Display the plots separately in their respective sections
        st.pyplot(fig_top)
        st.pyplot(fig_bottom)
    with tab7:
        st.header("Consumption by Fuel Type")
        col3, col4 = st.columns([2, 2])  # Divide the layout into two columns
        with col3:
            st.subheader("Fuel Consumption")
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
            st.pyplot(fig8)

        with col4:
            st.subheader("Electric Consumption")
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

    with tab8:
            fuelt_fuelc = cdata_normal_fuel.groupby('fuel_type')['fuel_consumption_l_100km'].mean().reset_index()
            fig10, ax10 = plt.subplots(figsize=(11, 7))

            sns.barplot(data=fuelt_fuelc, x='fuel_type', y='fuel_consumption_l_100km', ax=ax10, palette= 'tab10')

            for p in ax10.patches:
                ax10.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 10), textcoords='offset points')

            plt.title("Mean fuel consumption of each fuel type (litter per 100km)",fontweight='bold')
            plt.xlabel("Name of Fuel Type")
            plt.ylabel("Mean fuel consumption (litter/100km)")
            # Draw the chart
            st.header("Mean fuel consumption of each fuel type L/100Km")
            st.pyplot(fig10)

    with tab9:
        st.header("Power PS")
        col7, col8 = st.columns([2, 2])
        with col7:
            st.subheader("Power PS Fuel")
            cdata_electric = cdata.loc[cdata['fuel_type'] == 'Electric'].copy()
            cdata_electric.rename(columns={'fuel_consumption_l_100km':'charge_time_100km'},inplace=True)
            cdata_electric.reset_index(drop=True,inplace=True)


            cdata_normal_fuel = cdata.loc[cdata['fuel_type'] != 'Electric'].copy()
            selected_columns = ['power_ps', 'charge_time_100km']
            selected_data = cdata_electric[selected_columns]
            correlation_matrix = selected_data.corr()
            #print(correlation_matrix)
            sns.scatterplot(data=cdata_electric, x='power_ps', y='charge_time_100km')
            plt.title('Scatter Plot of Power (PS) vs. Electric Consumption (kW/100km)')
            plt.xlabel('Power (PS)')
            plt.ylabel('Electric Consumption (kW/100km)')
            st.pyplot(plt)
        with col8: 
            st.subheader("Power PS Electric")
            st.title('Scatter Plot and Linear Regression')

            aa = st.slider('Select aa value', min_value=1, max_value=10, step=1, value=1)
            bb = st.slider('Select bb value', min_value=0, max_value=10, step=1, value=1)
            show_line = st.checkbox('Show Linear Regression Line')
            text = st.text_input('Enter Text for Plot', 'Linear Regression')

            xx = np.array([50, 700])
            yy = xx * (aa / 100) + bb

            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df3.sample(frac=0.3), x='power_ps', y='fuel_consumption_l_100km')
            plt.title('Scatter Plot of Power (PS) vs. Fuel Consumption (l/100km)')
            plt.xlabel('Power (PS)')
            plt.ylabel('Fuel Consumption (l/100km)')

            if show_line:
                xx_line = np.array([50, 700])
                yy_line = xx_line * (aa / 100) + bb

                sns.lineplot(x=xx_line, y=yy_line, color='red', label=f'Linear Regression: {aa/100}*x + {bb}')
                plt.legend()

            st.pyplot(plt)
    with tab10:
        st.header("Information of Fuel Consumption")
        col5, col6 = st.columns([2, 2])
        with col5:
            st.subheader("Electric")
            cdata_electric = df3.loc[df3['fuel_type'] == 'Electric'].copy()
            cdata_electric.rename(columns={'fuel_consumption_l_100km':'charge_time_100km'},inplace=True)
            cdata_electric.reset_index(drop=True,inplace=True)
            cdata_normal_fuel = df3.loc[df3['fuel_type'] != 'Electric'].copy()
            fig5, ax5 =plt.subplots(figsize=(10,7),num=1)
            ax5.hist(cdata_electric['charge_time_100km'], bins = 10, color='y',edgecolor='black')
            ax5.axvline(344.555, color='blue',linestyle=':',linewidth=2)
            ax5.text(344.555 +10, 180, f"Mean:\n{344.555:.2f}", color="b")
            ax5.axvline(344,color='green', linestyle='--',linewidth=2)
            ax5.text(344-60,170 ,f"Median:\n{344:.2f}",color='green')
            ax5.axvline(201, color='violet',linestyle='--',linewidth=2)
            ax5.text(201-60,170,f"Mode:\n{201:0.2f}",color='violet')
            ax5.set_xlabel("Electric consumption (kW/100km)")
            plt.grid()
            st.pyplot(fig5)

        with col6:
            st.subheader("Fuel")
            cdata_normal_fuel = cdata.loc[cdata['fuel_type'] != 'Electric'].copy()
            fig6, ax6 =plt.subplots(figsize=(10,7),num=1)
            ax6.hist(cdata_normal_fuel['fuel_consumption_l_100km'], bins = 100, color='r',edgecolor='black')
            ax6.axvline(6.055, color='blue',linestyle=':',linewidth=2)
            ax6.text(6.055+0.2, 13000, f"Mean:\n{6.055:.2f}", color="b")
            ax6.axvline(5.7,color='green', linestyle='--',linewidth=2)
            ax6.text(5.7-2.5,13000 ,f"Median:\n{5.7:.2f}",color='green')
            ax6.axvline(5.25, color='violet',linestyle='--',linewidth=2)
            ax6.text(5.35-4,13000,f"Mode:\n{5.25:0.2f}",color='violet')
            ax6.set_xlabel("Fuel consumption (litter/100km)")
            plt.grid()
            st.pyplot(fig6)
            
    with tab11:
        cdata_electric = cdata.loc[cdata['fuel_type'] == 'Electric'].copy()
        cdata_electric.rename(columns={'fuel_consumption_l_100km':'charge_time_100km'},inplace=True)
        cdata_electric.reset_index(drop=True,inplace=True)
        cdata_normal_fuel = cdata.loc[cdata['fuel_type'] != 'Electric'].copy()
        g_data_fuel = g_data[g_data['fuel_type']!="Electric"]
        g_data_fuel.reset_index(drop=True,inplace=True)

        mean_g = g_data_fuel.groupby('brand')['fuel_consumption_g_km'].mean().reset_index()
        mean_g.sort_values('fuel_consumption_g_km',ascending=False,inplace=True)
        mean_g.reset_index(drop=True,inplace=True)
        fig11, axes11 = plt.subplots(nrows=2,ncols=2,figsize=(12,9),sharex=True)

        sns.barplot(data=mean_g.head(10),  x='fuel_consumption_g_km',y='brand', palette='viridis',
                ax=axes11[0,0])
        axes11[0,0].set_title('top 10 Brands')
        axes11[0,0].set_xlabel("Mean Emission (g/km)")
        axes11[0,0].set_ylabel('Brand Name')
        axes11[0,0].grid(axis='x')

        sns.barplot(data=mean_g.loc[10:20],x='fuel_consumption_g_km',y='brand', palette='rocket',
                ax=axes11[0,1])
        axes11[0,1].set_title('top 11-20 Brands')
        axes11[0,1].set_xlabel("Mean Emission (g/km)")
        axes11[0,1].set_ylabel('Brand Name')
        axes11[0,1].grid(axis='x')

        sns.barplot(data=mean_g.loc[20:30],x='fuel_consumption_g_km', y='brand', palette='mako',
                ax=axes11[1,0])
        axes11[1,0].set_title('top 21-30 Brands')
        axes11[1,0].set_xlabel("Mean Emission (g/km)")
        axes11[1,0].set_ylabel('Brand Name')
        axes11[1,0].grid(axis='x')

        sns.barplot(data=mean_g.loc[30:],x='fuel_consumption_g_km', y='brand', palette='plasma',
                ax=axes11[1,1])
        axes11[1,1].set_title('top 31-45 Brands')
        axes11[1,1].set_xlabel("Mean Emission (g/km)")
        axes11[1,1].set_ylabel('Brand Name')
        axes11[1,1].grid(axis='x')

        fig11.suptitle('The Mean Emission Comparision of Normal Fuel Cars of Brands ',
                    fontweight='bold', fontsize=16)

        plt.tight_layout(pad=2.0)
        fig11.patch.set_facecolor('#e6ccb3')
        st.pyplot(fig11)
    with tab12:
        st.header("The emssion following by Fuel Consumption")
        g_data_normal_fuel = g_data[g_data['fuel_type'] != "Electric"]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=g_data_normal_fuel, x='fuel_consumption_l_100km', y='fuel_consumption_g_km', color='orange')
        plt.title('Scatter Plot of Fuel Consumption (l/100km) vs. Emission (g/km)')
        plt.xlabel('Fuel Consumption (l/100km)')
        plt.ylabel('Emission (g/km)')
        st.pyplot(plt)
elif selected_plot == 'Analyze Prices': 
    st.header("Prices")
    tab13, tab14 = st.tabs(["Analyze Prices by Brand", "Predicting Prices"])
    with tab13:
        st.subheader("Analyze Prices by Brands")
        cdata_normal_fuel = cdata[cdata['fuel_type']!='electric']
        cdata_normal_fuel['price_in_euro'].describe()
        cdata_normal_fuel['fuel_consumption_l_100km'].describe()

        mean_price_brand = cdata_normal_fuel.groupby('brand')['price_in_euro'].mean().reset_index()
        mean_price_brand.sort_values('price_in_euro',ascending=False,inplace=True)
        mean_price_brand.reset_index(drop=True,inplace=True)
        fig10, axes10 = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), sharex=True)

        sns.barplot(data=mean_price_brand.head(10), x='price_in_euro', y='brand', palette='viridis', ax=axes10[0])
        axes10[0].set_title('Top 1-10 Brands')
        axes10[0].set_xlabel('Price')
        axes10[0].set_ylabel('Brand Name')
        axes10[0].grid(axis='x')

        sns.barplot(data=mean_price_brand.iloc[10:20], x='price_in_euro', y='brand', palette='mako', ax=axes10[1])
        axes10[1].set_title('Top 11-20 Brands')
        axes10[1].set_xlabel('Price')
        axes10[1].set_ylabel('Brand Name')
        axes10[1].grid(axis='x')
        fig10.suptitle('top 20 brands price comparision', fontweight='bold', fontsize=16)
        plt.tight_layout(pad=1.0)
        fig10.patch.set_facecolor('#e6e6e6')
        st.pyplot(fig10)
    with tab14:
        st.subheader("Predicting Prices Using Linear Regression")
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
# To run streamlit in websit local host: Please run the code and it will return the link for you
# Then, copy it and run it in the terminal












