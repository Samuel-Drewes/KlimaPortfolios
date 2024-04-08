import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Overview

st.title('BMZ Klima Dashboard')

page = st.sidebar.selectbox('Choose your page', ['Global Data', 'Country Breakdown', 'Country Comparison'])

# Get Globe Barchart Data

globe_df = pd.read_csv('upload_data/globe_df')

# Get Globe Waterfall Data

globe_waterfall_df = pd.read_csv('upload_data/globe_waterfall.csv')


# Design Fig Globe Bar

fig_globe_bar = px.bar(globe_df, x='Year', y='Amount', color='Type',
            title='Global Financing Totals',
            labels={'Amount': 'Financing Amount (Euros)', 'Year': 'Year'},
            category_orders={'Type': ['Other Funds','Climate Finance']},
            color_discrete_map={'Other Funds': 'orange', 'Climate Finance': 'green'})# This ensures consistent color ordering

fig_globe_bar.update_layout(title_x=0.5)

# Design Fig Globe Waterfall

initial_value = globe_waterfall_df['Percentage'].iloc[0] - globe_waterfall_df['Change'].iloc[1] # Subtract the first actual change to get the starting point


fig_globe_waterfall = go.Figure(go.Waterfall(
    name = "20", orientation = "v",
    measure = ["absolute"] + ["relative"] * (len(globe_waterfall_df) - 1), # The first measure is absolute, others are relative
    x = globe_waterfall_df['Year'].astype(str),
    textposition = "outside",
    text = globe_waterfall_df['Change'].round(2).astype(str),
    y = [initial_value] + globe_waterfall_df['Change'].tolist()[1:], # The initial value plus the changes
    connector = {"line":{"color":"rgb(63, 63, 63)"}},
))

fig_globe_waterfall.update_layout(
        title = {
            'text': "Yearly Percentage Change in Climate Finance",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis = {"type":"category"},
        yaxis = {"title":"Percentage"},
)




# # Get and process Country Specific Data

# df_country = pd.read_csv('upload_data/country_specific_df.csv')[['Recipient Name', 'Year', 'Value']]
# countries = df_country['Recipient Name'].unique()

# # Get and process Country Specific Data New Methodology

# df_country_2 = pd.read_csv('upload_data/newmethod_country_specific_df.csv')[['Recipient Name', 'Year', 'Value']]
# countries_2 = df_country_2['Recipient Name'].unique()



# Display Results

if page == 'Global Data':
    st.header("Global Data Overview")
    st.plotly_chart(fig_globe_bar)
    st.plotly_chart(fig_globe_waterfall)

if page == 'Country Breakdown':

    st.header('Country Breakdown')

#     selected_countries = st.multiselect('Which countries would you like to view?',
#                                          countries,
#                                            ['India', 'Brazil', 'Ukraine', 'Namibia', 'South Africa'])
    
#     selected_df = df_country[df_country['Recipient Name'].isin(selected_countries)]

#     fig = px.line(selected_df, 
#               x='Year', 
#               y='Value', 
#               color='Recipient Name',
#               title='Value by Year for Selected Country',
#               labels={'Value': 'Climate Relevant Financing (%)', 'Year': 'Year'},
#               markers=True)
    
#     st.plotly_chart(fig)

if page == 'Country Comparison':

    st.header('Country Comparison')

#     selected_countries = st.multiselect('Which countries would you like to view?',
#                                          countries_2,
#                                            ['India', 'Brazil', 'Ukraine', 'Namibia', 'South Africa'])
    
#     selected_df = df_country_2[df_country_2['Recipient Name'].isin(selected_countries)]

#     fig = px.line(selected_df, 
#               x='Year', 
#               y='Value', 
#               color='Recipient Name',
#               title='Value by Year for Selected Country',
#               labels={'Value': 'Climate Relevant Financing (%)', 'Year': 'Year'},
#               markers=True)
    
#     st.plotly_chart(fig)