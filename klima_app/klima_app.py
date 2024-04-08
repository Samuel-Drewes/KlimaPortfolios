import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Overview

st.title('BMZ Klima Dashboard')

page = st.sidebar.selectbox('Choose your page', ['Global Data', 'Country Breakdown', 'Country Comparison'])

# Get Globe Data

# merged_df = pd.read_csv('upload_data/global_df.csv')
# merged_df = merged_df.set_index(merged_df.columns[0])
# df_long = merged_df.stack().reset_index()
# df_long.columns = ['Climate Relevance', 'Year', 'Financing']
# df_long = df_long[df_long['Climate Relevance'] != 'Total Financing']

globe_df = pd.read_csv('upload_data/globe_df')



# Design Fig Globe

fig = px.bar(globe_df, x='Year', y='Amount', color='Type',
            title='Global Financing Totals',
            labels={'Amount': 'Financing Amount (Euros)', 'Year': 'Year'},
            category_orders={'Type': ['Other Funds','Climate Finance']},
            color_discrete_map={'Other Funds': 'orange', 'Climate Finance': 'green'})# This ensures consistent color ordering

fig.update_layout(title_x=0.5)

# # Get and process Country Specific Data

# df_country = pd.read_csv('upload_data/country_specific_df.csv')[['Recipient Name', 'Year', 'Value']]
# countries = df_country['Recipient Name'].unique()

# # Get and process Country Specific Data New Methodology

# df_country_2 = pd.read_csv('upload_data/newmethod_country_specific_df.csv')[['Recipient Name', 'Year', 'Value']]
# countries_2 = df_country_2['Recipient Name'].unique()



# Display Results

if page == 'Global Data':
    st.header("Global Data Overview")
    st.plotly_chart(fig)

# if page == 'Country Breakdown':

#     st.header('Country Breakdown')

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

# if page == 'Country Breakdown New Logic':

#     st.header('Country Breakdown New Logic')

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