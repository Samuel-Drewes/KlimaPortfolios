import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Overview

st.title('BMZ Klima Dashboard')

page = st.sidebar.selectbox('Choose your page', ['Aggregate Data', 'Country Breakdown'])

# Get and process Globe Data

merged_df = pd.read_csv('upload_data/global_df.csv')
merged_df = merged_df.set_index(merged_df.columns[0])
df_long = merged_df.stack().reset_index()
df_long.columns = ['Climate Relevance', 'Year', 'Financing']
df_long = df_long[df_long['Climate Relevance'] != 'Total Financing']


# Design Fig Globe

fig = px.bar(
    df_long,
    x='Year',
    y='Financing',
    color='Climate Relevance',  # This determines the stack segments
    hover_data=['Climate Relevance', 'Financing'],  # Customize hover info
    title='Global BMZ Financing by Climate Relevance'
)

fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Total Financing',
    barmode='stack'
)

# Get and process Country Specific Data

df_country = pd.read_csv('upload_data/country_specific_df.csv')[['Recipient Name', 'Year', 'Value']]
countries = df_country['Recipient Name'].unique()




# Display Results

if page == 'Aggregate Data':
    st.header("Aggregate Data Overview")
    st.dataframe(data=merged_df)
    st.plotly_chart(fig)

if page == 'Country Breakdown':

    st.write('country breakdown')

    selected_countries = st.multiselect('Which countries would you like to view?',
                                         countries,
                                           ['India', 'Brazil', 'Ukraine', 'Namibia', 'South Africa'])
    
    selected_df = df_country[df_country['Recipient Name'].isin(selected_countries)]

    fig = px.line(selected_df, 
              x='Year', 
              y='Value', 
              color='Recipient Name',
              title='Value by Year for Selected Country',
              labels={'Value': 'Climate Relevant Financing (%)', 'Year': 'Year'},
              markers=True)
    
    st.plotly_chart(fig)