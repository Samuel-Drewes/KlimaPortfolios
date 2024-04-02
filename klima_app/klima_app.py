import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Overview

st.title('BMZ Klima Dashboard')

page = st.sidebar.selectbox('Choose your page', ['Aggregate Data', 'Country Breakdown'])

# Get and process Data

merged_df = pd.read_csv('upload_data/global_df.csv')
merged_df = merged_df.set_index(merged_df.columns[0])
df_long = merged_df.stack().reset_index()
df_long.columns = ['Climate Relevance', 'Year', 'Financing']
df_long = df_long[df_long['Climate Relevance'] != 'Total Financing']


# Design Fig

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

# Display Results

if page == 'Aggregate Data':
    st.header("Aggregate Data Overview")
    st.dataframe(data=merged_df)
    st.plotly_chart(fig)

if page == 'Country Breakdown':
    st.write('country breakdown')