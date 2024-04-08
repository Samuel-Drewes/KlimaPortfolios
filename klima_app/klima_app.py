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


# DF to Show


full_globe_df = pd.read_csv('upload_data/globe_df_to_show.csv')






# Display Results

if page == 'Global Data':
    st.header("Global Data Overview")

    # Slider

    from_year, to_year = st.slider(
    'Select Years',
    min_value=2013,
    max_value=2022,
    value=[2013, 2022])

    # Globe Stacked 
    
    globe_df = globe_df[globe_df['Year'].between(from_year,to_year)]

    fig_globe_bar = px.bar(globe_df, x='Year', y='Amount', color='Type',
                title='Global Financing Totals',
                labels={'Amount': 'Financing Amount (Euros)', 'Year': 'Year'},
                category_orders={'Type': ['Other Funds','Climate Finance']},
                color_discrete_map={'Other Funds': 'orange', 'Climate Finance': 'green'})# This ensures consistent color ordering

    fig_globe_bar.update_layout(title_x=0.5)

    st.plotly_chart(fig_globe_bar)

    # Design Fig Globe Waterfall

    globe_waterfall_df = globe_waterfall_df[globe_waterfall_df['Year'].between(from_year,to_year)]


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

    st.plotly_chart(fig_globe_waterfall)


    # Percentage Text

    from_val = float(globe_waterfall_df[globe_waterfall_df['Year'] == from_year]['Percentage'])
    to_val = float(globe_waterfall_df[globe_waterfall_df['Year'] == to_year]['Percentage'])


    change_perc = round(to_val-from_val, 3)

    st.write(f"% Change in Climate finance between {from_year} and {to_year} was {change_perc}%")

    # DF to show

    full_globe_df = full_globe_df[full_globe_df['Year'].between(from_year,to_year)]

    st.dataframe(full_globe_df)


if page == 'Country Breakdown':

    st.header('Country Breakdown')



if page == 'Country Comparison':

    st.header('Country Comparison')

