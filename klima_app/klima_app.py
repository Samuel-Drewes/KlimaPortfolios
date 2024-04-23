import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Overview

st.title('BMZ Klima-Dashboard')
st.header(':red[_Prototyp_ Daten wurden noch nicht validiert]')

page = st.sidebar.selectbox('Seitenauswahl', ['Gesamtübersicht', 'Länderanalyse', 'Ländervergleich', 'Sektoranalyse Global', 'Sektoranalyse pro Land','Methodik Erklärung'])



# Get Globe Barchart Data

globe_df = pd.read_csv('upload_data/globe_df.csv')

# Get Globe Split Data

split_df = pd.read_csv('upload_data/split_globe.csv')

# Get Globe Waterfall Data

globe_waterfall_df = pd.read_csv('upload_data/globe_waterfall.csv')

# Get DF to Show

full_globe_df = pd.read_csv('upload_data/globe_df_to_show.csv')

# Get all country DF

all_country_df = pd.read_csv("upload_data/all_country_df.csv")
countries = all_country_df['Recipient Name'].unique()

# Country compare DF

country_compare_df = pd.read_csv('upload_data/country_specific_df.csv')

# Sektoranalyse DF

sector_df = pd.read_csv('upload_data/sector_analysis.csv')


# Display Results

if page == 'Gesamtübersicht':

    st.header("Gesamtübersicht")

    # Slider

    from_year, to_year = st.slider(
    'Jahrauswahl',
    min_value=2013,
    max_value=2022,
    value=[2013, 2022])

    # Text %

    globe_waterfall_df = globe_waterfall_df[globe_waterfall_df['Year'].between(from_year,to_year)]

    from_val = float(globe_waterfall_df[globe_waterfall_df['Year'] == from_year]['Percentage'])
    to_val = float(globe_waterfall_df[globe_waterfall_df['Year'] == to_year]['Percentage'])

    change_perc = round(to_val-from_val, 3)

    st.subheader(f"Die globale prozentuale Änderung der Klimafinanzierung zwischen {from_year} und {to_year} betrug {change_perc}%")


    # Globe Stacked 
    
    globe_df = globe_df[globe_df['Year'].between(from_year,to_year)]

    fig_globe_bar = px.bar(globe_df, x='Year', y='Amount', color='Type',
                title='Globale Finanzierungssummen',
                labels={'Amount': 'Finanzierungssumme ($)', 'Year': 'Jahr'},
                category_orders={'Type': ['Other Funds','Climate Finance']},
                color_discrete_map={'Other Funds': 'orange', 'Climate Finance': 'green'})# This ensures consistent color ordering

    fig_globe_bar.update_layout(title_x=0.5)

    st.plotly_chart(fig_globe_bar)

    # Globe Split

    split_df = split_df[split_df['Year'].between(from_year,to_year)]

    fig_split = px.bar(split_df, x='Year', y='Amount', color='Type',
            title='Globale Finanzierungssummen',
            labels={'Amount': 'Finanzierungssumme ($)', 'Year': 'Jahr'},
            category_orders={'Type': ['Andere ODA','Klimaschutz Finanzierung', 'Klimaanpassung Finanzierung']},
            color_discrete_map={'Andere ODA': 'orange', 'Klimaschutz Finanzierung': 'green', 'Klimaanpassung Finanzierung': 'blue'}
            )# This ensures consistent color ordering

    fig_split.update_layout(title_x=0.5)

    st.plotly_chart(fig_split)



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
                'text': "Jährliche prozentuale Veränderung der Klimafinanzierung",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis = {"type":"category"},
            yaxis = {"title":"Percentage"},
    )

    st.plotly_chart(fig_globe_waterfall)



    # DF to show

    full_globe_df = full_globe_df[full_globe_df['Year'].between(from_year,to_year)]

    st.dataframe(full_globe_df)


if page == 'Länderanalyse':

    st.header('Länderanalyse')

    # Slider

    from_year, to_year = st.slider(
    'Jahrauswahl',
    min_value=2013,
    max_value=2022,
    value=[2013, 2022])

    selected_countries = st.multiselect(
        'Welche Länder möchten Sie vergleichen?',
        countries,
        ['India', 'Brazil', 'Namibia', 'Ukraine', 'Tunisia', 'Mexico'])
    

    #Filter by country
    all_country_df = all_country_df[all_country_df['Recipient Name'].isin(selected_countries)]

    # Get Sum_df
    selected_columns = [col for col in all_country_df.columns if col.startswith('amount_') or col.startswith('clim_rel_amount_')]
    filtered_df = all_country_df[selected_columns]
    sums = filtered_df.sum()
    sum_df = pd.DataFrame(sums).transpose()

    # Created Sum_DF

    years = range(from_year, (to_year + 1))
    for year in years:
        amount_col = f'amount_{year}'
        clim_rel_amount_col = f'clim_rel_amount_{year}'
        non_clim_col = f'non_clim_amount_{year}'
        sum_df[non_clim_col] = sum_df[amount_col] - sum_df[clim_rel_amount_col]


    # Create Stacked Bar

    melted_df = sum_df.melt(value_vars=[f'clim_rel_amount_{year}' for year in years] + 
                                [f'non_clim_amount_{year}' for year in years],
                                var_name='Type_Year', value_name='Amount')

    melted_df['Year'] = melted_df['Type_Year'].apply(lambda x: x.split('_')[-1])
    melted_df['Type'] = melted_df['Type_Year'].apply(lambda x: 'Climate Finance' if 'clim_rel_amount' in x else 'Other Funds')

    melted_df['Amount'] = melted_df['Amount'] * 1_000_000

    fig_sel_bar = px.bar(melted_df, x='Year', y='Amount', color='Type',
                title='Finanzierungssummen für ausgewählte Lânder',
                labels={'Amount': 'Finanzierungssumme ($)', 'Year': 'Jahr'},
                category_orders={'Type': ['Other Funds','Climate Finance']},
                color_discrete_map={'Other Funds': 'orange', 'Climate Finance': 'green'})# This ensures consistent color ordering

    fig_sel_bar.update_layout(title_x=0.5)


    st.plotly_chart(fig_sel_bar)


    # Create Waterfall


    for year in years:
        amount_col = f'amount_{year}'
        clim_rel_amount_col = f'clim_rel_amount_{year}'
        clim_rel_percent_col = f'clim_rel_percent_{year}'
        
        sum_df[clim_rel_percent_col] = (sum_df[clim_rel_amount_col] / sum_df[amount_col]) * 100
        
    selected_columns = [col for col in sum_df.columns if col.startswith('clim_rel_percent_')]
    filtered_df = sum_df[selected_columns]


    melted_df = filtered_df.reset_index().melt(id_vars=['index'], var_name='Year', value_name='Percentage')

    melted_df['Year'] = melted_df['Year'].str.replace('clim_rel_percent_', '')

    melted_df = melted_df.drop(columns = 'index')
    melted_df['Change'] = melted_df['Percentage'].diff()

    initial_value = melted_df['Percentage'].iloc[0] - melted_df['Change'].iloc[1] # Subtract the first actual change to get the starting point


    fig_sel_waterfall = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["absolute"] + ["relative"] * (len(melted_df) - 1), # The first measure is absolute, others are relative
        x = melted_df['Year'].astype(str),
        textposition = "outside",
        text = melted_df['Change'].round(2).astype(str),
        y = [initial_value] + melted_df['Change'].tolist()[1:], # The initial value plus the changes
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig_sel_waterfall.update_layout(
            title = {
                'text': "Yearly Percentage Change in Climate Finance for Selected Countries",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis = {"type":"category"},
            yaxis = {"title":"Percentage"},
    )

    st.plotly_chart(fig_sel_waterfall)
    st.dataframe(all_country_df)

if page == 'Ländervergleich':

    st.header('Ländervergleich')

    # Slider

    from_year, to_year = st.slider(
    'Jahrauswahl',
    min_value=2013,
    max_value=2022,
    value=[2013, 2022])

    selected_countries = st.multiselect(
    'Welche Länder möchten Sie vergleichen?',
    countries,
    ['India', 'Brazil', 'Namibia', 'Ukraine', 'Tunisia', 'Mexico'])

    country_compare_df = country_compare_df[country_compare_df['Recipient Name'].isin(selected_countries)]

    country_compare_df = country_compare_df[country_compare_df['Year'].between(from_year,to_year)]

    # Line Graph

    fig_country_compare = px.line(country_compare_df, 
                x='Year', 
                y='Value', 
                color='Recipient Name',
                title='Value by Year for Selected Country',
                labels={'Value': 'Climate Relevant Financing (%)', 'Year': 'Year'},
                markers=True)

    st.plotly_chart(fig_country_compare)

    # Df Ranking

    from_values = country_compare_df[country_compare_df['Year'] == from_year]\
        .sort_values(by = 'Recipient Name')\
        .reset_index()['Value']

    to_values = country_compare_df[country_compare_df['Year'] == to_year]\
            .sort_values(by = 'Recipient Name')\
            .reset_index()['Value']

    index_countries = list(country_compare_df[country_compare_df['Year'] == to_year].sort_values(by = 'Recipient Name')['Recipient Name'])

    values_countries = (to_values - from_values)*100
    values_countries.index = index_countries

    ranked_df = pd.DataFrame(values_countries).rename(columns={'Value': '% Change in Period'})\
            .sort_values('% Change in Period', ascending = False)

    st.subheader("Ranking of Selected Countries for Time Period")
    st.dataframe(ranked_df)

    # Top 10 Bottom 10 Ranking

    country_compare_df = pd.read_csv('upload_data/country_specific_df.csv')

    country_compare_df = country_compare_df[country_compare_df['Year'].between(from_year,to_year)]

    from_values = country_compare_df[country_compare_df['Year'] == from_year]\
        .sort_values(by = 'Recipient Name')\
        .reset_index()['Value']

    to_values = country_compare_df[country_compare_df['Year'] == to_year]\
            .sort_values(by = 'Recipient Name')\
            .reset_index()['Value']

    index_countries = list(country_compare_df[country_compare_df['Year'] == to_year].sort_values(by = 'Recipient Name')['Recipient Name'])

    values_countries = to_values - from_values

    values_countries.index = index_countries

    ranked_df = pd.DataFrame(values_countries).rename(columns={'Value': '% Change in Period'})\
            .sort_values('% Change in Period', ascending = False)

    st.subheader("Top and Bottom 10 Countries for all Countries in selected Time Period")

    st.dataframe(ranked_df.dropna().head(10))
    st.dataframe(ranked_df.dropna().tail(10))

if page == 'Sektoranalyse Global':

    selected_year = st.slider(
    'Select Year',
    min_value=2013,
    max_value=2022,
    value=2022
    )

    year_select = [col for col in sector_df.columns if col.endswith(f'{selected_year}')]
    year_select.append('Sector')
    filtered_year_df = sector_df[year_select]

    st.write(f"{selected_year}")

    clim_rel_fig = px.sunburst(filtered_year_df, path=['Sector'], values=f'clim_rel_amount_{selected_year}', title='Climate Related Amount by Sector')
    non_clim_fig = px.sunburst(filtered_year_df, path=['Sector'], values=f'non_clim_{selected_year}', title='Non Climate Related Amount by Sector')
    clim_adapt_fig = px.sunburst(filtered_year_df, path=['Sector'], values=f'clim_adapt_amount_{selected_year}', title='Climate Apaptation Amount by Sector')
    clim_miti_fig = px.sunburst(filtered_year_df, path=['Sector'], values=f'clim_miti_amount_{selected_year}', title='Climate Mitigation Amount by Sector')

    st.plotly_chart(clim_rel_fig)
    st.plotly_chart(non_clim_fig)
    st.plotly_chart(clim_adapt_fig)
    st.plotly_chart(clim_miti_fig)

if page == 'Sektoranalyse pro Land':

    st.header('Sektoranalyse pro Land')

if page == 'Methodik Erklärung':

    st.header("Erklärung Methode")

    st.write("""
    Quelle: OECD CRS Datenbank (Daten bis einschließlich 2022)
    Bilaterale Entwicklungszusammenarbeit
    Finanzierungstyp: ODA Auszahlungen (konstante Preise; in Million USD)
    Geber: Deutschland
    Geberorganisation: BMZ (nur BMZ Haushalt)
    Finanzierungsarten:\n
    - Standard Grant\n
    - Standard Loan\n
    - Common Equity\n
    - Shares in collective investement vehicles\n
    - Debt forgiveness: ODA claims (P)\n
    - Debt forgiveness: ODA claims (I)\n
    - Debt rescheduling: ODA claims (I)\n

'Administrative Costs' des BMZ sind aus der Analyse entzogen.
             
**Erklärung der Klimafinanzierungsberechnung:**

*Bei der Berechnung der Gesamtklimafinanzierung für Projekte gibt es spezifische Regeln abhängig von den Bewertungen in den Kategorien "Climate Adaptation" und "Climate Mitigation":*

Vollständige Summe: Wenn ein Projekt entweder in der Kategorie "Climate Adaptation" oder "Climate Mitigation" eine '2' aufweist, oder in beiden Kategorien jeweils eine '1', wird die gesamte Projektsumme zur Klimafinanzierung gezählt.
Halbe Summe: Wenn nur in einer der beiden Kategorien (entweder "Climate Adaptation" oder "Climate Mitigation") der Marker '1' vorhanden ist, wird nur die halbe Summe des Projektes zur Klimafinanzierung gezählt.
Aufteilung zwischen Mitigation und Adaptation:

*Bei der spezifischen Zuordnung der Finanzmittel zu den Kategorien "Mitigation" und "Adaptation" gelten folgende Regeln:*

Gesamtsumme: Erhält eine Kategorie eine '2', wird die gesamte Summe des Projektes dieser Kategorie zugewiesen.
Halbe Summe: Wird einer Kategorie eine '1' zugeordnet, wird die Hälfte der Gesamtsumme des Projektes dieser Kategorie zugewiesen.
""")
    


