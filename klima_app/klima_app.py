import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.markdown("""
<style>
.intermediate-font {
    font-size:16px;  /* Example size, adjust as needed */
    font-weight: 1000; /* Adjust the weight to mimic subheader, if desired */
}
</style>
""", unsafe_allow_html=True)

tick_values = [2e9, 4e9, 6e9, 8e9, 10e9, 12e9]
tick_labels = ['2 M', '4 M', '6 M', '8 M', '10 M', '12 M']

# Function for Stacked Plotting

def stacked_area_chart(full_sector_df, year_start, year_end, category, top_n_sectors, abs_or_perc):
    
    # Select Category
    
    translate_dict = {
        "Gesamt ODA": "amount",
        "Klimarelevant": "clim_rel",
        "Klimaschutz": "clim_miti",
        "Klimaanpassung": "clim_adapt"
    }
    
    selected_cols = [col for col in full_sector_df.columns if col.startswith(translate_dict[category])]
    selected_cols.append('Sector')
    filtered_col_df = full_sector_df[selected_cols]

    # Select Top_N
    
    sector_avg = filtered_col_df.melt(id_vars=['Sector'], var_name='Year', value_name='Amount')[['Amount', 'Sector']].groupby('Sector').mean()
        
    top_sectors = sector_avg['Amount'].nlargest(top_n_sectors).index.tolist()
    filtered_col_df['Grouped Sector'] = filtered_col_df['Sector'].apply(lambda x: x if x in top_sectors else 'Andere Sektoren')
    grouped_df = filtered_col_df.groupby('Grouped Sector').sum().reset_index()
    
    # Prepare for Plotting
    
    long_df = grouped_df.melt(id_vars=['Grouped Sector'], var_name='Year', value_name='Amount')
    long_df['Year'] = long_df['Year'].str.extract('(\d+)')
    long_df['Year'] = pd.to_numeric(long_df['Year'])

    long_df =  long_df.dropna()
    
    # Year Select
    
    long_df = long_df[long_df['Year'].between(year_start,year_end)]
    

    # Option for Percentage Plot
    
    if abs_or_perc == 'Anteilig':
        
        # Calculate Percent
        
        total_per_year = long_df.groupby('Year')['Amount'].sum().reset_index(name='Total')


        long_df = pd.merge(long_df, total_per_year, on='Year')
        long_df['Percentage'] = (long_df['Amount'] / long_df['Total']) * 100
        
        # Create a filled area plot

        fig = px.area(long_df, x='Year', y='Percentage', color='Grouped Sector')
        # ,
        #               labels={'Percentage': 'Percentage of Total'},
        #               title='Stacked Area Plot of Grouped Sector as Percentage of Total per Year')

        # fig.update_layout(
        #     paper_bgcolor='white',
        #     plot_bgcolor='white',
        #     xaxis=dict(showgrid=False),
        #     yaxis=dict(showgrid=False, ticksuffix="%")  # Add a percentage sign to y-axis ticks
        # )

    
        return fig

    # Create a filled area plot
    
    fig = px.area(long_df, x='Year', y='Amount', color='Grouped Sector')
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        )
    
    return fig


# Overview

st.title('BMZ Klima-Dashboard')

page = st.sidebar.selectbox('Seitenauswahl', ['Gesamtübersicht', 'Ländervergleich', 'Länderanalyse', 'Sektoranalyse Global', 'Sektoranalyse pro Land','Methodik Erklärung'])



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

# Get all country split df

all_country_split_df = pd.read_csv('upload_data/all_country_split.csv')

# Get all country download df

all_country_df_download = pd.read_csv("upload_data/all_country_df_download.csv")


# Country compare DF

country_compare_df = pd.read_csv('upload_data/country_specific_df.csv')

# Sektoranalyse DF

sector_df = pd.read_csv('upload_data/sector_analysis.csv')

# Sektoranalyse pro Land DF

sector_per_country_df = pd.read_csv('upload_data/country_sector_analysis.csv')


# Display Results

if page == 'Gesamtübersicht':

    st.write(""" Die Analyse basiert auf Daten aus der OECD CRS Datenbank bis 
             einschließlich 2022, die Deutschlands bilaterale Entwicklungszusammenarbeit 
             dokumentieren. Betrachtet wurden die Offiziellen Entwicklungsassistenz-
             Auszahlungen (ODA) des Bundesministeriums für wirtschaftliche Zusammenarbeit 
             und Entwicklung (BMZ), angegeben in Millionen USD und zu konstanten Preisen.
            Für weitere Informationen siehe die Methodikerklärungsseite.""")

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

    st.markdown(f'<p class="intermediate-font">Die globale prozentuale Änderung der Klimafinanzierung zwischen {from_year} und {to_year} betrug {change_perc}%</p>', unsafe_allow_html=True)


    # Globe Stacked 
    
    globe_df = globe_df[globe_df['Year'].between(from_year,to_year)]

    fig_globe_bar = px.bar(globe_df, x='Year', y='Amount', color='Finanzierungstyp',
                title='Globale Finanzierungssummen',
                labels={'Amount': 'Finanzierungssumme ($)', 'Year': 'Jahr'},
                category_orders={'Finanzierungstyp': ['Andere ODA','Klimafinanzierung']},
                color_discrete_map={'Andere ODA': 'orange', 'Klimafinanzierung': 'green'})# This ensures consistent color ordering

    fig_globe_bar.update_layout(
        title_x=0.35,
        # Adjust margins to ensure plot utilizes more space and nothing is cut off
        margin=dict(l=20, r=20, t=50, b=20),
        # Adjust the legend's position if necessary
        legend=dict(
            x=0,  # Legend x position to the left
            xanchor='left',  # Anchor to the left side
            y=1,  # Top of the plot
            yanchor='top'  # Anchor to the top
        )
    )

    # Display the plot and allow it to use full container width
    st.plotly_chart(fig_globe_bar, use_container_width=True)

    # Globe Split

    chart_type = st.radio(
    "Wählen Sie eine Visualisierungsart",
    ["Anteilig", "Absolute Werte"],
    )

    split_df = split_df[split_df['Year'].between(from_year,to_year)]

    if chart_type == "Absolute Werte":


        fig_split = px.bar(split_df, x='Year', y='Amount', color='Finanzierungstyp',
                title='Globale Finanzierungssummen aufgeteilt',
                labels={'Amount': 'Finanzierungssumme ($)', 'Year': 'Jahr'},
                category_orders={'Finanzierungstyp': ['Andere ODA','Klimaschutz Finanzierung', 'Klimaanpassung Finanzierung']},
                color_discrete_map={'Andere ODA': 'orange', 'Klimaschutz Finanzierung': 'green', 'Klimaanpassung Finanzierung': 'blue'}
                )

        fig_split.update_layout(title_x=0.5)

            # Update the y-axis to display values in billions ('M' for Milliarden)
        fig_split.update_yaxes(tickprefix="", ticksuffix="",
                    tickvals=tick_values,
                    ticktext=tick_labels)

        fig_split.update_layout(
            title_x=0.35,
            # Adjust margins to ensure plot utilizes more space and nothing is cut off
            margin=dict(l=20, r=20, t=50, b=20),
            # Adjust the legend's position if necessary
            legend=dict(
                x=0,  # Legend x position to the left
                xanchor='left',  # Anchor to the left side
                y=1,  # Top of the plot
                yanchor='top'  # Anchor to the top
            ))

        st.plotly_chart(fig_split)

    if chart_type == "Anteilig":

        total_per_year = split_df.groupby('Year')['Amount'].transform('sum')
        split_df['Percentage'] = (split_df['Amount'] / total_per_year) * 100

        fig_split_percent = px.bar(split_df, x='Year', y='Percentage', color='Finanzierungstyp',
            title='Anteil Klimaschutz und Klimaanpassung an ODA-Auszahlungen',
            labels={'Percentage': 'Prozentsatz der Finanzierung', 'Year': 'Jahr'},
            category_orders={'Finanzierungstyp': ['Andere ODA', 'Klimaschutz Finanzierung', 'Klimaanpassung Finanzierung']},
            color_discrete_map={'Andere ODA': 'orange', 'Klimaschutz Finanzierung': 'green', 'Klimaanpassung Finanzierung': 'blue'}
            )

        fig_split_percent.update_layout(
            title_x=0.15,
            # Adjust margins to ensure plot utilizes more space and nothing is cut off
            margin=dict(l=20, r=20, t=50, b=20),
            # Adjust the legend's position if necessary
            legend=dict(
                x=0,  # Legend x position to the left
                xanchor='left',  # Anchor to the left side
                y=0,  # Top of the plot
                yanchor='bottom'  # Anchor to the top
            ))

        st.plotly_chart(fig_split_percent)


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
                'text': "Jährliche Veränderung der Klimafinanzierung (%)",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis = {"type":"category"},
            yaxis = {"title":"Prozent"},
    )

    st.plotly_chart(fig_globe_waterfall)

    # Sector View

    st.markdown(f'<p class="intermediate-font">Sektorübersicht erstellen</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        category = st.selectbox("Finanzierungstyp auswählen", options=["Gesamt ODA", "Klimarelevant", "Klimaschutz", "Klimaanpassung"])

    with col2:
        top_n_sectors = st.slider("Top N Sektoren anzeigen", min_value=2, max_value=10, value=5)

    with col3:
        abs_or_perc = st.radio("Visualisierungsart", options=['Anteilig', 'Absolute Werte'])


    if st.button("Sektorübersicht erstellen"):

        show_fig = stacked_area_chart(sector_df, from_year, to_year, category, top_n_sectors, abs_or_perc)
        # st.dataframe(show_fig)

        st.plotly_chart(show_fig)
        st.write(f"Flächendiagramm generiert von {from_year} bis {to_year} für {category}, {top_n_sectors} Top-Sektoren, Anzeigeart: {abs_or_perc}.")


    # DF to show

    st.markdown(f'<p class="intermediate-font">Daten zum Download:</p>', unsafe_allow_html=True)


    full_globe_df = full_globe_df[full_globe_df['Jahr'].between(from_year,to_year)]

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
        'Welche Länder möchten Sie summieren?',
        countries,
        ['India'])
    

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
    melted_df['Finanzierungstyp'] = melted_df['Type_Year'].apply(lambda x: 'Klimafinanzierung' if 'clim_rel_amount' in x else 'Andere ODA')

    melted_df['Amount'] = melted_df['Amount'] * 1_000_000

    fig_sel_bar = px.bar(melted_df, x='Year', y='Amount', color='Finanzierungstyp',
                title='Finanzierungssummen für ausgewählte Länder',
                labels={'Amount': 'Finanzierungssumme ($)', 'Year': 'Jahr'},
                category_orders={'Finanzierungstyp': ['Andere ODA','Klimafinanzierung']},
                color_discrete_map={'Andere ODA': 'orange', 'Klimafinanzierung': 'green'})# This ensures consistent color ordering

    fig_sel_bar.update_layout(
            title_x=0.35,
            # Adjust margins to ensure plot utilizes more space and nothing is cut off
            margin=dict(l=20, r=20, t=50, b=20),
            # Adjust the legend's position if necessary
            legend=dict(
                x=0,  # Legend x position to the left
                xanchor='left',  # Anchor to the left side
                y=1,  # Top of the plot
                yanchor='top'  # Anchor to the top
            ))


    st.plotly_chart(fig_sel_bar)

    # All country Split

    chart_type = st.radio(
    "Wählen Sie eine Visualisierungsart",
    ["Anteilig", "Absolute Werte"],
    )

    #Filter by country & Year
    all_country_split_df = all_country_split_df[all_country_split_df['Recipient Name'].isin(selected_countries)]
    all_country_split_df = all_country_split_df[all_country_split_df['Year'].between(from_year,to_year)]

    if chart_type == "Absolute Werte":


        fig_all_split = px.bar(all_country_split_df, x='Year', y='Amount', color='Finanzierungstyp',
                title='Finanzierungssummen ausgewählte Länder aufgeteilt',
                labels={'Amount': 'Finanzierungssumme ($)', 'Year': 'Jahr'},
                category_orders={'Finanzierungstyp': ['Andere ODA','Klimaschutz Finanzierung', 'Klimaanpassung Finanzierung']},
                color_discrete_map={'Andere ODA': 'orange', 'Klimaschutz Finanzierung': 'green', 'Klimaanpassung Finanzierung': 'blue'}
                )


        fig_all_split.update_layout(
            title_x=0.35,
            # Adjust margins to ensure plot utilizes more space and nothing is cut off
            margin=dict(l=20, r=20, t=50, b=20),
            # Adjust the legend's position if necessary
            legend=dict(
                x=0,  # Legend x position to the left
                xanchor='left',  # Anchor to the left side
                y=1,  # Top of the plot
                yanchor='top'  # Anchor to the top
            ))

        st.plotly_chart(fig_all_split)

    if chart_type == "Anteilig":

        total_per_year = all_country_split_df.groupby('Year')['Amount'].transform('sum')
        all_country_split_df['Percentage'] = (all_country_split_df['Amount'] / total_per_year) * 100

        fig_all_split_percent = px.bar(all_country_split_df, x='Year', y='Percentage', color='Finanzierungstyp',
            title='Anteil Klimaschutz und Klimaanpassung an ODA-Auszahlungen',
            labels={'Percentage': 'Prozentsatz der Finanzierung', 'Year': 'Jahr'},
            category_orders={'Finanzierungstyp': ['Andere ODA', 'Klimaschutz Finanzierung', 'Klimaanpassung Finanzierung']},
            color_discrete_map={'Andere ODA': 'orange', 'Klimaschutz Finanzierung': 'green', 'Klimaanpassung Finanzierung': 'blue'}
            )

        fig_all_split_percent.update_layout(
            title_x=0.15,
            # Adjust margins to ensure plot utilizes more space and nothing is cut off
            margin=dict(l=20, r=20, t=50, b=20),
            # Adjust the legend's position if necessary
            legend=dict(
                x=0,  # Legend x position to the left
                xanchor='left',  # Anchor to the left side
                y=0,  # Top of the plot
                yanchor='bottom'  # Anchor to the top
            ))


        st.plotly_chart(fig_all_split_percent)


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
                'text': "Jährliche Veränderung der Klimafinanzierung (%) für ausgewählte Länder",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis = {"type":"category"},
            yaxis = {"title":"Prozent"},
    )

    st.plotly_chart(fig_sel_waterfall)

    st.markdown(f'<p class="intermediate-font">Daten zum Download:</p>', unsafe_allow_html=True)

    all_country_df_download = all_country_df_download[all_country_df_download['Empfängerland'].isin(selected_countries)]

    st.dataframe(all_country_df_download)

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
                title='Anteil der Klimafinanzierung am gesamten Portfolio in ausgewählten Ländern',
                labels={'Value': 'Anteil der klimarelevanten Finanzierung (%)', 'Year': 'Jahr'},
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

    ranked_df = pd.DataFrame(values_countries).rename(columns={'Value': 'Prozentuale Veränderung im Zeitraum'})\
            .sort_values('Prozentuale Veränderung im Zeitraum', ascending = False)

    st.markdown('<p class="intermediate-font">Prozentuale Veränderung der Klimafinanzierung im ausgewählten Zeitraum</p>', unsafe_allow_html=True)

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

    ranked_df = pd.DataFrame(values_countries).rename(columns={'Value': 'Prozentuale Veränderung'})\
            .sort_values('Prozentuale Veränderung', ascending = False)

    st.markdown('<p class="intermediate-font">Länder mit größtem Zuwachs bzw. Rückgang an Klimafinanzierungsantiel im ausgewählten Zeitraum</p>', unsafe_allow_html=True)

    figu1 , figu2 = st.columns(2)

    with figu1:
            st.markdown('<p class="intermediate-font">Länder mit dem größten prozentualen Zuwachs</p>', unsafe_allow_html=True)

            st.dataframe(ranked_df.dropna().head(10), use_container_width = True)

        
    with figu2:
        st.markdown('<p class="intermediate-font">Länder mit größten prozentualen Rückgang</p>', unsafe_allow_html=True)

        st.dataframe(ranked_df.dropna().tail(10),use_container_width = True)


if page == 'Sektoranalyse Global':

    st.header('Sektoranalyse Global')


    selected_year = st.slider(
    'Jahr auswählen',
    min_value=2013,
    max_value=2022,
    value=2022
    )

    year_select = [col for col in sector_df.columns if col.endswith(f'{selected_year}')]
    year_select.append('Sector')
    filtered_year_df = sector_df[year_select]

    clim_rel_fig = px.sunburst(filtered_year_df, path=['Sector'], values=f'clim_rel_amount_{selected_year}', title='Sektoraufteilung klimarelevante Finanzierung')
    non_clim_fig = px.sunburst(filtered_year_df, path=['Sector'], values=f'non_clim_{selected_year}', title='Sektoraufteilung nicht-klimarelevante Finanzierung')
    clim_adapt_fig = px.sunburst(filtered_year_df, path=['Sector'], values=f'clim_adapt_amount_{selected_year}', title="Sektoraufteilung Klimaanpassungsfinanzierung")
    clim_miti_fig = px.sunburst(filtered_year_df, path=['Sector'], values=f'clim_miti_amount_{selected_year}', title="Sektoraufteilung Klimaschutzfinanzierung")


    # Customize margins to reduce unnecessary space
    layout_update = {
        'margin': dict(t=30, l=0, r=0, b=0)  # Adjust top, left, right, bottom margins as needed
    }

    clim_rel_fig.update_layout(**layout_update)
    non_clim_fig.update_layout(**layout_update)
    clim_adapt_fig.update_layout(**layout_update)
    clim_miti_fig.update_layout(**layout_update)


    figu1 , figu2 = st.columns(2)

    with figu1:
        st.plotly_chart(clim_rel_fig, use_container_width=True)
        
    with figu2:
        st.plotly_chart(non_clim_fig, use_container_width=True)

    figu3 , figu4 = st.columns(2)

    with figu3:
        st.plotly_chart(clim_adapt_fig, use_container_width=True)
        
    with figu4:
        st.plotly_chart(clim_miti_fig, use_container_width=True)



if page == 'Sektoranalyse pro Land':

    st.header('Sektoranalyse pro Land')

    selected_year = st.slider(
    'Jahr auswählen',
    min_value=2013,
    max_value=2022,
    value=2022
    )
    countries_list = countries.tolist()

    selected_country = st.selectbox(
        'Welches Land möchten Sie einblicken?',
        countries_list,
        index=countries_list.index('India') if 'India' in countries_list else 0)

    year_select = [col for col in sector_per_country_df.columns if col.endswith(f'{selected_year}')]
    year_select.append('Sector')
    df_merged = sector_per_country_df[sector_per_country_df['Recipient Name'] == selected_country]

    filtered_year_df = df_merged[year_select]


    clim_rel_fig = px.sunburst(filtered_year_df, path=['Sector'], values=f'clim_rel_amount_{selected_year}', title='Sektoraufteilung klimarelevante Finanzierung')
    non_clim_fig = px.sunburst(filtered_year_df, path=['Sector'], values=f'non_clim_{selected_year}', title='Sektoraufteilung nicht-klimarelevante Finanzierung')
    clim_adapt_fig = px.sunburst(filtered_year_df, path=['Sector'], values=f'clim_adapt_amount_{selected_year}', title='Sektoraufteilung Klimaanpassungsfinanzierung')
    clim_miti_fig = px.sunburst(filtered_year_df, path=['Sector'], values=f'clim_miti_amount_{selected_year}', title='Sektoraufteilung Klimaschutzfinanzierung')

    # Customize margins to reduce unnecessary space
    layout_update = {
        'margin': dict(t=30, l=0, r=0, b=0)  # Adjust top, left, right, bottom margins as needed
    }

    clim_rel_fig.update_layout(**layout_update)
    non_clim_fig.update_layout(**layout_update)
    clim_adapt_fig.update_layout(**layout_update)
    clim_miti_fig.update_layout(**layout_update)

    figu1 , figu2 = st.columns(2)

    with figu1:
        st.plotly_chart(clim_rel_fig, use_container_width=True)
        
    with figu2:
        st.plotly_chart(non_clim_fig, use_container_width=True)

    figu3 , figu4 = st.columns(2)

    with figu3:
        st.plotly_chart(clim_adapt_fig, use_container_width=True)
        
    with figu4:
        st.plotly_chart(clim_miti_fig, use_container_width=True)

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
    


