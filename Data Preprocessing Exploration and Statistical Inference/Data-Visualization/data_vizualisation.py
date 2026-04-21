
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Dashboard imports
import panel as pn
import hvplot.pandas

# Initialize Panel extension
pn.extension('plotly', sizing_mode='stretch_width')

# Set styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Data Acquisition
# Our World in Data Energy Dataset
def load_energy_data(filepath: str = None) -> pd.DataFrame:
    if filepath and Path(filepath).exists():
        df = pd.read_csv(filepath)
    else:
        # Download from Our World in Data
        url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
        print(f"Downloading data from {url}...")
        df = pd.read_csv(url)
        
        # Optionally save locally
        if not os.path.exists(filepath):
            os.makedirs('./data')
        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")
    
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    print(f"Countries/regions: {df['country'].nunique()}")
    
    return df

# Load the data
df_raw = load_energy_data('data/global_energy.csv')

# Data Exploration and Cleaning

def explore_data(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATA EXPLORATION SUMMARY")
    print("=" * 60)
    
    # Basic info
    print(f"\nShape: {df.shape}")
    print(f"\nColumn Types:")
    print(df.dtypes.value_counts())
    
    # Key columns for our analysis
    key_columns = [
        'country', 'year', 'iso_code', 'population', 'gdp',
        'primary_energy_consumption',
        'fossil_fuel_consumption', 'renewables_consumption',
        'nuclear_consumption', 'coal_consumption',
        'oil_consumption', 'gas_consumption',
        'solar_consumption', 'wind_consumption', 'hydro_consumption',
        'carbon_intensity_elec'
    ]
    
    available = [c for c in key_columns if c in df.columns]
    missing_cols = [c for c in key_columns if c not in df.columns]
    
    print(f"\nKey columns available: {len(available)}/{len(key_columns)}")
    if missing_cols:
        print(f"Missing: {missing_cols}")
    
    # Missing value analysis
    print("\nMissing Values (key columns):")
    missing_pct = (df[available].isnull().sum() / len(df) * 100).sort_values(ascending=False)
    print(missing_pct[missing_pct > 0].head(10))

explore_data(df_raw)

def clean_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Filter to actual countries (have ISO codes, not aggregates)
    aggregates = [
        'World', 'Africa', 'Asia', 'Europe', 'North America', 'South America',
        'Oceania', 'European Union', 'OECD', 'Non-OECD', 'High-income countries',
        'Low-income countries', 'Middle-income countries', 'Upper-middle-income countries',
        'Lower-middle-income countries'
    ]
    df = df[~df['country'].isin(aggregates)]
    df = df[df['iso_code'].notna()]
    df = df[df['iso_code'].str.len() == 3]  # Valid ISO codes
    
    # 2. Select key columns
    columns = [
        'country', 'year', 'iso_code', 'population', 'gdp',
        'primary_energy_consumption',
        'fossil_fuel_consumption', 'renewables_consumption',
        'nuclear_consumption', 'coal_consumption',
        'oil_consumption', 'gas_consumption',
        'solar_consumption', 'wind_consumption', 'hydro_consumption',
        'carbon_intensity_elec'
    ]
    available_columns = [c for c in columns if c in df.columns]
    df = df[available_columns]
    
    # 3. Focus on recent decades with better data coverage
    df = df[df['year'] >= 1990]
    
    # 4. Add derived metrics
    if 'primary_energy_consumption' in df.columns and 'population' in df.columns:
        df['energy_per_capita'] = df['primary_energy_consumption'] / (df['population'] / 1e6)
    
    if 'gdp' in df.columns and 'primary_energy_consumption' in df.columns:
        df['energy_intensity'] = df['primary_energy_consumption'] / (df['gdp'] / 1e9)
    
    if 'renewables_consumption' in df.columns and 'primary_energy_consumption' in df.columns:
        df['renewable_share'] = (df['renewables_consumption'] / 
                                  df['primary_energy_consumption'] * 100)
    
    print(f"Cleaned data: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    
    return df

df_clean = clean_energy_data(df_raw)
df_clean.head()

# Static Visualizations with Matplotlib and Seaborn
def create_global_energy_trends(df: pd.DataFrame) -> plt.Figure:
    # Aggregate globally by year
    global_yearly = df.groupby('year').agg({
        'primary_energy_consumption': 'sum',
        'fossil_fuel_consumption': 'sum',
        'renewables_consumption': 'sum',
        'nuclear_consumption': 'sum',
        'coal_consumption': 'sum',
        'oil_consumption': 'sum',
        'gas_consumption': 'sum'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Total Primary Energy Consumption
    ax = axes[0, 0]
    ax.fill_between(
        global_yearly['year'], 
        global_yearly['primary_energy_consumption'] / 1000,
        alpha=0.3, color='steelblue'
    )
    ax.plot(
        global_yearly['year'], 
        global_yearly['primary_energy_consumption'] / 1000,
        linewidth=2, color='steelblue'
    )
    ax.set_xlabel('Year')
    ax.set_ylabel('Energy Consumption (PWh)')
    ax.set_title('Global Primary Energy Consumption', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    
    # 2. Energy Mix Evolution (Stacked Area)
    ax = axes[0, 1]
    energy_sources = ['coal_consumption', 'oil_consumption', 'gas_consumption', 
                      'nuclear_consumption', 'renewables_consumption']
    labels = ['Coal', 'Oil', 'Natural Gas', 'Nuclear', 'Renewables']
    colors = ['#4a4a4a', '#2d5a27', '#ff7f0e', '#9467bd', '#2ca02c']
    
    # Stack the values
    stack_data = [global_yearly[col].fillna(0) / 1000 for col in energy_sources]
    ax.stackplot(global_yearly['year'], stack_data, labels=labels, colors=colors, alpha=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Energy Consumption (PWh)')
    ax.set_title('Global Energy Mix Evolution', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    
    # 3. Top 10 Countries by Energy Consumption (Latest Year)
    ax = axes[1, 0]
    latest_year = 2024 # 2025 does not contain enough data
    top_countries = (
        df[df['year'] == latest_year]
        .nlargest(10, 'primary_energy_consumption')
        [['country', 'primary_energy_consumption']]
    )
    
    bars = ax.barh(
        top_countries['country'], 
        top_countries['primary_energy_consumption'] / 1000,
        color='steelblue'
    )
    ax.set_xlabel('Primary Energy Consumption (PWh)')
    ax.set_title(f'Top 10 Countries by Energy Consumption ({latest_year})', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.invert_yaxis()
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}', va='center', fontsize=9)
    
    # 4. Renewable Share by Region (Box Plot)
    ax = axes[1, 1]
    
    # Add continent/region mapping (simplified)
    region_mapping = {
        'United States': 'Americas', 'Canada': 'Americas', 'Brazil': 'Americas',
        'Mexico': 'Americas', 'Argentina': 'Americas',
        'China': 'Asia', 'India': 'Asia', 'Japan': 'Asia', 
        'South Korea': 'Asia', 'Indonesia': 'Asia',
        'Germany': 'Europe', 'France': 'Europe', 'United Kingdom': 'Europe',
        'Italy': 'Europe', 'Spain': 'Europe', 'Poland': 'Europe',
        'Russia': 'Europe', 'Turkey': 'Europe',
        'Australia': 'Oceania', 'New Zealand': 'Oceania',
        'South Africa': 'Africa', 'Egypt': 'Africa', 'Nigeria': 'Africa',
        'Saudi Arabia': 'Middle East', 'Iran': 'Middle East', 
        'United Arab Emirates': 'Middle East'
    }
    
    df_latest = df[df['year'] == latest_year].copy()
    df_latest['region'] = df_latest['country'].map(region_mapping)
    df_latest = df_latest.dropna(subset=['region', 'renewable_share'])
    
    if not df_latest.empty:
        regions = ['Americas', 'Europe', 'Asia', 'Middle East', 'Africa', 'Oceania']
        box_data = [df_latest[df_latest['region'] == r]['renewable_share'].dropna() 
                    for r in regions if r in df_latest['region'].values]
        box_labels = [r for r in regions if r in df_latest['region'].values]
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors_box = plt.cm.Set2(np.linspace(0, 1, len(box_data)))
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_ylabel('Renewable Share (%)')
    ax.set_title(f'Renewable Energy Share by Region ({latest_year})', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    dir = "./images"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + '/global_energy_trends.png', dpi=150, bbox_inches='tight')
    
    return fig

# Create the visualization
fig = create_global_energy_trends(df_clean)
plt.show()

def create_energy_per_capita_analysis(df: pd.DataFrame) -> plt.Figure:
    latest_year = 2022 # 2023 does not have enough data
    df_latest = df[df['year'] == latest_year].dropna(subset=['energy_per_capita', 'gdp', 'population'])
    print(df_latest)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot with size = population, color = renewable share
    scatter = ax.scatter(
        df_latest['gdp'] / 1e12,  # Convert to trillions
        df_latest['energy_per_capita'],
        s=df_latest['population'] / 1e7,  # Scale for visibility
        c=df_latest['renewable_share'],
        cmap='RdYlGn',
        alpha=0.6,
        edgecolors='white',
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Renewable Share (%)', fontsize=10)
    
    # Label major countries
    major_countries = ['United States', 'China', 'India', 'Germany', 'Japan', 
                       'Brazil', 'Russia', 'United Kingdom', 'France', 'Canada']
    for country in major_countries:
        row = df_latest[df_latest['country'] == country]
        if not row.empty:
            ax.annotate(
                country,
                (row['gdp'].values[0] / 1e12, row['energy_per_capita'].values[0]),
                fontsize=8,
                alpha=0.8,
                xytext=(5, 5),
                textcoords='offset points'
            )
    
    ax.set_xlabel('GDP (Trillion USD)', fontsize=11)
    ax.set_ylabel('Energy per Capita (kWh/person)', fontsize=11)
    ax.set_title(f'Energy Consumption vs. Economic Development ({latest_year})\n'
                 f'Size = Population, Color = Renewable Share',
                 fontweight='bold', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    dir = "./images"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + '/energy_per_capita.png', dpi=150, bbox_inches='tight')
    
    return fig

fig = create_energy_per_capita_analysis(df_clean)
plt.show()

# Interactive Visualizations with Plotly
def create_interactive_energy_explorer(df: pd.DataFrame) -> go.Figure:
    # Prepare data - use 5-year intervals for smoother animation
    df_anim = df[df['year'].isin(range(1990, 2025, 5))].copy()
    df_anim = df_anim.dropna(subset=['energy_per_capita', 'gdp', 'population', 'renewable_share'])
    
    # Filter to countries with sufficient data
    country_counts = df_anim.groupby('country').size()
    valid_countries = country_counts[country_counts >= 3].index
    df_anim = df_anim[df_anim['country'].isin(valid_countries)]
    
    fig = px.scatter(
        df_anim,
        x='gdp',
        y='energy_per_capita',
        size='population',
        color='renewable_share',
        hover_name='country',
        animation_frame='year',
        animation_group='country',
        size_max=60,
        color_continuous_scale='RdYlGn',
        range_color=[0, 50],
        log_x=True,
        log_y=True,
        title='Global Energy Evolution: GDP vs Energy per Capita',
        labels={
            'gdp': 'GDP (USD)',
            'energy_per_capita': 'Energy per Capita (kWh/person)',
            'population': 'Population',
            'renewable_share': 'Renewable Share (%)'
        }
    )
    
    fig.update_layout(
        height=600,
        font=dict(family='Arial', size=12),
        title_font_size=16,
        title_x=0.5,
    )
    
    # Slow down animation
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1500
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 800
    
    return fig

fig = create_interactive_energy_explorer(df_clean)
fig.show()

# Sunburst chart showing energy mix hierarchy.
def create_energy_mix_sunburst(df: pd.DataFrame) -> go.Figure:
    latest_year = 2024 # 2025 does not have enough data

    # Aggregate by major countries
    top_countries = (
        df[df['year'] == latest_year]
        .nlargest(8, 'primary_energy_consumption')['country']
        .tolist()
    )

    # Prepare hierarchical data
    records = []
    energy_types = [
        ('Fossil Fuels', 'fossil_fuel_consumption', 
         [('Coal', 'coal_consumption'), ('Oil', 'oil_consumption'), ('Gas', 'gas_consumption')]),
        ('Renewables', 'renewables_consumption',
         [('Solar', 'solar_consumption'), ('Wind', 'wind_consumption'), ('Hydro', 'hydro_consumption')]),
        ('Nuclear', 'nuclear_consumption', [])
    ]
    
    for country in top_countries:
        country_data = df[(df['year'] == latest_year) & (df['country'] == country)].iloc[0]
        print(country_data)
        for energy_cat, cat_col, subtypes in energy_types:
            cat_value = country_data.get(cat_col, 0) or 0
            
            if subtypes:
                for subtype_name, subtype_col in subtypes:
                    subtype_value = country_data.get(subtype_col, 0) or 0
                    if subtype_value > 0:
                        records.append({
                            'country': country,
                            'category': energy_cat,
                            'source': subtype_name,
                            'value': subtype_value
                        })
            elif cat_value > 0:
                records.append({
                    'country': country,
                    'category': energy_cat,
                    'source': energy_cat,
                    'value': cat_value
                })
    
    df_sunburst = pd.DataFrame(records)

    fig = px.sunburst(
        df_sunburst,
        path=['country', 'category', 'source'],
        values='value',
        color='category',
        color_discrete_map={
            'Fossil Fuels': '#4a4a4a',
            'Renewables': '#2ca02c',
            'Nuclear': '#9467bd'
        },
        title=f'Energy Mix by Country and Source ({latest_year})'
    )
    
    fig.update_layout(
        height=600,
        font=dict(family='Arial', size=12),
        title_font_size=16,
        title_x=0.5,
    )
    
    return fig

fig = create_energy_mix_sunburst(df_clean)
fig.show()

# Faceted comparison of energy metrics for selected countries.
def create_country_comparison(df: pd.DataFrame, countries: list) -> go.Figure:
    df_subset = df[df['country'].isin(countries)].copy()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Primary Energy Consumption',
            'Renewable Share',
            'Energy per Capita',
            'Carbon Intensity'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set2[:len(countries)]
    
    for i, country in enumerate(countries):
        country_data = df_subset[df_subset['country'] == country].sort_values('year')
        color = colors[i]
        
        # Primary Energy
        fig.add_trace(
            go.Scatter(
                x=country_data['year'],
                y=country_data['primary_energy_consumption'] / 1000,
                name=country,
                line=dict(color=color, width=2),
                mode='lines',
                legendgroup=country,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Renewable Share
        fig.add_trace(
            go.Scatter(
                x=country_data['year'],
                y=country_data['renewable_share'],
                name=country,
                line=dict(color=color, width=2),
                mode='lines',
                legendgroup=country,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Energy per Capita
        fig.add_trace(
            go.Scatter(
                x=country_data['year'],
                y=country_data['energy_per_capita'],
                name=country,
                line=dict(color=color, width=2),
                mode='lines',
                legendgroup=country,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Carbon Intensity
        if 'carbon_intensity_elec' in country_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=country_data['year'],
                    y=country_data['carbon_intensity_elec'],
                    name=country,
                    line=dict(color=color, width=2),
                    mode='lines',
                    legendgroup=country,
                    showlegend=False
                ),
                row=2, col=2
            )
    
    fig.update_xaxes(title_text='Year', row=2, col=1)
    fig.update_xaxes(title_text='Year', row=2, col=2)
    fig.update_yaxes(title_text='PWh', row=1, col=1)
    fig.update_yaxes(title_text='%', row=1, col=2)
    fig.update_yaxes(title_text='kWh/person', row=2, col=1)
    fig.update_yaxes(title_text='gCO2/kWh', row=2, col=2)
    
    fig.update_layout(
        height=700,
        title_text='Country Energy Comparison',
        title_x=0.5,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig

countries = ['United States', 'China', 'Germany', 'India', 'Brazil']
fig = create_country_comparison(df_clean, countries)
fig.show()

# Interactive Multi-View Dashboard with Panel

# Multi-view dashboard for energy data exploration.
def create_energy_dashboard(df: pd.DataFrame):
    
    # Prepare data
    latest_year = 2024  # 2025 does not have enough data
    countries = sorted(df['country'].unique())
    years = sorted(df['year'].unique())
    
    # --- Widgets ---
    country_select = pn.widgets.Select(
        name='Country',
        options=countries,
        value='United States'
    )
    
    year_slider = pn.widgets.IntSlider(
        name='Year',
        start=int(min(years)),
        end=int(max(years)),
        value=int(latest_year),
        step=1
    )
    
    metric_select = pn.widgets.Select(
        name='Metric',
        options={
            'Primary Energy': 'primary_energy_consumption',
            'Renewable Share': 'renewable_share',
            'Energy per Capita': 'energy_per_capita',
            'Carbon Intensity': 'carbon_intensity_elec'
        },
        value='primary_energy_consumption'
    )
    
    # --- Reactive Components ---
    
    @pn.depends(country_select, metric_select)
    def country_time_series(country, metric):
        """Time series for selected country and metric."""
        country_data = df[df['country'] == country].sort_values('year')
        
        if metric not in country_data.columns or country_data[metric].isna().all():
            return pn.pane.Markdown(f"No data available for {metric}")
        
        fig = px.line(
            country_data,
            x='year',
            y=metric,
            title=f'{country}: {metric_select.name}',
            markers=True
        )
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title='Year',
            yaxis_title=metric_select.value
        )
        return fig
    
    @pn.depends(country_select, year_slider)
    def energy_mix_chart(country, year):
        """Energy mix breakdown for selected country and year."""
        row = df[(df['country'] == country) & (df['year'] == year)]
        
        if row.empty:
            return pn.pane.Markdown("No data for this selection")
        
        row = row.iloc[0]
        
        sources = ['Coal', 'Oil', 'Gas', 'Nuclear', 'Solar', 'Wind', 'Hydro']
        columns = ['coal_consumption', 'oil_consumption', 'gas_consumption',
                   'nuclear_consumption', 'solar_consumption', 'wind_consumption',
                   'hydro_consumption']
        
        values = [row.get(col, 0) or 0 for col in columns]
        
        # Filter out zero values
        data = [(s, v) for s, v in zip(sources, values) if v > 0]
        if not data:
            return pn.pane.Markdown("No energy mix data available")
        
        sources, values = zip(*data)
        
        fig = px.pie(
            names=sources,
            values=values,
            title=f'{country} Energy Mix ({year})',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
    
    @pn.depends(country_select, year_slider)
    def key_metrics(country, year):
        """Key metrics cards for selected country and year."""
        row = df[(df['country'] == country) & (df['year'] == year)]
        
        if row.empty:
            return pn.pane.Markdown("No data available")
        
        row = row.iloc[0]
        
        def format_metric(value, suffix=''):
            if pd.isna(value):
                return 'N/A'
            if value >= 1e9:
                return f'{value/1e9:.1f}B{suffix}'
            if value >= 1e6:
                return f'{value/1e6:.1f}M{suffix}'
            if value >= 1e3:
                return f'{value/1e3:.1f}K{suffix}'
            return f'{value:.1f}{suffix}'
        
        cards = pn.Row(
            pn.indicators.Number(
                name='Population',
                value=row.get('population', 0) / 1e6,
                format='{value:.1f}M',
                default_color='steelblue'
            ),
            pn.indicators.Number(
                name='Energy Consumption',
                value=(row.get('primary_energy_consumption', 0) or 0) / 1000,
                format='{value:.1f} PWh',
                default_color='darkorange'
            ),
            pn.indicators.Number(
                name='Renewable Share',
                value=row.get('renewable_share', 0) or 0,
                format='{value:.1f}%',
                default_color='seagreen'
            ),
            pn.indicators.Number(
                name='Per Capita',
                value=row.get('energy_per_capita', 0) or 0,
                format='{value:.0f} kWh',
                default_color='purple'
            )
        )
        return cards
    
    @pn.depends(year_slider, metric_select)
    def global_comparison(year, metric):
        """Bar chart comparing top countries for selected metric."""
        year_data = df[df['year'] == year].dropna(subset=[metric])
        
        if year_data.empty:
            return pn.pane.Markdown("No data for this year")
        
        top_10 = year_data.nlargest(10, metric)[['country', metric]]
        
        fig = px.bar(
            top_10,
            x=metric,
            y='country',
            orientation='h',
            title=f'Top 10 Countries by {metric_select.name} ({year})',
            color=metric,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        return fig
    
    # --- Layout ---
    dashboard = pn.template.FastListTemplate(
        title='Global Energy Dashboard',
        sidebar=[
            pn.pane.Markdown('## Controls'),
            country_select,
            year_slider,
            metric_select,
            pn.pane.Markdown('''
            ---
            ### About
            Explore global energy consumption patterns,
            renewable adoption, and carbon intensity.
            
            Data: Our World in Data
            ''')
        ],
        main=[
            pn.Row(
                pn.Column(
                    pn.pane.Markdown('## Key Metrics'),
                    key_metrics,
                    width=800
                )
            ),
            pn.Row(
                pn.Column(
                    pn.pane.Plotly(country_time_series, sizing_mode='stretch_width'),
                    width=500
                ),
                pn.Column(
                    pn.pane.Plotly(energy_mix_chart, sizing_mode='stretch_width'),
                    width=450
                )
            ),
            pn.Row(
                pn.Column(
                    pn.pane.Plotly(global_comparison, sizing_mode='stretch_width'),
                    width=800
                )
            )
        ],
        accent_base_color='#2196F3',
        header_background='#1976D2'
    )
    
    return dashboard

# Create and serve the dashboard
dashboard = create_energy_dashboard(df_clean)
dashboard.servable()
# 
# To launch: panel serve data_vizualisation.py --show
