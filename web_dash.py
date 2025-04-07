import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the merged DataFrame from the pickle file
merged_df = pd.read_pickle("/home/Ifena/merged.pkl")

# If the 'region' column isn’t already in your pickled DataFrame,
# you can create it using your region mapping.
region_mapping = {
    "Beijing": "East", "Tianjin": "East", "Liaoning": "East", "Shanghai": "East", "Jiangsu": "East",
    "Zhejiang": "East", "Fujian": "East", "Shandong": "East", "Guangdong": "East",
    "Hebei": "Central", "Shanxi": "Central", "Heilongjiang": "Central", "Jilin": "Central",
    "Anhui": "Central", "Jiangxi": "Central", "Henan": "Central", "Hubei": "Central", "Hunan": "Central", "Hainan": "Central",
    "Inner Mongolia": "West", "Guangxi": "West", "Chongqing": "West", "Sichuan": "West", "Guizhou": "West", "Yunnan": "West",
    "Tibet": "West", "Shaanxi": "West", "Gansu": "West", "Qinghai": "West", "Ningxia": "West", "Xinjiang": "West"
}
if 'region' not in merged_df.columns:
    merged_df["region"] = merged_df["Province"].map(region_mapping)

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "China Museum Dashboard"

# Create dropdown options based on unique province names
province_options = [{'label': prov, 'value': prov} for prov in sorted(merged_df['Province'].unique())]

# Define a color mapping for regions
region_color_map = {
    "East": "#4682B4",
    "Central": "orange",
    "West": "green"
}

# Define the layout of the app
app.layout = html.Div([
    html.H2("China Museum Data Dashboard (2016–2022)", style={'textAlign': 'center'}),
    html.Label("Select a Province:"),
    dcc.Dropdown(id='province-dropdown', options=province_options, value='Beijing'),
    html.Div(id='charts-container')
])

# Callback to update charts when a province is selected
@app.callback(
    Output('charts-container', 'children'),
    [Input('province-dropdown', 'value')]
)
def update_charts(province):
    df = merged_df[merged_df['Province'] == province].sort_values('Year')
    if df.empty:
        return html.Div("No data available for the selected province.")

    region = df['region'].iloc[0]
    color = region_color_map.get(region, 'gray')

    # Create the four charts using Plotly Express
    fig1 = px.line(df, x="Year", y="GDP_per_capita", title="GDP per Capita", markers=True)
    fig1.update_traces(line=dict(color=color))

    fig2 = px.bar(df, x="Year", y="Subsidy_per_sqm", title="Subsidy per Square Meter", color_discrete_sequence=[color])

    fig3 = px.line(df, x="Year", y="Museum size (10^6 ㎡)", title="Museum Size", markers=True)
    fig3.update_traces(line=dict(color=color))

    fig4 = px.bar(df, x="Year", y="Expenditure_per_sqm", title="Expenditure per Square Meter", color_discrete_sequence=[color])

    return html.Div([
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
        dcc.Graph(figure=fig3),
        dcc.Graph(figure=fig4)
    ])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)