import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import subprocess
import datetime
import pytz
import os
import scipy.stats as stats


# Application Initialization
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
server = app.server

# Configuration Constants
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_PATH, "projet.csv")
REPORT_FILE = os.path.join(BASE_PATH, "daily_report.csv")
TZ_PARIS = pytz.timezone("Europe/Paris")
MAX_DATA_POINTS = 100

# Design Theme
COLORS = {
    "background": "#1C1C1E",
    "text": "#FFFFFF",
    "bitcoin": "#F7931A",
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "card_bg": "#2C2C2E",
    "grid": "#3A3A3C"
}

def ensure_files_exist():
    """Ensure required files exist."""
    for file_path in [DATA_FILE, REPORT_FILE]:
        if not os.path.exists(file_path):
            open(file_path, 'a').close()
            print(f"Created file: {file_path}")

def load_data():
    """Load data from CSV file with error handling."""
    try:
        df = pd.read_csv(DATA_FILE, names=["Timestamp", "Price"], header=None)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = df.dropna().sort_values("Timestamp")
        return df.tail(MAX_DATA_POINTS)
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return pd.DataFrame(columns=["Timestamp", "Price"])

def load_daily_report():
    """Load the daily report from the CSV file with precise timestamp."""
    try:
        cols = ["Timestamp", "Open", "Close", "Max", "Min", "Evolution"]
        df = pd.read_csv(REPORT_FILE, names=cols, header=None)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        # If no data, return a default/empty report
        if df.empty:
            now = datetime.datetime.now()
            return pd.Series({
                "Timestamp": now,
                "Open": 0,
                "Close": 0,
                "Max": 0,
                "Min": 0,
                "Evolution": "0%"
            })
        
        return df.iloc[-1]  # Return the most recent report
    except Exception as e:
        print(f"❌ Report loading error: {e}")
        return None

def calculate_volatility(df, window=14):
    """Calculate volatility as the standard deviation of daily returns."""
    if len(df) < 2:
        return 0
    
    # Calculate returns
    df = df.copy()
    df['return'] = df['Price'].pct_change()
    
    # Calculate rolling volatility (annualized)
    volatility = df['return'].rolling(window=min(window, len(df))).std()
    
    # Annualize (assuming daily data)
    annualized_vol = volatility.iloc[-1] * np.sqrt(365) if not volatility.empty else 0
    return annualized_vol

def calculate_var(df, confidence=0.95, window=14):
    """Calculate Value at Risk using historical method."""
    if len(df) < window:
        return 0
    
    # Calculate returns
    df = df.copy()
    df['return'] = df['Price'].pct_change()
    
    # Filter out NaN values
    returns = df['return'].dropna()
    
    if len(returns) < 2:
        return 0
    
    # Calculate VaR
    var = np.percentile(returns, 100 * (1 - confidence))
    
    # Convert to percentage and make it positive for display
    var_pct = abs(var * 100)
    return var_pct

def create_price_graph(df):
    """Create a visually enhanced and interactive price graph."""
    if df.empty:
        return go.Figure()

    lower_percentile = np.percentile(df["Price"], 5)
    upper_percentile = np.percentile(df["Price"], 95)

    min_price = df["Price"].min()
    max_price = df["Price"].max()
    min_timestamp = df[df["Price"] == min_price]["Timestamp"].iloc[0]
    max_timestamp = df[df["Price"] == max_price]["Timestamp"].iloc[0]

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df["Price"],
        mode='lines',
        name='Prix',
        line=dict(color=COLORS["bitcoin"], width=3),
        hovertemplate='Heure: %{x}<br>Prix: $%{y:.2f}<extra></extra>',
    ))

    # Highlight min and max
    fig.add_trace(go.Scatter(
        x=[min_timestamp],
        y=[min_price],
        mode='markers+text',
        name='Min',
        marker=dict(color=COLORS["negative"], size=10),
        text=[f"Min: ${min_price:.2f}"],
        textposition="top right",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[max_timestamp],
        y=[max_price],
        mode='markers+text',
        name='Max',
        marker=dict(color=COLORS["positive"], size=10),
        text=[f"Max: ${max_price:.2f}"],
        textposition="bottom left",
        showlegend=False
    ))

    fig.update_layout(
        title="Evolution du Prix du Bitcoin",
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(family="Inter", color=COLORS["text"]),
        xaxis=dict(
            title="Date & Heure",
            showgrid=True,
            gridcolor=COLORS["grid"],
            tickangle=-45,
            rangeslider=dict(visible=True),  # Slider for navigation
            type="date",
            tickfont=dict(color=COLORS["text"])
        ),
        yaxis=dict(
            title="Prix (USD)",
            showgrid=True,
            gridcolor=COLORS["grid"],
            tickprefix="$",
            range=[lower_percentile * 0.98, upper_percentile * 1.02],
            tickfont=dict(color=COLORS["text"])
        ),
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color=COLORS["text"])
        )
    )

    return fig

def create_volatility_graph(df, window=14):
    """Create a volatility graph based on price data."""
    if len(df) < window:
        return go.Figure()
    
    # Calculate rolling volatility
    df = df.copy()
    df['return'] = df['Price'].pct_change()
    df['volatility'] = df['return'].rolling(window=min(window, len(df))).std() * np.sqrt(365) * 100  # Annualized and in percentage
    
    # Create the figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df["volatility"],
        mode='lines',
        name='Volatilité',
        line=dict(color="#FF9500", width=2),
        hovertemplate='Date: %{x}<br>Volatilité: %{y:.2f}%<extra></extra>',
    ))
    
    fig.update_layout(
        title="📊 Volatilité du Bitcoin (annualisée)",
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(family="Inter", color=COLORS["text"]),
        xaxis=dict(
            title="Date & Heure",
            showgrid=True,
            gridcolor=COLORS["grid"],
            tickangle=-45,
            type="date",
            tickfont=dict(color=COLORS["text"])
        ),
        yaxis=dict(
            title="Volatilité (%)",
            showgrid=True,
            gridcolor=COLORS["grid"],
            ticksuffix="%",
            tickfont=dict(color=COLORS["text"])
        ),
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_dashboard_layout():
    """Create the dashboard layout."""
    return html.Div([
        html.Div([
            html.H1("Bitcoin Live Monitor", className="dashboard-title"),
            html.Div(id="current-price", className="current-price")
        ], className="dashboard-header"),
        
        html.Div([
            html.Div([
                dcc.Graph(id="price-graph", config={'displayModeBar': False})
            ], className="graph-container"),
            
            html.Div([
                dcc.Graph(id="volatility-graph", config={'displayModeBar': False})
            ], className="graph-container"),
            
            html.Div([
                html.Div(id="daily-report", className="report-container")
            ], className="report-card"),
            
            html.Div([
                html.Div(id="risk-metrics", className="report-container")
            ], className="report-card")
        ], className="content-wrapper")
    ], className="dashboard-container")

# Single callback to update all components
@app.callback(
    [Output("price-graph", "figure"),
     Output("volatility-graph", "figure"),
     Output("current-price", "children"),
     Output("daily-report", "children"),
     Output("risk-metrics", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_dashboard(n):
    """Comprehensive dashboard update function."""
    try:
        # Run scraper and daily report scripts
        subprocess.run(["/bin/bash", os.path.join(BASE_PATH, "scraper.sh")], check=True)
        subprocess.run(["/bin/bash", os.path.join(BASE_PATH, "daily_report.sh")], check=True)

    except Exception as e:
        print(f"Script execution error: {e}")
    
    # Load data for graph
    df = load_data()
    
    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"]
        )
        return empty_fig, empty_fig, "N/A", html.Div("No data available"), html.Div("No data available")
    
    # Create price graph
    price_fig = create_price_graph(df)
    
    # Create volatility graph
    volatility_fig = create_volatility_graph(df)
    
    current_price = f"${df['Price'].iloc[-1]:,.2f}"
    
    # Calculate risk metrics
    volatility = calculate_volatility(df)
    var_95 = calculate_var(df, confidence=0.95)
    var_99 = calculate_var(df, confidence=0.99)
    
    # Load daily report
    report = load_daily_report()
    
    if report is None:
        daily_report_html = html.Div("No report available")
    else:
        daily_report_html = html.Div([
            html.H3("Rapport Quotidien Bitcoin", className="report-title"),
            html.Div([
                html.Div([
                    html.Span("Horodatage", className="report-label"),
                    html.Span(report["Timestamp"].strftime("%Y-%m-%d %H:%M:%S"), className="report-value")
                ], className="report-item"),
                html.Div([
                    html.Span("Prix d'ouverture", className="report-label"),
                    html.Span(f"${report['Open']:,.2f}", className="report-value")
                ], className="report-item"),
                html.Div([
                    html.Span("Prix de clôture", className="report-label"),
                    html.Span(f"${report['Close']:,.2f}", className="report-value")
                ], className="report-item"),
                html.Div([
                    html.Span("Maximum", className="report-label"),
                    html.Span(f"${report['Max']:,.2f}", className="report-value")
                ], className="report-item"),
                html.Div([
                    html.Span("Minimum", className="report-label"),
                    html.Span(f"${report['Min']:,.2f}", className="report-value")
                ], className="report-item"),
                html.Div([
                    html.Span("Evolution", className="report-label"),
                    html.Span(str(report["Evolution"]), 
                              className="report-value", 
                              style={"color": COLORS["positive"] if float(str(report["Evolution"]).rstrip('%')) >= 0 else COLORS["negative"]})
                ], className="report-item")
            ], className="report-grid")
        ], className="report-container")
    
    # Create risk metrics card with improved layout
    risk_metrics_html = html.Div([
        html.H3("Métriques de Risque", className="report-title"),
        html.Div([
            html.Div([
                html.Div([
                    html.Span("Volatilité (annualisée)", className="risk-label"),
                    html.Div([
                        html.Span(f"{volatility * 100:.2f}%", className="risk-value"),
                    ], className="risk-value-container")
                ], className="risk-header"),
                html.Div("Mesure de la variance des prix sur la période", className="risk-description")
            ], className="risk-item"),
            
            html.Div([
                html.Div([
                    html.Span("VaR 95%", className="risk-label"),
                    html.Div([
                        html.Span(f"{var_95:.2f}%", className="risk-value"),
                    ], className="risk-value-container")
                ], className="risk-header"),
                html.Div("Perte maximale sur une journée avec 95% de confiance", className="risk-description")
            ], className="risk-item"),
            
            html.Div([
                html.Div([
                    html.Span("VaR 99%", className="risk-label"),
                    html.Div([
                        html.Span(f"{var_99:.2f}%", className="risk-value"),
                    ], className="risk-value-container")
                ], className="risk-header"),
                html.Div("Perte maximale sur une journée avec 99% de confiance", className="risk-description")
            ], className="risk-item")
        ], className="risk-grid")
    ], className="report-container")
    
    return price_fig, volatility_fig, current_price, daily_report_html, risk_metrics_html

# Application Layout
app.layout = html.Div([
    create_dashboard_layout(),
    dcc.Interval(id="interval-component", interval=60000)  # Update every 60 seconds
])


# Custom Index String with Dark Mode Styling
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        <title>Bitcoin Live Dashboard</title>
        {%metas%}
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            body {
                font-family: 'Inter', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #1C1C1E;
                color: #FFFFFF;
                line-height: 1.6;
            }
            
            .dashboard-container {
                max-width: 100%; 
                width: 100%;      
                margin: 0 auto;   
                padding: 20px;    
                box-sizing: border-box; 
                overflow: hidden;
            }
            
            .dashboard-header {
                margin-bottom: 30px;
            }
            
            .content-wrapper {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                justify-content: center;
            }
            
            .dashboard-title {
                color: #F7931A;
                font-size: 32px;
                font-weight: 700;
                margin-bottom: 20px;
                text-align: center;
            }
            
            .current-price {
                text-align: center;
                font-size: 48px;
                font-weight: 600;
                color: #F7931A;
                margin-bottom: 30px;
            }
            
            .graph-container {
                background-color: #2C2C2E;
                border-radius: 12px;
                padding: 15px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                margin-bottom: 20px;
            }
            
            .report-card {
                background-color: #2C2C2E;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                margin-bottom: 20px;
            }
            
            .report-title {
                color: #F7931A;
                font-size: 22px;
                margin-bottom: 20px;
                border-bottom: 2px solid #F7931A;
                padding-bottom: 10px;
            }
            
            .report-grid {
                display: grid;
                gap: 15px;
            }
            
            .report-item {
                display: flex;
                flex-direction: column;
                padding: 15px;
                background-color: #3A3A3C;
                border-radius: 8px;
            }
            
            .report-label {
                color: #B0B0B0;
                font-size: 14px;
                margin-bottom: 5px;
            }
            
            .report-value {
                color: #FFFFFF;
                font-weight: 600;
                font-size: 18px;
            }
            
            /* Styles spécifiques pour les métriques de risque */
            .risk-grid {
                display: grid;
                gap: 15px;
            }
            
            .risk-item {
                background-color: #3A3A3C;
                border-radius: 8px;
                padding: 15px;
            }
            
            .risk-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }
            
            .risk-label {
                color: #B0B0B0;
                font-size: 14px;
                font-weight: 500;
            }
            
            .risk-value {
                font-size: 18px;
                font-weight: 600;
                color: #FFFFFF;
            }
            
            .risk-description {
                color: #8E8E93;
                font-size: 12px;
                font-style: italic;
            }
            
            @media (min-width: 992px) {
                .content-wrapper {
                    grid-template-columns: repeat(2, 1fr);
                }
                
                .graph-container:nth-child(1),
                .graph-container:nth-child(2) {
                    grid-column: span 2;
                }
            }
            
            @media (max-width: 768px) {
                .content-wrapper {
                    grid-template-columns: 1fr;
                }
                
                .graph-container,
                .report-card {
                    grid-column: span 1;
                }
                
                .current-price {
                    font-size: 36px;
                }
                
                .risk-header {
                    flex-direction: column;
                    align-items: flex-start;
                }
                
                .risk-value-container {
                    margin-top: 5px;
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# Ensure files exist before running
ensure_files_exist()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
