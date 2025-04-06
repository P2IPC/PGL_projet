
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
PREDICTION_MINUTES = 10  # Prédire pour les 10 prochaines minutes

# Design Theme
COLORS = {
    "background": "#1C1C1E",
    "text": "#FFFFFF",
    "bitcoin": "#F7931A",
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "prediction": "#9B59B6",  # Couleur pour les prédictions
    "card_bg": "#2C2C2E",
    "grid": "#3A3A3C"
}

def ensure_files_exist():
    """Ensure required files exist."""
    for file_path in [DATA_FILE, REPORT_FILE]:
        if not os.path.exists(file_path):
            open(file_path, 'a').close()
            print(f"Created file: {file_path}")
# Nouvelle fonction à ajouter
def convert_to_local_timezone(df):
    """Convertit les timestamps UTC en heure locale (Europe/Paris)"""
    if df.empty or "Timestamp" not in df.columns:
        return df
    
    df = df.copy()
    # Assure que les timestamps sont en UTC avant de les convertir
    df["Timestamp"] = df["Timestamp"].dt.tz_localize(pytz.UTC).dt.tz_convert(TZ_PARIS)
    return df

def load_data():
    """Load data from CSV file with error handling."""
    try:
        df = pd.read_csv(DATA_FILE, names=["Timestamp", "Price"], header=None)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = df.dropna().sort_values("Timestamp")
        # Conversion en heure locale
        df = convert_to_local_timezone(df)
        return df
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

def create_features(df):
    """Créer des caractéristiques pour le modèle prédictif."""
    df = df.copy()
    
    # Extraire des caractéristiques temporelles
    df['hour'] = df['Timestamp'].dt.hour
    df['minute'] = df['Timestamp'].dt.minute  # Ajout des minutes pour les prédictions plus précises
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
    df['quarter'] = df['Timestamp'].dt.quarter
    df['month'] = df['Timestamp'].dt.month
    df['year'] = df['Timestamp'].dt.year
    df['dayofyear'] = df['Timestamp'].dt.dayofyear
    df['dayofmonth'] = df['Timestamp'].dt.day
    df['weekofyear'] = df['Timestamp'].dt.isocalendar().week
    
    # Créer des caractéristiques de lag plus adaptées à des prédictions de court terme
    for lag in range(1, 11):  # Utiliser les 10 dernières valeurs
        df[f'lag_{lag}'] = df['Price'].shift(lag)
    
    # Calculer les moyennes mobiles de plus court terme
    df['rolling_mean_3'] = df['Price'].rolling(window=3).mean()
    df['rolling_mean_5'] = df['Price'].rolling(window=5).mean()
    df['rolling_mean_10'] = df['Price'].rolling(window=10).mean()
    
    # Calculer la volatilité à court terme
    df['volatility_10'] = df['Price'].rolling(window=10).std()
    
    # Calculer les variations de prix relatives à court terme
    df['price_change_1'] = df['Price'].pct_change(periods=1)
    df['price_change_5'] = df['Price'].pct_change(periods=5)
    
    # Supprimer les lignes avec des valeurs NaN (dues aux lag et rolling windows)
    df = df.dropna()
    
    return df

def prepare_prediction_data(df):
    """Préparer les données pour l'entraînement et les prédictions."""
    if len(df) < 30:  # Nécessite un minimum de données
        return None, None, None, None, None
    
    # Créer des features
    df_features = create_features(df)
    
    # Variable cible: le prix de la prochaine observation
    df_features['target'] = df_features['Price'].shift(-1)
    
    # Séparation des données récentes (pour la prédiction) et des données d'entraînement
    df_recent = df_features.iloc[-1:].copy()  # Dernière ligne pour les prédictions futures
    df_train = df_features.iloc[:-1].dropna().copy()  # Reste des données pour l'entraînement
    
    if df_train.empty:
        return None, None, None, None, None
    
    # Séparer les features et la cible
    features = ['hour', 'minute', 'dayofweek', 'lag_1', 'lag_2', 'lag_3', 'lag_5',
                'rolling_mean_3', 'rolling_mean_5', 'rolling_mean_10',
                'volatility_10', 'price_change_1', 'price_change_5']
    
    # S'assurer que toutes les colonnes existent
    features = [f for f in features if f in df_train.columns]
    
    X = df_train[features]
    y = df_train['target']
    
    # Données récentes pour la prédiction
    X_recent = df_recent[features]
    
    return X, y, X_recent, df_recent['Timestamp'].iloc[0], df_features

def train_model_and_predict(df):
    """Entraîner un modèle et faire des prédictions pour les 10 prochaines minutes."""
    if df.empty or len(df) < 30:
        return None, None
    
    X, y, X_recent, last_timestamp, df_features = prepare_prediction_data(df)
    
    if X is None or y is None:
        return None, None
    
    # Entraîner un modèle Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Pour les prédictions futures
    future_timestamps = []
    future_predictions = []
    
    # La dernière entrée connue
    current_data = X_recent.iloc[0].copy()
    current_timestamp = last_timestamp
    last_price = df["Price"].iloc[-1]  # Dernière valeur réelle
    
    # Faire des prédictions pour les prochaines 10 minutes
    for i in range(PREDICTION_MINUTES):
        # Calculer le prochain timestamp (1 minute plus tard)
        next_timestamp = current_timestamp + pd.Timedelta(minutes=1)
        
        # Mettre à jour les caractéristiques temporelles
        current_data['hour'] = next_timestamp.hour
        current_data['minute'] = next_timestamp.minute
        current_data['dayofweek'] = next_timestamp.dayofweek
        
        # Faire une prédiction
        prediction = model.predict(pd.DataFrame([current_data]))[0]
        
        # Stocker la prédiction
        future_timestamps.append(next_timestamp)
        future_predictions.append(prediction)
        
        # Mettre à jour pour la prochaine itération (important pour la continuité des prédictions)
        if i < PREDICTION_MINUTES - 1:
            # Mettre à jour les lags pour la prochaine prédiction
            for lag in range(10, 1, -1):
                lag_col = f'lag_{lag}'
                prev_lag_col = f'lag_{lag-1}'
                if lag_col in current_data and prev_lag_col in current_data:
                    current_data[lag_col] = current_data[prev_lag_col]
            
            if 'lag_1' in current_data:
                current_data['lag_1'] = prediction
            
            # Mettre à jour les moyennes mobiles (approximatif mais suffisant pour la prédiction)
            if 'rolling_mean_3' in current_data:
                current_data['rolling_mean_3'] = (current_data['lag_1'] + current_data.get('lag_2', 0) + current_data.get('lag_3', 0)) / 3
            
            if 'rolling_mean_5' in current_data:
                current_data['rolling_mean_5'] = (current_data['lag_1'] + current_data.get('lag_2', 0) + current_data.get('lag_3', 0) + 
                                                 current_data.get('lag_4', 0) + current_data.get('lag_5', 0)) / 5
            
            # Calculer la variation de prix basée sur notre prédiction
            if 'price_change_1' in current_data:
                current_data['price_change_1'] = (prediction - current_data['lag_1']) / current_data['lag_1'] if current_data['lag_1'] != 0 else 0
            
        current_timestamp = next_timestamp
    
    # Créer un DataFrame avec les prévisions
    predictions_df = pd.DataFrame({
        'Timestamp': future_timestamps,
        'Predicted_Price': future_predictions
    })
    
    # S'assurer que les timestamps sont dans le bon fuseau horaire
    if predictions_df['Timestamp'].dt.tz is None:
        predictions_df['Timestamp'] = predictions_df['Timestamp'].dt.tz_localize(TZ_PARIS)
    else:
        predictions_df['Timestamp'] = predictions_df['Timestamp'].dt.tz_convert(TZ_PARIS)
        
    # Calculer les métriques d'évaluation sur les données connues
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    eval_metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'Feature_Importance': dict(zip(X.columns, model.feature_importances_))
    }
    
    return predictions_df, eval_metrics
    
    # Créer un DataFrame avec les prévisions
    predictions_df = pd.DataFrame({
        'Timestamp': future_timestamps,
        'Predicted_Price': future_predictions
    })
    
    # Calculer les métriques d'évaluation sur les données connues
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    eval_metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'Feature_Importance': dict(zip(X.columns, model.feature_importances_))
    }
    
    return predictions_df, eval_metrics

def filter_data_by_range(df, minutes_limit):
    """Filter data to show only the specified number of minutes."""
    if df.empty:
        return df
    
    # If -1 is selected, return all data
    if minutes_limit == -1:
        return df
    
    # Use the most recent date in the dataset as the reference point
    latest_date = df["Timestamp"].max()
    
    # Filter to show only the last X minutes
    start_time = latest_date - pd.Timedelta(minutes=minutes_limit)
    return df[df["Timestamp"] >= start_time]

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

def create_price_graph(df, predictions_df=None):
    if df.empty:
        return go.Figure()

    # Utiliser toutes les données disponibles sans la limitation arbitraire
    df_display = df.copy()

    df_display['delta'] = df_display['Price'].pct_change()
    df_display['color'] = np.where(df_display['delta'] >= 0, COLORS["positive"], COLORS["negative"])

    # Moyenne + bandes
    df_display['ma'] = df_display['Price'].rolling(window=10).mean()
    df_display['upper'] = df_display['ma'] + df_display['Price'].rolling(window=10).std()
    df_display['lower'] = df_display['ma'] - df_display['Price'].rolling(window=10).std()

    # Trouver les indices et valeurs du maximum et minimum dans le timeframe actuel
    max_idx = df_display['Price'].idxmax()
    min_idx = df_display['Price'].idxmin()
    max_price = df_display.loc[max_idx, 'Price']
    min_price = df_display.loc[min_idx, 'Price']
    max_timestamp = df_display.loc[max_idx, 'Timestamp']
    min_timestamp = df_display.loc[min_idx, 'Timestamp']

    fig = go.Figure()

    # Bande supérieure et inférieure
    fig.add_trace(go.Scatter(
        x=df_display["Timestamp"],
        y=df_display["upper"],
        line=dict(color='rgba(255,255,255,0.2)', dash='dot'),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=df_display["Timestamp"],
        y=df_display["lower"],
        fill='tonexty',
        fillcolor='rgba(255,255,255,0.05)',
        line=dict(color='rgba(255,255,255,0.2)', dash='dot'),
        name='Bande de Volatilité',
        hoverinfo="skip"
    ))

    # Ligne principale avec couleur dynamique (via scattergl pour performance)
    fig.add_trace(go.Scattergl(
        x=df_display["Timestamp"],
        y=df_display["Price"],
        mode='lines',  # Remove markers, keep just the line
        line=dict(color=COLORS["bitcoin"], width=2),
        name="Prix",
        hovertemplate='Heure: %{x}<br>Prix: $%{y:.2f}<extra></extra>'
    ))

    # Ajouter le point pour le maximum (vert)
    fig.add_trace(go.Scatter(
        x=[max_timestamp],
        y=[max_price],
        mode='markers+text',
        marker=dict(color='green', size=10, symbol='circle'),
        text=["Max"],
        textposition="top center",
        name=f"Maximum: ${max_price:.2f}",
        hovertemplate='Max: $%{y:.2f}<br>%{x}<extra></extra>'
    ))

    # Ajouter le point pour le minimum (rouge)
    fig.add_trace(go.Scatter(
        x=[min_timestamp],
        y=[min_price],
        mode='markers+text',
        marker=dict(color='red', size=10, symbol='circle'),
        text=["Min"],
        textposition="bottom center",
        name=f"Minimum: ${min_price:.2f}",
        hovertemplate='Min: $%{y:.2f}<br>%{x}<extra></extra>'
    ))

    # Prédictions
    if predictions_df is not None and not predictions_df.empty:
        # Obtenir la dernière valeur réelle
        last_real_timestamp = df_display["Timestamp"].iloc[-1]
        last_real_price = df_display["Price"].iloc[-1]
        
        # S'assurer que les prédictions commencent après la dernière valeur réelle
        # Créer une nouvelle dataframe de prédiction qui commence 1 minute après la dernière valeur réelle
        new_start_time = last_real_timestamp + pd.Timedelta(minutes=1)
        
        # Créer un intervalle régulier de minutes pour les prédictions
        prediction_timestamps = pd.date_range(
            start=new_start_time, 
            periods=len(predictions_df), 
            freq="1min"
        )
        
        # Créer un nouveau dataframe de prédiction avec les timestamps ajustés
        adjusted_predictions = pd.DataFrame({
            "Timestamp": prediction_timestamps,
            "Predicted_Price": predictions_df["Predicted_Price"].values
        })
        
        # Ajouter la courbe de prédiction
        fig.add_trace(go.Scatter(
            x=adjusted_predictions["Timestamp"],
            y=adjusted_predictions["Predicted_Price"],
            mode='lines+text',
            name='Prédiction 10 min',
            line=dict(color=COLORS["prediction"], width=2, dash='dash'),
            text=["" for _ in range(len(adjusted_predictions)-1)] + ["➡️ Prédiction"],
            textposition="top right",
            textfont=dict(color=COLORS["prediction"]),
            hovertemplate='Prévu: %{x}<br>$%{y:.2f}<extra></extra>'
        ))

        # Transition visuelle entre la dernière valeur réelle et la première prédiction
        fig.add_trace(go.Scatter(
            x=[last_real_timestamp, adjusted_predictions["Timestamp"].iloc[0]],
            y=[last_real_price, adjusted_predictions["Predicted_Price"].iloc[0]],
            mode='lines',
            line=dict(color=COLORS["prediction"], width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title="📈 Fluctuation du Prix du Bitcoin avec Prédictions",
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(family="Inter", color=COLORS["text"]),
        xaxis=dict(title="Heure", showgrid=True, gridcolor=COLORS["grid"], tickangle=-45),
        yaxis=dict(title="Prix (USD)", showgrid=True, gridcolor=COLORS["grid"], tickprefix="$"),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    return fig

def create_volatility_graph(df, window=14):
    """Create a volatility graph based on price data."""
    if len(df) < window:
        return go.Figure()
    
    # Utiliser toutes les données disponibles sans la limitation
    df_display = df.copy()
    
    # Calculate rolling volatility
    df_display['return'] = df_display['Price'].pct_change()
    df_display['volatility'] = df_display['return'].rolling(window=min(window, len(df_display))).std() * np.sqrt(365) * 100  # Annualized and in percentage
    
    # Create the figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_display["Timestamp"],
        y=df_display["volatility"],
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
    """Create the dashboard layout with prediction metrics."""
    return html.Div([
        html.Div([
            html.H1("Bitcoin Live Monitor & Prédiction", className="dashboard-title"),
            html.Div(id="current-price", className="current-price"),
            
            # Remplacer le dropdown par des boutons de timeframe
            html.Div([
                html.Div("Période d'affichage:", className="timeframe-label"),
                html.Div([
                    html.Button("15min", id="timeframe-15", n_clicks=0, className="timeframe-button"),
                    html.Button("30min", id="timeframe-30", n_clicks=0, className="timeframe-button"),
                    html.Button("1h", id="timeframe-60", n_clicks=1, className="timeframe-button active"),
                    html.Button("2h", id="timeframe-120", n_clicks=0, className="timeframe-button"),
                    html.Button("4h", id="timeframe-240", n_clicks=0, className="timeframe-button"),
                    html.Button("8h", id="timeframe-480", n_clicks=0, className="timeframe-button"),
                    html.Button("12h", id="timeframe-720", n_clicks=0, className="timeframe-button"),
                    html.Button("1j", id="timeframe-1440", n_clicks=0, className="timeframe-button"),
                    html.Button("Tout", id="timeframe-all", n_clicks=0, className="timeframe-button"),
                ], className="timeframe-buttons-container"),
                # Hidden div to store the selected timeframe value
                html.Div(id="selected-timeframe", style={"display": "none"}, children="60")
            ], className="timeframe-selector"),
        ], className="dashboard-header"),
        
        # Reste du layout inchangé...
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
            ], className="report-card"),
            
            html.Div([
                html.Div(id="prediction-metrics", className="report-container")
            ], className="report-card")
        ], className="content-wrapper"),
        
        # Réduire l'intervalle pour des mises à jour plus fréquentes
        dcc.Interval(id="interval-component", interval=30000)  # Update every 30 seconds
    ], className="dashboard-container")

# Ajouter le callback pour gérer les clics sur les boutons de timeframe
@app.callback(
    [Output("selected-timeframe", "children"),
     Output("timeframe-15", "className"),
     Output("timeframe-30", "className"),
     Output("timeframe-60", "className"),
     Output("timeframe-120", "className"),
     Output("timeframe-240", "className"),
     Output("timeframe-480", "className"),
     Output("timeframe-720", "className"),
     Output("timeframe-1440", "className"),
     Output("timeframe-all", "className")],
    [Input("timeframe-15", "n_clicks"),
     Input("timeframe-30", "n_clicks"),
     Input("timeframe-60", "n_clicks"),
     Input("timeframe-120", "n_clicks"),
     Input("timeframe-240", "n_clicks"),
     Input("timeframe-480", "n_clicks"),
     Input("timeframe-720", "n_clicks"),
     Input("timeframe-1440", "n_clicks"),
     Input("timeframe-all", "n_clicks")]
)
def update_timeframe(n15, n30, n60, n120, n240, n480, n720, n1440, nall):
    ctx = dash.callback_context
    if not ctx.triggered:
        # Default à 1 heure (60 minutes)
        button_id = "timeframe-60"
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Mapping des IDs de boutons aux valeurs de timeframe
    timeframe_values = {
        "timeframe-15": "15",
        "timeframe-30": "30",
        "timeframe-60": "60",
        "timeframe-120": "120",
        "timeframe-240": "240",
        "timeframe-480": "480",
        "timeframe-720": "720",
        "timeframe-1440": "1440",
        "timeframe-all": "-1"
    }
    
    # Par défaut, tous les boutons ont la classe "timeframe-button"
    button_classes = {btn: "timeframe-button" for btn in timeframe_values.keys()}
    
    # Ajouter la classe "active" pour le bouton sélectionné
    button_classes[button_id] = "timeframe-button active"
    
    # Renvoie la valeur du timeframe et les classes pour tous les boutons
    return (
        timeframe_values[button_id],
        button_classes["timeframe-15"],
        button_classes["timeframe-30"],
        button_classes["timeframe-60"],
        button_classes["timeframe-120"],
        button_classes["timeframe-240"],
        button_classes["timeframe-480"],
        button_classes["timeframe-720"],
        button_classes["timeframe-1440"],
        button_classes["timeframe-all"]
    )

# Modifier le callback principal pour utiliser la valeur stockée dans selected-timeframe
@app.callback(
    [Output("price-graph", "figure"),
     Output("volatility-graph", "figure"),
     Output("current-price", "children"),
     Output("daily-report", "children"),
     Output("risk-metrics", "children"),
     Output("prediction-metrics", "children")],
    [Input("interval-component", "n_intervals"),
     Input("selected-timeframe", "children")]
)
def update_dashboard(n, minutes_limit):
    # Convertir la chaîne en entier
    minutes_limit = int(minutes_limit)
    
    try:
        subprocess.run(["/bin/bash", os.path.join(BASE_PATH, "scraper.sh")], check=True)
        subprocess.run(["/bin/bash", os.path.join(BASE_PATH, "daily_report.sh")], check=True)
    except Exception as e:
        print(f"❌ Script execution error: {e}")
    
    # Reste du code inchangé...
    df_full = load_data()
    if df_full.empty:
        empty_fig = go.Figure().update_layout(
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"]
        )
        empty_text = html.Div("Pas de données")
        return empty_fig, empty_fig, "N/A", empty_text, empty_text, empty_text
    
    # Filtrage basé sur le nombre de minutes sélectionnées
    df_display = filter_data_by_range(df_full.copy(), minutes_limit)
    if df_display.empty:
        empty_fig = go.Figure().update_layout(
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"]
        )
        empty_text = html.Div("Aucune donnée dans cette période")
        return empty_fig, empty_fig, "N/A", empty_text, empty_text, empty_text
    
    # Générer une nouvelle prédiction à chaque intervalle
    predictions_df, eval_metrics = train_model_and_predict(df_full)
    price_fig = create_price_graph(df_display, predictions_df)
    volatility_fig = create_volatility_graph(df_display)
    current_price = f"${df_display['Price'].iloc[-1]:,.2f}"
    
    # Calculate risk metrics
    volatility = calculate_volatility(df_display)
    var_95 = calculate_var(df_display, confidence=0.95)
    var_99 = calculate_var(df_display, confidence=0.99)
    
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
    
    # Créer la section des métriques de prédiction
    if predictions_df is not None and eval_metrics is not None and not predictions_df.empty:
        # Calculer la prédiction pour la prochaine minute et pour 10 minutes
        next_minute_prediction = predictions_df['Predicted_Price'].iloc[0]
        next_10min_prediction = predictions_df['Predicted_Price'].iloc[-1]
        current_price_value = df_display['Price'].iloc[-1]
        
        price_change_1min = (next_minute_prediction - current_price_value) / current_price_value * 100
        price_change_10min = (next_10min_prediction - current_price_value) / current_price_value * 100
        
        prediction_metrics_html = html.Div([
            html.H3("Prévisions à 10 Minutes", className="report-title"),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span("Prédiction à 1 min", className="risk-label"),
                        html.Div([
                            html.Span(f"${next_minute_prediction:.2f}", className="risk-value"),
                        ], className="risk-value-container")
                    ], className="risk-header"),
                    html.Div(f"Variation prévue: {price_change_1min:.2f}%", 
                             className="risk-description",
                             style={"color": COLORS["positive"] if price_change_1min >= 0 else COLORS["negative"]})
                ], className="risk-item"),
                
                html.Div([
                    html.Div([
                        html.Span("Prédiction à 10 min", className="risk-label"),
                        html.Div([
                            html.Span(f"${next_10min_prediction:.2f}", className="risk-value"),
                        ], className="risk-value-container")
                    ], className="risk-header"),
                    html.Div(f"Variation prévue: {price_change_10min:.2f}%", 
                             className="risk-description",
                             style={"color": COLORS["positive"] if price_change_10min >= 0 else COLORS["negative"]})
                ], className="risk-item"),
        ], className="risk-grid")
    ], className="report-container")
    
    # Créer la section des métriques de prédiction
    if predictions_df is not None and eval_metrics is not None and not predictions_df.empty:
        # Calculer la prédiction pour la prochaine minute et pour 10 minutes
        next_minute_prediction = predictions_df['Predicted_Price'].iloc[0]
        next_10min_prediction = predictions_df['Predicted_Price'].iloc[-1]
        current_price_value = df_display['Price'].iloc[-1]
        
        price_change_1min = (next_minute_prediction - current_price_value) / current_price_value * 100
        price_change_10min = (next_10min_prediction - current_price_value) / current_price_value * 100
        
        prediction_metrics_html = html.Div([
            html.H3("Prévisions à 10 Minutes", className="report-title"),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span("Prédiction à 1 min", className="risk-label"),
                        html.Div([
                            html.Span(f"${next_minute_prediction:.2f}", className="risk-value"),
                        ], className="risk-value-container")
                    ], className="risk-header"),
                    html.Div(f"Variation prévue: {price_change_1min:.2f}%", 
                             className="risk-description",
                             style={"color": COLORS["positive"] if price_change_1min >= 0 else COLORS["negative"]})
                ], className="risk-item"),
                
                html.Div([
                    html.Div([
                        html.Span("Prédiction à 10 min", className="risk-label"),
                        html.Div([
                            html.Span(f"${next_10min_prediction:.2f}", className="risk-value"),
                        ], className="risk-value-container")
                    ], className="risk-header"),
                    html.Div(f"Variation prévue: {price_change_10min:.2f}%", 
                             className="risk-description",
                             style={"color": COLORS["positive"] if price_change_10min >= 0 else COLORS["negative"]})
                ], className="risk-item"),
                
                html.Div([
                    html.Div([
                        html.Span("Précision du Modèle (MAE)", className="risk-label"),
                        html.Div([
                            html.Span(f"${eval_metrics['MAE']:.2f}", className="risk-value"),
                        ], className="risk-value-container")
                    ], className="risk-header"),
                    html.Div("Erreur absolue moyenne des prédictions", className="risk-description")
                ], className="risk-item"),
                
                html.Div([
                    html.Div([
                        html.Span("Facteur le Plus Influent", className="risk-label"),
                        html.Div([
                            html.Span(max(eval_metrics['Feature_Importance'].items(), key=lambda x: x[1])[0], className="risk-value"),
                        ], className="risk-value-container")
                    ], className="risk-header"),
                    html.Div("Variable ayant le plus d'impact sur les prédictions", className="risk-description")
                ], className="risk-item")
            ], className="risk-grid")
        ], className="report-container")
    else:
        prediction_metrics_html = html.Div([
            html.H3("Prévisions à 10 Minutes", className="report-title"),
            html.Div("Données insuffisantes pour générer des prédictions fiables.", 
                     style={"padding": "20px", "text-align": "center", "color": "#B0B0B0"})
        ], className="report-container")
    
    return price_fig, volatility_fig, current_price, daily_report_html, risk_metrics_html, prediction_metrics_html

# Application Layout
app.layout = create_dashboard_layout()
# Custom Index String with Dark Mode Styling
# Custom Index String with Dark Mode Styling
app.index_string = """
<!DOCTYPE html>
<html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bitcoin Live Dashboard & Prédiction</title>
        {%metas%}
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --background: #1C1C1E;
                --card-bg: #2C2C2E;
                --border-color: #3A3A3C;
                --text-primary: #FFFFFF;
                --text-secondary: #B0B0B0;
                --text-tertiary: #8E8E93;
                --bitcoin-color: #F7931A;
                --positive: #2ecc71;
                --negative: #e74c3c;
                --prediction: #9B59B6;
            }
            
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                margin: 0;
                padding: 0;
                background-color: var(--background);
                color: var(--text-primary);
                line-height: 1.6;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            .dashboard-container {
                width: 100%;      
                margin: 0 auto;   
                padding: 20px;    
                overflow-x: hidden;
            }
            
            .dashboard-header {
                margin-bottom: 30px;
                text-align: center;
            }
            
            .content-wrapper {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                justify-content: center;
                margin-bottom: 30px;
            }
            
            .dashboard-title {
                color: var(--bitcoin-color);
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 20px;
            }
            
            .current-price {
                font-size: 3rem;
                font-weight: 600;
                color: var(--bitcoin-color);
                margin-bottom: 30px;
            }
            
            .graph-container {
                background-color: var(--card-bg);
                border-radius: 12px;
                padding: 15px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                margin-bottom: 20px;
                overflow: hidden;
            }
            
            .report-card {
                background-color: var(--card-bg);
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                margin-bottom: 20px;
            }
            
            .report-title {
                color: var(--bitcoin-color);
                font-size: 1.375rem;
                margin-bottom: 20px;
                border-bottom: 2px solid var(--bitcoin-color);
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
                background-color: var(--border-color);
                border-radius: 8px;
            }
            
            .report-label {
                color: var(--text-secondary);
                font-size: 0.875rem;
                margin-bottom: 5px;
            }
            
            .report-value {
                color: var(--text-primary);
                font-weight: 600;
                font-size: 1.125rem;
            }
            
            /* Styles spécifiques pour les métriques de risque */
            .risk-grid {
                display: grid;
                gap: 15px;
            }
            
            .risk-item {
                background-color: var(--border-color);
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
                color: var(--text-secondary);
                font-size: 0.875rem;
                font-weight: 500;
            }
            
            .risk-value-container {
                display: flex;
                align-items: center;
            }
            
            .risk-value {
                font-size: 1.125rem;
                font-weight: 600;
                color: var(--text-primary);
            }
            
            .risk-description {
                color: var(--text-tertiary);
                font-size: 0.75rem;
                font-style: italic;
            }
            
            /* Styles de hover pour une meilleure interactivité */
            .report-item:hover, .risk-item:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
                transition: all 0.2s ease;
            }
            
            /* Media Queries pour une meilleure responsivité */
            @media (min-width: 1200px) {
                .dashboard-container {
                    max-width: 1400px;
                    padding: 30px;
                }
                
                .content-wrapper {
                    grid-template-columns: repeat(4, 1fr);
                }
                
                .graph-container:nth-child(1),
                .graph-container:nth-child(2) {
                    grid-column: span 2;
                }
            }
            
            @media (min-width: 992px) and (max-width: 1199px) {
                .content-wrapper {
                    grid-template-columns: repeat(2, 1fr);
                }
                
                .graph-container:nth-child(1),
                .graph-container:nth-child(2) {
                    grid-column: span 2;
                }
            }
            
            @media (min-width: 768px) and (max-width: 991px) {
                .content-wrapper {
                    grid-template-columns: repeat(2, 1fr);
                }
                
                .graph-container {
                    grid-column: span 2;
                }
            }
            
            @media (max-width: 767px) {
                .content-wrapper {
                    grid-template-columns: 1fr;
                }
                
                .dashboard-title {
                    font-size: 1.75rem;
                }
                
                .current-price {
                    font-size: 2.25rem;
                }
                
                .graph-container,
                .report-card {
                    margin-bottom: 15px;
                }
                
                .risk-header {
                    flex-direction: column;
                    align-items: flex-start;
                }
                
                .risk-value-container {
                    margin-top: 5px;
                    width: 100%;
                }
                
                .report-title {
                    font-size: 1.25rem;
                }
            }
            
            /* Pour la transition fluide lors du chargement des graphiques */
            .js-plotly-plot .plotly .modebar {
                opacity: 0.3;
                transition: opacity 0.3s ease-in-out;
            }
            
            .js-plotly-plot .plotly .modebar:hover {
                opacity: 1;
            }
            
            /* Animations pour améliorer l'expérience utilisateur */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .graph-container, .report-card {
                animation: fadeIn 0.4s ease-out forwards;
            }
            
            /* Amélioration pour les tooltips et hover states */
            [data-dash-is-loading="true"] {
                visibility: hidden;
            }
            
            [data-dash-is-loading="true"]::before {
                content: "Chargement...";
                display: block;
                text-align: center;
                color: var(--text-secondary);
                visibility: visible;
            }
            
            /* Dropdown styling fixes */
            .Select-menu-outer {
                background-color: var(--card-bg) !important;
                border: 1px solid var(--border-color) !important;
                color: var(--text-primary) !important;
                z-index: 1000 !important;
            }
            
            .Select-value, .Select-placeholder, .Select-input {
                color: var(--text-primary) !important;
                background-color: var(--card-bg) !important;
            }
            
            .Select-control {
                background-color: var(--card-bg) !important;
                border: 1px solid var(--border-color) !important;
            }
            
            .Select--single > .Select-control .Select-value, .Select-placeholder {
                color: var(--text-primary) !important;
            }
            
            .Select-menu {
                background-color: var(--card-bg) !important;
                color: var(--text-primary) !important;
            }
            
            .Select-option {
                background-color: var(--card-bg) !important;
                color: var(--text-primary) !important;
            }
            
            .Select-option:hover {
                background-color: var(--border-color) !important;
            }
            
            .Select-option.is-selected {
                background-color: var(--bitcoin-color) !important;
                color: white !important;
            }
            
            /* Range slider styling */
            .js-plotly-plot .plotly .rangeslider-container {
                background-color: var(--card-bg) !important;
            }
            
            .js-plotly-plot .plotly .rangeslider-mask {
                fill: rgba(247, 147, 26, 0.2) !important;
                fill-opacity: 0.3 !important;
            }
            
            .js-plotly-plot .plotly .rangeslider-handle {
                fill: var(--bitcoin-color) !important;
            }
            /* Styles pour les boutons de timeframe */
            .timeframe-selector {
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 20px;
                flex-wrap: wrap;
                gap: 10px;
            }

            .timeframe-label {
                font-size: 0.9rem;
                color: var(--text-secondary);
                margin-right: 10px;
            }

            .timeframe-buttons-container {
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                justify-content: center;
            }

            .timeframe-button {
                background-color: var(--card-bg);
                color: var(--text-secondary);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 0.85rem;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
                min-width: 60px;
                text-align: center;
            }

            .timeframe-button:hover {
                background-color: rgba(247, 147, 26, 0.1);
                color: var(--bitcoin-color);
                border-color: var(--bitcoin-color);
            }

            .timeframe-button.active {
                background-color: var(--bitcoin-color);
                color: white;
                border-color: var(--bitcoin-color);
                box-shadow: 0 2px 4px rgba(247, 147, 26, 0.3);
            }

            /* Responsive styles for timeframe buttons */
            @media (max-width: 767px) {
                .timeframe-selector {
                    flex-direction: column;
                }
                
                .timeframe-label {
                    margin-bottom: 10px;
                    margin-right: 0;
                }
                
                .timeframe-buttons-container {
                    width: 100%;
                }
                
                .timeframe-button {
                    flex: 1 0 calc(33.33% - 5px);
                    font-size: 0.8rem;
                    padding: 6px 8px;
                }
            }
        </style>
    </head>
    <body>
        <div id="react-entry-point">
            {%app_entry%}
        </div>
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