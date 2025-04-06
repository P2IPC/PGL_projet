import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PREDICTION_MINUTES = 10 #à mettre dans les constantes au début

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
     
    