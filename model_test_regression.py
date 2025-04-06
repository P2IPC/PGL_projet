import numpy as np
from sklearn.linear_model import LinearRegression

def prepare_data(df, window_size=60, horizon=10):
    prices = df["Price"].values
    X = []
    y = []

    if len(prices) < window_size + horizon:
        return np.array([]), np.array([])

    for i in range(len(prices) - window_size - horizon):
        window = prices[i:i + window_size]
        target = prices[i + window_size + horizon - 1]
        X.append(window)
        y.append(target)

    return np.array(X), np.array(y)

def predict_next_hour(df, seuil=0.001):  # seuil = 0.1% par défaut
    if len(df) < 80:  
        return "Données insuffisantes"

    X, y = prepare_data(df)

    if len(X) == 0:
        return "Modèle non disponible"

    try:
        model = LinearRegression()
        model.fit(X, y)

        last_window = df["Price"].values[-60:]
        if len(last_window) < 60:
            return "Fenêtre incomplète"

        prediction = model.predict(last_window.reshape(1, -1))[0]
        current_price = df["Price"].values[-1]

        # Détection d'opportunité de trading
        delta = prediction - current_price

        if delta > seuil * current_price:
            signal = "Signal d'achat (BUY)"
        elif delta < -seuil * current_price:
            signal = "Signal de vente (SELL)"
        else:
            signal = "Aucun signal (HOLD)"

        return (
            f"Prix actuel : ${current_price:,.2f}\n"
            f"Prix potentiel dans 10 min : ${prediction:,.2f}\n"
            f"{signal}"
        )

    except Exception as e:
        return f"Erreur : {str(e)}"
