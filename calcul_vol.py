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