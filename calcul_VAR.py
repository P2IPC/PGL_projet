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