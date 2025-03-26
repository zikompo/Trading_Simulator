import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def read_csv(path):
    """Read CSV file from the given path."""
    return pd.read_csv(path)

def get_data():
    """Read data from CSV, split into features and target, then into training and test sets."""
    df = pd.read_csv('/Users/arsencameron/Documents/Projects/Trading_Simulator/data/updated_data.csv', parse_dates=['Date'])
    print(f"Initial data shape: {df.shape}")
    
    # Ensure sorting by date (important for time series data)
    df = df.sort_values(by='Date')
    
    # Generate features
    df['returns'] = df['Close'].pct_change()
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_open_ratio'] = df['Close'] / df['Open']
    df['volume_change'] = df['Volume'].pct_change()
    
    # Generate lag features
    for lag in range(1, 6):
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        df[f'hl_ratio_lag_{lag}'] = df['high_low_ratio'].shift(lag)
        df[f'co_ratio_lag_{lag}'] = df['close_open_ratio'].shift(lag)
        df[f'volume_change_lag_{lag}'] = df['volume_change'].shift(lag)
    
    # Define target variable (e.g., predicting next day's return direction)
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)
    
    # Drop NaN values introduced by shifting
    df = df.dropna()
    print(f"Data shape after feature engineering: {df.shape}")
    
    # Select feature columns
    feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol', 'target']]
    X = df[feature_cols]
    y = df['target']
    
    # Split into training and test sets (no shuffle for time series data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def main():
    # Load the data
    X_train, X_test, y_train, y_test = get_data()
    
    # Create a linear regression model
    model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    import pickle  
    with open('linear_regression_model.pkl', "wb") as f:  
        pickle.dump(model, f)  # Save  

if __name__ == "__main__":
    main()