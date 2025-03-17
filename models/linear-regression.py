import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def read_csv(path):
    """Read CSV file from the given path."""
    return pd.read_csv(path)

def get_data():
    """Read data from CSV, split into features and target, then into training and test sets."""
    df = read_csv('data/data.csv')
    print(f"Initial data shape: {df.shape}")
    
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Select only relevant features for stock price prediction
    selected_features = [
        'returns', 'high_low_ratio', 'close_open_ratio', 'volume_change',
        'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_4', 'return_lag_5',
        'return_lag_6', 'return_lag_7', 'return_lag_8', 'return_lag_9', 'return_lag_10',
        'hl_ratio_lag_1', 'hl_ratio_lag_2', 'hl_ratio_lag_3', 'hl_ratio_lag_4', 'hl_ratio_lag_5',
        'co_ratio_lag_1', 'co_ratio_lag_2', 'co_ratio_lag_3', 'co_ratio_lag_4', 'co_ratio_lag_5',
        'volume_change_lag_1', 'volume_change_lag_2', 'volume_change_lag_3', 'volume_change_lag_4', 'volume_change_lag_5',
        'return_std_5', 'return_std_10', 'return_std_20'
    ]
    
    df = df[selected_features + ['target']]  # Include target column in the dataframe
    
    print(f"Columns after selecting features: {df.columns}")
    
    # Ensure all features are numeric and convert if necessary
    for feature in selected_features:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            print(f"Converting feature '{feature}' to numeric.")
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    # Drop rows with NaN values after conversion
    df = df.dropna()
    print(f"Data shape after cleaning: {df.shape}")

    # Check if the dataset is empty after dropping rows
    if df.empty:
        print("The dataset is empty after preprocessing.")
        return None, None, None, None
    
    # Separate features and target
    X = df[selected_features]
    y = df['target']

    # Split data into train and test sets (don't shuffle for time series data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

    
    # Check if the dataset is empty
    if df.empty:
        print("The dataset is empty after preprocessing.")
        return None, None, None, None
    
    X = df[features]
    y = df['target']

    # Using shuffle=False for time series data; remove if not needed
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