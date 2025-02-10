from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

def read_csv(path):
    """Read CSV file from the given path."""
    return pd.read_csv(path)

def get_data():
    """Read data from CSV, split into features and target, then into training and test sets."""
    df = read_csv('data/data.csv')
    columns = df.columns
    features = columns[:-1]
    X = df[features]
    y = df['target']
    # Using shuffle=False for time series data; remove if not needed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
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
    print(f'Mean Squared Error: {mse}')
    
    # Print model coefficients
    print(f'Intercept: {model.intercept_}')
    print(f'Coefficients: {model.coef_}')

if __name__ == "__main__":
    main()