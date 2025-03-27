import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def read_csv(path):
    """Read CSV file from the given path."""
    return pd.read_csv(path)

def get_data():
    """Load data, select features and target, then split into training and test sets.
    
    This version uses only the High, Open, and Low prices as features
    to predict the Close price.
    """
    # Load the data with Date parsed as datetime
    df = pd.read_csv('/Users/devshah/Documents/WorkSpace/University/year 3/CSC392/Trading_Simulator/data/updated_data.csv', parse_dates=['Date'])
    print(f"Initial data shape: {df.shape}")
    
    # Use only the essential price columns
    fin_df = df[['High', 'Open', 'Low', 'Close']]
    
    # Define features and target
    X = fin_df[['High', 'Open', 'Low']]  # independent variables
    y = fin_df['Close']                  # target variable
    
    # Split into training and test sets (70:30 ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test = get_data()
    
    # Create and train the linear regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = lr_model.predict(X_test)
    
    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Perform cross-validation using KFold with 20 splits
    kfold = KFold(n_splits=20, shuffle=True, random_state=2)
    cv_scores = cross_val_score(lr_model, X_test, y_test, cv=kfold, scoring='r2')
    print("Cross-validated R2: ", cv_scores.mean())
    
    # Save the trained model to disk
    with open('linear_regression_model.pkl', "wb") as f:
        pickle.dump(lr_model, f)

if __name__ == "__main__":
    main()
