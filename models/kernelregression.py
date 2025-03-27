import numpy as np
import pandas as pd
import os
import joblib
import gc

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def advanced_feature_engineering(df):
    """
    Create additional features for more robust prediction
    """
    df['Price_Change'] = df['Close'].pct_change()
    df['Moving_Average_5'] = df['Close'].rolling(window=5).mean()
    df['Moving_Average_10'] = df['Close'].rolling(window=10).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Open_Close_Ratio'] = df['Open'] / df['Close']
    
    return df

def prepare_data(file_path, test_size=0.2, look_back=5):
    """
    Comprehensive data preparation with advanced feature engineering
    """
    df = pd.read_csv(file_path, parse_dates=['Date'])
    
    df = df.sort_values('Date')
    
    df = advanced_feature_engineering(df)
    
    df.dropna(inplace=True)
    
    feature_columns = [
        'Open', 'High', 'Low', 'Volume', 
        'Price_Change', 'Moving_Average_5', 
        'Moving_Average_10', 'Volume_Change', 
        'High_Low_Ratio', 'Open_Close_Ratio'
    ]
    
    all_X, all_y = [], []
    
    for symbol in df['Symbol'].unique():
        symbol_df = df[df['Symbol'] == symbol]
        
        for i in range(len(symbol_df) - look_back):
            seq = symbol_df[feature_columns].iloc[i:i+look_back].values.flatten()
            target = symbol_df['Close'].iloc[i+look_back]
            
            all_X.append(seq)
            all_y.append(target)
    
    X = np.array(all_X)
    y = np.array(all_y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    split_idx = int(len(X_scaled) * (1 - test_size))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

def train_kernel_regression(file_path, model_save_path="kernel_model.pkl", epochs=10):
    """
    Advanced Kernel Ridge Regression with hyperparameter tuning and epoch tracking
    """
    X_train, X_test, y_train, y_test, scaler = prepare_data(file_path)
    
    param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf', 'polynomial'],
        'gamma': ['scale', 'auto', 0.1, 1.0]
    }
    
    krr = KernelRidge()
    grid_search = GridSearchCV(
        estimator=krr, 
        param_grid=param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Epoch tracking
    epoch_losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Predict and calculate loss
        y_pred_train = best_model.predict(X_train)
        epoch_loss = mean_squared_error(y_train, y_pred_train)
        epoch_losses.append(epoch_loss)
        
        print(f"Epoch Loss (MSE): {epoch_loss:.4f}")
        print("Best Hyperparameters:", grid_search.best_params_)
    
    # Final evaluation
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("\nFinal Training Metrics:")
    print(f"MSE: {train_mse:.4f}")
    print(f"MAE: {train_mae:.4f}")
    print(f"R2 Score: {train_r2:.4f}")
    
    print("\nFinal Testing Metrics:")
    print(f"MSE: {test_mse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"R2 Score: {test_r2:.4f}")
    
    joblib.dump({
        'model': best_model,
        'scaler': scaler
    }, model_save_path)
    
    print(f"\nModel saved to {model_save_path}")
    
    return best_model, (train_mse, test_mse, train_r2, test_r2)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    model, metrics = train_kernel_regression(
        "/Users/devshah/Documents/WorkSpace/University/year 3/CSC392/Trading_Simulator/data/updated_data.csv", 
        model_save_path="models/kernel_regression_model.pkl",
        epochs=10  # You can adjust the number of epochs
    )
