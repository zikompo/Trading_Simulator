import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

###############################################
# 1. DATA PREPARATION
###############################################

def prepare_data_kernel_regression(df, target_symbol, sequence_length=60):
    """
    Prepare data for Kernel Regression in a manner similar to the LSTM example.
    We create sequences of length 'sequence_length', each containing 5 features
    (open, high, low, close, volume), to predict the *next day's close* price.
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Filter for the target symbol and sort by date ascending
    target_df = df[df['symbol'] == target_symbol].sort_values('date')
    
    # We'll use these five features
    features = ['open', 'high', 'low', 'close', 'volume']
    data = target_df[features].values
    
    # Scale the data to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Build sequences of shape (sequence_length, 5) and the next day close as target
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        # Sequence of the last 'sequence_length' days
        sequence = scaled_data[i : i + sequence_length]
        
        # Flatten from (sequence_length, 5) -> (sequence_length*5,)
        sequence_flat = sequence.flatten()
        X.append(sequence_flat)
        
        # The target is the next day's close price (index=3)
        next_day_close = scaled_data[i + sequence_length, 3]
        y.append(next_day_close)
    
    X = np.array(X)
    y = np.array(y)
    
    # Train/test split without shuffling (time-series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test, scaler


###############################################
# 2. MODEL TRAINING
###############################################

def train_kernel_regression(X_train, y_train, alpha=1.0, kernel='rbf', gamma=0.1):
    """
    Train a Kernel Ridge regression model.
    
    You can tune 'alpha', 'kernel', and 'gamma' to avoid singularities or 
    improve performance. For example:
        alpha=1.0, kernel='rbf', gamma=0.1
        alpha=10.0, kernel='rbf', gamma=0.01
    """
    model = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)
    
    model.fit(X_train, y_train)
    return model


###############################################
# 3. EVALUATION
###############################################

def evaluate_model(model, X_test, y_test, scaler, features_shape=5):
    """
    Evaluate the model's predictions and print out error metrics and
    directional accuracy. Also computes buy/sell signal accuracy like the LSTM code.
    
    Returns:
        predictions_inversed, y_test_inversed
    """
    # Predict the next-day close in scaled space
    predictions_scaled = model.predict(X_test)
    
    # Now we need to invert the scaling only for the 'close' column
    # Create dummy arrays to feed to 'inverse_transform'
    # shape -> (#samples, 5). We'll place predictions in index=3 and zeros in others.
    dummy_pred = np.zeros((len(predictions_scaled), features_shape))
    dummy_pred[:, 3] = predictions_scaled  # put predicted close in index=3
    predictions_inversed = scaler.inverse_transform(dummy_pred)[:, 3]
    
    # Same for y_test
    dummy_true = np.zeros((len(y_test), features_shape))
    dummy_true[:, 3] = y_test
    y_test_inversed = scaler.inverse_transform(dummy_true)[:, 3]
    
    # Error metrics
    mse = mean_squared_error(y_test_inversed, predictions_inversed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inversed, predictions_inversed)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    # Directional accuracy
    # We'll compare day-to-day changes (diff)
    actual_dir = np.diff(y_test_inversed, prepend=y_test_inversed[0])
    pred_dir   = np.diff(predictions_inversed, prepend=predictions_inversed[0])
    
    # Skip the very first "diff" because it's from the same day; slice off index=0
    actual_dir = actual_dir[1:]
    pred_dir   = pred_dir[1:]
    
    correct_direction = (np.sign(actual_dir) == np.sign(pred_dir)).sum()
    directional_acc = correct_direction / len(actual_dir) * 100
    
    print(f"Directional Accuracy: {directional_acc:.2f}%")
    
    # Trading signals: same logic as your LSTM example
    threshold = 0.01  # 1% change
    actual_returns = actual_dir / y_test_inversed[:-1]
    predicted_returns = pred_dir / predictions_inversed[:-1]
    
    buy_signals = predicted_returns > threshold
    sell_signals = predicted_returns < -threshold
    
    if buy_signals.sum() > 0:
        correct_buys = (actual_returns[buy_signals] > 0).sum()
        buy_acc = correct_buys / buy_signals.sum() * 100
        print(f"Buy Signal Accuracy: {buy_acc:.2f}% ({correct_buys}/{buy_signals.sum()})")
    else:
        print("No buy signals generated.")
    
    if sell_signals.sum() > 0:
        correct_sells = (actual_returns[sell_signals] < 0).sum()
        sell_acc = correct_sells / sell_signals.sum() * 100
        print(f"Sell Signal Accuracy: {sell_acc:.2f}% ({correct_sells}/{sell_signals.sum()})")
    else:
        print("No sell signals generated.")
    
    trading_signals = np.logical_or(buy_signals, sell_signals)
    if trading_signals.sum() > 0:
        correct_trades = (
            (actual_returns[buy_signals] > 0).sum() +
            (actual_returns[sell_signals] < 0).sum()
        )
        trading_acc = correct_trades / trading_signals.sum() * 100
        print(f"Overall Trading Accuracy: {trading_acc:.2f}% ({correct_trades}/{trading_signals.sum()})")
    else:
        print("No trading signals generated.")
    
    return predictions_inversed, y_test_inversed


###############################################
# 4. PLOTTING
###############################################

def plot_results(actual, predictions, target_symbol):
    """
    Plot actual vs. predicted prices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, color='blue', label='Actual Prices')
    plt.plot(predictions, color='red', label='Predicted Prices')
    plt.title(f'{target_symbol} Price Prediction (Kernel Regression)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


###############################################
# 5. TRADING DECISION
###############################################

def make_trading_decision(model, latest_data, scaler, features_shape=5, threshold=0.01):
    """
    Similar to the LSTM version: we take the last 'sequence_length' days of data,
    flatten them, predict the next day's close, and decide 'buy', 'sell', or 'hold'.
    
    Returns:
        decision (str), predicted_next_price (float), current_price (float)
    """
    # latest_data shape is (1, sequence_length, 5)
    # Flatten for kernel regression
    latest_flat = latest_data.reshape(1, -1)  # (1, sequence_length*5)
    
    # Predict in scaled space
    next_price_scaled = model.predict(latest_flat)[0]  # single float
    
    # Current price scaled is the last day in the sequence
    current_price_scaled = latest_data[0, -1, 3]  # last day's close in scaled space
    
    # Invert scaling
    dummy = np.zeros((1, features_shape))
    
    dummy[0, 3] = next_price_scaled
    predicted_next_price = scaler.inverse_transform(dummy)[0, 3]
    
    dummy[0, 3] = current_price_scaled
    current_price = scaler.inverse_transform(dummy)[0, 3]
    
    # Trading decision
    expected_return = (predicted_next_price - current_price) / current_price
    if expected_return > threshold:
        decision = 'buy'
    elif expected_return < -threshold:
        decision = 'sell'
    else:
        decision = 'hold'
    
    return decision, predicted_next_price, current_price


###############################################
# 6. MAIN SCRIPT
###############################################

def main():
    # 1) Load your CSV data
    df = pd.read_csv('data/stock_data.csv', index_col=None)
    df['date'] = pd.to_datetime(df['date'])
    
    # 2) Select a stock symbol
    target_symbol = 'AAPL'
    
    # 3) Prepare data
    sequence_length = 60
    X_train, X_test, y_train, y_test, scaler = prepare_data_kernel_regression(
        df, target_symbol, sequence_length=sequence_length
    )
    
    # 4) Train the kernel regression model
    #    Adjust alpha, gamma, kernel, etc. if you get singularities or poor performance
    model = train_kernel_regression(
        X_train, 
        y_train, 
        alpha=1.0,   # Try bigger alpha like 10 or 100 if there's a singularity
        kernel='rbf',
        gamma=0.1
    )
    
    # 5) Evaluate
    predictions, actual = evaluate_model(model, X_test, y_test, scaler, features_shape=5)
    
    # 6) Plot
    plot_results(actual, predictions, target_symbol)
    
    # 7) Trading decision: use the last 60 days to predict next day
    target_df = df[df['symbol'] == target_symbol].sort_values('date')
    features = ['open', 'high', 'low', 'close', 'volume']
    
    # Extract the last 'sequence_length' rows and reshape
    recent_data = target_df[features].values[-sequence_length:]
    recent_data_scaled = scaler.transform(recent_data).reshape(1, sequence_length, 5)
    
    decision, next_price, current_price = make_trading_decision(
        model, recent_data_scaled, scaler, features_shape=5
    )
    print(f"Current price: {current_price:.2f}")
    print(f"Predicted next day's close: {next_price:.2f}")
    print(f"Trading decision: {decision.upper()}")

    import pickle  
    with open(f'backend/models/{target_symbol}kernel_regression_model.pkl', "wb") as f:  
        pickle.dump(model, f)  # Save  


if __name__ == "__main__":
    main()
