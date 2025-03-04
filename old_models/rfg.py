import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

###############################################################################
# 1) DATA PREPARATION
###############################################################################

def prepare_data_random_forest(df, target_symbol='AAPL', sequence_length=60):
    """
    Similar to the LSTM approach: we build rolling-window sequences
    of length 'sequence_length' for features [open, high, low, close, volume].
    Then the label is the *next day’s* close price.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Filter for the target symbol, sort by date
    symbol_df = df[df['symbol'] == target_symbol].sort_values('date')

    # We'll use these columns as features
    features = ['open', 'high', 'low', 'close', 'volume']
    data = symbol_df[features].values

    # Scale features to [0, 1] (optional for RandomForest, but can help consistency)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Build sequences + next-day close
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        # Last `sequence_length` rows are the feature window
        window = scaled_data[i : i + sequence_length]
        # Flatten => shape (sequence_length * 5,)
        window_flat = window.flatten()
        X.append(window_flat)

        # Next day’s close is at index i+sequence_length, col index=3 (close)
        next_close = scaled_data[i + sequence_length, 3]
        y.append(next_close)

    X = np.array(X)
    y = np.array(y)

    # 80% train, 20% test, no shuffle for time series
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test, scaler

###############################################################################
# 2) RANDOM FOREST TRAINING
###############################################################################

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Trains a RandomForestRegressor with the given hyperparameters.
    Adjust these as needed for better performance or to avoid overfitting.
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,  # for reproducibility
        n_jobs=-1         # use all cores
    )
    rf.fit(X_train, y_train)
    return rf

###############################################################################
# 3) EVALUATION
###############################################################################

def evaluate_model(model, X_test, y_test, scaler, features_shape=5):
    """
    Makes predictions, inverts scaling for the close column, and prints
    out MSE, RMSE, MAE, and directional accuracy. Also returns
    the unscaled predictions and actuals for optional plotting.

    features_shape=5 => We have [open, high, low, close, volume].
    """
    # Predict in scaled space
    y_pred_scaled = model.predict(X_test)

    # We only scaled the feature columns, not the y array, so let's do a trick:
    # Create dummy arrays of shape (#samples, 5). Put predictions in col=3, invert.
    dummy_pred = np.zeros((len(y_pred_scaled), features_shape))
    dummy_pred[:, 3] = y_pred_scaled
    predictions_inversed = scaler.inverse_transform(dummy_pred)[:, 3]

    # Same for y_test
    dummy_true = np.zeros((len(y_test), features_shape))
    dummy_true[:, 3] = y_test
    actual_inversed = scaler.inverse_transform(dummy_true)[:, 3]

    # Error metrics
    mse = mean_squared_error(actual_inversed, predictions_inversed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_inversed, predictions_inversed)

    print(f"Mean Squared Error:      {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error:     {mae:.4f}")

    # Directional accuracy
    # Compare day-to-day changes
    actual_dir = np.diff(actual_inversed, prepend=actual_inversed[0])[1:]
    pred_dir   = np.diff(predictions_inversed, prepend=predictions_inversed[0])[1:]
    correct_dir = (np.sign(actual_dir) == np.sign(pred_dir)).sum()
    direction_acc = correct_dir / len(actual_dir) * 100

    print(f"Directional Accuracy:    {direction_acc:.2f}%")

    return predictions_inversed, actual_inversed

###############################################################################
# 4) PLOTTING
###############################################################################

def plot_results(actual, predictions, target_symbol='AAPL'):
    """
    Simple line plot of actual vs. predicted close prices over the test set.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Close', color='blue')
    plt.plot(predictions, label='Predicted Close', color='red')
    plt.title(f'Random Forest: {target_symbol} Close Price Prediction')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

###############################################################################
# 5) TRADING DECISION (OPTIONAL)
###############################################################################

def make_trading_decision(model, df, scaler, target_symbol='AAPL',
                          sequence_length=60, threshold=0.01):
    """
    Creates a single forecast from the last `sequence_length` days,
    then compares predicted close to the most recent actual close
    to produce a toy 'buy', 'sell', or 'hold' decision.

    threshold=0.01 => 1% required to buy/sell
    """
    # Filter data, sort
    symbol_df = df[df['symbol'] == target_symbol].sort_values('date')
    features = ['open', 'high', 'low', 'close', 'volume']

    # Last `sequence_length` rows
    recent_data = symbol_df[features].values[-sequence_length:]
    recent_data_scaled = scaler.transform(recent_data)
    recent_data_scaled_flat = recent_data_scaled.flatten().reshape(1, -1)

    # Predict next day's close
    pred_close_scaled = model.predict(recent_data_scaled_flat)[0]

    # Invert scaling for that predicted close
    dummy = np.zeros((1, len(features)))
    dummy[0, 3] = pred_close_scaled
    pred_close = scaler.inverse_transform(dummy)[0, 3]

    # Current actual close is the last row's close
    current_close = symbol_df['close'].iloc[-1]

    expected_return = (pred_close - current_close) / current_close

    if expected_return > threshold:
        decision = 'buy'
    elif expected_return < -threshold:
        decision = 'sell'
    else:
        decision = 'hold'

    return decision, pred_close, current_close

###############################################################################
# 6) MAIN SCRIPT
###############################################################################

def main():
    # 1) Load your CSV
    df = pd.read_csv('data/stock_data.csv', index_col=None)
    df['date'] = pd.to_datetime(df['date'])

    # 2) Prepare data
    target_symbol = 'AAPL'
    sequence_length = 60  # last 60 days
    X_train, X_test, y_train, y_test, scaler = prepare_data_random_forest(
        df, target_symbol=target_symbol, sequence_length=sequence_length
    )

    # 3) Train the Random Forest
    # Adjust n_estimators, max_depth as needed
    model = train_random_forest(
        X_train, y_train,
        n_estimators=200,  # e.g. 200 trees
        max_depth=10       # or None, if you want fully grown trees
    )

    # 4) Evaluate on test
    predictions, actual = evaluate_model(model, X_test, y_test, scaler)

    # 5) Plot
    plot_results(actual, predictions, target_symbol)

    # 6) Trading decision example
    decision, next_price, current_price = make_trading_decision(
        model, df, scaler,
        target_symbol=target_symbol,
        sequence_length=sequence_length,
        threshold=0.01
    )
    print("\nTrading Decision:")
    print(f"  Current Close:    {current_price:.2f}")
    print(f"  Predicted Close:  {next_price:.2f}")
    print(f"  Decision:         {decision.upper()}")

if __name__ == "__main__":
    main()
