import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

###############################################
# 1. DATA PREPARATION
###############################################

def prepare_data_linear_regression(df, target_symbol, sequence_length=10, train_days=365, test_days=None):
    """
    Prepare data for Linear Regression.
    We create sequences of length 'sequence_length', each containing 5 features
    (open, high, low, close, volume), to predict the next day's close price.
    
    Parameters:
        df: DataFrame with stock data
        target_symbol: The stock symbol to analyze
        sequence_length: Number of days in each input sequence
        train_days: Number of days to use for training
        test_days: Number of days to use for testing (None = all remaining data)
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, dates_test
    """
    # Filter for the target symbol and sort by date ascending
    target_df = df[df['symbol'] == target_symbol].sort_values('date')
    
    # We'll use these five features
    features = ['open', 'high', 'low', 'close', 'volume']
    data = target_df[features].values
    dates = target_df['date'].values
    
    # Scale the data to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Build sequences and targets for the entire dataset
    X, y, dates_y = [], [], []
    for i in range(len(scaled_data) - sequence_length):
        # Sequence of the last 'sequence_length' days
        sequence = scaled_data[i : i + sequence_length]
        
        # Flatten from (sequence_length, 5) -> (sequence_length*5,)
        sequence_flat = sequence.flatten()
        X.append(sequence_flat)
        
        # The target is the next day's close price (index=3)
        next_day_close = scaled_data[i + sequence_length, 3]
        y.append(next_day_close)
        
        # Keep the date for the target day
        dates_y.append(dates[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    dates_y = np.array(dates_y)
    
    # Split data by time period instead of percentage
    if train_days + sequence_length >= len(X):
        raise ValueError(f"Not enough data for {train_days} training days plus {sequence_length} sequence days")
    
    # Calculate split indices
    train_end_idx = train_days
    
    # Split data
    X_train = X[:train_end_idx]
    y_train = y[:train_end_idx]
    
    # Determine test set size
    if test_days is None:
        # Use all remaining data
        X_test = X[train_end_idx:]
        y_test = y[train_end_idx:]
        dates_test = dates_y[train_end_idx:]
    else:
        # Use specified test days
        test_end_idx = train_end_idx + test_days
        if test_end_idx > len(X):
            test_end_idx = len(X)
        
        X_test = X[train_end_idx:test_end_idx]
        y_test = y[train_end_idx:test_end_idx]
        dates_test = dates_y[train_end_idx:test_end_idx]
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Testing data: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, scaler, dates_test


###############################################
# 2. MODEL TRAINING
###############################################

def train_linear_regression(X_train, y_train):
    """
    Train a simple Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


###############################################
# 3. EVALUATION
###############################################

def evaluate_model(model, X_test, y_test, scaler, dates_test=None, features_shape=5):
    """
    Evaluate the model's predictions and print out error metrics and
    directional accuracy.
    
    Parameters:
        model: The trained model
        X_test: Test features
        y_test: Test targets
        scaler: The fitted scaler
        dates_test: Dates corresponding to test data (optional)
        features_shape: Number of features in the original data
    
    Returns:
        predictions_inversed, y_test_inversed, dates_test
    """
    # Predict the next-day close in scaled space
    predictions_scaled = model.predict(X_test)
    
    # Invert the scaling
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
    actual_dir = np.diff(y_test_inversed, prepend=y_test_inversed[0])
    pred_dir = np.diff(predictions_inversed, prepend=predictions_inversed[0])
    
    # Skip the first diff
    actual_dir = actual_dir[1:]
    pred_dir = pred_dir[1:]
    
    correct_direction = (np.sign(actual_dir) == np.sign(pred_dir)).sum()
    directional_acc = correct_direction / len(actual_dir) * 100
    
    print(f"Directional Accuracy: {directional_acc:.2f}%")
    
    # Calculate trading signals
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
    
    return predictions_inversed, y_test_inversed, dates_test


###############################################
# 4. PLOTTING
###############################################

def plot_results(actual, predictions, target_symbol, dates=None):
    """
    Plot actual vs. predicted prices.
    
    Parameters:
        actual: Actual prices
        predictions: Predicted prices
        target_symbol: Stock symbol
        dates: Dates corresponding to the data points (optional)
    """
    plt.figure(figsize=(12, 6))
    
    if dates is not None:
        # Convert dates to datetime if they're not already
        if not isinstance(dates[0], pd.Timestamp):
            dates = pd.to_datetime(dates)
        
        # Plot with dates on x-axis
        plt.plot(dates, actual, color='blue', label='Actual Prices')
        plt.plot(dates, predictions, color='red', label='Predicted Prices')
        plt.gcf().autofmt_xdate()  # Rotate date labels
        
        # Add markers at fixed intervals if there are many data points
        if len(dates) > 30:
            interval = len(dates) // 10  # Show about 10 markers
            plt.plot(dates[::interval], actual[::interval], 'bo', alpha=0.7)
            plt.plot(dates[::interval], predictions[::interval], 'ro', alpha=0.7)
    else:
        # Plot without dates
        plt.plot(actual, color='blue', label='Actual Prices')
        plt.plot(predictions, color='red', label='Predicted Prices')
    
    # Calculate and show error metrics on the plot
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    
    # Add a text box with metrics
    textstr = f'RMSE: {rmse:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.title(f'{target_symbol} Price Prediction (Linear Regression)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


###############################################
# 5. TRADING DECISION
###############################################

def make_trading_decision(model, latest_data, scaler, features_shape=5, threshold=0.01):
    """
    Predict the next day's close price and decide to buy, sell, or hold.
    
    Returns:
        decision (str), predicted_next_price (float), current_price (float)
    """
    # Flatten the data for linear regression
    latest_flat = latest_data.reshape(1, -1)
    
    # Predict in scaled space
    next_price_scaled = model.predict(latest_flat)[0]
    
    # Current price scaled is the last day in the sequence
    current_price_scaled = latest_data[0, -1, 3]
    
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
    df = pd.read_csv('../data/stock_data.csv', index_col=None)
    df['date'] = pd.to_datetime(df['date'])
    
    # 2) Select a stock symbol
    target_symbol = 'AAPL'
    
    # 3) Prepare data with explicit time-based split
    sequence_length = 10
    train_days = 252  # Approximately 1 year of trading days
    test_days = None  # Use all remaining data for testing
    
    X_train, X_test, y_train, y_test, scaler, dates_test = prepare_data_linear_regression(
        df, 
        target_symbol, 
        sequence_length=sequence_length,
        train_days=train_days,
        test_days=test_days
    )
    
    # 4) Train the linear regression model
    model = train_linear_regression(X_train, y_train)
    
    # 5) Print model coefficients for interpretability
    feature_count = X_train.shape[1] // 5
    features = ['open', 'high', 'low', 'close', 'volume']
    print("\nModel Coefficients:")
    for i in range(feature_count):
        day = i // 5
        feature = features[i % 5]
        print(f"Day -{feature_count-day}, {feature}: {model.coef_[i]:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")
    
    # 6) Evaluate on the test set
    predictions, actual, dates = evaluate_model(
        model, X_test, y_test, scaler, dates_test, features_shape=5
    )
    
    # 7) Plot results over time
    plot_results(actual, predictions, target_symbol, dates)
    
    # 8) Analyze prediction accuracy over different time periods
    if len(predictions) > 60:  # If we have enough test data
        print("\nPerformance over different time horizons:")
        
        # First month (approximately 21 trading days)
        first_month = min(21, len(predictions))
        mse_first_month = mean_squared_error(actual[:first_month], predictions[:first_month])
        print(f"First month RMSE: {np.sqrt(mse_first_month):.4f}")
        
        # First quarter (approximately 63 trading days)
        first_quarter = min(63, len(predictions))
        if first_quarter > first_month:
            mse_first_quarter = mean_squared_error(
                actual[first_month:first_quarter], 
                predictions[first_month:first_quarter]
            )
            print(f"Rest of first quarter RMSE: {np.sqrt(mse_first_quarter):.4f}")
        
        # Rest of the data
        if len(predictions) > first_quarter:
            mse_rest = mean_squared_error(
                actual[first_quarter:], 
                predictions[first_quarter:]
            )
            print(f"Remaining period RMSE: {np.sqrt(mse_rest):.4f}")
    
    # 9) Trading decision for the latest data
    target_df = df[df['symbol'] == target_symbol].sort_values('date')
    features = ['open', 'high', 'low', 'close', 'volume']
    
    # Extract the last 'sequence_length' rows
    recent_data = target_df[features].values[-sequence_length:]
    recent_data_scaled = scaler.transform(recent_data).reshape(1, sequence_length, 5)
    
    decision, next_price, current_price = make_trading_decision(
        model, recent_data_scaled, scaler, features_shape=5
    )
    print(f"\nCurrent price: {current_price:.2f}")
    print(f"Predicted next day's close: {next_price:.2f}")
    print(f"Trading decision: {decision.upper()}")
    import pickle  
    with open(f'../backend/models/{target_symbol}_parametric_regression_model.pkl', "wb") as f:  
        pickle.dump(model, f)  # Save  


if __name__ == "__main__":
    main()
    
    # Alternatively, run with custom parameters
    def run_with_params(symbol='AAPL', train_days=252, test_days=None, sequence_length=10):
        """
        Run the model with custom parameters
        
        Parameters:
            symbol: Stock symbol to analyze
            train_days: Number of days to use for training
            test_days: Number of days to use for testing (None = all remaining data)
            sequence_length: Number of days in each sequence
        """
        print(f"\nRunning model for {symbol} with {train_days} training days and {sequence_length} sequence length")
        
        df = pd.read_csv('../data/stock_data.csv', index_col=None)
        df['date'] = pd.to_datetime(df['date'])
        
        X_train, X_test, y_train, y_test, scaler, dates_test = prepare_data_linear_regression(
            df, symbol, sequence_length, train_days, test_days
        )
        
        model = train_linear_regression(X_train, y_train)
        predictions, actual, dates = evaluate_model(model, X_test, y_test, scaler, dates_test)
        plot_results(actual, predictions, symbol, dates)   
        return model, scaler, (predictions, actual, dates)
    
    # Example: run with different parameters
    # model, scaler, results = run_with_params(symbol='AAPL', train_days=500, test_days=100, sequence_length=20)