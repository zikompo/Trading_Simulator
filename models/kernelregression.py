import os
import pickle
import pandas as pd
import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# New imports for hyperparameter tuning
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Your existing preprocessing function
from preprocessing import prepare_data_kernel_regression  

###############################################
# 1. HELPER: Train with optional hyperparameter tuning
###############################################

def train_kernel_regression_with_search(X_train, y_train):
    """
    Use GridSearchCV with TimeSeriesSplit to pick the best alpha/gamma
    for Kernel Ridge. Expand param_grid for more thorough search.
    """
    # Prepare a small param grid to try. Feel free to expand these ranges.
    param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'gamma': [0.001, 0.01, 0.1, 1.0],
        'kernel': ['rbf']  # We stick with RBF only
    }
    
    # Use TimeSeriesSplit so that training only sees the "past" folds
    tscv = TimeSeriesSplit(n_splits=3)  # or 5 folds, etc.
    
    # Set up the GridSearch with negative MSE as our scoring
    search = GridSearchCV(
        KernelRidge(),
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=tscv,
        n_jobs=-1,     # use all CPU cores
        verbose=0      # show less output; change to 2 or 3 if you want more detail
    )
    
    search.fit(X_train, y_train)
    print(f"Best params via GridSearch: {search.best_params_}")
    
    # Return the best model
    best_model = search.best_estimator_
    return best_model

###############################################
# 2. EVALUATION (unchanged)
###############################################

def evaluate_model(model, X_test, y_test, scaler, features_shape=5):
    predictions_scaled = model.predict(X_test)

    # Re-inject predicted close into dummy array for inverse_transform
    dummy_pred = np.zeros((len(predictions_scaled), features_shape))
    dummy_pred[:, 3] = predictions_scaled
    predictions_inversed = scaler.inverse_transform(dummy_pred)[:, 3]

    # Re-inject true close into dummy array for inverse_transform
    dummy_true = np.zeros((len(y_test), features_shape))
    dummy_true[:, 3] = y_test
    y_test_inversed = scaler.inverse_transform(dummy_true)[:, 3]

    mse = mean_squared_error(y_test_inversed, predictions_inversed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inversed, predictions_inversed)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    # Directional accuracy
    actual_dir = np.diff(y_test_inversed, prepend=y_test_inversed[0])[1:]
    pred_dir = np.diff(predictions_inversed, prepend=predictions_inversed[0])[1:]

    correct_direction = (np.sign(actual_dir) == np.sign(pred_dir)).sum()
    directional_acc = correct_direction / len(actual_dir) * 100
    print(f"Directional Accuracy: {directional_acc:.2f}%")

    # Trading signals
    threshold = 0.01
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
# 3. TRADING DECISION (unchanged)
###############################################

def make_trading_decision(model, latest_data, scaler, features_shape=5, threshold=0.01):
    latest_flat = latest_data.reshape(1, -1)
    next_price_scaled = model.predict(latest_flat)[0]
    current_price_scaled = latest_data[0, -1, 3]

    dummy = np.zeros((1, features_shape))
    dummy[0, 3] = next_price_scaled
    predicted_next_price = scaler.inverse_transform(dummy)[0, 3]

    dummy[0, 3] = current_price_scaled
    current_price = scaler.inverse_transform(dummy)[0, 3]

    expected_return = (predicted_next_price - current_price) / current_price
    if expected_return > threshold:
        decision = 'buy'
    elif expected_return < -threshold:
        decision = 'sell'
    else:
        decision = 'hold'

    return decision, predicted_next_price, current_price

###############################################
# 4. PLOTTING (unchanged)
###############################################

def plot_results(actual, predictions, target_symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, color='blue', label='Actual Prices')
    plt.plot(predictions, color='red', label='Predicted Prices')
    plt.title(f'{target_symbol} Price Prediction (Kernel Regression)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

###############################################
# 5. MAIN SCRIPT
###############################################

def main():
    # 1) Load CSV & fix column names
    df = pd.read_csv('data/updated_data.csv')
    df.columns = df.columns.str.strip() 
    df['date'] = pd.to_datetime(df['Date'])

    sequence_length = 60
    symbols = df['Symbol'].unique()

    for symbol in symbols:
        print(f"\n====== Processing {symbol} ======")
        try:
            # 2) Prepare data
            X_train, X_test, y_train, y_test, scaler = prepare_data_kernel_regression(
                df, symbol, sequence_length=sequence_length
            )
            
            # 3) Train with hyperparam search
            model = train_kernel_regression_with_search(X_train, y_train)

            # 4) Evaluate
            predictions, actual = evaluate_model(
                model, X_test, y_test, scaler, features_shape=5
            )

            # 5) Plot
            plot_results(actual, predictions, symbol)

            # 6) Trading decision
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            target_df = df[df['Symbol'] == symbol].sort_values('Date')
            recent_data = target_df[features].values[-sequence_length:]
            recent_data_scaled = scaler.transform(recent_data).reshape(1, sequence_length, 5)

            decision, next_price, current_price = make_trading_decision(
                model, recent_data_scaled, scaler, features_shape=5
            )
            print(f"Current price: {current_price:.2f}")
            print(f"Predicted next day's close: {next_price:.2f}")
            print(f"Trading decision: {decision.upper()}")

            # 7) Save model
            os.makedirs("backend/models", exist_ok=True)
            with open(f"backend/models/{symbol}_kernel_regression_model.pkl", "wb") as f:
                pickle.dump(model, f)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    main()
