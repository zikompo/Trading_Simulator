
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def prepare_data_kernel_regression_all(df, sequence_length=60):
    """
    Combines sequences from all stocks. Keeps time order.
    Uses only Open, High, Low, Close, Volume.
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    all_X, all_y = [], []

    # Fit one scaler globally (optional: you could also scale per stock if needed)
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    for symbol in df['Symbol'].unique():
        target_df = df[df['Symbol'] == symbol].sort_values('Date')
        data = target_df[features].values

        for i in range(len(data) - sequence_length):
            sequence = data[i:i + sequence_length]
            target_close = data[i + sequence_length, 3]  # next day's close
            all_X.append(sequence.flatten())
            all_y.append(target_close)

    X = np.array(all_X)
    y = np.array(all_y)

    # Split by time: no shuffle
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, scaler
