
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def prepare_data_kernel_regression(df, target_symbol, sequence_length=60):
    """
    Prepare time-series data for kernel regression.
    """
    target_df = df[df['Symbol'] == target_symbol].sort_values('Date')

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = target_df[features].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        sequence = scaled_data[i : i + sequence_length]
        sequence_flat = sequence.flatten()
        X.append(sequence_flat)
        y.append(scaled_data[i + sequence_length, 3])  # next day's close

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test, scaler
