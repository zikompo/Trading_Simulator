import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gc

def load_data(path):
    # Read in chunks to reduce memory usage
    return pd.read_csv(path, parse_dates=['Date' if 'Date' in pd.read_csv(path, nrows=1).columns else 'date'])

def data_generator(df, features, sequence_length, batch_size=32):
    """
    Generator function that yields batches of data sequences and targets.
    This avoids storing all sequences in memory at once.
    """
    symbols = df['Symbol'].unique()
    
    while True:
        # Randomly select symbols for this batch
        batch_symbols = np.random.choice(symbols, min(len(symbols), batch_size), replace=False)
        X_batch = []
        y_batch = []
        
        for symbol in batch_symbols:
            # Get data for this symbol
            symbol_df = df[df['Symbol'] == symbol].sort_values('Date' if 'Date' in df.columns else 'date')
            if len(symbol_df) <= sequence_length:
                continue
                
            # Get feature data
            data = symbol_df[features].values
            
            # Randomly select a starting point for a sequence
            start_idx = np.random.randint(0, len(data) - sequence_length - 1)
            sequence = data[start_idx:start_idx + sequence_length]
            target = data[start_idx + sequence_length, 3]  # Next day's close price
            
            X_batch.append(sequence.flatten())
            y_batch.append(target)
            
        if X_batch:  # Make sure we have data
            yield np.array(X_batch), np.array(y_batch)

def prepare_validation_data(df, features, sequence_length, validation_symbols=None):
    """
    Prepare a fixed validation set for model evaluation.
    Uses a subset of symbols to keep memory usage down.
    """
    # If no validation symbols provided, select a few
    if validation_symbols is None:
        all_symbols = df['Symbol'].unique()
        validation_symbols = np.random.choice(all_symbols, min(5, len(all_symbols)), replace=False)
    
    X_val = []
    y_val = []
    
    for symbol in validation_symbols:
        symbol_df = df[df['Symbol'] == symbol].sort_values('Date' if 'Date' in df.columns else 'date')
        if len(symbol_df) <= sequence_length:
            continue
            
        data = symbol_df[features].values
        
        # Use the last 20% of data for validation
        split_idx = int(len(data) * 0.8)
        val_data = data[split_idx:]
        
        for i in range(len(val_data) - sequence_length):
            sequence = val_data[i:i + sequence_length]
            target = val_data[i + sequence_length, 3]
            X_val.append(sequence.flatten())
            y_val.append(target)
            
    return np.array(X_val), np.array(y_val)

def scale_features(df, features):
    """
    Scale features and return the scaler for later use.
    """
    scaler = MinMaxScaler()
    # Only fit on a sample to save memory
    sample = df.sample(min(10000, len(df)))[features]
    scaler.fit(sample)
    
    # Process in chunks of 10,000 rows
    chunk_size = 10000
    for start_idx in range(0, len(df), chunk_size):
        end_idx = min(start_idx + chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx]
        df.iloc[start_idx:end_idx, df.columns.get_indexer(features)] = scaler.transform(chunk[features])
        
    return scaler

def prepare_data_for_training(file_path, sequence_length=60):
    """
    Main function to prepare data for model training with minimal memory usage.
    Returns a data generator and validation data.
    """
    print("Loading data...")
    df = load_data(file_path)
    
    # Convert column names to match your expected format
    if 'date' in df.columns and 'Date' not in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
    if 'symbol' in df.columns and 'Symbol' not in df.columns:
        df.rename(columns={'symbol': 'Symbol'}, inplace=True)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Verify required columns exist
    for col in ['Symbol', 'Date'] + features:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in data")
    
    print("Scaling features...")
    scaler = scale_features(df, features)
    
    # Select validation symbols
    validation_symbols = np.random.choice(df['Symbol'].unique(), 
                                         min(3, len(df['Symbol'].unique())),
                                         replace=False)
    
    print("Preparing validation data...")
    X_val, y_val = prepare_validation_data(df, features, sequence_length, validation_symbols)
    
    print(f"Validation data shape: X_val {X_val.shape}, y_val {y_val.shape}")
    
    # Create data generator for training
    train_gen = data_generator(
        df[~df['Symbol'].isin(validation_symbols)],  # Use symbols not in validation
        features, 
        sequence_length
    )
    
    return train_gen, X_val, y_val, scaler