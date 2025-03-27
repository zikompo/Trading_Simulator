import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import pickle
import h5py

# Import the data preparation function from your existing script
from preprocessing import prepare_data_for_training

def train_random_forest_model(file_path, 
                               sequence_length=60, 
                               n_estimators=100, 
                               max_depth=10, 
                               min_samples_split=2, 
                               min_samples_leaf=1,
                               random_state=42,
                               save_format='joblib'):
    """
    Train a Random Forest Regressor on stock price data
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing stock data
    sequence_length : int, optional (default=60)
        Number of previous time steps to use for prediction
    n_estimators : int, optional (default=100)
        Number of trees in the random forest
    max_depth : int, optional (default=10)
        Maximum depth of the trees
    min_samples_split : int, optional (default=2)
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, optional (default=1)
        Minimum number of samples required to be at a leaf node
    random_state : int, optional (default=42)
        Controls both the randomness of the bootstrapping of the samples 
        used when building trees and the sampling of the features to consider 
        when looking for the best split at each node
    save_format : str, optional (default='joblib')
        Format to save the model. Options: 'joblib', 'pkl', 'h5'
    
    Returns:
    --------
    dict
        A dictionary containing the trained model, validation metrics, and scaler
    """
    # Prepare the data
    train_gen, X_val, y_val, scaler = prepare_data_for_training(
        file_path, 
        sequence_length=sequence_length
    )
    
    # Instantiate the Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    # Collect training data
    print("Collecting training data...")
    X_train, y_train = [], []
    for _ in range(1000):  # Collect 1000 sequences
        X_batch, y_batch = next(train_gen)
        X_train.extend(X_batch)
        y_train.extend(y_batch)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Training data shape: X_train {X_train.shape}, y_train {y_train.shape}")
    
    # Train the model
    print("Training Random Forest Regressor...")
    rf_model.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred = rf_model.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save the model based on the specified format
    if save_format.lower() == 'joblib':
        model_path = 'results/random_forest_stock_model.joblib'
        joblib.dump(rf_model, model_path)
        print(f"Model saved to {model_path}")
        
        scaler_path = 'results/stock_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
    
    elif save_format.lower() == 'pkl':
        model_path = 'results/random_forest_stock_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"Model saved to {model_path}")
        
        scaler_path = 'results/stock_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
    
    elif save_format.lower() == 'h5':
        # For h5, we'll need to save both the model and the scaler
        model_path = 'results/random_forest_stock_model.h5'
        with h5py.File(model_path, 'w') as hf:
            # Save important attributes of the model
            hf.create_dataset('n_estimators', data=rf_model.n_estimators)
            hf.create_dataset('max_depth', data=rf_model.max_depth)
            
            # Save feature importances
            hf.create_dataset('feature_importances', data=rf_model.feature_importances_)
            
            # Note: h5py can't directly save sklearn models, so we'll use pickle within h5py
            estimators_group = hf.create_group('estimators')
            for i, estimator in enumerate(rf_model.estimators_):
                pickled_estimator = pickle.dumps(estimator)
                estimators_group.create_dataset(f'estimator_{i}', data=np.void(pickled_estimator))
        print(f"Model saved to {model_path}")
        
        scaler_path = 'results/stock_scaler.h5'
        with h5py.File(scaler_path, 'w') as hf:
            # Save scaler attributes
            hf.create_dataset('scale_', data=scaler.scale_)
            hf.create_dataset('min_', data=scaler.min_)
            hf.create_dataset('data_min_', data=scaler.data_min_)
            hf.create_dataset('data_max_', data=scaler.data_max_)
            hf.create_dataset('feature_range', data=scaler.feature_range)
        print(f"Scaler saved to {scaler_path}")
    
    else:
        raise ValueError("Unsupported save format. Choose 'joblib', 'pkl', or 'h5'.")
    
    # Print and return results
    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")
    
    return {
        'model': rf_model,
        'metrics': {
            'mse': mse,
            'mae': mae,
            'r2': r2
        },
        'scaler': scaler,
        'validation_data': {
            'X_val': X_val,
            'y_val': y_val
        }
    }

def load_model_and_scaler(model_path='results/random_forest_stock_model.joblib', 
                           scaler_path='results/stock_scaler.joblib',
                           load_format='joblib'):
    """
    Load a previously saved Random Forest model and its scaler
    
    Parameters:
    -----------
    model_path : str, optional
        Path to the saved model file
    scaler_path : str, optional
        Path to the saved scaler file
    load_format : str, optional
        Format of the saved model. Options: 'joblib', 'pkl', 'h5'
    
    Returns:
    --------
    tuple
        A tuple containing the loaded model and scaler
    """
    if load_format.lower() == 'joblib':
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    
    elif load_format.lower() == 'pkl':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    elif load_format.lower() == 'h5':
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import MinMaxScaler
        
        # Load model from h5
        with h5py.File(model_path, 'r') as hf:
            n_estimators = hf['n_estimators'][()]
            max_depth = hf['max_depth'][()]
            
            # Recreate the model
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth
            )
            
            # Restore estimators
            estimators_group = hf['estimators']
            model.estimators_ = []
            for key in estimators_group.keys():
                pickled_estimator = estimators_group[key][()].tostring()
                model.estimators_.append(pickle.loads(pickled_estimator))
            
            # Restore feature importances
            model.feature_importances_ = hf['feature_importances'][()]
        
        # Load scaler from h5
        with h5py.File(scaler_path, 'r') as hf:
            scaler = MinMaxScaler()
            scaler.scale_ = hf['scale_'][()]
            scaler.min_ = hf['min_'][()]
            scaler.data_min_ = hf['data_min_'][()]
            scaler.data_max_ = hf['data_max_'][()]
            scaler.feature_range = hf['feature_range'][()]
    
    else:
        raise ValueError("Unsupported load format. Choose 'joblib', 'pkl', or 'h5'.")
    
    return model, scaler

def predict_next_price(model, scaler, input_sequence):
    """
    Make a prediction for the next stock price
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained Random Forest model
    scaler : MinMaxScaler
        Fitted scaler used to scale the features
    input_sequence : numpy.ndarray
        Scaled input sequence of features
    
    Returns:
    --------
    float
        Predicted next day's closing price
    """
    # Ensure input is a 2D array
    if input_sequence.ndim == 1:
        input_sequence = input_sequence.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_sequence)
    return prediction[0]

# Example usage
if __name__ == '__main__':
    # Replace with your actual CSV file path
    stock_data_path = 'data/updated_data.csv'
    
    # Train the model
    results = train_random_forest_model(stock_data_path, save_format='h5')
    
    # Optionally, load and use the model
    loaded_model, loaded_scaler = load_model_and_scaler()
    
    # Example prediction (you'd need to prepare your input sequence)
    # sample_input_sequence = ... # Prepare your scaled input sequence
    # next_price = predict_next_price(loaded_model, loaded_scaler, sample_input_sequence)
    # print(f"Predicted next price: {next_price}")