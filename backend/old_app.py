# app.py - Flask application for stock prediction
import os
from database import get_latest_model, store_model
import asyncio
import pandas as pd
import numpy as np
import json
from database import get_latest_model, store_model, get_all_models
from datetime import datetime
import time
import matplotlib
matplotlib.use('Agg')  # Set the backend for matplotlib to avoid GUI issues
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify, send_file
import io
import base64
from werkzeug.utils import secure_filename
import nest_asyncio
import yfinance as yf  # New import for yfinance

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
DATA_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'h5','pkl'}

# Remove Alpha Vantage API key as it is no longer needed
# API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# Create necessary directories
for folder in [UPLOAD_FOLDER, MODEL_FOLDER, DATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER

nest_asyncio.apply()
loop = asyncio.get_event_loop()

def register_local_models():
    """
    On startup, scan backend/models/ for any .h5 or .pkl files
    and register them in MongoDB so they're known to the app.
    """
    folder_path = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(folder_path):
        print(f"Model folder not found at {folder_path}, skipping auto-register.")
        return
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.h5') or file_name.endswith('.pkl'):
            symbol = file_name.rsplit('.', 1)[0].upper()
            file_path = os.path.join(folder_path, file_name)
            # Store model in DB using our single event loop
            loop.run_until_complete(store_model(symbol, file_path))
            print(f"Registered local model for {symbol} -> {file_path}")

@app.route('/available-models')
def available_models():
    """Return a list of available models from the database."""
    models = loop.run_until_complete(get_all_models())
    return jsonify({'models': models})

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# New function to fetch stock data using yfinance
def get_yfinance_data(symbol, start_date="2022-03-19", end_date=None):
    """
    Fetch stock data using yfinance.
    Downloads data from start_date to today (or specified end_date),
    resets the index, renames the columns to lower-case,
    and sorts the data by date.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    
    # Reset index to bring the Date into a column
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['Date'])
    
    # Rename columns for consistency
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    df = df.sort_values('date')
    return df

def prepare_latest_data(df, sequence_length=60):
    """Prepare the latest data sequence for prediction"""
    features = ['open', 'high', 'low', 'close', 'volume']
    data = df[features].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Get the latest sequence
    if len(scaled_data) < sequence_length:
        raise ValueError(f"Not enough data. Need at least {sequence_length} days.")
        
    latest_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, len(features))
    
    return latest_sequence, scaler

def predict_future_prices(model, latest_sequence, scaler, days_ahead=10, features_count=5):
    """Predict future prices for multiple days ahead using a Keras model"""
    predictions = []
    current_sequence = latest_sequence.copy()
    
    for _ in range(days_ahead):
        # Predict the next day
        next_pred = model.predict(current_sequence, verbose=0)[0, 0]
        
        # Store the prediction
        dummy = np.zeros((1, features_count))
        dummy[0, 3] = next_pred  # 3 is the index for 'close'
        next_price = scaler.inverse_transform(dummy)[0, 3]
        predictions.append(next_price)
        
        # Update the sequence for the next prediction
        new_pred = np.zeros((1, 1, features_count))
        new_pred[0, 0, :] = current_sequence[0, -1, :].copy()
        new_pred[0, 0, 3] = next_pred  # Update only the close price
        
        # Remove the first day and add the new prediction
        current_sequence = np.append(current_sequence[:, 1:, :], new_pred, axis=1)
    
    return predictions

def predict_future_prices_non_keras(model, latest_sequence, scaler, days_ahead=10, features_count=5):
    """
    Predict future prices for non-Keras models (like sklearn or XGBoost)
    using recursive prediction.
    """
    predictions = []
    current_sequence = latest_sequence.copy()

    for _ in range(days_ahead):
        # Flatten the current sequence to feed into the model
        features = current_sequence.reshape(current_sequence.shape[0], -1)

        # Handle models with a specific number of expected features
        if hasattr(model, "n_features_in_"):
            expected_features = model.n_features_in_
            features = features[:, :expected_features]

        # Predict scaled close price
        next_scaled_close = model.predict(features)[0]

        # Convert to actual price using inverse transform
        dummy = np.zeros((1, features_count))
        dummy[0, 3] = next_scaled_close  # Only the 'close' index is set
        next_price = scaler.inverse_transform(dummy)[0, 3]
        predictions.append(next_price)

        # Update the sequence by shifting left and appending a new day
        new_day = current_sequence[0, -1, :].copy()
        new_day[3] = next_scaled_close  # Update only the 'close' value

        new_day = new_day.reshape(1, 1, features_count)
        current_sequence = np.append(current_sequence[:, 1:, :], new_day, axis=1)

    return predictions

def make_trading_decisions(predictions, current_price, threshold=0.01):
    """Make trading decisions based on predictions"""
    results = []
    prev_price = current_price
    
    for i, next_price in enumerate(predictions):
        expected_return = (next_price - prev_price) / prev_price
        
        if expected_return > threshold:
            decision = 'BUY'
        elif expected_return < -threshold:
            decision = 'SELL'
        else:
            decision = 'HOLD'
        
        results.append({
            'day': i + 1,
            'predicted_price': float(next_price),
            'expected_return': float(expected_return * 100),  # Convert to percentage
            'decision': decision
        })
        
        prev_price = next_price
    
    return results

def create_prediction_plot(historical_data, predictions, symbol):
    """Create a plot of historical and predicted prices"""
    # Get the last date from historical data
    last_date = historical_data['date'].iloc[-1]
    
    # Create date range for predictions
    future_dates = [last_date + BDay(i+1) for i in range(len(predictions))]
    predicted_prices = [p['predicted_price'] for p in predictions]
    
    # Get last 30 days of historical data for plotting
    last_30_days = historical_data.iloc[-30:]
    
    plt.figure(figsize=(10, 6))
    plt.plot(last_30_days['date'], last_30_days['close'], color='blue', label='Historical Prices')
    plt.plot(future_dates, predicted_prices, color='red', label='Predicted Prices', linestyle='--')
    plt.axvline(x=last_date, color='green', linestyle='-', label='Current Date')
    plt.title(f'{symbol} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Convert to base64 for embedding in HTML
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_data

# Flask Routes
@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle stock prediction requests using any selected model for AAPL stock."""
    try:
        # Get the selected model name
        model_name = request.form.get('symbol', '').upper()
        if not model_name:
            return jsonify({'error': 'No model selected'}), 400

        # Get model path from DB
        model_path = loop.run_until_complete(get_latest_model(model_name))
        if not model_path or not os.path.exists(model_path):
            return jsonify({'error': f'No trained model found for {model_name}. Please upload a model first.'}), 404

        # Always use AAPL stock data
        real_symbol = "AAPL"

        # Fetch AAPL stock data using yfinance
        try:
            stock_data = get_yfinance_data(real_symbol)
            data_path = os.path.join(app.config['DATA_FOLDER'], f'{real_symbol}_data.csv')
            stock_data.to_csv(data_path, index=False)
        except Exception as e:
            return jsonify({'error': f'Error fetching data for {real_symbol}: {str(e)}'}), 500

        if model_path.endswith('.h5'):
            model = load_model(model_path)
            is_keras_model = True
        elif model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            is_keras_model = False
        else:
            return jsonify({'error': 'Unsupported model format. Please upload a .h5 or .pkl model.'}), 400

        sequence, scaler = prepare_latest_data(stock_data)
        current_price = stock_data['close'].iloc[-1]
        last_date = stock_data['date'].iloc[-1]

        days_ahead_param = request.form.get('days_ahead', '10')
        try:
            days_ahead = int(days_ahead_param)
            if days_ahead <= 0:
                raise ValueError("days_ahead must be positive")
        except ValueError:
            return jsonify({'error': 'Invalid days_ahead. Must be a positive integer.'}), 400

        if is_keras_model:
            future_prices = predict_future_prices(model, sequence, scaler, days_ahead)
        else:
            future_prices = predict_future_prices_non_keras(model, sequence, scaler, days_ahead)

        # Generate trading decisions
        decisions = make_trading_decisions(future_prices, current_price)

        # Create the plot
        plot_data = create_prediction_plot(stock_data, decisions, model_name)

        return jsonify({
            'model_used': model_name,
            'symbol': real_symbol,
            'current_date': last_date.strftime('%Y-%m-%d'),
            'current_price': float(current_price),
            'predictions': decisions,
            'plot': plot_data
        })

    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_model():
    """Handle model upload."""
    if 'model' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['model']
    symbol = request.form.get('symbol', '').upper()

    if not file.filename:
        return jsonify({'error': 'No selected file'}), 400
    if not symbol:
        return jsonify({'error': 'No stock symbol provided'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only .h5 or .pkl allowed.'}), 400

    # Save the file
    extension = file.filename.rsplit('.', 1)[1].lower()
    filename = f'{symbol}.{extension}'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Store in DB with our single global loop
    loop.run_until_complete(store_model(symbol, file_path))

    return jsonify({'success': f'Model uploaded and stored for {symbol}'}), 200

# Run the application
if __name__ == '__main__':
    register_local_models()
    app.run(debug=True, host='0.0.0.0', port=4080)
