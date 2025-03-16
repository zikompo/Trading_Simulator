# app.py - Flask application for stock prediction
import os
import requests
import pandas as pd
import numpy as np
import json
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

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
DATA_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'h5'}
API_KEY = 'TKZMZK2F3VMKJ58C'  # Replace with your Alpha Vantage API key

# Create necessary directories
for folder in [UPLOAD_FOLDER, MODEL_FOLDER, DATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER

# Helper Functions
def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_alpha_vantage_data(symbol, api_key, function='TIME_SERIES_DAILY', output_size='compact'):
    """Fetch stock data from Alpha Vantage API"""
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': function,
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': output_size
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    # Check for error messages
    if "Error Message" in data:
        raise ValueError(f"API Error: {data['Error Message']}")
    
    # Extract time series data
    if function == 'TIME_SERIES_DAILY':
        time_series_key = 'Time Series (Daily)'
    elif function == 'TIME_SERIES_WEEKLY':
        time_series_key = 'Weekly Time Series'
    else:
        time_series_key = 'Time Series (Daily)'  # Default
    
    # Convert to DataFrame
    df = pd.DataFrame(data[time_series_key]).T
    
    # Convert string values to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    # Rename columns
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Add date as a column
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index
    
    # Sort by date
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
    """Predict future prices for multiple days ahead"""
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
        new_pred[0, 0, :] = current_sequence[0, -1, :]  # Copy the last day's values
        new_pred[0, 0, 3] = next_pred  # Update only the close price
        
        # Remove the first day and add the new prediction
        current_sequence = np.append(current_sequence[:, 1:, :], new_pred, axis=1)
    
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
    """Handle stock prediction requests"""
    try:
        # Get stock symbol from request
        symbol = request.form.get('symbol')
        if not symbol:
            return jsonify({'error': 'No stock symbol provided'}), 400
            
        symbol = symbol.upper()
        
        # Check if we have a saved model for this symbol
        model_path = os.path.join(app.config['MODEL_FOLDER'], f'{symbol}_lstm_model.h5')
        print(model_path)
        if not os.path.exists(model_path):
            return jsonify({'error': f'No trained model found for {symbol}. Please upload a model first.'}), 404
        
        # Fetch latest data for this symbol
        try:
            stock_data = get_alpha_vantage_data(symbol, API_KEY)
            data_path = os.path.join(app.config['DATA_FOLDER'], f'{symbol}_data.csv')
            stock_data.to_csv(data_path)
        except Exception as e:
            return jsonify({'error': f'Error fetching data for {symbol}: {str(e)}'}), 500
            
        # Load the model
        model = load_model(model_path)
        
        # Prepare data for prediction
        try:
            sequence, scaler = prepare_latest_data(stock_data)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
            
        # Get current price and last date
        current_price = stock_data['close'].iloc[-1]
        last_date = stock_data['date'].iloc[-1]
        
        # Get the number of days to predict from the form data, default to 10 if not provided
        days_ahead_param = request.form.get('days_ahead', '10')
        try:
            days_ahead = int(days_ahead_param)
            if days_ahead <= 0:
                raise ValueError("days_ahead must be a positive integer")
        except ValueError:
            return jsonify({'error': 'Invalid value for days_ahead. Please provide a positive integer.'}), 400
        
        # Make predictions
        predictions = predict_future_prices(model, sequence, scaler, days_ahead)
        
        # Get trading decisions
        results = make_trading_decisions(predictions, current_price)
        
        # Create plot (uses the number of prediction days based on results length)
        plot_data = create_prediction_plot(stock_data, results, symbol)
        
        # Return results
        return jsonify({
            'symbol': symbol,
            'current_date': last_date.strftime('%Y-%m-%d'),
            'current_price': float(current_price),
            'predictions': results,
            'plot': plot_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_model():
    """Handle model upload"""
    if 'model' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['model']
    symbol = request.form.get('symbol')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not symbol:
        return jsonify({'error': 'No stock symbol provided'}), 400
        
    symbol = symbol.upper()
    
    if file and allowed_file(file.filename):
        filename = f'{symbol}_lstm_model.h5'
        file.save(os.path.join(app.config['MODEL_FOLDER'], filename))
        return jsonify({'success': f'Model uploaded for {symbol}'}), 200
    else:
        return jsonify({'error': 'Invalid file type. Only .h5 files are allowed.'}), 400
        
@app.route('/available-models')
def available_models():
    """Return a list of available models"""
    models = []
    for filename in os.listdir(app.config['MODEL_FOLDER']):
        if filename.endswith('_lstm_model.h5'):
            symbol = filename.split('_')[0]
            models.append(symbol)
    return jsonify({'models': models})

# Run the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4080)