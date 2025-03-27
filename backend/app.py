import os
import io
import time
import base64
import asyncio
import nest_asyncio
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from database import store_model
from database import get_all_models  # make sure this is imported

# Apply nest_asyncio to support async operations if needed
nest_asyncio.apply()
loop = asyncio.get_event_loop()

# ---------------------------
# Flask App Configuration
# ---------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
DATA_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'h5', 'pkl'}

for folder in [UPLOAD_FOLDER, MODEL_FOLDER, DATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER

MODEL_FOLDER = "models"  # or just "models" if you're already inside backend/

async def sync_all_models():
    for filename in os.listdir(MODEL_FOLDER):
        if filename.endswith('.h5') or filename.endswith('.pkl'):
            symbol = filename.rsplit('.', 1)[0].upper()
            file_path = os.path.join(MODEL_FOLDER, filename)
            await store_model(symbol, file_path)

# ---------------------------
# Helper Functions
# ---------------------------
def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_local_model_path(symbol):
    """
    Search the local model folder for a file named <symbol>.h5 or <symbol>.pkl.
    Returns the path if found, else None.
    """
    folder = app.config['MODEL_FOLDER']
    for ext in ALLOWED_EXTENSIONS:
        model_file = os.path.join(folder, f"{symbol}.{ext}")
        if os.path.exists(model_file):
            return model_file
    return None

def get_yfinance_data(symbol, start_date="2022-03-19", end_date=None):
    """
    Fetch stock data using yfinance.
    Downloads data from start_date to today (or specified end_date),
    resets the index, and sorts the data by date.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
    print("Before flattening:", df.columns)
    if isinstance(df.columns, pd.MultiIndex):
        # Use the 'Price' level (or level 0) instead of the last level
        df.columns = df.columns.get_level_values('Price')
        df.columns.name = None
    print("After flattening:", df.columns)

    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Symbol'] = symbol
    df = df.sort_values('Date')
    return df

def prepare_latest_data(df, sequence_length=60):
    """Prepare the latest data sequence for prediction."""
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)
    
    if len(data) < sequence_length:
        raise ValueError(f"Not enough data. Need at least {sequence_length} days.")
        
    latest_sequence = data[-sequence_length:]
    latest_sequence_scaled = scaler.transform(latest_sequence).reshape(1, sequence_length, len(features))
    return latest_sequence_scaled, scaler

def predict_future_prices(model, latest_sequence, scaler, days_ahead=1, features_count=5):
    """
    Predict future prices for the next day using the Keras LSTM model.
    Returns a list with a single predicted price.
    """
    predictions = []
    current_sequence = latest_sequence.copy()
    
    # Since we always predict only the next day, run the loop once.
    for _ in range(days_ahead):
        next_pred = float(model.predict(current_sequence, verbose=0)[0, 0])
        dummy = np.zeros((1, features_count))
        dummy[0, 3] = next_pred  # index 3 corresponds to 'Close'
        next_price = float(scaler.inverse_transform(dummy)[0, 3])
        predictions.append(next_price)
        
        # Update sequence: remove the first day and append new prediction as the last day.
        new_pred = np.zeros((1, 1, features_count))
        new_pred[0, 0, :] = current_sequence[0, -1, :].copy()
        new_pred[0, 0, 3] = next_pred  # update close price with prediction
        current_sequence = np.append(current_sequence[:, 1:, :], new_pred, axis=1)
    
    return predictions



def make_trading_decisions(predictions, current_price, threshold=0.01):
    """Generate trading signals based on predicted prices."""
    results = []
    prev_price = current_price
    for i, next_price in enumerate(predictions):
        expected_return = (next_price - prev_price) / prev_price
        decision = 'BUY' if expected_return > threshold else 'SELL' if expected_return < -threshold else 'HOLD'
        results.append({
            'day': i + 1,
            'predicted_price': float(next_price),
            'expected_return': float(expected_return * 100),
            'decision': decision
        })
        prev_price = next_price
    return results

def create_prediction_plot(historical_data, decisions, symbol):
    """
    Create a plot showing historical prices (last 30 days) and the predicted future price.
    Returns the plot as a base64 encoded string.
    """
    last_date = historical_data['Date'].iloc[-1]
    from pandas.tseries.offsets import BDay
    future_dates = [last_date + BDay(i+1) for i in range(len(decisions))]
    predicted_prices = [d['predicted_price'] for d in decisions]
    historical_plot = historical_data.iloc[-30:]
    
    plt.figure(figsize=(10, 6))
    plt.plot(historical_plot['Date'], historical_plot['Close'], color='blue', marker='o', label='Historical Prices')
    plt.plot(future_dates, predicted_prices, color='red', linestyle='--', marker='o', label='Predicted Price')
    plt.axvline(x=last_date, color='green', linestyle='-', label='Current Date')
    plt.title(f'{symbol} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_base64

# ---------------------------
# Flask Endpoints
# ---------------------------
@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an index.html in your templates folder

@app.route('/available-models')
def available_models():
    """Return a list of available model symbols from the database."""
    models = loop.run_until_complete(get_all_models())
    return jsonify({'models': models})

@app.route('/upload', methods=['POST'])
def upload_model():
    """Handle model upload and save it locally."""
    if 'model' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['model']
    raw_symbol = request.form.get('symbol', '')
    model_symbol = str(raw_symbol).split()[0].strip().upper()
    
    if not file.filename:
        return jsonify({'error': 'No selected file'}), 400
    if model_symbol == "":
        return jsonify({'error': 'No stock symbol provided'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only .h5 or .pkl allowed.'}), 400

    extension = file.filename.rsplit('.', 1)[1].lower()
    filename = f'{model_symbol}.{extension}'
    model_path = os.path.join(app.config['MODEL_FOLDER'], filename)
    file.save(model_path)

    return jsonify({'success': f'Model uploaded and stored for {model_symbol}'}), 200

@app.route('/predict_lstm', methods=['POST'])
def predict_lstm():
    """
    LSTM prediction route.
    Expects form fields:
      - 'symbol': the model symbol (used to locate the model file)
      - 'stock': the stock ticker for which to fetch data
    This endpoint always predicts the next day's price.
    """
    try:
        model_symbol = str(request.form.get('symbol', '')).strip().upper()
        stock_field = str(request.form.get('stock', '')).strip().upper()
        
        # Always predict next day
        days_ahead = 1
        
        if model_symbol == "":
            return jsonify({'error': 'No model symbol provided'}), 400
        
        stock_symbol = stock_field if stock_field and stock_field != model_symbol else "AAPL"

        model_path = get_local_model_path(model_symbol)
        if model_path is None:
            return jsonify({'error': f'No trained model found for {model_symbol}. Please upload a model first.'}), 404
        if not model_path.endswith('.h5'):
            return jsonify({'error': 'This endpoint only supports LSTM models in .h5 format.'}), 400

        stock_data = get_yfinance_data(stock_symbol)
        data_path = os.path.join(app.config['DATA_FOLDER'], f'{stock_symbol}_data.csv')
        stock_data.to_csv(data_path, index=False)
        
        model = load_model(model_path)
        latest_sequence, scaler = prepare_latest_data(stock_data)
        current_price = stock_data['Close'].iloc[-1]
        last_date = stock_data['Date'].iloc[-1]
        
        future_prices = predict_future_prices(model, latest_sequence, scaler, days_ahead)
        decisions = make_trading_decisions(future_prices, current_price)
        plot_img = create_prediction_plot(stock_data, decisions, stock_symbol)
        
        return jsonify({
            'model_used': model_symbol,
            'symbol': stock_symbol,
            'current_date': last_date.strftime('%Y-%m-%d'),
            'current_price': float(current_price),
            'predictions': decisions,
            'plot': plot_img
        })
        
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500

@app.route('/predict_kernel', methods=['POST'])
def predict_kernel():
    """
    Kernel Ridge Regression prediction route.
    Expects form fields:
      - 'symbol': the model symbol (for lookup)
      - 'stock': the stock ticker for which to fetch data
    This endpoint predicts the next day's closing price using the improved
    Kernel Ridge Regression model (with Nystroem kernel approximation) trained on
    High, Open, and Low prices.
    """
    try:
        # Retrieve input parameters.
        model_symbol = str(request.form.get('symbol', '')).strip().upper()
        stock_field = str(request.form.get('stock', '')).strip().upper()
        
        if model_symbol == "":
            return jsonify({'error': 'No model symbol provided'}), 400
        
        # Default to AAPL if no valid stock ticker is provided.
        stock_symbol = stock_field if stock_field and stock_field != model_symbol else "AAPL"
        
        # Fetch stock data using yfinance.
        stock_data = get_yfinance_data(stock_symbol)
        if stock_data.empty:
            return jsonify({'error': f'No data found for stock symbol {stock_symbol}.'}), 404
        
        # For this improved model, we use only the High, Open, and Low prices.
        features = ['High', 'Open', 'Low']
        # Use the latest available row as the input features.
        X_input = stock_data[features].iloc[-1].values.reshape(1, -1)
        
        # Load the saved pipeline model.
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'kernel_ridge_model_approx.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': f'No trained Kernel Ridge model found at {model_path}. Please train and save the model first.'}), 404
        
        import pickle
        with open(model_path, "rb") as f:
            kr_pipeline = pickle.load(f)
        
        # Predict the next day's closing price.
        predicted_price = kr_pipeline.predict(X_input)[0]
        
        # Get the current closing price.
        current_price = stock_data['Close'].iloc[-1]
        
        # Compute expected return (percentage change).
        expected_return = (predicted_price - current_price) / current_price * 100
        
        # Generate a trading decision.
        decision = "BUY" if predicted_price > current_price else "SELL" if predicted_price < current_price else "HOLD"
        
        # Prepare the predictions list.
        predictions = [{
            'day': 1,
            'predicted_price': float(predicted_price),
            'expected_return': float(expected_return),
            'decision': decision
        }]
        
        # Generate a prediction plot using your helper function.
        plot_img = create_prediction_plot(stock_data, predictions, stock_symbol)
        last_date = stock_data['Date'].iloc[-1]
        
        return jsonify({
            'model_used': 'Kernel Ridge Regression (Approx)',
            'symbol': stock_symbol,
            'current_date': last_date.strftime('%Y-%m-%d'),
            'current_price': float(current_price),
            'predictions': predictions,
            'plot': plot_img
        })
        
    except Exception as e:
        return jsonify({'error': f'Error making Kernel Ridge prediction: {str(e)}'}), 500


@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    """
    KNN prediction route.
    Expects form fields:
      - 'symbol': the model symbol (used to locate the model file)
      - 'stock': the stock ticker for which to fetch data
    This endpoint always predicts the next day's price.
    """
    try:
        model_symbol = str(request.form.get('symbol', '')).strip().upper()
        stock_field = str(request.form.get('stock', '')).strip().upper()

        if model_symbol == "":
            return jsonify({'error': 'No model symbol provided'}), 400

        stock_symbol = stock_field if stock_field and stock_field != model_symbol else "AAPL"

        # Locate and load the KNN model and its scaler.
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'knn_model.pkl')
        scaler_path = os.path.join(app.config['MODEL_FOLDER'], 'scaler.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': f'No trained KNN model found at {model_path}. Please train the model first.'}), 404
        if not os.path.exists(scaler_path):
            return jsonify({'error': f'No scaler found for the KNN model at {scaler_path}. Please save the scaler along with the model.'}), 404

        import pickle
        with open(model_path, "rb") as f:
            knn_model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            knn_scaler = pickle.load(f)

        stock_data = get_yfinance_data(stock_symbol)
        if stock_data.empty:
            return jsonify({'error': f'No data found for stock symbol {stock_symbol}.'}), 404

        features = ['High', 'Low', 'Open', 'Volume']
        latest_features = stock_data[features].iloc[-1:].values

        latest_features_scaled = knn_scaler.transform(latest_features)
        next_close_pred = knn_model.predict(latest_features_scaled)[0]
        predicted_prices = [float(next_close_pred)]

        current_price = stock_data['Close'].iloc[-1]
        decisions = make_trading_decisions(predicted_prices, current_price)
        plot_img = create_prediction_plot(stock_data, decisions, stock_symbol)

        last_date = stock_data['Date'].iloc[-1]
        if not isinstance(last_date, str):
            last_date = last_date.strftime('%Y-%m-%d')

        return jsonify({
            'model_used': 'K-Nearest Neighbors Regression',
            'symbol': stock_symbol,
            'current_date': last_date,
            'current_price': float(current_price),
            'predictions': decisions,
            'plot': plot_img
        })

    except Exception as e:
        return jsonify({'error': f'Error making KNN prediction: {str(e)}'}), 500

def compute_linear_features(df):
    """
    Compute features for linear regression prediction based on the training pipeline.
    Expects df to have columns: Date, Symbol, Open, High, Low, Close, Volume.
    Returns the dataframe with computed features and the list of feature columns.
    """
    df = df.copy()
    df = df.sort_values(by='Date')
    df['returns'] = df['Close'].pct_change()
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_open_ratio'] = df['Close'] / df['Open']
    df['volume_change'] = df['Volume'].pct_change()
    
    for lag in range(1, 6):
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        df[f'hl_ratio_lag_{lag}'] = df['high_low_ratio'].shift(lag)
        df[f'co_ratio_lag_{lag}'] = df['close_open_ratio'].shift(lag)
        df[f'volume_change_lag_{lag}'] = df['volume_change'].shift(lag)
    
    # Define target as the next day's return direction (1 if >0, else 0)
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Feature columns exclude Date, Symbol, and target.
    feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol', 'target']]
    return df, feature_cols

@app.route('/predict_linear', methods=['POST'])
def predict_linear():
    """
    Linear Regression prediction route.
    Expects form fields:
      - 'symbol': the model symbol (for lookup)
      - 'stock': the stock ticker for which to fetch data
    This endpoint predicts the next day's closing price using the Linear Regression model
    trained on High, Open, and Low prices.
    """
    try:
        # Retrieve input parameters.
        model_symbol = str(request.form.get('symbol', '')).strip().upper()
        stock_field = str(request.form.get('stock', '')).strip().upper()
        
        if model_symbol == "":
            return jsonify({'error': 'No model symbol provided'}), 400
        
        # Default to AAPL if no valid stock ticker is provided.
        stock_symbol = stock_field if stock_field and stock_field != model_symbol else "AAPL"
        
        # Fetch stock data using yfinance.
        stock_data = get_yfinance_data(stock_symbol)
        if stock_data.empty:
            return jsonify({'error': f'No data found for stock symbol {stock_symbol}.'}), 404
        
        # For this improved model, we use only the High, Open, and Low prices.
        features = ['High', 'Open', 'Low']
        # Use the last available row as the input features.
        X_input = stock_data[features].iloc[-1].values.reshape(1, -1)
        
        # Load the saved linear regression model.
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'linear_regression_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': f'No trained linear regression model found at {model_path}. Please train and save the model first.'}), 404
        
        import pickle
        with open(model_path, "rb") as f:
            lr_model = pickle.load(f)
        
        # Predict the next day's closing price.
        predicted_price = lr_model.predict(X_input)[0]
        
        # Get the current closing price (latest available).
        current_price = stock_data['Close'].iloc[-1]
        
        # Compute expected return in percentage.
        expected_return = (predicted_price - current_price) / current_price * 100
        
        # Generate trading decision based on the price movement.
        decision = "BUY" if predicted_price > current_price else "SELL" if predicted_price < current_price else "HOLD"
        
        # Prepare predictions in the same format as other endpoints.
        predictions = [{
            'day': 1,
            'predicted_price': float(predicted_price),
            'expected_return': float(expected_return),
            'decision': decision
        }]
        
        # Generate a prediction plot.
        plot_img = create_prediction_plot(stock_data, predictions, stock_symbol)
        last_date = stock_data['Date'].iloc[-1]
        
        return jsonify({
            'model_used': 'Linear Regression',
            'symbol': stock_symbol,
            'current_date': last_date.strftime('%Y-%m-%d'),
            'current_price': float(current_price),
            'predictions': predictions,
            'plot': plot_img
        })
        
    except Exception as e:
        return jsonify({'error': f'Error making linear regression prediction: {str(e)}'}), 500






if __name__ == '__main__':
    asyncio.run(sync_all_models())
    app.run(debug=True, host='0.0.0.0', port=4080)
