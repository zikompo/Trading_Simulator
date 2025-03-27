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
from qlearningagent import QLearningAgent

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
    Predict future prices for the next days using the Keras model.
    The function iteratively predicts one day ahead and then updates the input sequence.
    """
    predictions = []
    current_sequence = latest_sequence.copy()
    
    for _ in range(days_ahead):
        next_pred = float(model.predict(current_sequence, verbose=0)[0, 0])
        # Inverse transform the predicted scaled "Close" price.
        dummy = np.zeros((1, features_count))
        dummy[0, 3] = next_pred  # index 3 corresponds to 'Close'
        next_price = float(scaler.inverse_transform(dummy)[0, 3])
        predictions.append(next_price)
        
        # Update sequence: remove the first day and append new prediction.
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
      - 'symbol': the model symbol (for lookup)
      - 'stock': the stock ticker for which to fetch data
      - 'days_ahead': number of days to predict (max 5)
    """
    try:
        model_symbol = str(request.form.get('symbol', '')).strip().upper()
        stock_field = str(request.form.get('stock', '')).strip().upper()
        days_ahead = int(request.form.get('days_ahead', 1))
        days_ahead = days_ahead
        
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
      - 'days_ahead': number of days to predict (max 5)
    """
    try:
        model_symbol = str(request.form.get('symbol', '')).strip().upper()
        stock_field = str(request.form.get('stock', '')).strip().upper()
        days_ahead = int(request.form.get('days_ahead', 1))
        days_ahead = days_ahead
        
        if model_symbol == "":
            return jsonify({'error': 'No model symbol provided'}), 400
        
        stock_symbol = stock_field if stock_field and stock_field != model_symbol else "AAPL"
        stock_data = get_yfinance_data(stock_symbol)
        if stock_data.empty:
            return jsonify({'error': f'No data found for stock symbol {stock_symbol}.'}), 404
        
        # Use only High, Open, and Low as features.
        features = ['High', 'Open', 'Low']
        X_input = stock_data[features].iloc[-1].values.reshape(1, -1)
        
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'kernel_ridge_model_approx.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': f'No trained Kernel Ridge model found at {model_path}. Please train and save the model first.'}), 404
        
        import pickle
        with open(model_path, "rb") as f:
            kr_pipeline = pickle.load(f)
        
        # Iterative prediction for days_ahead.
        predictions = []
        current_features = X_input.copy()  # shape (1, 3)
        for _ in range(days_ahead):
            pred = kr_pipeline.predict(current_features)[0]
            predictions.append(pred)
            # Naively update current_features: assume next day's low equals predicted closing price.
            # (This heuristic may be adjusted for your use-case.)
            current_features = np.array([[current_features[0,0], current_features[0,1], pred]])
        
        current_price = stock_data['Close'].iloc[-1]
        expected_returns = [(p - current_price)/current_price * 100 for p in predictions]
        decisions = make_trading_decisions(predictions, current_price)
        plot_img = create_prediction_plot(stock_data, decisions, stock_symbol)
        last_date = stock_data['Date'].iloc[-1]
        
        return jsonify({
            'model_used': 'Kernel Ridge Regression (Approx)',
            'symbol': stock_symbol,
            'current_date': last_date.strftime('%Y-%m-%d'),
            'current_price': float(current_price),
            'predictions': decisions,
            'plot': plot_img
        })
        
    except Exception as e:
        return jsonify({'error': f'Error making Kernel Ridge prediction: {str(e)}'}), 500
@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    """
    KNN prediction route.
    Expects form fields:
      - 'symbol': the model symbol (for lookup)
      - 'stock': the stock ticker for which to fetch data
      - 'days_ahead': number of days to predict (max 5)
    """
    try:
        model_symbol = str(request.form.get('symbol', '')).strip().upper()
        stock_field = str(request.form.get('stock', '')).strip().upper()
        days_ahead = int(request.form.get('days_ahead', 1))
        days_ahead = days_ahead
        
        if model_symbol == "":
            return jsonify({'error': 'No model symbol provided'}), 400

        stock_symbol = stock_field if stock_field and stock_field != model_symbol else "AAPL"
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

        # Use these features for KNN.
        features = ['High', 'Low', 'Open', 'Volume']
        # Start with the latest available row.
        current_features = stock_data[features].iloc[-1:].values  # shape (1,4)

        predictions = []
        # Compute recent average ratios for high and low relative to the close.
        avg_high_ratio = (stock_data['High'] / stock_data['Close']).tail(10).mean()
        avg_low_ratio = (stock_data['Low'] / stock_data['Close']).tail(10).mean()
        # Use the most recent volume or average volume over recent days.
        avg_volume = stock_data['Volume'].tail(10).mean()

        for _ in range(days_ahead):
            scaled = knn_scaler.transform(current_features)
            pred = knn_model.predict(scaled)[0]
            predictions.append(float(pred))
            # Update current_features:
            # For simplicity, assume:
            #   - Next day's open equals the predicted close.
            #   - Next day's high and low are computed via average ratios.
            #   - Next day's volume is the average volume.
            new_high = pred * avg_high_ratio
            new_low = pred * avg_low_ratio
            new_open = pred
            current_features = np.array([[new_high, new_low, new_open, avg_volume]])
        
        current_price = stock_data['Close'].iloc[-1]
        decisions = make_trading_decisions(predictions, current_price)
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


@app.route('/predict_linear', methods=['POST'])
def predict_linear():
    """
    Linear Regression prediction route.
    Expects form fields:
      - 'symbol': the model symbol (for lookup)
      - 'stock': the stock ticker for which to fetch data
      - 'days_ahead': number of days to predict (max 5)
    """
    try:
        model_symbol = str(request.form.get('symbol', '')).strip().upper()
        stock_field = str(request.form.get('stock', '')).strip().upper()
        days_ahead = int(request.form.get('days_ahead', 1))
        days_ahead = days_ahead
        
        if model_symbol == "":
            return jsonify({'error': 'No model symbol provided'}), 400
        
        stock_symbol = stock_field if stock_field and stock_field != model_symbol else "AAPL"
        stock_data = get_yfinance_data(stock_symbol)
        if stock_data.empty:
            return jsonify({'error': f'No data found for stock symbol {stock_symbol}.'}), 404

        # For linear regression we use these three features.
        features = ['High', 'Open', 'Low']
        # Start with the latest available row.
        current_features = stock_data[features].iloc[-1].values.reshape(1, -1)
        
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'linear_regression_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': f'No trained linear regression model found at {model_path}. Please train and save the model first.'}), 404
        
        import pickle
        with open(model_path, "rb") as f:
            lr_model = pickle.load(f)
        
        predictions = []
        # Compute recent average ratios for updating features.
        # We assume: next_high = predicted_close * avg_high_ratio and next_low = predicted_close * avg_low_ratio.
        avg_high_ratio = (stock_data['High'] / stock_data['Close']).tail(10).mean()
        avg_low_ratio = (stock_data['Low'] / stock_data['Close']).tail(10).mean()
        
        # Iteratively predict and update features.
        for _ in range(days_ahead):
            pred = lr_model.predict(current_features)[0]
            predictions.append(float(pred))
            # Update features: assume next day's open equals the predicted close.
            # And update high and low using recent average ratios.
            new_high = pred * avg_high_ratio
            new_open = pred
            new_low = pred * avg_low_ratio
            current_features = np.array([[new_high, new_open, new_low]])
        
        current_price = stock_data['Close'].iloc[-1]
        decisions = make_trading_decisions(predictions, current_price)
        plot_img = create_prediction_plot(stock_data, decisions, stock_symbol)
        last_date = stock_data['Date'].iloc[-1]
        
        return jsonify({
            'model_used': 'Linear Regression',
            'symbol': stock_symbol,
            'current_date': last_date.strftime('%Y-%m-%d'),
            'current_price': float(current_price),
            'predictions': decisions,
            'plot': plot_img
        })
        
    except Exception as e:
        return jsonify({'error': f'Error making linear regression prediction: {str(e)}'}), 500


def create_q_plot(q_values, actions):
    import matplotlib.pyplot as plt
    import io, base64
    plt.figure(figsize=(6,4))
    plt.bar(range(len(q_values)), q_values, tick_label=[actions[i] for i in range(len(actions))], color='skyblue')
    plt.title("Q-values for Current State")
    plt.xlabel("Actions")
    plt.ylabel("Q-value")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return plot_base64


@app.route('/predict_qlearning', methods=['POST'])
def predict_qlearning():
    """
    Q-learning prediction route.
    Expects form fields:
      - 'symbol': the model symbol (used to locate the model file)
      - 'stock': the stock ticker for which to fetch data
      - 'holding': (optional) 0 if not holding, 1 if holding (defaults to 0)
    
    This endpoint computes the current trend based on the last two closing prices,
    forms the state as (trend, holding), loads the saved Q-learning agent, retrieves
    the Q-values for that state, and returns only the recommended action.
    """
    try:
        # Retrieve input parameters.
        model_symbol = str(request.form.get('symbol', '')).strip().upper()
        stock_field = str(request.form.get('stock', '')).strip().upper()
        holding = request.form.get('holding', 0)
        
        if model_symbol == "":
            return jsonify({'error': 'No model symbol provided'}), 400
        
        # Default to AAPL if no valid stock ticker is provided.
        stock_symbol = stock_field if stock_field and stock_field != model_symbol else "AAPL"
        holding = int(holding)
        
        # Fetch stock data using yfinance.
        stock_data = get_yfinance_data(stock_symbol)
        if stock_data.empty:
            return jsonify({'error': f'No data found for stock symbol {stock_symbol}.'}), 404
        
        # Compute current trend based on the last two closing prices.
        if len(stock_data) < 2:
            return jsonify({'error': 'Not enough data to compute trend.'}), 400
        current_close = stock_data['Close'].iloc[-1]
        prev_close = stock_data['Close'].iloc[-2]
        ret = (current_close - prev_close) / prev_close
        threshold = 0.001
        if ret > threshold:
            trend = 'up'
        elif ret < -threshold:
            trend = 'down'
        else:
            trend = 'stable'
        
        # Form the state as a tuple (trend, holding)
        state = (trend, holding)
        
        # Load the saved Q-learning agent.
        import pickle
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'q_learning_agent.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': f'No Q-learning agent found at {model_path}. Please train and export the model first.'}), 404
        with open(model_path, 'rb') as f:
            agent = pickle.load(f)
        
        # Retrieve Q-values and determine the recommended action.
        q_values = agent.get_Q(state)
        action = agent.choose_action(state)
        action_mapping = {0: "Buy", 1: "Sell", 2: "Hold"}
        recommended_action = action_mapping.get(action, str(action))
        
        # Get current date and price.
        last_date = stock_data['Date'].iloc[-1]
        
        return jsonify({
            'model_used': 'Q-learning',
            'symbol': stock_symbol,
            'current_date': last_date.strftime('%Y-%m-%d'),
            'current_price': float(current_close),
            'predicted_action': recommended_action,
            'q_values': q_values.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': f'Error making Q-learning prediction: {str(e)}'}), 500


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

def create_q_plot(q_values, actions):
    import matplotlib.pyplot as plt
    import io, base64
    plt.figure(figsize=(6,4))
    plt.bar(range(len(q_values)), q_values, tick_label=[actions[i] for i in range(len(actions))], color='skyblue')
    plt.title("Q-values for Current State")
    plt.xlabel("Actions")
    plt.ylabel("Q-value")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return plot_base64


@app.route('/predict_rnn', methods=['POST'])
def predict_rnn():
    """
    RNN prediction route.
    Expects form fields:
      - 'symbol': the model symbol (for lookup)
      - 'stock': the stock ticker for which to fetch data (defaults to AAPL)
      - 'days_ahead': number of days to predict (max 5)
    
    This endpoint predicts future closing prices using the pre-trained
    General RNN model and returns the predicted prices, expected returns,
    trading decisions, and a plot of historical prices with the predictions.
    """
    try:
        import os
        import numpy as np
        import joblib
        from tensorflow.keras.models import load_model

        # Retrieve input parameters.
        model_symbol = str(request.form.get('symbol', '')).strip().upper()
        stock_field = str(request.form.get('stock', '')).strip().upper()
        days_ahead = int(request.form.get('days_ahead', 1))
        days_ahead = days_ahead
        
        if model_symbol == "":
            return jsonify({'error': 'No model symbol provided'}), 400
        
        # Default to AAPL if no valid stock ticker is provided.
        stock_symbol = stock_field if stock_field and stock_field != model_symbol else "AAPL"
        
        # Fetch stock data using yfinance helper function.
        stock_data = get_yfinance_data(stock_symbol)
        if stock_data.empty:
            return jsonify({'error': f'No data found for stock symbol {stock_symbol}.'}), 404
        
        # Define sequence length and features.
        sequence_length = 60
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        if len(stock_data) < sequence_length:
            return jsonify({'error': f'Not enough data. Need at least {sequence_length} days.'}), 400

        # Load the pre-trained RNN model.
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'general_rnn_model.h5')
        if not os.path.exists(model_path):
            return jsonify({'error': f'No trained RNN model found at {model_path}. Please train and save the model first.'}), 404
        rnn_model = load_model(model_path)
        
        # Load the scaler used during training.
        scaler_path = os.path.join(app.config['MODEL_FOLDER'], 'rnn_scaler.pkl')
        if not os.path.exists(scaler_path):
            return jsonify({'error': f'No scaler found for the RNN model at {scaler_path}. Please save the scaler along with the model.'}), 404
        scaler = joblib.load(scaler_path)
        
        # Prepare the latest sequence.
        latest_sequence = stock_data[features].values[-sequence_length:]
        latest_sequence_scaled = scaler.transform(latest_sequence).reshape(1, sequence_length, len(features))
        
        # Predict for the requested number of days ahead.
        future_prices = predict_future_prices(rnn_model, latest_sequence_scaled, scaler, days_ahead, features_count=len(features))
        
        # Get the current price (latest available closing price).
        current_price = stock_data['Close'].iloc[-1]
        
        # Compute trading decisions for each predicted day.
        decisions = make_trading_decisions(future_prices, current_price)
        
        # Generate a plot using your helper function.
        plot_img = create_prediction_plot(stock_data, decisions, stock_symbol)
        last_date = stock_data['Date'].iloc[-1]
        
        return jsonify({
            'model_used': 'General RNN',
            'symbol': stock_symbol,
            'current_date': last_date.strftime('%Y-%m-%d'),
            'current_price': float(current_price),
            'predictions': decisions,
            'plot': plot_img
        })
        
    except Exception as e:
        return jsonify({'error': f'Error making RNN prediction: {str(e)}'}), 500







if __name__ == '__main__':
    asyncio.run(sync_all_models())
    app.run(debug=True, host='0.0.0.0', port=4080)
