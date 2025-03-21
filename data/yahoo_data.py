import yfinance as yf
import time
import pandas as pd

# Define the stock symbols
symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK.B', 'JPM', 'V', 
    'UNH', 'JNJ', 'XOM', 'WMT', 'PG', 'MA', 'CVX', 'LLY', 'HD', 'ABBV', 
    'KO', 'PEP', 'AVGO', 'MRK', 'COST', 'TMO', 'MCD', 'AMD', 'BAC', 'NFLX', 
    'DIS', 'ADBE', 'PFE', 'CRM', 'CSCO', 'ABT', 'ACN', 'NKE', 'LIN', 'DHR', 
    'INTC', 'TXN', 'CMCSA', 'HON', 'AMGN', 'NEE', 'COP', 'PM', 'IBM', 'QCOM', 
    'LOW', 'SBUX', 'RTX', 'SPGI', 'INTU', 'GS', 'NOW', 'CAT', 'DE', 'T', 
    'MDT', 'BKNG', 'BLK', 'ISRG', 'GILD', 'AMT', 'LMT', 'MO', 'F', 'CVS', 
    'C', 'PLD', 'UBER', 'TGT', 'SO', 'VRTX', 'MS', 'DUK', 'SYK', 'ELV', 
    'ADP', 'PNC', 'SCHW', 'ZTS', 'CB', 'GM', 'PYPL', 'CI', 'MDLZ', 'MU', 
    'CCI', 'USB', 'BDX', 'TJX', 'NSC', 'EQIX', 'CME', 'TFC', 'REGN', 'SHW'
]

# Download data for multiple symbols individually
all_data = []

for symbol in symbols:
    try:
        print(f"Downloading data for {symbol}...")
        data = yf.download(symbol, start="2022-03-19", end="2025-03-19")
        data['symbol'] = symbol  # Add symbol as a column to the data
        all_data.append(data)
        time.sleep(1)  # Wait to avoid hitting rate limits
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

# Combine all data into a single DataFrame
if all_data:
    combined_data = pd.concat(all_data)
    # Save data to a CSV
    combined_data.to_csv('stocks_data.csv')
    print(f"Data saved to 'stocks_data.csv'.")
else:
    print("No data fetched.")