import yfinance as yf
import time
import pandas as pd

# Define the stock symbols
symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B', 'JPM', 'V', 
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
# Create an empty list to store all dataframes
all_data = []

for symbol in symbols:
    try:
        print(f"Downloading data for {symbol}...")
        
        # Download data for the symbol
        # The key fix: auto_adjust=True simplifies the columns
        data = yf.download(symbol, start="2022-03-19", end="2025-03-19", 
                          progress=False, auto_adjust=True)
        
        # Handle multi-level column indices if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        # Reset the index to make Date a column
        data = data.reset_index()
        
        # Add symbol as a column
        data['Symbol'] = symbol
        
        # Append to our list
        all_data.append(data)
        
        print(f"Added {len(data)} rows for {symbol}")
        
        # Sleep to avoid rate limits
        time.sleep(1)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

# Check if we have data
if all_data:
    # Combine all dataframes
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)
    
    # Verify column types (make sure they're strings, not tuples)
    print(f"Column types: {type(combined_data.columns[0])}")
    
    # Explicitly ensure column names are strings
    combined_data.columns = [str(col) for col in combined_data.columns]
    
    # Save to CSV
    combined_data.to_csv('updated_data.csv', index=False)
    print(f"Data saved to 'stocks_data.csv' with {len(combined_data)} total rows.")
    print(f"Columns: {', '.join(combined_data.columns)}")
    
    # Print the first few rows to verify format
    print("\nFirst 2 rows of data:")
    print(combined_data.head(2))
else:
    print("No data fetched.")