import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [symbol, setSymbol] = useState('');
  const [daysAhead, setDaysAhead] = useState(10); // New state for custom days ahead
  const [modelSymbol, setModelSymbol] = useState('');
  const [modelFile, setModelFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);

  useEffect(() => {
    loadAvailableModels();
  }, []);

  const loadAvailableModels = async () => {
    try {
      const res = await axios.get('/available-models');
      setAvailableModels(res.data.models);
    } catch (error) {
      console.error('Error loading models', error);
    }
  };

  const handlePredictSubmit = async (e) => {
    e.preventDefault();
    if (!symbol) {
      alert('Please enter a stock symbol');
      return;
    }
    setLoading(true);
    setPredictionResult(null);
  
    try {
      const formData = new URLSearchParams();
      formData.append('symbol', symbol.toUpperCase());
      // Append the custom days ahead value
      formData.append('days_ahead', daysAhead);

      const res = await axios.post('/predict', formData, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      });
      setPredictionResult(res.data);
    } catch (error) {
      const errMsg = error.response?.data?.error || 'An error occurred';
      alert(errMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (!modelSymbol || !modelFile) {
      alert('Please enter a stock symbol and select a model file');
      return;
    }
    const formData = new FormData();
    formData.append('symbol', modelSymbol.toUpperCase());
    formData.append('model', modelFile);

    try {
      await axios.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      alert('Model uploaded successfully!');
      setModelSymbol('');
      setModelFile(null);
      loadAvailableModels();
    } catch (error) {
      const errMsg = error.response?.data?.error || 'An error occurred';
      alert(errMsg);
    }
  };

  const computeFutureDate = (baseDate, daysAhead) => {
    let futureDate = new Date(baseDate);
    let actualDaysAdded = 0;
    while (actualDaysAdded < daysAhead) {
      futureDate.setDate(futureDate.getDate() + 1);
      const day = futureDate.getDay();
      if (day !== 0 && day !== 6) {
        actualDaysAdded++;
      }
    }
    return futureDate.toLocaleDateString();
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>StockSense</h1>
        <p>Advanced LSTM-based stock price forecasting</p>
      </div>

      <div className="content-grid">
        {/* Left column: Inputs and controls */}
        <div className="sidebar">
          <div className="card predict-card">
            <h2>Predict Stock</h2>
            <form id="predictForm" onSubmit={handlePredictSubmit}>
              <div className="input-group">
                <label htmlFor="symbol">Stock Symbol</label>
                <input
                  type="text"
                  id="symbol"
                  placeholder="e.g., AAPL"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                  required
                />
                <small>Enter stock ticker symbol (AAPL, MSFT, etc.)</small>
              </div>
              {/* New input for custom prediction days */}
              <div className="input-group">
                <label htmlFor="daysAhead">Days Ahead</label>
                <input
                  type="number"
                  id="daysAhead"
                  placeholder="e.g., 10"
                  value={daysAhead}
                  onChange={(e) => setDaysAhead(e.target.value)}
                  min="1"
                  required
                />
                <small>Enter number of days to predict</small>
              </div>
              <button type="submit" className="btn primary-btn">
                {loading ? (
                  <span className="loading-spinner"></span>
                ) : (
                  "Generate Prediction"
                )}
              </button>
            </form>
          </div>

          <div className="card models-card">
            <h2>Models Library</h2>
            {availableModels.length > 0 ? (
              <div className="models-list">
                <p>Select from available models:</p>
                <div className="model-chips">
                  {availableModels.map((m) => (
                    <button
                      key={m}
                      className="model-chip"
                      onClick={() => {
                        setSymbol(m);
                        setTimeout(() => document.getElementById('predictForm')?.dispatchEvent(
                          new Event('submit', { cancelable: true, bubbles: true })
                        ), 0);
                      }}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <p className="no-models">No models available. Upload one below.</p>
            )}
          </div>

          <div className="card upload-card">
            <h2>Upload New Model</h2>
            <form onSubmit={handleUploadSubmit}>
              <div className="input-group">
                <label htmlFor="modelSymbol">Stock Symbol</label>
                <input
                  type="text"
                  id="modelSymbol"
                  placeholder="e.g., AAPL"
                  value={modelSymbol}
                  onChange={(e) => setModelSymbol(e.target.value)}
                  required
                />
              </div>
              <div className="input-group file-upload">
                <label htmlFor="modelFile">Model File (.h5)</label>
                <input
                  type="file"
                  id="modelFile"
                  accept=".h5"
                  onChange={(e) => setModelFile(e.target.files[0])}
                  required
                />
                <span className="file-info">
                  {modelFile ? modelFile.name : "No file selected"}
                </span>
              </div>
              <button type="submit" className="btn secondary-btn">
                Upload Model
              </button>
            </form>
          </div>
        </div>

        {/* Right column: Results */}
        <div className="main-content">
          {!predictionResult && !loading && (
            <div className="placeholder-message">
              <div className="placeholder-icon">📈</div>
              <h2>Select a stock to view predictions</h2>
              <p>Enter a stock symbol or choose from available models</p>
            </div>
          )}

          {loading && (
            <div className="loading-container">
              <div className="loader"></div>
              <p>Analyzing market data and generating predictions...</p>
            </div>
          )}

          {predictionResult && (
            <>
              <div className="current-price-card">
                <div className="price-header">
                  <div>
                    <h2>{predictionResult.symbol}</h2>
                    <p className="date-info">Data as of {predictionResult.current_date}</p>
                  </div>
                  <div className="price-display">
                    <h1>${parseFloat(predictionResult.current_price).toFixed(2)}</h1>
                    <p>Current Price</p>
                  </div>
                </div>
              </div>

              <div className="card chart-card">
                <h2>Price Forecast</h2>
                <div className="chart-container">
                  <img
                    src={`data:image/png;base64,${predictionResult.plot}`}
                    alt={`${predictionResult.symbol} Price Prediction Chart`}
                    className="prediction-chart"
                  />
                </div>
              </div>

              <div className="card prediction-table-card">
                <h2>Trading Signals</h2>
                <div className="table-container">
                  <table className="prediction-table">
                    <thead>
                      <tr>
                        <th>Day</th>
                        <th>Date</th>
                        <th>Predicted Price</th>
                        <th>Expected Change</th>
                        <th>Signal</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictionResult.predictions.map((pred) => {
                        const futureDate = computeFutureDate(predictionResult.current_date, pred.day);
                        const changeFormatted = pred.expected_return > 0
                          ? `+${pred.expected_return.toFixed(2)}%`
                          : `${pred.expected_return.toFixed(2)}%`;
                        
                        return (
                          <tr key={pred.day}>
                            <td>{pred.day}</td>
                            <td>{futureDate}</td>
                            <td className="price-cell">${pred.predicted_price.toFixed(2)}</td>
                            <td className={pred.expected_return >= 0 ? "positive-change" : "negative-change"}>
                              {changeFormatted}
                            </td>
                            <td>
                              <span className={`signal-badge ${pred.decision.toLowerCase()}`}>
                                {pred.decision}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;