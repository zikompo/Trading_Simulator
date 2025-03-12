// src/App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css'; // For any custom styling

function App() {
  const [symbol, setSymbol] = useState('');
  const [modelSymbol, setModelSymbol] = useState('');
  const [modelFile, setModelFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);

  // Load available models on mount
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
      // Use URLSearchParams to encode data as form data
      const formData = new URLSearchParams();
      formData.append('symbol', symbol.toUpperCase());
  
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

  // Helper to compute future date from current_date skipping weekends
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
    <div className="container">
      <header className="header-container text-center my-4 p-4 bg-dark text-white rounded">
        <h1>Stock Price Prediction</h1>
        <p className="lead">LSTM-based stock price forecasting</p>
      </header>

      <div className="row">
        {/* Left Column: Input Form and Model Upload */}
        <div className="col-md-4">
          <div className="card mb-4 shadow-sm">
            <div className="card-header">Predict Stock Price</div>
            <div className="card-body">
              <form onSubmit={handlePredictSubmit}>
                <div className="mb-3">
                  <label htmlFor="symbol" className="form-label">Stock Symbol</label>
                  <input
                    type="text"
                    className="form-control"
                    id="symbol"
                    placeholder="AAPL"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value)}
                    required
                  />
                  <div className="form-text">Enter the stock symbol (e.g., AAPL for Apple)</div>
                </div>
                <button type="submit" className="btn btn-primary w-100">Predict</button>
              </form>
              {loading && (
                <div className="loading mt-3 text-center">
                  <div className="spinner-border text-primary" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </div>
                  <p className="mt-2">Fetching data and making predictions...</p>
                </div>
              )}
            </div>
          </div>

          <div className="card mb-4 shadow-sm">
            <div className="card-header">Available Models</div>
            <div className="card-body">
              {availableModels.length > 0 ? (
                <>
                  <p>Available models for:</p>
                  <ul>
                    {availableModels.map((m) => (
                      <li key={m}>
                        <button
                          className="btn btn-link p-0"
                          onClick={() => {
                            setSymbol(m);
                            // Trigger prediction on click:
                            setTimeout(() => document.getElementById('predictForm')?.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true })), 0);
                          }}
                        >
                          {m}
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                <p>No models available. Please upload a model first.</p>
              )}
              <hr />
              <h6>Upload New Model</h6>
              <form onSubmit={handleUploadSubmit}>
                <div className="mb-3">
                  <label htmlFor="modelSymbol" className="form-label">Stock Symbol</label>
                  <input
                    type="text"
                    className="form-control"
                    id="modelSymbol"
                    placeholder="AAPL"
                    value={modelSymbol}
                    onChange={(e) => setModelSymbol(e.target.value)}
                    required
                  />
                </div>
                <div className="mb-3">
                  <label htmlFor="modelFile" className="form-label">Model File (.h5)</label>
                  <input
                    type="file"
                    className="form-control"
                    id="modelFile"
                    accept=".h5"
                    onChange={(e) => setModelFile(e.target.files[0])}
                    required
                  />
                </div>
                <button type="submit" className="btn btn-secondary w-100">Upload Model</button>
              </form>
            </div>
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="col-md-8">
          {/* Current Price Information */}
          <div className="card mb-4 shadow-sm">
            <div className="card-header">Current Price Information</div>
            <div className="card-body">
              {predictionResult ? (
                <div className="row">
                  <div className="col-md-6">
                    <h3>{predictionResult.symbol}</h3>
                    <p>As of {predictionResult.current_date}</p>
                  </div>
                  <div className="col-md-6 text-end">
                    <h2>${parseFloat(predictionResult.current_price).toFixed(2)}</h2>
                    <p>Current Price</p>
                  </div>
                </div>
              ) : (
                <p className="text-muted">Select a stock and click "Predict" to see information</p>
              )}
            </div>
          </div>

          {/* Prediction Chart */}
          <div className="card mb-4 shadow-sm">
            <div className="card-header">Price Prediction Chart</div>
            <div className="card-body">
              {predictionResult ? (
                <img
                  id="predictionImage"
                  src={`data:image/png;base64,${predictionResult.plot}`}
                  alt="Prediction Chart"
                  className="img-fluid"
                />
              ) : (
                <p className="text-muted">Select a stock and click "Predict" to see the chart</p>
              )}
            </div>
          </div>

          {/* Trading Predictions */}
          <div className="card mb-4 shadow-sm">
            <div className="card-header">Trading Predictions</div>
            <div className="card-body">
              {predictionResult ? (
                <div className="table-responsive">
                  <table className="table table-striped">
                    <thead>
                      <tr>
                        <th>Day</th>
                        <th>Date</th>
                        <th>Price</th>
                        <th>Change</th>
                        <th>Decision</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictionResult.predictions.map((pred) => {
                        const futureDate = computeFutureDate(predictionResult.current_date, pred.day);
                        const decisionClass =
                          pred.decision === 'BUY'
                            ? 'text-success fw-bold'
                            : pred.decision === 'SELL'
                            ? 'text-danger fw-bold'
                            : 'text-warning fw-bold';
                        const changeFormatted =
                          pred.expected_return > 0
                            ? `+${pred.expected_return.toFixed(2)}%`
                            : `${pred.expected_return.toFixed(2)}%`;
                        return (
                          <tr key={pred.day}>
                            <td>{pred.day}</td>
                            <td>{futureDate}</td>
                            <td>${pred.predicted_price.toFixed(2)}</td>
                            <td>{changeFormatted}</td>
                            <td className={decisionClass}>{pred.decision}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-muted">Select a stock and click "Predict" to see predictions</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
