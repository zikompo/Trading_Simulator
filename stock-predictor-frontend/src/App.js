import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // Separate states for stock symbol, selected model, uploaded model file, prediction days, etc.
  const [stockSymbol, setStockSymbol] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [modelFile, setModelFile] = useState(null);
  const [predictionDays, setPredictionDays] = useState(1);
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
    if (!stockSymbol) {
      alert('Please enter a stock symbol');
      return;
    }
    if (!selectedModel) {
      alert('Please select a model from the library');
      return;
    }
    setLoading(true);
    setPredictionResult(null);
  
    try {
      const formData = new URLSearchParams();
      // 'symbol' holds the model name for lookup on the server.
      formData.append('symbol', selectedModel.toUpperCase());
      // 'stock' holds the ticker for which to fetch data.
      formData.append('stock', stockSymbol.toUpperCase());
      // Append the number of days to predict (up to 5)
      formData.append('days_ahead', predictionDays);
      
      console.log("Model selected:", selectedModel, "Days ahead:", predictionDays);
      // Determine the endpoint based on the selected model.
      let endpoint = '/predict_kernel'; // default endpoint
      if (selectedModel.toUpperCase() === 'GENERAL_LSTM_MODEL') {
        endpoint = '/predict_lstm';
      } else if (selectedModel.toUpperCase() === 'KNN_MODEL') {
        endpoint = '/predict_knn';
      } else if (selectedModel.toUpperCase() === 'LINEAR_REGRESSION_MODEL') {
        endpoint = '/predict_linear';
      } else if (selectedModel.toUpperCase() === 'Q_LEARNING_AGENT') {
        endpoint = '/predict_qlearning';
      } else if (selectedModel.toUpperCase() === 'GENERAL_RNN_MODEL') {
        endpoint = '/predict_rnn';
      }
  
      const res = await axios.post(endpoint, formData, {
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
                <label htmlFor="stockSymbol">Stock Symbol</label>
                <input
                  type="text"
                  id="stockSymbol"
                  placeholder="e.g., AAPL"
                  value={stockSymbol}
                  onChange={(e) => setStockSymbol(e.target.value)}
                  required
                />
                <small>Enter stock ticker symbol (AAPL, MSFT, etc.)</small>
              </div>
              <div className="input-group">
                <label htmlFor="predictionDays">Number of Days</label>
                <input
                  type="number"
                  id="predictionDays"
                  min="1"
                  max="500"
                  value={predictionDays}
                  onChange={(e) => setPredictionDays(e.target.value)}
                  required
                />
                <small>Select the number of days to predict (max 5)</small>
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
                  {availableModels.filter((m) => !m.toLowerCase().includes("scaler")) 
                 .map((m) =>  (
                    <button
                      key={m}
                      className={`model-chip ${selectedModel === m ? "selected" : ""}`}
                      onClick={() => setSelectedModel(m)}
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

        </div>

        {/* Right column: Results */}
        <div className="main-content">
          {!predictionResult && !loading && (
            <div className="placeholder-message">
              <div className="placeholder-icon">📈</div>
              <h2>Select a stock to view predictions</h2>
              <p>Enter a stock symbol and select a model</p>
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

              {predictionResult.model_used === "Q-learning" ? (
                <div className="card qlearning-card">
                  <h2>Trading Signal</h2>
                  <p><strong>Recommended Action:</strong> {predictionResult.predicted_action}</p>
                  <p><strong>Q-values:</strong> {predictionResult.q_values.join(", ")}</p>
                </div>
              ) : (
                <>
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
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
