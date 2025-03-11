import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Dashboard = () => {
  const [symbol, setSymbol] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [modelSymbol, setModelSymbol] = useState('');
  const [modelFile, setModelFile] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    loadAvailableModels();
  }, []);

  const loadAvailableModels = async () => {
    try {
      const response = await axios.get('/available-models');
      setAvailableModels(response.data.models);
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const handlePrediction = async (e) => {
    e.preventDefault();
    if (!symbol) {
      setError('Please enter a stock symbol');
      return;
    }

    setIsLoading(true);
    setError('');
    setPredictionData(null);

    try {
      const formData = new FormData();
      formData.append('symbol', symbol.toUpperCase());

      const response = await axios.post('/predict', formData);
      setPredictionData(response.data);
    } catch (error) {
      setError(error.response?.data?.error || 'An error occurred during prediction');
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelUpload = async (e) => {
    e.preventDefault();
    if (!modelSymbol || !modelFile) {
      setError('Please enter a stock symbol and select a model file');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('symbol', modelSymbol.toUpperCase());
      formData.append('model', modelFile);

      await axios.post('/upload', formData);
      setModelSymbol('');
      setModelFile(null);
      // Reset file input (requires reference)
      document.getElementById('modelFile').value = '';
      loadAvailableModels();
      alert('Model uploaded successfully!');
    } catch (error) {
      setError(error.response?.data?.error || 'An error occurred during upload');
    }
  };

  const handleModelSelect = (selectedSymbol) => {
    setSymbol(selectedSymbol);
    // Automatically trigger prediction
    setTimeout(() => {
      document.getElementById('predictionForm').dispatchEvent(new Event('submit'));
    }, 100);
  };

  // Calculate future date (skip weekends)
  const calculateFutureDate = (currentDate, daysToAdd) => {
    const futureDate = new Date(currentDate);
    let actualDaysAdded = 0;
    
    while (actualDaysAdded < daysToAdd) {
      futureDate.setDate(futureDate.getDate() + 1);
      // Skip weekends
      if (futureDate.getDay() !== 0 && futureDate.getDay() !== 6) {
        actualDaysAdded++;
      }
    }
    
    return futureDate.toLocaleDateString();
  };

  const getDecisionClass = (decision) => {
    if (decision === 'BUY') return 'text-green-600 font-bold';
    if (decision === 'SELL') return 'text-red-600 font-bold';
    return 'text-yellow-500 font-bold';
  };

  return (
    <div className="bg-gray-100 min-h-screen p-4">
      <div className="container mx-auto">
        <div className="bg-gray-800 text-white text-center p-6 rounded-md mb-8">
          <h1 className="text-3xl font-bold">Stock Price Prediction</h1>
          <p className="text-xl">LSTM-based stock price forecasting</p>
        </div>
        
        <div className="flex flex-col md:flex-row gap-6">
          {/* Left Column: Input Form */}
          <div className="w-full md:w-1/3">
            <div className="bg-white rounded-md shadow-md mb-6">
              <div className="bg-gray-50 p-4 font-semibold border-b">Predict Stock Price</div>
              <div className="p-4">
                <form id="predictionForm" onSubmit={handlePrediction}>
                  <div className="mb-4">
                    <label htmlFor="symbol" className="block mb-2">Stock Symbol</label>
                    <input 
                      type="text" 
                      className="w-full border rounded-md p-2" 
                      id="symbol" 
                      value={symbol}
                      onChange={(e) => setSymbol(e.target.value)}
                      placeholder="AAPL" 
                      required 
                    />
                    <div className="text-sm text-gray-500 mt-1">Enter the stock symbol (e.g., AAPL for Apple)</div>
                  </div>
                  <button type="submit" className="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700">
                    Predict
                  </button>
                </form>
                
                {isLoading && (
                  <div className="text-center mt-4">
                    <div className="inline-block w-6 h-6 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                    <p className="mt-2">Fetching data and making predictions...</p>
                  </div>
                )}
                
                {error && (
                  <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-md">
                    {error}
                  </div>
                )}
              </div>
            </div>

            <div className="bg-white rounded-md shadow-md">
              <div className="bg-gray-50 p-4 font-semibold border-b">Available Models</div>
              <div className="p-4">
                <div id="modelsList">
                  {availableModels.length > 0 ? (
                    <>
                      <p>Available models for:</p>
                      <ul className="list-disc pl-5 mt-2">
                        {availableModels.map((model, index) => (
                          <li key={index}>
                            <button 
                              className="text-blue-600 hover:underline" 
                              onClick={() => handleModelSelect(model)}
                            >
                              {model}
                            </button>
                          </li>
                        ))}
                      </ul>
                    </>
                  ) : (
                    <p>No models available. Please upload a model first.</p>
                  )}
                </div>
                
                <hr className="my-4" />
                
                <h6 className="font-semibold mb-2">Upload New Model</h6>
                <form onSubmit={handleModelUpload}>
                  <div className="mb-3">
                    <label htmlFor="modelSymbol" className="block mb-2">Stock Symbol</label>
                    <input 
                      type="text" 
                      className="w-full border rounded-md p-2" 
                      id="modelSymbol" 
                      value={modelSymbol}
                      onChange={(e) => setModelSymbol(e.target.value)}
                      placeholder="AAPL" 
                      required 
                    />
                  </div>
                  <div className="mb-3">
                    <label htmlFor="modelFile" className="block mb-2">Model File (.h5)</label>
                    <input 
                      type="file" 
                      className="w-full border rounded-md p-2" 
                      id="modelFile" 
                      onChange={(e) => setModelFile(e.target.files[0])}
                      accept=".h5" 
                      required 
                    />
                  </div>
                  <button type="submit" className="w-full bg-gray-600 text-white py-2 rounded-md hover:bg-gray-700">
                    Upload Model
                  </button>
                </form>
              </div>
            </div>
          </div>
          
          {/* Right Column: Results */}
          <div className="w-full md:w-2/3">
            <div className="bg-white rounded-md shadow-md mb-6">
              <div className="bg-gray-50 p-4 font-semibold border-b">Current Price Information</div>
              <div className="p-4">
                {predictionData ? (
                  <div className="flex justify-between">
                    <div>
                      <h3 className="text-xl font-bold">{predictionData.symbol}</h3>
                      <p>As of {predictionData.current_date}</p>
                    </div>
                    <div className="text-right">
                      <h2 className="text-2xl font-bold">${predictionData.current_price.toFixed(2)}</h2>
                      <p>Current Price</p>
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-500">Select a stock and click "Predict" to see information</p>
                )}
              </div>
            </div>
            
            <div className="bg-white rounded-md shadow-md mb-6">
              <div className="bg-gray-50 p-4 font-semibold border-b">Price Prediction Chart</div>
              <div className="p-4">
                {predictionData && predictionData.plot ? (
                  <img 
                    src={`data:image/png;base64,${predictionData.plot}`} 
                    alt="Prediction Chart" 
                    className="w-full h-auto mt-4"
                  />
                ) : (
                  <p className="text-gray-500">Select a stock and click "Predict" to see the chart</p>
                )}
              </div>
            </div>
            
            <div className="bg-white rounded-md shadow-md">
              <div className="bg-gray-50 p-4 font-semibold border-b">Trading Predictions</div>
              <div className="p-4">
                {predictionData && predictionData.predictions ? (
                  <div className="overflow-x-auto">
                    <table className="min-w-full border-collapse">
                      <thead>
                        <tr className="bg-gray-50">
                          <th className="border p-2 text-left">Day</th>
                          <th className="border p-2 text-left">Date</th>
                          <th className="border p-2 text-left">Price</th>
                          <th className="border p-2 text-left">Change</th>
                          <th className="border p-2 text-left">Decision</th>
                        </tr>
                      </thead>
                      <tbody>
                        {predictionData.predictions.map((pred, index) => {
                          const formattedDate = calculateFutureDate(predictionData.current_date, pred.day);
                          const changeFormatted = pred.expected_return > 0 ? 
                            `+${pred.expected_return.toFixed(2)}%` : 
                            `${pred.expected_return.toFixed(2)}%`;
                            
                          return (
                            <tr key={index} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                              <td className="border p-2">{pred.day}</td>
                              <td className="border p-2">{formattedDate}</td>
                              <td className="border p-2">${pred.predicted_price.toFixed(2)}</td>
                              <td className="border p-2">{changeFormatted}</td>
                              <td className={`border p-2 ${getDecisionClass(pred.decision)}`}>
                                {pred.decision}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-gray-500">Select a stock and click "Predict" to see predictions</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;