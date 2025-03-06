import React, { useState } from 'react';
import './Dashboard.css';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { Layers, Activity, DollarSign, TrendingUp, BarChart2, Settings, RefreshCw } from 'lucide-react';

// Sample data - replace with your actual API calls
const sampleStocks = [
  { symbol: 'AAPL', name: 'Apple Inc.', lastPrice: 178.72, change: 2.34, changePercent: 1.32 },
  { symbol: 'MSFT', name: 'Microsoft Corp.', lastPrice: 328.79, change: -1.21, changePercent: -0.37 },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', lastPrice: 137.14, change: 0.56, changePercent: 0.41 },
  { symbol: 'AMZN', name: 'Amazon.com Inc.', lastPrice: 178.75, change: 3.45, changePercent: 1.97 },
  { symbol: 'NVDA', name: 'NVIDIA Corp.', lastPrice: 950.02, change: 15.32, changePercent: 1.64 },
];

const samplePerformance = [
  { date: '2025-01-01', modelReturns: 2.1, marketReturns: 1.2 },
  { date: '2025-01-02', modelReturns: 2.8, marketReturns: 1.8 },
  { date: '2025-01-03', modelReturns: 1.9, marketReturns: 2.2 },
  { date: '2025-01-04', modelReturns: 3.4, marketReturns: 2.5 },
  { date: '2025-01-05', modelReturns: 3.9, marketReturns: 2.1 },
  { date: '2025-01-06', modelReturns: 4.2, marketReturns: 2.4 },
  { date: '2025-01-07', modelReturns: 4.5, marketReturns: 2.8 },
];

const samplePredictions = [
  { symbol: 'AAPL', predictedMove: 2.3, confidence: 85, signal: 'buy' },
  { symbol: 'MSFT', predictedMove: -0.8, confidence: 72, signal: 'hold' },
  { symbol: 'GOOGL', predictedMove: 1.5, confidence: 68, signal: 'buy' },
  { symbol: 'AMZN', predictedMove: 3.2, confidence: 92, signal: 'buy' },
  { symbol: 'NVDA', predictedMove: -1.2, confidence: 75, signal: 'sell' },
];

const samplePositions = [
  { symbol: 'AAPL', shares: 100, entryPrice: 165.42, currentPrice: 178.72, pl: 1330.0, plPercent: 8.03 },
  { symbol: 'MSFT', shares: 50, entryPrice: 310.15, currentPrice: 328.79, pl: 932.0, plPercent: 6.01 },
  { symbol: 'AMZN', shares: 75, entryPrice: 160.3, currentPrice: 178.75, pl: 1383.75, plPercent: 11.51 },
];

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [modelRunning, setModelRunning] = useState(true);
  const [riskLevel, setRiskLevel] = useState('medium');
  const [modelMetrics, setModelMetrics] = useState({
    accuracy: 78.5,
    sharpeRatio: 2.14,
    sortino: 2.68,
    maxDrawdown: -12.4,
    totalReturn: 24.8
  });

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-left">
          <TrendingUp className="icon" />
          <h1 className="header-title">Trading Model Dashboard</h1>
        </div>
        <div className="header-right">
          <span className={`model-badge ${modelRunning ? 'badge-active' : 'badge-paused'}`}>
            {modelRunning ? 'Model Active' : 'Model Paused'}
          </span>
          <button
            className="icon-button"
            onClick={() => setModelRunning(!modelRunning)}
            title="Toggle Model Status"
          >
            <RefreshCw className="icon" />
          </button>
        </div>
      </header>

      {/* Main layout */}
      <div className="main-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <nav className="sidebar-nav">
            <ul>
              <li>
                <button
                  className={`sidebar-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
                  onClick={() => setActiveTab('dashboard')}
                >
                  <Layers className="icon" />
                  <span className="btn-text">Dashboard</span>
                </button>
              </li>
              <li>
                <button
                  className={`sidebar-btn ${activeTab === 'predictions' ? 'active' : ''}`}
                  onClick={() => setActiveTab('predictions')}
                >
                  <Activity className="icon" />
                  <span className="btn-text">Predictions</span>
                </button>
              </li>
              <li>
                <button
                  className={`sidebar-btn ${activeTab === 'positions' ? 'active' : ''}`}
                  onClick={() => setActiveTab('positions')}
                >
                  <DollarSign className="icon" />
                  <span className="btn-text">Positions</span>
                </button>
              </li>
              <li>
                <button
                  className={`sidebar-btn ${activeTab === 'performance' ? 'active' : ''}`}
                  onClick={() => setActiveTab('performance')}
                >
                  <BarChart2 className="icon" />
                  <span className="btn-text">Performance</span>
                </button>
              </li>
              <li>
                <button
                  className={`sidebar-btn ${activeTab === 'settings' ? 'active' : ''}`}
                  onClick={() => setActiveTab('settings')}
                >
                  <Settings className="icon" />
                  <span className="btn-text">Settings</span>
                </button>
              </li>
            </ul>
          </nav>
        </aside>

        {/* Content */}
        <main className="content-area">
          {activeTab === 'dashboard' && (
            <div className="tab-content">
              <h2 className="section-title">Dashboard Overview</h2>

              {/* Model metrics */}
              <div className="metrics-grid">
                <div className="metric-card">
                  <h3 className="metric-label">Accuracy</h3>
                  <p className="metric-value">{modelMetrics.accuracy}%</p>
                </div>
                <div className="metric-card">
                  <h3 className="metric-label">Sharpe Ratio</h3>
                  <p className="metric-value">{modelMetrics.sharpeRatio}</p>
                </div>
                <div className="metric-card">
                  <h3 className="metric-label">Sortino Ratio</h3>
                  <p className="metric-value">{modelMetrics.sortino}</p>
                </div>
                <div className="metric-card">
                  <h3 className="metric-label">Max Drawdown</h3>
                  <p className="metric-value negative">{modelMetrics.maxDrawdown}%</p>
                </div>
                <div className="metric-card">
                  <h3 className="metric-label">Total Return</h3>
                  <p className="metric-value positive">+{modelMetrics.totalReturn}%</p>
                </div>
              </div>

              {/* Performance chart */}
              <div className="card chart-card">
                <h3 className="chart-title">Performance vs Market</h3>
                <div className="chart-wrapper">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={samplePerformance}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="modelReturns"
                        stroke="#3B82F6"
                        name="Model Returns %"
                      />
                      <Line
                        type="monotone"
                        dataKey="marketReturns"
                        stroke="#9CA3AF"
                        name="Market Returns %"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Latest predictions */}
              <div className="card table-card">
                <div className="table-header">
                  <h3 className="table-title">Latest Predictions</h3>
                  <button
                    className="link-button"
                    onClick={() => setActiveTab('predictions')}
                  >
                    View all
                  </button>
                </div>
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Predicted Move</th>
                        <th>Confidence</th>
                        <th>Signal</th>
                      </tr>
                    </thead>
                    <tbody>
                      {samplePredictions.slice(0, 3).map((prediction, i) => (
                        <tr key={i}>
                          <td>{prediction.symbol}</td>
                          <td
                            className={
                              prediction.predictedMove > 0 ? 'positive' : 'negative'
                            }
                          >
                            {prediction.predictedMove > 0 ? '+' : ''}
                            {prediction.predictedMove}%
                          </td>
                          <td>
                            <div className="progress-bg">
                              <div
                                className={
                                  prediction.confidence > 80
                                    ? 'progress-bar high'
                                    : prediction.confidence > 60
                                    ? 'progress-bar medium'
                                    : 'progress-bar low'
                                }
                                style={{ width: `${prediction.confidence}%` }}
                              ></div>
                            </div>
                            <span className="confidence-text">
                              {prediction.confidence}%
                            </span>
                          </td>
                          <td>
                            <span
                              className={`signal-badge ${
                                prediction.signal === 'buy'
                                  ? 'buy'
                                  : prediction.signal === 'sell'
                                  ? 'sell'
                                  : 'hold'
                              }`}
                            >
                              {prediction.signal.toUpperCase()}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'predictions' && (
            <div className="tab-content">
              <h2 className="section-title">Model Predictions</h2>
              <div className="card table-card">
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Predicted Move</th>
                        <th>Confidence</th>
                        <th>Signal</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {samplePredictions.map((prediction, i) => (
                        <tr key={i}>
                          <td>{prediction.symbol}</td>
                          <td
                            className={
                              prediction.predictedMove > 0 ? 'positive' : 'negative'
                            }
                          >
                            {prediction.predictedMove > 0 ? '+' : ''}
                            {prediction.predictedMove}%
                          </td>
                          <td>
                            <div className="progress-bg">
                              <div
                                className={
                                  prediction.confidence > 80
                                    ? 'progress-bar high'
                                    : prediction.confidence > 60
                                    ? 'progress-bar medium'
                                    : 'progress-bar low'
                                }
                                style={{ width: `${prediction.confidence}%` }}
                              ></div>
                            </div>
                            <span className="confidence-text">
                              {prediction.confidence}%
                            </span>
                          </td>
                          <td>
                            <span
                              className={`signal-badge ${
                                prediction.signal === 'buy'
                                  ? 'buy'
                                  : prediction.signal === 'sell'
                                  ? 'sell'
                                  : 'hold'
                              }`}
                            >
                              {prediction.signal.toUpperCase()}
                            </span>
                          </td>
                          <td>
                            <button
                              className={`action-button ${
                                prediction.signal === 'buy'
                                  ? 'buy-button'
                                  : prediction.signal === 'sell'
                                  ? 'sell-button'
                                  : 'hold-button'
                              }`}
                            >
                              {prediction.signal === 'buy'
                                ? 'Buy'
                                : prediction.signal === 'sell'
                                ? 'Sell'
                                : 'Hold'}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'positions' && (
            <div className="tab-content">
              <h2 className="section-title">Current Positions</h2>
              <div className="card table-card">
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Shares</th>
                        <th>Entry Price</th>
                        <th>Current Price</th>
                        <th>P&amp;L</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {samplePositions.map((position, i) => (
                        <tr key={i}>
                          <td>{position.symbol}</td>
                          <td>{position.shares}</td>
                          <td>${position.entryPrice.toFixed(2)}</td>
                          <td>${position.currentPrice.toFixed(2)}</td>
                          <td className={position.pl >= 0 ? 'positive' : 'negative'}>
                            ${position.pl.toFixed(2)} (
                            {position.plPercent >= 0 ? '+' : ''}
                            {position.plPercent.toFixed(2)}%)
                          </td>
                          <td>
                            <button className="action-button close-button">Close</button>
                            <button className="action-button edit-button">Edit</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'performance' && (
            <div className="tab-content">
              <h2 className="section-title">Model Performance</h2>
              <div className="performance-grid">
                <div className="card chart-card">
                  <h3 className="chart-title">Cumulative Returns</h3>
                  <div className="chart-wrapper">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={samplePerformance}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Area
                          type="monotone"
                          dataKey="modelReturns"
                          stackId="1"
                          stroke="#3B82F6"
                          fill="#93C5FD"
                          name="Model Returns %"
                        />
                        <Area
                          type="monotone"
                          dataKey="marketReturns"
                          stackId="2"
                          stroke="#9CA3AF"
                          fill="#E5E7EB"
                          name="Market Returns %"
                        />
                        <Legend />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="card chart-card">
                  <h3 className="chart-title">Daily Returns</h3>
                  <div className="chart-wrapper">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={samplePerformance}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="modelReturns" fill="#3B82F6" name="Model Returns %" />
                        <Bar dataKey="marketReturns" fill="#9CA3AF" name="Market Returns %" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div className="card">
                <h3 className="chart-title">Performance Metrics</h3>
                <div className="metrics-grid smaller">
                  <div className="metric-card">
                    <h4 className="metric-label">Accuracy</h4>
                    <p className="metric-value">{modelMetrics.accuracy}%</p>
                  </div>
                  <div className="metric-card">
                    <h4 className="metric-label">Sharpe Ratio</h4>
                    <p className="metric-value">{modelMetrics.sharpeRatio}</p>
                  </div>
                  <div className="metric-card">
                    <h4 className="metric-label">Sortino Ratio</h4>
                    <p className="metric-value">{modelMetrics.sortino}</p>
                  </div>
                  <div className="metric-card">
                    <h4 className="metric-label">Max Drawdown</h4>
                    <p className="metric-value negative">{modelMetrics.maxDrawdown}%</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="tab-content">
              <h2 className="section-title">Model Settings</h2>
              <div className="card settings-card">
                <h3 className="settings-title">Trading Parameters</h3>

                <div className="form-group">
                  <label className="label">Model Status</label>
                  <div className="button-group">
                    <button
                      className={`switch-button ${modelRunning ? 'switch-active' : ''}`}
                      onClick={() => setModelRunning(true)}
                    >
                      Active
                    </button>
                    <button
                      className={`switch-button ${!modelRunning ? 'switch-paused' : ''}`}
                      onClick={() => setModelRunning(false)}
                    >
                      Paused
                    </button>
                  </div>
                </div>

                <div className="form-group">
                  <label className="label">Risk Level</label>
                  <div className="button-group">
                    <button
                      className={`switch-button ${riskLevel === 'low' ? 'switch-active' : ''}`}
                      onClick={() => setRiskLevel('low')}
                    >
                      Low
                    </button>
                    <button
                      className={`switch-button ${riskLevel === 'medium' ? 'switch-active' : ''}`}
                      onClick={() => setRiskLevel('medium')}
                    >
                      Medium
                    </button>
                    <button
                      className={`switch-button ${riskLevel === 'high' ? 'switch-active' : ''}`}
                      onClick={() => setRiskLevel('high')}
                    >
                      High
                    </button>
                  </div>
                </div>

                <div className="form-group">
                  <label className="label">Maximum Position Size (% of Portfolio)</label>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    defaultValue="5"
                    className="range-input"
                  />
                  <div className="range-scale">
                    <span>1%</span>
                    <span>5%</span>
                    <span>10%</span>
                    <span>15%</span>
                    <span>20%</span>
                  </div>
                </div>

                <div className="form-group">
                  <label className="label">Confidence Threshold for Trades</label>
                  <input
                    type="range"
                    min="50"
                    max="95"
                    defaultValue="70"
                    className="range-input"
                  />
                  <div className="range-scale">
                    <span>50%</span>
                    <span>60%</span>
                    <span>70%</span>
                    <span>80%</span>
                    <span>90%</span>
                  </div>
                </div>

                <div className="form-group">
                  <button className="save-button">Save Settings</button>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default Dashboard;