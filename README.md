# bitHunter - LLM-Powered Crypto Trading Platform

## Overview

bitHunter is an advanced cryptocurrency trading platform that combines Large Language Models (LLM) with technical analysis to generate trading signals. The platform offers:

- **LLM-based sentiment analysis** of cryptocurrency news and market data
- **Backtesting engine** for evaluating trading strategies
- **Live data support** from cryptocurrency exchanges and news APIs
- **Advanced technical indicators** and risk metrics
- **Interactive dashboard** with real-time updates

## Architecture

The platform consists of five main modules:

1. **LLM Sentiment Analysis** - Analyzes news articles and social media content to gauge market sentiment
2. **Signal Generation** - Creates buy/sell signals based on sentiment and technical indicators
3. **Backtesting Engine** - Tests trading strategies against historical or live data
4. **Live Data Ingestion** - Fetches real-time cryptocurrency prices and news from multiple sources
5. **Dashboard** - Visualizes performance metrics, trading signals, and technical indicators

## Dashboard Modes

The platform supports multiple operational modes:

- **Historical Mode** - Analyze uploaded historical data or use sample simulation data
- **Live Mode** - Fetch live cryptocurrency prices and news from APIs for real-time analysis

## Getting Started

### Prerequisites

- Python 3.9+
- Required libraries (install using `pip install -r requirements.txt`):
  - Flask
  - Pandas
  - NumPy
  - Matplotlib
  - Requests
  - CCXT (for exchange API access)
  - OpenAI (for LLM sentiment analysis)

### Configuration

Set the following environment variables:

```
# API Keys for Live Data
CRYPTOCOMPARE_API_KEY=your_cryptocompare_api_key
COINGECKO_API_KEY=your_coingecko_api_key  
NEWSAPI_KEY=your_newsapi_key

# LLM API Configuration
OPENAI_API_KEY=your_openai_api_key
```

### Running the Platform

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the dashboard: `python dashboard/app.py`
4. Access the dashboard at `http://localhost:5000`

## Using Live Data Mode

1. Set up the required API keys in your environment variables
2. From the dashboard, click "Toggle Mode" to switch to Live Mode
3. Enter the cryptocurrency symbol you want to track (e.g., BTC/USD)
4. Click "Refresh" to fetch the latest data
5. The dashboard will display live price data, technical indicators, and news sentiment

Live data is automatically refreshed every 5 minutes, or you can manually refresh it at any time.

## Technical Indicators

The platform calculates and displays various technical indicators:

- **Moving Averages** (SMA, EMA)
- **Relative Strength Index (RSI)**
- **Moving Average Convergence Divergence (MACD)**
- **Bollinger Bands**

## Risk Metrics

Comprehensive risk assessment metrics are included:

- **Sharpe Ratio** - Return adjusted for risk
- **Sortino Ratio** - Return adjusted for downside risk
- **Maximum Drawdown** - Largest peak-to-trough decline
- **Volatility** - Price fluctuation measurement
- **Value at Risk (VaR)** - Maximum expected loss within a confidence interval
- **Calmar Ratio** - Return relative to maximum drawdown

## Backtesting

The backtesting engine allows you to test trading strategies against both historical and live data:

1. Upload historical data or use live data
2. Run the backtest from the dashboard
3. Analyze performance metrics and risk assessment
4. Visualize results through interactive charts

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project uses several open-source libraries and APIs for data collection and analysis
- Special thanks to the cryptocurrency and LLM communities for their valuable resources and documentation
