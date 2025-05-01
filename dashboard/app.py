import sys
import os
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set Agg backend to avoid threading issues
import matplotlib.pyplot as plt
import logging
import json
import csv # Import csv module
from datetime import datetime, timedelta, timezone # Add timezone
import time
from flask import Flask, render_template, jsonify, request, url_for, redirect, flash, session
from flask_socketio import SocketIO
from threading import Thread, Event

# Add the parent directory to the Python path so that we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Dashboard Application

This module provides a web interface to visualize:
1. Trading signals and their performance
2. Backtesting results and metrics
3. Market sentiment analysis
4. Advanced technical indicators and risk metrics
5. Live cryptocurrency data
"""

import random
from typing import List, Dict, Any, Optional, Tuple, Union
from werkzeug.utils import secure_filename

# Import project modules
from signal_generation.generator import SignalGenerator
from backtesting.backtest import BacktestEngine
from data_ingestion.collector import CryptoDataCollector  # To generate sample data and fetch live data
from llm_sentiment.analyzer import SentimentAnalyzer  # To generate sample sentiment
from analytics.indicators import (  # Import new analytics functions
    calculate_all_indicators,
    calculate_all_risk_metrics,
    calculate_moving_average,
    calculate_RSI,
    calculate_bollinger_bands,
    calculate_MACD,
    calculate_returns
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_flash_messages')
socketio = SocketIO(app)

# Application mode (historical, live, or demo)
MODE = 'demo'  # Default to demo mode

# Demo simulation thread
simulation_thread = None
simulation_stop_event = Event()

# Simulation data
simulation_data = {
    'current_day_index': 0,
    'price_data': [],
    'signals': [],
    'portfolio_value': 100000.0,
    'buy_signals': [],
    'sell_signals': [],
    'timestamps': []
}

# --- Custom Jinja Filters ---
@app.template_filter('format_datetime')
def format_datetime(value):
    """Format a datetime string by extracting just the date portion."""
    if not value:
        return 'N/A'
    # Split at 'T' to get just the date part if it's an ISO format string
    return value.split('T')[0] if 'T' in value else value

@app.template_filter('format_number')
def format_number(value, decimal_places=2):
    """Format a number with the specified number of decimal places."""
    if value is None:
        return 'N/A'
    try:
        return f"{float(value):.{decimal_places}f}"
    except (ValueError, TypeError):
        return value

@app.template_filter('format_percent')
def format_percent(value, decimal_places=2):
    """Format a decimal as a percentage with the specified number of decimal places."""
    if value is None:
        return 'N/A'
    try:
        return f"{float(value) * 100:.{decimal_places}f}%"
    except (ValueError, TypeError):
        return value

# --- Configuration ---
# Directory to save performance plots
PLOT_DIR = os.path.join(app.static_folder, 'plots')
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Directory to save indicator plots
INDICATOR_PLOT_DIR = os.path.join(app.static_folder, 'indicators')
if not os.path.exists(INDICATOR_PLOT_DIR):
    os.makedirs(INDICATOR_PLOT_DIR)

# Directory for historical data
DATA_DIR = os.path.join(app.root_path, '..', 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Default historical data file paths (can be overridden via settings)
DEFAULT_HISTORICAL_PRICE_DATA = os.path.join(DATA_DIR, 'historical_prices.csv')
DEFAULT_HISTORICAL_NEWS_DATA = os.path.join(DATA_DIR, 'historical_news.json')

# Path to Dogecoin data for demo
DOGE_DATA_PATH = os.path.join(DATA_DIR, 'previous_historical_data', 'doge-usd.csv')
# Path to Musk Tweet data for demo popups
MUSK_TWEET_DATA_PATH = os.path.join(DATA_DIR, 'previous_crypto_news', 'musk-tweet.csv')

# Application config defaults
DEFAULT_CRYPTO_SYMBOL = 'DOGE'  # Default crypto to track in demo mode
DEFAULT_DAYS_BACK = 10  # Default time window for data
SIMULATION_SPEED = 2  # Seconds between updates in demo mode

# --- Helper Functions ---

def load_dogecoin_data():
    """Load Dogecoin historical data for demo simulation."""
    try:
        df = pd.read_csv(DOGE_DATA_PATH)
        df['date'] = pd.to_datetime(df['snapped_at'])
        df['close'] = df['price']
        df = df[['date', 'close', 'market_cap', 'total_volume']]
        # Sort by date in ascending order
        df = df.sort_values('date')
        return df
    except Exception as e:
        app.logger.error(f"Error loading Dogecoin data: {str(e)}")
        # Generate fallback data if file can't be loaded
        return create_sample_historical_data(days=365)

def load_musk_tweets(file_path: str) -> List[Dict[str, Any]]:
    """Load Musk tweet data (date and image) for demo popups."""
    tweets = []
    if not os.path.exists(file_path):
        app.logger.warning(f"Musk tweet data file not found: {file_path}")
        return tweets

    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None) # Skip header row if it exists
            for row in reader:
                if len(row) >= 2:
                    date_str = row[0]
                    image_path_relative = row[1]
                    try:
                        # Parse date and convert to UTC milliseconds timestamp
                        dt_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        # Assume UTC for the date provided in the file
                        dt_obj_utc = dt_obj.replace(tzinfo=timezone.utc)
                        # Convert to milliseconds since epoch - crucial for ApexCharts/JS Date compatibility
                        timestamp_ms = int(dt_obj_utc.timestamp() * 1000)

                        # Generate the static URL for the image
                        # Use url_for within the application context
                        image_url = url_for('static', filename=image_path_relative.replace('\\\\', '/'))

                        tweets.append({
                            'timestamp': timestamp_ms,
                            'image_url': image_url
                        })
                    except ValueError as ve:
                        app.logger.error(f"Error parsing date '{date_str}' in {file_path}: {ve}")
                    except Exception as url_e:
                         # Catch potential errors during url_for if context is missing, though unlikely here
                         app.logger.error(f"Error generating URL for '{image_path_relative}': {url_e}")
    except Exception as e:
        app.logger.error(f"Error reading Musk tweet data file {file_path}: {str(e)}")

    # Remove duplicates based on timestamp (keeps the last encountered one)
    unique_tweets = {t['timestamp']: t for t in tweets}.values()
    app.logger.info(f"Loaded {len(unique_tweets)} unique Musk tweets from {file_path}")
    return list(unique_tweets)

def generate_simulation_signals(price_data, signal_frequency=0.15):
    """Generate simulated trading signals based on price movements."""
    signals = []
    buy_signals = []
    sell_signals = []
    timestamps = [pd.Timestamp(date).timestamp() * 1000 for date in price_data['date']]
    
    for i in range(1, len(price_data)):
        # Generate a signal randomly, but with higher probability on significant price changes
        price_change = (price_data['close'].iloc[i] - price_data['close'].iloc[i-1]) / price_data['close'].iloc[i-1]
        
        # Increase signal probability on significant price movements
        adjusted_probability = signal_frequency + abs(price_change) * 5  # Amplify based on price change
        
        signal_type = None
        if random.random() < adjusted_probability:
            # Determine if it's a buy or sell signal based on recent price movement
            # Plus some randomness for realism
            if price_change > 0.01 or (price_change > 0 and random.random() > 0.3):
                signal_type = 'buy'
                buy_signals.append({
                    'x': timestamps[i],
                    'y': price_data['close'].iloc[i],
                    'marker': {
                        'size': 8,
                        'fillColor': '#00B746', 
                        'strokeColor': '#00B746'
                    }
                })
            elif price_change < -0.01 or (price_change < 0 and random.random() > 0.3):
                signal_type = 'sell'
                sell_signals.append({
                    'x': timestamps[i],
                    'y': price_data['close'].iloc[i],
                    'marker': {
                        'size': 8,
                        'fillColor': '#FF4560',
                        'strokeColor': '#FF4560'
                    }
                })
        
        if signal_type:
            # Create a detailed signal
            confidence = min(0.9, 0.5 + abs(price_change) * 10)
            sentiment_score = 0.2 + price_change * 5 if signal_type == 'buy' else -0.2 + price_change * 5
            
            signals.append({
                'timestamp': price_data['date'].iloc[i].strftime('%Y-%m-%d'),
                'signal': signal_type,
                'confidence': confidence,
                'supporting_metrics': {
                    'sentiment_score': sentiment_score,
                    'volume': price_data['total_volume'].iloc[i],
                    'price_change': price_change,
                    'keywords': generate_random_keywords(signal_type)
                }
            })
    
    return signals, buy_signals, sell_signals, timestamps

def generate_random_keywords(signal_type):
    """Generate random relevant keywords based on signal type."""
    bullish_keywords = ['rally', 'adoption', 'growth', 'partnership', 'upgrade', 'development', 'breakthrough']
    bearish_keywords = ['dip', 'regulation', 'sell-off', 'correction', 'risk', 'volatility', 'concerns']
    neutral_keywords = ['market', 'trading', 'crypto', 'blockchain', 'technology', 'volume', 'trend']
    
    # Select keywords based on signal type with some overlap
    if signal_type == 'buy':
        pool = bullish_keywords + random.sample(neutral_keywords, 2)
    elif signal_type == 'sell':
        pool = bearish_keywords + random.sample(neutral_keywords, 2)
    else:
        pool = neutral_keywords
    
    # Return a random sample of 2-4 keywords
    return random.sample(pool, random.randint(2, min(4, len(pool))))

def calculate_portfolio_value(initial_value, price_data, signals, current_index):
    """Calculate portfolio value based on trading signals up to the current index."""
    if current_index <= 0:
        return initial_value
    
    portfolio_value = initial_value
    holding_coins = 0
    
    # Process all signals up to current_index
    relevant_signals = []
    for signal in signals:
        signal_date = datetime.strptime(signal['timestamp'], '%Y-%m-%d')
        if signal_date.date() <= price_data['date'].iloc[current_index].date():
            relevant_signals.append(signal)
    
    # Sort signals by date
    relevant_signals.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d'))
    
    for i, signal in enumerate(relevant_signals):
        signal_date = datetime.strptime(signal['timestamp'], '%Y-%m-%d')
        
        # Find the closest price data point for this signal
        price_idx = price_data['date'].searchsorted(signal_date) - 1
        price_idx = max(0, min(price_idx, current_index))
        
        current_price = price_data['close'].iloc[price_idx]
        
        if signal['signal'] == 'buy' and portfolio_value > 0:
            # Invest 30% of current portfolio in coins
            investment = portfolio_value * 0.3
            coins_bought = investment / current_price
            portfolio_value -= investment
            holding_coins += coins_bought
            
        elif signal['signal'] == 'sell' and holding_coins > 0:
            # Sell 70% of holding
            coins_sold = holding_coins * 0.7
            sale_value = coins_sold * current_price
            portfolio_value += sale_value
            holding_coins -= coins_sold
    
    # Add the value of any remaining coins to portfolio value
    if holding_coins > 0:
        final_price = price_data['close'].iloc[current_index]
        portfolio_value += holding_coins * final_price
    
    return portfolio_value

def start_simulation():
    """Start the simulation thread."""
    global simulation_thread, simulation_stop_event
    
    if simulation_thread is None or not simulation_thread.is_alive():
        simulation_stop_event.clear()
        simulation_thread = Thread(target=run_simulation)
        simulation_thread.daemon = True
        simulation_thread.start()
        return True
    return False

def stop_simulation():
    """Stop the simulation thread."""
    global simulation_thread, simulation_stop_event
    
    if simulation_thread and simulation_thread.is_alive():
        simulation_stop_event.set()
        simulation_thread.join(timeout=1.0)
        return True
    return False

def run_simulation():
    """Run the time-lapse simulation of Dogecoin trading."""
    global simulation_data
    
    # Load Dogecoin data
    price_data = load_dogecoin_data()
    
    # Generate signals for the entire dataset
    signals, buy_signals, sell_signals, timestamps = generate_simulation_signals(price_data)
    
    # Reset simulation data
    simulation_data = {
        'current_day_index': 0,
        'price_data': price_data,
        'signals': signals,
        'portfolio_value': 100000.0,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'timestamps': timestamps
    }
    
    # Start from day 0 and progress through the data
    day_index = 0
    
    while not simulation_stop_event.is_set() and day_index < len(price_data):
        # Update current day index
        simulation_data['current_day_index'] = day_index
        
        # Calculate portfolio value
        portfolio_value = calculate_portfolio_value(
            100000.0, price_data, signals, day_index
        )
        simulation_data['portfolio_value'] = portfolio_value
        
        # Emit update event with current data
        current_data = {
            'timestamp': price_data['date'].iloc[day_index].strftime('%Y-%m-%d'),
            'price': price_data['close'].iloc[day_index],
            'portfolio_value': portfolio_value,
            'day_index': day_index,
            'total_days': len(price_data),
            'recent_signals': [s for s in signals if datetime.strptime(s['timestamp'], '%Y-%m-%d').date() <= price_data['date'].iloc[day_index].date()][-5:] if signals else []
        }
        
        # Send update via websocket
        socketio.emit('simulation_update', current_data)
        
        # Increment day counter
        day_index += 1
        
        # Pause between updates
        time.sleep(SIMULATION_SPEED)
    
    # Signal that simulation has completed
    socketio.emit('simulation_completed', {
        'final_portfolio_value': simulation_data['portfolio_value'],
        'initial_value': 100000.0,
        'total_return': (simulation_data['portfolio_value'] - 100000.0) / 100000.0 * 100,
        'days_simulated': day_index
    })

def generate_sample_sentiment_and_signals(num_days=30) -> List[Dict[str, Any]]:
    """Generate sample sentiment data and corresponding trading signals."""
    analyzer = SentimentAnalyzer()
    generator = SignalGenerator()
    collector = CryptoDataCollector()
    signals_log = []

    # Fetch sample news data (which includes hidden sentiment for simulation)
    sample_news = collector.fetch_crypto_news(days_back=num_days)
    
    for news_item in sample_news:
        # Use the hidden sentiment or analyze content (using simulation)
        sentiment_result = analyzer.analyze_text(news_item['content'])
        
        # Generate signal based on sentiment
        signal = generator.generate_signal(sentiment_result)
        
        # Add timestamp to the signal (use news publication date)
        signal['timestamp'] = news_item['published_at']
        signals_log.append(signal)
        
    return signals_log

def create_sample_historical_data(num_days=30) -> pd.DataFrame:
    """Generate sample historical price data for backtesting."""
    start_date = datetime.now() - timedelta(days=num_days)
    end_date = datetime.now()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    prices = 100 + np.random.randn(len(dates)).cumsum() * 2 # More volatility
    historical_data = pd.DataFrame({'date': dates, 'close': prices})
    historical_data['close'] = historical_data['close'].clip(lower=1) # Ensure price > 0
    return historical_data

def process_historical_data(price_file_path: str, news_file_path: str, use_real_llm: bool = False) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Load and process historical data files for backtesting.
    
    Args:
        price_file_path (str): Path to the historical price data file
        news_file_path (str): Path to the historical news data file
        use_real_llm (bool): Whether to use the real LLM API for sentiment analysis
        
    Returns:
        Tuple[pd.DataFrame, List[Dict[str, Any]]]: Historical price data and generated signals
        
    Raises:
        ValueError: If there are issues with the data files
    """
    collector = CryptoDataCollector()
    analyzer = SentimentAnalyzer()
    generator = SignalGenerator()
    
    try:
        # Load historical price data
        historical_prices = collector.load_historical_data_from_file(price_file_path)
        
        # Load historical news data
        historical_news = collector.load_historical_crypto_news(news_file_path)
        
        # Process news articles to generate signals
        signals_log = []
        for news_item in historical_news:
            # Analyze sentiment
            if use_real_llm:
                # Use real LLM API for sentiment analysis
                sentiment_result = analyzer.analyze_sentiment_with_historical_data(news_item['content'])
            else:
                # Use simulated sentiment analysis
                sentiment_result = analyzer.analyze_text(news_item['content'])
            
            # Generate signal
            signal = generator.generate_signal(sentiment_result)
            
            # Add timestamp to the signal
            signal['timestamp'] = news_item['published_at']
            signals_log.append(signal)
        
        return historical_prices, signals_log
        
    except Exception as e:
        raise ValueError(f"Error processing historical data: {str(e)}")

def fetch_live_data(crypto_symbol: str, days_back: int = 30, fetch_news: bool = True) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Fetch live cryptocurrency data from APIs.
    
    Args:
        crypto_symbol (str): Symbol of the cryptocurrency to fetch data for
        days_back (int): Number of days of historical data to fetch
        fetch_news (bool): Whether to fetch news data
        
    Returns:
        Tuple[pd.DataFrame, List[Dict[str, Any]]]: Historical price data and news articles
        
    Raises:
        ConnectionError: If there's an issue connecting to the APIs
        ValueError: If the APIs return error responses
    """
    try:
        # Create live data directory if it doesn't exist
        live_data_dir = os.path.join(DATA_DIR, 'live_data')
        os.makedirs(live_data_dir, exist_ok=True)
        
        # Get live data from collector
        collector = CryptoDataCollector()
        
        # Format symbol for API call (BTC -> BTC/USD)
        symbol = f"{crypto_symbol}/USD" if '/' not in crypto_symbol else crypto_symbol
        
        # Fetch historical data
        df = collector.fetch_live_historical_data(symbol, days=days_back)
        
        news_data = []
        if fetch_news:
            # Fetch latest news
            news_df = collector.fetch_live_crypto_news(crypto_symbol.split('/')[0])
            if news_df is not None and not news_df.empty:
                # Save news to file
                safe_symbol = symbol.replace('/', '_')
                news_df.to_csv(os.path.join(live_data_dir, f'{safe_symbol}_news.csv'), index=False)
                news_data = news_df.to_dict('records')
        
        if df is not None and not df.empty:
            # Save price data to file
            safe_symbol = symbol.replace('/', '_')
            df.to_csv(os.path.join(live_data_dir, f'{safe_symbol}_latest.csv'), index=False)
            
            # Generate indicator plots
            generate_indicator_plots(df, symbol)
            
            return df, news_data
        else:
            raise ValueError(f"No data returned for {symbol}")
            
    except Exception as e:
        app.logger.error(f"Error fetching live data: {str(e)}")
        raise ConnectionError(f"Failed to fetch live data: {str(e)}")

def generate_indicator_plots(df, symbol):
    """
    Generate plots for technical indicators
    
    Args:
        df: DataFrame with price data
        symbol: Cryptocurrency symbol
    """
    try:
        # Create indicator plots directory if it doesn't exist
        indicator_plots_dir = os.path.join(app.static_folder, 'indicator_plots')
        os.makedirs(indicator_plots_dir, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # Plot 1: Price with Moving Averages
        df['date_idx'] = range(len(df))  # For x-axis
        axes[0].plot(df['date_idx'], df['close'], label='Close Price', color='blue')
        
        # Add MA lines
        ma_20 = calculate_moving_average(df['close'], 20)
        ma_50 = calculate_moving_average(df['close'], 50)
        axes[0].plot(df['date_idx'][-len(ma_20):], ma_20, label='MA (20)', color='orange')
        axes[0].plot(df['date_idx'][-len(ma_50):], ma_50, label='MA (50)', color='red')
        axes[0].set_title(f'{symbol} - Price and Moving Averages')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: RSI
        rsi = calculate_RSI(df['close'])
        axes[1].plot(df['date_idx'][-len(rsi):], rsi, color='purple')
        axes[1].axhline(y=70, color='red', linestyle='--')
        axes[1].axhline(y=30, color='green', linestyle='--')
        axes[1].set_title('RSI (14)')
        axes[1].set_ylim(0, 100)
        axes[1].grid(True)
        
        # Plot 3: MACD
        macd_dict = calculate_MACD(df['close'])
        macd = macd_dict['macd']
        signal = macd_dict['signal'] 
        histogram = macd_dict['histogram']
        axes[2].plot(df['date_idx'][-len(macd):], macd, label='MACD', color='blue')
        axes[2].plot(df['date_idx'][-len(signal):], signal, label='Signal', color='red')
        axes[2].bar(df['date_idx'][-len(histogram):], histogram, color=['green' if x > 0 else 'red' for x in histogram], label='Histogram')
        axes[2].set_title('MACD')
        axes[2].legend()
        axes[2].grid(True)
        
        # Plot 4: Bollinger Bands
        bollinger_dict = calculate_bollinger_bands(df['close'])
        upper = bollinger_dict['upper']
        middle = bollinger_dict['middle']
        lower = bollinger_dict['lower']
        axes[3].plot(df['date_idx'], df['close'], label='Close Price', color='blue')
        axes[3].plot(df['date_idx'][-len(upper):], upper, label='Upper Band', color='red', linestyle='--')
        axes[3].plot(df['date_idx'][-len(middle):], middle, label='Middle Band', color='orange')
        axes[3].plot(df['date_idx'][-len(lower):], lower, label='Lower Band', color='green', linestyle='--')
        axes[3].set_title('Bollinger Bands')
        axes[3].legend()
        axes[3].grid(True)
        
        # Format x-axis with actual dates
        date_labels = df['timestamp'].dt.strftime('%Y-%m-%d') if 'timestamp' in df.columns else [f'Day {i}' for i in range(len(df))]
        plt.xticks(df['date_idx'][::max(1, len(df)//10)], date_labels[::max(1, len(df)//10)], rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        safe_symbol = symbol.replace('/', '_')
        plot_path = os.path.join(indicator_plots_dir, f'{safe_symbol}_indicators.png')
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
        
    except Exception as e:
        app.logger.error(f"Error generating indicator plots: {str(e)}")
        return None

def crypto_to_name(symbol: str) -> str:
    """Convert a crypto symbol to its full name for searching news."""
    crypto_map = {
        'BTC': 'Bitcoin',
        'ETH': 'Ethereum',
        'XRP': 'Ripple',
        'ADA': 'Cardano',
        'SOL': 'Solana',
        'DOT': 'Polkadot',
        'DOGE': 'Dogecoin',
        'AVAX': 'Avalanche',
        'MATIC': 'Polygon'
    }
    return crypto_map.get(symbol.upper(), symbol)

# --- Routes ---
@app.route('/')
def index():
    """Main dashboard view showing trading data and signals."""
    global MODE
    
    crypto_symbol = request.args.get('crypto_symbol', DEFAULT_CRYPTO_SYMBOL)
    days_back = int(request.args.get('days_back', DEFAULT_DAYS_BACK))
    
    signals_log = []
    performance_metrics = {}
    indicator_plot_path = None
    risk_metrics = {}
    last_updated = None
    
    try:
        # Demo mode - uses Dogecoin data
        if MODE == 'demo':
            # Load Dogecoin data
            doge_data = load_dogecoin_data()
            
            # Generate signals, buy/sell points, and timestamps for the full dataset
            all_signals, all_buy_signals, all_sell_signals, all_timestamps = generate_simulation_signals(doge_data)
            
            
            # Determine initial visible data range
            initial_visible_days = min(days_back, len(doge_data))
            initial_visible_data = doge_data.iloc[:initial_visible_days]
            initial_visible_signals = [s for s in all_signals if datetime.strptime(s['timestamp'], '%Y-%m-%d').date() <= initial_visible_data['date'].iloc[-1].date()]

            # Initial placeholder performance metrics
            initial_portfolio = 100000.0
            performance_metrics = {
                'portfolio_value': initial_portfolio,
                'total_return': 0,  # Initial return is 0
                'max_drawdown': 0,  # Initial drawdown is 0
                'trade_count': len(initial_visible_signals), # Initial count based on visible signals
                'volatility': 0,
                'sharpe_ratio': 0
            }
            
            # Placeholder risk metrics for initial view
            risk_metrics = {
                'sharpe_ratio': 0, 'sortino_ratio': 0, 'volatility': 0,
                'max_drawdown': 0, 'calmar_ratio': 0, 'var_95': 0
            }

            # --- Load Musk Tweet Data ---
            musk_tweets_data = load_musk_tweets(MUSK_TWEET_DATA_PATH)
            # --------------------------

            # Pass full data needed for JS simulation, but only initial signals for display
            return render_template(
                'index.html',
                mode=MODE,
                crypto_symbol=crypto_symbol,
                days_back=initial_visible_days, # How many days are initially shown
                signals=initial_visible_signals[:10],  # Show only first 10 signals initially
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                last_updated=initial_visible_data['date'].iloc[-1].strftime('%Y-%m-%d'),
                
                # Data for JavaScript simulation - PASS PYTHON OBJECTS DIRECTLY
                price_data=doge_data['close'].tolist(), # Pass list directly
                timestamps=all_timestamps, # Pass list directly
                buy_signals=all_buy_signals, # Pass list directly
                sell_signals=all_sell_signals, # Pass list directly
                musk_tweets_data=musk_tweets_data, # Pass tweet data
                
                # Demo control parameters
                demo_interval=1000,  # Convert to milliseconds
                demo_start_date=doge_data['date'].iloc[0].strftime('%Y-%m-%d'),
                demo_end_date=doge_data['date'].iloc[-1].strftime('%Y-%m-%d'),
                demo_data_days=len(doge_data) # Total days in dataset
            )
            
        elif MODE == 'live':
            # Live data mode
            try:
                # Fetch live data
                price_data, news_data = fetch_live_data(crypto_symbol, days_back)
                
                # Process signals from news
                analyzer = SentimentAnalyzer()
                generator = SignalGenerator()
                
                signals_log = []
                for news_item in news_data:
                    # Analyze sentiment
                    sentiment_result = analyzer.analyze_text(news_item.get('content', news_item.get('title', '')))
                    
                    # Generate signal
                    signal = generator.generate_signal(sentiment_result)
                    
                    # Add timestamp
                    signal['timestamp'] = news_item.get('publishedAt', news_item.get('published_at', datetime.now().isoformat()))
                    signals_log.append(signal)
                
                # Get the last updated timestamp
                if not price_data.empty:
                    last_updated = price_data['timestamp'].iloc[-1]
                
                # Calculate indicators
                with_indicators = price_data.copy()
                with_indicators['returns'] = calculate_returns(with_indicators['close'])
                
                # Generate indicator plots
                indicator_plot_path = generate_indicator_plots(price_data, crypto_symbol)
                
                # Calculate risk metrics
                risk_metrics = calculate_all_risk_metrics(with_indicators['returns'], with_indicators['close'])
                
                # Add some performance metrics for display
                performance_metrics = {
                    'portfolio_value': price_data['close'].iloc[-1],
                    'total_return': (price_data['close'].iloc[-1] / price_data['close'].iloc[0] - 1) * 100 if len(price_data) > 1 else 0,
                    'volatility': risk_metrics.get('volatility', 0) * 100,
                    'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
                    'max_drawdown': risk_metrics.get('max_drawdown', 0) * 100,
                    'trades': len(signals_log)
                }
                
            except Exception as e:
                flash(f"Error fetching live data: {str(e)}", "error")
                # Fall back to historical mode
                MODE = 'historical'
                
        # Historical mode (default or fallback)
        if MODE == 'historical':
            # Get historical data (from simulation or uploaded files)
            historical_data = create_sample_historical_data(days_back)
            signals_log = generate_sample_sentiment_and_signals(days_back)
            
            # Calculate indicators
            with_indicators = historical_data.copy()
            with_indicators['returns'] = calculate_returns(with_indicators['close'])
            
            # Generate indicator plots
            indicator_plot_path = generate_indicator_plots(historical_data, crypto_symbol)
            
            # Filter signals to only include buy/sell for the backtest
            backtest_signals = [s for s in signals_log if s['signal'] in ['buy', 'sell']]
            
            # Run backtest
            backtest_engine = BacktestEngine()
            performance_metrics = backtest_engine.run_backtest(
                backtest_signals, 
                historical_data
            )
            
            # Add missing fields expected by the template
            performance_metrics['portfolio_value'] = performance_metrics.get('final_portfolio_value', 0)
            
            # Calculate risk metrics
            risk_metrics = calculate_all_risk_metrics(with_indicators['returns'], with_indicators['close'])
            
        # Get indicator plots path relative to static folder
        if indicator_plot_path:
            indicator_plot_path = indicator_plot_path.replace(app.static_folder, '').lstrip('/')
        
        # Render the dashboard template with data
        return render_template(
            'index.html',
            mode=MODE,
            crypto_symbol=crypto_symbol,
            days_back=days_back,
            signals=signals_log[:10],  # Show most recent 10 signals
            performance_metrics=performance_metrics,
            last_updated=last_updated,
            indicator_plot_path=indicator_plot_path,
            risk_metrics=risk_metrics,
        )
        
    except Exception as e:
        traceback.print_exc()
        error_message = f"Error loading dashboard: {str(e)}"
        return render_template('error.html', error=error_message)

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    """
    Toggle between historical, live, and demo data modes
    """
    global MODE
    
    # Circular toggle between modes
    if MODE == 'historical':
        MODE = 'live'
        # Initialize live data if needed
        if not os.path.exists(os.path.join(DATA_DIR, 'live_data')):
            os.makedirs(os.path.join(DATA_DIR, 'live_data'), exist_ok=True)
    elif MODE == 'live':
        MODE = 'demo'
        # Stop any existing simulation
        stop_simulation()
    else:  # demo mode
        MODE = 'historical'
        # Stop simulation when switching away from demo mode
        stop_simulation()
    
    return redirect(url_for('index'))

@app.route('/start_simulation', methods=['POST'])
def start_simulation_route():
    """
    Start the demo simulation
    """
    if MODE != 'demo':
        flash('Simulation can only be started in demo mode', 'error')
        return jsonify({'success': False, 'error': 'Not in demo mode'})
    
    success = start_simulation()
    
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Simulation already running'})

@app.route('/stop_simulation', methods=['POST'])
def stop_simulation_route():
    """
    Stop the running simulation
    """
    success = stop_simulation()
    
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'No simulation running'})

@app.route('/get_simulation_data')
def get_simulation_data():
    """
    API endpoint to get current simulation data
    """
    if MODE != 'demo':
        return jsonify({'success': False, 'error': 'Not in demo mode'})
    
    day_index = simulation_data['current_day_index']
    
    if day_index >= len(simulation_data['price_data']):
        return jsonify({'success': False, 'error': 'Simulation completed'})
    
    return jsonify({
        'success': True,
        'current_day': day_index,
        'total_days': len(simulation_data['price_data']),
        'timestamp': simulation_data['price_data']['date'].iloc[day_index].strftime('%Y-%m-%d'),
        'price': float(simulation_data['price_data']['close'].iloc[day_index]),
        'portfolio_value': float(simulation_data['portfolio_value']),
        'recent_signals': [s for s in simulation_data['signals'] 
                           if datetime.strptime(s['timestamp'], '%Y-%m-%d').date() <= 
                              simulation_data['price_data']['date'].iloc[day_index].date()][-5:]
    })

@app.route('/api/signals')
def get_signals():
    """API endpoint to get trading signals"""
    _, signals = generate_sample_sentiment_and_signals()
    return jsonify(signals)

@app.route('/api/backtest_results')
def get_backtest_results():
    """API endpoint to get backtest results"""
    # This would typically be fetched from a database or computed on demand
    results = {
        'final_portfolio_value': 110250.75,
        'total_return': 0.1025,
        'max_drawdown': 0.05,
        'trade_count': 12
    }
    return jsonify(results)

@app.route('/api/sentiment_data')
def get_sentiment_data():
    """API endpoint to get sentiment data"""
    sentiment_data, _ = generate_sample_sentiment_and_signals()
    return jsonify(sentiment_data)

@app.route('/demo_update')
def demo_update():
    """API endpoint to get the next day's data for demo mode time simulation."""
    if MODE != 'demo':
        return jsonify({'error': 'Not in demo mode'}), 400
    
    try:
        # Get the current day index from the request
        current_index = int(request.args.get('day', 0))
        
        # Load the Dogecoin data
        doge_data = load_dogecoin_data()
        
        # Check if we've reached the end of the data
        if current_index >= len(doge_data) - 1:
            return jsonify({
                'completed': True,
                'message': 'Simulation complete'
            })
        
        # Get the next day's data
        next_index = current_index + 1
        next_day = doge_data.iloc[next_index]
        
        # Generate signals up to this point
        signals, buy_signals, sell_signals, timestamps = generate_simulation_signals(doge_data.iloc[:next_index+1])
        
        # Calculate portfolio value (simplified simulation)
        initial_value = 100000.0
        portfolio_value = initial_value * (1 + (next_day['close'] / doge_data.iloc[0]['close'] - 1) * 0.8)
        
        # Calculate return metrics
        total_return = (portfolio_value / initial_value - 1) * 100
        
        # Find if there's a signal for this day
        day_signal = None
        for signal in signals:
            if signal['timestamp'] == next_day['date'].strftime('%Y-%m-%d'):
                day_signal = signal
                break
        
        # Check if this is a buy or sell point
        is_buy_point = any(bs['x'] == timestamps[next_index] for bs in buy_signals)
        is_sell_point = any(ss['x'] == timestamps[next_index] for ss in sell_signals)
        
        return jsonify({
            'completed': False,
            'timestamp': next_day['date'].strftime('%Y-%m-%d'),
            'price': float(next_day['close']),
            'portfolio_value': float(portfolio_value),
            'total_return': float(total_return),
            'signal': day_signal,
            'is_buy_point': is_buy_point,
            'is_sell_point': is_sell_point,
            'day_index': next_index,
            'total_days': len(doge_data)
        })
    
    except Exception as e:
        app.logger.error(f"Error in demo update: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --- Socket.IO event handlers ---
@socketio.on('connect')
def handle_connect():
    """Handle client connection to websocket."""
    # Start simulation if in demo mode and not already running
    if MODE == 'demo':
        start_simulation()

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    # Only stop the simulation if we're in demo mode and this is the last client
    if MODE == 'demo' and len(socketio.server.eio.sockets) <= 1:
        stop_simulation()

# --- Main Entry Point ---
if __name__ == "__main__":
    # Ensure necessary directories exist
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    if not os.path.exists(INDICATOR_PLOT_DIR):
        os.makedirs(INDICATOR_PLOT_DIR)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Check if doge data exists
    if not os.path.exists(DOGE_DATA_PATH):
        print(f"Warning: Dogecoin data file not found at {DOGE_DATA_PATH}")
        print("Demo mode will use generated data instead")
        
    # Run the Flask development server with Socket.IO
    socketio.run(app, debug=True)