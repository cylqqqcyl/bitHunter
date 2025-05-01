#!/usr/bin/env python
"""
Demo API for Dogecoin Price Simulation

This standalone Flask API provides an endpoint for simulating day-by-day
time elapsed Dogecoin price data for the BitHunter dashboard demo mode.
"""

import os
import sys
import pandas as pd
import numpy as np
import random
from datetime import datetime
from flask import Flask, jsonify, request

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# Path to Dogecoin data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DOGE_DATA_PATH = os.path.join(DATA_DIR, "previous_historical_data", "doge-usd.csv")

# Global variables to store cached data
doge_data = None
signals = None
buy_signals = None
sell_signals = None
timestamps = None

def load_dogecoin_data():
    """Load and cache Dogecoin historical data."""
    global doge_data
    
    if doge_data is not None:
        return doge_data
    
    try:
        df = pd.read_csv(DOGE_DATA_PATH)
        df['date'] = pd.to_datetime(df['snapped_at'])
        df['close'] = df['price']
        df = df[['date', 'close', 'market_cap', 'total_volume']]
        df = df.sort_values('date')
        doge_data = df
        return df
    except Exception as e:
        print(f"Error loading Dogecoin data: {e}")
        # Generate fallback data
        return generate_sample_data()

def generate_sample_data(days=365):
    """Generate sample data in case the real data can't be loaded."""
    start_date = datetime.now().replace(year=datetime.now().year - 1)
    dates = pd.date_range(start=start_date, periods=days)
    
    # Generate a price series with some realistic patterns
    base_price = 0.1  # Starting price
    price = base_price
    prices = []
    for i in range(days):
        # Add some random daily variation
        daily_change = np.random.normal(0, 0.005)  # Mean 0, std dev 0.5%
        
        # Add occasional jumps (both positive and negative)
        if random.random() < 0.02:  # 2% chance of a jump
            jump = np.random.normal(0, 0.05)  # Mean 0, std dev 5%
            daily_change += jump
        
        # Add a trend component
        trend = 0.0001 * np.sin(i / 30)  # Cyclical trend
        
        # Update price
        price *= (1 + daily_change + trend)
        price = max(0.01, price)  # Ensure price doesn't go too low
        prices.append(price)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'market_cap': [p * 1e11 for p in prices],  # Simulated market cap
        'total_volume': [p * 1e9 * (0.8 + 0.4 * random.random()) for p in prices]  # Simulated volume
    })
    
    return df

def generate_signals(price_data, signal_frequency=0.15):
    """Generate trading signals based on price movements."""
    global signals, buy_signals, sell_signals, timestamps
    
    if signals is not None and len(signals) > 0:
        return signals, buy_signals, sell_signals, timestamps
    
    signals = []
    buy_signals = []
    sell_signals = []
    timestamps = []
    
    for i in range(len(price_data)):
        # Convert date to timestamp for the chart
        timestamp = int(pd.Timestamp(price_data['date'].iloc[i]).timestamp() * 1000)
        timestamps.append(timestamp)
        
        # Skip the first data point as we need previous price for comparison
        if i == 0:
            continue
            
        # Calculate price change
        price_change = (price_data['close'].iloc[i] - price_data['close'].iloc[i-1]) / price_data['close'].iloc[i-1]
        
        # Adjust signal probability based on price movement
        adjusted_probability = signal_frequency + abs(price_change) * 5
        
        # Determine if this day has a signal
        if random.random() < adjusted_probability:
            signal_type = None
            
            # Determine signal type based on price movement and some randomness
            if price_change > 0.01 or (price_change > 0 and random.random() > 0.3):
                signal_type = 'buy'
                buy_signals.append({
                    'x': timestamp,
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
                    'x': timestamp,
                    'y': price_data['close'].iloc[i],
                    'marker': {
                        'size': 8,
                        'fillColor': '#FF4560',
                        'strokeColor': '#FF4560'
                    }
                })
            
            if signal_type:
                # Generate detailed signal information
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
                        'keywords': generate_keywords(signal_type)
                    }
                })
    
    return signals, buy_signals, sell_signals, timestamps

def generate_keywords(signal_type):
    """Generate relevant keywords based on signal type."""
    bullish_keywords = ['rally', 'adoption', 'growth', 'partnership', 'upgrade', 'development', 'breakthrough']
    bearish_keywords = ['dip', 'regulation', 'sell-off', 'correction', 'risk', 'volatility', 'concerns']
    neutral_keywords = ['market', 'trading', 'crypto', 'blockchain', 'technology', 'volume', 'trend']
    
    if signal_type == 'buy':
        pool = bullish_keywords + random.sample(neutral_keywords, 2)
    elif signal_type == 'sell':
        pool = bearish_keywords + random.sample(neutral_keywords, 2)
    else:
        pool = neutral_keywords
    
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
    
    for signal in relevant_signals:
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

@app.route('/demo_update')
def demo_update():
    """API endpoint to get the next day's data for the simulation."""
    try:
        # Get current day index from request
        day_index = int(request.args.get('day', 0))
        
        # Load data if not already loaded
        price_data = load_dogecoin_data()
        
        # Generate signals if not already generated
        all_signals, all_buy_signals, all_sell_signals, all_timestamps = generate_signals(price_data)
        
        # Check if we've reached the end of the data
        if day_index >= len(price_data) - 1:
            return jsonify({
                'completed': True,
                'message': 'Simulation complete'
            })
        
        # Get the next day's data
        next_index = day_index + 1
        next_day = price_data.iloc[next_index]
        
        # Calculate portfolio value
        initial_capital = 100000.0
        portfolio_value = calculate_portfolio_value(initial_capital, price_data, all_signals, next_index)
        
        # Calculate return metrics
        total_return = (portfolio_value / initial_capital - 1) * 100
        
        # Find if there's a signal for this day
        day_signal = None
        for signal in all_signals:
            if signal['timestamp'] == next_day['date'].strftime('%Y-%m-%d'):
                day_signal = signal
                break
        
        # Check if this is a buy or sell point
        is_buy_point = any(bs['x'] == all_timestamps[next_index] for bs in all_buy_signals)
        is_sell_point = any(ss['x'] == all_timestamps[next_index] for ss in all_sell_signals)
        
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
            'total_days': len(price_data)
        })
    
    except Exception as e:
        print(f"Error in demo update: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/demo_init')
def demo_init():
    """API endpoint to get initial data for the simulation."""
    try:
        # Load data
        price_data = load_dogecoin_data()
        
        # Generate signals
        all_signals, all_buy_signals, all_sell_signals, all_timestamps = generate_signals(price_data)
        
        # Get initial days range (default to 30 days or the parameter)
        days = int(request.args.get('days', 30))
        initial_days = min(days, len(price_data))
        
        return jsonify({
            'start_date': price_data['date'].iloc[0].strftime('%Y-%m-%d'),
            'end_date': price_data['date'].iloc[-1].strftime('%Y-%m-%d'),
            'total_days': len(price_data),
            'visible_days': initial_days,
            'price_data': price_data['close'].iloc[:initial_days].tolist(),
            'timestamps': all_timestamps[:initial_days],
            'buy_signals': [bs for bs in all_buy_signals if bs['x'] <= all_timestamps[initial_days-1]],
            'sell_signals': [ss for ss in all_sell_signals if ss['x'] <= all_timestamps[initial_days-1]],
            'initial_signals': [s for s in all_signals if datetime.strptime(s['timestamp'], '%Y-%m-%d').date() <= price_data['date'].iloc[initial_days-1].date()]
        })
    
    except Exception as e:
        print(f"Error in demo init: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Check if data file exists
    if not os.path.exists(DOGE_DATA_PATH):
        print(f"Warning: Dogecoin data file not found at {DOGE_DATA_PATH}")
        print("Demo will use generated data instead")
    
    # Run the standalone API
    app.run(debug=True, port=5001) 