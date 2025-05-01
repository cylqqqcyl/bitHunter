"""
bitHunter - Cryptocurrency LLM Trading Agent

This is the main entry point for the bitHunter application.
"""

import os
import argparse
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Import modules
from data_ingestion.collector import CryptoDataCollector
from llm_sentiment.analyzer import SentimentAnalyzer
from signal_generation.generator import SignalGenerator
from backtesting.backtest import BacktestEngine
from dashboard.app import app as dashboard_app

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="bitHunter - Cryptocurrency LLM Trading Agent")
    
    # Mode selection
    parser.add_argument('--mode', choices=['collect', 'analyze', 'generate', 'backtest', 'dashboard', 'pipeline'],
                        default='dashboard', help='Operation mode')
    
    # Data collection options
    parser.add_argument('--symbol', type=str, default='BTC',
                        help='Cryptocurrency symbol (default: BTC)')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to look back for data (default: 30)')
    
    # Backtest options
    parser.add_argument('--capital', type=float, default=10000,
                        help='Initial capital for backtesting (default: 10000)')
    
    # Dashboard options
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for the dashboard server (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                        help='Run dashboard in debug mode')
    
    return parser.parse_args()

def run_data_collection(args):
    """Run the data collection module."""
    print(f"Collecting data for {args.symbol} from the last {args.days} days...")
    
    collector = CryptoDataCollector()
    
    # Calculate start date
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    # Collect price data
    price_data = collector.fetch_price_data(args.symbol, start_date)
    collector.save_data(price_data, 'price')
    
    # Collect news data
    news_data = collector.fetch_news_data([args.symbol, 'cryptocurrency'], args.days)
    collector.save_data(news_data, 'news')
    
    print("Data collection completed.")

def run_sentiment_analysis(args):
    """Run the sentiment analysis module."""
    print(f"Analyzing sentiment for {args.symbol}...")
    
    # This would load data from files in a real implementation
    analyzer = SentimentAnalyzer()
    
    # Placeholder for demo
    sample_texts = [
        f"{args.symbol} price surges 5% amid positive market sentiment",
        f"Analysts predict bearish outlook for {args.symbol} in coming weeks",
        f"New regulations may impact {args.symbol} trading volumes"
    ]
    
    results = analyzer.analyze_batch(sample_texts)
    aggregate = analyzer.aggregate_sentiment(results)
    
    print(f"Sentiment analysis completed: {aggregate['overall_sentiment']}")

def run_signal_generation(args):
    """Run the signal generation module."""
    print(f"Generating trading signals for {args.symbol}...")
    
    # This would load sentiment data from files in a real implementation
    generator = SignalGenerator()
    
    # Placeholder for demo
    sample_sentiment = {
        "overall_sentiment": "bullish",
        "confidence": 7,
        "sample_size": 10
    }
    
    signal = generator.generate_signal(sample_sentiment)
    
    print(f"Signal generated: {signal['action']} with strength {signal['strength']}")

def run_backtest(args):
    """Run the backtesting module."""
    print(f"Running backtest for {args.symbol} with {args.capital} initial capital...")
    
    # This would load signals and price data from files in a real implementation
    backtest = BacktestEngine(initial_capital=args.capital)
    
    # Placeholder for demo
    import pandas as pd
    import numpy as np
    
    # Create dummy price data
    dates = pd.date_range(start=datetime.now() - timedelta(days=args.days), periods=args.days)
    close_prices = np.random.normal(10000, 500, args.days)  # Random prices around 10000
    price_data = pd.DataFrame({
        'timestamp': dates,
        'close': close_prices
    })
    
    # Create dummy signals
    signals = []
    for i in range(5):  # 5 random signals
        day = np.random.randint(0, args.days)
        action = np.random.choice(['buy', 'sell'])
        signals.append({
            'timestamp': dates[day].isoformat(),
            'action': action,
            'strength': np.random.random() * 0.5 + 0.5  # Random strength between 0.5 and 1.0
        })
    
    results = backtest.run_backtest(price_data, signals)
    
    print(f"Backtest completed. ROI: {results['roi']:.2%}")

def run_dashboard(args):
    """Run the dashboard application."""
    print(f"Starting dashboard on port {args.port}...")
    dashboard_app.run(host='0.0.0.0', port=args.port, debug=args.debug)

def run_full_pipeline(args):
    """Run the full pipeline from data collection to signal generation."""
    print(f"Running full pipeline for {args.symbol}...")
    
    # Run each module in sequence
    run_data_collection(args)
    run_sentiment_analysis(args)
    run_signal_generation(args)
    run_backtest(args)
    
    print("Pipeline completed successfully.")

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    # Run the selected mode
    if args.mode == 'collect':
        run_data_collection(args)
    elif args.mode == 'analyze':
        run_sentiment_analysis(args)
    elif args.mode == 'generate':
        run_signal_generation(args)
    elif args.mode == 'backtest':
        run_backtest(args)
    elif args.mode == 'dashboard':
        run_dashboard(args)
    elif args.mode == 'pipeline':
        run_full_pipeline(args)

if __name__ == "__main__":
    main() 