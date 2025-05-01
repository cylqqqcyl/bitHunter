#!/usr/bin/env python
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Add the parent directory to the path to allow importing from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our modules
from data_ingestion.news_fetcher import NewsFetcher
from data_ingestion.price_fetcher import PriceFetcher
from sentiment_analysis.analyzer import SentimentAnalyzer
from signal_generation.generator import SignalGenerator
from backtesting.backtest import BacktestEngine

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_historical_data():
    """Load historical price data from file."""
    try:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "data", "historical_prices.csv")
        return pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return None

def load_news_data():
    """Load historical news data from file."""
    try:
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "data", "historical_news.json")
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading news data: {e}")
        return None

def format_timestamp(timestamp):
    """Format timestamp for display."""
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return timestamp
    return timestamp

def run_demo():
    print("=" * 80)
    print("BitHunter - Crypto Trading Demo")
    print("=" * 80)
    
    # Step 1: Load historical data
    print("\n[1] Loading historical price data...")
    historical_data = load_historical_data()
    if historical_data is None:
        print("Failed to load historical data. Exiting.")
        return
    
    print(f"Loaded price data from {historical_data.index.min()} to {historical_data.index.max()}")
    print(f"Sample data:\n{historical_data.head()}")
    
    # Step 2: Load news data
    print("\n[2] Loading historical news data...")
    news_data = load_news_data()
    if news_data is None:
        print("Failed to load news data. Exiting.")
        return
    
    print(f"Loaded {len(news_data)} news articles")
    print(f"Sample article: {news_data[0]['title']}")
    
    # Step 3: Analyze sentiment
    print("\n[3] Analyzing sentiment for news articles...")
    analyzer = SentimentAnalyzer()
    
    sentiment_results = {}
    for article in news_data[:10]:  # Process only first 10 for demo
        timestamp = article.get('published_at', article.get('publishedAt', str(datetime.now())))
        sentiment = analyzer.analyze_text(article['title'] + ". " + article.get('description', ''))
        date_key = format_timestamp(timestamp)
        
        if date_key not in sentiment_results:
            sentiment_results[date_key] = []
        
        sentiment_results[date_key].append({
            "title": article['title'],
            "sentiment_score": sentiment['compound'],
            "sentiment_label": "positive" if sentiment['compound'] > 0.05 else 
                              "negative" if sentiment['compound'] < -0.05 else "neutral"
        })
    
    print(f"Analyzed sentiment for {sum(len(v) for v in sentiment_results.values())} articles")
    for date, sentiments in list(sentiment_results.items())[:3]:
        avg_score = sum(item['sentiment_score'] for item in sentiments) / len(sentiments)
        print(f"  {date}: {len(sentiments)} articles, avg score: {avg_score:.2f}")
    
    # Step 4: Generate trading signals
    print("\n[4] Generating trading signals...")
    generator = SignalGenerator()
    
    signals = []
    for date, sentiments in sentiment_results.items():
        # Calculate average sentiment for this date
        scores = [item['sentiment_score'] for item in sentiments]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Prepare input for signal generator
        sentiment_input = {
            "timestamp": date,
            "overall_score": avg_score,
            "confidence": 0.7 if len(scores) > 3 else 0.5,
            "article_count": len(scores),
            "keywords": ["crypto", "bitcoin", "market"]
        }
        
        signal = generator.generate_signal(sentiment_input)
        signal['timestamp'] = date
        signals.append(signal)
    
    # Print some sample signals
    print(f"Generated {len(signals)} trading signals")
    for signal in signals[:3]:
        print(f"  {signal['timestamp']}: {signal['signal']} (conf: {signal['confidence']:.2f})")
    
    # Step 5: Run backtest
    print("\n[5] Running backtest simulation...")
    
    # Prepare historical data in the required format
    backtest_data = historical_data[['close']].copy()
    backtest_data.columns = ['price']
    
    # Create backtest engine
    engine = BacktestEngine(initial_capital=10000, commission_rate=0.001)
    
    # Convert signals to the format expected by backtest engine
    backtest_signals = []
    for signal in signals:
        if signal['signal'] in ['buy', 'sell']:
            try:
                # Try to parse the timestamp string
                date = datetime.strptime(signal['timestamp'], "%Y-%m-%d %H:%M")
            except:
                # If parsing fails, use the string as is (the backtest engine will handle this)
                date = signal['timestamp']
                
            backtest_signals.append({
                'date': date,
                'action': signal['signal'],
                'confidence': signal['confidence']
            })
    
    # Run backtest
    results = engine.run_backtest(backtest_data, backtest_signals)
    
    # Plot and save results
    plot_file = os.path.join(OUTPUT_DIR, f"backtest_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    engine.plot_performance(save_to=plot_file)
    
    # Step 6: Display results summary
    print("\n[6] Backtest Results Summary:")
    print(f"  Initial Capital: ${engine.initial_capital:.2f}")
    print(f"  Final Portfolio Value: ${engine.portfolio_value:.2f}")
    print(f"  Total Return: {((engine.portfolio_value/engine.initial_capital)-1)*100:.2f}%")
    print(f"  Maximum Drawdown: {engine.max_drawdown*100:.2f}%")
    print(f"  Number of Trades: {engine.trade_count}")
    print(f"  Performance plot saved to: {plot_file}")
    
    # Save results to JSON for potential dashboard display
    results_file = os.path.join(OUTPUT_DIR, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    results_data = {
        "initial_capital": engine.initial_capital,
        "final_value": engine.portfolio_value,
        "total_return": ((engine.portfolio_value/engine.initial_capital)-1)*100,
        "max_drawdown": engine.max_drawdown*100,
        "trade_count": engine.trade_count,
        "trades": engine.trades,
        "performance_plot": plot_file,
        "signals": signals
    }
    
    with open(results_file, 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        json_data = {
            **results_data,
            "trades": [
                {**trade, "date": str(trade["date"])} 
                for trade in results_data["trades"]
            ]
        }
        json.dump(json_data, f, indent=2)
    
    print(f"  Results data saved to: {results_file}")
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    run_demo() 