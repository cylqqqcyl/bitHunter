#!/usr/bin/env python
"""
BitHunter Real-Time Trading Simulation
======================================

This script demonstrates a real-time simulation of the BitHunter trading system,
integrating data ingestion, sentiment analysis, signal generation, and backtesting
in a cohesive pipeline.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import random

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import BitHunter modules
from data_ingestion.collector import DataCollector
from sentiment_analysis.analyzer import SentimentAnalyzer
from signal_generation.generator import SignalGenerator
from backtesting.backtest import BacktestEngine

class RealTimeSimulation:
    """
    Simulates a real-time trading environment by processing historical data
    in sequential chunks as if it were streaming in real time.
    """
    
    def __init__(self, historical_prices_path, historical_news_path, 
                 initial_capital=100000.0, commission_rate=0.001,
                 data_window_days=5, simulation_delay=1.0):
        """
        Initialize the real-time simulation.
        
        Args:
            historical_prices_path (str): Path to historical price CSV file
            historical_news_path (str): Path to historical news JSON file
            initial_capital (float): Initial capital for backtesting
            commission_rate (float): Commission rate for trades
            data_window_days (int): Number of days of data to process in each step
            simulation_delay (float): Delay between steps to simulate real-time (seconds)
        """
        self.historical_prices_path = historical_prices_path
        self.historical_news_path = historical_news_path
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.data_window_days = data_window_days
        self.simulation_delay = simulation_delay
        
        # Initialize components
        self.data_collector = DataCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.signal_generator = SignalGenerator()
        self.backtest_engine = BacktestEngine(
            initial_capital=initial_capital,
            commission_rate=commission_rate
        )
        
        # Results storage
        self.portfolio_values = []
        self.signals_generated = []
        self.sentiment_results = []
        
        # Setup output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load historical price and news data."""
        print("Loading historical data...")
        
        # Load historical prices
        self.prices_df = pd.read_csv(self.historical_prices_path)
        self.prices_df['date'] = pd.to_datetime(self.prices_df['date'])
        
        # Load historical news
        with open(self.historical_news_path, 'r') as f:
            self.news_data = json.load(f)
        
        # Convert news timestamps to datetime
        for news in self.news_data:
            news['published_at'] = datetime.strptime(
                news['published_at'], '%Y-%m-%dT%H:%M:%S'
            )
        
        print(f"Loaded {len(self.prices_df)} price points and {len(self.news_data)} news items")
        
    def run_simulation(self):
        """Run the real-time trading simulation."""
        if not hasattr(self, 'prices_df') or not hasattr(self, 'news_data'):
            self.load_data()
            
        # Get the date range from the price data
        start_date = self.prices_df['date'].min()
        end_date = self.prices_df['date'].max()
        
        print(f"\n{'='*80}")
        print(f"Starting BitHunter Real-Time Trading Simulation")
        print(f"Simulation period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"{'='*80}\n")
        
        # Initialize simulation variables
        current_date = start_date
        step_counter = 1
        cumulative_signals = []
        
        # Reset the backtest engine
        self.backtest_engine.reset()
        
        # Simulation loop
        while current_date <= end_date:
            next_date = current_date + timedelta(days=self.data_window_days)
            
            print(f"\n--- Simulation Step {step_counter} ---")
            print(f"Processing data from {current_date.strftime('%Y-%m-%d')} to {next_date.strftime('%Y-%m-%d')}")
            
            # Step 1: Get price data for the current window
            current_prices = self.prices_df[
                (self.prices_df['date'] >= current_date) & 
                (self.prices_df['date'] < next_date)
            ]
            
            if len(current_prices) == 0:
                # Skip if no price data for this window
                current_date = next_date
                step_counter += 1
                continue
            
            print(f"Price data: {len(current_prices)} data points")
            
            # Step 2: Get news articles for the current window
            current_news = [
                news for news in self.news_data 
                if current_date <= news['published_at'] < next_date
            ]
            
            print(f"News data: {len(current_news)} articles")
            
            # Step 3: Perform sentiment analysis
            if current_news:
                news_texts = [news['content'] for news in current_news]
                sentiment_results = self.sentiment_analyzer.analyze_batch(news_texts)
                aggregate_sentiment = self.sentiment_analyzer.aggregate_sentiment(sentiment_results)
                self.sentiment_results.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'sentiment': aggregate_sentiment
                })
                print(f"Sentiment analysis: {aggregate_sentiment['overall_sentiment']} (confidence: {aggregate_sentiment['confidence']})")
            else:
                # If no news, use neutral sentiment
                aggregate_sentiment = {
                    'overall_sentiment': 'neutral',
                    'confidence': 5,
                    'sample_size': 0
                }
                print("No news articles found - using neutral sentiment")
            
            # Step 4: Generate trading signals
            signal = self.signal_generator.generate_signal(aggregate_sentiment)
            
            # Add timestamp to the signal
            signal['timestamp'] = current_date.strftime('%Y-%m-%d')
            self.signals_generated.append(signal)
            
            # Only process if we have an actionable signal (buy or sell)
            if signal['signal'] in ['buy', 'sell']:
                # Format signal for backtesting
                backtest_signal = {
                    'timestamp': current_date.strftime('%Y-%m-%d'),
                    'signal': signal['signal'],
                    'confidence': signal['confidence']
                }
                cumulative_signals.append(backtest_signal)
                
                print(f"Generated {signal['signal'].upper()} signal with {signal['confidence']:.2f} confidence")
            else:
                print(f"Generated HOLD signal (no action)")
            
            # Step 5: Update the backtest with the latest data and signals
            # For backtesting, we need to convert current prices to the expected format
            backtest_prices = current_prices[['date', 'close']].copy()
            backtest_prices.columns = ['date', 'close']
            
            # Run incremental backtest
            portfolio_value = self.backtest_engine.update_backtest(
                backtest_signals=cumulative_signals,
                price_data=backtest_prices
            )
            
            self.portfolio_values.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'portfolio_value': portfolio_value
            })
            
            print(f"Updated portfolio value: ${portfolio_value:,.2f}")
            
            # Move to the next time window
            current_date = next_date
            step_counter += 1
            
            # Add a delay to simulate real-time
            if self.simulation_delay > 0:
                time.sleep(self.simulation_delay)
        
        # After simulation completes, generate the final performance report
        self._generate_performance_report()
        
    def _generate_performance_report(self):
        """Generate a performance report at the end of the simulation."""
        if not self.portfolio_values:
            print("No portfolio data available to generate a report.")
            return
        
        # Calculate final metrics
        initial_value = self.initial_capital
        final_value = self.portfolio_values[-1]['portfolio_value']
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Calculate drawdown
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df.set_index('date', inplace=True)
        
        # Count actual trades
        buy_signals = sum(1 for s in self.signals_generated if s['signal'] == 'buy')
        sell_signals = sum(1 for s in self.signals_generated if s['signal'] == 'sell')
        
        # Plot portfolio value over time
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df.index, portfolio_df['portfolio_value'], 'b-')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(self.output_dir, f'simulation_performance_{timestamp}.png')
        plt.savefig(plot_path)
        
        # Generate performance summary
        print(f"\n{'='*80}")
        print(f"BitHunter Simulation Performance Summary")
        print(f"{'='*80}")
        print(f"Initial Portfolio Value: ${initial_value:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Buy Signals: {buy_signals}")
        print(f"Total Sell Signals: {sell_signals}")
        print(f"Performance Plot: {plot_path}")
        print(f"{'='*80}")
        
        # Save results to a JSON file for later analysis
        results = {
            'portfolio_values': self.portfolio_values,
            'signals': self.signals_generated,
            'sentiment': self.sentiment_results,
            'summary': {
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': total_return,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals
            }
        }
        
        results_path = os.path.join(self.output_dir, f'simulation_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")


# Update the BacktestEngine class to support incremental updates
def update_backtest_engine():
    """
    Add an update_backtest method to the BacktestEngine class to support
    incremental backtesting for real-time simulation.
    """
    # This should be implemented in the actual backtest.py file,
    # but for this simulation we'll simulate it here
    
    BacktestEngine.update_backtest = lambda self, backtest_signals, price_data: \
        self.run_backtest(backtest_signals, price_data)['final_portfolio_value']

    
# Run the simulation if executed as a script
if __name__ == "__main__":
    # Update BacktestEngine for real-time simulation
    update_backtest_engine()
    
    # Paths to data files
    prices_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_prices.csv')
    news_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_news.json')
    
    # Initialize and run the simulation
    simulation = RealTimeSimulation(
        historical_prices_path=prices_path,
        historical_news_path=news_path,
        initial_capital=100000.0,
        commission_rate=0.001,
        data_window_days=7,  # Process 7 days at a time
        simulation_delay=0.5  # 0.5 second delay between steps
    )
    
    # Run the simulation
    simulation.run_simulation() 