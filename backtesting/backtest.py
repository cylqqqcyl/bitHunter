"""
Backtesting Module

This module evaluates trading signals against historical data:
1. Simulates trading based on generated signals
2. Calculates performance metrics (ROI, Sharpe ratio, etc.)
3. Visualizes backtest results for strategy evaluation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import os # Add os import

# Optional: Import matplotlib for plotting
try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class BacktestEngine:
    """Engine for backtesting trading signals against historical cryptocurrency data."""
    
    def __init__(self, initial_capital=100000.0, commission_rate=0.001):
        """
        Initialize the backtest engine.
        
        Args:
            initial_capital (float): Starting capital for the backtest. Defaults to 100000.0.
            commission_rate (float): Broker commission fee per trade. Defaults to 0.001 (0.1%).
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        
        # State variables to be reset for each run
        self.reset_state()
    
    def reset_state(self):
        """Reset the internal state for a new backtest run."""
        self.current_capital = self.initial_capital
        self.holdings = 0.0  # Quantity of cryptocurrency held
        self.portfolio_value_history = []
        self.trades = []
        self.position_value = 0.0
        self.in_position = False
        
    def run_backtest(self, signals: List[Dict[str, Any]], 
                       historical_data: pd.DataFrame, 
                       initial_capital: float = 100000.0) -> Dict[str, Any]:
        """
        Run a backtest simulation using historical price data and trading signals.

        Args:
            signals (List[Dict[str, Any]]): List of trading signals, each containing
                                            'signal' ('buy', 'sell', 'hold'), 'timestamp',
                                            and optionally 'confidence'.
            historical_data (pd.DataFrame): DataFrame with historical price data. 
                                            Must contain 'date' and 'close' columns. 
                                            'date' should be parsable by pd.to_datetime.
            initial_capital (float, optional): Starting capital for this specific run. 
                                                Defaults to 100000.0.

        Returns:
            Dict[str, Any]: Dictionary containing backtest results and performance metrics.
                            Example: {
                                'final_portfolio_value': 120000.0,
                                'total_return': 0.20,
                                'max_drawdown': -0.15,
                                'trade_count': 15
                            }
                            
        Raises:
            ValueError: If historical_data or signals are invalid or empty.
        """
        # Validate inputs
        if historical_data.empty or 'date' not in historical_data.columns or 'close' not in historical_data.columns:
            raise ValueError("Historical data must be a non-empty DataFrame with 'date' and 'close' columns.")
        if not signals:
            raise ValueError("Signals list cannot be empty.")

        # Reset state for the new run
        self.initial_capital = initial_capital
        self.reset_state()

        # Prepare data: ensure date column is datetime and set as index
        if not pd.api.types.is_datetime64_any_dtype(historical_data['date']):
            historical_data['date'] = pd.to_datetime(historical_data['date'])
        historical_data = historical_data.set_index('date').sort_index()
        
        # Prepare signals: convert timestamps and create a lookup
        signals_dict = {}
        for sig in signals:
            try:
                timestamp = pd.to_datetime(sig['timestamp']).normalize() # Normalize to date part
                signals_dict[timestamp] = sig
            except (KeyError, ValueError):
                print(f"Warning: Skipping invalid signal: {sig}")
                continue
        
        # Simulate trading day by day
        for date, row in historical_data.iterrows():
            current_price = row['close']
            current_portfolio_value = self.current_capital + self.holdings * current_price
            self.portfolio_value_history.append({"date": date, "value": current_portfolio_value})

            # Check if there's a signal for today
            if date in signals_dict:
                signal_data = signals_dict[date]
                action = signal_data.get('signal', 'hold').lower()
                confidence = signal_data.get('confidence', 0.5)
                
                # --- Execute Trade Logic --- 
                
                # Buy Logic: Enter position if not already in one
                if action == 'buy' and not self.in_position:
                    # Simple strategy: invest all available capital
                    investment_amount = self.current_capital
                    commission = investment_amount * self.commission_rate
                    amount_to_invest = investment_amount - commission
                    
                    if amount_to_invest > 0 and current_price > 0:
                        self.holdings = amount_to_invest / current_price
                        self.current_capital = 0.0 # Invested all capital
                        self.in_position = True
                        self.position_value = self.holdings * current_price # Value at time of purchase
                        
                        self.trades.append({
                            'date': date,
                            'action': 'buy',
                            'price': current_price,
                            'quantity': self.holdings,
                            'commission': commission,
                            'portfolio_value': current_portfolio_value
                        })
                        print(f"{date.strftime('%Y-%m-%d')}: BUY {self.holdings:.4f} @ {current_price:.2f}")
                
                # Sell Logic: Exit position if currently holding
                elif action == 'sell' and self.in_position:
                    proceeds = self.holdings * current_price
                    commission = proceeds * self.commission_rate
                    final_proceeds = proceeds - commission
                    
                    self.current_capital += final_proceeds
                    sold_value = self.holdings * current_price # Value at time of sale
                    self.holdings = 0.0
                    self.in_position = False
                    
                    # Calculate profit/loss for this trade
                    profit_loss = sold_value - self.position_value
                    self.position_value = 0.0 # Reset position value

                    self.trades.append({
                        'date': date,
                        'action': 'sell',
                        'price': current_price,
                        'quantity': -1, # Indicate selling all holdings
                        'commission': commission,
                        'profit_loss': profit_loss,
                        'portfolio_value': current_portfolio_value + profit_loss # Approximate value after sale
                    })
                    print(f"{date.strftime('%Y-%m-%d')}: SELL @ {current_price:.2f}, P/L: {profit_loss:.2f}")

        # Final portfolio value
        final_portfolio_value = self.portfolio_value_history[-1]['value']
        
        # Calculate performance metrics
        metrics = self.calculate_metrics(self.portfolio_value_history)

        print(f"\nBacktest completed.")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Trade Count: {len(self.trades)}")

        results = {
            "final_portfolio_value": final_portfolio_value,
            **metrics, # Add all calculated metrics
            "trade_count": len(self.trades),
            "portfolio_history": self.portfolio_value_history, # Include history for plotting
            "trades_log": self.trades
        }
        
        return results
    
    def calculate_metrics(self, portfolio_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate performance metrics from the portfolio value history.

        Args:
            portfolio_history (List[Dict[str, Any]]): List of dictionaries with 'date' and 'value'.

        Returns:
            Dict[str, float]: Dictionary containing key performance metrics.
                              Includes 'total_return', 'max_drawdown'.
        """
        if not portfolio_history:
            return {"total_return": 0.0, "max_drawdown": 0.0}

        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df = portfolio_df.set_index('date')
        
        # Total Return
        total_return = (portfolio_df['value'].iloc[-1] / self.initial_capital) - 1

        # Maximum Drawdown
        rolling_max = portfolio_df['value'].cummax()
        daily_drawdown = portfolio_df['value'] / rolling_max - 1.0
        max_drawdown = daily_drawdown.min()
        
        # (Optional) Add other metrics like Sharpe Ratio, Volatility, etc.
        # These would require calculating daily/periodic returns.

        metrics = {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            # Add other metrics here if calculated
        }
        
        return metrics

    def plot_performance(self, backtest_results: Dict[str, Any], plot_filename: Optional[str] = None):
        """
        Plot the portfolio value over time using matplotlib (if available).
        
        Args:
            backtest_results (Dict[str, Any]): The dictionary returned by run_backtest.
            plot_filename (Optional[str]): The full path to save the plot file. 
                                           If None, generates a default filename.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not found. Skipping plot generation.")
            return None # Return None if plot not generated
            
        if 'portfolio_history' not in backtest_results or not backtest_results['portfolio_history']:
            print("No portfolio history found in results. Cannot plot performance.")
            return None # Return None if plot not generated
        
        # Set non-interactive backend for matplotlib to avoid thread-related errors
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
            
        history_df = pd.DataFrame(backtest_results['portfolio_history'])
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df = history_df.set_index('date')
        
        fig, ax = plt.subplots(figsize=(12, 6)) # Use fig, ax for clarity
        ax.plot(history_df.index, history_df['value'], label='Portfolio Value')
        
        # Mark trades on the plot
        trades_log = backtest_results.get('trades_log', [])
        buy_dates = [pd.to_datetime(t['date']) for t in trades_log if t['action'] == 'buy']
        sell_dates = [pd.to_datetime(t['date']) for t in trades_log if t['action'] == 'sell']
        
        # Filter dates to only those present in the history index
        valid_buy_dates = [d for d in buy_dates if d in history_df.index]
        valid_sell_dates = [d for d in sell_dates if d in history_df.index]

        if valid_buy_dates:
            ax.scatter(valid_buy_dates, history_df.loc[valid_buy_dates]['value'], marker='^', color='green', label='Buy Signal', s=100, zorder=5)
        if valid_sell_dates:
             ax.scatter(valid_sell_dates, history_df.loc[valid_sell_dates]['value'], marker='v', color='red', label='Sell Signal', s=100, zorder=5)

        ax.set_title('Portfolio Performance Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        
        # Save or show the plot
        if plot_filename is None:
            # Default filename if none provided
            plot_filename = f"backtest_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        # Ensure the directory exists
        plot_dir = os.path.dirname(plot_filename)
        if plot_dir and not os.path.exists(plot_dir):
            try:
                os.makedirs(plot_dir)
                print(f"Created directory: {plot_dir}")
            except OSError as e:
                print(f"Error creating directory {plot_dir}: {e}")
                # Fallback to saving in the current directory
                plot_filename = os.path.basename(plot_filename)

        try:
            plt.savefig(plot_filename)
            print(f"Performance plot saved as {plot_filename}")
            plt.close(fig) # Close the figure to free memory
            return plot_filename # Return the filename where plot was saved
        except Exception as e:
            print(f"Could not save plot: {e}")
            plt.close(fig) # Close the figure even if saving failed
            return None # Return None if saving failed
        # plt.show() # Uncomment to display plot interactively

# Example usage and simple testing
if __name__ == "__main__":
    # 1. Create sample historical data
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq='D')
    prices = 100 + np.random.randn(len(dates)).cumsum() # Simple random walk
    historical_data = pd.DataFrame({'date': dates, 'close': prices})
    historical_data['close'] = historical_data['close'].clip(lower=1) # Ensure price > 0
    print("--- Sample Historical Data --- ")
    print(historical_data.head())

    # 2. Create sample trading signals
    signals = [
        {'timestamp': '2023-01-05', 'signal': 'buy', 'confidence': 0.8},
        {'timestamp': '2023-01-15', 'signal': 'sell', 'confidence': 0.7},
        {'timestamp': '2023-01-20', 'signal': 'buy', 'confidence': 0.9},
        {'timestamp': '2023-01-28', 'signal': 'sell', 'confidence': 0.6}
    ]
    print("\n--- Sample Signals --- ")
    print(signals)

    # 3. Run the backtest
    engine = BacktestEngine(initial_capital=100000.0)
    print("\n--- Running Backtest --- ")
    try:
        backtest_results = engine.run_backtest(signals, historical_data.copy()) # Use copy to avoid modifying original
        print("\n--- Backtest Results Summary --- ")
        print(f"Final Value: ${backtest_results['final_portfolio_value']:,.2f}")
        print(f"Total Return: {backtest_results['total_return']:.2%}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        print(f"Trade Count: {backtest_results['trade_count']}")
        
        # 4. Plot performance
        print("\n--- Generating Performance Plot --- ")
        # Example of specifying a path (adjust as needed for testing)
        # plot_path = os.path.join("dashboard", "static", "plots", f"test_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        # engine.plot_performance(backtest_results, plot_filename=plot_path)
        engine.plot_performance(backtest_results) # Use default filename for simple testing
        
    except ValueError as e:
        print(f"Backtest Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    print("\nBacktesting example finished.") 