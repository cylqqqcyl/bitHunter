"""
Technical Indicators Module

This module provides functions to calculate:
1. Technical indicators (Moving Averages, RSI, MACD, etc.)
2. Risk and performance metrics (Sharpe ratio, Sortino ratio, etc.)
3. Additional analytics for backtesting
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Technical Indicators ---

def calculate_moving_average(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA) for a price series.
    
    Args:
        data (pd.Series): Series of price data
        window (int): Window period for moving average calculation
    
    Returns:
        pd.Series: Series containing the moving average values
    
    Raises:
        ValueError: If window size is invalid or data is empty
    """
    if window <= 0:
        raise ValueError("Window size must be positive")
    if data is None or len(data) == 0:
        raise ValueError("Input data is empty")
    
    # Calculate the simple moving average
    ma = data.rolling(window=window).mean()
    
    return ma

def calculate_exponential_moving_average(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA) for a price series.
    
    Args:
        data (pd.Series): Series of price data
        window (int): Window period for EMA calculation
    
    Returns:
        pd.Series: Series containing the EMA values
    
    Raises:
        ValueError: If window size is invalid or data is empty
    """
    if window <= 0:
        raise ValueError("Window size must be positive")
    if data is None or len(data) == 0:
        raise ValueError("Input data is empty")
    
    # Calculate the exponential moving average
    ema = data.ewm(span=window, adjust=False).mean()
    
    return ema

def calculate_RSI(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a price series.
    
    Args:
        data (pd.Series): Series of price data
        window (int, optional): Window period for RSI calculation. Defaults to 14.
    
    Returns:
        pd.Series: Series containing the RSI values (0-100)
    
    Raises:
        ValueError: If window size is invalid or data is empty
    """
    if window <= 0:
        raise ValueError("Window size must be positive")
    if data is None or len(data) == 0:
        raise ValueError("Input data is empty")
    
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculate average gain and loss over the window
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate relative strength
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_MACD(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a price series.
    
    Args:
        data (pd.Series): Series of price data
        fast_period (int, optional): Fast EMA period. Defaults to 12.
        slow_period (int, optional): Slow EMA period. Defaults to 26.
        signal_period (int, optional): Signal line period. Defaults to 9.
    
    Returns:
        Dict[str, pd.Series]: Dictionary containing MACD, signal line, and histogram
        {
            'macd': MACD line,
            'signal': Signal line,
            'histogram': MACD histogram
        }
    
    Raises:
        ValueError: If period values are invalid or data is empty
    """
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("Period values must be positive")
    if fast_period >= slow_period:
        raise ValueError("Fast period must be smaller than slow period")
    if data is None or len(data) == 0:
        raise ValueError("Input data is empty")
    
    # Calculate EMAs
    fast_ema = calculate_exponential_moving_average(data, fast_period)
    slow_ema = calculate_exponential_moving_average(data, slow_period)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line (EMA of MACD line)
    signal_line = calculate_exponential_moving_average(macd_line, signal_period)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands for a price series.
    
    Args:
        data (pd.Series): Series of price data
        window (int, optional): Window period for moving average. Defaults to 20.
        num_std (float, optional): Number of standard deviations. Defaults to 2.0.
    
    Returns:
        Dict[str, pd.Series]: Dictionary containing upper band, middle band (SMA), and lower band
        {
            'upper': Upper Bollinger Band,
            'middle': Middle Band (SMA),
            'lower': Lower Bollinger Band
        }
    
    Raises:
        ValueError: If window size is invalid or data is empty
    """
    if window <= 0:
        raise ValueError("Window size must be positive")
    if num_std < 0:
        raise ValueError("Number of standard deviations must be non-negative")
    if data is None or len(data) == 0:
        raise ValueError("Input data is empty")
    
    # Calculate middle band (SMA)
    middle_band = calculate_moving_average(data, window)
    
    # Calculate standard deviation
    std = data.rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return {
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band
    }

# --- Risk & Performance Metrics ---

def calculate_returns(prices: pd.Series, method: str = 'percent') -> pd.Series:
    """
    Calculate returns from a price series.
    
    Args:
        prices (pd.Series): Series of price data
        method (str, optional): Method to calculate returns ('percent' or 'log'). Defaults to 'percent'.
    
    Returns:
        pd.Series: Series of returns
    
    Raises:
        ValueError: If method is invalid or data is empty
    """
    if prices is None or len(prices) == 0:
        raise ValueError("Input data is empty")
    
    if method.lower() == 'percent':
        # Calculate percentage returns (price_t / price_t-1 - 1)
        returns = prices.pct_change()
    elif method.lower() == 'log':
        # Calculate log returns (ln(price_t / price_t-1))
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError("Method must be 'percent' or 'log'")
    
    return returns

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sharpe Ratio for a series of returns.
    
    Args:
        returns (pd.Series): Series of returns (not prices)
        risk_free_rate (float, optional): Risk-free rate (annualized). Defaults to 0.0.
        periods_per_year (int, optional): Number of periods in a year. Defaults to 252 (trading days).
    
    Returns:
        float: Sharpe Ratio
    
    Raises:
        ValueError: If data is empty or standard deviation is zero
    """
    if returns is None or len(returns) == 0:
        raise ValueError("Input returns data is empty")
    
    # Remove NaN values
    returns = returns.dropna()
    
    if len(returns) == 0:
        raise ValueError("No valid returns data after dropping NaN values")
    
    # Calculate mean return and standard deviation
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0:
        raise ValueError("Standard deviation of returns is zero")
    
    # Convert to annualized figures
    annualized_return = mean_return * periods_per_year
    annualized_std = std_return * np.sqrt(periods_per_year)
    
    # Calculate Sharpe ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std
    
    return sharpe_ratio

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sortino Ratio for a series of returns.
    The Sortino ratio is similar to the Sharpe ratio, but it only considers downside volatility.
    
    Args:
        returns (pd.Series): Series of returns (not prices)
        risk_free_rate (float, optional): Risk-free rate (annualized). Defaults to 0.0.
        periods_per_year (int, optional): Number of periods in a year. Defaults to 252 (trading days).
    
    Returns:
        float: Sortino Ratio
    
    Raises:
        ValueError: If data is empty or downside deviation is zero
    """
    if returns is None or len(returns) == 0:
        raise ValueError("Input returns data is empty")
    
    # Remove NaN values
    returns = returns.dropna()
    
    if len(returns) == 0:
        raise ValueError("No valid returns data after dropping NaN values")
    
    # Calculate mean return
    mean_return = returns.mean()
    
    # Calculate downside deviation (only negative returns)
    downside_returns = returns[returns < 0].copy()
    if len(downside_returns) == 0:
        # No negative returns, perfect performance
        return float('inf') if mean_return > 0 else 0.0
    
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    
    if downside_deviation == 0:
        raise ValueError("Downside deviation of returns is zero")
    
    # Convert to annualized figures
    annualized_return = mean_return * periods_per_year
    annualized_downside = downside_deviation * np.sqrt(periods_per_year)
    
    # Calculate Sortino ratio
    sortino_ratio = (annualized_return - risk_free_rate) / annualized_downside
    
    return sortino_ratio

def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized volatility of returns.
    
    Args:
        returns (pd.Series): Series of returns (not prices)
        periods_per_year (int, optional): Number of periods in a year. Defaults to 252 (trading days).
    
    Returns:
        float: Annualized volatility
    
    Raises:
        ValueError: If data is empty
    """
    if returns is None or len(returns) == 0:
        raise ValueError("Input returns data is empty")
    
    # Remove NaN values
    returns = returns.dropna()
    
    if len(returns) == 0:
        raise ValueError("No valid returns data after dropping NaN values")
    
    # Calculate standard deviation
    std_return = returns.std()
    
    # Calculate annualized volatility
    annualized_volatility = std_return * np.sqrt(periods_per_year)
    
    return annualized_volatility

def calculate_max_drawdown(prices: pd.Series) -> Dict[str, Any]:
    """
    Calculate the maximum drawdown for a price series.
    
    Args:
        prices (pd.Series): Series of price data
    
    Returns:
        Dict[str, Any]: Dictionary containing drawdown metrics
        {
            'max_drawdown': Maximum drawdown as a percentage (0 to 1),
            'peak_date': Date of the peak,
            'trough_date': Date of the trough,
            'recovery_date': Date of recovery (or None if not recovered),
            'drawdown_length': Length of drawdown period in days,
            'recovery_length': Length of recovery period in days (or None if not recovered)
        }
    
    Raises:
        ValueError: If data is empty
    """
    if prices is None or len(prices) == 0:
        raise ValueError("Input price data is empty")
    
    # Remove NaN values
    prices = prices.dropna()
    
    if len(prices) == 0:
        raise ValueError("No valid price data after dropping NaN values")
    
    # Calculate running maximum
    running_max = prices.cummax()
    
    # Calculate drawdown series
    drawdown = (prices - running_max) / running_max
    
    # Find the maximum drawdown
    max_drawdown = drawdown.min()
    
    # Find the dates of peak and trough
    peak_idx = drawdown[drawdown == 0].index
    peak_idx = peak_idx[peak_idx < drawdown.idxmin()]
    if len(peak_idx) > 0:
        peak_date = peak_idx[-1]
    else:
        peak_date = prices.index[0]
    
    trough_date = drawdown.idxmin()
    
    # Find recovery date (if any)
    recovery_data = drawdown.loc[trough_date:]
    recovery_indices = recovery_data[recovery_data >= 0].index
    
    if len(recovery_indices) > 0:
        recovery_date = recovery_indices[0]
        if hasattr(recovery_date, 'days') and hasattr(trough_date, 'days'):
            recovery_length = (recovery_date - trough_date).days
        else:
            # If dates are integers (index positions), just calculate the difference
            recovery_length = recovery_date - trough_date
    else:
        recovery_date = None
        recovery_length = None
    
    # Calculate drawdown length
    if hasattr(trough_date, 'days') and hasattr(peak_date, 'days'):
        drawdown_length = (trough_date - peak_date).days
    else:
        # If dates are integers (index positions), just calculate the difference
        drawdown_length = trough_date - peak_date
    
    return {
        'max_drawdown': abs(max_drawdown),
        'peak_date': peak_date,
        'trough_date': trough_date,
        'recovery_date': recovery_date,
        'drawdown_length': drawdown_length,
        'recovery_length': recovery_length
    }

def calculate_calmar_ratio(returns: pd.Series, prices: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the Calmar Ratio, which is the ratio of annualized return to maximum drawdown.
    
    Args:
        returns (pd.Series): Series of returns (not prices)
        prices (pd.Series): Series of price data (used to calculate max drawdown)
        periods_per_year (int, optional): Number of periods in a year. Defaults to 252 (trading days).
    
    Returns:
        float: Calmar Ratio
    
    Raises:
        ValueError: If data is empty or max drawdown is zero
    """
    if returns is None or len(returns) == 0 or prices is None or len(prices) == 0:
        raise ValueError("Input data is empty")
    
    # Remove NaN values
    returns = returns.dropna()
    
    if len(returns) == 0:
        raise ValueError("No valid returns data after dropping NaN values")
    
    # Calculate annualized return
    annualized_return = returns.mean() * periods_per_year
    
    # Calculate maximum drawdown
    max_drawdown_result = calculate_max_drawdown(prices)
    max_drawdown = max_drawdown_result['max_drawdown']
    
    if max_drawdown == 0:
        # No drawdown, perfect performance
        return float('inf') if annualized_return > 0 else 0.0
    
    # Calculate Calmar ratio
    calmar_ratio = annualized_return / max_drawdown
    
    return calmar_ratio

def calculate_value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) for a series of returns using the historical method.
    
    Args:
        returns (pd.Series): Series of returns (not prices)
        confidence_level (float, optional): Confidence level for VaR (0 to 1). Defaults to 0.95.
    
    Returns:
        float: Value at Risk (as a positive number)
    
    Raises:
        ValueError: If data is empty or confidence level is invalid
    """
    if returns is None or len(returns) == 0:
        raise ValueError("Input returns data is empty")
    
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("Confidence level must be between 0 and 1 (exclusive)")
    
    # Remove NaN values
    returns = returns.dropna()
    
    if len(returns) == 0:
        raise ValueError("No valid returns data after dropping NaN values")
    
    # Calculate the percentile corresponding to the confidence level
    var = abs(returns.quantile(1 - confidence_level))
    
    return var

# --- Combined Analytics ---

def calculate_all_indicators(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Calculate all technical indicators for a price DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        price_col (str, optional): Column name for price data. Defaults to 'close'.
    
    Returns:
        pd.DataFrame: Original DataFrame with additional columns for indicators
    
    Raises:
        ValueError: If data is empty or price column doesn't exist
    """
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is empty")
    
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Get the price series
    prices = df[price_col]
    
    # Calculate moving averages
    try:
        result_df['SMA_20'] = calculate_moving_average(prices, 20)
        result_df['SMA_50'] = calculate_moving_average(prices, 50)
        result_df['SMA_200'] = calculate_moving_average(prices, 200)
        result_df['EMA_12'] = calculate_exponential_moving_average(prices, 12)
        result_df['EMA_26'] = calculate_exponential_moving_average(prices, 26)
    except Exception as e:
        logger.warning(f"Error calculating moving averages: {str(e)}")
    
    # Calculate RSI
    try:
        result_df['RSI_14'] = calculate_RSI(prices, 14)
    except Exception as e:
        logger.warning(f"Error calculating RSI: {str(e)}")
    
    # Calculate MACD
    try:
        macd_result = calculate_MACD(prices)
        result_df['MACD'] = macd_result['macd']
        result_df['MACD_Signal'] = macd_result['signal']
        result_df['MACD_Histogram'] = macd_result['histogram']
    except Exception as e:
        logger.warning(f"Error calculating MACD: {str(e)}")
    
    # Calculate Bollinger Bands
    try:
        bb_result = calculate_bollinger_bands(prices)
        result_df['BB_Upper'] = bb_result['upper']
        result_df['BB_Middle'] = bb_result['middle']
        result_df['BB_Lower'] = bb_result['lower']
    except Exception as e:
        logger.warning(f"Error calculating Bollinger Bands: {str(e)}")
    
    # Calculate returns
    try:
        result_df['Returns'] = calculate_returns(prices)
    except Exception as e:
        logger.warning(f"Error calculating returns: {str(e)}")
    
    logger.info("Calculated all technical indicators successfully")
    return result_df

def calculate_all_risk_metrics(returns: pd.Series, prices: pd.Series) -> Dict[str, float]:
    """
    Calculate all risk and performance metrics for a series of returns.
    
    Args:
        returns (pd.Series): Series of returns (not prices)
        prices (pd.Series): Series of price data (for drawdown calculations)
    
    Returns:
        Dict[str, float]: Dictionary containing all risk metrics
    
    Raises:
        ValueError: If data is empty
    """
    if returns is None or len(returns) == 0 or prices is None or len(prices) == 0:
        raise ValueError("Input data is empty")
    
    metrics = {}
    
    # Calculate volatility
    try:
        metrics['volatility'] = calculate_volatility(returns)
    except Exception as e:
        logger.warning(f"Error calculating volatility: {str(e)}")
        metrics['volatility'] = None
    
    # Calculate Sharpe ratio
    try:
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
    except Exception as e:
        logger.warning(f"Error calculating Sharpe ratio: {str(e)}")
        metrics['sharpe_ratio'] = None
    
    # Calculate Sortino ratio
    try:
        metrics['sortino_ratio'] = calculate_sortino_ratio(returns)
    except Exception as e:
        logger.warning(f"Error calculating Sortino ratio: {str(e)}")
        metrics['sortino_ratio'] = None
    
    # Calculate maximum drawdown
    try:
        drawdown_result = calculate_max_drawdown(prices)
        metrics['max_drawdown'] = drawdown_result['max_drawdown']
        metrics['drawdown_length'] = drawdown_result['drawdown_length']
    except Exception as e:
        logger.warning(f"Error calculating maximum drawdown: {str(e)}")
        metrics['max_drawdown'] = None
        metrics['drawdown_length'] = None
    
    # Calculate Calmar ratio
    try:
        metrics['calmar_ratio'] = calculate_calmar_ratio(returns, prices)
    except Exception as e:
        logger.warning(f"Error calculating Calmar ratio: {str(e)}")
        metrics['calmar_ratio'] = None
    
    # Calculate VaR
    try:
        metrics['var_95'] = calculate_value_at_risk(returns, 0.95)
        metrics['var_99'] = calculate_value_at_risk(returns, 0.99)
    except Exception as e:
        logger.warning(f"Error calculating VaR: {str(e)}")
        metrics['var_95'] = None
        metrics['var_99'] = None
    
    # Calculate annualized return
    try:
        metrics['annualized_return'] = returns.mean() * 252
    except Exception as e:
        logger.warning(f"Error calculating annualized return: {str(e)}")
        metrics['annualized_return'] = None
    
    logger.info("Calculated all risk metrics successfully")
    return metrics

# Test function to demonstrate usage
def run_test():
    """Run a test calculation on sample data."""
    # Create sample price data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = pd.Series(np.cumsum(np.random.normal(0.001, 0.02, 100)) + 100, index=dates)
    
    # Calculate returns
    returns = calculate_returns(prices)
    
    # Calculate technical indicators
    sma_20 = calculate_moving_average(prices, 20)
    ema_12 = calculate_exponential_moving_average(prices, 12)
    rsi = calculate_RSI(prices)
    macd = calculate_MACD(prices)
    bollinger = calculate_bollinger_bands(prices)
    
    # Calculate risk metrics
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    volatility = calculate_volatility(returns)
    max_dd = calculate_max_drawdown(prices)
    
    # Print results
    print("Sample Technical Indicators:")
    print(f"  SMA(20): {sma_20.iloc[-1]:.2f}")
    print(f"  EMA(12): {ema_12.iloc[-1]:.2f}")
    print(f"  RSI(14): {rsi.iloc[-1]:.2f}")
    print(f"  MACD: {macd['macd'].iloc[-1]:.4f}")
    
    print("\nSample Risk Metrics:")
    print(f"  Sharpe Ratio: {sharpe:.4f}")
    print(f"  Sortino Ratio: {sortino:.4f}")
    print(f"  Volatility (annualized): {volatility:.4f}")
    print(f"  Maximum Drawdown: {max_dd['max_drawdown']:.4%}")
    
    # Calculate all indicators at once
    df = pd.DataFrame({'close': prices, 'date': dates})
    df_with_indicators = calculate_all_indicators(df)
    
    # Calculate all risk metrics at once
    all_metrics = calculate_all_risk_metrics(returns, prices)
    print("\nAll Risk Metrics:")
    for key, value in all_metrics.items():
        if value is not None:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    run_test() 