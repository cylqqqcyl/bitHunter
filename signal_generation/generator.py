"""
Signal Generator Module

This module converts sentiment analysis data into actionable trading signals:
1. Processes sentiment data from the LLM analyzer
2. Applies trading logic to generate buy/sell/hold signals
3. Calculates signal strength and confidence metrics
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

class SignalGenerator:
    """Generates trading signals from sentiment analysis data."""
    
    def __init__(self, config=None):
        """
        Initialize the signal generator with configuration.
        
        Args:
            config (Dict, optional): Configuration parameters. Defaults to None.
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Create default configuration settings.
        
        Returns:
            Dict: Default configuration
        """
        return {
            # Sentiment-based signal thresholds
            "buy_score_threshold": 0.5,    # Reduced from 0.75
            "sell_score_threshold": 0.5,    # Increased from 0.35
            "buy_confidence_threshold": 5,   # Reduced from 6
            "sell_confidence_threshold": 5,   # Reduced from 6
            
            # Keywords that strengthen buy signals
            "buy_keywords": ["bullish", "surge", "rally", "adoption", "breakthrough"],
            # Keywords that strengthen sell signals
            "sell_keywords": ["bearish", "crash", "plunge", "hack", "regulation", "fear"],
            
            # Optional: Technical indicator thresholds (not implemented yet)
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "sma_cross_period": 50  # Example: Check against 50-day SMA
        }
    
    def generate_signal(self, news_sentiment: Dict[str, Any], 
                        historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generates a trading signal based on sentiment analysis and optional historical data.

        Args:
            news_sentiment (dict): A dictionary containing the sentiment analysis result.
                                   Expected keys: 'sentiment', 'score', 'keywords', 'confidence'.
            historical_data (DataFrame, optional): A pandas DataFrame containing historical price data.
                                                   Expected columns: 'date', 'close'. Defaults to None.

        Returns:
            dict: A dictionary representing the trading signal.
                  Example: {
                      'signal': 'buy', 
                      'confidence': 0.8, 
                      'supporting_metrics': {
                          'sentiment_score': 0.85, 
                          'keyword_hits': ['bullish', 'adoption']
                      }
                  }

        Raises:
            ValueError: If the news_sentiment dictionary is incomplete or invalid.
        """
        # Validate input
        if not news_sentiment or not isinstance(news_sentiment, dict):
            raise ValueError("Invalid or empty sentiment data provided")
            
        sentiment = news_sentiment.get("sentiment", "neutral").lower()
        score = news_sentiment.get("score", 0.5)  # Normalized score (0.0 to 1.0)
        confidence = news_sentiment.get("confidence", 5)  # Confidence (1-10 scale)
        keywords = news_sentiment.get("keywords", [])

        # Initialize signal
        trade_signal = "hold"
        signal_confidence = confidence / 10.0  # Convert confidence to 0.0-1.0 scale
        supporting_metrics = {
            "sentiment_score": score,
            "keyword_hits": []
        }

        # Determine Buy signal based on heuristics
        if (sentiment == "positive" or sentiment == "bullish" or sentiment == "very bullish") and \
           score >= self.config["buy_score_threshold"] and \
           confidence >= self.config["buy_confidence_threshold"]:
            trade_signal = "buy"
            supporting_metrics["keyword_hits"] = [k for k in keywords if k in self.config["buy_keywords"]]
            # Boost confidence if strong buy keywords are present
            if supporting_metrics["keyword_hits"]:
                signal_confidence = min(1.0, signal_confidence + 0.1)

        # Determine Sell signal based on heuristics
        elif (sentiment == "negative" or sentiment == "bearish" or sentiment == "very bearish") and \
             score <= self.config["sell_score_threshold"] and \
             confidence >= self.config["sell_confidence_threshold"]:
            trade_signal = "sell"
            supporting_metrics["keyword_hits"] = [k for k in keywords if k in self.config["sell_keywords"]]
            # Boost confidence if strong sell keywords are present
            if supporting_metrics["keyword_hits"]:
                signal_confidence = min(1.0, signal_confidence + 0.1)

        # TODO: Incorporate historical data analysis (e.g., price trends, technical indicators)
        # if historical_data is not None and not historical_data.empty:
        #     # Example: Check recent price trend
        #     recent_trend = self._calculate_trend(historical_data)
        #     supporting_metrics["recent_trend"] = recent_trend
        #     # Adjust signal based on trend (e.g., require stronger sentiment if trend is opposing)

        print(f"Generated signal: {trade_signal} (Confidence: {signal_confidence:.2f}) based on sentiment: {sentiment} (Score: {score:.2f}, Conf: {confidence})")

        return {
            "signal": trade_signal,
            "confidence": round(signal_confidence, 2),
            "supporting_metrics": supporting_metrics
        }

    def _calculate_trend(self, historical_data: pd.DataFrame, window: int = 7) -> str:
        """
        Calculate a simple price trend based on the last few data points.
        (Placeholder - needs implementation)
        
        Args:
            historical_data (pd.DataFrame): DataFrame with 'close' prices.
            window (int): Number of days to consider for trend calculation.
            
        Returns:
            str: Trend indication ('upward', 'downward', 'sideways')
        """
        if len(historical_data) < window:
            return "insufficient data"
        
        recent_closes = historical_data['close'].tail(window)
        # Simple trend: compare start and end of window
        if recent_closes.iloc[-1] > recent_closes.iloc[0]:
            return "upward"
        elif recent_closes.iloc[-1] < recent_closes.iloc[0]:
            return "downward"
        else:
            return "sideways"
            
    # Previous generate_batch_signals method might need updating if signature changes
    def generate_batch_signals(self, sentiment_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate signals for a batch of sentiment analyses.
        (Assuming generate_signal now takes news_sentiment dict)
        
        Args:
            sentiment_batch (List[Dict]): List of sentiment analyses
            
        Returns:
            List[Dict]: List of trading signals
        """
        signals = []
        for sentiment_data in sentiment_batch:
            # Assuming no historical data is passed for batch generation for now
            signal = self.generate_signal(sentiment_data)
            signals.append(signal)
        return signals

# Example usage and simple testing
if __name__ == "__main__":
    generator = SignalGenerator()

    # Test case 1: Strong bullish sentiment
    bullish_sentiment = {
        "sentiment": "positive", 
        "score": 0.9, 
        "confidence": 8, 
        "keywords": ["bullish", "surge", "adoption"]
    }
    print("\n--- Test Case 1: Strong Bullish --- ")
    signal1 = generator.generate_signal(bullish_sentiment)
    print("Output Signal:", signal1)
    assert signal1["signal"] == "buy"

    # Test case 2: Weak bullish sentiment (below threshold)
    weak_bullish_sentiment = {
        "sentiment": "positive", 
        "score": 0.6, 
        "confidence": 7, 
        "keywords": ["positive", "growth"]
    }
    print("\n--- Test Case 2: Weak Bullish --- ")
    signal2 = generator.generate_signal(weak_bullish_sentiment)
    print("Output Signal:", signal2)
    assert signal2["signal"] == "hold"
    
    # Test case 3: Strong bearish sentiment
    bearish_sentiment = {
        "sentiment": "negative", 
        "score": 0.1, 
        "confidence": 9, 
        "keywords": ["bearish", "crash", "fear"]
    }
    print("\n--- Test Case 3: Strong Bearish --- ")
    signal3 = generator.generate_signal(bearish_sentiment)
    print("Output Signal:", signal3)
    assert signal3["signal"] == "sell"
    
    # Test case 4: Neutral sentiment
    neutral_sentiment = {
        "sentiment": "neutral", 
        "score": 0.5, 
        "confidence": 6, 
        "keywords": ["stable", "market"]
    }
    print("\n--- Test Case 4: Neutral --- ")
    signal4 = generator.generate_signal(neutral_sentiment)
    print("Output Signal:", signal4)
    assert signal4["signal"] == "hold"
    
    # Test case 5: Missing data (should raise error or handle gracefully)
    print("\n--- Test Case 5: Incomplete Data --- ")
    incomplete_sentiment = {"sentiment": "positive", "score": 0.8} # Missing confidence
    try:
        signal5 = generator.generate_signal(incomplete_sentiment)
        print("Output Signal:", signal5)
        # If it handles gracefully (uses default confidence), check signal
        assert signal5["signal"] == "hold" # Default confidence is 5, below buy threshold 6
    except ValueError as e:
        print(f"Handled incomplete data as expected: {e}")

    print("\nSignal Generator tests passed!") 