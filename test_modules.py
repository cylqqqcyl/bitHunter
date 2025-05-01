"""
Test script for data ingestion and sentiment analysis modules
"""

from data_ingestion.collector import CryptoDataCollector
from llm_sentiment.analyzer import SentimentAnalyzer

def test_data_ingestion():
    """Test the data ingestion module."""
    print("=== Testing Data Ingestion Module ===")
    collector = CryptoDataCollector()
    # Test news collection
    print("\n1. Testing fetch_crypto_news()")
    news = collector.fetch_crypto_news(keywords=["Bitcoin", "Ethereum"], days_back=5)
    print(f"Retrieved {len(news)} news articles")
    if news:
        print("Sample news item:")
        print(f"  Title: {news[0]['title']}")
        print(f"  Source: {news[0]['source']}")
        print(f"  Date: {news[0]['published_at']}")
        print(f"  Hidden sentiment: {news[0]['_sentiment']}")
    # Test historical data collection
    print("\n2. Testing fetch_historical_data()")
    hist_data = collector.fetch_historical_data("BTC", "2023-01-01", "2023-01-05")
    print(f"Retrieved historical data with {len(hist_data)} rows")
    if not hist_data.empty:
        print("Sample data:")
        print(hist_data.head(3))

def test_sentiment_analysis():
    """Test the sentiment analysis module."""
    print("\n=== Testing LLM Sentiment Analysis Module ===")
    analyzer = SentimentAnalyzer()
    # Test texts with different sentiment
    test_texts = [
        "Bitcoin surges 10% as institutional adoption increases",
        "Ethereum drops amid concerns about network congestion",
        "Cryptocurrency markets remain stable as trading volume decreases",
        "New regulations could pose challenges for DeFi projects"
    ]
    # Test individual sentiment analysis
    print("\n1. Testing analyze_sentiment()")
    for i, text in enumerate(test_texts):
        result = analyzer.analyze_sentiment(text)
        print(f"\nText {i+1}: {text}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Score: {result['score']}")
        print(f"  Keywords: {', '.join(result['keywords']) if result['keywords'] else 'None'}")
        print(f"  Cryptocurrencies: {', '.join(result['cryptocurrencies']) if result['cryptocurrencies'] else 'None'}")
        print(f"  Confidence: {result['confidence']}")
    # Test LLM-simulated analysis
    print("\n2. Testing analyze_text() (LLM simulation)")
    llm_result = analyzer.analyze_text(test_texts[0])
    print(f"Enhanced LLM analysis for: {test_texts[0]}")
    print(f"  Sentiment: {llm_result['sentiment']}")
    print(f"  Market indicators: {', '.join(llm_result['market_indicators'])}")
    print(f"  Cryptocurrencies: {', '.join(llm_result['cryptocurrencies'])}")
    print(f"  Events: {', '.join(llm_result['events']) if llm_result['events'] else 'None'}")
    print(f"  Market impact (short-term): {llm_result['market_impact']['short_term']}")
    print(f"  Market impact (long-term): {llm_result['market_impact']['long_term']}")
    print(f"  Confidence: {llm_result['confidence']}/10")
    # Test batch analysis and aggregation
    print("\n3. Testing batch analysis and aggregation")
    batch_results = analyzer.analyze_batch(test_texts)
    aggregated = analyzer.aggregate_sentiment(batch_results)
    print(f"Analyzed {aggregated['sample_size']} texts")
    print(f"Overall sentiment: {aggregated['overall_sentiment']}")
    print(f"Confidence: {aggregated['confidence']}/10")

if __name__ == "__main__":
    try:
        test_data_ingestion()
        test_sentiment_analysis()
        print("\n=== All tests completed successfully ===")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
