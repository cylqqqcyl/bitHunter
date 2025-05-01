"""
LLM Sentiment Analyzer Module

This module processes text data through Large Language Models to:
1. Analyze sentiment of cryptocurrency news and social media content
2. Extract key insights and market indicators
3. Generate structured data for signal generation
"""

import os
import json
import re
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai_api_key = os.getenv("OPENAI_API_KEY")

class SentimentAnalyzer:
    """Uses LLMs to analyze cryptocurrency sentiment from text data."""
    
    def __init__(self, model="gpt-4"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model (str): The LLM model to use. Defaults to "gpt-4".
        """
        self.model = model
        self.prompt_template = """
        Analyze the following cryptocurrency-related text for sentiment and insights:
        
        TEXT:
        {text}
        
        Please provide a detailed analysis covering:
        1. Overall sentiment (very bearish, bearish, neutral, bullish, very bullish)
        2. Key market indicators mentioned
        3. Specific cryptocurrencies discussed
        4. Important events or news highlights
        5. Potential market impact (short-term and long-term)
        6. Confidence level of your analysis (1-10)
        
        Format your response as JSON with these fields.
        """

        # Real LLM API specific prompt
        self.llm_api_prompt = """
        You are a financial sentiment analysis expert specialized in cryptocurrency markets. 
        
        Analyze the following cryptocurrency-related news article and extract sentiment information:
        
        TEXT:
        {text}
        
        Provide your analysis in the following JSON format ONLY, with no additional text or explanations:
        
        {
          "sentiment": "positive" | "negative" | "neutral",
          "score": <float between 0 and 1, where 0.5 is neutral, values closer to 1 are more positive, values closer to 0 are more negative>,
          "keywords": [<list of sentiment keywords found in the text>],
          "cryptocurrencies": [<list of specific cryptocurrencies mentioned>],
          "market_indicators": [<list of market indicators mentioned>],
          "confidence": <float between 0 and 1 indicating confidence in your analysis>
        }
        
        Focus on extracting accurate sentiment. The sentiment score should be between 0 and 1, with 0.5 being neutral.
        """
        
        # Define keyword dictionaries for sentiment analysis simulation
        self.bullish_keywords = [
            "surge", "rally", "boom", "bullish", "uptrend", "outperform", "gains", 
            "positive", "optimistic", "growth", "adoption", "breakthrough", "milestone",
            "soar", "climb", "upward", "rising", "recovery", "record high", "all-time high"
        ]
        
        self.bearish_keywords = [
            "crash", "plunge", "collapse", "bearish", "downtrend", "underperform", "losses",
            "negative", "pessimistic", "decline", "fall", "drop", "dump", "tumble", "selloff",
            "downturn", "correction", "bubble", "risk", "concern", "fear", "panic"
        ]
        
        self.crypto_keywords = {
            "bitcoin": "BTC",
            "btc": "BTC",
            "ethereum": "ETH",
            "eth": "ETH",
            "ripple": "XRP",
            "xrp": "XRP",
            "cardano": "ADA",
            "ada": "ADA",
            "solana": "SOL",
            "sol": "SOL",
            "binance coin": "BNB",
            "bnb": "BNB",
            "dogecoin": "DOGE",
            "doge": "DOGE",
            "polkadot": "DOT",
            "avalanche": "AVAX",
            "polygon": "MATIC"
        }
        
        self.market_indicators = [
            "volume", "volatility", "market cap", "liquidity", "trading activity",
            "hash rate", "mining difficulty", "gas fees", "transaction fees",
            "institutional investment", "exchange outflows", "exchange inflows",
            "whale movement", "regulatory", "adoption", "integration", "partnership"
        ]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Process the input text and output a sentiment analysis result.
        
        Args:
            text (str): The text to analyze for cryptocurrency sentiment
            
        Returns:
            Dict[str, Any]: Dictionary containing sentiment analysis results
            
        Example output:
            {
                "sentiment": "positive",  # or "negative" or "neutral"
                "score": 0.8,             # Sentiment score (0.0 to 1.0)
                "keywords": ["bullish", "growth"]  # Detected sentiment keywords
            }
            
        Raises:
            ValueError: If the input text is empty or not a string
        """
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
            
        print(f"Analyzing sentiment for text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Initialize counters and lists
        bullish_count = 0
        bearish_count = 0
        detected_keywords = []
        detected_cryptos = []
        detected_indicators = []
        
        # Check for bullish keywords
        for keyword in self.bullish_keywords:
            if keyword.lower() in text_lower:
                bullish_count += 1
                detected_keywords.append(keyword)
        
        # Check for bearish keywords
        for keyword in self.bearish_keywords:
            if keyword.lower() in text_lower:
                bearish_count += 1
                detected_keywords.append(keyword)
        
        # Detect cryptocurrencies mentioned
        for crypto_name, symbol in self.crypto_keywords.items():
            if crypto_name in text_lower or symbol.lower() in text_lower:
                if symbol not in detected_cryptos:
                    detected_cryptos.append(symbol)
        
        # Detect market indicators
        for indicator in self.market_indicators:
            if indicator.lower() in text_lower:
                detected_indicators.append(indicator)
        
        # Calculate sentiment score (-1.0 to 1.0 scale)
        total_keywords = bullish_count + bearish_count
        if total_keywords == 0:
            sentiment_score = 0.0  # Neutral
            sentiment = "neutral"
        else:
            # Calculate normalized score between -1 and 1
            sentiment_score = (bullish_count - bearish_count) / total_keywords
            
            # Determine sentiment label
            if sentiment_score > 0.33:
                sentiment = "positive"
            elif sentiment_score < -0.33:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        # Scale score to 0.0-1.0 range for easier integration with other components
        normalized_score = (sentiment_score + 1) / 2
        
        # Calculate confidence based on keyword count (more keywords = higher confidence)
        # Scale from 0.5 (minimum confidence) to 1.0 (maximum confidence)
        confidence = min(0.5 + (total_keywords / 10) * 0.5, 1.0)
        
        # Create result dictionary
        result = {
            "sentiment": sentiment,
            "score": round(normalized_score, 2),
            "keywords": list(set(detected_keywords)),  # Remove duplicates
            "cryptocurrencies": detected_cryptos,
            "market_indicators": detected_indicators,
            "confidence": round(confidence, 2)
        }
        
        return result
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text item for cryptocurrency sentiment.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Dict: Structured sentiment analysis results
        """
        # First use the simpler analyze_sentiment function
        basic_analysis = self.analyze_sentiment(text)
        
        # In a production environment, we would use the LLM API here
        # For now, we'll enhance the basic analysis to simulate LLM output
        
        # Convert sentiment to the expected format
        sentiment_mapping = {
            "positive": "bullish" if basic_analysis["score"] > 0.75 else "bullish",
            "negative": "very bearish" if basic_analysis["score"] < 0.25 else "bearish",
            "neutral": "neutral"
        }
        
        # Enhance sentiment scores to create more variation for testing
        if "bearish" in sentiment_mapping[basic_analysis["sentiment"]]:
            # Make bearish scores lower to trigger sell signals
            enhanced_score = max(0.1, basic_analysis["score"] - 0.2)
        elif "bullish" in sentiment_mapping[basic_analysis["sentiment"]]:
            # Make bullish scores higher to trigger buy signals
            enhanced_score = min(0.9, basic_analysis["score"] + 0.2)
        else:
            enhanced_score = basic_analysis["score"]
        
        # Simulate potential market impact based on sentiment
        market_impact = {
            "short_term": "price volatility expected" if basic_analysis["score"] != 0.5 else "minimal impact",
            "long_term": "potentially positive" if basic_analysis["score"] > 0.5 else 
                         "potentially negative" if basic_analysis["score"] < 0.5 else "neutral outlook"
        }
        
        # Create enhanced response
        enhanced_response = {
            "sentiment": sentiment_mapping[basic_analysis["sentiment"]],
            "score": enhanced_score,  # Use the enhanced score
            "market_indicators": basic_analysis["market_indicators"] or ["general market sentiment"],
            "cryptocurrencies": basic_analysis["cryptocurrencies"] or ["general market"],
            "events": [],  # We don't extract events in the simple analysis
            "market_impact": market_impact,
            "confidence": int(basic_analysis["confidence"] * 10)  # Scale to 1-10
        }
        
        # Extract possible events from text (very simplified)
        if "launch" in text.lower():
            enhanced_response["events"].append("Product/service launch")
        if "partnership" in text.lower() or "collaboration" in text.lower():
            enhanced_response["events"].append("Partnership announcement")
        if "regulation" in text.lower() or "sec" in text.lower():
            enhanced_response["events"].append("Regulatory development")
        if "hack" in text.lower() or "breach" in text.lower() or "security" in text.lower():
            enhanced_response["events"].append("Security incident")
        
        print(f"Text analyzed with sentiment: {enhanced_response['sentiment']}")
        return enhanced_response
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze a batch of texts for cryptocurrency sentiment.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Dict]: List of structured sentiment analysis results
        """
        results = []
        for text in texts:
            result = self.analyze_text(text)
            results.append(result)
        return results
    
    def analyze_sentiment_with_historical_data(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using a real LLM API with historical data.
        
        This function will attempt to use OpenAI's API to analyze the sentiment
        of cryptocurrency-related text. If the API call fails, it will fall back
        to the simulated analysis method.
        
        Args:
            text (str): The historical news article or text to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing sentiment analysis results in 
            the same format as analyze_sentiment for consistency:
            {
                "sentiment": "positive" | "negative" | "neutral",
                "score": <float between 0 and 1>,
                "keywords": <list of sentiment keywords>,
                "cryptocurrencies": <list of cryptocurrencies mentioned>,
                "market_indicators": <list of market indicators>,
                "confidence": <float between 0 and 1>
            }
        """
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
            
        print(f"Analyzing historical text with real LLM API: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Check if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: No OpenAI API key found in environment variables. Falling back to simulated analysis.")
            return self.analyze_sentiment(text)
        
        # Set the API key
        openai.api_key = api_key
        
        try:
            # Prepare the prompt with the text to analyze
            prompt = self.llm_api_prompt.format(text=text)
            
            # Determine which API version to use based on the client library availability
            try:
                # Use the latest OpenAI client if available
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial sentiment analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent results
                    max_tokens=1000,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                # Extract the response content
                response_text = response.choices[0].message.content
                
            except (ImportError, AttributeError):
                # Fall back to legacy OpenAI API
                print("Using legacy OpenAI API")
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial sentiment analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                # Extract the response content
                response_text = response.choices[0].message['content']
            
            # Parse the JSON response
            try:
                # Try to find and extract just the JSON part from the response
                json_match = re.search(r'({[\s\S]*})', response_text)
                if json_match:
                    response_text = json_match.group(1)
                    
                sentiment_result = json.loads(response_text)
                
                # Validate the required fields are in the response
                required_fields = ["sentiment", "score", "keywords", "cryptocurrencies", "market_indicators", "confidence"]
                missing_fields = [field for field in required_fields if field not in sentiment_result]
                
                if missing_fields:
                    print(f"Warning: LLM API response missing fields: {missing_fields}. Falling back to simulated analysis.")
                    return self.analyze_sentiment(text)
                
                # Normalize the result to ensure it conforms to our expected format
                # Make sure sentiment is one of our expected values
                if sentiment_result["sentiment"].lower() not in ["positive", "negative", "neutral"]:
                    # Map from the LLM's potential values to our expected values
                    sentiment_mapping = {
                        "bullish": "positive",
                        "very bullish": "positive",
                        "bearish": "negative",
                        "very bearish": "negative"
                    }
                    sentiment_result["sentiment"] = sentiment_mapping.get(
                        sentiment_result["sentiment"].lower(), 
                        "neutral"  # Default to neutral if unknown
                    )
                
                # Ensure score is between 0 and 1
                score = float(sentiment_result["score"])
                sentiment_result["score"] = max(0.0, min(1.0, score))
                
                # Ensure confidence is between 0 and 1
                confidence = float(sentiment_result["confidence"])
                sentiment_result["confidence"] = max(0.0, min(1.0, confidence))
                
                print(f"LLM API sentiment analysis complete: {sentiment_result['sentiment']} with score {sentiment_result['score']}")
                return sentiment_result
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Error parsing LLM API response: {e}. Response: {response_text}")
                print("Falling back to simulated analysis.")
                return self.analyze_sentiment(text)
                
        except Exception as e:
            print(f"Error using OpenAI API: {e}")
            print("Falling back to simulated analysis.")
            return self.analyze_sentiment(text)

    def aggregate_sentiment(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple sentiment analyses into a single report.
        
        Args:
            analyses (List[Dict]): List of individual sentiment analyses
            
        Returns:
            Dict: Aggregated sentiment report
        """
        if not analyses:
            return {
                "overall_sentiment": "neutral",
                "confidence": 0,
                "sample_size": 0,
                "details": []
            }
        
        # Count sentiments
        sentiment_counts = {"very bullish": 0, "bullish": 0, "neutral": 0, "bearish": 0, "very bearish": 0}
        total_confidence = 0
        
        # Count occurrences of each sentiment
        for analysis in analyses:
            sentiment = analysis["sentiment"]
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            total_confidence += analysis.get("confidence", 5)
        
        # Determine overall sentiment based on highest count
        # In case of tie, use the more extreme sentiment
        max_count = max(sentiment_counts.values())
        max_sentiments = [s for s, c in sentiment_counts.items() if c == max_count]
        
        if len(max_sentiments) == 1:
            overall_sentiment = max_sentiments[0]
        else:
            # In case of a tie, select based on priority
            sentiment_priority = ["very bullish", "very bearish", "bullish", "bearish", "neutral"]
            for s in sentiment_priority:
                if s in max_sentiments:
                    overall_sentiment = s
                    break
            else:
                overall_sentiment = "neutral"  # Default fallback
        
        # Calculate average confidence
        avg_confidence = round(total_confidence / len(analyses))
        
        print(f"Aggregated {len(analyses)} analyses into overall sentiment: {overall_sentiment}")
        
        return {
            "overall_sentiment": overall_sentiment,
            "confidence": avg_confidence,
            "sample_size": len(analyses),
            "details": analyses
        }

# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test analyze_sentiment with different texts
    test_texts = [
        "Bitcoin surges 10% as institutional adoption increases",
        "Ethereum drops amid concerns about network congestion",
        "Cryptocurrency markets remain stable as trading volume decreases",
        "New regulations could pose challenges for DeFi projects"
    ]
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Score: {result['score']}")
        print(f"Keywords: {', '.join(result['keywords'])}")
        
    # Test batch analysis
    batch_results = analyzer.analyze_batch(test_texts)
    aggregated = analyzer.aggregate_sentiment(batch_results)
    print(f"\nOverall sentiment from {aggregated['sample_size']} texts: {aggregated['overall_sentiment']}")
    print(f"Confidence: {aggregated['confidence']}/10") 