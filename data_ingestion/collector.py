"""
Data Collector Module

This module is responsible for fetching cryptocurrency data from various sources:
1. Price data from crypto exchanges
2. News data from news APIs
3. Social media data from relevant platforms
"""

import os
import requests
import pandas as pd
import json
import random
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CryptoDataCollector:
    """Collects cryptocurrency data from multiple sources."""

    def __init__(self):
        """Initialize the data collector with API keys."""
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.exchange_api_key = os.getenv("EXCHANGE_API_KEY")
        self.exchange_api_secret = os.getenv("EXCHANGE_API_SECRET")
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY")
        self.cryptocompare_api_key = os.getenv("CRYPTOCOMPARE_API_KEY")
        
        # API endpoints
        self.coingecko_endpoint = "https://api.coingecko.com/api/v3"
        self.cryptocompare_endpoint = "https://min-api.cryptocompare.com/data"
        self.newsapi_endpoint = "https://newsapi.org/v2/everything"

    def fetch_crypto_news(self, keywords: List[str] = None, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Retrieve cryptocurrency news headlines and articles.
        
        Args:
            keywords (List[str], optional): List of keywords to filter news by. Defaults to None.
            days_back (int, optional): Number of days to look back. Defaults to 7.
            
        Returns:
            List[Dict[str, Any]]: List of news articles with title, source, content, and published_at fields
            
        Raises:
            ConnectionError: If there's an issue connecting to the news API
            ValueError: If the API returns an error response
        """
        print(f"Fetching crypto news for the past {days_back} days...")
        # In a real implementation, this would call a news API
        # For now, we'll return dummy data
        try:
            # Simulate API call delay
            # time.sleep(1)
            # Generate dummy news data
            news_data = []
            start_date = datetime.now() - timedelta(days=days_back)

            # Sample news headlines with varying sentiment - with strong buy/sell signals
            headlines = [
                {"title": "Bitcoin Surges Past $60,000 as Institutional Adoption Grows", "sentiment": "positive"},
                {"title": "Ethereum Upgrade Delayed, Price Drops 10%", "sentiment": "negative"},
                {"title": "Regulatory Concerns Loom Over Cryptocurrency Markets", "sentiment": "negative"},
                {"title": "New DeFi Protocol Gains Traction Among Investors", "sentiment": "positive"},
                {"title": "Analysts Predict Cryptocurrency Bull Run in Coming Months", "sentiment": "positive"},
                {"title": "Major Exchange Faces Security Breach, Crypto Prices Tumble", "sentiment": "negative"},
                {"title": "Central Banks Explore Digital Currency Options", "sentiment": "neutral"},
                {"title": "Mining Difficulty Increases as Hash Rate Reaches All-Time High", "sentiment": "neutral"},
                {"title": "Whale Moves $1 Billion in Bitcoin, Market Remains Stable", "sentiment": "neutral"},
                {"title": "NFT Market Cools Down After Months of Explosive Growth", "sentiment": "negative"},
                # Add more varied headlines with explicit buy/sell signals
                {"title": "BREAKING: Bitcoin Surges 20% in Massive Rally, Experts Call It Beginning of Bull Run", "sentiment": "very positive"},
                {"title": "Major Investment Bank Allocates $1B to Cryptocurrency, Triggering Market Surge", "sentiment": "very positive"},
                {"title": "URGENT: Market Crash as Regulatory Crackdown Wipes 30% Off Crypto Valuations", "sentiment": "very negative"},
                {"title": "Panic Selling: Investors Flee Crypto Markets Amid Systemic Risk Concerns", "sentiment": "very negative"},
                {"title": "Historic Bull Run: Cryptocurrency Market Cap Reaches New All-Time High", "sentiment": "very positive"}
            ]

            # Generate random news articles over the time period
            sources = ["CryptoNews", "CoinDesk", "Bloomberg", "Reuters", "CNBC", "CoinTelegraph"]

            for i in range(len(headlines)):
                # Create a random date within the range
                days_offset = random.randint(0, days_back)
                pub_date = start_date + timedelta(days=days_offset)

                # Select a headline
                headline = headlines[i]

                # Create the article
                article = {
                    "title": headline["title"],
                    "source": random.choice(sources),
                    "content": f"This is a sample content for the article titled '{headline['title']}'. In a real implementation, this would contain the actual news article text.",
                    "published_at": pub_date.isoformat(),
                    "_sentiment": headline["sentiment"]  # Hidden field for testing
                }

                # Filter by keywords if provided
                if keywords:
                    if any(keyword.lower() in headline["title"].lower() for keyword in keywords):
                        news_data.append(article)
                else:
                    news_data.append(article)

            return news_data

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to news API: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing news data: {str(e)}")

    def fetch_historical_data(self, crypto_symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve historical price data for a given cryptocurrency.

        Args:
            crypto_symbol (str): Symbol of the cryptocurrency (e.g., 'BTC')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.

        Returns:
            pd.DataFrame: DataFrame with columns for date, open, high, low, close, and volume

        Raises:
            ConnectionError: If there's an issue connecting to the exchange API
            ValueError: If the API returns an error response or if date format is incorrect
        """
        # Validate input parameters
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            if end_date:
                end = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end = datetime.now()
        except ValueError:
            raise ValueError("Date format should be YYYY-MM-DD")

        print(f"Fetching historical data for {crypto_symbol} from {start_date} to {end_date or 'today'}")

        try:
            # In a real implementation, this would call an exchange API (e.g., using ccxt library)
            # For now, generate dummy data

            # Calculate number of days
            delta = end - start
            num_days = delta.days + 1

            # Generate dates
            dates = [start + timedelta(days=i) for i in range(num_days)]

            # Set base price based on the crypto symbol
            base_price = 0
            if crypto_symbol.upper() == "BTC":
                base_price = 40000  # Base price for Bitcoin
            elif crypto_symbol.upper() == "ETH":
                base_price = 2000   # Base price for Ethereum
            else:
                base_price = 100    # Default base price

            # Generate price data with some randomness and trend
            data = []
            current_price = base_price
            for date in dates:
                # Add a small random change to the price
                change_pct = random.uniform(-0.03, 0.03)  # -3% to +3%

                # Add a trend bias
                if random.random() > 0.5:  # Slight bullish bias
                    change_pct += 0.01

                # Calculate price change
                change = current_price * change_pct

                # Calculate prices for the day
                open_price = current_price
                close_price = current_price + change
                high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
                low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
                volume = random.randint(1000, 10000) * base_price / 100

                # Add to data
                data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume
                })

                # Update current price for next day
                current_price = close_price

            # Convert to DataFrame
            df = pd.DataFrame(data)

            return df

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to exchange API: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing historical data: {str(e)}")

    def fetch_live_historical_data(self, crypto_symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve live historical price data for a given cryptocurrency from a real API.
        
        Args:
            crypto_symbol (str): Symbol of the cryptocurrency (e.g., 'BTC')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.
        
        Returns:
            pd.DataFrame: DataFrame with columns for date, open, high, low, close, and volume
            
        Raises:
            ConnectionError: If there's an issue connecting to the API
            ValueError: If the API returns an error response or date format is incorrect
            RuntimeError: If no API key is provided when required
        """
        # Validate input parameters
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            if end_date:
                end = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end = datetime.now()
        except ValueError:
            raise ValueError("Date format should be YYYY-MM-DD")
        
        # Convert crypto symbol to expected format (lowercase for CoinGecko)
        crypto_id = crypto_symbol.lower()
        if crypto_id == 'btc':
            crypto_id = 'bitcoin'
        elif crypto_id == 'eth':
            crypto_id = 'ethereum'
        
        logger.info(f"Fetching live historical data for {crypto_symbol} from {start_date} to {end_date or 'today'}")
        
        # Try CoinGecko API first
        try:
            return self._fetch_from_coingecko(crypto_id, start, end)
        except Exception as e:
            logger.warning(f"CoinGecko API failed: {str(e)}. Trying CryptoCompare...")
            
            # Fallback to CryptoCompare
            try:
                return self._fetch_from_cryptocompare(crypto_symbol.upper(), start, end)
            except Exception as e2:
                logger.error(f"All API attempts failed: CoinGecko: {str(e)}, CryptoCompare: {str(e2)}")
                raise ConnectionError(f"Failed to fetch live historical data: {str(e2)}")
    
    def _fetch_from_coingecko(self, crypto_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Helper method to fetch data from CoinGecko API."""
        # Convert dates to UNIX timestamps (milliseconds)
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # Prepare API URL
        url = f"{self.coingecko_endpoint}/coins/{crypto_id}/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': start_timestamp,
            'to': end_timestamp
        }
        
        # Add API key if available
        if self.coingecko_api_key:
            params['x_cg_pro_api_key'] = self.coingecko_api_key
        
        # Make the API request
        response = requests.get(url, params=params)
        
        # Check for errors
        if response.status_code != 200:
            raise ValueError(f"CoinGecko API error: {response.status_code} - {response.text}")
        
        # Parse the response
        data = response.json()
        
        # CoinGecko returns separate arrays for prices, market caps, and volumes
        # We need to convert this to OHLC format
        if 'prices' not in data or not data['prices']:
            raise ValueError(f"No price data returned for {crypto_id}")
        
        # Create a DataFrame with timestamp and close price
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        
        # Convert timestamp from milliseconds to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # For CoinGecko, we only get closing prices, so we'll estimate OHLC
        # Note: This is a simplification, in a production system you'd want to use proper OHLC data
        df['open'] = df['close'].shift(1)
        df['high'] = df['close'] * 1.01  # Estimate: 1% higher than close
        df['low'] = df['close'] * 0.99   # Estimate: 1% lower than close
        
        # Handle the first row with missing open
        if len(df) > 0:
            df.loc[df.index[0], 'open'] = df.loc[df.index[0], 'close']
        
        # Add volume data if available
        if 'total_volumes' in data and data['total_volumes']:
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
            # Merge with the main dataframe
            df = pd.merge_asof(df, volumes_df, left_on='timestamp', right_on='timestamp')
        else:
            # If volume data is not available, set volumes to NaN
            df['volume'] = float('nan')
        
        # Clean up the DataFrame
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        # Convert date to string format for consistency with other methods
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        return df
    
    def _fetch_from_cryptocompare(self, crypto_symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Helper method to fetch data from CryptoCompare API."""
        # Calculate number of days
        days = (end_date - start_date).days + 1
        
        # Prepare API URL
        url = f"{self.cryptocompare_endpoint}/v2/histoday"
        params = {
            'fsym': crypto_symbol,
            'tsym': 'USD',
            'limit': min(2000, days),  # API has a limit of 2000 datapoints
            'toTs': int(end_date.timestamp())
        }
        
        # Add API key if available
        if self.cryptocompare_api_key:
            params['api_key'] = self.cryptocompare_api_key
        
        # Make the API request
        response = requests.get(url, params=params)
        
        # Check for errors
        if response.status_code != 200:
            raise ValueError(f"CryptoCompare API error: {response.status_code} - {response.text}")
        
        # Parse the response
        data = response.json()
        
        # Check if data is valid
        if data['Response'] != 'Success' or 'Data' not in data or 'Data' not in data['Data']:
            raise ValueError(f"Invalid response from CryptoCompare: {data}")
        
        # Extract the price data
        price_data = data['Data']['Data']
        
        # Create a DataFrame
        df = pd.DataFrame(price_data)
        
        # Rename columns to match our standard format
        df = df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volumefrom': 'volume'
        })
        
        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Clean up the DataFrame
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        # Convert date to string format for consistency with other methods
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        return df

    def fetch_live_crypto_news(self, keywords: List[str] = None, from_date: str = None, to_date: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve live cryptocurrency news articles from a real API.
        
        Args:
            keywords (List[str], optional): List of keywords to filter news by. Defaults to None.
            from_date (str, optional): Start date in YYYY-MM-DD format. Defaults to 7 days ago.
            to_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.
            
        Returns:
            List[Dict[str, Any]]: List of news articles with title, source, content, and published_at fields
            
        Raises:
            ConnectionError: If there's an issue connecting to the news API
            ValueError: If the API returns an error response
            RuntimeError: If no API key is provided
        """
        if not self.news_api_key:
            raise RuntimeError("NEWS_API_KEY is required for fetching live news")
        
        # Set default dates if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
            
        # Default keywords if none provided
        if not keywords or len(keywords) == 0:
            keywords = ["cryptocurrency", "bitcoin", "ethereum", "crypto"]
        
        logger.info(f"Fetching live crypto news for keywords {keywords} from {from_date} to {to_date}")
        
        try:
            # Try fetching from NewsAPI
            return self._fetch_from_newsapi(keywords, from_date, to_date)
        except Exception as e:
            logger.error(f"NewsAPI failed: {str(e)}")
            # If no fallback is available, raise the error
            raise ConnectionError(f"Failed to fetch live news data: {str(e)}")
    
    def _fetch_from_newsapi(self, keywords: List[str], from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Helper method to fetch news from NewsAPI."""
        # Construct the query string from keywords
        query = " OR ".join(keywords)
        
        # Prepare API URL and parameters
        url = self.newsapi_endpoint
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': self.news_api_key
        }
        
        # Make the API request
        response = requests.get(url, params=params)
        
        # Check for errors
        if response.status_code != 200:
            raise ValueError(f"NewsAPI error: {response.status_code} - {response.text}")
        
        # Parse the response
        data = response.json()
        
        # Check if the response contains articles
        if 'articles' not in data or not data['articles']:
            logger.warning(f"No news articles found for query: {query}")
            return []
        
        # Transform the response to our expected format
        articles = []
        for article in data['articles']:
            processed_article = {
                'title': article.get('title', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'content': article.get('content', article.get('description', '')),
                'published_at': article.get('publishedAt', ''),
                'url': article.get('url', '')
            }
            articles.append(processed_article)
        
        logger.info(f"Retrieved {len(articles)} news articles")
        return articles

    def fetch_news_data(self, keywords, days_back=7):
        """
        Fetch news articles related to cryptocurrencies.

        Args:
            keywords (list): List of keywords to search for
            days_back (int): Number of days to look back

        Returns:
            list: List of news articles
        """
        # Redirect to new function for compatibility
        return self.fetch_crypto_news(keywords, days_back)

    def fetch_price_data(self, symbol, start_date, end_date=None):
        """
        Fetch historical price data for a cryptocurrency.

        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.

        Returns:
            pd.DataFrame: Dataframe with price data
        """
        # Redirect to new function for compatibility
        return self.fetch_historical_data(symbol, start_date, end_date)

    def fetch_social_media_data(self, keywords, platform="twitter", days_back=3):
        """
        Fetch social media posts related to cryptocurrencies.

        Args:
            keywords (list): List of keywords to search for
            platform (str): Social media platform to search
            days_back (int): Number of days to look back

        Returns:
            list: List of social media posts
        """
        # TODO: Implement social media API integration
        print(f"Fetching {platform} posts for {keywords} from the last {days_back} days")
        return []  # Placeholder

    def load_historical_data_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Load historical cryptocurrency price data from a file.
        
        Args:
            file_path (str): Path to the file containing historical data (CSV or JSON)
            
        Returns:
            pd.DataFrame: DataFrame with columns for date, open, high, low, close, volume
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is unsupported or if there are parsing issues
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Historical data file not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                print(f"Loading historical data from CSV: {file_path}")
                df = pd.read_csv(file_path)
                
                # Validate required columns
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"CSV file is missing required columns: {', '.join(missing_cols)}")
                
                # Ensure date column is in the right format
                if 'date' in df.columns:
                    try:
                        df['date'] = pd.to_datetime(df['date'])
                        # Convert back to string format for consistency with other methods
                        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                    except Exception as e:
                        print(f"Warning: Could not parse date column: {e}")
                        # Keep the dates as is if they can't be parsed
                
            elif file_extension == '.json':
                print(f"Loading historical data from JSON: {file_path}")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Handle both array and object formats
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict) and 'data' in data:
                    # Some APIs return data in a nested structure
                    df = pd.DataFrame(data['data'])
                else:
                    raise ValueError("JSON file has an unsupported structure")
                
                # Validate required columns
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"JSON file is missing required columns: {', '.join(missing_cols)}")
                
                # Ensure date column is in the right format
                if 'date' in df.columns:
                    try:
                        df['date'] = pd.to_datetime(df['date'])
                        # Convert back to string format for consistency with other methods
                        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                    except Exception as e:
                        print(f"Warning: Could not parse date column: {e}")
                        # Keep the dates as is if they can't be parsed
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Use .csv or .json")
            
            # Verify that numeric columns are indeed numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date if available
            if 'date' in df.columns:
                df = df.sort_values('date')
            
            print(f"Successfully loaded {len(df)} records from {file_path}")
            return df
            
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing file {file_path}: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON in file {file_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading historical data from {file_path}: {str(e)}")

    def load_historical_crypto_news(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load historical cryptocurrency news articles from a file.
        
        Args:
            file_path (str): Path to the file containing historical news data (JSON or CSV)
            
        Returns:
            List[Dict[str, Any]]: List of news articles with title, source, content, and published_at fields
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is unsupported or if there are parsing issues
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Historical news file not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            news_data = []
            
            if file_extension == '.json':
                print(f"Loading historical news from JSON: {file_path}")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Handle both array and object formats
                if isinstance(data, list):
                    news_data = data
                elif isinstance(data, dict) and 'articles' in data:
                    # Some news APIs return data in a nested structure
                    news_data = data['articles']
                elif isinstance(data, dict) and 'data' in data:
                    news_data = data['data']
                else:
                    raise ValueError("JSON file has an unsupported structure")
                
            elif file_extension == '.csv':
                print(f"Loading historical news from CSV: {file_path}")
                df = pd.read_csv(file_path)
                
                # Required columns for news articles
                required_cols = ['title', 'content', 'published_at']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"CSV file is missing required columns: {', '.join(missing_cols)}")
                
                # Convert DataFrame to list of dictionaries
                news_data = df.to_dict(orient='records')
                
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Use .csv or .json")
            
            # Validate the structure of the news data
            validated_news = []
            for article in news_data:
                # Check for required fields
                if not all(key in article for key in ['title', 'content', 'published_at']):
                    print(f"Warning: Skipping article missing required fields: {article.get('title', 'Unknown')}")
                    continue
                
                # Ensure published_at is a valid date
                try:
                    # Parse the date to validate it (but keep original string)
                    datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    # If not ISO format, try to convert it
                    try:
                        date_obj = pd.to_datetime(article['published_at'])
                        article['published_at'] = date_obj.isoformat()
                    except:
                        print(f"Warning: Invalid date format in article: {article['title']}")
                        # Keep the original format if conversion fails
                
                # Add source if missing
                if 'source' not in article:
                    article['source'] = "Historical Data"
                
                validated_news.append(article)
            
            if not validated_news:
                raise ValueError("No valid news articles found in the file")
            
            print(f"Successfully loaded {len(validated_news)} news articles from {file_path}")
            return validated_news
            
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file {file_path}: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON in file {file_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading historical news from {file_path}: {str(e)}")

    def save_data(self, data, data_type, filename=None):
        """
        Save collected data to disk.
        
        Args:
            data: The data to save
            data_type (str): Type of data ('price', 'news', 'social')
            filename (str, optional): Custom filename. Defaults to auto-generated.

        Returns:
            str: Path to saved data
        """
        # TODO: Implement data saving logic
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type}_{timestamp}.csv"

        print(f"Saving {data_type} data to {filename}")
        return filename

# Example usage
if __name__ == "__main__":
    collector = CryptoDataCollector()

    # Test fetch_live_historical_data
    try:
        print("\nTesting live historical data fetch...")
        live_data = collector.fetch_live_historical_data('BTC', '2023-01-01', '2023-01-10')
        print(f"Retrieved live historical data with {len(live_data)} rows")
        if not live_data.empty:
            print("Sample data:")
            print(live_data.head())
    except Exception as e:
        print(f"Error testing live data: {str(e)}")
    
    # Test fetch_live_crypto_news
    try:
        print("\nTesting live crypto news fetch...")
        live_news = collector.fetch_live_crypto_news(['Bitcoin', 'Ethereum'])
        print(f"Retrieved {len(live_news)} live news articles")
        if live_news:
            print("Sample news item:", live_news[0])
    except Exception as e:
        print(f"Error testing live news: {str(e)}")
    
    # Fall back to simulated data tests if live APIs fail
    print("\nTesting simulated data (fallback)...")
    
    # Test fetch_crypto_news
    news = collector.fetch_crypto_news(["Bitcoin", "Ethereum"], days_back=5)
    print(f"Retrieved {len(news)} news articles")
    if news:
        print("Sample news item:", news[0])

    # Test fetch_historical_data
    hist_data = collector.fetch_historical_data("BTC", "2023-01-01", "2023-01-10")
    print(f"Retrieved historical data with {len(hist_data)} rows")
    if not hist_data.empty:
        print("Sample data:")
        print(hist_data.head())
 