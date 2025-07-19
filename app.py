import os
import re
import logging
import sys
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import feedparser
import random
import gunicorn
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, abort
from flask_caching import Cache
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from newsapi.newsapi_client import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func
from nsepython import nse_fiidii
from werkzeug.utils import secure_filename
from flask_wtf.csrf import CSRFProtect
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.exc import SQLAlchemyError
from flask_migrate import Migrate
from whitenoise import WhiteNoise
import torch  # Added for transformers support

load_dotenv()
app = Flask(__name__)

csrf = CSRFProtect(app)
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # CSRF token expires in 1 hour
@app.after_request
def apply_security_headers(response):
    """Add security headers to every response"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store, max-age=0'
    return response

app.secret_key = os.getenv('SECRET_KEY', 'fallback_secret_key')
# Improved database configuration with connection pooling
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL or 'sqlite:///stocks.db'
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 5,
    'max_overflow': 10,
    'pool_pre_ping': True,
    'pool_recycle': 300  # Recycle connections after 5 minutes
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True, 'pool_recycle': 300}
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
app.config['CACHE_TYPE'] = 'RedisCache'
app.config['CACHE_REDIS_URL'] = redis_url
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # 5 minutes
app.config['CACHE_KEY_PREFIX'] = 'howismystock_'

cache = Cache(app)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')
app.wsgi_app.add_files('static/css/', prefix='css/')
app.wsgi_app.add_files('static/js/', prefix='js/')
app.wsgi_app.add_files('static/images/', prefix='images/')

newsapi = NewsApiClient(api_key=os.getenv('NEWSAPI_KEY', 'fallback_newsapi_key'))
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', 'fallback_alpha_vantage_key')
FINNHUB_KEY = os.getenv('FINNHUB_KEY', 'fallback_finnhub_key')

# ===== ENVIRONMENT VALIDATION =====
required_vars = ['SECRET_KEY', 'NEWSAPI_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars and not app.debug:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing_vars)}. "
        "Please set these in your Railway environment variables."
    )

NEWS_SOURCES = {
    "primary": [
        {"name": "NewsAPI", "function": "fetch_newsapi"},
        {"name": "AlphaVantage", "function": "fetch_alphavantage"},
    ],
    "secondary": [
        {"name": "MoneyControl", "function": "scrape_moneycontrol_news"},
        {"name": "Investing.com", "function": "scrape_investing_news"},
        {"name": "BusinessStandard", "function": "parse_rss_feed"},
        {"name": "EconomicTimes", "function": "fetch_economictimes"},
        {"name": "FinancialExpress", "function": "fetch_financialexpress"},
    ],
    "tertiary": [
        {"name": "GoogleNews", "function": "fetch_googlenews"},
        {"name": "YahooFinance", "function": "fetch_yahoofinance"},
        {"name": "Reuters", "function": "fetch_reuters"},
    ]
}

def handle_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except SQLAlchemyError as e:
            db.session.rollback()
            logger.error(f"Database error: {str(e)}")
            flash('Database operation failed', 'error')
            return redirect(url_for('home'))
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            flash('Market data service unavailable', 'error')
            return redirect(url_for('home'))
        except yf.YFinanceError as e:
            logger.error(f"YFinance error: {str(e)}")
            flash('Stock data service unavailable', 'error')
            return redirect(url_for('home'))
        except Exception as e:
            logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
            flash('System temporarily unavailable. Please try again later.', 'error')
            return redirect(url_for('home'))
    return wrapper

try:
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    logger.info("Sentiment analysis model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentiment model: {str(e)}")
    sentiment_pipeline = None

try:
    nse_stocks = pd.read_csv('NSE_list.csv')
    nse_stocks.columns = nse_stocks.columns.str.strip()
    stock_list = nse_stocks.to_dict('records')
    company_to_symbol = nse_stocks.set_index('NAME OF COMPANY')['SYMBOL'].to_dict()
    symbol_to_company = nse_stocks.set_index('SYMBOL')['NAME OF COMPANY'].to_dict()
    all_companies = nse_stocks['NAME OF COMPANY'].dropna().unique().tolist()
    logger.info(f"Loaded {len(stock_list)} stocks from NSE_list.csv")
except Exception as e:
    logger.error(f"Error loading stock data: {str(e)}")
    stock_list = []
    company_to_symbol = {}
    symbol_to_company = {}
    all_companies = []

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    full_name = db.Column(db.String(100), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    notification_prefs = db.Column(db.JSON, default={'email': True, 'app': True})

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    name = db.Column(db.String(100))
    current_price = db.Column(db.Float)
    market_cap = db.Column(db.Float)
    pe_ratio = db.Column(db.Float)
    dividend_yield = db.Column(db.Float)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    sector = db.Column(db.String(50))
    industry = db.Column(db.String(100))

class Vote(db.Model):
    __table_args__ = (db.Index('idx_vote_stock_user', 'stock_symbol', 'user_id'), db.Index('idx_vote_timestamp', 'timestamp'))
    id = db.Column(db.Integer, primary_key=True)
    stock_symbol = db.Column(db.String(10), nullable=False)
    vote = db.Column(db.String(10), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='votes')

class SentimentAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stock_symbol = db.Column(db.String(10), nullable=False)
    source = db.Column(db.String(20))
    content = db.Column(db.Text)
    score = db.Column(db.Float)
    label = db.Column(db.String(10))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    confidence = db.Column(db.Float)

class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    stock_symbol = db.Column(db.String(10))
    search_query = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='search_history')

class Watchlist(db.Model):
    __tablename__ = 'watchlist'
    __table_args__ = (db.Index('idx_watchlist_user_stock', 'user_id', 'stock_symbol'),)
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    stock_symbol = db.Column(db.String(10), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)  # This is the missing column
    notes = db.Column(db.Text, nullable=True)
    
    user = db.relationship('User', backref='watchlist_items')
    
class Discussion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stock_symbol = db.Column(db.String(10), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('discussion.id'))  # For replies
    sentiment = db.Column(db.String(10))  # bullish/bearish/neutral
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    likes = db.Column(db.Integer, default=0)

    user = db.relationship('User', backref='discussions')
    replies = db.relationship('Discussion', backref=db.backref('parent', remote_side=[id]))

def format_indian_currency(value):
    if value == 'N/A': return 'N/A'
    try:
        value = float(value)
        if value >= 1e7: return f'₹{value/1e7:.2f} Cr'
        elif value >= 1e5: return f'₹{value/1e5:.2f} L'
        return f'₹{value:.2f}'
    except (ValueError, TypeError): return 'N/A'

def is_market_open():
    now = datetime.now()
    if now.weekday() >= 5: return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close

def get_cache_timeout():
    now = datetime.now()
    if now.hour >= 3: return 3600
    else: return 21600

def parse_rss_feed(url):
    try:
        feed = feedparser.parse(url)
        news_items = []
        for entry in feed.entries[:5]:
            news_items.append({
                'title': entry.title,
                'url': entry.link,
                'publishedAt': entry.published,
                'source': {'name': 'Business Standard'}
            })
        return news_items
    except Exception as e:
        logger.error(f"Error parsing RSS feed: {e}")
        return []

def is_highly_relevant(title, content, company_name, symbol):
    patterns = [
        rf"\b{re.escape(company_name)}\b",
        rf"\b{re.escape(symbol)}\b",
        rf"\b{re.escape(company_name.split()[0])}\b",
    ]
    irrelevant_patterns = [
        "market update", "sensex", "nifty", "stock market", 
        "market highlights", "top gainers", "general market"
    ]
    title_lower = title.lower()
    content_lower = content.lower() if content else ""
    has_company = any(re.search(pattern, title_lower, re.IGNORECASE) or 
                   re.search(pattern, content_lower, re.IGNORECASE) 
                   for pattern in patterns)
    is_irrelevant = any(ip in title_lower for ip in irrelevant_patterns)
    return has_company and not is_irrelevant

def fetch_combined_news(company_name, stock_symbol):
    results = []
    source_errors = []
    
    # Stage 1: Try primary sources
    for source in NEWS_SOURCES["primary"]:
        try:
            func = globals()[source["function"]]
            articles = func(company_name, stock_symbol)
            if articles:  # Only process if we got results
                relevant = [a for a in articles if is_highly_relevant(a["title"], a.get("content", ""), company_name, stock_symbol)]
                results.extend(relevant)
                logger.info(f"Got {len(relevant)} relevant articles from {source['name']}")
                
                if len(results) >= 5:  # Early exit if we have enough
                    break
        except Exception as e:
            err_msg = f"{source['name']} error: {str(e)}"
            logger.error(err_msg)
            source_errors.append(err_msg)
    
    # Stage 2: Try secondary sources if needed
    if len(results) < 5:
        for source in NEWS_SOURCES["secondary"]:
            try:
                func = globals()[source["function"]]
                articles = func(company_name, stock_symbol)
                if articles:
                    relevant = [a for a in articles if is_highly_relevant(a["title"], a.get("content", ""), company_name, stock_symbol)]
                    results.extend(relevant)
                    if len(results) >= 5:
                        break
            except Exception as e:
                logger.error(f"{source['name']} error: {str(e)}")
    
    # Stage 3: Final fallback to tertiary sources
    if len(results) < 3:
        for source in NEWS_SOURCES["tertiary"]:
            try:
                func = globals()[source["function"]]
                articles = func(company_name, stock_symbol)
                if articles:
                    relevant = [a for a in articles if is_highly_relevant(a["title"], a.get("content", ""), company_name, stock_symbol)]
                    results.extend(relevant)
            except Exception as e:
                logger.error(f"{source['name']} error: {str(e)}")
    
    # Deduplicate results
    seen_urls = set()
    unique_results = []
    for article in results:
        if article["url"] not in seen_urls:
            seen_urls.add(article["url"])
            unique_results.append(article)
    
    return unique_results[:10]  # Return max 10 articles

def fetch_alphavantage(company_name, symbol):
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}.BSE&apikey={ALPHA_VANTAGE_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        articles = []
        for item in data.get("feed", [])[:10]:
            articles.append({
                "title": item.get("title", ""),
                "url": item.get("url", "#"),
                "publishedAt": item.get("time_published", ""),
                "source": {"name": item.get("source", "AlphaVantage")},
                "content": item.get("summary", "")
            })
        return articles
    except Exception as e:
        logger.error(f"AlphaVantage error: {str(e)}")
        return []

def fetch_googlenews(company_name, symbol):
    try:
        query = f"{company_name} {symbol} site:moneycontrol.com OR site:livemint.com OR site:business-standard.com"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:5]:
            articles.append({
                "title": entry.title,
                "url": entry.link,
                "publishedAt": entry.published,
                "source": {"name": "GoogleNews"}
            })
        return articles
    except Exception as e:
        logger.error(f"GoogleNews error: {str(e)}")
        return []

def fetch_yahoofinance(company_name, symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}.NS/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        for item in soup.select('li.js-stream-content'):
            title = item.select_one('h3').get_text(strip=True)
            link = item.select_one('a')['href']
            if not link.startswith('http'):
                link = f'https://finance.yahoo.com{link}'
            source = item.select_one('div span')['data-reactid'].split('.')[2]
            date = item.select_one('div span')['data-reactid'].split('.')[4]
            articles.append({
                "title": title,
                "url": link,
                "publishedAt": date,
                "source": {"name": source}
            })
        return articles[:5]
    except Exception as e:
        logger.error(f"YahooFinance error: {str(e)}")
        return []

def fetch_newsapi(company_name, symbol):
    try:
        query = f"{company_name} OR {symbol}"
        domains = [
            'economictimes.indiatimes.com', 
            'moneycontrol.com',
            'livemint.com',
            'business-standard.com',
            'financialexpress.com',
            'bloombergquint.com',
            'thehindubusinessline.com',
            'investing.com',
            'reuters.com',
            'bloomberg.com'
        ]
        news = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='relevancy',
            page_size=20,
            domains=','.join(domains))
        
        irrelevant_keywords = [
            'loan', 'credit card', 'personal finance', 
            'savings account', 'fixed deposit', 
            'interest rate', 'emi', 'home loan',
            'car loan', 'education loan'
        ]
        
        relevant_articles = []
        for article in news['articles']:
            title = article['title'].lower() if article['title'] else ''
            description = article['description'].lower() if article['description'] else ''
            content = article['content'].lower() if article['content'] else ''
            
            if any(keyword in title or keyword in description or keyword in content 
                  for keyword in irrelevant_keywords):
                continue
                
            relevant_articles.append({
                'title': article['title'],
                'url': article['url'],
                'publishedAt': article['publishedAt'],
                'source': {'name': article['source']['name']},
                'content': article.get('description', '') + ' ' + article.get('content', '')
            })
            
            if len(relevant_articles) >= 10:
                break
                
        return relevant_articles
    except Exception as e:
        logger.error(f"NewsAPI error: {str(e)}")
        return []

def fetch_economictimes(company_name, symbol):
    try:
        url = f"https://economictimes.indiatimes.com/markets/stocks/news/{symbol.lower()}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        for item in soup.select('.newsList li'):
            title = item.find('h3').get_text(strip=True)
            link = item.find('a')['href']
            if not link.startswith('http'):
                link = f'https://economictimes.indiatimes.com{link}'
            date = item.find('time').get_text(strip=True)
            articles.append({
                'title': title,
                'url': link,
                'publishedAt': date,
                'source': {'name': 'Economic Times'}
            })
        return articles[:5]
    except Exception as e:
        logger.error(f"EconomicTimes error: {str(e)}")
        return []

def fetch_financialexpress(company_name, symbol):
    try:
        url = f"https://www.financialexpress.com/?s={company_name.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        for item in soup.select('.article-list article'):
            title = item.select_one('h3 a').get_text(strip=True)
            link = item.select_one('h3 a')['href']
            date = item.select_one('time').get_text(strip=True)
            articles.append({
                'title': title,
                'url': link,
                'publishedAt': date,
                'source': {'name': 'Financial Express'}
            })
        return articles[:5]
    except Exception as e:
        logger.error(f"FinancialExpress error: {str(e)}")
        return []

def fetch_reuters(company_name, symbol):
    try:
        url = f"https://www.reuters.com/search/news?blob={company_name.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        for item in soup.select('.search-result-content'):
            title = item.select_one('h3').get_text(strip=True)
            link = "https://www.reuters.com" + item.select_one('a')['href']
            date = item.select_one('.search-result-timestamp').get_text(strip=True)
            articles.append({
                'title': title,
                'url': link,
                'publishedAt': date,
                'source': {'name': 'Reuters'}
            })
        return articles[:5]
    except Exception as e:
        logger.error(f"Reuters error: {str(e)}")
        return []

def scrape_moneycontrol_news(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        for item in soup.select('.newslist li'):
            title = item.find('h2').get_text(strip=True)
            link = item.find('a')['href']
            date = item.find('span').get_text(strip=True)
            news_items.append({
                'title': title,
                'url': link,
                'publishedAt': date,
                'source': {'name': 'MoneyControl'}
            })
        return news_items[:5]
    except Exception as e:
        logger.error(f"Error scraping MoneyControl: {e}")
        return []

def scrape_investing_news(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        for item in soup.select('.largeTitle .articleItem'):
            title = item.select_one('a.title').get_text(strip=True)
            link = "https://in.investing.com" + item.select_one('a.title')['href']
            date = item.select_one('.date').get_text(strip=True)
            news_items.append({
                'title': title,
                'url': link,
                'publishedAt': date,
                'source': {'name': 'Investing.com'}
            })
        return news_items[:5]
    except Exception as e:
        logger.error(f"Error scraping Investing.com: {e}")
        return []

def preprocess_text(text):
    if not text: return ""
    text = re.sub(r'\$[A-Za-z]+|\b[A-Z]{2,}\b(?=\s)|\.NS\b', '', text)
    replacements = {"Q1": "quarter one", "Q2": "quarter two", "Q3": "quarter three", "Q4": "quarter four", "EPS": "earnings per share", "P/E": "price to earnings ratio"}
    for abbr, full in replacements.items():
        text = text.replace(abbr, full)
    return text.strip()

@lru_cache(maxsize=1000)
def analyze_sentiment(text):
    if not text or not isinstance(text, str):
        return 0, "Neutral"
    
    text = preprocess_text(text)
    
    # Enhanced keyword analysis
    bullish_phrases = [
        "beat estimates", "raised guidance", "bullish", "positive", 
        "surged", "jumped", "growth", "outperform", "upgrade",
        "strong results", "profit increase", "dividend increase"
    ]
    
    bearish_phrases = [
        "missed estimates", "cut guidance", "bearish", "negative",
        "plummeted", "slashed", "decline", "downgrade", "weak",
        "loss", "profit warning", "dividend cut"
    ]
    
    text_lower = text.lower()
    bull_count = sum(1 for phrase in bullish_phrases if phrase in text_lower)
    bear_count = sum(1 for phrase in bearish_phrases if phrase in text_lower)
    
    # Clear sentiment based on keywords
    if bull_count > bear_count + 1:
        return 1, "Bullish"
    elif bear_count > bull_count + 1:
        return -1, "Bearish"
    
    # Use model if available
    if sentiment_pipeline:
        try:
            result = sentiment_pipeline(text)
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            label_map = {1: 'Bullish', 0: 'Neutral', -1: 'Bearish'}
            label = result[0]['label'].lower()
            return sentiment_map[label], label_map[sentiment_map[label]]
        except Exception as e:
            logger.error(f"Sentiment model failed: {str(e)}")
    
    return 0, "Neutral"

def populate_initial_data():
    with app.app_context():
        try:
            if Stock.query.count() == 0 and stock_list:
                app.logger.info("Populating initial stock data...")
                
                # Batch insert for better performance
                stocks_to_add = []
                for stock in stock_list[:100]:  # Limit to 100 initially
                    if not Stock.query.filter_by(symbol=stock['SYMBOL']).first():
                        try:
                            yf_stock = yf.Ticker(stock['SYMBOL'] + ".NS")
                            info = yf_stock.info
                            
                            stocks_to_add.append(Stock(
                                symbol=stock['SYMBOL'],
                                name=stock['NAME OF COMPANY'],
                                current_price=info.get('currentPrice', 0),
                                market_cap=info.get('marketCap', 0),
                                pe_ratio=info.get('trailingPE', 0),
                                dividend_yield=info.get('dividendYield', 0),
                                sector=info.get('sector', 'N/A'),
                                industry=info.get('industry', 'N/A')
                            ))
                            
                            # Commit in batches of 20
                            if len(stocks_to_add) >= 20:
                                db.session.bulk_save_objects(stocks_to_add)
                                db.session.commit()
                                stocks_to_add = []
                                
                        except Exception as e:
                            app.logger.error(f"Error adding stock {stock.get('SYMBOL', 'UNKNOWN')}: {e}")
                            continue
                
                # Add any remaining stocks
                if stocks_to_add:
                    db.session.bulk_save_objects(stocks_to_add)
                    db.session.commit()
                    
                app.logger.info(f"Added {Stock.query.count()} stocks to database")
                
        except Exception as e:
            app.logger.error(f"Error in populate_initial_data: {e}")
            db.session.rollback()

@app.context_processor
def utility_processor():
    return dict(is_market_open=is_market_open)

@app.template_filter('format_datetime')
def format_datetime(value, format="%b %d, %Y %I:%M %p"):
    if isinstance(value, str):
        try: value = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError: return value
    if isinstance(value, datetime): return value.strftime(format)
    return value

@app.route('/')
def home():
    popular_stocks = Stock.query.order_by(Stock.market_cap.desc()).limit(10).all()
    recent_searches = []
    if 'user_id' in session:
        recent_searches = SearchHistory.query.filter_by(user_id=session['user_id']).order_by(SearchHistory.timestamp.desc()).limit(5).all()
    return render_template('index.html', stock_list=stock_list, all_companies=all_companies, company_to_symbol=company_to_symbol, popular_stocks=popular_stocks, recent_searches=recent_searches, now=datetime.now())

@app.route('/results', methods=['GET', 'POST'])
@handle_errors
def results():
    try:
        if request.method == 'GET':
            stock_symbol = request.args.get('stock_symbol')
            company_name = request.args.get('company_name')
        else:
            stock_symbol = request.form.get('stock_symbol')
            company_name = None

        if not stock_symbol:
            if company_name:
                stock_symbol = company_to_symbol.get(company_name)
                if not stock_symbol:
                    flash('Company not found. Please try another name.', 'error')
                    return redirect(url_for('home'))
            else:
                flash('Please select a stock', 'error')
                return redirect(url_for('home'))

        if 'user_id' in session:
            try:
                db.session.begin()
                search_entry = SearchHistory(
                    user_id=session['user_id'],
                    stock_symbol=stock_symbol,
                    search_query=request.form.get('search_query', '') if request.method == 'POST' else company_name or stock_symbol
                )
                db.session.add(search_entry)
                db.session.commit()
            except Exception as db_error:
                db.session.rollback()
                logger.error(f"Database error: {db_error}")

        try:
            stock_info, historical_data = fetch_stock_data(stock_symbol)
            if not stock_info:
                flash('Failed to fetch data for this stock. Please try another.', 'error')
                return redirect(url_for('home'))
        except Exception as fetch_error:
            logger.error(f"Stock fetch error: {fetch_error}")
            cached_stock = Stock.query.filter_by(symbol=stock_symbol).first()
            if cached_stock:
                stock_info = {
                    'currentPrice': cached_stock.current_price,
                    'sector': cached_stock.sector,
                    'industry': cached_stock.industry
                }
                historical_data = None
            else:
                flash('Service temporarily unavailable', 'error')
                return redirect(url_for('home'))

        company_name = symbol_to_company.get(stock_symbol, stock_symbol)
        technicals = fetch_technical_details(stock_symbol) or {}

        chart_data = None
        chart_type = 'historical'
        try:
            if is_market_open():
                chart_data = fetch_realtime_chart_data(stock_symbol)
                chart_type = 'realtime'
            else:
                chart_data = fetch_last_trading_day_data(stock_symbol)
                if not chart_data:
                    chart_data = {
                        'times': [date.strftime('%Y-%m-%d') for date in historical_data.index],
                        'prices': historical_data['Close'].tolist()
                    }
        except Exception as chart_error:
            logger.error(f"Chart error: {chart_error}")
            chart_data = {'times': [], 'prices': []}

        news_articles = []
        sentiment_labels = []
        sentiment_scores = []

        try:
            raw_news = fetch_combined_news(company_name, stock_symbol)
            for article in raw_news[:10]:  # Process max 10 articles
                if isinstance(article, dict):
                    # Ensure required fields exist
                    validated_article = {
                        'title': article.get('title', 'No title available'),
                        'url': article.get('url', '#'),
                        'publishedAt': article.get('publishedAt', ''),
                        'source': article.get('source', {'name': 'Unknown'})
                    }
                    news_articles.append(validated_article)
                    
                    # Analyze sentiment
                    if validated_article['title']:
                        score, label = analyze_sentiment(validated_article['title'])
                        sentiment_scores.append(score)
                        sentiment_labels.append(label)
        except Exception as news_error:
            logger.error(f"News processing error: {news_error}")
            # Fallback to empty lists if news processing fails

        # Calculate average sentiment
        avg_sentiment_value = sum(sentiment_scores)/len(sentiment_scores) if sentiment_scores else 0
        avg_sentiment_label = 'Bullish' if avg_sentiment_value > 0.2 else 'Bearish' if avg_sentiment_value < -0.2 else 'Neutral'
        recommendation = "Hold"
        try:
            pe_ratio = technicals.get('pe_ratio', 0)
            if isinstance(pe_ratio, str) and pe_ratio.replace('.', '').isdigit():
                pe_ratio = float(pe_ratio)
            if avg_sentiment_label == 'Bullish' and pe_ratio and pe_ratio < 25:
                recommendation = "Strong Buy"
            elif avg_sentiment_label == 'Bullish':
                recommendation = "Buy"
            elif avg_sentiment_label == 'Bearish' and pe_ratio and pe_ratio > 30:
                recommendation = "Strong Sell"
            elif avg_sentiment_label == 'Bearish':
                recommendation = "Sell"
        except Exception as rec_error:
            logger.error(f"Recommendation error: {rec_error}")

        # Voting system implementation - UPDATED SECTION
        votes = Vote.query.filter_by(stock_symbol=stock_symbol).all()
        
        # Create a unique key for this stock's mock data
        mock_data_key = f"mock_votes_{stock_symbol}"
        
        if len(votes) < 10:  # If not enough real votes
            # Try to get existing mock data from session
            vote_counts = session.get(mock_data_key)
            
            if not vote_counts:  # First time, generate new mock data
                if avg_sentiment_label.lower() == 'bullish':
                    base_weights = {'Buy': 0.6, 'Hold': 0.3, 'Sell': 0.1}
                elif avg_sentiment_label.lower() == 'bearish':
                    base_weights = {'Buy': 0.1, 'Hold': 0.3, 'Sell': 0.6}
                else:  # neutral
                    base_weights = {'Buy': 0.4, 'Hold': 0.4, 'Sell': 0.2}

                total_votes = random.randint(500, 5000)
                vote_counts = {
                    'Buy': int(base_weights['Buy'] * total_votes),
                    'Hold': int(base_weights['Hold'] * total_votes),
                    'Sell': total_votes - int(base_weights['Buy'] * total_votes) - int(base_weights['Hold'] * total_votes)
                }
                # Store this mock data for future requests
                session[mock_data_key] = vote_counts
        else:
            # Use real vote counts
            vote_counts = {'Buy': 0, 'Hold': 0, 'Sell': 0}
            for vote in votes:
                vote_counts[vote.vote] += 1

        fii_dii_data = []
        try:
            fii_dii_response = get_daily_fii_dii()
            if hasattr(fii_dii_response, 'get_json'):
                fii_dii_data = fii_dii_response.get_json().get('data', [])[:5]
        except Exception as inst_error:
            logger.error(f"Institutional data error: {inst_error}")

        technical_indicators = None
        try:
            tech_response = get_technical_indicators(stock_symbol)
            if hasattr(tech_response, 'get_json'):
                technical_indicators = tech_response.get_json()
        except Exception as tech_error:
            logger.error(f"Technical indicators error: {tech_error}")

        return render_template('results.html',
            stock_name=company_name,
            stock_symbol=stock_symbol,
            chart_data=chart_data,
            chart_type=chart_type,
            technicals=technicals,
            current_price=stock_info.get('currentPrice', 'N/A'),
            day_high=stock_info.get('dayHigh', 'N/A'),
            day_low=stock_info.get('dayLow', 'N/A'),
            pe_ratio=technicals.get('pe_ratio', 'N/A'),
            market_cap=format_indian_currency(stock_info.get('marketCap', 'N/A')),
            sector=stock_info.get('sector', 'N/A'),
            industry=stock_info.get('industry', 'N/A'),
            fifty_two_week_high=stock_info.get('fiftyTwoWeekHigh', 'N/A'),
            fifty_two_week_low=stock_info.get('fiftyTwoWeekLow', 'N/A'),
            dividend_yield=stock_info.get('dividendYield', 'N/A'),
            volume=format_indian_currency(stock_info.get('volume', 'N/A')),
            beta=stock_info.get('beta', 'N/A'),
            eps=stock_info.get('trailingEps', 'N/A'),
            book_value=stock_info.get('bookValue', 'N/A'),
            profit_margins=stock_info.get('profitMargins', 'N/A'),
            vote_counts=vote_counts,
            total_votes=sum(vote_counts.values()),
            avg_sentiment=avg_sentiment_label,
            recommendation=recommendation,
            news=news_articles,
            sentiment_labels=sentiment_labels,
            sentiment_score=avg_sentiment_value,
            fii_dii_data=fii_dii_data,
            technical_indicators=technical_indicators,
            is_market_open=is_market_open(),
            now=datetime.now(),
            error_message=None
        )

    except Exception as e:
        logger.critical(f"Critical error in results route: {str(e)}", exc_info=True)
        flash('System temporarily unavailable. Please try again later.', 'error')
        return redirect(url_for('home'))

@app.route('/emergency_fallback/<symbol>')
@cache.cached(timeout=3600)
def emergency_fallback(symbol):
    cached = Stock.query.filter_by(symbol=symbol).first()
    if cached:
        return render_template('results.html',
            stock_name=symbol_to_company.get(symbol, symbol),
            stock_symbol=symbol,
            current_price=cached.current_price,
            sector=cached.sector,
            industry=cached.industry,
            is_market_open=is_market_open(),
            emergency_mode=True
        )
    abort(404)

@app.route('/api/technical_indicators/<symbol>')
@cache.cached(timeout=3600)
def get_technical_indicators(symbol):
    try:
        stock = yf.Ticker(f"{symbol}.NS")
        hist = stock.history(period="1y")
        if hist.empty:
            return jsonify({'status': 'error', 'message': 'No data available'})
        closes = hist['Close'].values
        rsi = calculate_rsi(closes[-14:]) if len(closes) >= 14 else None
        macd = calculate_macd(closes[-26:]) if len(closes) >= 26 else None
        bollinger = calculate_bollinger_bands(closes[-20:]) if len(closes) >= 20 else None
        recent_prices = closes[-30:]
        support = min(recent_prices)
        resistance = max(recent_prices)
        return jsonify({
            'rsi': rsi,
            'macd': macd,
            'bollinger': bollinger,
            'support': support,
            'resistance': resistance,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error fetching technical indicators: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/daily-fii-dii')
def get_daily_fii_dii():
    try:
        try:
            fiidii_data = nse_fiidii()
            if fiidii_data and isinstance(fiidii_data, (pd.DataFrame, list)):
                if isinstance(fiidii_data, pd.DataFrame):
                    latest_data = fiidii_data.iloc[0].to_dict()
                else:
                    latest_data = fiidii_data[0]
                return jsonify({
                    'status': 'success',
                    'source': 'nsepython',
                    'data': [{
                        'date': latest_data.get('tradedDate', ''),
                        'fii_buy': latest_data.get('fii_Buy_Value', 'N/A'),
                        'fii_sell': latest_data.get('fii_Sell_Value', 'N/A'),
                        'fii_net': latest_data.get('fii_Net', 'N/A'),
                        'dii_buy': latest_data.get('dii_Buy_Value', 'N/A'),
                        'dii_sell': latest_data.get('dii_Sell_Value', 'N/A'),
                        'dii_net': latest_data.get('dii_Net', 'N/A')
                    }],
                    'latest': {
                        'date': latest_data.get('tradedDate', ''),
                        'fii_buy': latest_data.get('fii_Buy_Value', 'N/A'),
                        'fii_sell': latest_data.get('fii_Sell_Value', 'N/A'),
                        'fii_net': latest_data.get('fii_Net', 'N/A'),
                        'dii_buy': latest_data.get('dii_Buy_Value', 'N/A'),
                        'dii_sell': latest_data.get('dii_Sell_Value', 'N/A'),
                        'dii_net': latest_data.get('dii_Net', 'N/A')
                    }
                })
        except Exception as nse_error:
            logger.warning(f"NSEPython failed, trying alternative sources: {str(nse_error)}")

        sources = [
            {
                'name': 'StockEdge',
                'url': 'https://www.stockedge.com/data/fii-dii-data',
                'parser': 'stockedge'
            },
            {
                'name': 'Research360',
                'url': 'https://www.research360.net/fii-dii-data',
                'parser': 'research360'
            },
            {
                'name': 'BlinkX',
                'url': 'https://www.blinkx.in/fii-dii-data',
                'parser': 'blinkx'
            }
        ]

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        for source in sources:
            try:
                response = requests.get(source['url'], headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    if source['parser'] == 'stockedge':
                        table = soup.find('table', {'class': 'fii-dii-table'})
                        if table:
                            rows = table.find_all('tr')[1:6]
                            data = []
                            for row in rows:
                                cols = row.find_all('td')
                                if len(cols) >= 7:
                                    data.append({
                                        'date': cols[0].text.strip(),
                                        'fii_buy': cols[1].text.strip(),
                                        'fii_sell': cols[2].text.strip(),
                                        'fii_net': cols[3].text.strip(),
                                        'dii_buy': cols[4].text.strip(),
                                        'dii_sell': cols[5].text.strip(),
                                        'dii_net': cols[6].text.strip()
                                    })
                            if data:
                                return jsonify({
                                    'status': 'success',
                                    'source': source['name'],
                                    'data': data,
                                    'latest': data[0]
                                })
                    elif source['parser'] == 'research360':
                        table = soup.find('div', {'class': 'fii-dii-data'})
                        if table:
                            pass
                    elif source['parser'] == 'blinkx':
                        table = soup.find('section', {'id': 'fii-dii-section'})
                        if table:
                            pass
            except Exception as e:
                logger.warning(f"Failed to fetch from {source['name']}: {str(e)}")
                continue

        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch FII/DII data from all sources'
        }), 500
    except Exception as e:
        logger.error(f"Error in FII/DII endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/fundamentals/<symbol>')
@cache.cached(timeout=86400)
def get_fundamentals(symbol):
    try:
        stock = yf.Ticker(f"{symbol}.NS")
        info = stock.info
        earnings = stock.quarterly_earnings
        earnings_surprise = ((earnings['Actual'] - earnings['Estimate']) / earnings['Estimate'] * 100).mean() if not earnings.empty else None
        holdings_data = fetch_institutional_data(symbol)
        return jsonify({
            'earnings_surprise': earnings_surprise,
            'revenue_growth': info.get('revenueGrowth'),
            'fii_trend': holdings_data.get('fii_trend', 'N/A'),
            'promoter_trend': holdings_data.get('promoter_trend', 'N/A'),
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error fetching fundamentals: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/fii-dii')
def get_fii_dii_data():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
            "Referer": "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php"
        }
        url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'mctable1'})
        if not table:
            return jsonify({'error': 'Table not found'}), 404
        rows = table.find_all('tr')[1:]
        data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 7:
                entry = {
                    'date': cols[0].text.strip(),
                    'fii_buy': cols[1].text.strip(),
                    'fii_sell': cols[2].text.strip(),
                    'fii_net': cols[3].text.strip(),
                    'dii_buy': cols[4].text.strip(),
                    'dii_sell': cols[5].text.strip(),
                    'dii_net': cols[6].text.strip()
                }
                data.append(entry)
        return jsonify({'data': data, 'status': 'success'})
    except Exception as e:
        logger.error(f"Error fetching FII/DII data: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@cache.memoize(timeout=86400)
def fetch_institutional_data(symbol):
    try:
        search_url = f"https://www.screener.in/search/?q={symbol}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        search_response = requests.get(search_url, headers=headers, timeout=10)
        search_response.raise_for_status()
        soup = BeautifulSoup(search_response.text, 'html.parser')
        company_link = soup.find('a', class_='company-link')['href']
        company_url = f"https://www.screener.in{company_link}"
        company_response = requests.get(company_url, headers=headers, timeout=10)
        company_response.raise_for_status()
        company_soup = BeautifulSoup(company_response.text, 'html.parser')
        shareholding_section = company_soup.find('section', id='shareholding')
        data = {}
        for row in shareholding_section.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) >= 3:
                holder_name = cols[0].get_text(strip=True)
                if 'FII' in holder_name:
                    data['fii_trend'] = cols[2].get_text(strip=True)
                elif 'Promoter' in holder_name:
                    data['promoter_trend'] = cols[2].get_text(strip=True)
        return data if data else {'fii_trend': 'N/A', 'promoter_trend': 'N/A'}
    except Exception as e:
        logger.error(f"Error fetching institutional data: {e}")
        return {'fii_trend': 'N/A', 'promoter_trend': 'N/A'}

def calculate_rsi(prices, period=14):
    try:
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = 100 - (100/(1 + rs))
        for i in range(period, len(prices)-1):
            delta = deltas[i]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down
            rsi = np.append(rsi, 100 - (100/(1 + rs)))
        return rsi[-1]
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return None

def calculate_macd(prices, fast=12, slow=26, signal=9):
    try:
        prices_series = pd.Series(prices)
        ema_fast = prices_series.ewm(span=fast, adjust=False).mean()
        ema_slow = prices_series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            'macd_line': macd_line.iloc[-1],
            'signal_line': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1],
            'trend': 'bullish' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'bearish'
        }
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return None

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    try:
        prices_series = pd.Series(prices)
        sma = prices_series.rolling(window=period).mean().iloc[-1]
        std = prices_series.rolling(window=period).std().iloc[-1]
        current_price = prices[-1]
        position = 'upper' if current_price > sma + std_dev*std else \
                  'lower' if current_price < sma - std_dev*std else 'middle'
        return {
            'upper': sma + std_dev * std,
            'middle': sma,
            'lower': sma - std_dev * std,
            'price_position': position
        }
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return None

@app.route('/api/search_companies')
def search_companies():
    query = request.args.get('q', '').lower().strip()
    if not query or len(query) < 2: return jsonify([])
    results = []
    for company, symbol in company_to_symbol.items():
        if query in company.lower():
            results.append({'name': company, 'symbol': symbol})
            if len(results) >= 20: break
    return jsonify(results)

@app.route('/api/discussion/<symbol>', methods=['GET', 'POST'])
@handle_errors
def handle_discussion(symbol):
    if request.method == 'POST':
        if 'user_id' not in session:
            return jsonify({'error': 'Login required'}), 401
            
        data = request.get_json()
        new_comment = Discussion(
            stock_symbol=symbol,
            user_id=session['user_id'],
            content=data['content'],
            parent_id=data.get('parent_id'),
            sentiment=analyze_sentiment(data['content'])[1]
        )
        db.session.add(new_comment)
        db.session.commit()
        return jsonify({'success': True})
    
    # GET: Fetch comments for this stock
    comments = Discussion.query.filter_by(stock_symbol=symbol, parent_id=None)\
               .order_by(Discussion.timestamp.desc()).all()
    return jsonify([{
        'id': c.id,
        'user_name': c.user.username,
        'content': c.content,
        'timestamp': c.timestamp.isoformat(),
        'likes': c.likes,
        'sentiment': c.sentiment,
        'replies': [{
            'id': r.id,
            'user_name': r.user.username,
            'content': r.content,
            'timestamp': r.timestamp.isoformat(),
            'likes': r.likes,
            'sentiment': r.sentiment
        } for r in c.replies]
    } for c in comments])

@app.route('/vote', methods=['POST'])
@csrf.exempt
def handle_vote():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'You must be logged in to vote.'}), 401

    data = request.get_json()
    stock_symbol = data.get('stock_symbol')
    user_vote = data.get('vote')
    user_id = session['user_id']

    if not stock_symbol or not user_vote or user_vote not in ['Buy', 'Hold', 'Sell']:
        return jsonify({'success': False, 'message': 'Invalid vote request.'}), 400

    # Check if user has voted for this stock in the last 24 hours
    last_vote = Vote.query.filter_by(
        stock_symbol=stock_symbol,
        user_id=user_id
    ).order_by(Vote.timestamp.desc()).first()

    if last_vote and (datetime.utcnow() - last_vote.timestamp) < timedelta(hours=24):
        return jsonify({
            'success': False,
            'message': 'You can only vote once per stock in 24 hours.'
        }), 400

    # Get current vote counts (either from mock data or real votes)
    votes = Vote.query.filter_by(stock_symbol=stock_symbol).all()
    mock_data_key = f"mock_votes_{stock_symbol}"
    
    if len(votes) < 10:  # Using mock data
        vote_counts = session.get(mock_data_key, {
            'Buy': 0,
            'Hold': 0,
            'Sell': 0
        })
    else:  # Using real votes
        vote_counts = {
            'Buy': Vote.query.filter_by(stock_symbol=stock_symbol, vote='Buy').count(),
            'Hold': Vote.query.filter_by(stock_symbol=stock_symbol, vote='Hold').count(),
            'Sell': Vote.query.filter_by(stock_symbol=stock_symbol, vote='Sell').count()
        }

    # Increment the voted option
    vote_counts[user_vote] += 1

    # Store updated counts if using mock data
    if len(votes) < 10:
        session[mock_data_key] = vote_counts

    # Save the real vote to database
    new_vote = Vote(
        stock_symbol=stock_symbol,
        vote=user_vote,
        user_id=user_id
    )
    db.session.add(new_vote)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': 'Your vote has been recorded!',
        'vote_counts': vote_counts,
        'total_votes': sum(vote_counts.values())
    })

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect(url_for('register'))
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            user.last_login = datetime.utcnow()
            db.session.commit()
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/realtime_data', methods=['GET'])
@cache.cached(timeout=60)
def realtime_data():
    symbol = request.args.get('symbol')
    try:
        stock = yf.Ticker(symbol + ".NS")
        data = stock.history(period="1d", interval="1m")
        if data.empty: return jsonify({'error': 'No data found'}), 404
        times = data.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        prices = data['Close'].tolist()
        return jsonify({'times': times, 'prices': prices})
    except Exception as e:
        logger.error(f"Error fetching real-time data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/watchlist/add', methods=['POST'])
@handle_errors
def add_to_watchlist():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login to use watchlist'}), 401
    
    stock_symbol = request.form.get('stock_symbol')
    if not stock_symbol:
        return jsonify({'success': False, 'message': 'Stock symbol required'}), 400
    
    # Check if already in watchlist
    existing = Watchlist.query.filter_by(
        user_id=session['user_id'],
        stock_symbol=stock_symbol
    ).first()
    
    if existing:
        return jsonify({'success': False, 'message': 'Stock already in watchlist'}), 400
    
    # Add to watchlist
    watchlist_item = Watchlist(
        user_id=session['user_id'],
        stock_symbol=stock_symbol
    )
    db.session.add(watchlist_item)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Added to watchlist'
    })

@app.route('/watchlist/remove', methods=['POST'])
@handle_errors
def remove_from_watchlist():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login to modify watchlist'}), 401
    
    stock_symbol = request.form.get('stock_symbol')
    if not stock_symbol:
        return jsonify({'success': False, 'message': 'Stock symbol required'}), 400
    
    # Remove from watchlist
    Watchlist.query.filter_by(
        user_id=session['user_id'],
        stock_symbol=stock_symbol
    ).delete()
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Removed from watchlist'
    })

@app.route('/watchlist', methods=['GET'])
@handle_errors
def get_watchlist():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    watchlist_items = Watchlist.query.filter_by(
        user_id=session['user_id']
    ).order_by(Watchlist.added_at.desc()).all()
    
    # Get stock data for each watchlist item
    watchlist_data = []
    for item in watchlist_items:
        stock = Stock.query.filter_by(symbol=item.stock_symbol).first()
        if stock:
            watchlist_data.append({
                'symbol': item.stock_symbol,
                'name': stock.name,
                'current_price': stock.current_price,
                'added_at': item.added_at,
                'notes': item.notes
            })
    
    return render_template('watchlist.html', watchlist=watchlist_data)

@app.route('/profile')
@cache.cached(timeout=60)
def profile():
    if 'user_id' not in session:
        flash('Please login to view your profile', 'error')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    
    # Get basic stats
    search_count = SearchHistory.query.filter_by(user_id=user.id).count()
    vote_count = Vote.query.filter_by(user_id=user.id).count()
    recent_votes = Vote.query.filter_by(user_id=user.id).order_by(Vote.timestamp.desc()).limit(5).all()
    
    top_stocks = db.session.query(
        Vote.stock_symbol,
        func.count(Vote.id).label('vote_count')
    ).filter_by(user_id=user.id)\
     .group_by(Vote.stock_symbol)\
     .order_by(func.count(Vote.id).desc())\
     .limit(3).all()
    
    watchlist_items = Watchlist.query.filter_by(user_id=user.id).order_by(Watchlist.added_at.desc()).limit(5).all()
    
    # Fetch prices with proper formatting - ensure None instead of 'N/A'
    stock_prices = {}
    for item in watchlist_items:
        try:
            stock = yf.Ticker(f"{item.stock_symbol}.NS")
            hist = stock.history(period='1d')
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                stock_prices[item.stock_symbol] = float(price)  # Store as float
            else:
                cached = Stock.query.filter_by(symbol=item.stock_symbol).first()
                stock_prices[item.stock_symbol] = float(cached.current_price) if cached and cached.current_price else None
        except Exception as e:
            logger.error(f"Error fetching price for {item.stock_symbol}: {e}")
            stock_prices[item.stock_symbol] = None  # Use None instead of 'N/A'

    return render_template('profile.html',
        user=user,
        search_count=search_count,
        vote_count=vote_count,
        recent_votes=recent_votes,
        top_stocks=top_stocks,
        watchlist_items=watchlist_items,
        symbol_to_company=symbol_to_company,
        stock_prices=stock_prices  # Pass the dictionary of prices (numbers or None)
    )
    
@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        flash('Please login to update your profile', 'error')
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    user.full_name = request.form.get('full_name')
    user.email = request.form.get('email')
    user.notification_prefs = {
        'email': 'email_notifications' in request.form,
        'app': 'app_notifications' in request.form
    }
    db.session.commit()
    flash('Profile updated successfully', 'success')
    return redirect(url_for('profile'))

@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user_id' not in session:
        flash('Please login to change your password', 'error')
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    current_password = request.form['current_password']
    new_password = request.form['new_password']
    confirm_password = request.form['confirm_password']
    if not user.check_password(current_password):
        flash('Current password is incorrect', 'error')
    elif new_password != confirm_password:
        flash('New passwords do not match', 'error')
    else:
        user.set_password(new_password)
        db.session.commit()
        flash('Password changed successfully', 'success')
    return redirect(url_for('profile'))

@app.route('/health')
def health_check():
    """Endpoint for health checks"""
    try:
        # Test database connection
        db.session.execute(db.select(1)).scalar()
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response

@cache.cached(timeout=60, key_prefix='realtime_chart_data')
def fetch_realtime_chart_data(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol + ".NS")
        data = stock.history(period="1d", interval="5m")
        if data.empty: return None
        return {
            'times': data.index.strftime('%H:%M').tolist(),
            'prices': data['Close'].tolist()
        }
    except Exception as e:
        logger.error(f"Error fetching real-time data for {stock_symbol}: {e}")
        return None

@cache.cached(timeout=3600, key_prefix='last_trading_day_data')
def fetch_last_trading_day_data(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol + ".NS")
        data = stock.history(period="1d", interval="30m")
        if data.empty: return None
        return {
            'times': data.index.strftime('%H:%M').tolist(),
            'prices': data['Close'].tolist()
        }
    except Exception as e:
        logger.error(f"Error fetching last trading day data for {stock_symbol}: {e}")
        return None

@cache.memoize(timeout=timedelta(hours=36).total_seconds())
def fetch_cached_news(stock_symbol, company_name):
    query = f"{company_name} OR {stock_symbol}"
    news = fetch_newsapi(query)
    alt_news = fetch_combined_news(stock_symbol, company_name)
    return news['articles'] + alt_news

def fetch_technical_details(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol + ".NS")
        info = stock.info
        technicals = {
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'peg_ratio': info.get('pegRatio', 'N/A'),
            'eps': info.get('trailingEps', 'N/A'),
            'book_value': info.get('bookValue', 'N/A'),
            'price_to_book': info.get('priceToBook', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'market_cap': format_indian_currency(info.get('marketCap', 'N/A')),
            'avg_volume': format_indian_currency(info.get('averageVolume', 'N/A')),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
            'ceo': fetch_ceo_name(stock_symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
        }
        current_price = info.get('currentPrice')
        fifty_two_week_high = info.get('fiftyTwoWeekHigh')
        fifty_two_week_low = info.get('fiftyTwoWeekLow')
        if current_price and fifty_two_week_high and fifty_two_week_low:
            technicals['from_52w_high'] = f"{(float(fifty_two_week_high) - current_price) / float(fifty_two_week_high) * 100:.2f}%"
            technicals['from_52w_low'] = f"{(current_price - float(fifty_two_week_low)) / float(fifty_two_week_low) * 100:.2f}%"
        else:
            technicals['from_52w_high'] = 'N/A'
            technicals['from_52w_low'] = 'N/A'
        return technicals
    except Exception as e:
        logger.error(f"Error fetching technical details for {stock_symbol}: {e}")
        return None

@cache.cached(timeout=86400, key_prefix='ceo_name')
def fetch_ceo_name(stock_symbol):
    try:
        url = f"https://www.moneycontrol.com/india/stockpricequote/telecommunications-service/{stock_symbol}"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        ceo_name = soup.find('div', class_='CEO_name').text.strip()
        return ceo_name
    except Exception as e:
        logger.error(f"Error fetching CEO name: {e}")
        return 'N/A'

@cache.memoize(3600)
def fetch_stock_data(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol + ".NS")
        stock_info = stock.info
        historical_data = stock.history(period="1mo")
        return stock_info, historical_data
    except Exception as e:
        logger.error(f"Error fetching stock data for {stock_symbol}: {e}")
        return {}, pd.DataFrame()

with app.app_context():
    db.create_all()
    populate_initial_data()

# ===== APPLICATION INITIALIZATION =====
def initialize_app():
    """Initialize all application components"""
    with app.app_context():
        # Create tables if they don't exist
        db.create_all()
        
        # Populate initial data
        populate_initial_data()
        
        # Verify essential services
        verify_services()

def verify_services():
    """Verify all external services are reachable"""
    services = {
        'NewsAPI': lambda: NewsApiClient(api_key=os.getenv('NEWSAPI_KEY')).get_sources(),
        'Yahoo Finance': lambda: yf.Ticker("RELIANCE.NS").info,
        'Database': lambda: db.session.execute(db.select(1)).scalar()
    }
    
    for name, test in services.items():
        try:
            test()
            app.logger.info(f"{name} service verified")
        except Exception as e:
            app.logger.error(f"{name} service check failed: {str(e)}")

if __name__ == '__main__':
    initialize_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)