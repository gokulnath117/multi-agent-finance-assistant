# tools.py
from langchain.tools import tool
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from newspaper import Article

@tool
def search_tool(query: str) -> str:
    """Fetches recent stock data for a given ticker (e.g., 'AAPL')"""
    ticker = query.strip().upper()
    try:
        df = yf.download(ticker, period="5d")
        return df.tail().to_string()
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def news_tool(company: str) -> str:
    """Fetch unique news articles about a company"""
    news_urls = get_news_urls(company)

    if news_urls:
        all_news = []
        for url in news_urls:
            news = extract_news(url)
            if news:
                all_news.append(news)
                if len(all_news) >= 20:  # Fetch more to account for duplicate removal
                    break

        return all_news
    return None

def get_news_urls(company, max_results=10):
    """Fetches news article URLs related to a given company from Bing News"""
    search_url = f"https://www.google.com/search?q={company.replace(' ', '+')}+news&tbm=nws"

    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    remove_url=['https://www.google.com','https://maps.google.com','https://play.google.com','https://policies.google.com','https://support.google.com','https://accounts.google']

    news_links = set()
    for link in soup.find_all('a', href=True):
        url = link['href']
        filter_url = url.split("/url?q=")[-1].split("&")[0]
        if "http" in filter_url and 'msn' not in filter_url and not any(remove in filter_url for remove in remove_url):
            news_links.add(filter_url)
            if len(news_links) >= max_results:
                break
    
    return list(news_links)

def extract_news(url):
    """Scrapes article title, summary, and text using newspaper3k."""
    try:
        article = Article(url)
        article.download()
        article.parse()

        source = url.split('/')[2]

        return {
            "title": article.title,
            "summary": article.meta_description if article.meta_description else article.text[:250],
            "source": source,
            "publish_date": str(article.publish_date) if article.publish_date else "No Date Available",
        }
    except Exception as e:
        # print(f"Error extracting {url}: {e}")
        return None