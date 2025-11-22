import os
import uvicorn
import requests
from openai import OpenAI  # <--- NEW IMPORT
from fastapi import FastAPI
from gnews import GNews
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import yfinance as yf
from cachetools import TTLCache # <--- NEW IMPORT
from datetime import datetime, timedelta
from dotenv import load_dotenv

import re

CACHE_TTL_SECONDS = 12 * 60 * 60
CACHE_MAX_SIZE = 100
analysis_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)
# 1. Define client variable globally, initialize it to None
client = None
MODEL_NAME = 'gpt-4o-mini' # Use 'gpt-5-flash' for higher performance/cost
# 2. Load the environment variables from .env
load_dotenv()

# 3. Read the API Key from the environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Configure the OpenAI client
try:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment.")

    # The client variable is defined/assigned here
    client = OpenAI(api_key=OPENAI_API_KEY)

except Exception as e:
    # If anything fails (key missing, initialization error), client remains None
    print(f"Warning: Failed to initialize OpenAI client. Error: {e}")

def get_cached_analysis(ticker: str) -> list[dict[str, Any]] | None:
    """Retrieves cached result if available and not expired."""
    return analysis_cache.get(ticker.upper())

def set_cached_analysis(ticker: str, result: list[dict[str, Any]]):
    """Stores the analysis result in the cache."""
    analysis_cache[ticker.upper()] = result


def is_chat_or_spam_link(url: str) -> bool:
    """
    Checks if a URL is a direct link to a chat platform.
    """
    # Common patterns for WhatsApp, Telegram, and known shorteners/spam
    blacklist_patterns = [
        "wa.me",  # Official WhatsApp Click to Chat
        "whatsapp.com/send",
        "t.me",  # Official Telegram links
        "telegram.me",
        "m.me",  # Official Facebook Messenger links
        "bit.ly",  # Common shorteners often used for redirects
        "linktr.ee"  # Bio link aggregators
    ]

    url_lower = url.lower()

    # Return True (Spam/Chat) if any pattern is found in the URL
    return any(pattern in url_lower for pattern in blacklist_patterns)




app = FastAPI(title="AI Stock News Analyst (OpenAI)")


def filter_and_sort_news(news_results: List[Dict[str, Any]], ticker: str, company_name: str) -> List[Dict[str, Any]]:
    """
    STRICTLY filters news: only keeps articles where the company name/ticker AND a catalyst
    keyword are both present in the title/description. It then sorts the remaining relevant news.
    """

    # 1. Define the individual catalyst keywords
    catalyst_keywords = ['anti-dumping', 'government', 'policy', 'duty', 'tariff',
                         'capex', 'expansion', 'result', 'earnings', 'concall',
                         'margin', 'guidance', 'acquisition', 'divestiture', 'Growing TAM',
                         'Operating leverage', 'Value chain climb (forward integration/value-added)',
                         'capex', 'Geographic expansion', 'New products','Deleveraging', 'segments',
                         'fundraise', 'demerger', 'warrants', 'M&A/strategic stakes',
                         'Management', 'strategy', 'Order book', 'pipeline expansion',
                         'breaks', 'deals', 'shares', 'bags', 'surges', 'down', 'pledge', 'sell',
                         'buyback', 'order win', 'export growth', 'margin expansion',
                         'new product launch', 'regulatory approval', 'capacity addition', 'debt reduction',
                         'strategic partnership', 'government incentive', 'strong guidance', 'record revenue',
                         'promoter buying', 'anti-dumping duty', 'penalty', 'QIP', 'ED',
                         'investigation', 'fraud', 'plant shutdown', 'weak guidance',
                         'debt default', 'supply disruption', 'promoter selling', 'margin compression',
                         'layoffs', 'price hike denial', 'loss of contract', 'regulatory ban', 'litigation',
                         'order inflow', 'favorable regulation', 'price hike', 'volume growth',
                         'market share gain', 'operational efficiency', 'cost optimization',
                         'resumption of production', 'channel expansion', 'distribution expansion',
                         'technology upgrade', 'patent approval', 'ESG compliance', 'asset monetization',
                         'dividend announcement', 'bonus issue', 'stock split', 'global expansion',
                         'license approval', 'supply agreement', 'long-term contract', 'turnaround plan',
                         'profit warning', 'inventory buildup', 'management resignation',
                         'whistleblower complaint', 'downgrade by rating agency', 'credit rating upgrade',
                         'cyber attack', 'data breach', 'environmental violation', 'export ban', 'import duty',
                         'raw material shortage', 'fed policy impact', 'currency fluctuation', 'geo-political risk',
                         'commodity price surge', 'commodity price fall', 'downgrade', 'upgrade']

    # 2. Define the search terms (ticker and full name)
    search_terms = {ticker.upper(), company_name.upper()}

    strictly_filtered_results = []

    for item in news_results:
        text = (item.get('title', '') + ' ' + item.get('description', '')).upper()

        # Check 1: Must contain the Company Name/Ticker
        company_matched = any(term in text for term in search_terms)

        # Check 2: Must contain at least one Catalyst Keyword
        catalyst_matched = any(keyword.upper() in text for keyword in catalyst_keywords)

        # 3. Apply the STRICT FILTERING Logic
        if company_matched and catalyst_matched:
            # ONLY articles matching BOTH criteria are kept.
            item['relevance_score'] = 100
            strictly_filtered_results.append(item)
        # Articles that fail this strict check are discarded (not added to the list)

    # Sort the remaining (strictly relevant) results by score (descending)
    strictly_filtered_results.sort(key=lambda x: x['relevance_score'], reverse=True)

    return strictly_filtered_results

def get_company_name_and_query(ticker: str) -> str:
    """
    Fetches the company's full name and constructs the dynamic, filtered GNews query.
    """
    # 1. Fetch Company Data using yfinance
    try:
        # Ticker(ticker).info fetches a dictionary of company data
        info = yf.Ticker(ticker).info
        company_name = info.get('longName', ticker)
        # Use the full name in quotes for maximum precision
        search_term = f'"{company_name}" OR {ticker}'

    except Exception:
        # Fallback if yfinance fails (e.g., due to an invalid ticker or API issues)
        print(f"yfinance lookup failed for {ticker}. Falling back to basic search.")
        search_term = f'"{ticker} stock"'

    # 2. Define the Universal Filtering Components

    # Exclude common, low-impact news items (works for any stock)
    low_value_exclusion = "NOT ('52 week' OR '52w'  OR 'technical analysis' OR 'brokerage' OR 'sensex' OR 'nifty')"

    # Target high-impact, catalytic events (works for any stock)
    # catalyst_keywords = "(anti-dumping OR government OR policy OR duty OR tariff OR capex OR expansion OR result OR earnings OR concall OR margin OR guidance OR acquisition OR divestiture OR Growing TAM OR Operating leverage OR Value chain climb (forward integration/value-added) OR capex OR Geographic expansion OR New products OR Deleveraging OR segments OR fundraise OR demerger OR warrants OR M&A/strategic stakes OR Management OR strategy OR Order book OR pipeline expansion OR breaks OR deals OR shares OR bags OR surges OR down OR pledge OR sell OR buyback OR order win OR export growth OR margin expansion OR new product launch OR regulatory approval OR capacity addition OR debt reduction OR strategic partnership OR government incentive OR strong guidance OR record revenue OR promoter buying OR anti-dumping duty OR penalty OR QIP OR ED OR investigation OR fraud OR plant shutdown OR weak guidance OR debt default OR supply disruption OR promoter selling OR margin compression OR layoffs OR price hike denial OR loss of contract OR regulatory ban OR litigation OR order inflow OR favorable regulation OR price hike OR volume growth OR market share gain OR operational efficiency OR cost optimization OR resumption of production OR channel expansion OR distribution expansion OR technology upgrade OR patent approval OR ESG compliance OR asset monetization OR dividend announcement OR bonus issue OR stock split OR global expansion OR license approval OR supply agreement OR long-term contract OR turnaround plan OR profit warning OR inventory buildup OR management resignation OR whistleblower complaint OR downgrade by rating agency OR credit rating upgrade OR cyber attack OR data breach OR environmental violation OR export ban OR import duty OR raw material shortage OR fed policy impact OR currency fluctuation OR geo-political risk OR commodity price surge OR commodity price fall OR downgrade OR upgrade)"

    # 3. Combine them into the final, dynamic GNews query
    query = f'{search_term}  {low_value_exclusion}'
    return query

# --- CORE HELPER FUNCTIONS (Scraping remains the same) ---

def scrape_article_content(url: str) -> str:
    """Uses requests and BeautifulSoup to extract text from a news URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = " ".join([p.get_text() for p in paragraphs])

        return content[:4000]

    except Exception as e:
        return f"Scraping Failed: {e}"


def analyze_with_llm(ticker: str, title: str, content: str) -> Dict[str, str]:
    """Sends news data to OpenAI's GPT for structured analysis."""

    if "Scraping Failed" in content:
        content = f"The scraping failed, please analyze based ONLY on the headline: {title}"

    # --- The LLM Prompt (Leveraging Chat Completions structure) ---
    prompt = f"""
You are a professional financial analyst. Your task is to evaluate a news article concerning the stock ticker {ticker}.
Based on the content provided, you MUST perform a structured analysis and extract the requested highlights.

Article Title: {title}
Article Content: {content}

---
STRICT EXTRACTION REQUIREMENTS:
1. RESULTS_HIGHLIGHT: Identify the most important positive or negative highlight from the latest earnings result or conference call (e.g., "Margins beat estimates," or "New capex plan announced"). If not found, use 'N/A'.
2. POLICY_IMPACT: Identify any mention of new government rules, anti-dumping duties, tariffs, or trade policy changes that specifically affect the company. If not found, use 'N/A'.

Return a strictly formatted response with these five labels:
1. SUMMARY: (One concise sentence summarizing the core financial event, combining the title and content.)
2. RESULTS_HIGHLIGHT: (The extracted data point from requirement #1.)
3. POLICY_IMPACT: (The extracted data point from requirement #2.)
4. IMPACT: (Choose one: BULLISH, BEARISH, or NEUTRAL, based on ALL extracted information.)
5. REASON: (One sentence explaining the primary reason for the determined impact, referring to the extracted data.)
"""

    try:
        response = client.chat.completions.create(  # <--- NEW API CALL METHOD
            model=MODEL_NAME,
            messages=[
                {"role": "system",
                 "content": "You are a specialized financial analyst. Your output must strictly follow the SUMMARY, IMPACT, REASON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0  # Use low temperature for consistent financial analysis
        )

        ai_response_text = response.choices[0].message.content

        # Basic parsing of the structured LLM output (same logic as before)
        lines = ai_response_text.split('\n')

        result = {
            "summary": next((line.split(': ', 1)[1] for line in lines if 'SUMMARY:' in line), "Analysis error or N/A"),
            "impact": next((line.split(': ', 1)[1] for line in lines if 'IMPACT:' in line), "NEUTRAL"),
            "reason": next((line.split(': ', 1)[1] for line in lines if 'REASON:' in line),
                           "N/A: Could not parse LLM reason.")
        }
        return result

    except Exception as e:
        return {
            "summary": "LLM API failed to connect or process.",
            "impact": "NEUTRAL",
            "reason": f"API Error: {e}"
        }


# --- API ENDPOINT (Remains the same) ---

@app.get("/agent/financial-analysis")
def get_stock_analysis(ticker: str, limit: int = 10) -> Dict[str, Any]:
    """
        Fetches the latest news for a ticker using dynamic filtering,
        and analyzes the sentiment using the LLM.
        """
    print(f"Starting dynamic analysis for {ticker}. Fetching {limit} news items.")
    ticker_upper = ticker.upper()

    # 1. 🔍 CHECK CACHE
    cached_result = get_cached_analysis(ticker_upper)
    # if cached_result:
    #     return {
    #         "stock": ticker,
    #         "total_articles_analyzed": len(cached_result),
    #         "articles": cached_result
    #     }
    #
    # print(f"❌ Cache Miss: Performing fresh analysis for {ticker_upper}.")

    google_news = GNews(language='en', country='IN', period='2d', max_results=limit)
    # 🌟 NEW: Get the dynamically generated, filtered query
 # Ensure ticker is uppercase for yfinance

    company_name = get_company_name_and_query(ticker.upper()).split('"')[1]
    news_results = google_news.get_news(company_name)
    news_results_filtered = filter_and_sort_news(news_results, ticker, company_name)
    analyzed_data: List[Dict[str, Any]] = []
    # 1. Fetch News Links using the dynamic query

    print(f"Searching GNews with query: {company_name}")

    analyzed_data: List[Dict[str, Any]] = []

    for news_item in news_results_filtered[:limit]:
        url = news_item.get('url', '')
        title = news_item.get('title', '')
        source = news_item.get('source', '')
        if is_chat_or_spam_link(url) or news_item.get('publisher', {}).get('title') == "earlytimes.in":
            print(f"Skipping link (Chat/Spam detected): {url}")
            continue  # Skip the rest of the loop for this item
        full_content = scrape_article_content(url)
        analysis = analyze_with_llm(ticker, title, full_content)

        analyzed_data.append({
            "title": title,
            "link": url,
            "published_date": news_item.get('published date'),
            "source": news_item.get('publisher', {}).get('title'),
            "analysis": analysis
        })
    set_cached_analysis(ticker_upper, analyzed_data)
    return {
        "stock": ticker,
        "total_articles_analyzed": len(analyzed_data),
        "articles": analyzed_data
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)