"""
Configuration settings for NIFTY/SENSEX Manual Trader
All sensitive credentials are loaded from Streamlit secrets
"""

import streamlit as st
import pytz
from datetime import datetime
import os

# ═══════════════════════════════════════════════════════════════════════
# TIMEZONE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Indian Standard Time (IST) - Use this for all datetime operations
IST = pytz.timezone('Asia/Kolkata')

def get_current_time_ist():
    """Get current time in IST timezone"""
    return datetime.now(IST)

# ═══════════════════════════════════════════════════════════════════════
# CREDENTIALS - Loaded from Streamlit Secrets
# ═══════════════════════════════════════════════════════════════════════

def get_dhan_credentials():
    """Load DhanHQ credentials from secrets"""
    try:
        return {
            'client_id': st.secrets["DHAN"]["CLIENT_ID"],
            'access_token': st.secrets["DHAN"].get("ACCESS_TOKEN", ""),
            'api_key': st.secrets["DHAN"].get("API_KEY", ""),
            'api_secret': st.secrets["DHAN"].get("API_SECRET", "")
        }
    except Exception as e:
        print(f"⚠️ DhanHQ credentials missing: {e}")
        return None

def get_telegram_credentials():
    """Load Telegram credentials from secrets"""
    try:
        return {
            'bot_token': st.secrets["TELEGRAM"]["BOT_TOKEN"],
            'chat_id': st.secrets["TELEGRAM"]["CHAT_ID"],
            'enabled': True
        }
    except Exception as e:
        print(f"⚠️ Telegram credentials missing: {e}")
        return {'enabled': False}

# ═══════════════════════════════════════════════════════════════════════
# AI CONFIGURATION - Loaded from Streamlit Secrets
# ═══════════════════════════════════════════════════════════════════════

def get_perplexity_credentials():
    """Load Perplexity credentials from secrets"""
    try:
        return {
            'api_key': st.secrets["PERPLEXITY"]["API_KEY"],
            'model': st.secrets["PERPLEXITY"].get("MODEL", "sonar"),
            'search_depth': st.secrets["PERPLEXITY"].get("SEARCH_DEPTH", "medium"),
            'enabled': True
        }
    except Exception as e:
        print(f"⚠️ Perplexity credentials missing: {e}")
        return {'enabled': False}

def get_newsdata_credentials():
    """Load NewsData credentials from secrets"""
    try:
        return {
            'api_key': st.secrets["NEWSDATA"]["API_KEY"],
            'enabled': True
        }
    except Exception as e:
        print(f"⚠️ NewsData credentials missing: {e}")
        return {'enabled': False}

# AI Settings
AI_RUN_ONLY_DIRECTIONAL = os.environ.get("AI_RUN_ONLY_DIRECTIONAL", "") == "1"
AI_REPORT_DIR = os.environ.get("AI_REPORT_DIR", "ai_reports")

# ═══════════════════════════════════════════════════════════════════════
# MARKET HOURS SETTINGS (All times in IST - Indian Standard Time)
# ═══════════════════════════════════════════════════════════════════════

MARKET_HOURS_ENABLED = True  # Set to False to disable market hours checking

# Market session timings (IST)
MARKET_HOURS = {
    'pre_market_open': '08:30',    # 8:30 AM IST
    'market_open': '09:15',        # 9:15 AM IST
    'market_close': '15:30',       # 3:30 PM IST
    'post_market_close': '15:45'   # 3:45 PM IST (App will run until this time)
}

# Session-based refresh intervals (seconds)
# Optimized to prevent API rate limiting (HTTP 429)
REFRESH_INTERVALS = {
    'pre_market': 60,      # 60 seconds during pre-market
    'regular': 60,         # 60 seconds during regular trading
    'post_market': 60,     # 60 seconds during post-market
    'closed': 60           # 60 seconds when market is closed (1 minute)
}

# ═══════════════════════════════════════════════════════════════════════
# TRADING SETTINGS
# ═══════════════════════════════════════════════════════════════════════

LOT_SIZES = {
    "NIFTY": 75,
    "SENSEX": 30
}

STRIKE_INTERVALS = {
    "NIFTY": 50,
    "SENSEX": 100
}

SENSEX_NIFTY_RATIO = 3.3  # SENSEX ≈ 3.3 × NIFTY

STOP_LOSS_OFFSET = 10  # Points
SIGNALS_REQUIRED = 3
VOB_TOUCH_TOLERANCE = 5  # Points

# ═══════════════════════════════════════════════════════════════════════
# UI SETTINGS
# ═══════════════════════════════════════════════════════════════════════

# Auto-refresh interval: 1 minute (60 seconds)
AUTO_REFRESH_INTERVAL = 60  # seconds (1 minute - optimized for fast clicks)
DEMO_MODE = False

APP_TITLE = "🎯 NIFTY/SENSEX Manual Trader"
APP_SUBTITLE = "VOB-Based Trading | Manual Signal Entry"

COLORS = {
    'bullish': '#089981',
    'bearish': '#f23645',
    'neutral': '#787B86'
}

# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def get_all_credentials():
    """Get all credentials in a single call"""
    return {
        'dhan': get_dhan_credentials(),
        'telegram': get_telegram_credentials(),
        'perplexity': get_perplexity_credentials(),
        'newsdata': get_newsdata_credentials()
    }

def is_ai_enabled():
    """Check if AI features are enabled"""
    perplexity_creds = get_perplexity_credentials()
    newsdata_creds = get_newsdata_credentials()
    return perplexity_creds.get('enabled', False) and newsdata_creds.get('enabled', False)
