import streamlit as st
import time
from datetime import datetime
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
from pytz import timezone as pytz_timezone
import plotly.graph_objects as go
import io
import requests
from streamlit_autorefresh import st_autorefresh
import os
import asyncio
import logging

# Import modules
from config import *
from market_data import *
from market_hours_scheduler import scheduler, is_within_trading_hours, should_run_app
from signal_manager import SignalManager
from strike_calculator import calculate_strike, calculate_levels
from trade_executor import TradeExecutor
from telegram_alerts import TelegramBot, send_test_message
from dhan_api import check_dhan_connection
from bias_analysis import BiasAnalysisPro
from advanced_chart_analysis import AdvancedChartAnalysis
from overall_market_sentiment import render_overall_market_sentiment, calculate_overall_sentiment, run_ai_analysis, shutdown_ai_engine
from advanced_proximity_alerts import get_proximity_alert_system
from data_cache_manager import (
    get_cache_manager,
    preload_all_data,
    get_cached_nifty_data,
    get_cached_sensex_data,
    get_cached_bias_analysis_results
)
from vob_signal_generator import VOBSignalGenerator
from htf_sr_signal_generator import HTFSRSignalGenerator
from ai_tab_integration import render_master_ai_analysis_tab, render_advanced_analytics_tab

# Configure logging
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG & PERFORMANCE OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NIFTY/SENSEX Trader",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════
# CUSTOM CSS - PREVENT BLUR/LOADING OVERLAY DURING REFRESH
# ═══════════════════════════════════════════════════════════════════════
# This CSS prevents the app from showing blur/white screen during refresh
# allowing users to continue viewing data while refresh happens in background

st.markdown("""
<style>
    /* Hide the Streamlit loading spinner and blur overlay */
    .stApp > div[data-testid="stAppViewContainer"] > div:first-child {
        display: none !important;
    }

    /* Prevent blur overlay during rerun */
    .stApp [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }

    /* Hide the "Running..." indicator in top right */
    .stApp [data-testid="stStatusWidget"] {
        visibility: hidden;
    }

    /* Hide all spinner elements */
    .stSpinner {
        display: none !important;
    }

    /* Hide loading indicator */
    div[data-testid="stLoadingIndicator"] {
        display: none !important;
    }

    /* Keep app content visible during refresh - no opacity change */
    .element-container {
        opacity: 1 !important;
        transition: none !important;
    }

    .stMarkdown {
        opacity: 1 !important;
        transition: none !important;
    }

    /* Prevent white flash during page reload */
    body {
        background-color: #0E1117;
        transition: none !important;
    }

    /* Smooth transitions for dynamic content - DISABLED to prevent blur */
    .stApp {
        transition: none !important;
    }

    /* Hide loading overlay completely */
    div[data-testid="stAppViewContainer"] > div[style*="position: absolute"] {
        display: none !important;
    }

    /* Ensure dataframes remain visible during refresh */
    .dataframe {
        opacity: 1 !important;
        transition: none !important;
    }

    /* Keep charts visible during refresh */
    .stPlotlyChart {
        opacity: 1 !important;
        transition: none !important;
    }

    /* Hide the app header spinner */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Prevent flickering on dynamic updates */
    section[data-testid="stSidebar"],
    section[data-testid="stMain"] {
        transition: none !important;
    }

    /* Keep all content visible - override any opacity changes */
    [data-testid="stVerticalBlock"] {
        opacity: 1 !important;
    }

    /* Remove blur filter if applied */
    * {
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }

    /* ═══════════════════════════════════════════════════════════════════ */
    /* MOBILE & DESKTOP RESPONSIVE IMPROVEMENTS */
    /* ═══════════════════════════════════════════════════════════════════ */

    /* Ensure tabs are scrollable on mobile */
    [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: thin;
    }

    /* Make tabs touch-friendly on mobile */
    [data-baseweb="tab"] {
        min-height: 48px !important;
        padding: 12px 16px !important;
    }

    /* Responsive font sizes for mobile */
    @media (max-width: 768px) {
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }

        /* Reduce padding on mobile */
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        /* Make tables scrollable on mobile */
        .dataframe-container {
            overflow-x: auto !important;
        }

        /* Stack columns on mobile */
        [data-testid="column"] {
            min-width: 100% !important;
        }
    }

    /* Improve touch targets for buttons */
    button {
        min-height: 44px !important;
        min-width: 44px !important;
    }

    /* Smooth scrolling for better UX */
    html {
        scroll-behavior: smooth;
    }
</style>
""", unsafe_allow_html=True)

# Performance optimization: Reduce widget refresh overhead
# This improves app responsiveness and reduces lag
if 'performance_mode' not in st.session_state:
    st.session_state.performance_mode = True

# ═══════════════════════════════════════════════════════════════════════
# AI MARKET ANALYSIS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Get API keys from Streamlit secrets
NEWSDATA_API_KEY = st.secrets.get("NEWSDATA_API_KEY", "")
PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY", "")

# Initialize AI analysis tracking
if 'last_ai_analysis_time' not in st.session_state:
    st.session_state.last_ai_analysis_time = 0

if 'ai_analysis_interval' not in st.session_state:
    # Run AI analysis every 30 minutes (1800 seconds)
    st.session_state.ai_analysis_interval = 1800

if 'ai_analysis_results' not in st.session_state:
    st.session_state.ai_analysis_results = None

async def run_ai_market_analysis():
    """Run AI market analysis and send alerts"""
    try:
        # Get overall market sentiment from cached sentiment
        overall_market = "NEUTRAL"
        if st.session_state.cached_sentiment:
            sentiment_map = {
                'BULLISH': 'BULL',
                'BEARISH': 'BEAR',
                'NEUTRAL': 'NEUTRAL'
            }
            overall_market = sentiment_map.get(st.session_state.cached_sentiment.get('overall_sentiment', 'NEUTRAL'), 'NEUTRAL')
        
        # Calculate module biases from various indicators
        module_biases = {
            "htf_sr": 0.5,  # Default neutral
            "vob": 0.5,
            "overall_sentiment": 0.5,
            "option_chain": 0.5,
            "proximity_alerts": 0.5,
        }
        
        # Try to get more accurate biases from actual data
        try:
            # Get bias analysis results if available
            if st.session_state.bias_analysis_results and st.session_state.bias_analysis_results.get('success'):
                bias_results = st.session_state.bias_analysis_results
                overall_score = bias_results.get('overall_score', 0)
                
                # Convert score to 0-1 bias
                if overall_score > 0:
                    module_biases["overall_sentiment"] = min(1.0, 0.5 + (overall_score / 200))
                elif overall_score < 0:
                    module_biases["overall_sentiment"] = max(0.0, 0.5 + (overall_score / 200))
        except:
            pass
        
        # Market metadata
        nifty_data = get_cached_nifty_data()
        market_meta = {
            "volatility": 0.15,  # Default
            "volume_change": 0.05,
            "query": "NSE India market",
            "current_price": nifty_data.get('spot_price', 0) if nifty_data else 0,
            "market_status": get_market_status().get('session', 'unknown')
        }
        
        # Run AI analysis with save and telegram send
        report = await run_ai_analysis(
            overall_market,
            module_biases,
            market_meta,
            news_api_key=NEWSDATA_API_KEY,  # Uses the variable from secrets
            perplexity_api_key=PERPLEXITY_API_KEY,  # Uses the variable from secrets
            save_report=True,
            telegram_send=True
        )
        
        if not report.get("triggered"):
            logger.info("AI not triggered: %s", report.get("reason"))
            return None
        
        logger.info("AI Market Report: label=%s confidence=%.2f recommendation=%s", 
                   report.get("label"), report.get("confidence"), report.get("recommendation"))
        
        return report
        
    except Exception as e:
        logger.error(f"Error in AI market analysis: {e}")
        return None

# Function to check and run AI analysis if needed
def check_and_run_ai_analysis():
    """Check if it's time to run AI analysis and run it if needed"""
    current_time = time.time()
    
    # Check if enough time has passed since last analysis
    if current_time - st.session_state.last_ai_analysis_time > st.session_state.ai_analysis_interval:
        # Only run during market hours for more relevant analysis
        market_status = get_market_status()
        if market_status.get('open', False) and market_status.get('session') == 'regular':
            # Update last analysis time
            st.session_state.last_ai_analysis_time = current_time
            
            # Run AI analysis asynchronously
            try:
                # We'll run it in a separate thread to not block the main app
                import threading
                
                async def run_async():
                    report = await run_ai_market_analysis()
                    if report:
                        st.session_state.ai_analysis_results = report
                
                # Start the async task in a new thread
                thread = threading.Thread(
                    target=lambda: asyncio.run(run_async()),
                    daemon=True
                )
                thread.start()
                
                return True
            except Exception as e:
                logger.error(f"Failed to start AI analysis thread: {e}")
    
    return False

# ═══════════════════════════════════════════════════════════════════════
# INITIALIZE SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

if 'signal_manager' not in st.session_state:
    st.session_state.signal_manager = SignalManager()

if 'vob_signal_generator' not in st.session_state:
    st.session_state.vob_signal_generator = VOBSignalGenerator(proximity_threshold=8.0)

if 'active_vob_signals' not in st.session_state:
    st.session_state.active_vob_signals = []

if 'last_vob_check_time' not in st.session_state:
    st.session_state.last_vob_check_time = 0

if 'htf_sr_signal_generator' not in st.session_state:
    st.session_state.htf_sr_signal_generator = HTFSRSignalGenerator(proximity_threshold=8.0)

if 'active_htf_sr_signals' not in st.session_state:
    st.session_state.active_htf_sr_signals = []

if 'last_htf_sr_check_time' not in st.session_state:
    st.session_state.last_htf_sr_check_time = 0


# VOB and HTF data storage
if 'vob_data_nifty' not in st.session_state:
    st.session_state.vob_data_nifty = None

if 'vob_data_sensex' not in st.session_state:
    st.session_state.vob_data_sensex = None

if 'htf_data_nifty' not in st.session_state:
    st.session_state.htf_data_nifty = None

if 'htf_data_sensex' not in st.session_state:
    st.session_state.htf_data_sensex = None

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

if 'active_setup_id' not in st.session_state:
    st.session_state.active_setup_id = None

# Lazy initialization - only create these objects when needed (on tab access)
# This significantly reduces initial load time
def get_bias_analyzer():
    """Lazy load bias analyzer"""
    if 'bias_analyzer' not in st.session_state:
        st.session_state.bias_analyzer = BiasAnalysisPro()
    return st.session_state.bias_analyzer

def get_advanced_chart_analyzer():
    """Lazy load advanced chart analyzer"""
    if 'advanced_chart_analyzer' not in st.session_state:
        st.session_state.advanced_chart_analyzer = AdvancedChartAnalysis()
    return st.session_state.advanced_chart_analyzer

if 'bias_analysis_results' not in st.session_state:
    st.session_state.bias_analysis_results = None

if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None

# Initialize background data loading
if 'data_preloaded' not in st.session_state:
    st.session_state.data_preloaded = False
    # Start background data loading on first run
    preload_all_data()
    st.session_state.data_preloaded = True

# NSE Options Analyzer - Initialize instruments session state
NSE_INSTRUMENTS = {
    'indices': {
        'NIFTY': {'lot_size': 75, 'zone_size': 20, 'atm_range': 200},
        'BANKNIFTY': {'lot_size': 25, 'zone_size': 100, 'atm_range': 500},
        'SENSEX': {'lot_size': 10, 'zone_size': 50, 'atm_range': 300},
        'NIFTY IT': {'lot_size': 50, 'zone_size': 50, 'atm_range': 300},
        'NIFTY AUTO': {'lot_size': 50, 'zone_size': 50, 'atm_range': 300}
    },
    'stocks': {
        'TCS': {'lot_size': 150, 'zone_size': 30, 'atm_range': 150},
        'RELIANCE': {'lot_size': 250, 'zone_size': 40, 'atm_range': 200},
        'HDFCBANK': {'lot_size': 550, 'zone_size': 50, 'atm_range': 250}
    }
}

# Initialize session states for all NSE instruments
for category in NSE_INSTRUMENTS:
    for instrument in NSE_INSTRUMENTS[category]:
        if f'{instrument}_price_data' not in st.session_state:
            st.session_state[f'{instrument}_price_data'] = pd.DataFrame(columns=["Time", "Spot"])

        if f'{instrument}_trade_log' not in st.session_state:
            st.session_state[f'{instrument}_trade_log'] = []

        if f'{instrument}_call_log_book' not in st.session_state:
            st.session_state[f'{instrument}_call_log_book'] = []

        if f'{instrument}_support_zone' not in st.session_state:
            st.session_state[f'{instrument}_support_zone'] = (None, None)

        if f'{instrument}_resistance_zone' not in st.session_state:
            st.session_state[f'{instrument}_resistance_zone'] = (None, None)

# Initialize overall option chain data
if 'overall_option_data' not in st.session_state:
    st.session_state['overall_option_data'] = {}

# ═══════════════════════════════════════════════════════════════════════
# AUTO REFRESH & PERFORMANCE OPTIMIZATIONS
# ═══════════════════════════════════════════════════════════════════════
# Optimized for fast loading and refresh:
# - Chart data cached for 60 seconds
# - Signal checks reduced to 30-second intervals
# - Lazy loading for tab-specific data
# - Streamlit caching for expensive computations

# Auto-refresh every 1 minute (configurable via AUTO_REFRESH_INTERVAL)
# This ensures the app stays updated with latest market data
# The refresh is seamless - no blur/flash thanks to custom CSS above
refresh_count = st_autorefresh(interval=AUTO_REFRESH_INTERVAL * 1000, key="data_refresh")

# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# Check and run AI analysis if needed
check_and_run_ai_analysis()

# ═══════════════════════════════════════════════════════════════════════
# MARKET HOURS WARNING BANNER
# ═══════════════════════════════════════════════════════════════════════

if MARKET_HOURS_ENABLED:
    should_run, reason = should_run_app()

    if not should_run:
        # Display prominent warning banner when market is closed
        st.error(f"""
        ⚠️ **MARKET CLOSED - APP RUNNING IN LIMITED MODE**

        **Reason:** {reason}

        **Trading Hours:** 8:30 AM - 3:45 PM IST (Monday - Friday, excluding holidays)

        The app will automatically resume full operation during market hours.
        Background data refresh is paused to conserve API quota.
        """)

        # Show next market open time if available
        market_status = get_market_status()
        if 'next_open' in market_status:
            st.info(f"📅 **Next Market Open:** {market_status['next_open']}")
    else:
        # Show market session info when market is open
        market_status = get_market_status()
        session = market_status.get('session', 'unknown')

        if session == 'pre_market':
            st.info(f"⏰ **{reason}** - Limited liquidity expected")
        elif session == 'post_market':
            st.warning(f"⏰ **{reason}** - Trading session ending soon")
        # Don't show banner during regular market hours to save space

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR - SYSTEM STATUS
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ System Status")
    
    # Market status
    market_status = get_market_status()
    if market_status['open']:
        st.success(f"{market_status['message']} | {market_status['time']}")
    else:
        st.error(market_status['message'])
    
    st.divider()
    
    # DhanHQ connection
    st.subheader("🔌 DhanHQ API")
    if DEMO_MODE:
        st.info("🧪 DEMO MODE Active")
    else:
        if check_dhan_connection():
            st.success("✅ Connected")
        else:
            st.error("❌ Connection Failed")
    
    st.divider()
    
    # Telegram status
    st.subheader("📱 Telegram Alerts")
    telegram_creds = get_telegram_credentials()
    if telegram_creds['enabled']:
        st.success("✅ Connected")
        if st.button("Send Test Message"):
            if send_test_message():
                st.success("Test message sent!")
            else:
                st.error("Failed to send")
    else:
        st.warning("⚠️ Not Configured")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    st.write(f"**Auto Refresh:** {AUTO_REFRESH_INTERVAL}s")
    st.write(f"**NIFTY Lot Size:** {LOT_SIZES['NIFTY']}")
    st.write(f"**SENSEX Lot Size:** {LOT_SIZES['SENSEX']}")
    st.write(f"**SL Offset:** {STOP_LOSS_OFFSET} points")
    
    st.divider()
    
    # AI Analysis Status
    st.subheader("🤖 AI Market Analysis (Powered by Perplexity AI)")
    if NEWSDATA_API_KEY and PERPLEXITY_API_KEY:
        st.success("✅ API Keys Configured (Perplexity AI)")
        
        # Show last AI analysis time
        if st.session_state.last_ai_analysis_time > 0:
            last_time = datetime.fromtimestamp(st.session_state.last_ai_analysis_time)
            time_str = last_time.strftime("%H:%M:%S")
            time_ago = int(time.time() - st.session_state.last_ai_analysis_time)
            
            if time_ago < 60:
                st.info(f"Last analysis: {time_ago}s ago")
            elif time_ago < 3600:
                st.info(f"Last analysis: {time_ago//60}m ago")
            else:
                st.info(f"Last analysis: {time_ago//3600}h ago")
        else:
            st.info("⏳ Never run")
        
        # Manual trigger button
        if st.button("Run AI Analysis Now", key="run_ai_analysis"):
            with st.spinner("Running AI market analysis..."):
                try:
                    # Run AI analysis
                    async def run_ai():
                        report = await run_ai_market_analysis()
                        if report:
                            st.session_state.ai_analysis_results = report
                            st.session_state.last_ai_analysis_time = time.time()
                            st.success(f"✅ AI Analysis Complete: {report.get('label')}")
                        else:
                            st.warning("⚠️ AI analysis not triggered (not enough signals)")
                    
                    # Run async in current event loop
                    import asyncio
                    asyncio.run(run_ai())
                except Exception as e:
                    st.error(f"❌ AI analysis failed: {e}")
    else:
        st.warning("⚠️ API Keys Required")
        st.caption("Set NEWSDATA_API_KEY and PERPLEXITY_API_KEY in Streamlit secrets (.streamlit/secrets.toml)")
    
    st.divider()

    # Background Data Loading Status
    st.subheader("🔄 Data Loading")
    cache_manager = get_cache_manager()

    # Check cache status for each data type
    data_status = {
        'Market Data': cache_manager.is_valid('nifty_data'),
        'Bias Analysis': cache_manager.is_valid('bias_analysis'),
    }

    for name, is_valid in data_status.items():
        if is_valid:
            st.success(f"✅ {name}")
        else:
            st.info(f"⏳ {name} Loading...")

    st.caption("🔄 Auto-refreshing every 60-120 seconds (optimized for performance)")

# ═══════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════

# Get NIFTY data from cache (loaded in background)
# PERFORMANCE OPTIMIZED: Never blocks - shows loading message if data not ready
nifty_data = get_cached_nifty_data()

if not nifty_data or not nifty_data.get('success'):
    # Show loading message (non-blocking)
    # Background thread will load data and it will appear on next refresh
    st.info("⏳ Loading NIFTY data in background... Please wait a moment and the data will appear automatically.")

    # Check if there's an error message to display
    if nifty_data and nifty_data.get('error'):
        st.error(f"⚠️ **Error:** {nifty_data['error']}")

        # Show help message if it's a credentials error
        if 'credentials' in nifty_data['error'].lower() or 'secrets.toml' in nifty_data['error'].lower():
            st.warning("""
            **Setup Required:**
            1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
            2. Fill in your DhanHQ API credentials
            3. Restart the application
            """)
    else:
        st.info("💡 **Performance Note:** Tab and button clicks should be instant now. Data loads in background without blocking UI.")

    # Use a placeholder/default data structure to prevent errors
    nifty_data = {
        'success': False,
        'spot_price': None,
        'atm_strike': None,
        'open': None,
        'high': None,
        'low': None,
        'close': None,
        'expiry_dates': [],
        'current_expiry': 'N/A',
        'chart_data': None,
        'error': nifty_data.get('error', 'Loading...') if nifty_data else 'Loading...',
        'timestamp': None
    }
    # Note: We don't stop() here - let the app continue and show what it can

# ═══════════════════════════════════════════════════════════════════════
# UPDATE BROWSER TAB TITLE WITH SPOT PRICE
# ═══════════════════════════════════════════════════════════════════════
# Dynamically update browser tab title to show NIFTY spot price for easy monitoring
if nifty_data and nifty_data.get('success') and nifty_data.get('spot_price'):
    spot_price = nifty_data['spot_price']
    # Use JavaScript to update the browser tab title
    st.markdown(f"""
    <script>
        document.title = "NIFTY ₹{spot_price:,.2f} | Trader";
    </script>
    """, unsafe_allow_html=True)
else:
    # Fallback title when spot price is not available
    st.markdown("""
    <script>
        document.title = "NIFTY/SENSEX Trader";
    </script>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# CACHED CHART DATA FETCHER (Performance Optimization)
# ═══════════════════════════════════════════════════════════════════════
# Cache chart data for 60 seconds to avoid repeated API calls
if 'chart_data_cache' not in st.session_state:
    st.session_state.chart_data_cache = {}
if 'chart_data_cache_time' not in st.session_state:
    st.session_state.chart_data_cache_time = {}

@st.cache_data(ttl=60, show_spinner=False)  # 60 seconds (1 minute)
def get_cached_chart_data(symbol, period, interval):
    """Cached chart data fetcher - reduces API calls"""
    chart_analyzer = AdvancedChartAnalysis()
    return chart_analyzer.fetch_intraday_data(symbol, period=period, interval=interval)

@st.cache_data(ttl=60, show_spinner=False)  # 60 seconds (1 minute)
def calculate_vob_indicators(df_key, sensitivity=5):
    """Cached VOB calculation - reduces redundant computations"""
    from indicators.volume_order_blocks import VolumeOrderBlocks

    # Get dataframe from cache
    cache_manager = get_cache_manager()
    df = cache_manager.get(df_key)

    if df is None or len(df) == 0:
        return None

    vob_indicator = VolumeOrderBlocks(sensitivity=sensitivity)
    return vob_indicator.calculate(df)

@st.cache_data(ttl=60, show_spinner=False)  # 60 seconds (1 minute)
def calculate_sentiment():
    """Cached sentiment calculation"""
    try:
        return calculate_overall_sentiment()
    except:
        return None

# ═══════════════════════════════════════════════════════════════════════
# VOB-BASED SIGNAL MONITORING (Optimized)
# ═══════════════════════════════════════════════════════════════════════
# Initialize sentiment cache in session state
if 'cached_sentiment' not in st.session_state:
    st.session_state.cached_sentiment = None
if 'sentiment_cache_time' not in st.session_state:
    st.session_state.sentiment_cache_time = 0

# Calculate sentiment once every 120 seconds and cache it (increased from 60s for better performance)
current_time = time.time()
if current_time - st.session_state.sentiment_cache_time > 120:
    sentiment_result = calculate_sentiment()
    if sentiment_result:
        st.session_state.cached_sentiment = sentiment_result
        st.session_state.sentiment_cache_time = current_time

# ═══════════════════════════════════════════════════════════════════════
# PERFORMANCE OPTIMIZATION: Only run expensive signal checks during market hours
# and increase check interval to reduce load (60 seconds instead of 30)
# ═══════════════════════════════════════════════════════════════════════
market_status = get_market_status()
should_run_signal_check = market_status.get('open', False)

# Check for VOB signals every 60 seconds (increased from 30s for better performance)
# Only run during market hours to reduce unnecessary processing
if should_run_signal_check and (current_time - st.session_state.last_vob_check_time > 60):
    st.session_state.last_vob_check_time = current_time

    try:
        # Use cached sentiment (calculated once per minute to avoid redundancy)
        if st.session_state.cached_sentiment:
            overall_sentiment = st.session_state.cached_sentiment.get('overall_sentiment', 'NEUTRAL')
        else:
            overall_sentiment = 'NEUTRAL'

        # Optimize: Create single VOB indicator instance for reuse
        from indicators.volume_order_blocks import VolumeOrderBlocks
        vob_indicator = VolumeOrderBlocks(sensitivity=5)

        # Fetch chart data and calculate VOB for NIFTY (using cached function)
        df = get_cached_chart_data('^NSEI', '1d', '1m')

        if df is not None and len(df) > 0:
            # Calculate VOB blocks (reusing indicator instance)
            vob_data = vob_indicator.calculate(df)

            # Store VOB data in session state for display
            st.session_state.vob_data_nifty = vob_data

            # Check for NIFTY signal - with safe spot price handling
            nifty_spot_for_signal = None
            if nifty_data.get('spot_price') and nifty_data['spot_price'] not in [None, 0, 'N/A']:
                try:
                    nifty_spot_for_signal = float(nifty_data['spot_price'])
                except (TypeError, ValueError):
                    nifty_spot_for_signal = None

            nifty_signal = None
            if nifty_spot_for_signal is not None:
                nifty_signal = st.session_state.vob_signal_generator.check_for_signal(
                    spot_price=nifty_spot_for_signal,
                    market_sentiment=overall_sentiment,
                    bullish_blocks=vob_data['bullish_blocks'],
                    bearish_blocks=vob_data['bearish_blocks'],
                    index='NIFTY',
                    df=df  # Pass dataframe for strength analysis
                )

            if nifty_signal:
                # Check if this is a new signal (not already in active signals)
                is_new = True
                for existing_signal in st.session_state.active_vob_signals:
                    if (existing_signal['index'] == nifty_signal['index'] and
                        existing_signal['direction'] == nifty_signal['direction'] and
                        abs(existing_signal['entry_price'] - nifty_signal['entry_price']) < 5):
                        is_new = False
                        break

                if is_new:
                    # Add to active signals
                    st.session_state.active_vob_signals.append(nifty_signal)

                    # Send telegram notification
                    telegram_bot = TelegramBot()
                    telegram_bot.send_vob_entry_signal(nifty_signal)

        # Fetch chart data and calculate VOB for SENSEX (using cached function)
        df_sensex = get_cached_chart_data('^BSESN', '1d', '1m')

        if df_sensex is not None and len(df_sensex) > 0:
            # Calculate VOB blocks for SENSEX (reusing indicator instance)
            vob_data_sensex = vob_indicator.calculate(df_sensex)

            # Store VOB data in session state for display
            st.session_state.vob_data_sensex = vob_data_sensex

            # Get SENSEX spot price
            sensex_data = get_cached_sensex_data()
            if sensex_data:
                # Safe handling for SENSEX spot price in signal check
                sensex_spot_for_signal = None
                if sensex_data.get('spot_price') and sensex_data['spot_price'] not in [None, 0, 'N/A']:
                    try:
                        sensex_spot_for_signal = float(sensex_data['spot_price'])
                    except (TypeError, ValueError):
                        sensex_spot_for_signal = None

                sensex_signal = None
                if sensex_spot_for_signal is not None:
                    sensex_signal = st.session_state.vob_signal_generator.check_for_signal(
                        spot_price=sensex_spot_for_signal,
                        market_sentiment=overall_sentiment,
                        bullish_blocks=vob_data_sensex['bullish_blocks'],
                        bearish_blocks=vob_data_sensex['bearish_blocks'],
                        index='SENSEX',
                        df=df_sensex  # Pass dataframe for strength analysis
                    )
            else:
                sensex_signal = None

                if sensex_signal:
                    # Check if this is a new signal
                    is_new = True
                    for existing_signal in st.session_state.active_vob_signals:
                        if (existing_signal['index'] == sensex_signal['index'] and
                            existing_signal['direction'] == sensex_signal['direction'] and
                            abs(existing_signal['entry_price'] - sensex_signal['entry_price']) < 5):
                            is_new = False
                            break

                    if is_new:
                        # Add to active signals
                        st.session_state.active_vob_signals.append(sensex_signal)

                        # Send telegram notification
                        telegram_bot = TelegramBot()
                        telegram_bot.send_vob_entry_signal(sensex_signal)

        # Clean up old signals (older than 30 minutes)
        st.session_state.active_vob_signals = [
            sig for sig in st.session_state.active_vob_signals
            if (current_time - sig['timestamp'].timestamp()) < 1800
        ]

        # ═══════════════════════════════════════════════════════════════
        # PROXIMITY ALERTS - Check if price is near VOB levels
        # ═══════════════════════════════════════════════════════════════
        try:
            proximity_system = get_proximity_alert_system(cooldown_minutes=10)

            # Check NIFTY VOB proximity
            if st.session_state.vob_data_nifty and nifty_data:
                # Safe handling for NIFTY spot price
                nifty_price = None
                if nifty_data.get('spot_price') and nifty_data['spot_price'] not in [None, 0, 'N/A']:
                    try:
                        nifty_price = float(nifty_data['spot_price'])
                    except (TypeError, ValueError):
                        nifty_price = None

                if nifty_price is not None:
                    # Convert HTF data to list format expected by proximity system
                    htf_data_list = []
                    if st.session_state.htf_data_nifty:
                        # htf_data_nifty is already a list, iterate directly
                        for level_data in st.session_state.htf_data_nifty:
                            if level_data:
                                htf_data_list.append({
                                    'timeframe': level_data.get('timeframe'),
                                    'pivot_high': level_data.get('pivot_high'),
                                    'pivot_low': level_data.get('pivot_low')
                                })

                    # Process proximity alerts for NIFTY
                    nifty_alerts, nifty_notifications = proximity_system.process_market_data(
                        symbol='NIFTY',
                        current_price=nifty_price,
                        vob_data=st.session_state.vob_data_nifty,
                        htf_data=htf_data_list
                    )

            # Check SENSEX VOB proximity
            if st.session_state.vob_data_sensex and sensex_data:
                # Safe handling for SENSEX spot price
                sensex_price = None
                if sensex_data.get('spot_price') and sensex_data['spot_price'] not in [None, 0, 'N/A']:
                    try:
                        sensex_price = float(sensex_data['spot_price'])
                    except (TypeError, ValueError):
                        sensex_price = None

                if sensex_price is not None:
                    # Convert HTF data to list format
                    htf_data_list_sensex = []
                    if st.session_state.htf_data_sensex:
                        # htf_data_sensex is already a list, iterate directly
                        for level_data in st.session_state.htf_data_sensex:
                            if level_data:
                                htf_data_list_sensex.append({
                                    'timeframe': level_data.get('timeframe'),
                                    'pivot_high': level_data.get('pivot_high'),
                                    'pivot_low': level_data.get('pivot_low')
                                })

                    # Process proximity alerts for SENSEX
                    sensex_alerts, sensex_notifications = proximity_system.process_market_data(
                        symbol='SENSEX',
                        current_price=sensex_price,
                        vob_data=st.session_state.vob_data_sensex,
                        htf_data=htf_data_list_sensex
                    )

        except Exception as prox_e:
            # Silently fail proximity alerts
            pass


    except Exception as e:
        # Silently fail - don't disrupt the app
        pass

# ═══════════════════════════════════════════════════════════════════════
# HTF SUPPORT/RESISTANCE SIGNAL MONITORING (Optimized)
# ═══════════════════════════════════════════════════════════════════════
# Check for HTF S/R signals every 60 seconds (increased from 30s for better performance)
# Only run during market hours to reduce unnecessary processing
if should_run_signal_check and (current_time - st.session_state.last_htf_sr_check_time > 60):
    st.session_state.last_htf_sr_check_time = current_time

    try:
        # Use cached sentiment (calculated once per minute to avoid redundancy)
        if st.session_state.cached_sentiment:
            overall_sentiment = st.session_state.cached_sentiment.get('overall_sentiment', 'NEUTRAL')
        else:
            overall_sentiment = 'NEUTRAL'

        # Only check if market sentiment is BULLISH or BEARISH
        if overall_sentiment in ['BULLISH', 'BEARISH']:
            # Optimize: Create single HTF S/R indicator instance for reuse
            from indicators.htf_support_resistance import HTFSupportResistance
            htf_sr = HTFSupportResistance()
            levels_config = [
                {'timeframe': '5T', 'length': 5},
                {'timeframe': '10T', 'length': 5},
                {'timeframe': '15T', 'length': 5}
            ]

            # Fetch chart data for NIFTY (using cached function)
            df_nifty = get_cached_chart_data('^NSEI', '7d', '1m')

            if df_nifty is not None and len(df_nifty) > 0:
                # Calculate HTF S/R levels (reusing indicator instance)
                htf_levels = htf_sr.calculate_multi_timeframe(df_nifty, levels_config)

                # Store HTF data in session state for display
                st.session_state.htf_data_nifty = htf_levels

                # Check for NIFTY signal - with safe spot price handling
                nifty_spot_htf = None
                if nifty_data.get('spot_price') and nifty_data['spot_price'] not in [None, 0, 'N/A']:
                    try:
                        nifty_spot_htf = float(nifty_data['spot_price'])
                    except (TypeError, ValueError):
                        nifty_spot_htf = None

                nifty_htf_signal = None
                if nifty_spot_htf is not None:
                    nifty_htf_signal = st.session_state.htf_sr_signal_generator.check_for_signal(
                        spot_price=nifty_spot_htf,
                        market_sentiment=overall_sentiment,
                        htf_levels=htf_levels,
                        index='NIFTY',
                        df=df_nifty  # Pass dataframe for strength analysis
                    )

                if nifty_htf_signal:
                    # Check if this is a new signal (not already in active signals)
                    is_new = True
                    for existing_signal in st.session_state.active_htf_sr_signals:
                        if (existing_signal['index'] == nifty_htf_signal['index'] and
                            existing_signal['direction'] == nifty_htf_signal['direction'] and
                            abs(existing_signal['entry_price'] - nifty_htf_signal['entry_price']) < 5):
                            is_new = False
                            break

                    if is_new:
                        # Add to active signals
                        st.session_state.active_htf_sr_signals.append(nifty_htf_signal)

                        # Send telegram notification
                        telegram_bot = TelegramBot()
                        telegram_bot.send_htf_sr_entry_signal(nifty_htf_signal)

            # Fetch chart data for SENSEX (using cached function)
            df_sensex = get_cached_chart_data('^BSESN', '7d', '1m')

            if df_sensex is not None and len(df_sensex) > 0:
                # Calculate HTF S/R levels for SENSEX (reusing indicator instance)
                htf_levels_sensex = htf_sr.calculate_multi_timeframe(df_sensex, levels_config)

                # Store HTF data in session state for display
                st.session_state.htf_data_sensex = htf_levels_sensex

                # Get SENSEX spot price
                sensex_data = get_cached_sensex_data()
                if sensex_data:
                    # Safe handling for SENSEX spot price in HTF signal check
                    sensex_spot_htf = None
                    if sensex_data.get('spot_price') and sensex_data['spot_price'] not in [None, 0, 'N/A']:
                        try:
                            sensex_spot_htf = float(sensex_data['spot_price'])
                        except (TypeError, ValueError):
                            sensex_spot_htf = None

                    sensex_htf_signal = None
                    if sensex_spot_htf is not None:
                        sensex_htf_signal = st.session_state.htf_sr_signal_generator.check_for_signal(
                            spot_price=sensex_spot_htf,
                            market_sentiment=overall_sentiment,
                            htf_levels=htf_levels_sensex,
                            index='SENSEX',
                            df=df_sensex  # Pass dataframe for strength analysis
                        )
                else:
                    sensex_htf_signal = None

                if sensex_htf_signal:
                    # Check if this is a new signal
                    is_new = True
                    for existing_signal in st.session_state.active_htf_sr_signals:
                        if (existing_signal['index'] == sensex_htf_signal['index'] and
                            existing_signal['direction'] == sensex_htf_signal['direction'] and
                            abs(existing_signal['entry_price'] - sensex_htf_signal['entry_price']) < 5):
                            is_new = False
                            break

                    if is_new:
                        # Add to active signals
                        st.session_state.active_htf_sr_signals.append(sensex_htf_signal)

                        # Send telegram notification
                        telegram_bot = TelegramBot()
                        telegram_bot.send_htf_sr_entry_signal(sensex_htf_signal)

            # Clean up old HTF S/R signals (older than 30 minutes)
            st.session_state.active_htf_sr_signals = [
                sig for sig in st.session_state.active_htf_sr_signals
                if (current_time - sig['timestamp'].timestamp()) < 1800
            ]

    except Exception as e:
        # Silently fail - don't disrupt the app
        pass

# Display market data
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Handle None or 0 values for spot price
    if nifty_data.get('spot_price') is not None and nifty_data['spot_price'] != 0:
        st.metric(
            "NIFTY Spot",
            f"₹{nifty_data['spot_price']:,.2f}",
            delta=None
        )
    else:
        st.metric(
            "NIFTY Spot",
            "N/A",
            delta=None
        )
        if nifty_data.get('error'):
            st.error(f"⚠️ {nifty_data['error']}")

with col2:
    # Handle None or 0 values for ATM strike
    if nifty_data.get('atm_strike') is not None and nifty_data['atm_strike'] != 0:
        st.metric(
            "ATM Strike",
            f"{nifty_data['atm_strike']}"
        )
    else:
        st.metric(
            "ATM Strike",
            "N/A"
        )

with col3:
    st.metric(
        "Current Expiry",
        nifty_data.get('current_expiry', 'N/A')
    )

with col4:
    # Show data freshness status
    cache_manager = get_cache_manager()
    nifty_cache_time = cache_manager._cache_timestamps.get('nifty_data', 0)
    if nifty_cache_time > 0:
        age_seconds = int(time.time() - nifty_cache_time)
        if age_seconds < 15:
            st.success(f"🟢 Live ({age_seconds}s ago)")
        elif age_seconds < 60:
            st.info(f"🔵 Fresh ({age_seconds}s ago)")
        else:
            st.warning(f"🟡 Updating...")
    else:
        st.info("📅 Loading...")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# VOB TRADING SIGNALS DISPLAY
# ═══════════════════════════════════════════════════════════════════════
st.markdown("### 🎯 NIFTY/SENSEX Manual Trader")
st.markdown("**VOB-Based Trading | Manual Signal Entry**")

# ═══════════════════════════════════════════════════════════════════
# 🎯 CURRENT POSITION & ACTION - MOST IMPORTANT INFO FIRST!
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🎯 CURRENT POSITION & ACTION")

# Get current price
current_price_app = 0.0
if 'bias_analysis_results' in st.session_state and isinstance(st.session_state.bias_analysis_results, dict):
    df_app = st.session_state.bias_analysis_results.get('df')
    if df_app is not None and len(df_app) > 0:
        current_price_app = df_app['close'].iloc[-1]

# Get VOB data for nearest levels
nearest_support_app = None
nearest_resistance_app = None

if 'vob_data_nifty' in st.session_state and st.session_state.vob_data_nifty:
    vob_data_app = st.session_state.vob_data_nifty

    # Find nearest bullish VOB below current price (support)
    bullish_blocks_app = vob_data_app.get('bullish_blocks', [])
    min_dist_sup_app = float('inf')
    for block in bullish_blocks_app:
        if isinstance(block, dict):
            block_mid = (block.get('upper', 0) + block.get('lower', 0)) / 2
            if block_mid < current_price_app:
                dist = current_price_app - block_mid
                if dist < min_dist_sup_app:
                    min_dist_sup_app = dist
                    nearest_support_app = {
                        'price': block_mid,
                        'type': 'VOB Support',
                        'lower': block.get('lower', block_mid),
                        'upper': block.get('upper', block_mid)
                    }

    # Find nearest bearish VOB above current price (resistance)
    bearish_blocks_app = vob_data_app.get('bearish_blocks', [])
    min_dist_res_app = float('inf')
    for block in bearish_blocks_app:
        if isinstance(block, dict):
            block_mid = (block.get('upper', 0) + block.get('lower', 0)) / 2
            if block_mid > current_price_app:
                dist = block_mid - current_price_app
                if dist < min_dist_res_app:
                    min_dist_res_app = dist
                    nearest_resistance_app = {
                        'price': block_mid,
                        'type': 'VOB Resistance',
                        'lower': block.get('lower', block_mid),
                        'upper': block.get('upper', block_mid)
                    }

# Display action based on position
if current_price_app > 0 and nearest_support_app and nearest_resistance_app:
    dist_to_sup_app = current_price_app - nearest_support_app['price']
    dist_to_res_app = nearest_resistance_app['price'] - current_price_app

    if dist_to_sup_app <= 5:
        st.success(f"""
### 🟢 AT SUPPORT - LONG SETUP ACTIVE

**Entry NOW:** ₹{nearest_support_app['lower']:,.0f} - ₹{nearest_support_app['upper']:,.0f} ({nearest_support_app['type']})
**Stop Loss:** ₹{nearest_support_app['lower'] - 20:,.0f} (below support zone)
**Target 1:** ₹{current_price_app + 30:,.0f} (+30 pts, Quick scalp)
**Target 2:** ₹{nearest_resistance_app['price']:,.0f} (Next resistance, +{dist_to_res_app + dist_to_sup_app:.0f} pts)

**✅ Entry Confirmation Required:**
1. Price bounces FROM support zone (don't chase if already moved up)
2. Volume increases on bounce candle
3. Regime supports LONG (check AI Trading Signal tab)
4. ATM Bias BULLISH (check NIFTY Option Screener tab)
        """)
    elif dist_to_res_app <= 5:
        st.error(f"""
### 🔴 AT RESISTANCE - SHORT SETUP ACTIVE

**Entry NOW:** ₹{nearest_resistance_app['lower']:,.0f} - ₹{nearest_resistance_app['upper']:,.0f} ({nearest_resistance_app['type']})
**Stop Loss:** ₹{nearest_resistance_app['upper'] + 20:,.0f} (above resistance zone)
**Target 1:** ₹{current_price_app - 30:,.0f} (-30 pts, Quick scalp)
**Target 2:** ₹{nearest_support_app['price']:,.0f} (Next support, -{dist_to_res_app + dist_to_sup_app:.0f} pts)

**✅ Entry Confirmation Required:**
1. Price rejects FROM resistance zone (don't chase if already moved down)
2. Volume increases on rejection candle
3. Regime supports SHORT (check AI Trading Signal tab)
4. ATM Bias BEARISH (check NIFTY Option Screener tab)
        """)
    else:
        st.info(f"""
### ⚠️ MID-ZONE - WAIT FOR ENTRY ZONES

**Current Price:** ₹{current_price_app:,.2f}
**Nearest Support:** ₹{nearest_support_app['price']:,.0f} (-{dist_to_sup_app:.0f} pts) - {nearest_support_app['type']}
**Nearest Resistance:** ₹{nearest_resistance_app['price']:,.0f} (+{dist_to_res_app:.0f} pts) - {nearest_resistance_app['type']}

**🚫 DO NOT TRADE HERE:**
- Poor risk/reward ratio in the middle
- Wait for price to reach entry zones (±5 pts of levels)
- Set alerts at ₹{nearest_support_app['price']:,.0f} (LONG) and ₹{nearest_resistance_app['price']:,.0f} (SHORT)

**Missing a trade is 100x better than a bad entry!**
        """)
elif current_price_app > 0:
    st.warning(f"""
### ⏳ LOADING ENTRY ZONES...

**Current Price:** ₹{current_price_app:,.2f}

Waiting for Volume Order Block data to identify precise entry zones.
Check **Advanced Chart Analysis** tab to load VOB data.
    """)
else:
    st.warning("""
### ⏳ WAITING FOR MARKET DATA...

Loading current price and entry zones. Please wait...
    """)

st.markdown("---")

if st.session_state.active_vob_signals:
    for signal in st.session_state.active_vob_signals:
        signal_emoji = "🟢" if signal['direction'] == 'CALL' else "🔴"
        direction_label = "BULLISH" if signal['direction'] == 'CALL' else "BEARISH"
        sentiment_color = "#26ba9f" if signal['market_sentiment'] == 'BULLISH' else "#ba2646"

        # Get strength data if available
        strength = signal.get('strength')
        strength_html = ""
        if strength:
            strength_score = strength.get('strength_score', 0)
            strength_label = strength.get('strength_label', 'UNKNOWN')
            trend = strength.get('trend', 'UNKNOWN')
            times_tested = strength.get('times_tested', 0)
            respect_rate = strength.get('respect_rate', 0)

            # Determine strength color
            if strength_score >= 70:
                strength_color = "#4caf50"  # Green
            elif strength_score >= 50:
                strength_color = "#ffc107"  # Yellow
            else:
                strength_color = "#ff5252"  # Red

            # Trend indicator
            trend_emoji = "🔺" if trend == "STRENGTHENING" else "🔻" if trend == "WEAKENING" else "➖"

            strength_html = f"""
                <hr style='margin: 10px 0;'>
                <div style='background-color: rgba(0,0,0,0.05); padding: 10px; border-radius: 5px;'>
                    <p style='margin: 0; font-size: 14px; font-weight: bold;'>📊 Order Block Strength Analysis</p>
                    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px;'>
                        <div>
                            <p style='margin: 0; font-size: 11px; color: #888;'>Strength Score</p>
                            <p style='margin: 0; font-size: 16px; font-weight: bold; color: {strength_color};'>{strength_score}/100</p>
                        </div>
                        <div>
                            <p style='margin: 0; font-size: 11px; color: #888;'>Status</p>
                            <p style='margin: 0; font-size: 14px;'>{strength_label.replace('_', ' ')}</p>
                        </div>
                        <div>
                            <p style='margin: 0; font-size: 11px; color: #888;'>Trend</p>
                            <p style='margin: 0; font-size: 14px;'>{trend_emoji} {trend}</p>
                        </div>
                        <div>
                            <p style='margin: 0; font-size: 11px; color: #888;'>Tests / Respect Rate</p>
                            <p style='margin: 0; font-size: 14px;'>{times_tested} / {respect_rate}%</p>
                        </div>
                    </div>
                </div>
            """

        with st.container():
            st.markdown(f"""
            <div style='border: 2px solid {sentiment_color}; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: rgba(38, 186, 159, 0.1);'>
                <h3 style='margin: 0;'>{signal_emoji} {signal['index']} {direction_label} ENTRY SIGNAL</h3>
                <p style='margin: 5px 0;'><b>Market Sentiment:</b> {signal['market_sentiment']}</p>
                <hr style='margin: 10px 0;'>
                <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;'>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>Entry Price</p>
                        <p style='margin: 0; font-size: 18px; font-weight: bold;'>₹{signal['entry_price']}</p>
                    </div>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>Stop Loss</p>
                        <p style='margin: 0; font-size: 18px; font-weight: bold; color: #ff5252;'>₹{signal['stop_loss']}</p>
                    </div>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>Target</p>
                        <p style='margin: 0; font-size: 18px; font-weight: bold; color: #4caf50;'>₹{signal['target']}</p>
                    </div>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>Risk:Reward</p>
                        <p style='margin: 0; font-size: 18px; font-weight: bold;'>{signal['risk_reward']}</p>
                    </div>
                </div>
                <hr style='margin: 10px 0;'>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>VOB Level</p>
                        <p style='margin: 0; font-size: 14px;'>₹{signal['vob_level']}</p>
                    </div>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>Distance from VOB</p>
                        <p style='margin: 0; font-size: 14px;'>{signal['distance_from_vob']} points</p>
                    </div>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>Signal Time</p>
                        <p style='margin: 0; font-size: 14px;'>{signal['timestamp'].strftime('%H:%M:%S')}</p>
                    </div>
                </div>
                {strength_html}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("⏳ Monitoring market for VOB-based entry signals... No active signals at the moment.")
    st.caption("Signals are generated when spot price is within 8 points of a Volume Order Block and aligned with overall market sentiment.")

# Display VOB Summary for NIFTY and SENSEX
st.markdown("#### 📈 Volume Order Block Status")

col1, col2 = st.columns(2)

# NIFTY VOB Summary
with col1:
    st.markdown("**NIFTY VOB**")
    if st.session_state.vob_data_nifty:
        from indicators.vob_strength_tracker import VOBStrengthTracker
        vob_tracker = VOBStrengthTracker()

        df_nifty = get_cached_chart_data('^NSEI', '1d', '1m')

        # Get latest bullish and bearish blocks
        bullish_blocks = st.session_state.vob_data_nifty.get('bullish_blocks', [])
        bearish_blocks = st.session_state.vob_data_nifty.get('bearish_blocks', [])

        # Calculate strength for latest blocks
        bull_strength = None
        bear_strength = None

        if bullish_blocks and df_nifty is not None:
            latest_bull = bullish_blocks[-1]
            bull_strength = vob_tracker.calculate_strength(latest_bull, df_nifty)

        if bearish_blocks and df_nifty is not None:
            latest_bear = bearish_blocks[-1]
            bear_strength = vob_tracker.calculate_strength(latest_bear, df_nifty)

        # Display bullish block info
        if bull_strength:
            trend_emoji = "🔺" if bull_strength['trend'] == "STRENGTHENING" else "🔻" if bull_strength['trend'] == "WEAKENING" else "➖"
            strength_color = "#4caf50" if bull_strength['strength_score'] >= 70 else "#ffc107" if bull_strength['strength_score'] >= 50 else "#ff5252"
            st.markdown(f"**🟢 Bullish VOB:** ₹{latest_bull['lower']:.2f} - ₹{latest_bull['upper']:.2f}")
            st.markdown(f"**Strength:** <span style='color: {strength_color}; font-weight: bold;'>{bull_strength['strength_score']}/100</span> {trend_emoji} {bull_strength['trend']}", unsafe_allow_html=True)
        else:
            st.info("No bullish VOB data available")

        # Display bearish block info
        if bear_strength:
            trend_emoji = "🔺" if bear_strength['trend'] == "STRENGTHENING" else "🔻" if bear_strength['trend'] == "WEAKENING" else "➖"
            strength_color = "#4caf50" if bear_strength['strength_score'] >= 70 else "#ffc107" if bear_strength['strength_score'] >= 50 else "#ff5252"
            st.markdown(f"**🔴 Bearish VOB:** ₹{latest_bear['lower']:.2f} - ₹{latest_bear['upper']:.2f}")
            st.markdown(f"**Strength:** <span style='color: {strength_color}; font-weight: bold;'>{bear_strength['strength_score']}/100</span> {trend_emoji} {bear_strength['trend']}", unsafe_allow_html=True)
        else:
            st.info("No bearish VOB data available")
    else:
        st.info("VOB data loading...")

# SENSEX VOB Summary
with col2:
    st.markdown("**SENSEX VOB**")
    if st.session_state.vob_data_sensex:
        from indicators.vob_strength_tracker import VOBStrengthTracker
        vob_tracker = VOBStrengthTracker()

        df_sensex = get_cached_chart_data('^BSESN', '1d', '1m')

        # Get latest bullish and bearish blocks
        bullish_blocks = st.session_state.vob_data_sensex.get('bullish_blocks', [])
        bearish_blocks = st.session_state.vob_data_sensex.get('bearish_blocks', [])

        # Calculate strength for latest blocks
        bull_strength = None
        bear_strength = None

        if bullish_blocks and df_sensex is not None:
            latest_bull = bullish_blocks[-1]
            bull_strength = vob_tracker.calculate_strength(latest_bull, df_sensex)

        if bearish_blocks and df_sensex is not None:
            latest_bear = bearish_blocks[-1]
            bear_strength = vob_tracker.calculate_strength(latest_bear, df_sensex)

        # Display bullish block info
        if bull_strength:
            trend_emoji = "🔺" if bull_strength['trend'] == "STRENGTHENING" else "🔻" if bull_strength['trend'] == "WEAKENING" else "➖"
            strength_color = "#4caf50" if bull_strength['strength_score'] >= 70 else "#ffc107" if bull_strength['strength_score'] >= 50 else "#ff5252"
            st.markdown(f"**🟢 Bullish VOB:** ₹{latest_bull['lower']:.2f} - ₹{latest_bull['upper']:.2f}")
            st.markdown(f"**Strength:** <span style='color: {strength_color}; font-weight: bold;'>{bull_strength['strength_score']}/100</span> {trend_emoji} {bull_strength['trend']}", unsafe_allow_html=True)
        else:
            st.info("No bullish VOB data available")

        # Display bearish block info
        if bear_strength:
            trend_emoji = "🔺" if bear_strength['trend'] == "STRENGTHENING" else "🔻" if bear_strength['trend'] == "WEAKENING" else "➖"
            strength_color = "#4caf50" if bear_strength['strength_score'] >= 70 else "#ffc107" if bear_strength['strength_score'] >= 50 else "#ff5252"
            st.markdown(f"**🔴 Bearish VOB:** ₹{latest_bear['lower']:.2f} - ₹{latest_bear['upper']:.2f}")
            st.markdown(f"**Strength:** <span style='color: {strength_color}; font-weight: bold;'>{bear_strength['strength_score']}/100</span> {trend_emoji} {bear_strength['trend']}", unsafe_allow_html=True)
        else:
            st.info("No bearish VOB data available")
    else:
        st.info("VOB data loading...")

# Add button to send VOB status to Telegram
st.markdown("")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("📱 Send VOB Status to Telegram", use_container_width=True):
        try:
            from indicators.vob_strength_tracker import VOBStrengthTracker
            vob_tracker = VOBStrengthTracker()

            # Prepare NIFTY VOB data
            nifty_vob_summary = {}
            if st.session_state.vob_data_nifty:
                df_nifty = get_cached_chart_data('^NSEI', '1d', '1m')
                if df_nifty is not None:
                    bullish_blocks = st.session_state.vob_data_nifty.get('bullish_blocks', [])
                    bearish_blocks = st.session_state.vob_data_nifty.get('bearish_blocks', [])

                    if bullish_blocks:
                        latest_bull = bullish_blocks[-1]
                        bull_strength = vob_tracker.calculate_strength(latest_bull, df_nifty)
                        nifty_vob_summary['bullish'] = {
                            'lower': latest_bull['lower'],
                            'upper': latest_bull['upper'],
                            'strength_score': bull_strength['strength_score'],
                            'trend': bull_strength['trend']
                        }

                    if bearish_blocks:
                        latest_bear = bearish_blocks[-1]
                        bear_strength = vob_tracker.calculate_strength(latest_bear, df_nifty)
                        nifty_vob_summary['bearish'] = {
                            'lower': latest_bear['lower'],
                            'upper': latest_bear['upper'],
                            'strength_score': bear_strength['strength_score'],
                            'trend': bear_strength['trend']
                        }

            # Prepare SENSEX VOB data
            sensex_vob_summary = {}
            if st.session_state.vob_data_sensex:
                df_sensex = get_cached_chart_data('^BSESN', '1d', '1m')
                if df_sensex is not None:
                    bullish_blocks = st.session_state.vob_data_sensex.get('bullish_blocks', [])
                    bearish_blocks = st.session_state.vob_data_sensex.get('bearish_blocks', [])

                    if bullish_blocks:
                        latest_bull = bullish_blocks[-1]
                        bull_strength = vob_tracker.calculate_strength(latest_bull, df_sensex)
                        sensex_vob_summary['bullish'] = {
                            'lower': latest_bull['lower'],
                            'upper': latest_bull['upper'],
                            'strength_score': bull_strength['strength_score'],
                            'trend': bull_strength['trend']
                        }

                    if bearish_blocks:
                        latest_bear = bearish_blocks[-1]
                        bear_strength = vob_tracker.calculate_strength(latest_bear, df_sensex)
                        sensex_vob_summary['bearish'] = {
                            'lower': latest_bear['lower'],
                            'upper': latest_bear['upper'],
                            'strength_score': bear_strength['strength_score'],
                            'trend': bear_strength['trend']
                        }

            # Send to Telegram
            if nifty_vob_summary or sensex_vob_summary:
                telegram_bot = TelegramBot()
                success = telegram_bot.send_vob_status_summary(nifty_vob_summary, sensex_vob_summary)
                if success:
                    st.success("✅ VOB status sent to Telegram!")
                else:
                    st.error("❌ Failed to send to Telegram. Check your Telegram credentials.")
            else:
                st.warning("⚠️ No VOB data available to send")

        except Exception as e:
            st.error(f"❌ Error sending VOB status: {str(e)}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# AI MARKET ANALYSIS RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════════════

# Display AI analysis results if available
if st.session_state.ai_analysis_results:
    report = st.session_state.ai_analysis_results
    
    st.markdown("### 🤖 AI Market Analysis")
    
    # Determine color based on label
    label = report.get('label', 'NEUTRAL')
    if label == 'BULL':
        label_color = "#4caf50"
        label_emoji = "🐂"
    elif label == 'BEAR':
        label_color = "#f44336"
        label_emoji = "🐻"
    else:
        label_color = "#ff9800"
        label_emoji = "⚖️"
    
    # Confidence level
    confidence = report.get('confidence', 0)
    if confidence >= 80:
        confidence_color = "#4caf50"
        confidence_text = "High"
    elif confidence >= 60:
        confidence_color = "#ff9800"
        confidence_text = "Moderate"
    else:
        confidence_color = "#f44336"
        confidence_text = "Low"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"<h2 style='color:{label_color}; text-align: center;'>{label_emoji} {label}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Market Direction</p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<h2 style='color:{confidence_color}; text-align: center;'>{confidence:.0f}%</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Confidence ({confidence_text})</p>", unsafe_allow_html=True)
    
    with col3:
        # Show report age
        if 'timestamp' in report:
            try:
                report_time = datetime.fromisoformat(str(report['timestamp']).replace('Z', '+00:00'))
                now = datetime.now(report_time.tzinfo) if report_time.tzinfo else datetime.now()
                age_minutes = int((now - report_time).total_seconds() / 60)
                
                if age_minutes < 1:
                    age_text = "Just now"
                elif age_minutes == 1:
                    age_text = "1 minute ago"
                else:
                    age_text = f"{age_minutes} minutes ago"
                
                st.markdown(f"<h4 style='text-align: center;'>{age_text}</h4>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Last Updated</p>", unsafe_allow_html=True)
            except:
                st.markdown("<p style='text-align: center;'>Time unknown</p>", unsafe_allow_html=True)
    
    # Recommendation
    st.markdown("#### 💡 AI Recommendation")
    recommendation = report.get('recommendation', 'No specific recommendation')
    st.info(recommendation)
    
    # Key Findings
    if 'key_findings' in report and report['key_findings']:
        with st.expander("📊 Key Findings", expanded=False):
            for finding in report['key_findings'][:5]:  # Show top 5 findings
                st.write(f"• {finding}")
    
    # Risk Assessment
    if 'risk_assessment' in report:
        with st.expander("⚠️ Risk Assessment", expanded=False):
            risk = report['risk_assessment']
            st.write(f"**Level:** {risk.get('level', 'Medium')}")
            st.write(f"**Score:** {risk.get('score', 50)}/100")
            if 'factors' in risk:
                st.write("**Factors:**")
                for factor in risk.get('factors', []):
                    st.write(f"  - {factor}")
    
    # View full report button
    if st.button("📄 View Full AI Report", key="view_full_ai_report"):
        # Show the full report in a modal or expander
        with st.expander("Full AI Market Analysis Report", expanded=True):
            st.json(report)
    
    st.divider()

# ═══════════════════════════════════════════════════════════════════════
# HTF S/R TRADING SIGNALS DISPLAY
# ═══════════════════════════════════════════════════════════════════════
st.markdown("### 📊 HTF Support/Resistance Signals")
st.markdown("**HTF S/R Based Trading | 5min, 10min, 15min Timeframes**")

# Display VOB and HTF S/R Status Summary
st.markdown("#### 📊 VOB & HTF S/R Strength Status")

col1, col2 = st.columns(2)

# NIFTY Summary
with col1:
    st.markdown("**NIFTY**")

    # VOB Status
    if st.session_state.vob_data_nifty:
        from indicators.vob_strength_tracker import VOBStrengthTracker
        vob_tracker = VOBStrengthTracker()
        df_nifty = get_cached_chart_data('^NSEI', '1d', '1m')

        bullish_blocks = st.session_state.vob_data_nifty.get('bullish_blocks', [])
        bearish_blocks = st.session_state.vob_data_nifty.get('bearish_blocks', [])

        if bullish_blocks and df_nifty is not None:
            latest_bull = bullish_blocks[-1]
            bull_strength = vob_tracker.calculate_strength(latest_bull, df_nifty)
            trend_emoji = "🔺" if bull_strength['trend'] == "STRENGTHENING" else "🔻" if bull_strength['trend'] == "WEAKENING" else "➖"
            st.caption(f"**VOB Bull:** {bull_strength['strength_score']}/100 {trend_emoji} {bull_strength['trend']}")

        if bearish_blocks and df_nifty is not None:
            latest_bear = bearish_blocks[-1]
            bear_strength = vob_tracker.calculate_strength(latest_bear, df_nifty)
            trend_emoji = "🔺" if bear_strength['trend'] == "STRENGTHENING" else "🔻" if bear_strength['trend'] == "WEAKENING" else "➖"
            st.caption(f"**VOB Bear:** {bear_strength['strength_score']}/100 {trend_emoji} {bear_strength['trend']}")

    # HTF S/R Status
    if st.session_state.htf_data_nifty:
        from indicators.htf_sr_strength_tracker import HTFSRStrengthTracker
        htf_tracker = HTFSRStrengthTracker()
        df_nifty = get_cached_chart_data('^NSEI', '7d', '1m')

        # Get latest support and resistance levels
        # htf_data_nifty is a list, iterate directly
        for level_data in st.session_state.htf_data_nifty:
            if level_data and df_nifty is not None:
                timeframe = level_data.get('timeframe')
                support = level_data.get('pivot_low')
                resistance = level_data.get('pivot_high')

                if support:
                    support_strength = htf_tracker.calculate_strength(support, 'SUPPORT', df_nifty, lookback_periods=100)
                    trend_emoji = "🔺" if support_strength['trend'] == "STRENGTHENING" else "🔻" if support_strength['trend'] == "WEAKENING" else "➖"
                    st.caption(f"**HTF Support ({timeframe}):** {support_strength['strength_score']}/100 {trend_emoji} {support_strength['trend']}")

                if resistance:
                    resistance_strength = htf_tracker.calculate_strength(resistance, 'RESISTANCE', df_nifty, lookback_periods=100)
                    trend_emoji = "🔺" if resistance_strength['trend'] == "STRENGTHENING" else "🔻" if resistance_strength['trend'] == "WEAKENING" else "➖"
                    st.caption(f"**HTF Resistance ({timeframe}):** {resistance_strength['strength_score']}/100 {trend_emoji} {resistance_strength['trend']}")

                break  # Only show one timeframe for brevity

# SENSEX Summary
with col2:
    st.markdown("**SENSEX**")

    # VOB Status
    if st.session_state.vob_data_sensex:
        from indicators.vob_strength_tracker import VOBStrengthTracker
        vob_tracker = VOBStrengthTracker()
        df_sensex = get_cached_chart_data('^BSESN', '1d', '1m')

        bullish_blocks = st.session_state.vob_data_sensex.get('bullish_blocks', [])
        bearish_blocks = st.session_state.vob_data_sensex.get('bearish_blocks', [])

        if bullish_blocks and df_sensex is not None:
            latest_bull = bullish_blocks[-1]
            bull_strength = vob_tracker.calculate_strength(latest_bull, df_sensex)
            trend_emoji = "🔺" if bull_strength['trend'] == "STRENGTHENING" else "🔻" if bull_strength['trend'] == "WEAKENING" else "➖"
            st.caption(f"**VOB Bull:** {bull_strength['strength_score']}/100 {trend_emoji} {bull_strength['trend']}")

        if bearish_blocks and df_sensex is not None:
            latest_bear = bearish_blocks[-1]
            bear_strength = vob_tracker.calculate_strength(latest_bear, df_sensex)
            trend_emoji = "🔺" if bear_strength['trend'] == "STRENGTHENING" else "🔻" if bear_strength['trend'] == "WEAKENING" else "➖"
            st.caption(f"**VOB Bear:** {bear_strength['strength_score']}/100 {trend_emoji} {bear_strength['trend']}")

    # HTF S/R Status
    if st.session_state.htf_data_sensex:
        from indicators.htf_sr_strength_tracker import HTFSRStrengthTracker
        htf_tracker = HTFSRStrengthTracker()
        df_sensex = get_cached_chart_data('^BSESN', '7d', '1m')

        # Get latest support and resistance levels
        # htf_data_sensex is a list, iterate directly
        for level_data in st.session_state.htf_data_sensex:
            if level_data and df_sensex is not None:
                timeframe = level_data.get('timeframe')
                support = level_data.get('pivot_low')
                resistance = level_data.get('pivot_high')

                if support:
                    support_strength = htf_tracker.calculate_strength(support, 'SUPPORT', df_sensex, lookback_periods=100)
                    trend_emoji = "🔺" if support_strength['trend'] == "STRENGTHENING" else "🔻" if support_strength['trend'] == "WEAKENING" else "➖"
                    st.caption(f"**HTF Support ({timeframe}):** {support_strength['strength_score']}/100 {trend_emoji} {support_strength['trend']}")

                if resistance:
                    resistance_strength = htf_tracker.calculate_strength(resistance, 'RESISTANCE', df_sensex, lookback_periods=100)
                    trend_emoji = "🔺" if resistance_strength['trend'] == "STRENGTHENING" else "🔻" if resistance_strength['trend'] == "WEAKENING" else "➖"
                    st.caption(f"**HTF Resistance ({timeframe}):** {resistance_strength['strength_score']}/100 {trend_emoji} {resistance_strength['trend']}")

                break  # Only show one timeframe for brevity

st.divider()

if st.session_state.active_htf_sr_signals:
    for signal in st.session_state.active_htf_sr_signals:
        signal_emoji = "🟢" if signal['direction'] == 'CALL' else "🔴"
        direction_label = "BULLISH" if signal['direction'] == 'CALL' else "BEARISH"
        sentiment_color = "#26ba9f" if signal['market_sentiment'] == 'BULLISH' else "#ba2646"

        # Format timeframe for display
        timeframe_display = {
            '5T': '5 Min',
            '10T': '10 Min',
            '15T': '15 Min'
        }.get(signal.get('timeframe', ''), signal.get('timeframe', 'N/A'))

        # Determine level type and value
        if signal['direction'] == 'CALL':
            level_type = "Support Level"
            level_value = signal['support_level']
        else:
            level_type = "Resistance Level"
            level_value = signal.get('resistance_level', 'N/A')

        # Get strength data if available
        strength = signal.get('strength')
        strength_html = ""
        if strength:
            strength_score = strength.get('strength_score', 0)
            strength_label = strength.get('strength_label', 'UNKNOWN')
            trend = strength.get('trend', 'UNKNOWN')
            times_tested = strength.get('times_tested', 0)
            hold_rate = strength.get('hold_rate', 0)

            # Determine strength color
            if strength_score >= 70:
                strength_color = "#4caf50"  # Green
            elif strength_score >= 50:
                strength_color = "#ffc107"  # Yellow
            else:
                strength_color = "#ff5252"  # Red

            # Trend indicator
            trend_emoji = "🔺" if trend == "STRENGTHENING" else "🔻" if trend == "WEAKENING" else "➖"

            strength_html = f"""
                <hr style='margin: 10px 0;'>
                <div style='background-color: rgba(0,0,0,0.05); padding: 10px; border-radius: 5px;'>
                    <p style='margin: 0; font-size: 14px; font-weight: bold;'>📊 Support/Resistance Strength Analysis</p>
                    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px;'>
                        <div>
                            <p style='margin: 0; font-size: 11px; color: #888;'>Strength Score</p>
                            <p style='margin: 0; font-size: 16px; font-weight: bold; color: {strength_color};'>{strength_score}/100</p>
                        </div>
                        <div>
                            <p style='margin: 0; font-size: 11px; color: #888;'>Status</p>
                            <p style='margin: 0; font-size: 14px;'>{strength_label.replace('_', ' ')}</p>
                        </div>
                        <div>
                            <p style='margin: 0; font-size: 11px; color: #888;'>Trend</p>
                            <p style='margin: 0; font-size: 14px;'>{trend_emoji} {trend}</p>
                        </div>
                        <div>
                            <p style='margin: 0; font-size: 11px; color: #888;'>Tests / Hold Rate</p>
                            <p style='margin: 0; font-size: 14px;'>{times_tested} / {hold_rate}%</p>
                        </div>
                    </div>
                </div>
            """

        with st.container():
            st.markdown(f"""
            <div style='border: 2px solid {sentiment_color}; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: rgba(38, 186, 159, 0.1);'>
                <h3 style='margin: 0;'>{signal_emoji} {signal['index']} {direction_label} HTF S/R ENTRY SIGNAL</h3>
                <p style='margin: 5px 0;'><b>Market Sentiment:</b> {signal['market_sentiment']} | <b>Timeframe:</b> {timeframe_display}</p>
                <hr style='margin: 10px 0;'>
                <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;'>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>Entry Price</p>
                        <p style='margin: 0; font-size: 18px; font-weight: bold;'>₹{signal['entry_price']}</p>
                    </div>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>Stop Loss</p>
                        <p style='margin: 0; font-size: 18px; font-weight: bold; color: #ff5252;'>₹{signal['stop_loss']}</p>
                    </div>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>Target</p>
                        <p style='margin: 0; font-size: 18px; font-weight: bold; color: #4caf50;'>₹{signal['target']}</p>
                    </div>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>Risk:Reward</p>
                        <p style='margin: 0; font-size: 18px; font-weight: bold;'>{signal['risk_reward']}</p>
                    </div>
                </div>
                <hr style='margin: 10px 0;'>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>{level_type}</p>
                        <p style='margin: 0; font-size: 14px;'>₹{level_value}</p>
                    </div>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>Distance from Level</p>
                        <p style='margin: 0; font-size: 14px;'>{signal['distance_from_level']} points</p>
                    </div>
                    <div>
                        <p style='margin: 0; font-size: 12px; color: #888;'>Signal Time</p>
                        <p style='margin: 0; font-size: 14px;'>{signal['timestamp'].strftime('%H:%M:%S')}</p>
                    </div>
                </div>
                {strength_html}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("⏳ Monitoring market for HTF S/R entry signals... No active signals at the moment.")
    st.caption("Signals are generated when spot price is within 8 points of HTF Support (for bullish) or Resistance (for bearish) and aligned with overall market sentiment.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# TABS - USING NATIVE STREAMLIT TABS FOR BETTER UX
# ═══════════════════════════════════════════════════════════════════════

# Native tabs - work seamlessly on mobile and desktop, no multiple clicks needed
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "🌟 Overall Market Sentiment",
    "🎯 Trade Setup",
    "📊 Active Signals",
    "📈 Positions",
    "🎲 Bias Analysis Pro",
    "🔍 Option Chain Analysis",
    "📉 Advanced Chart Analysis",
    "🎯 NIFTY Option Screener v7.0",
    "🌐 Enhanced Market Data",
    "🤖 MASTER AI ANALYSIS",
    "🔬 Advanced Analytics",
    "📜 Signal History & Performance"
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: OVERALL MARKET SENTIMENT
# ═══════════════════════════════════════════════════════════════════════

with tab1:
    render_overall_market_sentiment(NSE_INSTRUMENTS)

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: TRADE SETUP
# ═══════════════════════════════════════════════════════════════════════

with tab2:
    st.header("🎯 Create New Trade Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_index = st.selectbox(
            "Select Index",
            ["NIFTY", "SENSEX"],
            key="setup_index"
        )
    
    with col2:
        selected_direction = st.selectbox(
            "Select Direction",
            ["CALL", "PUT"],
            key="setup_direction"
        )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Safe default value handling for VOB support level
        try:
            default_support = max(0.0, float(nifty_data['spot_price']) - 50.0) if nifty_data.get('spot_price') and nifty_data['spot_price'] not in [None, 0, 'N/A'] else 25000.0
        except (TypeError, ValueError):
            default_support = 25000.0

        vob_support = st.number_input(
            "VOB Support Level",
            min_value=0.0,
            value=default_support,
            step=10.0,
            key="vob_support"
        )

    with col2:
        # Safe default value handling for VOB resistance level
        try:
            default_resistance = max(0.0, float(nifty_data['spot_price']) + 50.0) if nifty_data.get('spot_price') and nifty_data['spot_price'] not in [None, 0, 'N/A'] else 25100.0
        except (TypeError, ValueError):
            default_resistance = 25100.0

        vob_resistance = st.number_input(
            "VOB Resistance Level",
            min_value=0.0,
            value=default_resistance,
            step=10.0,
            key="vob_resistance"
        )
    
    st.divider()
    
    # Preview calculated levels
    st.subheader("📋 Preview Trade Levels")
    
    levels = calculate_levels(
        selected_index,
        selected_direction,
        vob_support,
        vob_resistance,
        STOP_LOSS_OFFSET
    )

    # Safe handling for spot price in strike calculation
    safe_spot_price = 25000.0  # Default fallback value
    if nifty_data.get('spot_price') and nifty_data['spot_price'] not in [None, 0, 'N/A']:
        try:
            safe_spot_price = float(nifty_data['spot_price'])
        except (TypeError, ValueError):
            safe_spot_price = 25000.0

    strike_info = calculate_strike(
        selected_index,
        safe_spot_price,
        selected_direction,
        nifty_data.get('current_expiry', 'N/A')
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Entry Level", f"{levels['entry_level']:.2f}")
    
    with col2:
        st.metric("Stop Loss", f"{levels['sl_level']:.2f}")
    
    with col3:
        st.metric("Target", f"{levels['target_level']:.2f}")
    
    with col4:
        st.metric("Risk:Reward", f"1:{levels['rr_ratio']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Strike:** {strike_info['strike']} {strike_info['option_type']} ({strike_info['strike_type']})")
    
    with col2:
        lot_size = LOT_SIZES[selected_index]
        st.info(f"**Quantity:** {lot_size} ({selected_index} lot size)")
    
    st.divider()
    
    # Create setup button
    if st.button("✅ Create Signal Setup", type="primary", use_container_width=True):
        signal_id = st.session_state.signal_manager.create_setup(
            selected_index,
            selected_direction,
            vob_support,
            vob_resistance
        )
        st.session_state.active_setup_id = signal_id
        st.success(f"✅ Signal setup created! ID: {signal_id[:20]}...")
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: ACTIVE SIGNALS
# ═══════════════════════════════════════════════════════════════════════

with tab3:
    st.header("📊 Active Signal Setups")

    active_setups = st.session_state.signal_manager.get_active_setups()

    if not active_setups:
        st.info("No active signal setups. Create one in the Trade Setup tab.")
    else:
        for signal_id, setup in active_setups.items():
            with st.container():
                st.subheader(f"{setup['index']} {setup['direction']}")

                # Signal count display
                signal_count = setup['signal_count']
                signals_display = "⭐" * signal_count + "☆" * (3 - signal_count)

                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**Signals:** {signals_display} ({signal_count}/3)")
                    st.write(f"**VOB Support:** {setup['vob_support']}")
                    st.write(f"**VOB Resistance:** {setup['vob_resistance']}")

                with col2:
                    if signal_count < 3:
                        if st.button(f"➕ Add Signal", key=f"add_{signal_id}"):
                            st.session_state.signal_manager.add_signal(signal_id)

                            # Check if ready and send Telegram
                            updated_setup = st.session_state.signal_manager.get_setup(signal_id)
                            if updated_setup['status'] == 'ready':
                                telegram = TelegramBot()
                                telegram.send_signal_ready(updated_setup)

                            st.rerun()

                    if signal_count > 0:
                        if st.button(f"➖ Remove Signal", key=f"remove_{signal_id}"):
                            st.session_state.signal_manager.remove_signal(signal_id)
                            st.rerun()

                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{signal_id}"):
                        st.session_state.signal_manager.delete_setup(signal_id)
                        st.rerun()

                # NEW FEATURE: Trade Execution Button with Index Selection
                if setup['status'] == 'ready' or signal_count >= 3:
                    st.markdown("---")
                    st.markdown("### 🚀 Execute Trade")

                    # Index Selection
                    trade_col1, trade_col2 = st.columns([1, 2])

                    with trade_col1:
                        selected_index = st.radio(
                            "Select Index to Trade:",
                            ["NIFTY", "SENSEX"],
                            key=f"index_select_{signal_id}",
                            horizontal=True
                        )

                    # Get data based on selected index
                    if selected_index == "NIFTY":
                        index_data = nifty_data
                    else:
                        index_data = get_cached_sensex_data()
                        if not index_data:
                            st.error("❌ SENSEX data not available")
                            continue

                    current_price = index_data['spot_price']

                    # Display current market status
                    with trade_col2:
                        st.metric(
                            label=f"{selected_index} Spot Price",
                            value=f"{current_price:.2f}",
                            delta=f"Support: {setup['vob_support']:.2f} | Resistance: {setup['vob_resistance']:.2f}"
                        )

                    # Calculate if signal conditions are met
                    if setup['direction'] == 'CALL':
                        vob_touched = check_vob_touch(current_price, setup['vob_support'], VOB_TOUCH_TOLERANCE)
                        vob_type = "Support"
                        vob_level = setup['vob_support']
                    else:
                        vob_touched = check_vob_touch(current_price, setup['vob_resistance'], VOB_TOUCH_TOLERANCE)
                        vob_type = "Resistance"
                        vob_level = setup['vob_resistance']

                    # Show signal status
                    if vob_touched:
                        st.success(f"✅ {vob_type} LEVEL TOUCHED! Ready to execute.")
                    else:
                        st.info(f"⏳ Waiting for {vob_type} touch | Current: {current_price:.2f} | Target: {vob_level:.2f} | Distance: {abs(current_price - vob_level):.2f} pts")

                    # Execute Button (always available when signal is ready)
                    exec_col1, exec_col2, exec_col3 = st.columns([1, 1, 1])

                    with exec_col2:
                        if st.button(
                            f"🚀 EXECUTE {selected_index} TRADE",
                            key=f"execute_{signal_id}",
                            type="primary",
                            use_container_width=True
                        ):
                            with st.spinner(f"Executing {selected_index} trade..."):
                                # Update setup with selected index
                                trade_setup = setup.copy()
                                trade_setup['index'] = selected_index

                                executor = TradeExecutor()
                                result = executor.execute_trade(
                                    trade_setup,
                                    current_price,
                                    index_data['current_expiry']
                                )

                                if result['success']:
                                    st.success(f"✅ {result['message']}")
                                    st.success(f"**Order ID:** {result['order_id']}")

                                    # Mark as executed
                                    st.session_state.signal_manager.mark_executed(signal_id, result['order_id'])

                                    # Store position in session state for monitoring
                                    if 'active_positions' not in st.session_state:
                                        st.session_state.active_positions = {}

                                    details = result['order_details']
                                    st.session_state.active_positions[result['order_id']] = {
                                        'order_id': result['order_id'],
                                        'index': selected_index,
                                        'direction': setup['direction'],
                                        'strike': details['strike'],
                                        'option_type': details['option_type'],
                                        'quantity': details['quantity'],
                                        'entry': details['entry_level'],
                                        'sl': details['sl_price'],
                                        'target': details['target_price'],
                                        'timestamp': get_current_time_ist().isoformat(),
                                        'status': 'active'
                                    }

                                    # Display order details
                                    st.write("**Order Details:**")
                                    st.write(f"- Index: {selected_index}")
                                    st.write(f"- Strike: {details['strike']} {details['option_type']} ({details['strike_type']})")
                                    st.write(f"- Quantity: {details['quantity']}")
                                    st.write(f"- Entry: {details['entry_level']:.2f}")
                                    st.write(f"- Stop Loss: {details['sl_price']:.2f} (Support/Resistance {'-' if setup['direction'] == 'CALL' else '+'} 8 points)")
                                    st.write(f"- Target: {details['target_price']:.2f} (Opposite level)")
                                    st.write(f"- R:R Ratio: 1:{details['rr_ratio']}")

                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result['message']}")
                                    if 'error' in result:
                                        st.error(f"Error: {result['error']}")

                st.divider()

# ═══════════════════════════════════════════════════════════════════════
# TAB 4: POSITIONS
# ═══════════════════════════════════════════════════════════════════════

with tab4:
    st.header("📈 Active Positions & Monitoring")

    # Initialize active_positions in session state if not exists
    if 'active_positions' not in st.session_state:
        st.session_state.active_positions = {}

    # Get current spot prices for monitoring - with safe null handling
    nifty_spot = 0
    if nifty_data.get('spot_price') and nifty_data['spot_price'] not in [None, 'N/A']:
        try:
            nifty_spot = float(nifty_data['spot_price'])
        except (TypeError, ValueError):
            nifty_spot = 0

    sensex_data_obj = get_cached_sensex_data()
    sensex_spot = 0
    if sensex_data_obj and sensex_data_obj.get('spot_price') and sensex_data_obj['spot_price'] not in [None, 'N/A']:
        try:
            sensex_spot = float(sensex_data_obj['spot_price'])
        except (TypeError, ValueError):
            sensex_spot = 0

    # Check for auto-exit conditions
    positions_to_exit = []
    for order_id, pos in st.session_state.active_positions.items():
        if pos['status'] != 'active':
            continue

        # Get current spot price based on index
        current_spot = nifty_spot if pos['index'] == 'NIFTY' else sensex_spot

        # Check if target or SL is hit
        if pos['direction'] == 'CALL':
            # For CALL: Target is resistance, SL is support - 8
            target_hit = current_spot >= pos['target']
            sl_hit = current_spot <= pos['sl']
        else:  # PUT
            # For PUT: Target is support, SL is resistance + 8
            target_hit = current_spot <= pos['target']
            sl_hit = current_spot >= pos['sl']

        if target_hit:
            st.success(f"🎯 TARGET HIT for {pos['index']} {pos['direction']} | Order ID: {order_id}")
            positions_to_exit.append((order_id, 'target'))
        elif sl_hit:
            st.error(f"🛑 STOP LOSS HIT for {pos['index']} {pos['direction']} | Order ID: {order_id}")
            positions_to_exit.append((order_id, 'stoploss'))

    # Display tracked positions
    if not st.session_state.active_positions:
        st.info("No active positions tracked. Execute a trade from the Active Signals tab.")
    else:
        active_count = sum(1 for p in st.session_state.active_positions.values() if p['status'] == 'active')
        st.info(f"📊 Tracking {active_count} active position(s)")

        for order_id, pos in st.session_state.active_positions.items():
            if pos['status'] != 'active':
                continue

            with st.container():
                # Position header
                st.subheader(f"{pos['index']} {pos['direction']} - {pos['option_type']}")

                # Get current spot price
                current_spot = nifty_spot if pos['index'] == 'NIFTY' else sensex_spot

                # Calculate distance to target and SL
                if pos['direction'] == 'CALL':
                    dist_to_target = pos['target'] - current_spot
                    dist_to_sl = current_spot - pos['sl']
                else:
                    dist_to_target = current_spot - pos['target']
                    dist_to_sl = pos['sl'] - current_spot

                # Create columns for display
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

                with col1:
                    st.metric(
                        label="Current Spot",
                        value=f"{current_spot:.2f}",
                        delta=f"Entry: {pos['entry']:.2f}"
                    )
                    st.write(f"**Strike:** {pos['strike']} {pos['option_type']}")
                    st.write(f"**Quantity:** {pos['quantity']}")

                with col2:
                    target_color = "green" if dist_to_target >= 0 else "orange"
                    st.metric(
                        label="🎯 Target",
                        value=f"{pos['target']:.2f}",
                        delta=f"{dist_to_target:+.2f} pts away"
                    )
                    if pos['direction'] == 'CALL':
                        st.caption("Trigger: When spot reaches resistance (VOB/HTF)")
                    else:
                        st.caption("Trigger: When spot reaches support (VOB/HTF)")

                with col3:
                    sl_color = "red" if dist_to_sl <= 0 else "green"
                    st.metric(
                        label="🛑 Stop Loss",
                        value=f"{pos['sl']:.2f}",
                        delta=f"{dist_to_sl:+.2f} pts away"
                    )
                    if pos['direction'] == 'CALL':
                        st.caption("Support - 8 points")
                    else:
                        st.caption("Resistance + 8 points")

                with col4:
                    st.write("")  # Spacer
                    st.write("")  # Spacer
                    if st.button("❌ Exit", key=f"exit_{order_id}", type="secondary"):
                        # Exit position
                        if DEMO_MODE:
                            st.session_state.active_positions[order_id]['status'] = 'closed'
                            st.success("✅ Position exited (DEMO MODE)")
                            time.sleep(1)
                            st.rerun()
                        else:
                            from dhan_api import DhanAPI
                            dhan = DhanAPI()
                            result = dhan.exit_position(order_id)
                            if result['success']:
                                st.session_state.active_positions[order_id]['status'] = 'closed'
                                st.success("✅ Position exited successfully!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ Exit failed: {result.get('error', 'Unknown error')}")

                # Progress bar for position
                entry_to_target = abs(pos['target'] - pos['entry'])
                current_progress = abs(current_spot - pos['entry'])
                progress = min(current_progress / entry_to_target, 1.0) if entry_to_target > 0 else 0

                st.progress(progress, text=f"Progress to Target: {progress*100:.1f}%")

                # Status indicators
                status_col1, status_col2 = st.columns(2)
                with status_col1:
                    if dist_to_target <= 5:
                        st.warning(f"⚠️ Near Target ({dist_to_target:.2f} pts)")
                with status_col2:
                    if dist_to_sl <= 5:
                        st.error(f"⚠️ Near Stop Loss ({dist_to_sl:.2f} pts)")

                st.divider()

    # Auto-exit positions
    if positions_to_exit:
        for order_id, exit_reason in positions_to_exit:
            st.info(f"Auto-exiting position {order_id} due to {exit_reason}...")

            if not DEMO_MODE:
                from dhan_api import DhanAPI
                dhan = DhanAPI()
                result = dhan.exit_position(order_id)

                if result['success']:
                    st.session_state.active_positions[order_id]['status'] = 'closed'
                    st.success(f"✅ Position auto-exited: {exit_reason}")
                else:
                    st.error(f"❌ Auto-exit failed: {result.get('error')}")
            else:
                st.session_state.active_positions[order_id]['status'] = 'closed'
                st.success(f"✅ Position auto-exited (DEMO): {exit_reason}")

        time.sleep(2)
        st.rerun()

    # Show API positions (if not DEMO_MODE)
    if not DEMO_MODE:
        st.markdown("---")
        st.subheader("📡 Live Positions from Dhan API")

        from dhan_api import DhanAPI
        try:
            dhan = DhanAPI()
            positions_result = dhan.get_positions()

            if positions_result['success']:
                positions = positions_result['positions']

                if not positions:
                    st.info("No live positions from API")
                else:
                    for idx, pos in enumerate(positions):
                        with st.container():
                            col1, col2, col3 = st.columns([3, 2, 1])

                            with col1:
                                st.write(f"**Symbol:** {pos.get('tradingSymbol', 'N/A')}")
                                st.write(f"**Quantity:** {pos.get('netQty', 0)}")

                            with col2:
                                pnl = pos.get('unrealizedProfit', 0)
                                pnl_color = "green" if pnl > 0 else "red"
                                st.markdown(f"**P&L:** <span style='color:{pnl_color}'>₹{pnl:,.2f}</span>", unsafe_allow_html=True)

                            with col3:
                                # Use index and trading symbol for unique key to avoid duplicate key errors
                                unique_key = f"exit_api_{idx}_{pos.get('tradingSymbol', 'pos')}"
                                if st.button("❌ Exit", key=unique_key):
                                    result = dhan.exit_position(pos.get('orderId'))
                                    if result['success']:
                                        st.success("Position exited!")
                                        st.rerun()
                                    else:
                                        st.error("Exit failed")

                            st.divider()
            else:
                st.error(f"Failed to fetch positions: {positions_result.get('error')}")

        except Exception as e:
            st.error(f"Error fetching API positions: {e}")
    else:
        st.info("🧪 DEMO MODE - Connect to Dhan API to see live positions")

# ═══════════════════════════════════════════════════════════════════════
# TAB 5: BIAS ANALYSIS PRO
# ═══════════════════════════════════════════════════════════════════════

with tab5:
    st.header("🎯 Comprehensive Bias Analysis Pro")
    st.caption("13 Bias Indicators with Adaptive Weighted Scoring | 🔄 Auto-refreshing every 60 seconds")

    # Auto-load cached results if not already in session state
    if not st.session_state.bias_analysis_results:
        cached_results = get_cached_bias_analysis_results()
        if cached_results:
            st.session_state.bias_analysis_results = cached_results

    # Analysis controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        bias_symbol = st.selectbox(
            "Select Market for Bias Analysis",
            ["^NSEI (NIFTY 50)", "^BSESN (SENSEX)", "^DJI (DOW JONES)"],
            key="bias_analysis_symbol"
        )
        symbol_code = bias_symbol.split()[0]

    with col2:
        if st.button("🔍 Analyze All Bias", type="primary", use_container_width=True):
            with st.spinner("Analyzing bias indicators..."):
                try:
                    bias_analyzer = get_bias_analyzer()
                    results = bias_analyzer.analyze_all_bias_indicators(symbol_code)
                    st.session_state.bias_analysis_results = results
                    # Update cache
                    cache_manager = get_cache_manager()
                    cache_manager.set('bias_analysis', results)
                    if results['success']:
                        st.success("✅ Bias analysis completed!")
                    else:
                        st.error(f"❌ Analysis failed: {results.get('error')}")
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")

    with col3:
        if st.session_state.bias_analysis_results:
            if st.button("🗑️ Clear Analysis", use_container_width=True):
                st.session_state.bias_analysis_results = None
                st.rerun()

    st.divider()

    # Display results if available
    if st.session_state.bias_analysis_results and st.session_state.bias_analysis_results.get('success'):
        results = st.session_state.bias_analysis_results

        # =====================================================================
        # OVERALL BIAS SUMMARY
        # =====================================================================
        st.subheader("📊 Overall Market Bias")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Current Price",
                f"₹{results['current_price']:,.2f}"
            )

        with col2:
            overall_bias = results['overall_bias']
            bias_emoji = "🐂" if overall_bias == "BULLISH" else "🐻" if overall_bias == "BEARISH" else "⚖️"
            bias_color = "green" if overall_bias == "BULLISH" else "red" if overall_bias == "BEARISH" else "gray"

            st.markdown(f"<h3 style='color:{bias_color};'>{bias_emoji} {overall_bias}</h3>",
                       unsafe_allow_html=True)
            st.caption("Overall Market Bias")

        with col3:
            score = results['overall_score']
            score_color = "green" if score > 0 else "red" if score < 0 else "gray"
            st.markdown(f"<h3 style='color:{score_color};'>{score:.1f}</h3>",
                       unsafe_allow_html=True)
            st.caption("Overall Score")

        with col4:
            confidence = results['overall_confidence']
            st.metric(
                "Confidence",
                f"{confidence:.1f}%"
            )

        with col5:
            st.metric(
                "Total Indicators",
                results['total_indicators']
            )

        # Bias distribution
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🐂 Bullish Signals", results['bullish_count'])

        with col2:
            st.metric("🐻 Bearish Signals", results['bearish_count'])

        with col3:
            st.metric("⚖️ Neutral Signals", results['neutral_count'])

        st.divider()

        # =====================================================================
        # DETAILED BIAS BREAKDOWN TABLE
        # =====================================================================
        st.subheader("📋 Detailed Bias Breakdown")

        # Convert bias results to DataFrame
        bias_df = pd.DataFrame(results['bias_results'])

        # Function to color code bias
        def color_bias(val):
            if 'BULLISH' in str(val):
                return 'background-color: #26a69a; color: white;'
            elif 'BEARISH' in str(val):
                return 'background-color: #ef5350; color: white;'
            else:
                return 'background-color: #78909c; color: white;'

        # Function to color code scores
        def color_score(val):
            try:
                score = float(val)
                if score > 50:
                    return 'background-color: #1b5e20; color: white; font-weight: bold;'
                elif score > 0:
                    return 'background-color: #4caf50; color: white;'
                elif score < -50:
                    return 'background-color: #b71c1c; color: white; font-weight: bold;'
                elif score < 0:
                    return 'background-color: #f44336; color: white;'
                else:
                    return 'background-color: #616161; color: white;'
            except:
                return ''

        # Create styled dataframe
        styled_df = bias_df.style.applymap(color_bias, subset=['bias']) \
                                 .applymap(color_score, subset=['score']) \
                                 .format({'score': '{:.2f}', 'weight': '{:.1f}'})

        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=600)

        st.divider()

        # =====================================================================
        # VISUAL SCORE REPRESENTATION
        # =====================================================================
        st.subheader("📊 Visual Bias Representation")

        # Create a chart showing each indicator's contribution
        chart_data = pd.DataFrame({
            'Indicator': [b['indicator'] for b in results['bias_results']],
            'Weighted Score': [b['score'] * b['weight'] for b in results['bias_results']]
        })

        # Sort by weighted score
        chart_data = chart_data.sort_values('Weighted Score', ascending=True)

        # Display bar chart
        st.bar_chart(chart_data.set_index('Indicator'))

        st.divider()

        # =====================================================================
        # BIAS CATEGORY BREAKDOWN
        # =====================================================================
        st.subheader("📈 Bias by Category")

        # Display mode info
        if 'mode' in results:
            mode_color = "🔄" if results['mode'] == "REVERSAL" else "📊"
            st.info(f"{mode_color} **Mode:** {results['mode']} | Fast: {results.get('fast_bull_pct', 0):.0f}% Bull | Slow: {results.get('slow_bull_pct', 0):.0f}% Bull")

        col1, col2, col3 = st.columns(3)

        # Optimize: Create DataFrame once and filter using vectorized operations
        all_bias_df = pd.DataFrame(results['bias_results'])

        if not all_bias_df.empty:
            # Pre-compute bullish/bearish flags for all rows
            is_bullish = all_bias_df['bias'].str.contains('BULLISH', na=False)
            is_bearish = all_bias_df['bias'].str.contains('BEARISH', na=False)

            with col1:
                st.markdown("**⚡ Fast Indicators (8)**")
                fast_df = all_bias_df[all_bias_df['category'] == 'fast']
                if not fast_df.empty:
                    fast_bull = is_bullish[fast_df.index].sum()
                    fast_bear = is_bearish[fast_df.index].sum()
                    fast_neutral = len(fast_df) - fast_bull - fast_bear

                    st.write(f"🐂 {fast_bull} | 🐻 {fast_bear} | ⚖️ {fast_neutral}")
                    st.dataframe(fast_df[['indicator', 'bias', 'score']],
                               use_container_width=True, hide_index=True)

            with col2:
                st.markdown("**📊 Medium Indicators (0)**")
                med_df = all_bias_df[all_bias_df['category'] == 'medium']
                if not med_df.empty:
                    med_bull = is_bullish[med_df.index].sum()
                    med_bear = is_bearish[med_df.index].sum()
                    med_neutral = len(med_df) - med_bull - med_bear

                    st.write(f"🐂 {med_bull} | 🐻 {med_bear} | ⚖️ {med_neutral}")
                    st.dataframe(med_df[['indicator', 'bias', 'score']],
                               use_container_width=True, hide_index=True)
                else:
                    st.info("No medium indicators configured")

            with col3:
                st.markdown("**🐢 Slow Indicators (0)**")
                slow_df = all_bias_df[all_bias_df['category'] == 'slow']
                if not slow_df.empty:
                    slow_bull = is_bullish[slow_df.index].sum()
                    slow_bear = is_bearish[slow_df.index].sum()
                    slow_neutral = len(slow_df) - slow_bull - slow_bear

                    st.write(f"🐂 {slow_bull} | 🐻 {slow_bear} | ⚖️ {slow_neutral}")
                    st.dataframe(slow_df[['indicator', 'bias', 'score']],
                               use_container_width=True, hide_index=True)
                else:
                    st.info("No slow indicators configured")

        st.divider()

        # =====================================================================
        # TRADING RECOMMENDATION
        # =====================================================================
        st.subheader("💡 Trading Recommendation")

        overall_bias = results['overall_bias']
        overall_score = results['overall_score']
        confidence = results['overall_confidence']

        if overall_bias == "BULLISH" and confidence > 70:
            st.success("### 🐂 STRONG BULLISH SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - ✅ Look for LONG entries on dips
            - ✅ Wait for support levels or VOB support touch
            - ✅ Set stop loss below recent swing low
            - ✅ Target: Risk-Reward ratio 1:2 or higher
            """)
        elif overall_bias == "BULLISH" and confidence >= 50:
            st.success("### 🐂 MODERATE BULLISH SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - ⚠️ Consider LONG entries with caution
            - ⚠️ Use tighter stop losses
            - ⚠️ Take partial profits at resistance levels
            - ⚠️ Monitor for trend confirmation
            """)
        elif overall_bias == "BEARISH" and confidence > 70:
            st.error("### 🐻 STRONG BEARISH SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - ✅ Look for SHORT entries on rallies
            - ✅ Wait for resistance levels or VOB resistance touch
            - ✅ Set stop loss above recent swing high
            - ✅ Target: Risk-Reward ratio 1:2 or higher
            """)
        elif overall_bias == "BEARISH" and confidence >= 50:
            st.error("### 🐻 MODERATE BEARISH SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - ⚠️ Consider SHORT entries with caution
            - ⚠️ Use tighter stop losses
            - ⚠️ Take partial profits at support levels
            - ⚠️ Monitor for trend reversal
            """)
        else:
            st.warning("### ⚖️ NEUTRAL / NO CLEAR SIGNAL")
            st.info("""
            **Recommended Strategy:**
            - 🔄 Stay out of the market or use range trading
            - 🔄 Wait for clearer bias formation
            - 🔄 Monitor key support/resistance levels
            - 🔄 Reduce position sizes if trading
            """)

        # Key levels for entry
        st.divider()
        st.subheader("🎯 Key Considerations")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Bullish Signals Count**")
            st.markdown(f"<h2 style='color:green;'>{results['bullish_count']}/{results['total_indicators']}</h2>",
                       unsafe_allow_html=True)

        with col2:
            st.markdown("**Bearish Signals Count**")
            st.markdown(f"<h2 style='color:red;'>{results['bearish_count']}/{results['total_indicators']}</h2>",
                       unsafe_allow_html=True)

        with col3:
            st.markdown("**Confidence Level**")
            confidence_color = "green" if confidence > 70 else "orange" if confidence > 50 else "red"
            st.markdown(f"<h2 style='color:{confidence_color};'>{confidence:.1f}%</h2>",
                       unsafe_allow_html=True)

        # Timestamp
        st.caption(f"Analysis Time: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    else:
        st.info("👆 Click 'Analyze All Bias' button to start comprehensive bias analysis")

        st.markdown("""
        ### About Bias Analysis Pro

        This comprehensive bias analyzer evaluates **13 bias indicators** matching Pine Script EXACTLY:

        #### ⚡ Fast Indicators (8)
        - **Volume Delta** (Up Vol - Down Vol)
        - **HVP** (High Volume Pivots)
        - **VOB** (Volume Order Blocks)
        - **Order Blocks** (EMA 5/18 Crossover)
        - **RSI** (Relative Strength Index)
        - **DMI** (Directional Movement Index)
        - **VIDYA** (Variable Index Dynamic Average)
        - **MFI** (Money Flow Index)

        #### 📊 Medium Indicators (2)
        - **Close vs VWAP** (Price above/below VWAP)
        - **Price vs VWAP** (Position relative to VWAP)

        #### 🐢 Slow Indicators (3)
        - **Weighted Stocks (Daily)** (Top 9 NSE stocks)
        - **Weighted Stocks (15m)** (Intraday trend)
        - **Weighted Stocks (1h)** (Higher timeframe trend)

        #### 🎯 Adaptive Scoring System
        - **Normal Mode:** Fast (2x), Medium (3x), Slow (5x) weights
        - **Reversal Mode:** Fast (5x), Medium (3x), Slow (2x) weights
        - Mode switches automatically when divergence detected
        - Scores range from **-100 (Strong Bearish)** to **+100 (Strong Bullish)**
        - Overall bias requires **60%+ strength** for directional bias

        #### ✅ How to Use
        1. Select the market (NIFTY, SENSEX, or DOW)
        2. Click "Analyze All Bias" button
        3. Review comprehensive bias breakdown by category
        4. Check for REVERSAL mode warnings
        5. Use signals to inform your trading decisions

        **Note:** This tool is converted from the Pine Script "Smart Trading Dashboard - Adaptive + VOB" indicator with EXACT matching logic.
        """)

# ═══════════════════════════════════════════════════════════════════════
# TAB 6: OPTION CHAIN ANALYSIS (Nifty Option Screener v7.0)
# ═══════════════════════════════════════════════════════════════════════

with tab6:
    st.header("🔍 Option Chain Analysis")
    st.info("📌 The NIFTY Option Screener v7.0 has been moved to **Tab 8** for better organization.")
    st.markdown("""
    ### Available in Tab 8:
    - 🎯 NIFTY Option Screener v7.0
    - 100% SELLER'S PERSPECTIVE
    - ATM BIAS ANALYZER
    - MOMENT DETECTOR
    - EXPIRY SPIKE DETECTOR
    - ENHANCED OI/PCR ANALYTICS
    """)

# ═══════════════════════════════════════════════════════════════════════
# TAB 7: ADVANCED CHART ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

with tab7:
    st.header("📈 Advanced Chart Analysis")
    st.caption("TradingView-style Chart with Advanced Indicators: Volume Bars, Volume Order Blocks, HTF Support/Resistance (3min, 5min, 10min, 15min levels), Volume Footprint (1D timeframe, 10 bins, Dynamic POC), Ultimate RSI, OM Indicator (Order Flow & Momentum), Advanced Price Action (BOS, CHOCH, Fibonacci, Geometric Patterns)")

    # Chart controls
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

    with col1:
        chart_symbol = st.selectbox(
            "Select Market",
            ["^NSEI (NIFTY 50)", "^NSEBANK (BANK NIFTY)", "^BSESN (SENSEX)", "^DJI (DOW JONES)"],
            key="chart_symbol"
        )
        symbol_code = chart_symbol.split()[0]

    with col2:
        chart_period = st.selectbox(
            "Period",
            ["1d", "5d", "1mo"],
            index=0,
            key="chart_period"
        )

    with col3:
        chart_interval = st.selectbox(
            "Interval",
            ["1m", "5m", "15m", "1h"],
            index=0,
            key="chart_interval"
        )

    with col4:
        chart_auto_refresh = st.selectbox(
            "Auto Refresh",
            ["Off", "30s", "60s", "2m", "5m"],
            index=2,
            key="chart_auto_refresh",
            help="Automatically refresh chart at selected interval"
        )

    with col5:
        if st.button("🔄 Refresh", type="primary", use_container_width=True, key="manual_refresh_chart"):
            st.session_state.chart_needs_refresh = True

    st.divider()

    # Initialize chart session state variables
    if 'chart_data' not in st.session_state:
        st.session_state.chart_data = None
    if 'chart_needs_refresh' not in st.session_state:
        st.session_state.chart_needs_refresh = True
    if 'last_chart_params' not in st.session_state:
        st.session_state.last_chart_params = None
    if 'last_chart_update' not in st.session_state:
        st.session_state.last_chart_update = None

    # Check if chart parameters changed
    current_params = (symbol_code, chart_period, chart_interval)
    if st.session_state.last_chart_params != current_params:
        st.session_state.chart_needs_refresh = True
        st.session_state.last_chart_params = current_params

    # Handle auto-refresh timing
    should_auto_refresh = False
    if chart_auto_refresh != "Off" and st.session_state.last_chart_update is not None:
        # Convert refresh interval to seconds
        refresh_seconds = {
            "30s": 30,
            "60s": 60,
            "2m": 120,
            "5m": 300
        }.get(chart_auto_refresh, 60)

        # Check if enough time has passed
        from datetime import timedelta
        time_since_update = (get_current_time_ist() - st.session_state.last_chart_update).total_seconds()
        if time_since_update >= refresh_seconds:
            should_auto_refresh = True
            st.session_state.chart_needs_refresh = True

    # Auto-load chart data on first load or when refresh is needed
    if st.session_state.chart_needs_refresh:
        with st.spinner("Loading chart data and calculating indicators..."):
            try:
                # Fetch data using cached function (60s cache)
                df = get_cached_chart_data(symbol_code, chart_period, chart_interval)

                if df is not None and len(df) > 0:
                    st.session_state.chart_data = df
                    st.session_state.last_chart_update = get_current_time_ist()
                    st.success(f"✅ Loaded {len(df)} candles | Last updated (IST): {st.session_state.last_chart_update.strftime('%H:%M:%S %Z')}")
                else:
                    st.error("❌ Failed to fetch data. Try a different period or interval.")
                    st.session_state.chart_data = None

            except Exception as e:
                st.error(f"❌ Error loading chart: {e}")
                st.session_state.chart_data = None

        st.session_state.chart_needs_refresh = False

    # Show auto-refresh countdown
    if chart_auto_refresh != "Off" and st.session_state.chart_data is not None and st.session_state.last_chart_update is not None:
        refresh_seconds = {
            "30s": 30,
            "60s": 60,
            "2m": 120,
            "5m": 300
        }.get(chart_auto_refresh, 60)

        time_since_update = (get_current_time_ist() - st.session_state.last_chart_update).total_seconds()
        time_until_refresh = max(0, refresh_seconds - time_since_update)

        if time_until_refresh <= 0:
            # Time for refresh
            st.session_state.chart_needs_refresh = True
            st.rerun()
        else:
            st.info(f"⏱️ Next auto-refresh in {int(time_until_refresh)} seconds (auto-refresh enabled)")

    st.divider()

    # Indicator toggles
    st.subheader("🔧 Indicator Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        show_vob = st.checkbox("📦 Volume Order Blocks", value=True, key="show_vob")
        show_htf_sr = st.checkbox("📊 HTF Support/Resistance", value=True, key="show_htf_sr")

    with col2:
        show_footprint = st.checkbox("👣 Volume Footprint", value=True, key="show_footprint")
        show_rsi = st.checkbox("📈 Ultimate RSI", value=True, key="show_rsi")

    with col3:
        show_volume = st.checkbox("📊 Volume Bars", value=True, key="show_volume")
        show_om = st.checkbox("🎯 OM Indicator", value=True, key="show_om")

    # Volume Profile Indicators
    st.markdown("**📊 Volume Profile Indicators**")
    col1, col2, col3 = st.columns(3)

    with col1:
        show_liquidity_profile = st.checkbox("💧 Liquidity Sentiment Profile", value=False, key="show_liquidity_profile")

    with col2:
        show_money_flow_profile = st.checkbox("💰 Money Flow Profile", value=True, key="show_money_flow_profile")

    with col3:
        show_deltaflow_profile = st.checkbox("⚡ DeltaFlow Profile", value=True, key="show_deltaflow_profile")

    # Advanced Price Action Indicators
    st.markdown("**🎯 Advanced Price Action**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_bos = st.checkbox("🔺 BOS (Break of Structure)", value=True, key="show_bos")

    with col2:
        show_choch = st.checkbox("🔄 CHOCH (Change of Character)", value=True, key="show_choch")

    with col3:
        show_fibonacci = st.checkbox("📐 Fibonacci Levels", value=True, key="show_fibonacci")

    with col4:
        show_patterns = st.checkbox("📊 Geometric Patterns", value=True, key="show_patterns")

    st.divider()

    # Indicator Configuration Section
    st.subheader("⚙️ Indicator Configuration")
    st.caption("Configure each indicator's parameters below")

    # Volume Order Blocks Settings
    if show_vob:
        with st.expander("📦 Volume Order Blocks Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                vob_sensitivity = st.slider(
                    "Sensitivity",
                    min_value=3,
                    max_value=15,
                    value=5,
                    step=1,
                    help="Detection sensitivity for order blocks",
                    key="vob_sensitivity"
                )
            with col2:
                vob_mid_line = st.checkbox("Show Mid Line", value=True, key="vob_mid_line")
                vob_trend_shadow = st.checkbox("Show Trend Shadow", value=True, key="vob_trend_shadow")

    # HTF Support/Resistance Settings
    if show_htf_sr:
        with st.expander("📊 HTF Support/Resistance Settings", expanded=False):
            st.markdown("**Timeframe Configuration**")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                htf_3min_enabled = st.checkbox("3min", value=True, key="htf_3min_enabled")
                htf_3min_length = st.number_input("3min Length", min_value=2, max_value=10, value=4, step=1, key="htf_3min_length")

            with col2:
                htf_5min_enabled = st.checkbox("5min", value=True, key="htf_5min_enabled")
                htf_5min_length = st.number_input("5min Length", min_value=2, max_value=10, value=5, step=1, key="htf_5min_length")

            with col3:
                htf_10min_enabled = st.checkbox("10min", value=True, key="htf_10min_enabled")
                htf_10min_length = st.number_input("10min Length", min_value=2, max_value=10, value=5, step=1, key="htf_10min_length")

            with col4:
                htf_15min_enabled = st.checkbox("15min", value=True, key="htf_15min_enabled")
                htf_15min_length = st.number_input("15min Length", min_value=2, max_value=10, value=5, step=1, key="htf_15min_length")

    # Volume Footprint Settings
    if show_footprint:
        with st.expander("👣 Volume Footprint Settings", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                footprint_bins = st.slider(
                    "Number of Bins",
                    min_value=5,
                    max_value=30,
                    value=10,
                    step=1,
                    help="Number of price bins for volume distribution",
                    key="footprint_bins"
                )
                footprint_timeframe = st.selectbox(
                    "Timeframe",
                    options=["D", "W", "2W", "M"],
                    index=0,
                    help="Higher timeframe for footprint analysis",
                    key="footprint_timeframe"
                )

            with col2:
                footprint_dynamic_poc = st.checkbox(
                    "Dynamic POC",
                    value=True,
                    help="Show dynamic Point of Control",
                    key="footprint_dynamic_poc"
                )

    # Ultimate RSI Settings
    if show_rsi:
        with st.expander("📈 Ultimate RSI Settings", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                rsi_length = st.slider(
                    "RSI Length",
                    min_value=5,
                    max_value=30,
                    value=14,
                    step=1,
                    help="RSI calculation period",
                    key="rsi_length"
                )
                rsi_method = st.selectbox(
                    "Calculation Method",
                    options=["RMA", "EMA", "SMA", "TMA"],
                    index=0,
                    help="Moving average method for RSI",
                    key="rsi_method"
                )

            with col2:
                rsi_smooth = st.slider(
                    "Signal Smoothing",
                    min_value=5,
                    max_value=30,
                    value=14,
                    step=1,
                    help="Signal line smoothing period",
                    key="rsi_smooth"
                )
                rsi_signal_method = st.selectbox(
                    "Signal Method",
                    options=["EMA", "SMA", "RMA", "TMA"],
                    index=0,
                    help="Signal line calculation method",
                    key="rsi_signal_method"
                )

            with col3:
                rsi_ob_level = st.slider(
                    "Overbought Level",
                    min_value=70,
                    max_value=90,
                    value=80,
                    step=5,
                    help="Overbought threshold",
                    key="rsi_ob_level"
                )
                rsi_os_level = st.slider(
                    "Oversold Level",
                    min_value=10,
                    max_value=30,
                    value=20,
                    step=5,
                    help="Oversold threshold",
                    key="rsi_os_level"
                )

    # Liquidity Sentiment Profile Settings
    if show_liquidity_profile:
        with st.expander("💧 Liquidity Sentiment Profile Settings", expanded=False):
            st.markdown("**Profile Configuration**")
            col1, col2, col3 = st.columns(3)

            with col1:
                lsp_anchor_period = st.selectbox(
                    "Anchor Period",
                    options=["Auto", "Session", "Day", "Week", "Month", "Quarter", "Year"],
                    index=0,
                    key="lsp_anchor_period"
                )
                lsp_num_rows = st.slider(
                    "Number of Rows",
                    min_value=10,
                    max_value=100,
                    value=25,
                    step=5,
                    key="lsp_num_rows"
                )

            with col2:
                lsp_profile_width = st.slider(
                    "Profile Width %",
                    min_value=10,
                    max_value=50,
                    value=50,
                    step=5,
                    key="lsp_profile_width"
                ) / 100.0
                lsp_show_liquidity = st.checkbox("Show Liquidity Profile", value=True, key="lsp_show_liquidity")
                lsp_show_sentiment = st.checkbox("Show Sentiment Profile", value=True, key="lsp_show_sentiment")

            with col3:
                lsp_show_poc = st.checkbox("Show Level of Significance", value=False, key="lsp_show_poc")
                lsp_show_price_levels = st.checkbox("Show Price Levels", value=False, key="lsp_show_price_levels")
                lsp_show_range_bg = st.checkbox("Show Range Background", value=True, key="lsp_show_range_bg")

            st.markdown("**Volume Thresholds**")
            col1, col2 = st.columns(2)

            with col1:
                lsp_hv_threshold = st.slider(
                    "High Volume Threshold %",
                    min_value=50,
                    max_value=99,
                    value=73,
                    step=1,
                    key="lsp_hv_threshold"
                ) / 100.0

            with col2:
                lsp_lv_threshold = st.slider(
                    "Low Volume Threshold %",
                    min_value=10,
                    max_value=40,
                    value=21,
                    step=1,
                    key="lsp_lv_threshold"
                ) / 100.0

    # OM Indicator Settings
    if show_om:
        with st.expander("🎯 OM Indicator Settings", expanded=False):
            st.markdown("**Volume Order Blocks (VOB) Module**")
            col1, col2 = st.columns(2)
            with col1:
                om_vob_sensitivity = st.slider(
                    "VOB Sensitivity",
                    min_value=3,
                    max_value=15,
                    value=5,
                    step=1,
                    key="om_vob_sensitivity"
                )

            st.markdown("**High Volume Pivots (HVP) Module**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                om_show_hvp = st.checkbox("Show HVP", value=True, key="om_show_hvp")
            with col2:
                om_hvp_left_bars = st.number_input("Left Bars", min_value=5, max_value=30, value=15, step=1, key="om_hvp_left_bars")
            with col3:
                om_hvp_right_bars = st.number_input("Right Bars", min_value=5, max_value=30, value=15, step=1, key="om_hvp_right_bars")
            with col4:
                om_hvp_volume_filter = st.number_input("Volume Filter", min_value=1.0, max_value=5.0, value=2.0, step=0.5, key="om_hvp_volume_filter")

            st.markdown("**Delta Module**")
            col1, col2 = st.columns(2)
            with col1:
                om_delta_length = st.slider("Delta Length", min_value=5, max_value=30, value=10, step=1, key="om_delta_length")
            with col2:
                om_delta_threshold = st.slider("Delta Threshold", min_value=0.5, max_value=5.0, value=1.5, step=0.5, key="om_delta_threshold")

            st.markdown("**VIDYA Module**")
            col1, col2, col3 = st.columns(3)
            with col1:
                om_vidya_length = st.slider("VIDYA Length", min_value=5, max_value=30, value=10, step=1, key="om_vidya_length")
            with col2:
                om_vidya_momentum = st.slider("Momentum Period", min_value=10, max_value=40, value=20, step=5, key="om_vidya_momentum")
            with col3:
                om_band_distance = st.slider("Band Distance", min_value=1.0, max_value=5.0, value=2.0, step=0.5, key="om_band_distance")

    # Money Flow Profile Settings
    if show_money_flow_profile:
        with st.expander("💰 Money Flow Profile Settings", expanded=False):
            st.markdown("**Profile Configuration**")
            col1, col2, col3 = st.columns(3)

            with col1:
                mfp_lookback = st.slider(
                    "Lookback Length",
                    min_value=50,
                    max_value=500,
                    value=200,
                    step=50,
                    help="Lookback period for profile calculation",
                    key="mfp_lookback"
                )
                mfp_num_rows = st.slider(
                    "Number of Rows",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    help="Number of price bins (default: 10)",
                    key="mfp_num_rows"
                )

            with col2:
                mfp_source = st.selectbox(
                    "Profile Source",
                    options=["Volume", "Money Flow"],
                    index=0,
                    help="Volume or Money Flow (volume × price)",
                    key="mfp_source"
                )
                mfp_sentiment_method = st.selectbox(
                    "Sentiment Method",
                    options=["Bar Polarity", "Bar Buying/Selling Pressure"],
                    index=0,
                    help="Method to determine bullish vs bearish bars",
                    key="mfp_sentiment_method"
                )

            with col3:
                mfp_show_poc = st.selectbox(
                    "POC Display",
                    options=["Last(Zone)", "Last(Line)", "Developing", "None"],
                    index=0,
                    help="Point of Control display mode",
                    key="mfp_show_poc"
                )
                mfp_show_consolidation = st.checkbox(
                    "Show Consolidation Zones",
                    value=True,
                    help="Highlight high volume consolidation areas",
                    key="mfp_show_consolidation"
                )

            st.markdown("**Volume Thresholds**")
            col1, col2, col3 = st.columns(3)

            with col1:
                mfp_hv_threshold = st.slider(
                    "High Volume %",
                    min_value=50,
                    max_value=99,
                    value=53,
                    step=1,
                    key="mfp_hv_threshold"
                ) / 100.0

            with col2:
                mfp_lv_threshold = st.slider(
                    "Low Volume %",
                    min_value=10,
                    max_value=40,
                    value=37,
                    step=1,
                    key="mfp_lv_threshold"
                ) / 100.0

            with col3:
                mfp_consolidation_threshold = st.slider(
                    "Consolidation %",
                    min_value=0,
                    max_value=100,
                    value=25,
                    step=5,
                    key="mfp_consolidation_threshold"
                ) / 100.0

    # DeltaFlow Profile Settings
    if show_deltaflow_profile:
        with st.expander("⚡ DeltaFlow Profile Settings", expanded=False):
            st.markdown("**Profile Configuration**")
            col1, col2, col3 = st.columns(3)

            with col1:
                dfp_lookback = st.slider(
                    "Lookback Length",
                    min_value=50,
                    max_value=500,
                    value=200,
                    step=50,
                    help="Lookback period for profile calculation",
                    key="dfp_lookback"
                )
                dfp_bins = st.slider(
                    "Number of Bins",
                    min_value=10,
                    max_value=100,
                    value=30,
                    step=10,
                    help="Number of price bins for delta analysis",
                    key="dfp_bins"
                )

            with col2:
                dfp_show_poc = st.checkbox(
                    "Show POC Line",
                    value=True,
                    help="Display Point of Control line",
                    key="dfp_show_poc"
                )
                dfp_show_delta_heatmap = st.checkbox(
                    "Show Delta Heatmap",
                    value=True,
                    help="Color bins by delta strength",
                    key="dfp_show_delta_heatmap"
                )

            with col3:
                dfp_show_delta_display = st.checkbox(
                    "Show Delta Labels",
                    value=True,
                    help="Display delta percentage per level",
                    key="dfp_show_delta_display"
                )
                dfp_show_volume_bars = st.checkbox(
                    "Show Volume Bars",
                    value=True,
                    help="Display buy/sell volume bars per bin",
                    key="dfp_show_volume_bars"
                )

    st.divider()

    # Display chart if data is available
    if st.session_state.chart_data is not None:
        try:
            with st.spinner("Rendering chart with indicators..."):
                # Prepare indicator parameters
                vob_params = {
                    'sensitivity': vob_sensitivity if show_vob else 5,
                    'mid_line': vob_mid_line if show_vob else True,
                    'trend_shadow': vob_trend_shadow if show_vob else True
                } if show_vob else None

                htf_params = {
                    'levels_config': []
                }
                if show_htf_sr:
                    if htf_3min_enabled:
                        htf_params['levels_config'].append({'timeframe': '3T', 'length': htf_3min_length, 'style': 'Solid', 'color': '#26a69a'})
                    if htf_5min_enabled:
                        htf_params['levels_config'].append({'timeframe': '5T', 'length': htf_5min_length, 'style': 'Solid', 'color': '#2196f3'})
                    if htf_10min_enabled:
                        htf_params['levels_config'].append({'timeframe': '10T', 'length': htf_10min_length, 'style': 'Solid', 'color': '#9c27b0'})
                    if htf_15min_enabled:
                        htf_params['levels_config'].append({'timeframe': '15T', 'length': htf_15min_length, 'style': 'Solid', 'color': '#ff9800'})

                footprint_params = {
                    'bins': footprint_bins if show_footprint else 10,
                    'timeframe': footprint_timeframe if show_footprint else 'D',
                    'dynamic_poc': footprint_dynamic_poc if show_footprint else True
                } if show_footprint else None

                rsi_params = {
                    'length': rsi_length if show_rsi else 14,
                    'smooth': rsi_smooth if show_rsi else 14,
                    'method': rsi_method if show_rsi else 'RMA',
                    'signal_method': rsi_signal_method if show_rsi else 'EMA',
                    'ob_level': rsi_ob_level if show_rsi else 80,
                    'os_level': rsi_os_level if show_rsi else 20
                } if show_rsi else None

                om_params = {
                    'vob_sensitivity': om_vob_sensitivity if show_om else 5,
                    'hvp_left_bars': om_hvp_left_bars if show_om else 15,
                    'hvp_right_bars': om_hvp_right_bars if show_om else 15,
                    'hvp_volume_filter': om_hvp_volume_filter if show_om else 2.0,
                    'delta_length': om_delta_length if show_om else 10,
                    'delta_threshold': om_delta_threshold if show_om else 1.5,
                    'vidya_length': om_vidya_length if show_om else 10,
                    'vidya_momentum': om_vidya_momentum if show_om else 20,
                    'band_distance': om_band_distance if show_om else 2.0,
                    'show_hvp': om_show_hvp if show_om else True
                } if show_om else None

                liquidity_params = {
                    'anchor_period': lsp_anchor_period if show_liquidity_profile else 'Auto',
                    'num_rows': lsp_num_rows if show_liquidity_profile else 25,
                    'profile_width': lsp_profile_width if show_liquidity_profile else 0.50,
                    'show_liquidity_profile': lsp_show_liquidity if show_liquidity_profile else True,
                    'show_sentiment_profile': lsp_show_sentiment if show_liquidity_profile else True,
                    'show_poc': lsp_show_poc if show_liquidity_profile else False,
                    'show_price_levels': lsp_show_price_levels if show_liquidity_profile else False,
                    'show_range_bg': lsp_show_range_bg if show_liquidity_profile else True,
                    'hv_threshold': lsp_hv_threshold if show_liquidity_profile else 0.73,
                    'lv_threshold': lsp_lv_threshold if show_liquidity_profile else 0.21
                } if show_liquidity_profile else None

                money_flow_params = {
                    'lookback': mfp_lookback if show_money_flow_profile else 200,
                    'num_rows': mfp_num_rows if show_money_flow_profile else 10,
                    'profile_source': mfp_source if show_money_flow_profile else 'Volume',
                    'sentiment_method': mfp_sentiment_method if show_money_flow_profile else 'Bar Polarity',
                    'show_poc': mfp_show_poc if show_money_flow_profile else 'Last(Zone)',
                    'show_consolidation': mfp_show_consolidation if show_money_flow_profile else True,
                    'hv_threshold': mfp_hv_threshold if show_money_flow_profile else 0.53,
                    'lv_threshold': mfp_lv_threshold if show_money_flow_profile else 0.37,
                    'consolidation_threshold': mfp_consolidation_threshold if show_money_flow_profile else 0.25
                } if show_money_flow_profile else None

                deltaflow_params = {
                    'lookback': dfp_lookback if show_deltaflow_profile else 200,
                    'bins': dfp_bins if show_deltaflow_profile else 30,
                    'show_poc': dfp_show_poc if show_deltaflow_profile else True,
                    'show_delta_heatmap': dfp_show_delta_heatmap if show_deltaflow_profile else True,
                    'show_delta_display': dfp_show_delta_display if show_deltaflow_profile else True,
                    'show_volume_bars': dfp_show_volume_bars if show_deltaflow_profile else True
                } if show_deltaflow_profile else None

                # Create chart with selected indicators
                chart_analyzer = get_advanced_chart_analyzer()
                fig = chart_analyzer.create_advanced_chart(
                    st.session_state.chart_data,
                    symbol_code,
                    show_vob=show_vob,
                    show_htf_sr=show_htf_sr,
                    show_footprint=show_footprint,
                    show_rsi=show_rsi,
                    show_om=show_om,
                    show_volume=show_volume,
                    show_liquidity_profile=show_liquidity_profile,
                    show_money_flow_profile=show_money_flow_profile,
                    show_deltaflow_profile=show_deltaflow_profile,
                    show_bos=show_bos,
                    show_choch=show_choch,
                    show_fibonacci=show_fibonacci,
                    show_patterns=show_patterns,
                    vob_params=vob_params,
                    htf_params=htf_params,
                    footprint_params=footprint_params,
                    rsi_params=rsi_params,
                    om_params=om_params,
                    liquidity_params=liquidity_params,
                    money_flow_params=money_flow_params,
                    deltaflow_params=deltaflow_params
                )

                # Display chart
                st.plotly_chart(fig, use_container_width=True)

                # Chart statistics
                st.subheader("📊 Chart Statistics")

                col1, col2, col3, col4, col5 = st.columns(5)

                df_stats = st.session_state.chart_data

                with col1:
                    st.metric("Current Price", f"₹{df_stats['close'].iloc[-1]:,.2f}")

                with col2:
                    price_change = df_stats['close'].iloc[-1] - df_stats['close'].iloc[0]
                    price_change_pct = (price_change / df_stats['close'].iloc[0]) * 100
                    st.metric("Change", f"₹{price_change:,.2f}", delta=f"{price_change_pct:.2f}%")

                with col3:
                    st.metric("High", f"₹{df_stats['high'].max():,.2f}")

                with col4:
                    st.metric("Low", f"₹{df_stats['low'].min():,.2f}")

                with col5:
                    avg_volume = df_stats['volume'].mean()
                    st.metric("Avg Volume", f"{avg_volume:,.0f}")

                # Indicator Data Tables
                st.divider()
                st.subheader("📊 Indicator Data Tables")
                st.caption("Detailed data from all enabled indicators")

                # Create tabs for each indicator category
                indicator_tabs = []
                # Always show Market Regime tab
                indicator_tabs.append("🎯 Market Regime")
                if show_vob:
                    indicator_tabs.append("📦 Volume Order Blocks")
                if show_htf_sr:
                    indicator_tabs.append("📊 HTF Support/Resistance")
                if show_footprint:
                    indicator_tabs.append("👣 Volume Footprint")
                if show_rsi:
                    indicator_tabs.append("📈 Ultimate RSI")
                if show_om:
                    indicator_tabs.append("🎯 OM Indicator")
                if show_liquidity_profile:
                    indicator_tabs.append("💧 Liquidity Profile")
                if show_money_flow_profile:
                    indicator_tabs.append("💰 Money Flow Profile")
                if show_deltaflow_profile:
                    indicator_tabs.append("⚡ DeltaFlow Profile")
                if show_bos or show_choch or show_fibonacci or show_patterns:
                    indicator_tabs.append("🎯 Price Action")

                if indicator_tabs:
                    tabs = st.tabs(indicator_tabs)
                    tab_idx = 0

                    # Market Regime Dashboard (Always first tab)
                    with tabs[tab_idx]:
                        from ml.market_regime_detector import MarketRegimeDetector

                        st.markdown("### 🎯 Market Regime Analysis")
                        st.caption("AI-powered market regime detection using all indicators")

                        # Collect indicator data
                        regime_indicator_data = {}

                        # BOS data
                        if show_bos:
                            from indicators.advanced_price_action import AdvancedPriceAction
                            pa_indicator = AdvancedPriceAction()
                            regime_indicator_data['bos'] = pa_indicator.detect_bos(df_stats)
                            regime_indicator_data['choch'] = pa_indicator.detect_choch(df_stats)
                        else:
                            regime_indicator_data['bos'] = []
                            regime_indicator_data['choch'] = []

                        # HTF SR data
                        if show_htf_sr and htf_params:
                            from indicators.htf_support_resistance import HTFSupportResistance
                            htf_levels_for_regime = []
                            for level_config in htf_params.get('levels_config', []):
                                htf_indicator = HTFSupportResistance(
                                    timeframes=[level_config['timeframe']],
                                    pivot_length=level_config['length']
                                )
                                levels = htf_indicator.calculate_levels(df_stats)
                                htf_levels_for_regime.extend(levels)
                            regime_indicator_data['htf_sr'] = htf_levels_for_regime
                        else:
                            regime_indicator_data['htf_sr'] = []

                        # Order Blocks data
                        if show_vob:
                            from indicators.volume_order_blocks import VolumeOrderBlocks
                            vob_for_regime = VolumeOrderBlocks(**vob_params) if vob_params else VolumeOrderBlocks()
                            regime_indicator_data['order_blocks'] = vob_for_regime.calculate(df_stats)

                        # RSI data
                        if show_rsi:
                            from indicators.ultimate_rsi import UltimateRSI
                            rsi_for_regime = UltimateRSI(**rsi_params) if rsi_params else UltimateRSI()
                            regime_indicator_data['rsi'] = rsi_for_regime.get_signals(df_stats)

                        # Money Flow Profile data
                        if show_money_flow_profile:
                            from indicators.money_flow_profile import MoneyFlowProfile
                            mfp_for_regime = MoneyFlowProfile(**money_flow_params) if money_flow_params else MoneyFlowProfile(num_rows=10)
                            regime_indicator_data['money_flow_profile'] = mfp_for_regime.get_signals(df_stats)

                        # DeltaFlow Profile data
                        if show_deltaflow_profile:
                            from indicators.deltaflow_volume_profile import DeltaFlowVolumeProfile
                            dfp_for_regime = DeltaFlowVolumeProfile(**deltaflow_params) if deltaflow_params else DeltaFlowVolumeProfile(bins=30)
                            regime_indicator_data['deltaflow_profile'] = dfp_for_regime.get_signals(df_stats)

                        # Detect regime
                        regime_detector = MarketRegimeDetector()
                        regime_result = regime_detector.detect_regime(df_stats, regime_indicator_data)

                        # Display regime info
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            regime = regime_result['regime']
                            regime_emoji = {
                                'STRONG_UPTREND': '🚀',
                                'WEAK_UPTREND': '📈',
                                'RANGING': '↔️',
                                'WEAK_DOWNTREND': '📉',
                                'STRONG_DOWNTREND': '💥',
                                'REVERSAL_TO_UPTREND': '🔄📈',
                                'REVERSAL_TO_DOWNTREND': '🔄📉',
                                'UNCERTAIN': '❓'
                            }
                            st.metric(
                                "Current Regime",
                                f"{regime_emoji.get(regime, '🎯')} {regime.replace('_', ' ').title()}",
                                delta=None
                            )

                        with col2:
                            confidence_pct = regime_result['confidence'] * 100
                            confidence_color = '🟢' if confidence_pct > 70 else '🟡' if confidence_pct > 50 else '🔴'
                            st.metric(
                                "Confidence",
                                f"{confidence_color} {confidence_pct:.1f}%",
                                delta=None
                            )

                        with col3:
                            volatility = regime_result['volatility']
                            vol_emoji = {'HIGH_VOLATILITY': '⚡', 'NORMAL_VOLATILITY': '📊', 'LOW_VOLATILITY': '💤'}
                            st.metric(
                                "Volatility",
                                f"{vol_emoji.get(volatility, '📊')} {volatility.replace('_', ' ').title()}",
                                delta=None
                            )

                        st.divider()

                        # Regime Details
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### 📊 Regime Indicators")
                            trend_dir = regime_result['trend_direction']
                            trend_strength = regime_result['trend_strength']

                            indicator_data = []
                            indicator_data.append({
                                'Indicator': 'Trend Direction',
                                'Value': f"{'🟢' if trend_dir == 'BULLISH' else '🔴' if trend_dir == 'BEARISH' else '⚪'} {trend_dir}"
                            })
                            indicator_data.append({
                                'Indicator': 'Trend Strength',
                                'Value': f"{trend_strength:.1%}"
                            })
                            indicator_data.append({
                                'Indicator': 'Ranging Market',
                                'Value': '✅ Yes' if regime_result['is_ranging'] else '❌ No'
                            })
                            indicator_data.append({
                                'Indicator': 'Reversal Signal',
                                'Value': '⚠️ Detected' if regime_result['reversal_signal'] else '✅ None'
                            })

                            st.dataframe(pd.DataFrame(indicator_data), use_container_width=True, hide_index=True)

                        with col2:
                            st.markdown("#### 💡 Trading Recommendations")
                            recs = regime_result['recommendations']

                            rec_data = []
                            rec_data.append({
                                'Recommendation': 'Position Bias',
                                'Value': recs['position_bias']
                            })
                            rec_data.append({
                                'Recommendation': 'Strategy',
                                'Value': recs['strategy'].replace('_', ' ').title()
                            })
                            rec_data.append({
                                'Recommendation': 'Position Size',
                                'Value': f"{recs['position_size_multiplier']:.1f}x"
                            })
                            rec_data.append({
                                'Recommendation': 'Stop Loss Width',
                                'Value': f"{recs['stop_loss_multiplier']:.1f}x"
                            })

                            st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

                        # Allowed Setups
                        st.markdown("#### ✅ Recommended Trade Setups")
                        allowed_setups = recs['allowed_setups']
                        if allowed_setups:
                            for setup in allowed_setups:
                                st.markdown(f"• {setup}")
                        else:
                            st.info("No specific setups recommended in current regime")

                    tab_idx += 1

                    # Volume Order Blocks Data
                    if show_vob:
                        with tabs[tab_idx]:
                            from indicators.volume_order_blocks import VolumeOrderBlocks
                            vob_indicator = VolumeOrderBlocks(**vob_params) if vob_params else VolumeOrderBlocks()
                            vob_data = vob_indicator.calculate(df_stats)

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("#### 🟢 Bullish Order Blocks")
                                if vob_data['bullish_blocks']:
                                    bullish_data = []
                                    for i, block in enumerate(vob_data['bullish_blocks']):
                                        bullish_data.append({
                                            'Block #': i + 1,
                                            'Lower': f"₹{block['lower']:.2f}",
                                            'Upper': f"₹{block['upper']:.2f}",
                                            'Mid': f"₹{block['mid']:.2f}",
                                            'Volume': f"{block['volume']:,.0f}",
                                            'Active': '✅' if block['active'] else '❌'
                                        })
                                    st.dataframe(pd.DataFrame(bullish_data), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No bullish order blocks detected")

                            with col2:
                                st.markdown("#### 🔴 Bearish Order Blocks")
                                if vob_data['bearish_blocks']:
                                    bearish_data = []
                                    for i, block in enumerate(vob_data['bearish_blocks']):
                                        bearish_data.append({
                                            'Block #': i + 1,
                                            'Lower': f"₹{block['lower']:.2f}",
                                            'Upper': f"₹{block['upper']:.2f}",
                                            'Mid': f"₹{block['mid']:.2f}",
                                            'Volume': f"{block['volume']:,.0f}",
                                            'Active': '✅' if block['active'] else '❌'
                                        })
                                    st.dataframe(pd.DataFrame(bearish_data), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No bearish order blocks detected")
                        tab_idx += 1

                    # HTF Support/Resistance Data
                    if show_htf_sr:
                        with tabs[tab_idx]:
                            st.markdown("#### 📊 Higher Timeframe Levels")
                            if htf_params and htf_params.get('levels_config'):
                                from indicators.htf_support_resistance import HTFSupportResistance

                                for level_config in htf_params['levels_config']:
                                    htf_indicator = HTFSupportResistance(
                                        timeframes=[level_config['timeframe']],
                                        pivot_length=level_config['length']
                                    )
                                    levels = htf_indicator.calculate_levels(df_stats)

                                    if levels:
                                        st.markdown(f"**{level_config['timeframe']} Timeframe** (Length: {level_config['length']})")
                                        levels_data = []
                                        for level in levels:
                                            levels_data.append({
                                                'Type': level['type'].upper(),
                                                'Price': f"₹{level['price']:.2f}",
                                                'Timeframe': level['timeframe'],
                                                'Strength': '🔴' * int(level.get('strength', 1))
                                            })
                                        st.dataframe(pd.DataFrame(levels_data), use_container_width=True, hide_index=True)
                                        st.divider()
                            else:
                                st.info("No HTF levels configured")
                        tab_idx += 1

                    # Volume Footprint Data
                    if show_footprint:
                        with tabs[tab_idx]:
                            from indicators.htf_volume_footprint import HTFVolumeFootprint
                            footprint_indicator = HTFVolumeFootprint(**footprint_params) if footprint_params else HTFVolumeFootprint()
                            footprint_data = footprint_indicator.calculate(df_stats)

                            st.markdown("#### 👣 Volume Footprint Analysis")

                            # Recent bars analysis
                            recent_bars = df_stats.tail(10).copy()
                            footprint_table = []

                            for idx in range(len(recent_bars)):
                                row_idx = recent_bars.index[idx]
                                bar_data = {
                                    'Time': recent_bars.index[idx].strftime('%H:%M') if hasattr(recent_bars.index[idx], 'strftime') else str(recent_bars.index[idx]),
                                    'Open': f"₹{recent_bars['open'].iloc[idx]:.2f}",
                                    'High': f"₹{recent_bars['high'].iloc[idx]:.2f}",
                                    'Low': f"₹{recent_bars['low'].iloc[idx]:.2f}",
                                    'Close': f"₹{recent_bars['close'].iloc[idx]:.2f}",
                                    'Volume': f"{recent_bars['volume'].iloc[idx]:,.0f}",
                                }

                                # Add footprint data if available
                                if 'buy_volume' in footprint_data:
                                    buy_vol = footprint_data['buy_volume'][row_idx] if row_idx in footprint_data['buy_volume'].index else 0
                                    sell_vol = footprint_data['sell_volume'][row_idx] if row_idx in footprint_data['sell_volume'].index else 0
                                    bar_data['Buy Vol'] = f"{buy_vol:,.0f}"
                                    bar_data['Sell Vol'] = f"{sell_vol:,.0f}"
                                    bar_data['Delta'] = f"{buy_vol - sell_vol:,.0f}"

                                footprint_table.append(bar_data)

                            st.dataframe(pd.DataFrame(footprint_table), use_container_width=True, hide_index=True)
                        tab_idx += 1

                    # Ultimate RSI Data
                    if show_rsi:
                        with tabs[tab_idx]:
                            from indicators.ultimate_rsi import UltimateRSI
                            rsi_indicator = UltimateRSI(**rsi_params) if rsi_params else UltimateRSI()
                            rsi_signals = rsi_indicator.get_signals(df_stats)

                            st.markdown("#### 📈 Ultimate RSI History")

                            # Last 20 values
                            rsi_table = []
                            for i in range(max(0, len(df_stats) - 20), len(df_stats)):
                                rsi_table.append({
                                    'Time': df_stats.index[i].strftime('%H:%M') if hasattr(df_stats.index[i], 'strftime') else str(df_stats.index[i]),
                                    'Price': f"₹{df_stats['close'].iloc[i]:.2f}",
                                    'RSI': f"{rsi_signals['ultimate_rsi'][i]:.2f}",
                                    'Signal': f"{rsi_signals['signal'][i]:.2f}",
                                    'Status': '🔴 OB' if rsi_signals['ultimate_rsi'][i] > (rsi_params.get('ob_level', 80) if rsi_params else 80) else '🟢 OS' if rsi_signals['ultimate_rsi'][i] < (rsi_params.get('os_level', 20) if rsi_params else 20) else '⚪ Neutral'
                                })
                            st.dataframe(pd.DataFrame(rsi_table), use_container_width=True, hide_index=True)

                            # Current values
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current RSI", f"{rsi_signals['ultimate_rsi'][-1]:.2f}")
                            with col2:
                                st.metric("Signal Line", f"{rsi_signals['signal'][-1]:.2f}")
                            with col3:
                                divergence = rsi_signals['ultimate_rsi'][-1] - rsi_signals['signal'][-1]
                                st.metric("Divergence", f"{divergence:.2f}")
                        tab_idx += 1

                    # OM Indicator Data
                    if show_om:
                        with tabs[tab_idx]:
                            from indicators.om_indicator import OMIndicator
                            om_indicator = OMIndicator(**om_params) if om_params else OMIndicator()
                            om_results = om_indicator.calculate(df_stats)

                            st.markdown("#### 🎯 OM Indicator Modules")

                            # VOB Module
                            if 'vob' in om_results:
                                st.markdown("**📦 Volume Order Blocks Module**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    if om_results['vob'].get('bullish_blocks'):
                                        st.markdown("🟢 **Bullish Blocks**")
                                        vob_bull = []
                                        for i, block in enumerate(om_results['vob']['bullish_blocks'][:5]):
                                            vob_bull.append({
                                                '#': i + 1,
                                                'Lower': f"₹{block['lower']:.2f}",
                                                'Upper': f"₹{block['upper']:.2f}",
                                                'Active': '✅' if block['active'] else '❌'
                                            })
                                        st.dataframe(pd.DataFrame(vob_bull), use_container_width=True, hide_index=True)
                                with col2:
                                    if om_results['vob'].get('bearish_blocks'):
                                        st.markdown("🔴 **Bearish Blocks**")
                                        vob_bear = []
                                        for i, block in enumerate(om_results['vob']['bearish_blocks'][:5]):
                                            vob_bear.append({
                                                '#': i + 1,
                                                'Lower': f"₹{block['lower']:.2f}",
                                                'Upper': f"₹{block['upper']:.2f}",
                                                'Active': '✅' if block['active'] else '❌'
                                            })
                                        st.dataframe(pd.DataFrame(vob_bear), use_container_width=True, hide_index=True)
                                st.divider()

                            # Delta Module
                            if 'delta' in om_results:
                                st.markdown("**📊 Delta Module (Buy/Sell Pressure)**")
                                delta_data = []
                                delta_values = om_results['delta']
                                for i in range(max(0, len(df_stats) - 10), len(df_stats)):
                                    delta_data.append({
                                        'Time': df_stats.index[i].strftime('%H:%M') if hasattr(df_stats.index[i], 'strftime') else str(df_stats.index[i]),
                                        'Delta': f"{delta_values[i]:.2f}" if i < len(delta_values) else 'N/A',
                                        'Signal': '🟢 Bullish' if i < len(delta_values) and delta_values[i] > 0 else '🔴 Bearish' if i < len(delta_values) else 'N/A'
                                    })
                                st.dataframe(pd.DataFrame(delta_data), use_container_width=True, hide_index=True)
                                st.divider()

                            # VIDYA Module
                            if 'vidya' in om_results:
                                st.markdown("**📈 VIDYA Trend Module**")
                                vidya_data = []
                                vidya_values = om_results['vidya']
                                for i in range(max(0, len(df_stats) - 10), len(df_stats)):
                                    vidya_data.append({
                                        'Time': df_stats.index[i].strftime('%H:%M') if hasattr(df_stats.index[i], 'strftime') else str(df_stats.index[i]),
                                        'Price': f"₹{df_stats['close'].iloc[i]:.2f}",
                                        'VIDYA': f"₹{vidya_values[i]:.2f}" if i < len(vidya_values) else 'N/A',
                                    })
                                st.dataframe(pd.DataFrame(vidya_data), use_container_width=True, hide_index=True)
                                st.divider()

                            # HVP Module
                            if 'hvp' in om_results and om_results['hvp'].get('pivots'):
                                st.markdown("**🎯 High Volume Pivots**")
                                hvp_data = []
                                for i, pivot in enumerate(om_results['hvp']['pivots'][:10]):
                                    hvp_data.append({
                                        '#': i + 1,
                                        'Price': f"₹{pivot['price']:.2f}",
                                        'Type': pivot['type'].upper(),
                                        'Volume': f"{pivot['volume']:,.0f}"
                                    })
                                st.dataframe(pd.DataFrame(hvp_data), use_container_width=True, hide_index=True)

                        tab_idx += 1

                    # Liquidity Sentiment Profile Data
                    if show_liquidity_profile:
                        with tabs[tab_idx]:
                            from indicators.liquidity_sentiment_profile import LiquiditySentimentProfile
                            lsp_indicator = LiquiditySentimentProfile(**liquidity_params) if liquidity_params else LiquiditySentimentProfile()
                            lsp_data = lsp_indicator.calculate(df_stats)

                            st.markdown("#### 💧 Liquidity & Sentiment Profile")

                            if 'price_levels' in lsp_data and len(lsp_data['price_levels']) > 0:
                                # Profile data table
                                profile_data = []
                                for level in lsp_data['price_levels']:
                                    profile_data.append({
                                        'Price': f"₹{level['price']:.2f}",
                                        'Total Volume': f"{level.get('total_volume', 0):,.0f}",
                                        'Buy Volume': f"{level.get('buy_volume', 0):,.0f}",
                                        'Sell Volume': f"{level.get('sell_volume', 0):,.0f}",
                                        'Sentiment': '🟢 Bullish' if level.get('buy_volume', 0) > level.get('sell_volume', 0) else '🔴 Bearish',
                                        'Activity': '🔥' * min(5, int(level.get('total_volume', 0) / max(1, lsp_data.get('avg_volume', 1))))
                                    })
                                st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True, height=400)

                                # Key levels
                                st.divider()
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if 'poc' in lsp_data:
                                        st.metric("Point of Control", f"₹{lsp_data['poc']:.2f}")
                                with col2:
                                    if 'high_volume_node' in lsp_data:
                                        st.metric("High Volume Node", f"₹{lsp_data['high_volume_node']:.2f}")
                                with col3:
                                    if 'low_volume_node' in lsp_data:
                                        st.metric("Low Volume Node", f"₹{lsp_data['low_volume_node']:.2f}")
                            else:
                                st.info("No liquidity profile data available")
                        tab_idx += 1

                    # Money Flow Profile Data
                    if show_money_flow_profile:
                        with tabs[tab_idx]:
                            from indicators.money_flow_profile import MoneyFlowProfile
                            mfp_indicator = MoneyFlowProfile(**money_flow_params) if money_flow_params else MoneyFlowProfile(num_rows=10)
                            mfp_signals = mfp_indicator.get_signals(df_stats)

                            st.markdown("#### 💰 Money Flow Profile Analysis")

                            if mfp_signals.get('success'):
                                # Key metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Sentiment", mfp_signals['sentiment'])
                                with col2:
                                    st.metric("POC Price", f"₹{mfp_signals['poc_price']:.2f}")
                                with col3:
                                    st.metric("Bullish Volume", f"{mfp_signals['bullish_volume_pct']:.1f}%")
                                with col4:
                                    st.metric("Bearish Volume", f"{mfp_signals['bearish_volume_pct']:.1f}%")

                                st.divider()

                                # Price position
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Current Price",
                                        f"₹{mfp_signals['current_price']:.2f}",
                                        delta=f"{mfp_signals['distance_from_poc']:+.2f} from POC"
                                    )
                                with col2:
                                    st.metric("Position", mfp_signals['price_position'])

                                # Volume levels
                                st.divider()
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**🔥 High Volume Levels (Consolidation)**")
                                    if mfp_signals['high_volume_levels']:
                                        hv_data = []
                                        for level in mfp_signals['high_volume_levels']:
                                            hv_data.append({
                                                'Price': f"₹{level:.2f}",
                                                'Type': '🟡 Value Area'
                                            })
                                        st.dataframe(pd.DataFrame(hv_data), use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No high volume levels detected")

                                with col2:
                                    st.markdown("**⚡ Low Volume Levels (Supply/Demand)**")
                                    if mfp_signals['low_volume_levels']:
                                        lv_data = []
                                        for level in mfp_signals['low_volume_levels']:
                                            lv_data.append({
                                                'Price': f"₹{level:.2f}",
                                                'Type': '🔵 Breakout Zone'
                                            })
                                        st.dataframe(pd.DataFrame(lv_data), use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No low volume levels detected")

                                # Consolidation zones
                                if mfp_signals.get('consolidation_zones'):
                                    st.divider()
                                    st.markdown("**📊 Consolidation Zones**")
                                    cz_data = []
                                    for zone in mfp_signals['consolidation_zones']:
                                        cz_data.append({
                                            'Lower': f"₹{zone['lower']:.2f}",
                                            'Upper': f"₹{zone['upper']:.2f}",
                                            'Range': f"₹{zone['upper'] - zone['lower']:.2f}"
                                        })
                                    st.dataframe(pd.DataFrame(cz_data), use_container_width=True, hide_index=True)

                                # Trading insights
                                st.divider()
                                st.markdown("**💡 Trading Insights**")
                                insights = []
                                if mfp_signals['sentiment'] == 'BULLISH':
                                    insights.append("🟢 Strong bullish volume dominance - Look for dip buying opportunities")
                                elif mfp_signals['sentiment'] == 'BEARISH':
                                    insights.append("🔴 Strong bearish volume dominance - Look for short opportunities")
                                else:
                                    insights.append("🟡 Neutral sentiment - Wait for clearer direction")

                                if mfp_signals['price_position'] == 'Above POC':
                                    insights.append("📈 Price above POC - Bulls in control, POC acts as support")
                                elif mfp_signals['price_position'] == 'Below POC':
                                    insights.append("📉 Price below POC - Bears in control, POC acts as resistance")

                                for insight in insights:
                                    st.info(insight)

                            else:
                                st.error(f"Error: {mfp_signals.get('error', 'Unknown error')}")
                        tab_idx += 1

                    # DeltaFlow Profile Data
                    if show_deltaflow_profile:
                        with tabs[tab_idx]:
                            from indicators.deltaflow_volume_profile import DeltaFlowVolumeProfile
                            dfp_indicator = DeltaFlowVolumeProfile(**deltaflow_params) if deltaflow_params else DeltaFlowVolumeProfile(bins=30)
                            dfp_signals = dfp_indicator.get_signals(df_stats)

                            st.markdown("#### ⚡ DeltaFlow Volume Profile Analysis")

                            if dfp_signals.get('success'):
                                # Key metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    sentiment_emoji = "🟢" if "BULLISH" in dfp_signals['sentiment'] else "🔴" if "BEARISH" in dfp_signals['sentiment'] else "🟡"
                                    st.metric("Sentiment", f"{sentiment_emoji} {dfp_signals['sentiment']}")
                                with col2:
                                    delta_color = "normal" if abs(dfp_signals['overall_delta']) < 10 else "inverse" if dfp_signals['overall_delta'] < 0 else "off"
                                    st.metric("Overall Delta", f"{dfp_signals['overall_delta']:+.1f}%", delta_color=delta_color)
                                with col3:
                                    st.metric("Buy Volume", f"{dfp_signals['overall_bull_pct']:.1f}%")
                                with col4:
                                    st.metric("Sell Volume", f"{dfp_signals['overall_bear_pct']:.1f}%")

                                st.divider()

                                # POC and position
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "POC Price",
                                        f"₹{dfp_signals['poc_price']:.2f}",
                                        help="Point of Control - Highest volume price level"
                                    )
                                with col2:
                                    st.metric(
                                        "Distance from POC",
                                        f"{dfp_signals['distance_from_poc_pct']:+.2f}%",
                                        help=f"Position: {dfp_signals['price_position']}"
                                    )

                                # Delta levels
                                st.divider()
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**🟢 Strong Buy Levels (Delta > +30%)**")
                                    if dfp_signals['strong_buy_levels']:
                                        buy_data = []
                                        for level in dfp_signals['strong_buy_levels']:
                                            buy_data.append({
                                                'Price': f"₹{level['price']:.2f}",
                                                'Delta': f"{level['delta']:+.1f}%",
                                                'Strength': '🟢' * min(5, int(abs(level['delta']) / 10))
                                            })
                                        st.dataframe(pd.DataFrame(buy_data), use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No strong buy levels detected")

                                with col2:
                                    st.markdown("**🔴 Strong Sell Levels (Delta < -30%)**")
                                    if dfp_signals['strong_sell_levels']:
                                        sell_data = []
                                        for level in dfp_signals['strong_sell_levels']:
                                            sell_data.append({
                                                'Price': f"₹{level['price']:.2f}",
                                                'Delta': f"{level['delta']:+.1f}%",
                                                'Strength': '🔴' * min(5, int(abs(level['delta']) / 10))
                                            })
                                        st.dataframe(pd.DataFrame(sell_data), use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No strong sell levels detected")

                                # Absorption zones
                                if dfp_signals['absorption_zones']:
                                    st.divider()
                                    st.markdown("**🛡️ Absorption Zones (High Volume, Low Delta)**")
                                    st.caption("Areas where large orders are being absorbed - potential reversal zones")
                                    az_data = []
                                    for zone in dfp_signals['absorption_zones']:
                                        az_data.append({
                                            'Lower': f"₹{zone['lower']:.2f}",
                                            'Upper': f"₹{zone['upper']:.2f}",
                                            'Volume': f"{zone['volume']:,.0f}",
                                            'Significance': '🛡️🛡️🛡️'
                                        })
                                    st.dataframe(pd.DataFrame(az_data), use_container_width=True, hide_index=True)

                                # Delta distribution summary
                                dfp_summary = dfp_indicator.get_delta_levels_summary(df_stats)
                                if dfp_summary.get('success'):
                                    st.divider()
                                    st.markdown("**📊 Delta Distribution Across Price Levels**")
                                    col1, col2, col3, col4, col5 = st.columns(5)
                                    with col1:
                                        st.metric("Strong Buy Bins", f"{dfp_summary['strong_buy']}")
                                    with col2:
                                        st.metric("Moderate Buy", f"{dfp_summary['moderate_buy']}")
                                    with col3:
                                        st.metric("Neutral", f"{dfp_summary['neutral']}")
                                    with col4:
                                        st.metric("Moderate Sell", f"{dfp_summary['moderate_sell']}")
                                    with col5:
                                        st.metric("Strong Sell Bins", f"{dfp_summary['strong_sell']}")

                                # Trading insights
                                st.divider()
                                st.markdown("**💡 Trading Insights**")
                                insights = []

                                if dfp_signals['sentiment'] == 'STRONG BULLISH':
                                    insights.append("🟢🟢 Extremely strong buying pressure - Momentum likely to continue")
                                elif dfp_signals['sentiment'] == 'BULLISH':
                                    insights.append("🟢 Bullish delta - Buyers in control")
                                elif dfp_signals['sentiment'] == 'STRONG BEARISH':
                                    insights.append("🔴🔴 Extremely strong selling pressure - Downtrend likely")
                                elif dfp_signals['sentiment'] == 'BEARISH':
                                    insights.append("🔴 Bearish delta - Sellers in control")
                                else:
                                    insights.append("🟡 Neutral delta - Balanced orderflow, wait for confirmation")

                                if dfp_signals['strong_buy_levels']:
                                    insights.append(f"📍 {len(dfp_signals['strong_buy_levels'])} strong buy level(s) detected - Potential support zones")

                                if dfp_signals['strong_sell_levels']:
                                    insights.append(f"📍 {len(dfp_signals['strong_sell_levels'])} strong sell level(s) detected - Potential resistance zones")

                                if dfp_signals['absorption_zones']:
                                    insights.append(f"🛡️ {len(dfp_signals['absorption_zones'])} absorption zone(s) - Watch for reversals at these levels")

                                for insight in insights:
                                    st.info(insight)

                            else:
                                st.error(f"Error: {dfp_signals.get('error', 'Unknown error')}")
                        tab_idx += 1

                    # Price Action Data
                    if show_bos or show_choch or show_fibonacci or show_patterns:
                        with tabs[tab_idx]:
                            from indicators.advanced_price_action import AdvancedPriceAction
                            pa_indicator = AdvancedPriceAction()

                            st.markdown("#### 🎯 Advanced Price Action Signals")

                            # BOS Detection
                            if show_bos:
                                st.markdown("**🔺 Break of Structure (BOS)**")
                                bos_data = pa_indicator.detect_bos(df_stats)
                                if bos_data and len(bos_data) > 0:
                                    bos_table = []
                                    for bos in bos_data[-10:]:
                                        bos_table.append({
                                            'Index': bos.get('index', 'N/A'),
                                            'Type': '🟢 Bullish' if bos['type'] == 'bullish' else '🔴 Bearish',
                                            'Price': f"₹{bos['price']:.2f}",
                                            'Strength': '🔴' * min(5, int(bos.get('strength', 1)))
                                        })
                                    st.dataframe(pd.DataFrame(bos_table), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No BOS signals detected")
                                st.divider()

                            # CHOCH Detection
                            if show_choch:
                                st.markdown("**🔄 Change of Character (CHOCH)**")
                                choch_data = pa_indicator.detect_choch(df_stats)
                                if choch_data and len(choch_data) > 0:
                                    choch_table = []
                                    for choch in choch_data[-10:]:
                                        choch_table.append({
                                            'Index': choch.get('index', 'N/A'),
                                            'Type': '🟢 Bullish' if choch['type'] == 'bullish' else '🔴 Bearish',
                                            'Price': f"₹{choch['price']:.2f}",
                                        })
                                    st.dataframe(pd.DataFrame(choch_table), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No CHOCH signals detected")
                                st.divider()

                            # Fibonacci Levels
                            if show_fibonacci:
                                st.markdown("**📐 Fibonacci Retracement Levels**")
                                fib_levels = pa_indicator.calculate_fibonacci(df_stats)
                                if fib_levels:
                                    fib_table = []
                                    for level_name, level_price in fib_levels.items():
                                        fib_table.append({
                                            'Level': level_name,
                                            'Price': f"₹{level_price:.2f}"
                                        })
                                    st.dataframe(pd.DataFrame(fib_table), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No Fibonacci levels calculated")
                                st.divider()

                            # Patterns
                            if show_patterns:
                                st.markdown("**📊 Geometric Patterns**")
                                patterns = pa_indicator.detect_patterns(df_stats)
                                if patterns and len(patterns) > 0:
                                    pattern_table = []
                                    for pattern in patterns[-10:]:
                                        pattern_table.append({
                                            'Pattern': pattern['name'],
                                            'Type': pattern['type'],
                                            'Start': pattern.get('start_idx', 'N/A'),
                                            'End': pattern.get('end_idx', 'N/A'),
                                        })
                                    st.dataframe(pd.DataFrame(pattern_table), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No patterns detected")

                # Trading signals based on indicators
                st.divider()
                st.subheader("🎯 Trading Signals")

                if show_rsi:
                    from indicators.ultimate_rsi import UltimateRSI
                    rsi_indicator = UltimateRSI(**rsi_params) if rsi_params else UltimateRSI()
                    rsi_signals = rsi_indicator.get_signals(df_stats)

                    latest_rsi = rsi_signals['ultimate_rsi'][-1]
                    latest_signal = rsi_signals['signal'][-1]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Ultimate RSI Analysis**")
                        ob_threshold = rsi_params['ob_level'] if rsi_params else 80
                        os_threshold = rsi_params['os_level'] if rsi_params else 20
                        rsi_state = "Overbought" if latest_rsi > ob_threshold else "Oversold" if latest_rsi < os_threshold else "Neutral"
                        rsi_color = "red" if latest_rsi > ob_threshold else "green" if latest_rsi < os_threshold else "gray"

                        st.markdown(f"Current RSI: <span style='color:{rsi_color}; font-size:24px;'>{latest_rsi:.2f}</span>", unsafe_allow_html=True)
                        st.write(f"Signal Line: {latest_signal:.2f}")
                        st.write(f"State: **{rsi_state}**")

                    with col2:
                        st.markdown("**RSI Trading Recommendation**")
                        if latest_rsi > ob_threshold:
                            st.warning("⚠️ **OVERBOUGHT** - Consider taking profits or waiting for pullback")
                        elif latest_rsi < os_threshold:
                            st.success("✅ **OVERSOLD** - Potential buying opportunity")
                        elif latest_rsi > latest_signal:
                            st.info("📈 **BULLISH** - RSI above signal line")
                        elif latest_rsi < latest_signal:
                            st.info("📉 **BEARISH** - RSI below signal line")
                        else:
                            st.info("⏸ **NEUTRAL** - No clear signal")

                if show_vob:
                    from indicators.volume_order_blocks import VolumeOrderBlocks
                    vob_indicator = VolumeOrderBlocks(**vob_params) if vob_params else VolumeOrderBlocks()
                    vob_data = vob_indicator.calculate(df_stats)

                    st.divider()
                    st.markdown("**Volume Order Blocks**")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"🟢 **Bullish OBs:** {len([b for b in vob_data['bullish_blocks'] if b['active']])}")
                        if len(vob_data['bullish_blocks']) > 0:
                            for i, block in enumerate(vob_data['bullish_blocks'][-3:]):
                                if block['active']:
                                    st.write(f"  - Support: {block['lower']:.2f} - {block['upper']:.2f}")

                    with col2:
                        st.write(f"🔴 **Bearish OBs:** {len([b for b in vob_data['bearish_blocks'] if b['active']])}")
                        if len(vob_data['bearish_blocks']) > 0:
                            for i, block in enumerate(vob_data['bearish_blocks'][-3:]):
                                if block['active']:
                                    st.write(f"  - Resistance: {block['lower']:.2f} - {block['upper']:.2f}")

                # Proximity Alerts Section
                st.divider()
                st.markdown("**🔔 Proximity Alerts**")
                st.caption("Automatic Telegram notifications when price is near key levels (Rate limited: 10 minutes)")

                try:
                    # Get proximity alert system
                    proximity_system = get_proximity_alert_system(cooldown_minutes=10)

                    # Get current price
                    current_price = df_stats['close'].iloc[-1]

                    # Determine symbol for alerts
                    alert_symbol = "NIFTY" if "NSEI" in symbol_code else "SENSEX" if "BSESN" in symbol_code else symbol_code

                    # Get HTF data if enabled
                    htf_data = []
                    if show_htf_sr and htf_params and htf_params.get('levels_config'):
                        from indicators.htf_support_resistance import HTFSupportResistance
                        for level_config in htf_params['levels_config']:
                            htf_indicator = HTFSupportResistance(
                                timeframes=[level_config['timeframe']],
                                pivot_length=level_config['length']
                            )
                            levels = htf_indicator.calculate_levels(df_stats)
                            htf_data.extend(levels)

                    # Process proximity alerts
                    all_alerts, notifications_sent = proximity_system.process_market_data(
                        symbol=alert_symbol,
                        current_price=current_price,
                        vob_data=vob_data if show_vob else {},
                        htf_data=htf_data
                    )

                    # Display alert summary
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Current Price", f"₹{current_price:,.2f}")

                    with col2:
                        vob_alert_count = len([a for a in all_alerts if a.alert_type == 'VOB'])
                        st.metric("VOB Alerts", vob_alert_count,
                                 delta="Within 7 pts" if vob_alert_count > 0 else None)

                    with col3:
                        htf_alert_count = len([a for a in all_alerts if a.alert_type == 'HTF'])
                        st.metric("HTF Alerts", htf_alert_count,
                                 delta="Within 5 pts" if htf_alert_count > 0 else None)

                    # Gather and display comprehensive market context
                    market_context = proximity_system._gather_market_context()

                    # Display market context in an expander
                    with st.expander("📊 **Comprehensive Market Context**", expanded=True):
                        st.markdown("### Overall Market Sentiment")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Overall sentiment
                            sentiment = market_context['overall_sentiment']
                            if sentiment == 'BULLISH':
                                st.success(f"🐂 **{sentiment}**")
                            elif sentiment == 'BEARISH':
                                st.error(f"🐻 **{sentiment}**")
                            else:
                                st.info(f"⚖️ **{sentiment}**")

                        with col2:
                            st.metric("Overall Score", f"{market_context['overall_score']:.1f}")

                        st.divider()

                        # Enhanced Market Analysis
                        st.markdown("### 📈 Enhanced Market Analysis")
                        col1, col2 = st.columns(2)

                        with col1:
                            bias = market_context['technical_indicators_bias']
                            if bias == 'BULLISH':
                                st.success(f"🐂 **{bias}**")
                            elif bias == 'BEARISH':
                                st.error(f"🐻 **{bias}**")
                            else:
                                st.info(f"⚖️ **{bias}**")

                        with col2:
                            st.metric("Score", f"{market_context['technical_indicators_score']:.1f}")

                        # Market Breadth
                        st.markdown("### 🔍 Market Breadth")
                        col1, col2 = st.columns(2)

                        with col1:
                            bias = market_context['market_breadth_bias']
                            if bias == 'BULLISH':
                                st.success(f"🐂 **{bias}**")
                            elif bias == 'BEARISH':
                                st.error(f"🐻 **{bias}**")
                            else:
                                st.info(f"⚖️ **{bias}**")

                        with col2:
                            st.metric("Breadth %", f"{market_context['market_breadth_pct']:.1f}%")

                        # PCR Analysis
                        st.markdown("### 📉 PCR Analysis (Put-Call Ratio)")
                        col1, col2 = st.columns(2)

                        with col1:
                            bias = market_context['pcr_analysis_bias']
                            if bias == 'BULLISH':
                                st.success(f"🐂 **{bias}**")
                            elif bias == 'BEARISH':
                                st.error(f"🐻 **{bias}**")
                            else:
                                st.info(f"⚖️ **{bias}**")

                        with col2:
                            st.metric("Score", f"{market_context['pcr_analysis_score']:.1f}")

                        # NIFTY ATM Zone
                        st.markdown("### 🎯 NIFTY ATM Zone Summary")
                        verdict = market_context['nifty_atm_verdict']
                        if 'Bullish' in verdict:
                            st.success(f"🐂 **{verdict}**")
                        elif 'Bearish' in verdict:
                            st.error(f"🐻 **{verdict}**")
                        else:
                            st.info(f"⚖️ **{verdict}**")

                        # Option Chain Analysis
                        st.markdown("### 🔗 Option Chain ATM Zone Analysis")
                        col1, col2 = st.columns(2)

                        with col1:
                            bias = market_context['option_chain_bias']
                            if bias == 'BULLISH':
                                st.success(f"🐂 **{bias}**")
                            elif bias == 'BEARISH':
                                st.error(f"🐻 **{bias}**")
                            else:
                                st.info(f"⚖️ **{bias}**")

                        with col2:
                            st.metric("Score", f"{market_context['option_chain_score']:.1f}")

                    # Show active alerts
                    if all_alerts:
                        st.markdown("**Active Proximity Alerts:**")
                        for alert in all_alerts[:5]:  # Show top 5 alerts
                            if alert.alert_type == 'VOB':
                                emoji = "🟢" if "Bull" in alert.level_type else "🔴"
                                st.write(f"{emoji} **{alert.alert_type}** {alert.level_type}: "
                                        f"₹{alert.level:.2f} "
                                        f"({alert.distance:.2f} pts away)")
                            else:  # HTF
                                emoji = "🟢" if alert.level_type == 'Support' else "🔴"
                                tf_readable = alert.timeframe.replace('T', 'm') if alert.timeframe else ''
                                st.write(f"{emoji} **{alert.alert_type}** {alert.level_type} ({tf_readable}): "
                                        f"₹{alert.level:.2f} "
                                        f"({alert.distance:.2f} pts away)")

                        if notifications_sent > 0:
                            st.success(f"📱 Sent {notifications_sent} Telegram notification(s) with comprehensive market context")
                    else:
                        st.info("ℹ️ No proximity alerts at current price")

                except Exception as e:
                    st.warning(f"⚠️ Proximity alerts unavailable: {str(e)}")

                # Data table
                st.divider()
                st.subheader("📋 Recent Candles Data")

                # Show last 20 candles
                display_df = df_stats.tail(20).copy()
                display_df = display_df.reset_index()

                if 'timestamp' in display_df.columns:
                    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                elif display_df.index.name == 'Datetime':
                    display_df['Time'] = display_df.index.strftime('%Y-%m-%d %H:%M')

                st.dataframe(display_df[['open', 'high', 'low', 'close', 'volume']].tail(20),
                           use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error rendering chart: {e}")
            st.write("Error details:", str(e))

    else:
        st.info("⏳ Loading chart... The chart will automatically load and refresh based on your settings")

        st.markdown("""
        ### About Advanced Chart Analysis

        This advanced charting module provides professional-grade technical analysis with 6 powerful indicators:

        #### 📊 Volume Bars
        - TradingView-style volume histogram
        - Green bars for bullish candles (close > open)
        - Red bars for bearish candles (close < open)
        - Essential for confirming price movements and identifying volume spikes

        #### 📦 Volume Order Blocks (BigBeluga)
        - Detects institutional order blocks based on volume and EMA crossovers
        - Shows bullish (support) and bearish (resistance) zones
        - Helps identify high-probability entry/exit zones

        #### 📊 HTF Support/Resistance (BigBeluga)
        - Multi-timeframe pivot analysis (4H, 12H, Daily, Weekly)
        - Identifies key support and resistance levels
        - Non-repainting pivot detection

        #### 👣 Real-Time HTF Volume Footprint (BigBeluga)
        - Volume distribution across price levels
        - Point of Control (POC) - highest volume traded price
        - Value Area - where 70% of volume occurred

        #### 📈 Ultimate RSI (LuxAlgo)
        - Enhanced RSI using price range instead of just price change
        - More responsive to market conditions
        - Signal line for trend confirmation
        - Overbought/Oversold detection

        #### 🎯 OM Indicator (Order Flow & Momentum)
        - **VWAP**: Volume Weighted Average Price for intraday trading
        - **VOB**: Volume Order Blocks with EMA-based detection
        - **HVP**: High Volume Pivots marking significant support/resistance
        - **Delta Module**: Buy/Sell pressure analysis with spike detection
        - **VIDYA**: Variable Index Dynamic Average with trend detection
        - **LTP Trap**: Last Traded Price trap signals for reversal detection
        - Comprehensive order flow analysis combining 6 sub-indicators

        #### 🎯 How to Use
        1. Select market (NIFTY, SENSEX, or DOW)
        2. Choose period and interval (default: 1 day, 1 minute)
        3. Chart loads automatically - no need to click any button!
        4. Set auto-refresh interval (default: 60s) or click Refresh for manual update
        5. Toggle indicators on/off as needed
        6. Analyze chart and trading signals
        7. Use signals to inform your trading decisions

        **Note:** All indicators are converted from Pine Script with high accuracy and optimized for Python/Plotly.
        """)

# ═══════════════════════════════════════════════════════════════════════
# TAB 8: NIFTY OPTION SCREENER V7.0
# ═══════════════════════════════════════════════════════════════════════

with tab8:
    st.header("🎯 NIFTY Option Screener v7.0")
    st.caption("100% SELLER'S PERSPECTIVE + ATM BIAS ANALYZER + MOMENT DETECTOR + EXPIRY SPIKE DETECTOR + ENHANCED OI/PCR ANALYTICS")

    # Import and render the Nifty Option Screener v7.0
    try:
        from NiftyOptionScreener import render_nifty_option_screener
        render_nifty_option_screener()
    except ImportError as e:
        st.error(f"❌ Failed to load Nifty Option Screener v7.0: {e}")
        st.info("Please ensure NiftyOptionScreener.py is in the project directory")
    except Exception as e:
        st.error(f"❌ Error rendering Nifty Option Screener: {e}")
        st.exception(e)

# ═══════════════════════════════════════════════════════════════════════
# TAB 9: ENHANCED MARKET DATA
# ═══════════════════════════════════════════════════════════════════════

with tab9:
    st.header("🌐 Enhanced Market Data Analysis")
    st.caption("Comprehensive market data from Dhan API + Yahoo Finance | India VIX, Sector Rotation, Global Markets, Intermarket Data, Gamma Squeeze, Intraday Timing")

    # Auto-fetch enhanced market data if not already loaded or stale (older than 1 minute)
    should_fetch = False

    if 'enhanced_market_data' not in st.session_state:
        should_fetch = True
    else:
        # Check if data is older than 1 minute
        from datetime import timedelta
        data_age = get_current_time_ist() - st.session_state.enhanced_market_data.get('timestamp', get_current_time_ist())
        if data_age > timedelta(minutes=1):
            should_fetch = True

    if should_fetch:
        with st.spinner("🔄 Auto-loading comprehensive market data from all sources..."):
            try:
                from enhanced_market_data import get_enhanced_market_data
                enhanced_data = get_enhanced_market_data()
                st.session_state.enhanced_market_data = enhanced_data
                st.success("✅ Enhanced market data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to fetch enhanced data: {e}")
                import traceback
                st.error(traceback.format_exc())

    # Control buttons
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True, key="refresh_enhanced_data_btn"):
            with st.spinner("Refreshing market data..."):
                try:
                    from enhanced_market_data import get_enhanced_market_data
                    enhanced_data = get_enhanced_market_data()
                    st.session_state.enhanced_market_data = enhanced_data
                    st.success("✅ Data refreshed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to refresh data: {e}")

    with col2:
        if 'enhanced_market_data' in st.session_state:
            data = st.session_state.enhanced_market_data
            st.caption(f"📅 Last Updated: {data['timestamp'].strftime('%Y-%m-%d %H:%M:%S IST')}")

    # Display enhanced market data if available
    if 'enhanced_market_data' in st.session_state:
        try:
            from enhanced_market_display import render_enhanced_market_data_tab
            render_enhanced_market_data_tab(st.session_state.enhanced_market_data)
        except Exception as e:
            st.error(f"❌ Error displaying enhanced data: {e}")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.info("""
        ℹ️ Enhanced market data will auto-load on first visit and refresh every 5 minutes.

        **Data Sources:**
        - 📊 **Dhan API:** India VIX, All Sector Indices (IT, Auto, Pharma, Metal, FMCG, Realty, Energy)
        - 🌍 **Yahoo Finance:** Global Markets (S&P 500, Nasdaq, Dow, Nikkei, Hang Seng, etc.)
        - 💰 **Intermarket:** USD Index, Crude Oil, Gold, USD/INR, US 10Y Treasury, Bitcoin

        **Advanced Analysis:**
        - ⚡ **India VIX Analysis:** Fear & Greed Index with sentiment scoring
        - 🏢 **Sector Rotation Model:** Identify market leadership and rotation patterns
        - 🎯 **Gamma Squeeze Detection:** Option market makers hedging analysis
        - ⏰ **Intraday Seasonality:** Time-based trading recommendations
        - 🌐 **Global Correlation:** How worldwide markets affect Indian markets

        **All data is presented in comprehensive tables with bias scores and trading insights!**
        """)

# ═══════════════════════════════════════════════════════════════════════
# TAB 10: MASTER AI ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

with tab10:
    # Add button to open in new tab
    st.markdown("""
    <div style="background-color: #1f77b4; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
        <h3 style="color: white; margin: 0 0 10px 0;">🚀 Open AI Analysis in New Browser Tab</h3>
        <p style="color: white; margin: 0 0 10px 0;">Get a dedicated full-screen view of all AI analysis modules</p>
        <a href="/🤖_AI_Analysis" target="_blank" style="text-decoration: none;">
            <button style="
                background-color: #4CAF50;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            ">
                🪟 Open in New Tab
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    try:
        # Auto-load market data if not already loaded
        if 'data_df' not in st.session_state or st.session_state.data_df is None:
            with st.spinner("📊 Loading NIFTY market data for analysis..."):
                # Fetch NIFTY data
                df = get_cached_chart_data('^NSEI', '1d', '1m')

                if df is not None and not df.empty:
                    # Add ATR indicator if not present
                    if 'ATR' not in df.columns:
                        from advanced_chart_analysis import AdvancedChartAnalysis
                        chart_analyzer = AdvancedChartAnalysis()
                        df = chart_analyzer.add_indicators(df)

                    # Store in session state for reuse
                    st.session_state.data_df = df
                else:
                    st.error("❌ Failed to load market data. Please check your connection and try again.")
                    st.stop()

        df = st.session_state.data_df

        # Get option chain data
        option_chain = {}
        if 'overall_option_data' in st.session_state and st.session_state.overall_option_data:
            option_chain = st.session_state.overall_option_data

        # Get VIX data
        vix_current = 15.0  # Default
        if 'enhanced_market_data' in st.session_state:
            try:
                vix_current = st.session_state.enhanced_market_data.get('india_vix', {}).get('current', 15.0)
            except:
                pass

        # VIX history (use recent values or create simple series)
        vix_history = pd.Series([vix_current] * 50)  # Simple placeholder

        # Get instrument
        selected_instrument = st.session_state.get('selected_index', 'NIFTY')

        # Calculate days to expiry (simple approximation)
        from datetime import datetime
        today = datetime.now()
        # Find next Thursday (weekly expiry)
        days_ahead = (3 - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        days_to_expiry = days_ahead

        # Render the AI analysis tab
        render_master_ai_analysis_tab(
            df=df,
            option_chain=option_chain,
            vix_current=vix_current,
            vix_history=vix_history,
            instrument=selected_instrument,
            days_to_expiry=days_to_expiry
        )
    except Exception as e:
        st.error(f"Error in Master AI Analysis: {e}")
        import traceback
        st.code(traceback.format_exc())

# Show info below (removed the warning about loading data)
    st.info("""
            📊 **Master AI Analysis** combines ALL advanced modules:

            1. 🌡️ **Volatility Regime Detection** - Detects market volatility state
            2. 🎯 **OI Trap Detection** - Identifies retail trapping patterns
            3. 📊 **CVD & Delta Imbalance** - Professional orderflow analysis
            4. 🏦 **Institutional vs Retail** - Detects smart money activity
            5. 🧲 **Liquidity Gravity** - Predicts price magnet levels
            6. 💰 **Position Sizing** - Dynamic lot calculation with Kelly Criterion
            7. 🛡️ **Risk Management** - Trailing stops, partial profits
            8. 📈 **Expectancy Model** - Statistical edge validation
            9. 🤖 **ML Market Regime** - AI-powered regime classification
            10. 📋 **Market Summary** - Comprehensive actionable insights

            **Result**: 75-85%+ win rate potential 🎯
            """)

# ═══════════════════════════════════════════════════════════════════════
# TAB 11: ADVANCED ANALYTICS
# ═══════════════════════════════════════════════════════════════════════

with tab11:
    # Add button to open in new tab
    st.markdown("""
    <div style="background-color: #9c27b0; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
        <h3 style="color: white; margin: 0 0 10px 0;">🚀 Open Advanced Analytics in New Browser Tab</h3>
        <p style="color: white; margin: 0 0 10px 0;">Explore individual AI modules in a dedicated full-screen view</p>
        <a href="/🤖_AI_Analysis" target="_blank" style="text-decoration: none;">
            <button style="
                background-color: #4CAF50;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            ">
                🪟 Open in New Tab
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    try:
        # Auto-load market data if not already loaded
        if 'data_df' not in st.session_state or st.session_state.data_df is None:
            with st.spinner("📊 Loading NIFTY market data for analysis..."):
                # Fetch NIFTY data
                df = get_cached_chart_data('^NSEI', '1d', '1m')

                if df is not None and not df.empty:
                    # Add ATR indicator if not present
                    if 'ATR' not in df.columns:
                        from advanced_chart_analysis import AdvancedChartAnalysis
                        chart_analyzer = AdvancedChartAnalysis()
                        df = chart_analyzer.add_indicators(df)

                    # Store in session state for reuse
                    st.session_state.data_df = df
                else:
                    st.error("❌ Failed to load market data. Please check your connection and try again.")
                    st.stop()

        df = st.session_state.data_df

        # Get option chain data
        option_chain = {}
        if 'overall_option_data' in st.session_state and st.session_state.overall_option_data:
            option_chain = st.session_state.overall_option_data

        # Get VIX data
        vix_current = 15.0
        if 'enhanced_market_data' in st.session_state:
            try:
                vix_current = st.session_state.enhanced_market_data.get('india_vix', {}).get('current', 15.0)
            except:
                pass

        vix_history = pd.Series([vix_current] * 50)

        # Render individual modules
        render_advanced_analytics_tab(
            df=df,
            option_chain=option_chain,
            vix_current=vix_current,
            vix_history=vix_history
        )
    except Exception as e:
        st.error(f"Error in Advanced Analytics: {e}")
        import traceback
        st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════
# TAB 12: SIGNAL HISTORY & PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════

with tab12:
    try:
        from signal_tracker import display_signal_history_tab, update_active_signals

        # Auto-update active signals with current market data
        if 'bias_analysis_results' in st.session_state and isinstance(st.session_state.bias_analysis_results, dict):
            df_price = st.session_state.bias_analysis_results.get('df')
            if df_price is not None and len(df_price) > 0:
                current_price = df_price['close'].iloc[-1]

                # Gather market data for ML exit analysis
                market_data = {}

                # Add ML regime
                if 'ml_regime_result' in st.session_state:
                    market_data['ml_regime'] = st.session_state.ml_regime_result

                # Add money flow signals
                if 'money_flow_signals' in st.session_state:
                    market_data['money_flow_signals'] = st.session_state.money_flow_signals

                # Add deltaflow signals
                if 'deltaflow_signals' in st.session_state:
                    market_data['deltaflow_signals'] = st.session_state.deltaflow_signals

                # Add ATM bias data
                if 'atm_bias_data' in st.session_state:
                    market_data['atm_bias_data'] = st.session_state.atm_bias_data

                # Add volatility result
                if 'volatility_result' in st.session_state:
                    market_data['volatility_result'] = st.session_state.volatility_result

                # Update active signals with ML analysis
                update_result = update_active_signals(current_price, market_data=market_data)

        # Display signal history and performance metrics
        display_signal_history_tab()

    except Exception as e:
        st.error(f"Error loading Signal History: {e}")
        import traceback
        st.code(traceback.format_exc())

        st.info("""
        **Signal Tracker Not Available**

        To use this feature:
        1. Ensure signal_tracker.py is in the project root
        2. Set up Supabase credentials in .streamlit/secrets.toml
        3. Run the SQL schema in Supabase (see signal_tracker.py for SQL)
        """)

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.divider()
st.caption(f"Last Updated (IST): {get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')} | Auto-refresh: {AUTO_REFRESH_INTERVAL}s")
st.caption(f"🤖 AI Market Analysis: Runs every 30 minutes during market hours | Last AI analysis: {datetime.fromtimestamp(st.session_state.last_ai_analysis_time).strftime('%H:%M:%S') if st.session_state.last_ai_analysis_time > 0 else 'Never'}")
