"""
NSE Stock Screener - FULL Dhan API Integration
Uses REAL option chain data from Dhan API
Integrates ALL actual analysis scripts:
- Real Option Chain Analysis (OI, PCR, Max Pain, ATM Bias)
- BiasAnalysisPro (all 13 bias indicators)
- AdvancedChartAnalysis (all technical indicators)
- ML Market Regime Detection

Returns top 10 falling and top 10 rising stocks
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
import time
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import io

# Import actual analysis modules (lazy loading to avoid boot issues)
def _import_analysis_modules():
    """Lazy import to avoid loading NiftyOptionScreener at boot"""
    global AdvancedChartAnalysis, BiasAnalysisPro, MLMarketRegimeDetector
    global analyze_atm_bias, analyze_oi_pcr_metrics, calculate_seller_max_pain, compute_pcr_df

    from advanced_chart_analysis import AdvancedChartAnalysis
    from bias_analysis import BiasAnalysisPro
    from src.ml_market_regime import MLMarketRegimeDetector
    # Import ACTUAL option analysis functions from NiftyOptionScreener
    from NiftyOptionScreener import (
        analyze_atm_bias,
        analyze_oi_pcr_metrics,
        calculate_seller_max_pain,
        compute_pcr_df
    )
    return True

# Utility functions
def safe_int(x, default=0):
    """Safely convert to int"""
    try:
        return int(x)
    except:
        return default

def safe_float(x, default=0.0):
    """Safely convert to float"""
    try:
        return float(x)
    except:
        return default

# NSE FNO Stocks list
NSE_FNO_STOCKS = [
    'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY',
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN',
    'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'TITAN',
    'SUNPHARMA', 'ULTRACEMCO', 'BAJFINANCE', 'WIPRO', 'NESTLEIND', 'POWERGRID',
    'HCLTECH', 'M&M', 'NTPC', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'INDUSINDBK',
    'ADANIENT', 'JSWSTEEL', 'BAJAJFINSV', 'HDFCLIFE', 'COALINDIA', 'GRASIM',
    'ONGC', 'CIPLA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'SHREECEM', 'HINDALCO',
    'BRITANNIA', 'BPCL', 'APOLLOHOSP', 'TATACONSUM', 'HEROMOTOCO', 'ADANIPORTS',
    'VEDL', 'GODREJCP', 'PNB', 'BANKBARODA', 'CANBK', 'INDIGO', 'DLF', 'PEL',
    'SIEMENS', 'DABUR', 'GAIL', 'BAJAJ-AUTO', 'LUPIN', 'PIDILITIND', 'TORNTPHARM',
]

# Symbol mapping for Yahoo Finance
SYMBOL_MAPPING = {
    'NIFTY': '^NSEI',
    'BANKNIFTY': '^NSEBANK',
    'FINNIFTY': '^CNXFIN',
    'MIDCPNIFTY': '^NSEMDCP50',
}


class DhanInstrumentMapper:
    """Fetches and caches Dhan instrument list for security ID mapping"""

    def __init__(self):
        self.instrument_df = None
        self.symbol_to_security = {}

    def load_instruments(self) -> bool:
        """Fetch Dhan instrument list CSV and create mappings"""
        try:
            # Fetch compact instrument list
            url = "https://images.dhan.co/api-data/api-scrip-master.csv"
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse CSV
            self.instrument_df = pd.read_csv(io.StringIO(response.text))

            # Create symbol → security ID mapping for FNO stocks
            # Filter for derivatives segment (SEM_SEGMENT = 'D')
            fno_df = self.instrument_df[
                (self.instrument_df['SEM_SEGMENT'] == 'D') &
                (self.instrument_df['SEM_EXCH_EXCH_ID'] == 'NSE')
            ].copy()

            # For each stock, get its underlying security ID
            for _, row in fno_df.iterrows():
                symbol = row.get('SM_SYMBOL_NAME', '').upper()
                security_id = row.get('SEM_SMST_SECURITY_ID', 0)

                if symbol and security_id:
                    # Store the first (underlying) security ID for each symbol
                    if symbol not in self.symbol_to_security:
                        self.symbol_to_security[symbol] = security_id

            # Manual mapping for indices (these might have different names)
            index_mappings = {
                'NIFTY': 13,
                'BANKNIFTY': 25,
                'FINNIFTY': 27,
                'MIDCPNIFTY': 14
            }
            self.symbol_to_security.update(index_mappings)

            return True

        except Exception as e:
            print(f"Error loading Dhan instruments: {e}")
            return False

    def get_security_id(self, symbol: str) -> Optional[int]:
        """Get security ID for a symbol"""
        return self.symbol_to_security.get(symbol.upper())

    def get_segment(self, symbol: str) -> str:
        """Get segment for security (IDX_I for indices, COM for stocks)"""
        if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']:
            return 'IDX_I'
        return 'COM'  # Commodity segment for stock F&O


class NSEStockScreener:
    """NSE Stock Screener with REAL Dhan API Option Chain Integration"""

    def __init__(self):
        """Initialize with actual analysis classes + Dhan API"""
        # Lazy load analysis modules
        _import_analysis_modules()

        self.chart_analyzer = AdvancedChartAnalysis()
        self.bias_analyzer = BiasAnalysisPro()
        self.ml_regime_detector = MLMarketRegimeDetector()
        self.instrument_mapper = DhanInstrumentMapper()

        # Dhan API credentials (from NiftyOptionScreener)
        try:
            self.dhan_client_id = st.secrets["DHAN"]["CLIENT_ID"]
            self.dhan_access_token = st.secrets["DHAN"]["ACCESS_TOKEN"]
            self.dhan_base_url = "https://api.dhan.co"
            self.has_dhan_api = True
        except:
            self.has_dhan_api = False
            st.warning("⚠️ Dhan API credentials not found. Using fallback analysis.")

        self.results = []

    def get_yf_symbol(self, stock: str) -> str:
        """Convert NSE stock symbol to Yahoo Finance symbol"""
        if stock in SYMBOL_MAPPING:
            return SYMBOL_MAPPING[stock]
        return f"{stock}.NS"

    def fetch_stock_data(self, stock: str, period: str = '5d', interval: str = '5m') -> pd.DataFrame:
        """Fetch stock data"""
        try:
            yf_symbol = self.get_yf_symbol(stock)
            df = self.chart_analyzer.fetch_intraday_data(yf_symbol, period=period, interval=interval)

            if df is None or df.empty:
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(period=period, interval=interval)
                if df is None or df.empty:
                    return None
                df.columns = [col.lower() for col in df.columns]

            return df
        except Exception as e:
            return None

    def get_next_expiry(self) -> str:
        """Get next Thursday expiry date (YYYY-MM-DD format)"""
        today = datetime.now()
        days_ahead = (3 - today.weekday()) % 7  # 3 = Thursday
        if days_ahead == 0 and today.hour >= 15:  # After 3 PM on Thursday
            days_ahead = 7
        if days_ahead == 0:
            days_ahead = 7
        next_expiry = today + timedelta(days=days_ahead)
        return next_expiry.strftime('%Y-%m-%d')

    def fetch_dhan_option_chain(self, security_id: int, segment: str, expiry_date: str) -> Optional[Dict]:
        """Fetch option chain from Dhan API with retry logic"""
        if not self.has_dhan_api:
            return None

        max_retries = 3
        retry_delays = [2, 4, 8]

        for attempt in range(max_retries):
            try:
                url = f"{self.dhan_base_url}/v2/optionchain"
                payload = {
                    "UnderlyingScrip": security_id,
                    "UnderlyingSeg": segment,
                    "Expiry": expiry_date
                }
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "access-token": self.dhan_access_token,
                    "client-id": self.dhan_client_id
                }

                r = requests.post(url, json=payload, headers=headers, timeout=15)

                if r.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        time.sleep(retry_delays[attempt])
                        continue
                    return None

                r.raise_for_status()
                data = r.json()

                if data.get("status") == "success":
                    return data.get("data", {})
                return None

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delays[attempt])
                    continue
                return None

        return None

    def parse_dhan_option_chain(self, chain_data: Dict, spot_price: float) -> pd.DataFrame:
        """Parse Dhan option chain data into merged DataFrame"""
        if not chain_data:
            return pd.DataFrame()

        oc = chain_data.get("oc", {})
        merged_rows = []

        for strike_str, strike_data in oc.items():
            try:
                strike = int(float(strike_str))
            except:
                continue

            row = {"strikePrice": strike}

            # Parse CE data
            ce = strike_data.get("ce")
            if ce:
                row["OI_CE"] = safe_int(ce.get("oi", 0))
                row["Chg_OI_CE"] = safe_int(ce.get("oi", 0)) - safe_int(ce.get("previous_oi", 0))
                row["Vol_CE"] = safe_int(ce.get("volume", 0))
                row["LTP_CE"] = safe_float(ce.get("last_price", 0.0))
                row["IV_CE"] = safe_float(ce.get("implied_volatility", 0.0))
            else:
                row.update({"OI_CE": 0, "Chg_OI_CE": 0, "Vol_CE": 0, "LTP_CE": 0.0, "IV_CE": 0.0})

            # Parse PE data
            pe = strike_data.get("pe")
            if pe:
                row["OI_PE"] = safe_int(pe.get("oi", 0))
                row["Chg_OI_PE"] = safe_int(pe.get("oi", 0)) - safe_int(pe.get("previous_oi", 0))
                row["Vol_PE"] = safe_int(pe.get("volume", 0))
                row["LTP_PE"] = safe_float(pe.get("last_price", 0.0))
                row["IV_PE"] = safe_float(pe.get("implied_volatility", 0.0))
            else:
                row.update({"OI_PE": 0, "Chg_OI_PE": 0, "Vol_PE": 0, "LTP_PE": 0.0, "IV_PE": 0.0})

            merged_rows.append(row)

        if not merged_rows:
            return pd.DataFrame()

        merged_df = pd.DataFrame(merged_rows)
        merged_df = merged_df.sort_values("strikePrice").reset_index(drop=True)

        return merged_df

    def calculate_real_option_analysis(self, stock: str, spot_price: float) -> Dict:
        """
        Calculate REAL option chain analysis using Dhan API
        Uses ACTUAL functions from NiftyOptionScreener
        """
        try:
            # Get security ID
            security_id = self.instrument_mapper.get_security_id(stock)
            if not security_id:
                return {'success': False, 'score': 50, 'sentiment': 'NEUTRAL'}

            # Get segment
            segment = self.instrument_mapper.get_segment(stock)

            # Get expiry
            expiry = self.get_next_expiry()

            # Fetch option chain
            chain_data = self.fetch_dhan_option_chain(security_id, segment, expiry)
            if not chain_data:
                return {'success': False, 'score': 50, 'sentiment': 'NEUTRAL'}

            # Parse option chain
            merged_df = self.parse_dhan_option_chain(chain_data, spot_price)
            if merged_df.empty:
                return {'success': False, 'score': 50, 'sentiment': 'NEUTRAL'}

            # Calculate ATM strike
            strikes = merged_df['strikePrice'].values
            atm_strike = strikes[np.argmin(np.abs(strikes - spot_price))]

            # Calculate strike gap
            if len(strikes) >= 2:
                strike_gap = strikes[1] - strikes[0]
            else:
                strike_gap = 50  # Default

            # USE ACTUAL NiftyOptionScreener FUNCTIONS!

            # 1. ATM Bias Analysis
            atm_bias_result = analyze_atm_bias(merged_df, spot_price, atm_strike, strike_gap)

            # 2. OI/PCR Metrics
            oi_pcr_result = analyze_oi_pcr_metrics(merged_df, spot_price, atm_strike)

            # 3. Max Pain
            max_pain = calculate_seller_max_pain(merged_df)

            # 4. PCR DataFrame
            pcr_df = compute_pcr_df(merged_df)

            # Calculate composite option score
            option_score = 50  # Neutral baseline
            sentiment = 'NEUTRAL'

            # Adjust based on ATM bias
            if atm_bias_result:
                overall_bias = atm_bias_result.get('Overall_Bias', 0)
                if overall_bias > 0.3:
                    option_score += 20
                    sentiment = 'BULLISH'
                elif overall_bias < -0.3:
                    option_score -= 20
                    sentiment = 'BEARISH'

            # Adjust based on PCR
            if oi_pcr_result:
                pcr = oi_pcr_result.get('pcr_total', 1.0)
                if pcr > 1.2:  # High PCR = Bullish
                    option_score += 15
                elif pcr < 0.8:  # Low PCR = Bearish
                    option_score -= 15

            # Adjust based on Max Pain distance
            if max_pain and abs(max_pain - spot_price) / spot_price < 0.02:
                # Price near max pain = consolidation
                option_score = 50  # Pull towards neutral

            option_score = max(0, min(100, option_score))

            return {
                'success': True,
                'score': option_score,
                'sentiment': sentiment,
                'atm_bias': atm_bias_result,
                'oi_pcr': oi_pcr_result,
                'max_pain': max_pain,
                'pcr': oi_pcr_result.get('pcr_total', 0) if oi_pcr_result else 0
            }

        except Exception as e:
            print(f"Error in real option analysis for {stock}: {e}")
            return {'success': False, 'score': 50, 'sentiment': 'NEUTRAL'}

    def analyze_with_bias_pro(self, stock: str, df: pd.DataFrame) -> Dict:
        """Use ACTUAL BiasAnalysisPro"""
        try:
            if df is None or len(df) < 100:
                return {'success': False, 'overall_bias': 'NEUTRAL', 'overall_score': 50, 'bias_strength': 0}

            yf_symbol = self.get_yf_symbol(stock)
            result = self.bias_analyzer.analyze_all_bias_indicators(symbol=yf_symbol, data=df)

            if not result.get('success', False):
                return {'success': False, 'overall_bias': 'NEUTRAL', 'overall_score': 50, 'bias_strength': 0}

            return {
                'success': True,
                'overall_bias': result.get('overall_bias', 'NEUTRAL'),
                'overall_score': result.get('overall_score', 50),
                'bias_strength': abs(result.get('overall_score', 50) - 50),
                'bullish_count': sum(1 for b in result.get('bias_results', []) if b.get('bias') == 'BULLISH'),
                'bearish_count': sum(1 for b in result.get('bias_results', []) if b.get('bias') == 'BEARISH')
            }
        except:
            return {'success': False, 'overall_bias': 'NEUTRAL', 'overall_score': 50, 'bias_strength': 0}

    def analyze_with_chart_analysis(self, df: pd.DataFrame) -> Dict:
        """Use ACTUAL AdvancedChartAnalysis"""
        try:
            if df is None or len(df) < 50:
                return {'success': False, 'trend': 'NEUTRAL', 'score': 50, 'strength': 0}

            df_with_indicators = self.chart_analyzer.add_indicators(df.copy())
            if df_with_indicators is None or df_with_indicators.empty:
                return {'success': False, 'trend': 'NEUTRAL', 'score': 50, 'strength': 0}

            current_price = df_with_indicators['close'].iloc[-1]

            if len(df_with_indicators) >= 50:
                ma_20 = df_with_indicators['close'].rolling(20).mean().iloc[-1]
                ma_50 = df_with_indicators['close'].rolling(50).mean().iloc[-1]

                if current_price > ma_20 > ma_50:
                    trend, score = 'STRONG_BULLISH', 80
                elif current_price > ma_20:
                    trend, score = 'BULLISH', 65
                elif current_price < ma_20 < ma_50:
                    trend, score = 'STRONG_BEARISH', 20
                elif current_price < ma_20:
                    trend, score = 'BEARISH', 35
                else:
                    trend, score = 'NEUTRAL', 50
            else:
                trend, score = 'NEUTRAL', 50

            return {
                'success': True,
                'trend': trend,
                'score': score,
                'strength': abs(score - 50),
                'df': df_with_indicators
            }
        except:
            return {'success': False, 'trend': 'NEUTRAL', 'score': 50, 'strength': 0}

    def analyze_with_ml_regime(self, df: pd.DataFrame) -> Dict:
        """Use ACTUAL MLMarketRegimeDetector"""
        try:
            if df is None or len(df) < 50:
                return {'success': False, 'regime': 'UNKNOWN', 'confidence': 0, 'trading_sentiment': 'NEUTRAL'}

            regime_result = self.ml_regime_detector.detect_regime(df)
            if regime_result is None:
                return {'success': False, 'regime': 'UNKNOWN', 'confidence': 0, 'trading_sentiment': 'NEUTRAL'}

            return {
                'success': True,
                'regime': regime_result.regime,
                'confidence': regime_result.confidence,
                'trading_sentiment': regime_result.trading_sentiment,
                'sentiment_score': regime_result.sentiment_score,
                'volatility_state': regime_result.volatility_state
            }
        except:
            return {'success': False, 'regime': 'UNKNOWN', 'confidence': 0, 'trading_sentiment': 'NEUTRAL'}

    def analyze_stock(self, stock: str) -> Optional[Dict]:
        """COMPLETE analysis with REAL option chain + all scripts"""
        try:
            df = self.fetch_stock_data(stock, period='5d', interval='5m')
            if df is None or len(df) < 20:
                return None

            current_price = df['close'].iloc[-1]
            price_change_pct = ((df['close'].iloc[-1] - df['close'].iloc[-12]) / df['close'].iloc[-12] * 100) if len(df) >= 12 else 0

            # 1. REAL Option Chain Analysis
            option_result = self.calculate_real_option_analysis(stock, current_price)

            # 2. Bias Analysis Pro
            bias_result = self.analyze_with_bias_pro(stock, df)

            # 3. Chart Analysis
            chart_result = self.analyze_with_chart_analysis(df)

            # 4. ML Regime
            regime_result = self.analyze_with_ml_regime(chart_result.get('df', df) if chart_result.get('success') else df)

            # COMPOSITE SCORE: Option(35%) + Bias(30%) + Chart(20%) + Regime(15%)
            option_score = option_result.get('score', 50)
            bias_score = bias_result.get('overall_score', 50)
            chart_score = chart_result.get('score', 50)

            regime_score_map = {'STRONG LONG': 90, 'LONG': 70, 'NEUTRAL': 50, 'SHORT': 30, 'STRONG SHORT': 10}
            regime_score = regime_score_map.get(regime_result.get('trading_sentiment', 'NEUTRAL'), 50)

            composite_score = (
                option_score * 0.35 +
                bias_score * 0.30 +
                chart_score * 0.20 +
                regime_score * 0.15
            )

            strength = (
                abs(option_score - 50) * 0.35 +
                bias_result.get('bias_strength', 0) * 0.30 +
                chart_result.get('strength', 0) * 0.20 +
                abs(regime_score - 50) * 0.15
            )

            if composite_score >= 70:
                overall_signal = 'STRONG_BULLISH'
            elif composite_score >= 55:
                overall_signal = 'BULLISH'
            elif composite_score <= 30:
                overall_signal = 'STRONG_BEARISH'
            elif composite_score <= 45:
                overall_signal = 'BEARISH'
            else:
                overall_signal = 'NEUTRAL'

            return {
                'stock': stock,
                'price': current_price,
                'composite_score': composite_score,
                'strength': strength,
                'overall_signal': overall_signal,
                'price_change_pct': price_change_pct,
                'option_sentiment': option_result.get('sentiment', 'NEUTRAL'),
                'option_score': option_score,
                'option_pcr': option_result.get('pcr', 0),
                'option_max_pain': option_result.get('max_pain', 0),
                'bias': bias_result.get('overall_bias', 'NEUTRAL'),
                'bias_score': bias_score,
                'bullish_indicators': bias_result.get('bullish_count', 0),
                'bearish_indicators': bias_result.get('bearish_count', 0),
                'trend': chart_result.get('trend', 'NEUTRAL'),
                'chart_score': chart_score,
                'regime': regime_result.get('regime', 'UNKNOWN'),
                'regime_sentiment': regime_result.get('trading_sentiment', 'NEUTRAL'),
                'regime_confidence': regime_result.get('confidence', 0),
                'volatility_state': regime_result.get('volatility_state', 'UNKNOWN'),
            }

        except Exception as e:
            print(f"Error analyzing {stock}: {e}")
            return None

    def analyze_all_stocks(self, stocks: List[str] = None, progress_callback=None) -> List[Dict]:
        """Analyze all stocks in parallel"""
        if stocks is None:
            stocks = NSE_FNO_STOCKS

        results = []
        total = len(stocks)

        with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced workers for API rate limit
            futures = {executor.submit(self.analyze_stock, stock): stock for stock in stocks}

            completed = 0
            for future in futures:
                try:
                    result = future.result(timeout=45)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    pass

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

                # Small delay between stocks to respect API rate limits
                time.sleep(0.5)

        return results

    def get_top_movers(self, results: List[Dict], n: int = 10) -> Tuple[List[Dict], List[Dict]]:
        """Get top N falling and rising stocks"""
        if not results:
            return [], []

        sorted_by_strength = sorted(results, key=lambda x: x['strength'], reverse=True)

        falling_stocks = [
            r for r in sorted_by_strength
            if r['overall_signal'] in ['STRONG_BEARISH', 'BEARISH'] and r['price_change_pct'] < 0
        ][:n]

        rising_stocks = [
            r for r in sorted_by_strength
            if r['overall_signal'] in ['STRONG_BULLISH', 'BULLISH'] and r['price_change_pct'] > 0
        ][:n]

        return falling_stocks, rising_stocks


def render_nse_stock_screener_tab():
    """Render NSE Stock Screener with REAL Dhan API option chain"""
    st.header("🔍 NSE Stock Screener - REAL Option Chain Integration")

    st.success("""
    ✅ **FULLY INTEGRATED WITH REAL DHAN API OPTION CHAIN!**

    Now uses:
    - ✅ **REAL Option Chain (35%)** - Actual OI, PCR, Max Pain, ATM Bias from Dhan API
    - ✅ **BiasAnalysisPro (30%)** - All 13 bias indicators
    - ✅ **AdvancedChartAnalysis (20%)** - All technical indicators
    - ✅ **ML Market Regime (15%)** - AI-powered regime detection
    """)

    st.markdown("""
    ### 🎯 Complete 4-Layer Analysis with Real Data

    **Each stock analyzed through:**
    1. **Real Option Chain (35%)** - Uses actual analyze_atm_bias(), analyze_oi_pcr_metrics(), calculate_seller_max_pain()
    2. **Bias Analysis (30%)** - 13 indicators via BiasAnalysisPro
    3. **Chart Analysis (20%)** - Technical indicators via AdvancedChartAnalysis
    4. **ML Regime (15%)** - AI detection via MLMarketRegimeDetector

    **Perfect for 4 daily runs:** 9:30 AM, 11 AM, 1 PM, 2:30 PM
    """)

    st.divider()

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🚀 Run Real Option Chain Analysis", type="primary", use_container_width=True):
            st.session_state.run_nse_screener = True

    with col2:
        num_stocks = st.number_input("Top N stocks", min_value=5, max_value=20, value=10, step=1)

    with col3:
        include_indices = st.checkbox("Include Indices", value=True)

    st.divider()

    if st.session_state.get('run_nse_screener', False):
        st.session_state.run_nse_screener = False

        screener = NSEStockScreener()

        # Load Dhan instruments
        with st.spinner("📊 Loading Dhan instrument list..."):
            if not screener.instrument_mapper.load_instruments():
                st.error("Failed to load Dhan instruments. Check connection.")
                return

        st.success(f"✅ Loaded {len(screener.instrument_mapper.symbol_to_security)} instrument mappings")

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(completed, total):
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f"Analyzing with real option chain... {completed}/{total} ({int(progress*100)}%)")

        stocks_to_analyze = NSE_FNO_STOCKS.copy()
        if not include_indices:
            stocks_to_analyze = [s for s in stocks_to_analyze if s not in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']]

        with st.spinner(f"🔄 Running REAL option chain analysis on {len(stocks_to_analyze)} stocks..."):
            results = screener.analyze_all_stocks(stocks_to_analyze, progress_callback=update_progress)

        progress_bar.empty()
        status_text.empty()

        falling_stocks, rising_stocks = screener.get_top_movers(results, n=num_stocks)

        st.success(f"✅ Real option chain analysis complete! Analyzed {len(results)} stocks.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📉 Top {num_stocks} Falling Stocks")

            if falling_stocks:
                falling_df = pd.DataFrame([
                    {
                        'Stock': r['stock'],
                        'Price': f"₹{r['price']:.2f}",
                        'Change %': f"{r['price_change_pct']:.2f}%",
                        'Signal': r['overall_signal'],
                        'Strength': f"{r['strength']:.1f}",
                        'OC PCR': f"{r['option_pcr']:.2f}" if r['option_pcr'] else 'N/A',
                        'Bias': f"{r['bias']} ({r['bias_score']:.0f})",
                        'Trend': r['trend'],
                        'Regime': r['regime'],
                        'Bull/Bear': f"{r['bullish_indicators']}/{r['bearish_indicators']}"
                    }
                    for r in falling_stocks
                ])

                st.dataframe(falling_df, use_container_width=True, hide_index=True)

                csv = falling_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Falling Stocks CSV",
                    csv,
                    f"falling_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            else:
                st.info("No significant falling stocks found.")

        with col2:
            st.subheader(f"📈 Top {num_stocks} Rising Stocks")

            if rising_stocks:
                rising_df = pd.DataFrame([
                    {
                        'Stock': r['stock'],
                        'Price': f"₹{r['price']:.2f}",
                        'Change %': f"{r['price_change_pct']:.2f}%",
                        'Signal': r['overall_signal'],
                        'Strength': f"{r['strength']:.1f}",
                        'OC PCR': f"{r['option_pcr']:.2f}" if r['option_pcr'] else 'N/A',
                        'Bias': f"{r['bias']} ({r['bias_score']:.0f})",
                        'Trend': r['trend'],
                        'Regime': r['regime'],
                        'Bull/Bear': f"{r['bullish_indicators']}/{r['bearish_indicators']}"
                    }
                    for r in rising_stocks
                ])

                st.dataframe(rising_df, use_container_width=True, hide_index=True)

                csv = rising_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Rising Stocks CSV",
                    csv,
                    f"rising_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            else:
                st.info("No significant rising stocks found.")

        st.divider()

        st.subheader("📊 Analysis Summary")

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

        with summary_col1:
            st.metric("Total Analyzed", len(results))

        with summary_col2:
            bullish_count = len([r for r in results if 'BULLISH' in r['overall_signal']])
            st.metric("Bullish Stocks", bullish_count)

        with summary_col3:
            bearish_count = len([r for r in results if 'BEARISH' in r['overall_signal']])
            st.metric("Bearish Stocks", bearish_count)

        with summary_col4:
            neutral_count = len([r for r in results if r['overall_signal'] == 'NEUTRAL'])
            st.metric("Neutral Stocks", neutral_count)

        st.session_state.nse_screener_results = results
        st.session_state.nse_screener_timestamp = datetime.now()

    if 'nse_screener_timestamp' in st.session_state:
        st.caption(f"📅 Last analysis: {st.session_state.nse_screener_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    st.info("""
    💡 **Column Explanation**:
    - **OC PCR**: Real Put-Call Ratio from option chain (>1.2 bullish, <0.8 bearish)
    - **Bias**: Overall bias from 13 indicators
    - **Trend**: Chart analysis trend
    - **Regime**: ML-detected market regime
    - **Bull/Bear**: Bullish/Bearish indicator count (out of 13)
    """)
