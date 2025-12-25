"""
NSE Stock Screener - On-Demand Comprehensive Analysis
Analyzes all NSE FNO stocks using:
- Option Chain Analysis (from NIFTY Option Screener logic)
- Bias Analysis Pro (13 bias indicators)
- Advanced Chart Analysis (with ML Market Regime)
Returns top 10 falling and top 10 rising stocks
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from typing import Dict, List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback

# Import existing analysis modules
try:
    from advanced_chart_analysis import AdvancedChartAnalysis
    from bias_analysis import BiasAnalysisPro
    from src.ml_market_regime import MLMarketRegimeDetector
except ImportError as e:
    st.error(f"Error importing analysis modules: {e}")

# Comprehensive list of NSE FNO stocks (stocks with options trading)
NSE_FNO_STOCKS = [
    # Major Indices
    'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY',

    # Large Cap Stocks
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN',
    'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'TITAN',
    'SUNPHARMA', 'ULTRACEMCO', 'BAJFINANCE', 'WIPRO', 'NESTLEIND', 'POWERGRID',
    'HCLTECH', 'M&M', 'NTPC', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'INDUSINDBK',
    'ADANIENT', 'JSWSTEEL', 'BAJAJFINSV', 'HDFCLIFE', 'COALINDIA', 'GRASIM',
    'ONGC', 'CIPLA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'SHREECEM', 'HINDALCO',
    'BRITANNIA', 'BPCL', 'APOLLOHOSP', 'TATACONSUM', 'HEROMOTOCO', 'ADANIPORTS',

    # Mid Cap Stocks with high liquidity
    'VEDL', 'GODREJCP', 'PNB', 'BANKBARODA', 'CANBK', 'INDIGO', 'DLF', 'PEL',
    'SIEMENS', 'DABUR', 'GAIL', 'BAJAJ-AUTO', 'LUPIN', 'PIDILITIND', 'TORNTPHARM',
    'OFSS', 'BERGEPAINT', 'TATAPOWER', 'NMDC', 'SAIL', 'ZEEL', 'CONCOR', 'MRF',
    'ASHOKLEY', 'BANDHANBNK', 'CHOLAFIN', 'PFC', 'RECLTD', 'LICHSGFIN', 'MUTHOOTFIN',
    'IDFCFIRSTB', 'AUBANK', 'FEDERALBNK', 'INDUSTOWER', 'PAGEIND', 'DIXON',
    'ABCAPITAL', 'ASTRAL', 'ATUL', 'BALRAMCHIN', 'BEL', 'BIOCON', 'BOSCHLTD',
    'CUMMINSIND', 'ESCORTS', 'EXIDEIND', 'GMRINFRA', 'HAVELLS', 'IDEA', 'IRCTC',
    'JINDALSTEL', 'JUBLFOOD', 'L&TFH', 'LTTS', 'MANAPPURAM', 'MARICO', 'MFSL',
    'NAUKRI', 'PERSISTENT', 'PIIND', 'POLYCAB', 'PVR', 'RBLBANK', 'SRF',
    'SRTRANSFIN', 'VOLTAS', 'WHIRLPOOL', 'ABFRL', 'ACC', 'AMBUJACEM', 'APLLTD',
]

# Symbol mapping for Yahoo Finance
SYMBOL_MAPPING = {
    'NIFTY': '^NSEI',
    'BANKNIFTY': '^NSEBANK',
    'FINNIFTY': '^CNXFIN',
    'MIDCPNIFTY': '^NSEMDCP50',
}


class NSEStockScreener:
    """Comprehensive NSE Stock Screener"""

    def __init__(self):
        self.chart_analyzer = AdvancedChartAnalysis()
        self.bias_analyzer = BiasAnalysisPro()
        self.ml_regime_detector = MLMarketRegimeDetector()
        self.results = []

    def get_yf_symbol(self, stock: str) -> str:
        """Convert NSE stock symbol to Yahoo Finance symbol"""
        if stock in SYMBOL_MAPPING:
            return SYMBOL_MAPPING[stock]
        return f"{stock}.NS"

    def fetch_stock_data(self, stock: str, period: str = '1d', interval: str = '5m') -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        try:
            yf_symbol = self.get_yf_symbol(stock)
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)

            if df is None or df.empty:
                return None

            # Rename columns to match expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            return df
        except Exception as e:
            return None

    def calculate_option_chain_score(self, stock: str, df: pd.DataFrame) -> Dict:
        """
        Calculate option chain score based on price action
        (Simplified version - full option chain requires API access)
        """
        try:
            if df is None or len(df) < 20:
                return {'score': 0, 'signal': 'NEUTRAL', 'strength': 0}

            # Calculate price momentum
            current_price = df['close'].iloc[-1]
            price_1h_ago = df['close'].iloc[-12] if len(df) >= 12 else df['close'].iloc[0]
            price_change_pct = ((current_price - price_1h_ago) / price_1h_ago) * 100

            # Calculate volume surge
            avg_volume = df['volume'].tail(20).mean()
            current_volume = df['volume'].tail(5).mean()
            volume_surge = (current_volume / avg_volume) if avg_volume > 0 else 1

            # Determine signal strength
            if price_change_pct > 2 and volume_surge > 1.5:
                signal = 'STRONG_BULLISH'
                score = min(100, abs(price_change_pct) * 10)
            elif price_change_pct > 1:
                signal = 'BULLISH'
                score = min(100, abs(price_change_pct) * 8)
            elif price_change_pct < -2 and volume_surge > 1.5:
                signal = 'STRONG_BEARISH'
                score = min(100, abs(price_change_pct) * 10)
            elif price_change_pct < -1:
                signal = 'BEARISH'
                score = min(100, abs(price_change_pct) * 8)
            else:
                signal = 'NEUTRAL'
                score = 0

            return {
                'score': score,
                'signal': signal,
                'price_change_pct': price_change_pct,
                'volume_surge': volume_surge,
                'strength': abs(price_change_pct)
            }
        except Exception as e:
            return {'score': 0, 'signal': 'NEUTRAL', 'strength': 0}

    def calculate_bias_score(self, stock: str, df: pd.DataFrame) -> Dict:
        """Calculate bias score using BiasAnalysisPro logic"""
        try:
            if df is None or len(df) < 50:
                return {'score': 0, 'bias': 'NEUTRAL', 'strength': 0}

            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50

            # Calculate MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9, adjust=False).mean()
            macd_current = macd.iloc[-1] if len(macd) > 0 else 0
            signal_current = signal_line.iloc[-1] if len(signal_line) > 0 else 0

            # Calculate Volume Delta
            volume_delta = 0
            for i in range(min(20, len(df))):
                if df['close'].iloc[-i-1] > df['open'].iloc[-i-1]:
                    volume_delta += df['volume'].iloc[-i-1]
                else:
                    volume_delta -= df['volume'].iloc[-i-1]

            # Determine bias
            bullish_signals = 0
            bearish_signals = 0

            if current_rsi > 60:
                bullish_signals += 1
            elif current_rsi < 40:
                bearish_signals += 1

            if macd_current > signal_current:
                bullish_signals += 1
            else:
                bearish_signals += 1

            if volume_delta > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1

            # Calculate score
            if bullish_signals > bearish_signals:
                bias = 'BULLISH'
                score = (bullish_signals / 3) * 100
            elif bearish_signals > bullish_signals:
                bias = 'BEARISH'
                score = (bearish_signals / 3) * 100
            else:
                bias = 'NEUTRAL'
                score = 50

            return {
                'score': score,
                'bias': bias,
                'rsi': current_rsi,
                'strength': abs(score - 50)
            }
        except Exception as e:
            return {'score': 0, 'bias': 'NEUTRAL', 'strength': 0}

    def calculate_chart_analysis_score(self, stock: str, df: pd.DataFrame) -> Dict:
        """Calculate chart analysis score with market regime"""
        try:
            if df is None or len(df) < 50:
                return {'score': 0, 'regime': 'UNKNOWN', 'trend': 'NEUTRAL', 'strength': 0}

            # Add indicators
            df_with_indicators = self.chart_analyzer.add_indicators(df.copy())

            # Calculate trend strength
            if 'ATR' in df_with_indicators.columns:
                atr = df_with_indicators['ATR'].iloc[-1]
            else:
                atr = df_with_indicators['high'].tail(14).max() - df_with_indicators['low'].tail(14).min()

            # Calculate price position vs moving averages
            if len(df_with_indicators) >= 50:
                ma_20 = df_with_indicators['close'].rolling(20).mean().iloc[-1]
                ma_50 = df_with_indicators['close'].rolling(50).mean().iloc[-1]
                current_price = df_with_indicators['close'].iloc[-1]

                # Determine trend
                if current_price > ma_20 > ma_50:
                    trend = 'STRONG_BULLISH'
                    score = 80
                elif current_price > ma_20:
                    trend = 'BULLISH'
                    score = 65
                elif current_price < ma_20 < ma_50:
                    trend = 'STRONG_BEARISH'
                    score = 20
                elif current_price < ma_20:
                    trend = 'BEARISH'
                    score = 35
                else:
                    trend = 'NEUTRAL'
                    score = 50
            else:
                trend = 'NEUTRAL'
                score = 50

            # Try to detect market regime using ML
            try:
                regime_result = self.ml_regime_detector.predict_regime(df_with_indicators)
                regime = regime_result.get('regime', 'UNKNOWN')
                regime_confidence = regime_result.get('confidence', 0)
            except:
                regime = 'RANGE_BOUND'
                regime_confidence = 0.5

            return {
                'score': score,
                'regime': regime,
                'trend': trend,
                'strength': abs(score - 50),
                'regime_confidence': regime_confidence
            }
        except Exception as e:
            return {'score': 0, 'regime': 'UNKNOWN', 'trend': 'NEUTRAL', 'strength': 0}

    def analyze_stock(self, stock: str) -> Dict:
        """Comprehensive analysis of a single stock"""
        try:
            # Fetch data
            df = self.fetch_stock_data(stock, period='5d', interval='5m')

            if df is None or len(df) < 20:
                return None

            # Get current price
            current_price = df['close'].iloc[-1]

            # Perform all analyses
            option_score = self.calculate_option_chain_score(stock, df)
            bias_score = self.calculate_bias_score(stock, df)
            chart_score = self.calculate_chart_analysis_score(stock, df)

            # Calculate composite score
            composite_score = (
                option_score['score'] * 0.4 +
                bias_score['score'] * 0.3 +
                chart_score['score'] * 0.3
            )

            # Calculate overall strength (how much it moved)
            strength = (
                option_score['strength'] * 0.4 +
                bias_score['strength'] * 0.3 +
                chart_score['strength'] * 0.3
            )

            # Determine overall signal
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

            result = {
                'stock': stock,
                'price': current_price,
                'composite_score': composite_score,
                'strength': strength,
                'overall_signal': overall_signal,
                'option_signal': option_score['signal'],
                'bias': bias_score['bias'],
                'trend': chart_score['trend'],
                'regime': chart_score['regime'],
                'price_change_pct': option_score.get('price_change_pct', 0),
                'rsi': bias_score.get('rsi', 50),
                'volume_surge': option_score.get('volume_surge', 1)
            }

            return result

        except Exception as e:
            return None

    def analyze_all_stocks(self, stocks: List[str] = None, progress_callback=None) -> List[Dict]:
        """Analyze all stocks in parallel"""
        if stocks is None:
            stocks = NSE_FNO_STOCKS

        results = []
        total = len(stocks)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.analyze_stock, stock): stock for stock in stocks}

            completed = 0
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    pass

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        return results

    def get_top_movers(self, results: List[Dict], n: int = 10) -> Tuple[List[Dict], List[Dict]]:
        """Get top N falling and rising stocks"""
        if not results:
            return [], []

        # Sort by strength (absolute movement)
        sorted_by_strength = sorted(results, key=lambda x: x['strength'], reverse=True)

        # Filter falling stocks (bearish signals)
        falling_stocks = [
            r for r in sorted_by_strength
            if r['overall_signal'] in ['STRONG_BEARISH', 'BEARISH'] and r['price_change_pct'] < 0
        ][:n]

        # Filter rising stocks (bullish signals)
        rising_stocks = [
            r for r in sorted_by_strength
            if r['overall_signal'] in ['STRONG_BULLISH', 'BULLISH'] and r['price_change_pct'] > 0
        ][:n]

        return falling_stocks, rising_stocks


def render_nse_stock_screener_tab():
    """Render the NSE Stock Screener tab in Streamlit"""
    st.header("🔍 NSE Stock Screener - On-Demand Analysis")

    st.markdown("""
    ### Comprehensive Stock Analysis

    This screener analyzes all NSE FNO stocks using:
    - **Option Chain Analysis** - Price momentum and volume analysis
    - **Bias Analysis Pro** - 13 bias indicators across fast, medium, and slow timeframes
    - **Advanced Chart Analysis** - Technical indicators with ML Market Regime Detection

    Click the button below to run the analysis and get top 10 falling and top 10 rising stocks.
    """)

    st.divider()

    # Analysis controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
            st.session_state.run_nse_screener = True

    with col2:
        num_stocks = st.number_input("Top N stocks", min_value=5, max_value=20, value=10, step=1)

    with col3:
        include_indices = st.checkbox("Include Indices", value=True)

    st.divider()

    # Run analysis
    if st.session_state.get('run_nse_screener', False):
        st.session_state.run_nse_screener = False  # Reset flag

        # Initialize screener
        screener = NSEStockScreener()

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(completed, total):
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f"Analyzing stocks... {completed}/{total} completed")

        # Filter stocks
        stocks_to_analyze = NSE_FNO_STOCKS.copy()
        if not include_indices:
            stocks_to_analyze = [s for s in stocks_to_analyze if s not in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']]

        # Run analysis
        with st.spinner(f"Analyzing {len(stocks_to_analyze)} stocks..."):
            results = screener.analyze_all_stocks(stocks_to_analyze, progress_callback=update_progress)

        progress_bar.empty()
        status_text.empty()

        # Get top movers
        falling_stocks, rising_stocks = screener.get_top_movers(results, n=num_stocks)

        # Display results
        st.success(f"✅ Analysis complete! Found {len(results)} stocks with valid data.")

        # Create two columns for falling and rising stocks
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
                        'RSI': f"{r['rsi']:.1f}",
                        'Bias': r['bias'],
                        'Trend': r['trend'],
                        'Regime': r['regime']
                    }
                    for r in falling_stocks
                ])

                st.dataframe(
                    falling_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Export option
                csv = falling_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Falling Stocks CSV",
                    data=csv,
                    file_name=f"falling_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No significant falling stocks found in current market conditions.")

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
                        'RSI': f"{r['rsi']:.1f}",
                        'Bias': r['bias'],
                        'Trend': r['trend'],
                        'Regime': r['regime']
                    }
                    for r in rising_stocks
                ])

                st.dataframe(
                    rising_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Export option
                csv = rising_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Rising Stocks CSV",
                    data=csv,
                    file_name=f"rising_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No significant rising stocks found in current market conditions.")

        st.divider()

        # Summary statistics
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

        # Store results in session state
        st.session_state.nse_screener_results = results
        st.session_state.nse_screener_timestamp = datetime.now()

    # Show last analysis time if available
    if 'nse_screener_timestamp' in st.session_state:
        st.caption(f"Last analysis: {st.session_state.nse_screener_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
