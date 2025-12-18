"""
Enhanced Market Data Fetcher
=============================

Fetches comprehensive market data from multiple sources:
1. Dhan API: India VIX, Sector Indices, Futures Data
2. Yahoo Finance: Global Markets, Intermarket Data
3. NSE: FII/DII Data (optional)

All data is fetched efficiently and displayed in tabulated format
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yfinance as yf
import warnings
import pytz
from config import IST, get_current_time_ist
warnings.filterwarnings('ignore')

# Try to import Dhan API
try:
    from dhan_data_fetcher import DhanDataFetcher, SECURITY_IDS
    DHAN_AVAILABLE = True
except ImportError:
    DHAN_AVAILABLE = False


class EnhancedMarketData:
    """
    Comprehensive market data fetcher from multiple sources
    """

    def __init__(self):
        """Initialize enhanced market data fetcher"""
        self.dhan_fetcher = None
        if DHAN_AVAILABLE:
            try:
                self.dhan_fetcher = DhanDataFetcher()
            except Exception as e:
                print(f"Dhan API not available: {e}")

    # =========================================================================
    # DHAN API DATA FETCHING
    # =========================================================================

    def fetch_india_vix(self) -> Dict[str, Any]:
        """
        Fetch India VIX from Yahoo Finance (Primary Source)

        Returns:
            Dict with VIX data and sentiment
        """
        # Use Yahoo Finance as primary source
        return self._fetch_india_vix_yfinance()

    def _fetch_india_vix_yfinance(self) -> Dict[str, Any]:
        """Fetch India VIX from Yahoo Finance (^INDIAVIX)"""
        try:
            ticker = yf.Ticker("^INDIAVIX")
            hist = ticker.history(period="1d", interval="1m")

            if not hist.empty:
                vix_value = hist['Close'].iloc[-1]

                # VIX Interpretation
                if vix_value > 25:
                    vix_sentiment = "HIGH FEAR"
                    vix_bias = "BEARISH"
                    vix_score = -75
                elif vix_value > 20:
                    vix_sentiment = "ELEVATED FEAR"
                    vix_bias = "BEARISH"
                    vix_score = -50
                elif vix_value > 15:
                    vix_sentiment = "MODERATE"
                    vix_bias = "NEUTRAL"
                    vix_score = 0
                elif vix_value > 12:
                    vix_sentiment = "LOW VOLATILITY"
                    vix_bias = "BULLISH"
                    vix_score = 40
                else:
                    vix_sentiment = "COMPLACENCY"
                    vix_bias = "NEUTRAL"
                    vix_score = 0

                return {
                    'success': True,
                    'source': 'Yahoo Finance',
                    'value': vix_value,
                    'sentiment': vix_sentiment,
                    'bias': vix_bias,
                    'score': vix_score,
                    'timestamp': get_current_time_ist()
                }
        except Exception as e:
            pass

        return {'success': False, 'error': 'India VIX data not available'}

    def fetch_sector_indices(self) -> List[Dict[str, Any]]:
        """
        Fetch all sector indices from Dhan API

        Returns:
            List of sector data with performance and bias
        """
        sectors = ['BANKNIFTY', 'NIFTY_IT', 'NIFTY_AUTO', 'NIFTY_PHARMA', 'NIFTY_METAL',
                   'NIFTY_REALTY', 'NIFTY_ENERGY', 'NIFTY_FMCG']

        sector_data = []

        if self.dhan_fetcher:
            try:
                # Fetch all sectors in one call
                data = self.dhan_fetcher.fetch_ohlc_data(sectors)

                for sector in sectors:
                    if data.get(sector, {}).get('success'):
                        sector_info = data[sector]

                        last_price = sector_info.get('last_price', 0)
                        open_price = sector_info.get('open', last_price)

                        # Calculate change %
                        if open_price > 0:
                            change_pct = ((last_price - open_price) / open_price) * 100
                        else:
                            change_pct = 0

                        # Determine bias
                        if change_pct > 1.5:
                            bias = "STRONG BULLISH"
                            score = 75
                        elif change_pct > 0.5:
                            bias = "BULLISH"
                            score = 50
                        elif change_pct < -1.5:
                            bias = "STRONG BEARISH"
                            score = -75
                        elif change_pct < -0.5:
                            bias = "BEARISH"
                            score = -50
                        else:
                            bias = "NEUTRAL"
                            score = 0

                        # Format sector name for display
                        display_name = sector
                        if sector == 'BANKNIFTY':
                            display_name = 'BANK NIFTY'
                        else:
                            display_name = sector.replace('NIFTY_', 'NIFTY ')

                        sector_data.append({
                            'sector': display_name,
                            'last_price': last_price,
                            'open': open_price,
                            'high': sector_info.get('high', 0),
                            'low': sector_info.get('low', 0),
                            'change_pct': change_pct,
                            'bias': bias,
                            'score': score,
                            'source': 'Dhan API'
                        })
            except Exception as e:
                print(f"Dhan sector fetch error: {e}")

        # Fallback to Yahoo Finance if Dhan failed
        if not sector_data:
            sector_data = self._fetch_sector_indices_yfinance()

        return sector_data

    def _fetch_sector_indices_yfinance(self) -> List[Dict[str, Any]]:
        """Fallback: Fetch sector indices from Yahoo Finance"""
        sectors_map = {
            '^NSEBANK': 'BANK NIFTY',
            '^CNXIT': 'NIFTY IT',
            '^CNXAUTO': 'NIFTY AUTO',
            '^CNXPHARMA': 'NIFTY PHARMA',
            '^CNXMETAL': 'NIFTY METAL',
            '^CNXREALTY': 'NIFTY REALTY',
            '^CNXFMCG': 'NIFTY FMCG'
        }

        sector_data = []

        for symbol, name in sectors_map.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="1m")

                if not hist.empty:
                    last_price = hist['Close'].iloc[-1]
                    open_price = hist['Open'].iloc[0]
                    high_price = hist['High'].max()
                    low_price = hist['Low'].min()

                    change_pct = ((last_price - open_price) / open_price) * 100

                    # Determine bias
                    if change_pct > 1.5:
                        bias = "STRONG BULLISH"
                        score = 75
                    elif change_pct > 0.5:
                        bias = "BULLISH"
                        score = 50
                    elif change_pct < -1.5:
                        bias = "STRONG BEARISH"
                        score = -75
                    elif change_pct < -0.5:
                        bias = "BEARISH"
                        score = -50
                    else:
                        bias = "NEUTRAL"
                        score = 0

                    sector_data.append({
                        'sector': name,
                        'last_price': last_price,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score,
                        'source': 'Yahoo Finance'
                    })
            except Exception as e:
                print(f"Error fetching {name}: {e}")

        return sector_data

    # =========================================================================
    # YAHOO FINANCE DATA FETCHING
    # =========================================================================

    def fetch_global_markets(self) -> List[Dict[str, Any]]:
        """
        Fetch global market indices from Yahoo Finance

        Returns:
            List of global market data with bias
        """
        global_markets = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'DOW JONES',
            '^N225': 'NIKKEI 225',
            '^HSI': 'HANG SENG',
            '^FTSE': 'FTSE 100',
            '^GDAXI': 'DAX',
            '000001.SS': 'SHANGHAI'
        }

        market_data = []

        for symbol, name in global_markets.items():
            try:
                ticker = yf.Ticker(symbol)
                # Get last 2 days to calculate change
                hist = ticker.history(period="2d")

                if len(hist) >= 2:
                    current_close = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]

                    change_pct = ((current_close - prev_close) / prev_close) * 100

                    # Determine bias
                    if change_pct > 1.5:
                        bias = "STRONG BULLISH"
                        score = 75
                    elif change_pct > 0.5:
                        bias = "BULLISH"
                        score = 50
                    elif change_pct < -1.5:
                        bias = "STRONG BEARISH"
                        score = -75
                    elif change_pct < -0.5:
                        bias = "BEARISH"
                        score = -50
                    else:
                        bias = "NEUTRAL"
                        score = 0

                    market_data.append({
                        'market': name,
                        'symbol': symbol,
                        'last_price': current_close,
                        'prev_close': prev_close,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score
                    })
            except Exception as e:
                print(f"Error fetching {name}: {e}")

        return market_data

    def fetch_intermarket_data(self) -> List[Dict[str, Any]]:
        """
        Fetch intermarket data (commodities, currencies, bonds)

        Returns:
            List of intermarket data with bias
        """
        intermarket_assets = {
            'DX-Y.NYB': 'US DOLLAR INDEX',
            'CL=F': 'CRUDE OIL',
            'GC=F': 'GOLD',
            'INR=X': 'USD/INR',
            '^TNX': 'US 10Y TREASURY',
            'BTC-USD': 'BITCOIN'
        }

        intermarket_data = []

        for symbol, name in intermarket_assets.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")

                if len(hist) >= 2:
                    current_close = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]

                    change_pct = ((current_close - prev_close) / prev_close) * 100

                    # Specific interpretations for each asset
                    if 'DOLLAR' in name:
                        # Strong dollar = bearish for emerging markets
                        if change_pct > 0.5:
                            bias = "BEARISH (for India)"
                            score = -40
                        elif change_pct < -0.5:
                            bias = "BULLISH (for India)"
                            score = 40
                        else:
                            bias = "NEUTRAL"
                            score = 0
                    elif 'OIL' in name:
                        # High oil = bearish for India (import dependent)
                        if change_pct > 2:
                            bias = "BEARISH (for India)"
                            score = -50
                        elif change_pct < -2:
                            bias = "BULLISH (for India)"
                            score = 50
                        else:
                            bias = "NEUTRAL"
                            score = 0
                    elif 'GOLD' in name:
                        # Gold up = risk-off sentiment
                        if change_pct > 1:
                            bias = "RISK OFF"
                            score = -40
                        elif change_pct < -1:
                            bias = "RISK ON"
                            score = 40
                        else:
                            bias = "NEUTRAL"
                            score = 0
                    elif 'INR' in name:
                        # USD/INR up = INR weakening = bearish
                        if change_pct > 0.5:
                            bias = "BEARISH (INR Weak)"
                            score = -40
                        elif change_pct < -0.5:
                            bias = "BULLISH (INR Strong)"
                            score = 40
                        else:
                            bias = "NEUTRAL"
                            score = 0
                    elif 'TREASURY' in name:
                        # Yields up = risk-off
                        if change_pct > 2:
                            bias = "RISK OFF"
                            score = -40
                        elif change_pct < -2:
                            bias = "RISK ON"
                            score = 40
                        else:
                            bias = "NEUTRAL"
                            score = 0
                    else:
                        # Generic bias
                        if change_pct > 1:
                            bias = "BULLISH"
                            score = 40
                        elif change_pct < -1:
                            bias = "BEARISH"
                            score = -40
                        else:
                            bias = "NEUTRAL"
                            score = 0

                    intermarket_data.append({
                        'asset': name,
                        'symbol': symbol,
                        'last_price': current_close,
                        'prev_close': prev_close,
                        'change_pct': change_pct,
                        'bias': bias,
                        'score': score
                    })
            except Exception as e:
                print(f"Error fetching {name}: {e}")

        return intermarket_data

    # =========================================================================
    # COMPREHENSIVE DATA FETCH
    # =========================================================================

    def fetch_all_enhanced_data(self) -> Dict[str, Any]:
        """
        Fetch all enhanced market data from all sources

        Returns:
            Dict containing all market data organized by category
        """
        print("Fetching enhanced market data...")

        result = {
            'timestamp': get_current_time_ist(),
            'india_vix': {},
            'sector_indices': [],
            'global_markets': [],
            'intermarket': [],
            'gamma_squeeze': {},
            'sector_rotation': {},
            'intraday_seasonality': {},
            'summary': {}
        }

        # 1. Fetch India VIX
        print("  - Fetching India VIX...")
        result['india_vix'] = self.fetch_india_vix()

        # 2. Fetch Sector Indices
        print("  - Fetching sector indices...")
        result['sector_indices'] = self.fetch_sector_indices()

        # 3. Fetch Global Markets
        print("  - Fetching global markets...")
        result['global_markets'] = self.fetch_global_markets()

        # 4. Fetch Intermarket Data
        print("  - Fetching intermarket data...")
        result['intermarket'] = self.fetch_intermarket_data()

        # 5. Detect Gamma Squeeze
        print("  - Analyzing Gamma Squeeze...")
        result['gamma_squeeze'] = self.detect_gamma_squeeze('NIFTY')

        # 6. Analyze Sector Rotation
        print("  - Analyzing Sector Rotation...")
        result['sector_rotation'] = self.analyze_sector_rotation()

        # 7. Analyze Intraday Seasonality
        print("  - Analyzing Intraday Seasonality...")
        result['intraday_seasonality'] = self.analyze_intraday_seasonality()

        # 8. Calculate summary statistics
        result['summary'] = self._calculate_summary(result)

        print("✓ Enhanced market data fetch completed!")

        return result

    def _calculate_summary(self, data: Dict) -> Dict[str, Any]:
        """Calculate summary statistics from all data"""
        summary = {
            'total_data_points': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'avg_score': 0,
            'overall_sentiment': 'NEUTRAL'
        }

        all_scores = []

        # Count India VIX
        if data['india_vix'].get('success'):
            summary['total_data_points'] += 1
            all_scores.append(data['india_vix']['score'])
            bias = data['india_vix']['bias']
            if 'BULLISH' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        # Count sectors
        for sector in data['sector_indices']:
            summary['total_data_points'] += 1
            all_scores.append(sector['score'])
            bias = sector['bias']
            if 'BULLISH' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        # Count global markets
        for market in data['global_markets']:
            summary['total_data_points'] += 1
            all_scores.append(market['score'])
            bias = market['bias']
            if 'BULLISH' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        # Count intermarket
        for asset in data['intermarket']:
            summary['total_data_points'] += 1
            all_scores.append(asset['score'])
            bias = asset['bias']
            if 'BULLISH' in bias or 'RISK ON' in bias:
                summary['bullish_count'] += 1
            elif 'BEARISH' in bias or 'RISK OFF' in bias:
                summary['bearish_count'] += 1
            else:
                summary['neutral_count'] += 1

        # Calculate average score
        if all_scores:
            summary['avg_score'] = np.mean(all_scores)

            # Determine overall sentiment
            if summary['avg_score'] > 25:
                summary['overall_sentiment'] = 'BULLISH'
            elif summary['avg_score'] < -25:
                summary['overall_sentiment'] = 'BEARISH'
            else:
                summary['overall_sentiment'] = 'NEUTRAL'

        return summary

    # =========================================================================
    # GAMMA SQUEEZE DETECTION
    # =========================================================================

    def detect_gamma_squeeze(self, instrument: str = 'NIFTY') -> Dict[str, Any]:
        """
        Detect gamma squeeze potential from option chain data

        A gamma squeeze occurs when market makers need to hedge large gamma exposure,
        potentially causing rapid price movements.

        Args:
            instrument: Instrument name (NIFTY, BANKNIFTY, etc.)

        Returns:
            Dict with gamma squeeze analysis
        """
        import streamlit as st

        # Check if option chain data is available
        if 'overall_option_data' not in st.session_state:
            return {'success': False, 'error': 'Option chain data not available'}

        option_data = st.session_state.overall_option_data.get(instrument)
        if not option_data or not option_data.get('success'):
            return {'success': False, 'error': f'No option data for {instrument}'}

        try:
            spot = option_data.get('spot', 0)

            # Get total gamma exposure
            total_call_gamma = option_data.get('total_call_gamma', 0)
            total_put_gamma = option_data.get('total_put_gamma', 0)

            # Net gamma exposure
            net_gamma = total_put_gamma - total_call_gamma

            # Get ATM gamma concentration
            # Check session state for ATM zone data
            atm_key = f'{instrument}_atm_zone_bias'

            gamma_concentration = 0
            if atm_key in st.session_state:
                df_atm = st.session_state[atm_key]
                atm_row = df_atm[df_atm["Zone"] == "ATM"]
                if not atm_row.empty:
                    # Calculate gamma concentration at ATM
                    gamma_concentration = atm_row.iloc[0].get('Gamma_Exposure_Bias', 0)

            # Gamma squeeze risk levels
            if abs(net_gamma) > 1000000:  # High gamma exposure
                if net_gamma > 0:
                    squeeze_risk = "HIGH UPSIDE RISK"
                    squeeze_bias = "BULLISH GAMMA SQUEEZE"
                    squeeze_score = 80
                    interpretation = "Large positive gamma → MMs will buy on dips, sell on rallies (resistance to movement)"
                else:
                    squeeze_risk = "HIGH DOWNSIDE RISK"
                    squeeze_bias = "BEARISH GAMMA SQUEEZE"
                    squeeze_score = -80
                    interpretation = "Large negative gamma → MMs will sell on dips, buy on rallies (amplified movement)"
            elif abs(net_gamma) > 500000:
                if net_gamma > 0:
                    squeeze_risk = "MODERATE UPSIDE RISK"
                    squeeze_bias = "BULLISH"
                    squeeze_score = 50
                    interpretation = "Moderate positive gamma → Some resistance to downward movement"
                else:
                    squeeze_risk = "MODERATE DOWNSIDE RISK"
                    squeeze_bias = "BEARISH"
                    squeeze_score = -50
                    interpretation = "Moderate negative gamma → Some amplification of movement"
            else:
                squeeze_risk = "LOW"
                squeeze_bias = "NEUTRAL"
                squeeze_score = 0
                interpretation = "Low gamma exposure → Normal market conditions"

            return {
                'success': True,
                'instrument': instrument,
                'spot': spot,
                'total_call_gamma': total_call_gamma,
                'total_put_gamma': total_put_gamma,
                'net_gamma': net_gamma,
                'gamma_concentration': gamma_concentration,
                'squeeze_risk': squeeze_risk,
                'squeeze_bias': squeeze_bias,
                'squeeze_score': squeeze_score,
                'interpretation': interpretation,
                'timestamp': get_current_time_ist()
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # =========================================================================
    # SECTOR ROTATION MODEL
    # =========================================================================

    def analyze_sector_rotation(self) -> Dict[str, Any]:
        """
        Analyze sector rotation to identify market leadership changes

        Returns:
            Dict with sector rotation analysis
        """
        sectors = self.fetch_sector_indices()

        if not sectors:
            return {'success': False, 'error': 'No sector data available'}

        # Sort sectors by performance
        sectors_sorted = sorted(sectors, key=lambda x: x['change_pct'], reverse=True)

        # Identify leaders and laggards
        leaders = sectors_sorted[:3]  # Top 3 performing sectors
        laggards = sectors_sorted[-3:]  # Bottom 3 performing sectors

        # Calculate sector strength score
        # Lowered thresholds from 0.5% to 0.15% for more accurate sector classification
        # This ensures that sectors with small positive/negative changes are properly classified
        bullish_sectors = [s for s in sectors if s['change_pct'] >= 0.15]
        bearish_sectors = [s for s in sectors if s['change_pct'] <= -0.15]
        neutral_sectors = [s for s in sectors if -0.15 < s['change_pct'] < 0.15]

        # Market breadth from sectors
        if len(sectors) > 0:
            sector_breadth = (len(bullish_sectors) / len(sectors)) * 100
        else:
            sector_breadth = 50

        # Determine rotation pattern (for informational purposes only)
        if len(leaders) > 0 and leaders[0]['change_pct'] > 2:
            rotation_pattern = "STRONG ROTATION"
            if 'IT' in leaders[0]['sector'] or 'PHARMA' in leaders[0]['sector']:
                rotation_type = "DEFENSIVE ROTATION (Risk-off)"
            elif 'METAL' in leaders[0]['sector'] or 'ENERGY' in leaders[0]['sector']:
                rotation_type = "CYCLICAL ROTATION (Risk-on)"
            elif 'BANK' in leaders[0]['sector'] or 'AUTO' in leaders[0]['sector']:
                rotation_type = "GROWTH ROTATION (Risk-on)"
            else:
                rotation_type = "MIXED ROTATION"
        else:
            rotation_pattern = "NO CLEAR ROTATION"
            rotation_type = "CONSOLIDATION"

        # Overall sector sentiment (Based on Sector Breadth %)
        # Updated thresholds to align better with technical bias (60% threshold)
        if sector_breadth > 70:
            sector_sentiment = "STRONG BULLISH"
            sector_score = 75
        elif sector_breadth >= 60:
            sector_sentiment = "BULLISH"
            sector_score = 50
        elif sector_breadth < 30:
            sector_sentiment = "STRONG BEARISH"
            sector_score = -75
        elif sector_breadth <= 40:
            sector_sentiment = "BEARISH"
            sector_score = -50
        else:
            sector_sentiment = "NEUTRAL"
            sector_score = 0

        # IMPORTANT: Rotation Bias now ALWAYS matches Sector Breadth Sentiment
        # This ensures consistency between sector sentiment and rotation bias
        rotation_bias = sector_sentiment
        rotation_score = sector_score

        return {
            'success': True,
            'leaders': leaders,
            'laggards': laggards,
            'bullish_sectors_count': len(bullish_sectors),
            'bearish_sectors_count': len(bearish_sectors),
            'neutral_sectors_count': len(neutral_sectors),
            'sector_breadth': sector_breadth,
            'rotation_pattern': rotation_pattern,
            'rotation_type': rotation_type,
            'rotation_bias': rotation_bias,
            'rotation_score': rotation_score,
            'sector_sentiment': sector_sentiment,
            'sector_score': sector_score,
            'all_sectors': sectors,
            'timestamp': get_current_time_ist()
        }

    # =========================================================================
    # INTRADAY SEASONALITY
    # =========================================================================

    def analyze_intraday_seasonality(self) -> Dict[str, Any]:
        """
        Analyze intraday time-based patterns

        Common patterns:
        - Opening 15 minutes: High volatility
        - 10:00-11:00 AM: Post-opening trend
        - 11:00-14:30: Lunchtime lull
        - 14:30-15:30: Closing rally/selloff

        Returns:
            Dict with intraday seasonality analysis
        """
        now = get_current_time_ist()
        current_time = now.time()
        current_hour = now.hour
        current_minute = now.minute

        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()

        # Determine current market session
        if current_time < market_open:
            session = "PRE-MARKET"
            session_bias = "NEUTRAL"
            session_score = 0
            session_characteristics = "Low volume, wide spreads. Wait for market open."
            trading_recommendation = "AVOID - Wait for market open"
        elif current_time < datetime.strptime("09:30", "%H:%M").time():
            session = "OPENING RANGE (9:15-9:30)"
            session_bias = "HIGH VOLATILITY"
            session_score = 0
            session_characteristics = "High volatility, gap movements, institutional orders"
            trading_recommendation = "CAUTIOUS - Wait for range breakout or use tight stops"
        elif current_time < datetime.strptime("10:00", "%H:%M").time():
            session = "POST-OPENING (9:30-10:00)"
            session_bias = "TREND FORMATION"
            session_score = 40
            session_characteristics = "Trend develops, direction becomes clear"
            trading_recommendation = "ACTIVE - Trade in direction of trend"
        elif current_time < datetime.strptime("11:30", "%H:%M").time():
            session = "MID-MORNING (10:00-11:30)"
            session_bias = "TRENDING"
            session_score = 50
            session_characteristics = "Best trending period, follow momentum"
            trading_recommendation = "VERY ACTIVE - Best time for trend following"
        elif current_time < datetime.strptime("14:30", "%H:%M").time():
            session = "LUNCHTIME (11:30-14:30)"
            session_bias = "CONSOLIDATION"
            session_score = -20
            session_characteristics = "Low volume, choppy, range-bound"
            trading_recommendation = "REDUCE ACTIVITY - Scalping only or stay out"
        elif current_time < datetime.strptime("15:15", "%H:%M").time():
            session = "AFTERNOON SESSION (14:30-15:15)"
            session_bias = "MOMENTUM"
            session_score = 45
            session_characteristics = "Volume picks up, trends resume"
            trading_recommendation = "ACTIVE - Trade breakouts and momentum"
        elif current_time < market_close:
            session = "CLOSING RANGE (15:15-15:30)"
            session_bias = "HIGH VOLATILITY"
            session_score = 0
            session_characteristics = "High volume, squaring off positions, volatile"
            trading_recommendation = "CAUTIOUS - Close positions or use wide stops"
        else:
            session = "POST-MARKET"
            session_bias = "NEUTRAL"
            session_score = 0
            session_characteristics = "Market closed"
            trading_recommendation = "NO TRADING - Market closed"

        # Day of week patterns
        weekday = now.strftime("%A")

        if weekday == "Monday":
            day_bias = "GAP TENDENCY"
            day_characteristics = "Weekend news gaps, follow-through from Friday"
        elif weekday == "Tuesday" or weekday == "Wednesday":
            day_bias = "TRENDING"
            day_characteristics = "Best trending days, institutional activity high"
        elif weekday == "Thursday":
            day_bias = "CONSOLIDATION"
            day_characteristics = "Pre-Friday profit booking, consolidation"
        elif weekday == "Friday":
            day_bias = "PROFIT BOOKING"
            day_characteristics = "Week-end squaring off, typically weak close"
        else:
            day_bias = "WEEKEND"
            day_characteristics = "Market closed"

        return {
            'success': True,
            'current_time': now.strftime("%H:%M:%S"),
            'session': session,
            'session_bias': session_bias,
            'session_score': session_score,
            'session_characteristics': session_characteristics,
            'trading_recommendation': trading_recommendation,
            'weekday': weekday,
            'day_bias': day_bias,
            'day_characteristics': day_characteristics,
            'timestamp': now
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_enhanced_market_data() -> Dict[str, Any]:
    """
    Convenience function to fetch all enhanced market data

    Returns:
        Dict with all enhanced market data
    """
    fetcher = EnhancedMarketData()
    return fetcher.fetch_all_enhanced_data()


def get_india_vix_only() -> Dict[str, Any]:
    """
    Convenience function to fetch only India VIX

    Returns:
        Dict with India VIX data
    """
    fetcher = EnhancedMarketData()
    return fetcher.fetch_india_vix()


def get_sector_rotation() -> List[Dict[str, Any]]:
    """
    Convenience function to fetch sector rotation data

    Returns:
        List of sector performance data
    """
    fetcher = EnhancedMarketData()
    return fetcher.fetch_sector_indices()
