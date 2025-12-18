"""
Money Flow Profile Indicator
Converted from Pine Script v5 by LuxAlgo
License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/

This indicator displays volume/money flow profile and sentiment profile across price levels
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional


class MoneyFlowProfile:
    """
    Money Flow Profile Indicator

    Displays trading activity and sentiment distribution across price levels:
    - Volume/Money Flow Profile: Shows total trading activity at specific price levels
    - Sentiment Profile: Shows bullish/bearish dominance at each price level
    - Point of Control (POC): Price level with highest traded activity
    - Consolidation Zones: High volume areas (value areas)
    """

    def __init__(
        self,
        lookback: int = 200,
        num_rows: int = 10,
        profile_source: str = 'Volume',  # 'Volume' or 'Money Flow'
        show_volume_profile: bool = True,
        show_sentiment_profile: bool = True,
        sentiment_method: str = 'Bar Polarity',  # 'Bar Polarity' or 'Bar Buying/Selling Pressure'
        show_poc: str = 'Last(Zone)',  # 'Developing', 'Last(Line)', 'Last(Zone)', 'None'
        show_consolidation: bool = False,
        consolidation_threshold: float = 0.25,
        hv_threshold: float = 0.53,
        lv_threshold: float = 0.37,
        profile_width: float = 0.13,
        horizontal_offset: int = 13
    ):
        """
        Initialize Money Flow Profile

        Parameters:
        -----------
        lookback : int
            Lookback length / fixed range (10-1500)
        num_rows : int
            Number of price rows/bins for the profile (10-100)
        profile_source : str
            'Volume' or 'Money Flow' - what to measure
        show_volume_profile : bool
            Display volume/money flow profile
        show_sentiment_profile : bool
            Display sentiment profile (bull/bear bias)
        sentiment_method : str
            'Bar Polarity' or 'Bar Buying/Selling Pressure'
        show_poc : str
            POC display mode: 'Developing', 'Last(Line)', 'Last(Zone)', 'None'
        show_consolidation : bool
            Show consolidation zones (value areas)
        consolidation_threshold : float
            Consolidation threshold percentage (0-1)
        hv_threshold : float
            High volume threshold percentage (0.5-0.99)
        lv_threshold : float
            Low volume threshold percentage (0.1-0.4)
        profile_width : float
            Profile width as percentage (0.1-0.5)
        horizontal_offset : int
            Horizontal offset for profile display
        """
        self.lookback = max(10, min(1500, lookback))
        self.num_rows = max(10, min(100, num_rows))
        self.profile_source = profile_source
        self.show_volume_profile = show_volume_profile
        self.show_sentiment_profile = show_sentiment_profile
        self.sentiment_method = sentiment_method
        self.show_poc = show_poc
        self.show_consolidation = show_consolidation
        self.consolidation_threshold = consolidation_threshold
        self.hv_threshold = max(0.5, min(0.99, hv_threshold))
        self.lv_threshold = max(0.1, min(0.4, lv_threshold))
        self.profile_width = max(0.1, min(0.5, profile_width))
        self.horizontal_offset = horizontal_offset

        # Color scheme
        self.hv_color = 'rgba(255, 235, 59, 0.5)'  # High volume - yellow
        self.av_color = 'rgba(41, 98, 255, 0.5)'  # Average volume - blue
        self.lv_color = 'rgba(242, 54, 69, 0.5)'  # Low volume - red
        self.bullish_color = 'rgba(38, 166, 154, 0.5)'  # Bullish - teal
        self.bearish_color = 'rgba(239, 83, 80, 0.5)'  # Bearish - red
        self.poc_color = 'rgb(255, 235, 59)'
        self.consolidation_color = 'rgba(41, 98, 255, 0.27)'
        self.bg_color = 'rgba(0, 188, 212, 0.05)'

    def calculate(self, df: pd.DataFrame) -> Dict:
        """
        Calculate money flow profile

        Parameters:
        -----------
        df : DataFrame
            OHLCV data with DatetimeIndex

        Returns:
        --------
        dict with profile data for visualization
        """
        if df.empty or len(df) < self.num_rows:
            return {'success': False, 'error': 'Insufficient data'}

        # Use last N bars
        lookback_bars = min(self.lookback, len(df))
        df_period = df.tail(lookback_bars).copy()

        # Get high and low for the period
        period_high = df_period['high'].max()
        period_low = df_period['low'].min()

        if period_high == period_low:
            return {'success': False, 'error': 'No price range'}

        # Calculate price step
        price_step = (period_high - period_low) / self.num_rows

        # Initialize volume arrays
        volume_total = np.zeros(self.num_rows)
        volume_bullish = np.zeros(self.num_rows)

        # Distribute volume across price levels
        for idx, row in df_period.iterrows():
            bar_high = row['high']
            bar_low = row['low']
            bar_close = row['close']
            bar_open = row['open']
            bar_volume = row['volume']

            # Determine if bullish based on sentiment method
            if self.sentiment_method == 'Bar Polarity':
                is_bullish = bar_close > bar_open
            else:  # Bar Buying/Selling Pressure
                is_bullish = (bar_close - bar_low) > (bar_high - bar_close)

            # Distribute volume across bins
            for i in range(self.num_rows):
                bin_low = period_low + i * price_step
                bin_high = bin_low + price_step

                # Check if bar overlaps with this price level
                if bar_high >= bin_low and bar_low < bin_high:
                    # Calculate volume portion for this bin
                    if bar_low >= bin_low and bar_high > bin_high:
                        # Bar starts in bin, extends above
                        volume_portion = (bin_high - bar_low) / (bar_high - bar_low) if (bar_high - bar_low) > 0 else 1
                    elif bar_high <= bin_high and bar_low < bin_low:
                        # Bar extends below bin, ends in bin
                        volume_portion = (bar_high - bin_low) / (bar_high - bar_low) if (bar_high - bar_low) > 0 else 1
                    elif bar_low >= bin_low and bar_high <= bin_high:
                        # Bar completely within bin
                        volume_portion = 1
                    else:
                        # Bar spans entire bin
                        volume_portion = price_step / (bar_high - bar_low) if (bar_high - bar_low) > 0 else 1

                    # Apply money flow weighting if needed
                    if self.profile_source == 'Money Flow':
                        # Weight by price (money flow = volume × price)
                        bin_mid_price = bin_low + price_step / 2
                        weighted_volume = bar_volume * volume_portion * bin_mid_price
                    else:
                        weighted_volume = bar_volume * volume_portion

                    volume_total[i] += weighted_volume

                    if is_bullish:
                        volume_bullish[i] += weighted_volume

        # Calculate sentiment delta
        volume_delta = np.zeros(self.num_rows)
        for i in range(self.num_rows):
            bearish_volume = volume_total[i] - volume_bullish[i]
            delta = 2 * volume_bullish[i] - volume_total[i]
            volume_delta[i] = abs(delta)

        # Find POC (Point of Control) - level with highest volume
        if volume_total.max() > 0:
            poc_idx = np.argmax(volume_total)
            poc_price = period_low + (poc_idx + 0.5) * price_step
        else:
            poc_idx = self.num_rows // 2
            poc_price = (period_high + period_low) / 2

        # Find consolidation zones (high volume areas)
        consolidation_levels = []
        if self.show_consolidation:
            max_volume = volume_total.max()
            for i in range(self.num_rows):
                normalized_volume = volume_total[i] / max_volume if max_volume > 0 else 0
                if normalized_volume > self.consolidation_threshold and normalized_volume < 1:
                    consolidation_levels.append({
                        'lower': period_low + i * price_step,
                        'upper': period_low + (i + 1) * price_step,
                        'volume': volume_total[i]
                    })

        # Prepare bin data
        bins_data = []
        max_volume = volume_total.max() if volume_total.max() > 0 else 1
        max_delta = volume_delta.max() if volume_delta.max() > 0 else 1

        for i in range(self.num_rows):
            bin_lower = period_low + i * price_step
            bin_upper = bin_lower + price_step
            bin_mid = (bin_lower + bin_upper) / 2

            normalized_volume = volume_total[i] / max_volume
            normalized_delta = volume_delta[i] / max_delta

            # Determine color based on volume threshold
            if normalized_volume > self.hv_threshold:
                volume_color = self.hv_color
            elif normalized_volume < self.lv_threshold:
                volume_color = self.lv_color
            else:
                volume_color = self.av_color

            # Determine sentiment color
            bullish_vol = volume_bullish[i]
            bearish_vol = volume_total[i] - volume_bullish[i]
            sentiment_color = self.bullish_color if bullish_vol > bearish_vol else self.bearish_color

            bins_data.append({
                'lower': bin_lower,
                'upper': bin_upper,
                'mid': bin_mid,
                'volume': volume_total[i],
                'volume_bullish': volume_bullish[i],
                'volume_bearish': bearish_vol,
                'delta': volume_delta[i],
                'normalized_volume': normalized_volume,
                'normalized_delta': normalized_delta,
                'volume_color': volume_color,
                'sentiment_color': sentiment_color,
                'is_poc': (i == poc_idx)
            })

        return {
            'success': True,
            'period_high': period_high,
            'period_low': period_low,
            'poc_price': poc_price,
            'poc_idx': poc_idx,
            'bins': bins_data,
            'consolidation_zones': consolidation_levels,
            'total_volume': volume_total.sum(),
            'total_bullish': volume_bullish.sum(),
            'total_bearish': volume_total.sum() - volume_bullish.sum(),
            'num_bars': lookback_bars
        }

    def get_signals(self, df: pd.DataFrame) -> Dict:
        """
        Get trading signals from money flow profile

        Returns:
        --------
        dict with signal information
        """
        profile_data = self.calculate(df)

        if not profile_data.get('success'):
            return {'success': False, 'error': profile_data.get('error')}

        poc_price = profile_data['poc_price']
        period_high = profile_data['period_high']
        period_low = profile_data['period_low']
        total_bullish = profile_data['total_bullish']
        total_volume = profile_data['total_volume']

        # Calculate sentiment
        if total_volume == 0:
            sentiment = "NEUTRAL"
            bullish_pct = 0
        else:
            bullish_pct = (total_bullish / total_volume) * 100
            bearish_pct = 100 - bullish_pct

            if bullish_pct > 60:
                sentiment = "BULLISH"
            elif bullish_pct < 40:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"

        # Find high and low volume levels
        bins = profile_data['bins']
        hv_levels = [b['mid'] for b in bins if b['normalized_volume'] > self.hv_threshold]
        lv_levels = [b['mid'] for b in bins if b['normalized_volume'] < self.lv_threshold]

        current_price = df['close'].iloc[-1]

        # Determine price position relative to POC
        if current_price > poc_price * 1.01:
            price_position = "Above POC"
        elif current_price < poc_price * 0.99:
            price_position = "Below POC"
        else:
            price_position = "At POC"

        return {
            'success': True,
            'sentiment': sentiment,
            'poc_price': poc_price,
            'profile_high': period_high,
            'profile_low': period_low,
            'high_volume_levels': hv_levels,
            'low_volume_levels': lv_levels,
            'bullish_volume_pct': bullish_pct,
            'bearish_volume_pct': 100 - bullish_pct,
            'current_price': current_price,
            'distance_from_poc': current_price - poc_price,
            'distance_from_poc_pct': ((current_price - poc_price) / poc_price * 100) if poc_price > 0 else 0,
            'price_position': price_position,
            'consolidation_zones': profile_data['consolidation_zones']
        }

    def format_report(self, signals: Dict) -> str:
        """
        Format money flow profile signals as readable report

        Parameters:
        -----------
        signals : dict
            Signal data from get_signals()

        Returns:
        --------
        str : Formatted report
        """
        if not signals.get('success'):
            return f"Error: {signals.get('error', 'Unknown error')}"

        report = f"""
╔════════════════════════════════════════════════════════╗
║        MONEY FLOW PROFILE ANALYSIS                     ║
╚════════════════════════════════════════════════════════╝

📊 SENTIMENT: {signals['sentiment']}
💰 Bullish Volume: {signals['bullish_volume_pct']:.1f}%
📉 Bearish Volume: {signals['bearish_volume_pct']:.1f}%

─────────────────────────────────────────────────────────
🎯 POINT OF CONTROL (POC)
  • POC Price: {signals['poc_price']:.2f}
  • Current Price: {signals['current_price']:.2f}
  • Position: {signals['price_position']}
  • Distance: {signals['distance_from_poc']:+.2f} ({signals['distance_from_poc_pct']:+.2f}%)

─────────────────────────────────────────────────────────
📈 PRICE RANGE
  • Profile High: {signals['profile_high']:.2f}
  • Profile Low: {signals['profile_low']:.2f}
  • Range: {signals['profile_high'] - signals['profile_low']:.2f}

─────────────────────────────────────────────────────────
🔥 HIGH VOLUME LEVELS (Consolidation/Value Areas)
"""
        if signals['high_volume_levels']:
            for level in signals['high_volume_levels']:
                report += f"  • {level:.2f}\n"
        else:
            report += "  • None detected\n"

        report += """
─────────────────────────────────────────────────────────
⚡ LOW VOLUME LEVELS (Supply/Demand Zones)
"""
        if signals['low_volume_levels']:
            for level in signals['low_volume_levels']:
                report += f"  • {level:.2f}\n"
        else:
            report += "  • None detected\n"

        if signals.get('consolidation_zones'):
            report += """
─────────────────────────────────────────────────────────
🎯 CONSOLIDATION ZONES
"""
            for zone in signals['consolidation_zones']:
                report += f"  • {zone['lower']:.2f} - {zone['upper']:.2f}\n"

        return report
