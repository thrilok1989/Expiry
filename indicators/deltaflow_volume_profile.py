"""
DeltaFlow Volume Profile Indicator
Converted from Pine Script v6 by BigBeluga
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
https://creativecommons.org/licenses/by-nc-sa/4.0/

This indicator combines volume profile with delta analysis per price level
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional


class DeltaFlowVolumeProfile:
    """
    DeltaFlow Volume Profile Indicator

    Combines volume profile with delta analysis:
    - Volume distribution across price levels
    - Buy/sell volume separation per bin
    - Delta calculation per price level
    - Delta heatmap visualization
    - Point of Control (POC) tracking
    - Delta percentage per level
    """

    def __init__(
        self,
        lookback: int = 200,
        bins: int = 30,
        show_delta_heatmap: bool = True,
        show_delta_display: bool = True,
        show_volume_bars: bool = True,
        show_poc: bool = True,
        bull_color: str = 'rgba(0, 150, 136, 0.6)',
        bear_color: str = 'rgba(230, 150, 30, 0.6)',
        poc_color: str = 'rgb(0, 183, 255)',
        offset: int = 5
    ):
        """
        Initialize DeltaFlow Volume Profile

        Parameters:
        -----------
        lookback : int
            Lookback period for profile calculation
        bins : int
            Number of price bins (10-100)
        show_delta_heatmap : bool
            Display delta heatmap
        show_delta_display : bool
            Display delta percentage labels
        show_volume_bars : bool
            Display buy/sell volume bars
        show_poc : bool
            Display Point of Control
        bull_color : str
            Color for bullish volume
        bear_color : str
            Color for bearish volume
        poc_color : str
            Color for POC line
        offset : int
            Horizontal offset for profile display
        """
        self.lookback = lookback
        self.bins = max(10, min(100, bins))
        self.show_delta_heatmap = show_delta_heatmap
        self.show_delta_display = show_delta_display
        self.show_volume_bars = show_volume_bars
        self.show_poc = show_poc
        self.bull_color = bull_color
        self.bear_color = bear_color
        self.poc_color = poc_color
        self.offset = offset

    def calculate(self, df: pd.DataFrame) -> Dict:
        """
        Calculate DeltaFlow volume profile

        Parameters:
        -----------
        df : DataFrame
            OHLCV data with DatetimeIndex

        Returns:
        --------
        dict with profile data for visualization
        """
        if df.empty or len(df) < self.bins:
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
        price_step = (period_high - period_low) / self.bins

        # Initialize arrays
        volume_total = np.zeros(self.bins)
        volume_bullish = np.zeros(self.bins)
        volume_bearish = np.zeros(self.bins)

        # Distribute volume across price bins
        for idx, row in df_period.iterrows():
            bar_close = row['close']
            bar_open = row['open']
            bar_volume = row['volume']

            is_bearish = bar_close < bar_open

            # Find which bin this price belongs to
            for k in range(self.bins):
                bin_lower = period_low + price_step * k
                bin_mid = bin_lower + price_step / 2

                # Check if close price is within tolerance of this bin
                if abs(bin_mid - bar_close) <= price_step:
                    volume_total[k] += bar_volume

                    if is_bearish:
                        volume_bearish[k] += bar_volume
                    else:
                        volume_bullish[k] += bar_volume

        # Find POC (Point of Control) - bin with highest volume
        poc_idx = np.argmax(volume_total) if volume_total.max() > 0 else self.bins // 2
        poc_price = period_low + (poc_idx + 0.5) * price_step

        # Calculate normalized volumes and deltas
        max_volume = volume_total.max() if volume_total.max() > 0 else 1

        bins_data = []
        for i in range(self.bins):
            bin_lower = period_low + price_step * i
            bin_upper = bin_lower + price_step
            bin_mid = (bin_lower + bin_upper) / 2

            # Normalized volumes (0-100)
            normalized_volume = (volume_total[i] / max_volume) * 100
            normalized_bull = (volume_bullish[i] / max_volume) * 100
            normalized_bear = (volume_bearish[i] / max_volume) * 100

            # Calculate delta percentage
            if volume_total[i] > 0:
                bull_pct = (volume_bullish[i] / volume_total[i]) * 100
                bear_pct = (volume_bearish[i] / volume_total[i]) * 100
                delta_pct = bull_pct - bear_pct
            else:
                bull_pct = 0
                bear_pct = 0
                delta_pct = 0

            # Determine delta color
            delta_color = self._calculate_delta_color(delta_pct)

            bins_data.append({
                'lower': bin_lower,
                'upper': bin_upper,
                'mid': bin_mid,
                'volume_total': volume_total[i],
                'volume_bullish': volume_bullish[i],
                'volume_bearish': volume_bearish[i],
                'normalized_volume': normalized_volume,
                'normalized_bull': normalized_bull,
                'normalized_bear': normalized_bear,
                'bull_pct': bull_pct,
                'bear_pct': bear_pct,
                'delta_pct': delta_pct,
                'delta_color': delta_color,
                'is_poc': (i == poc_idx)
            })

        # Calculate overall statistics
        total_volume = volume_total.sum()
        total_bullish = volume_bullish.sum()
        total_bearish = volume_bearish.sum()

        if total_volume > 0:
            overall_bull_pct = (total_bullish / total_volume) * 100
            overall_bear_pct = (total_bearish / total_volume) * 100
            overall_delta = overall_bull_pct - overall_bear_pct
        else:
            overall_bull_pct = 0
            overall_bear_pct = 0
            overall_delta = 0

        return {
            'success': True,
            'period_high': period_high,
            'period_low': period_low,
            'poc_price': poc_price,
            'poc_idx': poc_idx,
            'bins': bins_data,
            'total_volume': total_volume,
            'total_bullish': total_bullish,
            'total_bearish': total_bearish,
            'overall_bull_pct': overall_bull_pct,
            'overall_bear_pct': overall_bear_pct,
            'overall_delta': overall_delta,
            'num_bars': lookback_bars
        }

    def _calculate_delta_color(self, delta_pct: float) -> str:
        """
        Calculate color based on delta percentage using gradient

        Parameters:
        -----------
        delta_pct : float
            Delta percentage (-100 to +100)

        Returns:
        --------
        str : RGBA color string
        """
        # Gradient from -30 to +30 (bearish to bullish)
        delta_normalized = np.clip(delta_pct, -30, 30)

        # Normalize to 0-1 range
        t = (delta_normalized + 30) / 60

        # Parse colors
        bear_rgb = (230, 150, 30)  # Orange
        bull_rgb = (0, 150, 136)   # Teal

        # Interpolate
        r = int(bear_rgb[0] + t * (bull_rgb[0] - bear_rgb[0]))
        g = int(bear_rgb[1] + t * (bull_rgb[1] - bear_rgb[1]))
        b = int(bear_rgb[2] + t * (bull_rgb[2] - bear_rgb[2]))

        return f'rgba({r}, {g}, {b}, 0.6)'

    def get_signals(self, df: pd.DataFrame) -> Dict:
        """
        Get trading signals from DeltaFlow profile

        Returns:
        --------
        dict with signal information
        """
        profile_data = self.calculate(df)

        if not profile_data.get('success'):
            return {'success': False, 'error': profile_data.get('error')}

        poc_price = profile_data['poc_price']
        overall_delta = profile_data['overall_delta']
        overall_bull_pct = profile_data['overall_bull_pct']
        overall_bear_pct = profile_data['overall_bear_pct']

        # Determine sentiment
        if overall_delta > 20:
            sentiment = "STRONG BULLISH"
        elif overall_delta > 5:
            sentiment = "BULLISH"
        elif overall_delta < -20:
            sentiment = "STRONG BEARISH"
        elif overall_delta < -5:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        # Find levels with strong delta imbalance
        bins = profile_data['bins']
        strong_buy_levels = [b for b in bins if b['delta_pct'] > 30 and b['volume_total'] > 0]
        strong_sell_levels = [b for b in bins if b['delta_pct'] < -30 and b['volume_total'] > 0]

        # Find absorption zones (high volume, low delta)
        absorption_levels = [b for b in bins if b['normalized_volume'] > 50 and abs(b['delta_pct']) < 10]

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
            'overall_delta': overall_delta,
            'overall_bull_pct': overall_bull_pct,
            'overall_bear_pct': overall_bear_pct,
            'current_price': current_price,
            'price_position': price_position,
            'distance_from_poc': current_price - poc_price,
            'distance_from_poc_pct': ((current_price - poc_price) / poc_price * 100) if poc_price > 0 else 0,
            'strong_buy_levels': [{'price': b['mid'], 'delta': b['delta_pct']} for b in strong_buy_levels],
            'strong_sell_levels': [{'price': b['mid'], 'delta': b['delta_pct']} for b in strong_sell_levels],
            'absorption_zones': [{'lower': b['lower'], 'upper': b['upper'], 'volume': b['volume_total']} for b in absorption_levels],
            'profile_high': profile_data['period_high'],
            'profile_low': profile_data['period_low']
        }

    def format_report(self, signals: Dict) -> str:
        """
        Format DeltaFlow profile signals as readable report

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
║        DELTAFLOW VOLUME PROFILE ANALYSIS               ║
╚════════════════════════════════════════════════════════╝

📊 SENTIMENT: {signals['sentiment']}
⚖️  OVERALL DELTA: {signals['overall_delta']:+.1f}%

💰 Buy Volume: {signals['overall_bull_pct']:.1f}%
📉 Sell Volume: {signals['overall_bear_pct']:.1f}%

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
🟢 STRONG BUY LEVELS (Delta > +30%)
"""
        if signals['strong_buy_levels']:
            for level in signals['strong_buy_levels']:
                report += f"  • {level['price']:.2f} (Δ {level['delta']:+.1f}%)\n"
        else:
            report += "  • None detected\n"

        report += """
─────────────────────────────────────────────────────────
🔴 STRONG SELL LEVELS (Delta < -30%)
"""
        if signals['strong_sell_levels']:
            for level in signals['strong_sell_levels']:
                report += f"  • {level['price']:.2f} (Δ {level['delta']:+.1f}%)\n"
        else:
            report += "  • None detected\n"

        report += """
─────────────────────────────────────────────────────────
🛡️  ABSORPTION ZONES (High Volume, Low Delta)
"""
        if signals['absorption_zones']:
            for zone in signals['absorption_zones']:
                report += f"  • {zone['lower']:.2f} - {zone['upper']:.2f} (Vol: {zone['volume']:,.0f})\n"
        else:
            report += "  • None detected\n"

        return report

    def get_delta_levels_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of delta at each price level

        Parameters:
        -----------
        df : DataFrame
            OHLCV data

        Returns:
        --------
        dict : Summary of delta distribution
        """
        profile_data = self.calculate(df)

        if not profile_data.get('success'):
            return {'success': False, 'error': profile_data.get('error')}

        bins = profile_data['bins']

        # Categorize bins by delta
        strong_buy_bins = [b for b in bins if b['delta_pct'] > 30]
        moderate_buy_bins = [b for b in bins if 10 < b['delta_pct'] <= 30]
        neutral_bins = [b for b in bins if -10 <= b['delta_pct'] <= 10]
        moderate_sell_bins = [b for b in bins if -30 <= b['delta_pct'] < -10]
        strong_sell_bins = [b for b in bins if b['delta_pct'] < -30]

        return {
            'success': True,
            'strong_buy': len(strong_buy_bins),
            'moderate_buy': len(moderate_buy_bins),
            'neutral': len(neutral_bins),
            'moderate_sell': len(moderate_sell_bins),
            'strong_sell': len(strong_sell_bins),
            'total_bins': self.bins,
            'poc_price': profile_data['poc_price'],
            'overall_delta': profile_data['overall_delta']
        }
