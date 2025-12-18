"""
Liquidity Sentiment Profile Indicator
Converted from Pine Script v5 by LuxAlgo
License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/

This indicator displays liquidity profile and sentiment profile across different price levels
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional


class LiquiditySentimentProfile:
    """
    Liquidity Sentiment Profile Indicator

    Displays trading activity and sentiment distribution across price levels:
    - Liquidity Profile: Shows total trading activity at specific price levels
    - Sentiment Profile: Shows bullish/bearish dominance at each price level
    - Level of Significance: Tracks price levels with highest traded activity
    """

    def __init__(
        self,
        anchor_period: str = 'Auto',  # 'Auto', 'Session', 'Day', 'Week', 'Month', 'Quarter', 'Year'
        num_rows: int = 25,
        profile_width: float = 0.50,
        show_liquidity_profile: bool = True,
        show_sentiment_profile: bool = True,
        show_poc: bool = False,
        show_price_levels: bool = False,
        show_range_bg: bool = True,
        hv_threshold: float = 0.73,
        lv_threshold: float = 0.21,
    ):
        """
        Initialize Liquidity Sentiment Profile

        Parameters:
        -----------
        anchor_period : str
            Timeframe for profile calculation ('Auto', 'Day', 'Week', 'Month', etc.)
        num_rows : int
            Number of price rows/bins for the profile (10-100)
        profile_width : float
            Width of profile bars as percentage (0.1-0.5)
        show_liquidity_profile : bool
            Display liquidity profile (total volume activity)
        show_sentiment_profile : bool
            Display sentiment profile (bull/bear bias)
        show_poc : bool
            Show Point of Control line (level of significance)
        show_price_levels : bool
            Show price level labels
        show_range_bg : bool
            Show background fill for profile range
        hv_threshold : float
            High volume threshold percentage (0.5-0.99)
        lv_threshold : float
            Low volume threshold percentage (0.1-0.4)
        """
        self.anchor_period = anchor_period
        self.num_rows = max(10, min(100, num_rows))
        self.profile_width = max(0.1, min(0.5, profile_width))
        self.show_liquidity_profile = show_liquidity_profile
        self.show_sentiment_profile = show_sentiment_profile
        self.show_poc = show_poc
        self.show_price_levels = show_price_levels
        self.show_range_bg = show_range_bg
        self.hv_threshold = max(0.5, min(0.99, hv_threshold))
        self.lv_threshold = max(0.1, min(0.4, lv_threshold))

        # Color scheme
        self.hv_color = 'rgba(255, 152, 0, 0.2)'  # High volume - orange
        self.av_color = 'rgba(120, 123, 134, 0.2)'  # Average volume - gray
        self.lv_color = 'rgba(41, 98, 255, 0.2)'  # Low volume - blue
        self.bullish_color = 'rgba(38, 166, 154, 0.2)'  # Bullish - teal
        self.bearish_color = 'rgba(239, 83, 80, 0.2)'  # Bearish - red
        self.poc_color = 'rgb(255, 0, 0)'
        self.bg_color = 'rgba(0, 188, 212, 0.05)'

    def _determine_anchor_timeframe(self, df: pd.DataFrame) -> str:
        """Determine the anchor timeframe based on data frequency"""
        if self.anchor_period != 'Auto':
            return self.anchor_period

        # Detect data frequency
        time_diff = (df.index[-1] - df.index[0]).total_seconds() / len(df)

        # Intraday data
        if time_diff < 3600:  # Less than 1 hour
            return 'D'  # Day
        elif time_diff < 14400:  # Less than 4 hours
            return 'W'  # Week
        elif time_diff < 86400:  # Less than 1 day
            return 'W'  # Week
        elif time_diff < 604800:  # Less than 1 week
            return 'M'  # Month
        elif time_diff < 2592000:  # Less than 1 month
            return '3M'  # Quarter
        else:
            return '12M'  # Year

    def _get_period_boundaries(self, df: pd.DataFrame, timeframe: str) -> List[Tuple[int, int]]:
        """Get start and end indices for each period"""
        boundaries = []

        if timeframe == 'D':
            # Group by day
            groups = df.groupby(df.index.date)
        elif timeframe == 'W':
            # Group by week
            groups = df.groupby(pd.Grouper(freq='W'))
        elif timeframe == 'M':
            # Group by month
            groups = df.groupby(pd.Grouper(freq='M'))
        elif timeframe == '3M':
            # Group by quarter
            groups = df.groupby(pd.Grouper(freq='Q'))
        elif timeframe == '12M':
            # Group by year
            groups = df.groupby(pd.Grouper(freq='Y'))
        else:
            # Default to day
            groups = df.groupby(df.index.date)

        for name, group in groups:
            if len(group) > 0:
                start_idx = df.index.get_loc(group.index[0])
                end_idx = df.index.get_loc(group.index[-1])
                boundaries.append((start_idx, end_idx))

        return boundaries

    def _calculate_volume_profile(
        self,
        df_period: pd.DataFrame,
        high: float,
        low: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate volume profile for a period

        Returns:
        --------
        volume_total : array of total volume per price level
        volume_bullish : array of bullish volume per price level
        volume_delta : array of volume delta (bullish - bearish) per price level
        """
        step = (high - low) / self.num_rows

        volume_total = np.zeros(self.num_rows)
        volume_bullish = np.zeros(self.num_rows)

        for idx, row in df_period.iterrows():
            bar_high = row['high']
            bar_low = row['low']
            bar_close = row['close']
            bar_open = row['open']
            bar_volume = row['volume']

            is_bullish = bar_close > bar_open

            # Distribute volume across price levels
            for i in range(self.num_rows):
                level_low = low + i * step
                level_high = low + (i + 1) * step

                # Check if bar overlaps with this price level
                if bar_high >= level_low and bar_low < level_high:
                    # Calculate volume distribution
                    bar_range = bar_high - bar_low
                    if bar_range == 0:
                        volume_weight = bar_volume
                    else:
                        volume_weight = bar_volume * (step / bar_range)

                    volume_total[i] += volume_weight

                    if is_bullish:
                        volume_bullish[i] += volume_weight

        # Calculate volume delta (absolute difference between bull and bear)
        volume_delta = np.zeros(self.num_rows)
        for i in range(self.num_rows):
            bearish_volume = volume_total[i] - volume_bullish[i]
            delta = 2 * volume_bullish[i] - volume_total[i]
            volume_delta[i] = abs(delta)

        return volume_total, volume_bullish, volume_delta

    def calculate(self, df: pd.DataFrame) -> Dict:
        """
        Calculate liquidity sentiment profile

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

        # Determine anchor timeframe
        timeframe = self._determine_anchor_timeframe(df)

        # Get period boundaries
        boundaries = self._get_period_boundaries(df, timeframe)

        if not boundaries:
            return {'success': False, 'error': 'No valid periods found'}

        profiles = []

        # Calculate profiles for each complete period
        for i, (start_idx, end_idx) in enumerate(boundaries[:-1]):
            df_period = df.iloc[start_idx:end_idx+1]

            period_high = df_period['high'].max()
            period_low = df_period['low'].min()

            if period_high == period_low:
                continue

            volume_total, volume_bullish, volume_delta = self._calculate_volume_profile(
                df_period, period_high, period_low
            )

            # Find POC (Point of Control) - level with highest volume
            poc_idx = np.argmax(volume_total)
            poc_price = period_low + (poc_idx + 0.5) * ((period_high - period_low) / self.num_rows)

            profiles.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'high': period_high,
                'low': period_low,
                'volume_total': volume_total,
                'volume_bullish': volume_bullish,
                'volume_delta': volume_delta,
                'poc_price': poc_price,
                'poc_idx': poc_idx
            })

        # Calculate current (incomplete) period profile
        if len(boundaries) > 0:
            start_idx, _ = boundaries[-1]
            df_current = df.iloc[start_idx:]

            current_high = df_current['high'].max()
            current_low = df_current['low'].min()

            if current_high != current_low:
                volume_total, volume_bullish, volume_delta = self._calculate_volume_profile(
                    df_current, current_high, current_low
                )

                poc_idx = np.argmax(volume_total)
                poc_price = current_low + (poc_idx + 0.5) * ((current_high - current_low) / self.num_rows)

                profiles.append({
                    'start_idx': start_idx,
                    'end_idx': len(df) - 1,
                    'high': current_high,
                    'low': current_low,
                    'volume_total': volume_total,
                    'volume_bullish': volume_bullish,
                    'volume_delta': volume_delta,
                    'poc_price': poc_price,
                    'poc_idx': poc_idx,
                    'is_current': True
                })

        return {
            'success': True,
            'profiles': profiles,
            'timeframe': timeframe,
            'num_rows': self.num_rows
        }

    def add_to_chart(self, fig: go.Figure, df: pd.DataFrame, profile_data: Dict) -> go.Figure:
        """
        Add liquidity sentiment profile to existing Plotly chart

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Existing chart figure
        df : DataFrame
            OHLCV data
        profile_data : dict
            Profile calculation results

        Returns:
        --------
        Updated figure
        """
        if not profile_data.get('success'):
            return fig

        profiles = profile_data['profiles']

        for profile in profiles:
            start_idx = profile['start_idx']
            end_idx = profile['end_idx']
            high = profile['high']
            low = profile['low']
            volume_total = profile['volume_total']
            volume_bullish = profile['volume_bullish']
            volume_delta = profile['volume_delta']
            poc_price = profile['poc_price']
            is_current = profile.get('is_current', False)

            step = (high - low) / self.num_rows

            # Calculate mid point for profile bars
            mid_idx = (start_idx + end_idx) // 2
            mid_time = df.index[mid_idx]

            # Calculate profile bar width in time units
            time_range = end_idx - start_idx
            bar_width = int(time_range * self.profile_width)

            # Normalize volumes
            max_volume_total = volume_total.max() if volume_total.max() > 0 else 1
            max_volume_delta = volume_delta.max() if volume_delta.max() > 0 else 1

            # Add background fill for profile range
            if self.show_range_bg:
                fig.add_shape(
                    type="rect",
                    x0=df.index[start_idx],
                    x1=df.index[end_idx],
                    y0=low,
                    y1=high,
                    fillcolor=self.bg_color,
                    line=dict(width=0),
                    layer="below"
                )

            # Draw liquidity profile (horizontal bars)
            if self.show_liquidity_profile:
                for i in range(self.num_rows):
                    level_low = low + i * step
                    level_high = low + (i + 1) * step
                    level_mid = (level_low + level_high) / 2

                    # Calculate bar length
                    normalized_volume = volume_total[i] / max_volume_total
                    bar_length = int(normalized_volume * bar_width)

                    if bar_length > 0:
                        # Determine color based on volume threshold
                        if normalized_volume > self.hv_threshold:
                            color = self.hv_color
                        elif normalized_volume < self.lv_threshold:
                            color = self.lv_color
                        else:
                            color = self.av_color

                        # Add horizontal bar
                        fig.add_shape(
                            type="rect",
                            x0=df.index[mid_idx],
                            x1=df.index[min(mid_idx + bar_length, end_idx)],
                            y0=level_low,
                            y1=level_high,
                            fillcolor=color,
                            line=dict(width=0),
                            layer="above"
                        )

            # Draw sentiment profile (horizontal bars in opposite direction)
            if self.show_sentiment_profile:
                for i in range(self.num_rows):
                    level_low = low + i * step
                    level_high = low + (i + 1) * step

                    # Calculate bar length
                    normalized_delta = volume_delta[i] / max_volume_delta
                    bar_length = int(normalized_delta * bar_width)

                    if bar_length > 0:
                        # Determine color based on bullish vs bearish
                        bullish_vol = volume_bullish[i]
                        bearish_vol = volume_total[i] - volume_bullish[i]

                        if bullish_vol > bearish_vol:
                            color = self.bullish_color
                        else:
                            color = self.bearish_color

                        # Add horizontal bar (left side)
                        fig.add_shape(
                            type="rect",
                            x0=df.index[max(mid_idx - bar_length, start_idx)],
                            x1=df.index[mid_idx],
                            y0=level_low,
                            y1=level_high,
                            fillcolor=color,
                            line=dict(width=0),
                            layer="above"
                        )

            # Draw POC line
            if self.show_poc:
                fig.add_shape(
                    type="line",
                    x0=df.index[start_idx],
                    x1=df.index[end_idx],
                    y0=poc_price,
                    y1=poc_price,
                    line=dict(color=self.poc_color, width=2),
                    layer="above"
                )

            # Add price level labels
            if self.show_price_levels and is_current:
                fig.add_annotation(
                    x=df.index[mid_idx],
                    y=high,
                    text=f"{high:.2f}",
                    showarrow=False,
                    font=dict(size=10, color='rgb(0, 188, 212)'),
                    bgcolor='rgba(0, 0, 0, 0.5)',
                    yshift=10
                )
                fig.add_annotation(
                    x=df.index[mid_idx],
                    y=low,
                    text=f"{low:.2f}",
                    showarrow=False,
                    font=dict(size=10, color='rgb(0, 188, 212)'),
                    bgcolor='rgba(0, 0, 0, 0.5)',
                    yshift=-10
                )

        return fig

    def get_signals(self, df: pd.DataFrame) -> Dict:
        """
        Get trading signals from liquidity sentiment profile

        Returns:
        --------
        dict with signal information
        """
        profile_data = self.calculate(df)

        if not profile_data.get('success'):
            return {'success': False, 'error': profile_data.get('error')}

        profiles = profile_data['profiles']
        if not profiles:
            return {'success': False, 'error': 'No profiles available'}

        # Get current (last) profile
        current_profile = profiles[-1]

        poc_price = current_profile['poc_price']
        high = current_profile['high']
        low = current_profile['low']
        volume_total = current_profile['volume_total']
        volume_bullish = current_profile['volume_bullish']

        # Calculate sentiment
        total_bullish = volume_bullish.sum()
        total_volume = volume_total.sum()
        total_bearish = total_volume - total_bullish

        if total_volume == 0:
            sentiment = "NEUTRAL"
        elif total_bullish > total_bearish * 1.2:
            sentiment = "BULLISH"
        elif total_bearish > total_bullish * 1.2:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        # Find high and low volume levels
        max_volume = volume_total.max()
        hv_levels = []
        lv_levels = []

        step = (high - low) / self.num_rows

        for i in range(self.num_rows):
            normalized = volume_total[i] / max_volume if max_volume > 0 else 0
            price_level = low + (i + 0.5) * step

            if normalized > self.hv_threshold:
                hv_levels.append(price_level)
            elif normalized < self.lv_threshold:
                lv_levels.append(price_level)

        current_price = df['close'].iloc[-1]

        return {
            'success': True,
            'sentiment': sentiment,
            'poc_price': poc_price,
            'profile_high': high,
            'profile_low': low,
            'high_volume_levels': hv_levels,
            'low_volume_levels': lv_levels,
            'bullish_volume_pct': (total_bullish / total_volume * 100) if total_volume > 0 else 0,
            'bearish_volume_pct': (total_bearish / total_volume * 100) if total_volume > 0 else 0,
            'current_price': current_price,
            'distance_from_poc': current_price - poc_price
        }
