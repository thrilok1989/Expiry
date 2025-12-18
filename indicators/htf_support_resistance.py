"""
HTF Support/Resistance Indicator
Calculates pivot-based support and resistance levels across multiple timeframes
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class HTFSupportResistance:
    """
    Higher Time Frame Support/Resistance indicator that calculates
    pivot points (support and resistance levels) across multiple timeframes
    """

    def __init__(self, timeframes: Optional[List[str]] = None, pivot_length: int = 5):
        """
        Initialize HTF Support/Resistance indicator

        Args:
            timeframes: List of timeframe strings (e.g., ['5T', '15T', '1H'])
            pivot_length: Lookback period for pivot calculation
        """
        self.timeframes = timeframes or ['5T', '15T', '1H']
        self.pivot_length = pivot_length

    def calculate_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Calculate support and resistance levels for configured timeframes

        Args:
            df: DataFrame with OHLC data

        Returns:
            List of dicts with 'type', 'price', and 'timeframe' keys
        """
        levels = []

        for timeframe in self.timeframes:
            pivot_data = self._calculate_pivot_levels(df, timeframe, self.pivot_length)

            if pivot_data:
                # Add resistance level
                levels.append({
                    'type': 'resistance',
                    'price': pivot_data['pivot_high'],
                    'timeframe': timeframe
                })

                # Add support level
                levels.append({
                    'type': 'support',
                    'price': pivot_data['pivot_low'],
                    'timeframe': timeframe
                })

        return levels

    def calculate_multi_timeframe(self, df: pd.DataFrame, levels_config: List[Dict]) -> List[Dict]:
        """
        Calculate support/resistance levels for multiple timeframes

        Args:
            df: DataFrame with OHLC data (1-minute or higher)
            levels_config: List of timeframe configurations, each containing:
                - timeframe: Pandas offset alias (e.g., '1T', '5T', '15T', 'D', 'W')
                - length: Lookback period for pivot calculation
                - style: Line style (not used in calculation)
                - color: Color for display (passed through to output)

        Returns:
            List of dicts, each containing:
                - timeframe: The timeframe code
                - pivot_high: Resistance level (pivot high)
                - pivot_low: Support level (pivot low)
                - color: Color for display
        """
        if df is None or len(df) == 0:
            return []

        df = df.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif 'datetime' in df.columns:
                df = df.set_index('datetime')

        results = []

        for config in levels_config:
            timeframe = config.get('timeframe', '5T')
            length = config.get('length', 5)
            color = config.get('color', '#2196f3')

            # Calculate pivot levels for this timeframe
            pivot_data = self._calculate_pivot_levels(df, timeframe, length)

            if pivot_data:
                results.append({
                    'timeframe': timeframe,
                    'pivot_high': pivot_data['pivot_high'],
                    'pivot_low': pivot_data['pivot_low'],
                    'color': color
                })

        return results

    def _calculate_pivot_levels(self, df: pd.DataFrame, timeframe: str, length: int) -> Optional[Dict]:
        """
        Calculate pivot high and pivot low for a specific timeframe

        Args:
            df: DataFrame with OHLC data
            timeframe: Pandas offset alias (e.g., '1T', '5T', '15T')
            length: Lookback period for pivot calculation

        Returns:
            Dict with pivot_high and pivot_low, or None if insufficient data
        """
        try:
            # Resample data to the target timeframe
            df_resampled = df.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            if len(df_resampled) < length + 2:
                return None

            # Calculate pivot highs and lows using rolling windows
            # A pivot high is a high that is higher than the highs before and after it
            # A pivot low is a low that is lower than the lows before and after it

            pivot_high = self._find_pivot_high(df_resampled, length)
            pivot_low = self._find_pivot_low(df_resampled, length)

            return {
                'pivot_high': pivot_high,
                'pivot_low': pivot_low
            }

        except Exception as e:
            print(f"Error calculating pivot levels for {timeframe}: {e}")
            return None

    def _find_pivot_high(self, df: pd.DataFrame, length: int) -> Optional[float]:
        """
        Find the most recent pivot high (resistance level)

        A pivot high is a high that is the highest in a window of 'length' bars
        on either side.

        Args:
            df: Resampled DataFrame
            length: Lookback period

        Returns:
            Pivot high price, or None if not found
        """
        highs = df['high'].values

        # Look for pivot highs in recent data
        # Start from the end and work backwards (skip the last bar as it's incomplete)
        for i in range(len(highs) - 2, length, -1):
            is_pivot = True

            # Check if this high is higher than 'length' bars before and after
            for j in range(1, length + 1):
                # Check bars before
                if i - j >= 0 and highs[i] <= highs[i - j]:
                    is_pivot = False
                    break
                # Check bars after
                if i + j < len(highs) and highs[i] <= highs[i + j]:
                    is_pivot = False
                    break

            if is_pivot:
                return float(highs[i])

        # If no pivot found, return the recent high
        return float(df['high'].tail(length * 2).max())

    def _find_pivot_low(self, df: pd.DataFrame, length: int) -> Optional[float]:
        """
        Find the most recent pivot low (support level)

        A pivot low is a low that is the lowest in a window of 'length' bars
        on either side.

        Args:
            df: Resampled DataFrame
            length: Lookback period

        Returns:
            Pivot low price, or None if not found
        """
        lows = df['low'].values

        # Look for pivot lows in recent data
        # Start from the end and work backwards (skip the last bar as it's incomplete)
        for i in range(len(lows) - 2, length, -1):
            is_pivot = True

            # Check if this low is lower than 'length' bars before and after
            for j in range(1, length + 1):
                # Check bars before
                if i - j >= 0 and lows[i] >= lows[i - j]:
                    is_pivot = False
                    break
                # Check bars after
                if i + j < len(lows) and lows[i] >= lows[i + j]:
                    is_pivot = False
                    break

            if is_pivot:
                return float(lows[i])

        # If no pivot found, return the recent low
        return float(df['low'].tail(length * 2).min())
