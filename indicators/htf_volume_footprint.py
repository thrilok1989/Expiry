"""
Real-Time HTF Volume Footprint Indicator
Converted from Pine Script by BigBeluga
https://creativecommons.org/licenses/by-nc-sa/4.0/

Shows volume distribution across price levels on higher timeframes
Identifies Point of Control (POC) - the price with highest volume
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VolumeLevel:
    """Represents a single volume level in the footprint"""
    price_low: float
    price_high: float
    volume: float
    is_poc: bool = False


@dataclass
class VolumeFootprint:
    """Complete volume footprint for a timeframe period"""
    start_bar: int
    end_bar: int
    high: float
    low: float
    levels: List[VolumeLevel]
    poc_price: float
    poc_volume: float


class HTFVolumeFootprint:
    """
    Higher Timeframe Volume Footprint Indicator

    Displays volume distribution across price levels for higher timeframes
    Shows Point of Control (POC) - price level with highest volume
    Useful for identifying institutional activity and key support/resistance
    """

    def __init__(
        self,
        bins: int = 20,
        timeframe: str = '1D',
        show_dynamic_poc: bool = False
    ):
        """
        Initialize HTF Volume Footprint indicator

        Args:
            bins: Number of price levels to divide the range into
            timeframe: Higher timeframe to analyze ('1D', '1W', '1M')
            show_dynamic_poc: Whether to show POC as it updates
        """
        self.bins = bins
        self.timeframe = timeframe
        self.show_dynamic_poc = show_dynamic_poc

        # Map timeframe to bars
        self.timeframe_bars = {
            '1D': 1,
            '2D': 2,
            '3D': 3,
            '4D': 4,
            '5D': 5,
            '1W': 5,
            '2W': 10,
            '3W': 15,
            '1M': 20,
            '2M': 40,
            '3M': 60
        }

    def _get_column_names(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get correct column names (handle both uppercase and lowercase)"""
        return {
            'open': 'Open' if 'Open' in df.columns else 'open',
            'high': 'High' if 'High' in df.columns else 'high',
            'low': 'Low' if 'Low' in df.columns else 'low',
            'close': 'Close' if 'Close' in df.columns else 'close',
            'volume': 'Volume' if 'Volume' in df.columns else 'volume' if 'volume' in df.columns else None
        }

    def _normalize_volume(self, df: pd.DataFrame, cols: Dict[str, str]) -> pd.Series:
        """Normalize volume using standard deviation"""
        if cols['volume'] is None:
            return pd.Series([1.0] * len(df), index=df.index)

        volume_stdev = df[cols['volume']].rolling(window=min(200, len(df))).std()
        volume_stdev = volume_stdev.fillna(1.0).replace(0, 1.0)

        normalized = df[cols['volume']] / volume_stdev
        return normalized.fillna(1.0)

    def _detect_htf_periods(self, df: pd.DataFrame, period_bars: int) -> List[Tuple[int, int]]:
        """
        Detect higher timeframe periods in the data

        Args:
            df: DataFrame with OHLC data
            period_bars: Number of bars per HTF period

        Returns:
            List of (start_index, end_index) tuples for each HTF period
        """
        periods = []
        total_bars = len(df)

        for i in range(0, total_bars, period_bars):
            start = i
            end = min(i + period_bars, total_bars)
            if end > start:
                periods.append((start, end))

        return periods

    def calculate(self, df: pd.DataFrame) -> Dict:
        """
        Calculate HTF volume footprint

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict containing footprint data and POC levels
        """
        if len(df) < 10:
            return {
                'success': False,
                'error': 'Insufficient data'
            }

        cols = self._get_column_names(df)

        if cols['volume'] is None:
            return {
                'success': False,
                'error': 'Volume data not available'
            }

        # Get timeframe period in bars
        period_bars = self.timeframe_bars.get(self.timeframe, 5)

        # Normalize volume
        normalized_volume = self._normalize_volume(df, cols)

        # Detect HTF periods
        periods = self._detect_htf_periods(df, period_bars)

        if not periods:
            return {
                'success': False,
                'error': 'No HTF periods detected'
            }

        # Calculate footprint for most recent complete period and current period
        footprints = []

        for start_idx, end_idx in periods[-3:]:  # Last 3 periods
            period_df = df.iloc[start_idx:end_idx]

            if len(period_df) == 0:
                continue

            # Get period high and low
            period_high = period_df[cols['high']].max()
            period_low = period_df[cols['low']].min()

            if period_high == period_low:
                continue

            # Calculate bin size
            bin_size = (period_high - period_low) / self.bins

            # Initialize volume levels
            levels = []
            for i in range(self.bins):
                level_low = period_low + i * bin_size
                level_high = level_low + bin_size
                levels.append(VolumeLevel(
                    price_low=level_low,
                    price_high=level_high,
                    volume=0.0
                ))

            # Distribute volume across levels
            for j in range(len(period_df)):
                close_price = period_df[cols['close']].iloc[j]
                vol = normalized_volume.iloc[start_idx + j]

                # Find which level this close price belongs to
                for level in levels:
                    if level.price_low <= close_price < level.price_high:
                        level.volume += vol
                        break

            # Find POC (level with highest volume)
            if levels:
                max_volume = max(level.volume for level in levels)
                for level in levels:
                    if level.volume == max_volume:
                        level.is_poc = True

                # Get POC price (midpoint of POC level)
                poc_level = next(level for level in levels if level.is_poc)
                poc_price = (poc_level.price_low + poc_level.price_high) / 2

                footprint = VolumeFootprint(
                    start_bar=start_idx,
                    end_bar=end_idx - 1,
                    high=period_high,
                    low=period_low,
                    levels=levels,
                    poc_price=poc_price,
                    poc_volume=max_volume
                )

                footprints.append(footprint)

        if not footprints:
            return {
                'success': False,
                'error': 'No footprints calculated'
            }

        # Get current footprint (last one)
        current_footprint = footprints[-1]

        # Calculate value area (levels containing 70% of volume)
        total_volume = sum(level.volume for level in current_footprint.levels)
        sorted_levels = sorted(current_footprint.levels, key=lambda x: x.volume, reverse=True)

        value_area_volume = 0
        value_area_high = current_footprint.low
        value_area_low = current_footprint.high

        for level in sorted_levels:
            value_area_volume += level.volume
            value_area_high = max(value_area_high, level.price_high)
            value_area_low = min(value_area_low, level.price_low)

            if value_area_volume >= total_volume * 0.7:
                break

        return {
            'success': True,
            'footprints': footprints,
            'current_footprint': current_footprint,
            'poc_price': current_footprint.poc_price,
            'poc_volume': current_footprint.poc_volume,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'htf_high': current_footprint.high,
            'htf_low': current_footprint.low,
            'timeframe': self.timeframe,
            'bins': self.bins,
            'show_dynamic_poc': self.show_dynamic_poc
        }

    def get_signals(self, df: pd.DataFrame) -> Dict:
        """
        Get trading signals from volume footprint

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict containing signals and analysis
        """
        result = self.calculate(df)

        if not result['success']:
            return result

        cols = self._get_column_names(df)
        current_price = df[cols['close']].iloc[-1]

        signals = []

        # Price near POC
        poc_distance = abs(current_price - result['poc_price'])
        poc_distance_pct = (poc_distance / current_price) * 100

        if poc_distance_pct < 0.5:  # Within 0.5% of POC
            signals.append({
                'type': 'POC_PROXIMITY',
                'message': f"Price near POC ({result['poc_price']:.2f})",
                'bias': 'NEUTRAL',
                'strength': 'HIGH'
            })

        # Price near value area bounds
        if abs(current_price - result['value_area_high']) / current_price * 100 < 0.5:
            signals.append({
                'type': 'VALUE_AREA_HIGH',
                'message': f"Price at Value Area High ({result['value_area_high']:.2f})",
                'bias': 'BEARISH',
                'strength': 'MEDIUM'
            })

        if abs(current_price - result['value_area_low']) / current_price * 100 < 0.5:
            signals.append({
                'type': 'VALUE_AREA_LOW',
                'message': f"Price at Value Area Low ({result['value_area_low']:.2f})",
                'bias': 'BULLISH',
                'strength': 'MEDIUM'
            })

        # Price near HTF extremes
        if abs(current_price - result['htf_high']) / current_price * 100 < 0.5:
            signals.append({
                'type': 'HTF_HIGH',
                'message': f"Price at {self.timeframe} High ({result['htf_high']:.2f})",
                'bias': 'BEARISH',
                'strength': 'HIGH'
            })

        if abs(current_price - result['htf_low']) / current_price * 100 < 0.5:
            signals.append({
                'type': 'HTF_LOW',
                'message': f"Price at {self.timeframe} Low ({result['htf_low']:.2f})",
                'bias': 'BULLISH',
                'strength': 'HIGH'
            })

        result['signals'] = signals
        result['current_price'] = current_price

        return result
