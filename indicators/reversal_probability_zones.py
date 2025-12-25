"""
Reversal Probability Zones & Levels Indicator
Converted from Pine Script by LuxAlgo
https://creativecommons.org/licenses/by-nc-sa/4.0/

Predicts reversal zones based on statistical analysis of historical swings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ReversalZone:
    """Represents a reversal probability zone"""
    start_bar: int
    price: float
    is_bullish: bool
    percentile_25_price: float
    percentile_25_bars: int
    percentile_50_price: float
    percentile_50_bars: int
    percentile_75_price: float
    percentile_75_bars: int
    percentile_90_price: float
    percentile_90_bars: int
    max_price: float
    max_bars: int


class ReversalProbabilityZones:
    """
    Reversal Probability Zones & Levels Indicator

    Analyzes historical swing patterns to predict where and when reversals are likely
    Uses percentile analysis to show probability zones at different confidence levels
    """

    def __init__(
        self,
        swing_length: int = 20,
        max_reversals: int = 1000,
        normalize_data: bool = False,
        percentile_25: bool = True,
        percentile_50: bool = True,
        percentile_75: bool = True,
        percentile_90: bool = True
    ):
        """
        Initialize Reversal Probability Zones indicator

        Args:
            swing_length: Lookback period for swing detection
            max_reversals: Maximum number of historical reversals to track
            normalize_data: Whether to normalize price data (percentage-based)
            percentile_25/50/75/90: Which percentile levels to show
        """
        self.swing_length = swing_length
        self.max_reversals = max_reversals
        self.normalize_data = normalize_data

        self.percentile_25 = percentile_25
        self.percentile_50 = percentile_50
        self.percentile_75 = percentile_75
        self.percentile_90 = percentile_90

        # Storage for historical reversals
        self.bullish_price_deltas = []
        self.bullish_bar_deltas = []
        self.bearish_price_deltas = []
        self.bearish_bar_deltas = []

    def _get_column_names(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get correct column names (handle both uppercase and lowercase)"""
        return {
            'open': 'Open' if 'Open' in df.columns else 'open',
            'high': 'High' if 'High' in df.columns else 'high',
            'low': 'Low' if 'Low' in df.columns else 'low',
            'close': 'Close' if 'Close' in df.columns else 'close'
        }

    def _find_swing_highs_lows(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """
        Find swing highs and lows using pivot detection

        Returns:
            Tuple of (swing_highs, swing_lows) lists
        """
        cols = self._get_column_names(df)

        swing_highs = []
        swing_lows = []

        length = self.swing_length

        # Detect swings
        for i in range(length, len(df) - length):
            # Check for swing high
            is_swing_high = True
            for j in range(i - length, i + length + 1):
                if j != i:
                    current_high = max(df[cols['close']].iloc[i], df[cols['open']].iloc[i])
                    compare_high = max(df[cols['close']].iloc[j], df[cols['open']].iloc[j])
                    if compare_high >= current_high:
                        is_swing_high = False
                        break

            if is_swing_high:
                swing_highs.append({
                    'index': i,
                    'price': max(df[cols['close']].iloc[i], df[cols['open']].iloc[i]),
                    'bar': i
                })

            # Check for swing low
            is_swing_low = True
            for j in range(i - length, i + length + 1):
                if j != i:
                    current_low = min(df[cols['close']].iloc[i], df[cols['open']].iloc[i])
                    compare_low = min(df[cols['close']].iloc[j], df[cols['open']].iloc[j])
                    if compare_low <= current_low:
                        is_swing_low = False
                        break

            if is_swing_low:
                swing_lows.append({
                    'index': i,
                    'price': min(df[cols['close']].iloc[i], df[cols['open']].iloc[i]),
                    'bar': i
                })

        return swing_highs, swing_lows

    def _calculate_deltas(
        self,
        swing_highs: List[Dict],
        swing_lows: List[Dict]
    ) -> None:
        """
        Calculate price and bar deltas between swings
        Stores results in instance variables for percentile calculation
        """
        # Combine and sort all swings by bar index
        all_swings = []
        for sh in swing_highs:
            all_swings.append({'price': sh['price'], 'bar': sh['bar'], 'is_high': True})
        for sl in swing_lows:
            all_swings.append({'price': sl['price'], 'bar': sl['bar'], 'is_high': False})

        all_swings.sort(key=lambda x: x['bar'])

        # Calculate deltas
        for i in range(1, len(all_swings)):
            current = all_swings[i]
            previous = all_swings[i - 1]

            # Price delta
            price_delta = abs(current['price'] - previous['price'])
            if self.normalize_data and previous['price'] != 0:
                price_delta = price_delta / previous['price']

            # Bar delta
            bar_delta = current['bar'] - previous['bar']

            if bar_delta > 0 and price_delta > 0:
                # Bullish swing (from low to high)
                if not previous['is_high'] and current['is_high']:
                    self.bullish_price_deltas.append(price_delta)
                    self.bullish_bar_deltas.append(bar_delta)

                # Bearish swing (from high to low)
                elif previous['is_high'] and not current['is_high']:
                    self.bearish_price_deltas.append(price_delta)
                    self.bearish_bar_deltas.append(bar_delta)

        # Limit to max_reversals
        if len(self.bullish_price_deltas) > self.max_reversals:
            self.bullish_price_deltas = self.bullish_price_deltas[-self.max_reversals:]
            self.bullish_bar_deltas = self.bullish_bar_deltas[-self.max_reversals:]

        if len(self.bearish_price_deltas) > self.max_reversals:
            self.bearish_price_deltas = self.bearish_price_deltas[-self.max_reversals:]
            self.bearish_bar_deltas = self.bearish_bar_deltas[-self.max_reversals:]

    def _percentile_nearest_rank(self, data: List[float], percentile: int) -> float:
        """Calculate percentile using nearest rank method"""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        n = len(sorted_data)
        rank = int(np.ceil(percentile / 100.0 * n))
        rank = max(1, min(rank, n))  # Clamp to valid range

        return sorted_data[rank - 1]

    def calculate(self, df: pd.DataFrame) -> Dict:
        """
        Calculate reversal probability zones

        Args:
            df: DataFrame with OHLC data

        Returns:
            Dict containing reversal zones and signals
        """
        if len(df) < self.swing_length * 2:
            return {
                'success': False,
                'error': 'Insufficient data',
                'zones': []
            }

        cols = self._get_column_names(df)

        # Find swings
        swing_highs, swing_lows = self._find_swing_highs_lows(df)

        if not swing_highs or not swing_lows:
            return {
                'success': False,
                'error': 'No swings detected',
                'zones': []
            }

        # Calculate historical deltas
        self._calculate_deltas(swing_highs, swing_lows)

        # Get most recent swing
        last_high = swing_highs[-1] if swing_highs else None
        last_low = swing_lows[-1] if swing_lows else None

        if not last_high and not last_low:
            return {
                'success': False,
                'error': 'No recent swings',
                'zones': []
            }

        # Determine current bias (last swing type)
        is_bullish = False
        current_price = 0
        current_bar = 0

        if last_high and last_low:
            if last_high['bar'] > last_low['bar']:
                # Last swing was high -> expect bearish reversal
                is_bullish = False
                current_price = last_high['price']
                current_bar = last_high['bar']
            else:
                # Last swing was low -> expect bullish reversal
                is_bullish = True
                current_price = last_low['price']
                current_bar = last_low['bar']
        elif last_high:
            is_bullish = False
            current_price = last_high['price']
            current_bar = last_high['bar']
        else:
            is_bullish = True
            current_price = last_low['price']
            current_bar = last_low['bar']

        # Select appropriate historical data
        price_deltas = self.bullish_price_deltas if is_bullish else self.bearish_price_deltas
        bar_deltas = self.bullish_bar_deltas if is_bullish else self.bearish_bar_deltas

        if not price_deltas or not bar_deltas:
            return {
                'success': False,
                'error': 'Insufficient historical data',
                'zones': []
            }

        # Calculate percentiles
        zone = ReversalZone(
            start_bar=current_bar,
            price=current_price,
            is_bullish=is_bullish,
            percentile_25_price=0, percentile_25_bars=0,
            percentile_50_price=0, percentile_50_bars=0,
            percentile_75_price=0, percentile_75_bars=0,
            percentile_90_price=0, percentile_90_bars=0,
            max_price=0, max_bars=0
        )

        if self.percentile_25:
            zone.percentile_25_price = self._percentile_nearest_rank(price_deltas, 25)
            zone.percentile_25_bars = int(self._percentile_nearest_rank(bar_deltas, 25))

        if self.percentile_50:
            zone.percentile_50_price = self._percentile_nearest_rank(price_deltas, 50)
            zone.percentile_50_bars = int(self._percentile_nearest_rank(bar_deltas, 50))

        if self.percentile_75:
            zone.percentile_75_price = self._percentile_nearest_rank(price_deltas, 75)
            zone.percentile_75_bars = int(self._percentile_nearest_rank(bar_deltas, 75))

        if self.percentile_90:
            zone.percentile_90_price = self._percentile_nearest_rank(price_deltas, 90)
            zone.percentile_90_bars = int(self._percentile_nearest_rank(bar_deltas, 90))

        # Calculate max (for zone boundary)
        zone.max_price = max(
            zone.percentile_25_price if self.percentile_25 else 0,
            zone.percentile_50_price if self.percentile_50 else 0,
            zone.percentile_75_price if self.percentile_75 else 0,
            zone.percentile_90_price if self.percentile_90 else 0
        )
        zone.max_bars = max(
            zone.percentile_25_bars if self.percentile_25 else 0,
            zone.percentile_50_bars if self.percentile_50 else 0,
            zone.percentile_75_bars if self.percentile_75 else 0,
            zone.percentile_90_bars if self.percentile_90 else 0
        )

        # Calculate target prices (denormalize if needed)
        direction = 1 if is_bullish else -1

        if self.normalize_data:
            zone.percentile_25_price = current_price * (1 + direction * zone.percentile_25_price)
            zone.percentile_50_price = current_price * (1 + direction * zone.percentile_50_price)
            zone.percentile_75_price = current_price * (1 + direction * zone.percentile_75_price)
            zone.percentile_90_price = current_price * (1 + direction * zone.percentile_90_price)
            zone.max_price = current_price * (1 + direction * zone.max_price)
        else:
            zone.percentile_25_price = current_price + direction * zone.percentile_25_price
            zone.percentile_50_price = current_price + direction * zone.percentile_50_price
            zone.percentile_75_price = current_price + direction * zone.percentile_75_price
            zone.percentile_90_price = current_price + direction * zone.percentile_90_price
            zone.max_price = current_price + direction * zone.max_price

        return {
            'success': True,
            'zone': zone,
            'current_price': df[cols['close']].iloc[-1],
            'swing_highs': swing_highs[-5:],  # Last 5 for visualization
            'swing_lows': swing_lows[-5:],
            'total_bullish_samples': len(self.bullish_price_deltas),
            'total_bearish_samples': len(self.bearish_price_deltas)
        }
