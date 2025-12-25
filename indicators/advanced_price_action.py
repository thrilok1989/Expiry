"""
Advanced Price Action Analysis Module
Implements BOS, CHOCH, Fibonacci levels, and Geometrical Pattern Detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class AdvancedPriceAction:
    """
    Advanced price action analysis including:
    - BOS (Break of Structure)
    - CHOCH (Change of Character)
    - Fibonacci Retracement/Extension levels
    - Geometrical Pattern Detection (Head & Shoulders, Triangles, Flags, etc.)
    """

    def __init__(self, swing_length: int = 5):
        """
        Initialize Advanced Price Action analyzer

        Args:
            swing_length: Number of bars to use for swing high/low detection
        """
        self.swing_length = swing_length

    # =========================================================================
    # SWING HIGH/LOW DETECTION
    # =========================================================================

    def find_swing_highs_lows(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """
        Find swing highs and lows in the price data

        Args:
            df: DataFrame with OHLC data

        Returns:
            Tuple of (swing_highs, swing_lows) lists with dicts containing index, price, time
        """
        # Handle both uppercase and lowercase column names
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'

        swing_highs = []
        swing_lows = []

        length = self.swing_length

        for i in range(length, len(df) - length):
            # Check for swing high
            is_swing_high = True
            for j in range(i - length, i + length + 1):
                if j != i and df[high_col].iloc[j] >= df[high_col].iloc[i]:
                    is_swing_high = False
                    break

            if is_swing_high:
                swing_highs.append({
                    'index': i,
                    'price': df[high_col].iloc[i],
                    'time': df.index[i]
                })

            # Check for swing low
            is_swing_low = True
            for j in range(i - length, i + length + 1):
                if j != i and df[low_col].iloc[j] <= df[low_col].iloc[i]:
                    is_swing_low = False
                    break

            if is_swing_low:
                swing_lows.append({
                    'index': i,
                    'price': df[low_col].iloc[i],
                    'time': df.index[i]
                })

        return swing_highs, swing_lows

    # =========================================================================
    # BOS (BREAK OF STRUCTURE) DETECTION
    # =========================================================================

    def detect_bos(self, df: pd.DataFrame, swing_highs: List[Dict] = None, swing_lows: List[Dict] = None) -> List[Dict]:
        """
        Detect Break of Structure (BOS) events

        This is a convenience wrapper that can be called with just a DataFrame.
        If swing points are not provided, they will be calculated automatically.

        Args:
            df: DataFrame with OHLC data
            swing_highs: Optional list of swing high points (will be calculated if not provided)
            swing_lows: Optional list of swing low points (will be calculated if not provided)

        Returns:
            List of BOS events
        """
        if swing_highs is None or swing_lows is None:
            swing_highs, swing_lows = self.find_swing_highs_lows(df)

        return self._detect_bos_internal(df, swing_highs, swing_lows)

    def _detect_bos_internal(self, df: pd.DataFrame, swing_highs: List[Dict], swing_lows: List[Dict]) -> List[Dict]:
        """
        Detect Break of Structure (BOS) events

        BOS occurs when price breaks above the most recent swing high (bullish BOS)
        or below the most recent swing low (bearish BOS)

        Args:
            df: DataFrame with OHLC data
            swing_highs: List of swing high points
            swing_lows: List of swing low points

        Returns:
            List of BOS events with type, price, time, and previous structure level
        """
        # Handle both uppercase and lowercase column names
        close_col = 'Close' if 'Close' in df.columns else 'close'

        bos_events = []

        # Track the most recent swing high and low
        for i in range(len(df)):
            # Get most recent swing high before this bar
            recent_swing_high = None
            for sh in reversed(swing_highs):
                if sh['index'] < i:
                    recent_swing_high = sh
                    break

            # Get most recent swing low before this bar
            recent_swing_low = None
            for sl in reversed(swing_lows):
                if sl['index'] < i:
                    recent_swing_low = sl
                    break

            # Check for bullish BOS (price breaks above recent swing high)
            if recent_swing_high and df[close_col].iloc[i] > recent_swing_high['price']:
                # Check if we haven't already recorded this BOS
                if not any(bos['type'] == 'BULLISH' and bos['structure_level'] == recent_swing_high['price']
                          for bos in bos_events):
                    bos_events.append({
                        'type': 'BULLISH',
                        'index': i,
                        'price': df[close_col].iloc[i],
                        'time': df.index[i],
                        'structure_level': recent_swing_high['price'],
                        'structure_time': recent_swing_high['time']
                    })

            # Check for bearish BOS (price breaks below recent swing low)
            if recent_swing_low and df[close_col].iloc[i] < recent_swing_low['price']:
                # Check if we haven't already recorded this BOS
                if not any(bos['type'] == 'BEARISH' and bos['structure_level'] == recent_swing_low['price']
                          for bos in bos_events):
                    bos_events.append({
                        'type': 'BEARISH',
                        'index': i,
                        'price': df[close_col].iloc[i],
                        'time': df.index[i],
                        'structure_level': recent_swing_low['price'],
                        'structure_time': recent_swing_low['time']
                    })

        return bos_events

    # =========================================================================
    # CHOCH (CHANGE OF CHARACTER) DETECTION
    # =========================================================================

    def detect_choch(self, df: pd.DataFrame, swing_highs: List[Dict] = None, swing_lows: List[Dict] = None) -> List[Dict]:
        """
        Detect Change of Character (CHOCH) events

        This is a convenience wrapper that can be called with just a DataFrame.
        If swing points are not provided, they will be calculated automatically.

        Args:
            df: DataFrame with OHLC data
            swing_highs: Optional list of swing high points (will be calculated if not provided)
            swing_lows: Optional list of swing low points (will be calculated if not provided)

        Returns:
            List of CHOCH events
        """
        if swing_highs is None or swing_lows is None:
            swing_highs, swing_lows = self.find_swing_highs_lows(df)

        return self._detect_choch_internal(df, swing_highs, swing_lows)

    def _detect_choch_internal(self, df: pd.DataFrame, swing_highs: List[Dict], swing_lows: List[Dict]) -> List[Dict]:
        """
        Detect Change of Character (CHOCH) events

        CHOCH occurs when price fails to make a higher high in an uptrend
        or fails to make a lower low in a downtrend (first sign of reversal)

        Args:
            df: DataFrame with OHLC data
            swing_highs: List of swing high points
            swing_lows: List of swing low points

        Returns:
            List of CHOCH events with type, price, time
        """
        choch_events = []

        # Identify trend by comparing consecutive swing highs/lows
        for i in range(1, len(swing_highs)):
            prev_high = swing_highs[i - 1]
            curr_high = swing_highs[i]

            # In uptrend, CHOCH occurs when we fail to make higher high
            if curr_high['price'] < prev_high['price']:
                choch_events.append({
                    'type': 'BEARISH',  # Potential downtrend starting
                    'index': curr_high['index'],
                    'price': curr_high['price'],
                    'time': curr_high['time'],
                    'prev_structure': prev_high['price']
                })

        for i in range(1, len(swing_lows)):
            prev_low = swing_lows[i - 1]
            curr_low = swing_lows[i]

            # In downtrend, CHOCH occurs when we fail to make lower low
            if curr_low['price'] > prev_low['price']:
                choch_events.append({
                    'type': 'BULLISH',  # Potential uptrend starting
                    'index': curr_low['index'],
                    'price': curr_low['price'],
                    'time': curr_low['time'],
                    'prev_structure': prev_low['price']
                })

        # Sort by index
        choch_events.sort(key=lambda x: x['index'])

        return choch_events

    # =========================================================================
    # FIBONACCI RETRACEMENT/EXTENSION LEVELS
    # =========================================================================

    def calculate_fibonacci_levels(self, df: pd.DataFrame, swing_highs: List[Dict],
                                   swing_lows: List[Dict], lookback: int = 3) -> Dict:
        """
        Calculate Fibonacci retracement and extension levels

        Uses the most recent significant swing high and low

        Args:
            df: DataFrame with OHLC data
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            lookback: Number of most recent swings to consider

        Returns:
            Dict with Fibonacci levels (retracement and extension)
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'success': False, 'error': 'Insufficient swing points'}

        # Get most recent significant high and low
        recent_highs = swing_highs[-lookback:]
        recent_lows = swing_lows[-lookback:]

        highest = max(recent_highs, key=lambda x: x['price'])
        lowest = min(recent_lows, key=lambda x: x['price'])

        # Determine trend direction
        trend_up = highest['index'] < lowest['index']  # Low came after high = uptrend

        # Calculate Fibonacci levels
        price_range = highest['price'] - lowest['price']

        # Retracement levels (from high to low for downtrend, low to high for uptrend)
        fib_ratios = {
            '0.0': 0.0,
            '0.236': 0.236,
            '0.382': 0.382,
            '0.5': 0.5,
            '0.618': 0.618,  # Golden ratio
            '0.786': 0.786,
            '1.0': 1.0
        }

        # Extension levels
        fib_extensions = {
            '1.272': 1.272,
            '1.414': 1.414,
            '1.618': 1.618,  # Golden ratio extension
            '2.0': 2.0,
            '2.618': 2.618
        }

        retracement_levels = {}
        extension_levels = {}

        if trend_up:
            # Uptrend: retracements from recent low
            for label, ratio in fib_ratios.items():
                retracement_levels[label] = lowest['price'] + (price_range * ratio)

            # Extensions above high
            for label, ratio in fib_extensions.items():
                extension_levels[label] = lowest['price'] + (price_range * ratio)
        else:
            # Downtrend: retracements from recent high
            for label, ratio in fib_ratios.items():
                retracement_levels[label] = highest['price'] - (price_range * ratio)

            # Extensions below low
            for label, ratio in fib_extensions.items():
                extension_levels[label] = highest['price'] - (price_range * ratio)

        return {
            'success': True,
            'trend_up': trend_up,
            'swing_high': highest,
            'swing_low': lowest,
            'retracement_levels': retracement_levels,
            'extension_levels': extension_levels,
            'price_range': price_range
        }

    # =========================================================================
    # GEOMETRICAL PATTERN DETECTION
    # =========================================================================

    def detect_head_and_shoulders(self, swing_highs: List[Dict], swing_lows: List[Dict],
                                  tolerance: float = 0.02) -> List[Dict]:
        """
        Detect Head and Shoulders pattern

        Pattern consists of:
        - Left shoulder (swing high)
        - Head (higher swing high)
        - Right shoulder (swing high similar to left shoulder)
        - Neckline (support through swing lows between shoulders)

        Args:
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            tolerance: Price tolerance for shoulder matching (default 2%)

        Returns:
            List of detected head & shoulders patterns
        """
        patterns = []

        # Need at least 3 swing highs for head and shoulders
        if len(swing_highs) < 3:
            return patterns

        for i in range(len(swing_highs) - 2):
            left_shoulder = swing_highs[i]
            head = swing_highs[i + 1]
            right_shoulder = swing_highs[i + 2]

            # Check if head is higher than both shoulders
            if head['price'] > left_shoulder['price'] and head['price'] > right_shoulder['price']:
                # Check if shoulders are roughly equal (within tolerance)
                shoulder_diff = abs(left_shoulder['price'] - right_shoulder['price'])
                avg_shoulder = (left_shoulder['price'] + right_shoulder['price']) / 2

                if shoulder_diff / avg_shoulder <= tolerance:
                    # Find neckline (lows between shoulders)
                    neckline_lows = [sl for sl in swing_lows
                                    if left_shoulder['index'] < sl['index'] < right_shoulder['index']]

                    if neckline_lows:
                        neckline_price = sum(sl['price'] for sl in neckline_lows) / len(neckline_lows)

                        patterns.append({
                            'type': 'HEAD_AND_SHOULDERS',
                            'left_shoulder': left_shoulder,
                            'head': head,
                            'right_shoulder': right_shoulder,
                            'neckline_price': neckline_price,
                            'target': neckline_price - (head['price'] - neckline_price),  # Measured move
                            'completed': False  # Will be true when neckline is broken
                        })

        return patterns

    def detect_inverse_head_and_shoulders(self, swing_highs: List[Dict], swing_lows: List[Dict],
                                         tolerance: float = 0.02) -> List[Dict]:
        """
        Detect Inverse Head and Shoulders pattern (bullish reversal)

        Args:
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            tolerance: Price tolerance for shoulder matching

        Returns:
            List of detected inverse head & shoulders patterns
        """
        patterns = []

        if len(swing_lows) < 3:
            return patterns

        for i in range(len(swing_lows) - 2):
            left_shoulder = swing_lows[i]
            head = swing_lows[i + 1]
            right_shoulder = swing_lows[i + 2]

            # Check if head is lower than both shoulders
            if head['price'] < left_shoulder['price'] and head['price'] < right_shoulder['price']:
                # Check if shoulders are roughly equal
                shoulder_diff = abs(left_shoulder['price'] - right_shoulder['price'])
                avg_shoulder = (left_shoulder['price'] + right_shoulder['price']) / 2

                if shoulder_diff / avg_shoulder <= tolerance:
                    # Find neckline (highs between shoulders)
                    neckline_highs = [sh for sh in swing_highs
                                     if left_shoulder['index'] < sh['index'] < right_shoulder['index']]

                    if neckline_highs:
                        neckline_price = sum(sh['price'] for sh in neckline_highs) / len(neckline_highs)

                        patterns.append({
                            'type': 'INVERSE_HEAD_AND_SHOULDERS',
                            'left_shoulder': left_shoulder,
                            'head': head,
                            'right_shoulder': right_shoulder,
                            'neckline_price': neckline_price,
                            'target': neckline_price + (neckline_price - head['price']),  # Measured move
                            'completed': False
                        })

        return patterns

    def detect_triangles(self, swing_highs: List[Dict], swing_lows: List[Dict],
                        min_touches: int = 2) -> List[Dict]:
        """
        Detect Triangle patterns (ascending, descending, symmetrical)

        Args:
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            min_touches: Minimum number of touches required for trendline

        Returns:
            List of detected triangle patterns
        """
        patterns = []

        if len(swing_highs) < min_touches or len(swing_lows) < min_touches:
            return patterns

        # Take last 4-6 swing points for analysis
        recent_highs = swing_highs[-6:]
        recent_lows = swing_lows[-6:]

        # Calculate trendline slopes
        if len(recent_highs) >= 2:
            high_slope = (recent_highs[-1]['price'] - recent_highs[0]['price']) / \
                        (recent_highs[-1]['index'] - recent_highs[0]['index'])
        else:
            high_slope = 0

        if len(recent_lows) >= 2:
            low_slope = (recent_lows[-1]['price'] - recent_lows[0]['price']) / \
                       (recent_lows[-1]['index'] - recent_lows[0]['index'])
        else:
            low_slope = 0

        # Classify triangle type
        if high_slope < 0 and low_slope > 0:
            # Converging lines = Symmetrical Triangle
            triangle_type = "SYMMETRICAL_TRIANGLE"
        elif abs(high_slope) < 0.01 and low_slope > 0:
            # Flat top, rising bottom = Ascending Triangle
            triangle_type = "ASCENDING_TRIANGLE"
        elif high_slope < 0 and abs(low_slope) < 0.01:
            # Falling top, flat bottom = Descending Triangle
            triangle_type = "DESCENDING_TRIANGLE"
        else:
            return patterns  # Not a triangle

        patterns.append({
            'type': triangle_type,
            'upper_trendline': recent_highs,
            'lower_trendline': recent_lows,
            'high_slope': high_slope,
            'low_slope': low_slope,
            'apex_estimate': None  # Can be calculated if needed
        })

        return patterns

    def detect_flags_and_pennants(self, df: pd.DataFrame, swing_highs: List[Dict],
                                  swing_lows: List[Dict], lookback: int = 20) -> List[Dict]:
        """
        Detect Flag and Pennant patterns (continuation patterns)

        Flags: Sharp move followed by parallel channel consolidation
        Pennants: Sharp move followed by converging channel consolidation

        Args:
            df: DataFrame with OHLC data
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            lookback: Number of bars to look back for trend

        Returns:
            List of detected flag/pennant patterns
        """
        # Handle both uppercase and lowercase column names
        close_col = 'Close' if 'Close' in df.columns else 'close'

        patterns = []

        if len(df) < lookback + 10:
            return patterns

        # Find strong trending moves (flagpole)
        for i in range(lookback, len(df) - 10):
            # Calculate move strength
            flagpole_start = i - lookback
            flagpole_end = i

            price_change = df[close_col].iloc[flagpole_end] - df[close_col].iloc[flagpole_start]
            percent_change = (price_change / df[close_col].iloc[flagpole_start]) * 100

            # Look for strong moves (> 5% in lookback period)
            if abs(percent_change) > 5:
                # Analyze consolidation after the move
                consolidation_highs = [sh for sh in swing_highs
                                      if flagpole_end < sh['index'] < flagpole_end + 15]
                consolidation_lows = [sl for sl in swing_lows
                                     if flagpole_end < sl['index'] < flagpole_end + 15]

                if len(consolidation_highs) >= 2 and len(consolidation_lows) >= 2:
                    # Calculate slopes
                    high_slope = (consolidation_highs[-1]['price'] - consolidation_highs[0]['price']) / \
                                (consolidation_highs[-1]['index'] - consolidation_highs[0]['index'])
                    low_slope = (consolidation_lows[-1]['price'] - consolidation_lows[0]['price']) / \
                               (consolidation_lows[-1]['index'] - consolidation_lows[0]['index'])

                    # Check if slopes are parallel (flag) or converging (pennant)
                    slope_diff = abs(high_slope - low_slope)

                    if slope_diff < 0.1:
                        # Parallel = Flag
                        pattern_type = "BULL_FLAG" if percent_change > 0 else "BEAR_FLAG"
                    else:
                        # Converging = Pennant
                        pattern_type = "BULL_PENNANT" if percent_change > 0 else "BEAR_PENNANT"

                    patterns.append({
                        'type': pattern_type,
                        'flagpole_start': flagpole_start,
                        'flagpole_end': flagpole_end,
                        'percent_change': percent_change,
                        'consolidation_highs': consolidation_highs,
                        'consolidation_lows': consolidation_lows,
                        'breakout_target': df[close_col].iloc[flagpole_end] + price_change  # Measured move
                    })

        return patterns

    # =========================================================================
    # CONVENIENCE WRAPPER METHODS FOR APP.PY
    # =========================================================================

    def calculate_fibonacci(self, df: pd.DataFrame) -> Dict:
        """
        Convenience wrapper for calculate_fibonacci_levels that matches app.py expectations

        Args:
            df: DataFrame with OHLC data

        Returns:
            Dict with Fibonacci retracement levels in simplified format
        """
        swing_highs, swing_lows = self.find_swing_highs_lows(df)
        fib_result = self.calculate_fibonacci_levels(df, swing_highs, swing_lows)

        if not fib_result.get('success', False):
            return {}

        # Return simplified format for display
        return fib_result.get('retracement_levels', {})

    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect all geometric patterns and return in simplified format for display

        Args:
            df: DataFrame with OHLC data

        Returns:
            List of detected patterns with name, type, and indices
        """
        swing_highs, swing_lows = self.find_swing_highs_lows(df)
        patterns = []

        # Detect all pattern types
        head_shoulders = self.detect_head_and_shoulders(swing_highs, swing_lows)
        for hs in head_shoulders:
            patterns.append({
                'name': 'Head and Shoulders',
                'type': 'Bearish Reversal',
                'start_idx': hs['left_shoulder']['index'],
                'end_idx': hs['right_shoulder']['index']
            })

        inv_head_shoulders = self.detect_inverse_head_and_shoulders(swing_highs, swing_lows)
        for ihs in inv_head_shoulders:
            patterns.append({
                'name': 'Inverse Head and Shoulders',
                'type': 'Bullish Reversal',
                'start_idx': ihs['left_shoulder']['index'],
                'end_idx': ihs['right_shoulder']['index']
            })

        triangles = self.detect_triangles(swing_highs, swing_lows)
        for tri in triangles:
            patterns.append({
                'name': tri['type'].replace('_', ' ').title(),
                'type': 'Continuation',
                'start_idx': tri['lower_trendline'][0]['index'] if tri['lower_trendline'] else 'N/A',
                'end_idx': tri['lower_trendline'][-1]['index'] if tri['lower_trendline'] else 'N/A'
            })

        flags_pennants = self.detect_flags_and_pennants(df, swing_highs, swing_lows)
        for fp in flags_pennants:
            patterns.append({
                'name': fp['type'].replace('_', ' ').title(),
                'type': 'Continuation',
                'start_idx': fp['flagpole_start'],
                'end_idx': fp['flagpole_end']
            })

        return patterns

    # =========================================================================
    # MAIN ANALYSIS FUNCTION
    # =========================================================================

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run complete advanced price action analysis

        Args:
            df: DataFrame with OHLC data

        Returns:
            Dict containing all detected patterns, BOS, CHOCH, and Fibonacci levels
        """
        # Find swing points
        swing_highs, swing_lows = self.find_swing_highs_lows(df)

        # Detect BOS and CHOCH
        bos_events = self._detect_bos_internal(df, swing_highs, swing_lows)
        choch_events = self._detect_choch_internal(df, swing_highs, swing_lows)

        # Calculate Fibonacci levels
        fib_levels = self.calculate_fibonacci_levels(df, swing_highs, swing_lows)

        # Detect patterns
        head_shoulders = self.detect_head_and_shoulders(swing_highs, swing_lows)
        inv_head_shoulders = self.detect_inverse_head_and_shoulders(swing_highs, swing_lows)
        triangles = self.detect_triangles(swing_highs, swing_lows)
        flags_pennants = self.detect_flags_and_pennants(df, swing_highs, swing_lows)

        return {
            'success': True,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'bos_events': bos_events,
            'choch_events': choch_events,
            'fibonacci': fib_levels,
            'patterns': {
                'head_and_shoulders': head_shoulders,
                'inverse_head_and_shoulders': inv_head_shoulders,
                'triangles': triangles,
                'flags_pennants': flags_pennants
            }
        }
