"""
Volume Order Block Strength Tracker

Tracks the strength/weakness of Volume Order Blocks over time by monitoring:
1. Number of times the block has been tested
2. Respect vs break ratio
3. Volume at each test
4. Strength deterioration patterns

This provides a quality score (0-100) for each order block to help determine
if the level is getting stronger or weaker.
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime


class VOBStrengthTracker:
    """
    Tracks and analyzes the strength of Volume Order Blocks
    """

    def __init__(self, respect_distance: float = 5.0):
        """
        Initialize VOB Strength Tracker

        Args:
            respect_distance: Points within block considered as "respect" (default 5.0)
        """
        self.respect_distance = respect_distance
        self.block_history = {}  # Store test history for each block

    def calculate_strength(self,
                          block: Dict,
                          df: pd.DataFrame,
                          lookback_periods: int = 50) -> Dict:
        """
        Calculate strength score for a Volume Order Block

        Args:
            block: VOB block dict with 'upper', 'lower', 'mid', 'volume'
            df: Price dataframe with OHLC data
            lookback_periods: Number of candles to analyze for tests

        Returns:
            Dict with strength metrics and score
        """
        if df is None or len(df) < 2:
            return self._default_strength_result()

        upper = block['upper']
        lower = block['lower']
        mid = block['mid']
        block_volume = block.get('volume', 0)

        # Analyze recent price action relative to this block
        recent_df = df.tail(lookback_periods).copy()

        # Count tests, respects, and breaks
        test_data = self._analyze_tests(recent_df, upper, lower, mid)

        # Calculate volume trends at tests
        volume_trend = self._analyze_volume_trend(test_data)

        # Calculate strength score (0-100)
        strength_score = self._calculate_score(test_data, volume_trend, block_volume)

        # Determine if block is strengthening or weakening
        trend = self._determine_trend(test_data, volume_trend)

        return {
            'strength_score': round(strength_score, 1),
            'strength_label': self._get_strength_label(strength_score),
            'times_tested': test_data['total_tests'],
            'times_respected': test_data['respects'],
            'times_broken': test_data['breaks'],
            'respect_rate': round(test_data['respect_rate'], 1),
            'avg_test_volume': round(test_data['avg_volume'], 0),
            'last_test_volume': test_data['last_test_volume'],
            'volume_trend': volume_trend,
            'is_strengthening': trend == 'STRENGTHENING',
            'is_weakening': trend == 'WEAKENING',
            'trend': trend,
            'confidence': self._calculate_confidence(test_data)
        }

    def _analyze_tests(self, df: pd.DataFrame, upper: float, lower: float, mid: float) -> Dict:
        """
        Analyze how price has tested the block

        Returns:
            Dict with test statistics
        """
        tests = []
        respects = 0
        breaks = 0

        for idx, row in df.iterrows():
            high = row['high']
            low = row['low']
            close = row['close']
            volume = row.get('volume', 0)

            # Check if candle touched or entered the block
            touched_block = (low <= upper and high >= lower)

            if touched_block:
                # Determine if it was a respect or break
                closed_inside = (lower <= close <= upper)
                broke_through = (close > upper + self.respect_distance) or (close < lower - self.respect_distance)

                test_type = 'break' if broke_through else 'respect'

                if test_type == 'respect':
                    respects += 1
                else:
                    breaks += 1

                tests.append({
                    'timestamp': idx,
                    'type': test_type,
                    'close': close,
                    'volume': volume,
                    'distance_from_mid': abs(close - mid)
                })

        total_tests = len(tests)
        respect_rate = (respects / total_tests * 100) if total_tests > 0 else 0
        avg_volume = sum(t['volume'] for t in tests) / total_tests if total_tests > 0 else 0
        last_test_volume = tests[-1]['volume'] if tests else 0

        return {
            'tests': tests,
            'total_tests': total_tests,
            'respects': respects,
            'breaks': breaks,
            'respect_rate': respect_rate,
            'avg_volume': avg_volume,
            'last_test_volume': last_test_volume
        }

    def _analyze_volume_trend(self, test_data: Dict) -> str:
        """
        Analyze if volume is increasing or decreasing at tests

        Returns:
            'INCREASING', 'DECREASING', or 'STABLE'
        """
        tests = test_data['tests']

        if len(tests) < 2:
            return 'INSUFFICIENT_DATA'

        # Compare first half vs second half of tests
        mid_point = len(tests) // 2
        first_half_avg = sum(t['volume'] for t in tests[:mid_point]) / mid_point if mid_point > 0 else 0
        second_half_avg = sum(t['volume'] for t in tests[mid_point:]) / (len(tests) - mid_point) if len(tests) > mid_point else 0

        if second_half_avg > first_half_avg * 1.2:
            return 'INCREASING'
        elif second_half_avg < first_half_avg * 0.8:
            return 'DECREASING'
        else:
            return 'STABLE'

    def _calculate_score(self, test_data: Dict, volume_trend: str, block_volume: float) -> float:
        """
        Calculate overall strength score (0-100)

        Factors:
        - Respect rate (40% weight)
        - Number of tests (20% weight) - more tests = more reliable
        - Volume trend (20% weight)
        - Last test volume vs block volume (20% weight)
        """
        score = 0

        # 1. Respect rate component (40 points max)
        respect_rate_score = (test_data['respect_rate'] / 100) * 40
        score += respect_rate_score

        # 2. Test count component (20 points max)
        # More tests = more reliable (capped at 5 tests)
        test_count_score = min(test_data['total_tests'] / 5, 1.0) * 20
        score += test_count_score

        # 3. Volume trend component (20 points max)
        if volume_trend == 'INCREASING':
            volume_trend_score = 20
        elif volume_trend == 'STABLE':
            volume_trend_score = 15
        elif volume_trend == 'DECREASING':
            volume_trend_score = 5
        else:
            volume_trend_score = 10
        score += volume_trend_score

        # 4. Last test volume vs block volume (20 points max)
        if test_data['last_test_volume'] > 0 and block_volume > 0:
            volume_ratio = test_data['last_test_volume'] / block_volume
            if volume_ratio >= 1.0:
                last_volume_score = 20
            elif volume_ratio >= 0.7:
                last_volume_score = 15
            elif volume_ratio >= 0.4:
                last_volume_score = 10
            else:
                last_volume_score = 5
        else:
            last_volume_score = 10  # Neutral score if no data
        score += last_volume_score

        return min(score, 100)  # Cap at 100

    def _determine_trend(self, test_data: Dict, volume_trend: str) -> str:
        """
        Determine if block is strengthening, weakening, or stable

        Returns:
            'STRENGTHENING', 'WEAKENING', or 'STABLE'
        """
        tests = test_data['tests']

        if len(tests) < 2:
            return 'INSUFFICIENT_DATA'

        # Analyze recent tests (last 3)
        recent_tests = tests[-3:]
        recent_respects = sum(1 for t in recent_tests if t['type'] == 'respect')
        recent_respect_rate = (recent_respects / len(recent_tests)) * 100

        # Compare to overall respect rate
        overall_respect_rate = test_data['respect_rate']

        # Strengthening conditions:
        # - Recent respect rate > overall rate
        # - Volume increasing
        if recent_respect_rate > overall_respect_rate and volume_trend == 'INCREASING':
            return 'STRENGTHENING'

        # Weakening conditions:
        # - Recent respect rate < overall rate
        # - Volume decreasing or recent breaks
        elif recent_respect_rate < overall_respect_rate or (volume_trend == 'DECREASING' and recent_respects < len(recent_tests) / 2):
            return 'WEAKENING'

        else:
            return 'STABLE'

    def _get_strength_label(self, score: float) -> str:
        """
        Convert numeric score to label

        Args:
            score: Strength score (0-100)

        Returns:
            'VERY_STRONG', 'STRONG', 'MODERATE', 'WEAK', or 'VERY_WEAK'
        """
        if score >= 80:
            return 'VERY_STRONG'
        elif score >= 65:
            return 'STRONG'
        elif score >= 45:
            return 'MODERATE'
        elif score >= 25:
            return 'WEAK'
        else:
            return 'VERY_WEAK'

    def _calculate_confidence(self, test_data: Dict) -> str:
        """
        Calculate confidence level based on number of tests

        Returns:
            'HIGH', 'MEDIUM', or 'LOW'
        """
        total_tests = test_data['total_tests']

        if total_tests >= 5:
            return 'HIGH'
        elif total_tests >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _default_strength_result(self) -> Dict:
        """
        Return default strength result when no data available
        """
        return {
            'strength_score': 50.0,
            'strength_label': 'UNKNOWN',
            'times_tested': 0,
            'times_respected': 0,
            'times_broken': 0,
            'respect_rate': 0.0,
            'avg_test_volume': 0.0,
            'last_test_volume': 0,
            'volume_trend': 'INSUFFICIENT_DATA',
            'is_strengthening': False,
            'is_weakening': False,
            'trend': 'INSUFFICIENT_DATA',
            'confidence': 'LOW'
        }


def get_emoji_for_strength(strength_label: str) -> str:
    """
    Get emoji representation for strength label

    Args:
        strength_label: Strength label string

    Returns:
        Emoji string
    """
    emoji_map = {
        'VERY_STRONG': '🟢🟢🟢',
        'STRONG': '🟢🟢',
        'MODERATE': '🟡',
        'WEAK': '🟠',
        'VERY_WEAK': '🔴',
        'UNKNOWN': '⚪'
    }
    return emoji_map.get(strength_label, '⚪')


def get_description_for_strength(strength_label: str, trend: str) -> str:
    """
    Get human-readable description of strength

    Args:
        strength_label: Strength label string
        trend: Trend string

    Returns:
        Description string
    """
    descriptions = {
        'VERY_STRONG': 'Very Strong - High confidence level',
        'STRONG': 'Strong - Good level to trade from',
        'MODERATE': 'Moderate - Use caution',
        'WEAK': 'Weak - Consider reducing position size',
        'VERY_WEAK': 'Very Weak - High risk of failure',
        'UNKNOWN': 'Insufficient data to assess strength'
    }

    base_desc = descriptions.get(strength_label, 'Unknown')

    if trend == 'STRENGTHENING':
        return f"{base_desc} (🔺 Strengthening)"
    elif trend == 'WEAKENING':
        return f"{base_desc} (🔻 Weakening)"
    else:
        return base_desc
