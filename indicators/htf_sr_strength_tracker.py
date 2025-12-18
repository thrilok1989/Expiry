"""
HTF Support/Resistance Strength Tracker

Tracks the strength/weakness of HTF Support and Resistance levels over time by monitoring:
1. Number of times the level has been tested
2. Hold vs break ratio
3. Volume at each test
4. Test spacing (closer tests = weakening)
5. Strength deterioration patterns

This provides a quality score (0-100) for each S/R level to help determine
if the level is getting stronger or weaker.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class HTFSRStrengthTracker:
    """
    Tracks and analyzes the strength of HTF Support and Resistance levels
    """

    def __init__(self, touch_distance: float = 10.0):
        """
        Initialize HTF S/R Strength Tracker

        Args:
            touch_distance: Points within level considered as "touch" (default 10.0)
        """
        self.touch_distance = touch_distance
        self.level_history = {}  # Store test history for each level

    def calculate_strength(self,
                          level: float,
                          level_type: str,
                          df: pd.DataFrame,
                          lookback_periods: int = 100) -> Dict:
        """
        Calculate strength score for a Support or Resistance level

        Args:
            level: Price level (support or resistance)
            level_type: 'SUPPORT' or 'RESISTANCE'
            df: Price dataframe with OHLC data
            lookback_periods: Number of candles to analyze for tests

        Returns:
            Dict with strength metrics and score
        """
        if df is None or len(df) < 2:
            return self._default_strength_result()

        # Analyze recent price action relative to this level
        recent_df = df.tail(lookback_periods).copy()

        # Count tests, holds, and breaks
        test_data = self._analyze_tests(recent_df, level, level_type)

        # Calculate volume trends at tests
        volume_analysis = self._analyze_volume_at_tests(test_data)

        # Analyze test spacing (close together = weakening)
        spacing_analysis = self._analyze_test_spacing(test_data)

        # Calculate strength score (0-100)
        strength_score = self._calculate_score(
            test_data,
            volume_analysis,
            spacing_analysis
        )

        # Determine if level is strengthening or weakening
        trend = self._determine_trend(test_data, volume_analysis, spacing_analysis)

        return {
            'strength_score': round(strength_score, 1),
            'strength_label': self._get_strength_label(strength_score),
            'times_tested': test_data['total_tests'],
            'times_held': test_data['holds'],
            'times_broken': test_data['breaks'],
            'hold_rate': round(test_data['hold_rate'], 1),
            'avg_test_volume': round(test_data['avg_volume'], 0),
            'last_test_volume': test_data['last_test_volume'],
            'volume_trend': volume_analysis['trend'],
            'avg_test_spacing_hours': round(spacing_analysis['avg_spacing_hours'], 1),
            'spacing_trend': spacing_analysis['trend'],
            'is_strengthening': trend == 'STRENGTHENING',
            'is_weakening': trend == 'WEAKENING',
            'trend': trend,
            'confidence': self._calculate_confidence(test_data),
            'last_test_time': test_data['last_test_time']
        }

    def _analyze_tests(self, df: pd.DataFrame, level: float, level_type: str) -> Dict:
        """
        Analyze how price has tested the level

        Args:
            df: Price dataframe
            level: S/R level price
            level_type: 'SUPPORT' or 'RESISTANCE'

        Returns:
            Dict with test statistics
        """
        tests = []
        holds = 0
        breaks = 0

        for idx, row in df.iterrows():
            high = row['high']
            low = row['low']
            close = row['close']
            volume = row.get('volume', 0)

            # Check if candle touched the level
            if level_type == 'SUPPORT':
                touched = low <= (level + self.touch_distance) and low >= (level - self.touch_distance)
                if touched:
                    # Support holds if close is above level
                    held = close >= (level - 5)  # 5 point tolerance
                    test_type = 'hold' if held else 'break'

                    if test_type == 'hold':
                        holds += 1
                    else:
                        breaks += 1

                    tests.append({
                        'timestamp': idx,
                        'type': test_type,
                        'close': close,
                        'low': low,
                        'volume': volume,
                        'distance': abs(close - level)
                    })

            else:  # RESISTANCE
                touched = high >= (level - self.touch_distance) and high <= (level + self.touch_distance)
                if touched:
                    # Resistance holds if close is below level
                    held = close <= (level + 5)  # 5 point tolerance
                    test_type = 'hold' if held else 'break'

                    if test_type == 'hold':
                        holds += 1
                    else:
                        breaks += 1

                    tests.append({
                        'timestamp': idx,
                        'type': test_type,
                        'close': close,
                        'high': high,
                        'volume': volume,
                        'distance': abs(close - level)
                    })

        total_tests = len(tests)
        hold_rate = (holds / total_tests * 100) if total_tests > 0 else 0
        avg_volume = sum(t['volume'] for t in tests) / total_tests if total_tests > 0 else 0
        last_test_volume = tests[-1]['volume'] if tests else 0
        last_test_time = tests[-1]['timestamp'] if tests else None

        return {
            'tests': tests,
            'total_tests': total_tests,
            'holds': holds,
            'breaks': breaks,
            'hold_rate': hold_rate,
            'avg_volume': avg_volume,
            'last_test_volume': last_test_volume,
            'last_test_time': last_test_time
        }

    def _analyze_volume_at_tests(self, test_data: Dict) -> Dict:
        """
        Analyze volume trends at level tests

        Returns:
            Dict with volume analysis
        """
        tests = test_data['tests']

        if len(tests) < 2:
            return {
                'trend': 'INSUFFICIENT_DATA',
                'increasing': False,
                'decreasing': False
            }

        # Compare first half vs second half
        mid_point = len(tests) // 2
        first_half_avg = sum(t['volume'] for t in tests[:mid_point]) / mid_point if mid_point > 0 else 0
        second_half_avg = sum(t['volume'] for t in tests[mid_point:]) / (len(tests) - mid_point) if len(tests) > mid_point else 0

        if second_half_avg > first_half_avg * 1.3:
            trend = 'INCREASING'
            increasing = True
            decreasing = False
        elif second_half_avg < first_half_avg * 0.7:
            trend = 'DECREASING'
            increasing = False
            decreasing = True
        else:
            trend = 'STABLE'
            increasing = False
            decreasing = False

        return {
            'trend': trend,
            'increasing': increasing,
            'decreasing': decreasing,
            'first_half_avg': first_half_avg,
            'second_half_avg': second_half_avg
        }

    def _analyze_test_spacing(self, test_data: Dict) -> Dict:
        """
        Analyze spacing between tests
        Closer spacing = level being tested more frequently = potential weakness

        Returns:
            Dict with spacing analysis
        """
        tests = test_data['tests']

        if len(tests) < 2:
            return {
                'avg_spacing_hours': 0,
                'trend': 'INSUFFICIENT_DATA',
                'getting_closer': False
            }

        # Calculate time between consecutive tests
        spacings = []
        for i in range(1, len(tests)):
            time_diff = tests[i]['timestamp'] - tests[i-1]['timestamp']
            hours = time_diff.total_seconds() / 3600
            spacings.append(hours)

        avg_spacing_hours = np.mean(spacings) if spacings else 0

        # Check if tests are getting closer together
        if len(spacings) >= 4:
            first_half_spacing = np.mean(spacings[:len(spacings)//2])
            second_half_spacing = np.mean(spacings[len(spacings)//2:])

            if second_half_spacing < first_half_spacing * 0.7:
                trend = 'GETTING_CLOSER'
                getting_closer = True
            elif second_half_spacing > first_half_spacing * 1.3:
                trend = 'GETTING_WIDER'
                getting_closer = False
            else:
                trend = 'STABLE'
                getting_closer = False
        else:
            trend = 'INSUFFICIENT_DATA'
            getting_closer = False

        return {
            'avg_spacing_hours': avg_spacing_hours,
            'trend': trend,
            'getting_closer': getting_closer
        }

    def _calculate_score(self,
                        test_data: Dict,
                        volume_analysis: Dict,
                        spacing_analysis: Dict) -> float:
        """
        Calculate overall strength score (0-100)

        Factors:
        - Hold rate (40% weight)
        - Number of tests (15% weight)
        - Volume trend (25% weight)
        - Test spacing (20% weight)
        """
        score = 0

        # 1. Hold rate component (40 points max)
        hold_rate_score = (test_data['hold_rate'] / 100) * 40
        score += hold_rate_score

        # 2. Test count component (15 points max)
        # More tests = more reliable (capped at 7 tests)
        test_count_score = min(test_data['total_tests'] / 7, 1.0) * 15
        score += test_count_score

        # 3. Volume trend component (25 points max)
        if volume_analysis['trend'] == 'INCREASING':
            volume_score = 25  # Strong participation
        elif volume_analysis['trend'] == 'STABLE':
            volume_score = 18
        elif volume_analysis['trend'] == 'DECREASING':
            volume_score = 8  # Weakening interest
        else:
            volume_score = 12
        score += volume_score

        # 4. Test spacing component (20 points max)
        # Wider spacing = stronger (level holds longer)
        if spacing_analysis['trend'] == 'GETTING_WIDER':
            spacing_score = 20  # Level holding well
        elif spacing_analysis['trend'] == 'STABLE':
            spacing_score = 15
        elif spacing_analysis['trend'] == 'GETTING_CLOSER':
            spacing_score = 5  # Being tested frequently = weakness
        else:
            spacing_score = 10
        score += spacing_score

        return min(score, 100)

    def _determine_trend(self,
                        test_data: Dict,
                        volume_analysis: Dict,
                        spacing_analysis: Dict) -> str:
        """
        Determine if level is strengthening, weakening, or stable

        Returns:
            'STRENGTHENING', 'WEAKENING', or 'STABLE'
        """
        tests = test_data['tests']

        if len(tests) < 3:
            return 'INSUFFICIENT_DATA'

        # Analyze recent tests (last 3)
        recent_tests = tests[-3:]
        recent_holds = sum(1 for t in recent_tests if t['type'] == 'hold')
        recent_hold_rate = (recent_holds / len(recent_tests)) * 100

        overall_hold_rate = test_data['hold_rate']

        # Strengthening conditions:
        # - Recent holds > overall rate
        # - Volume increasing
        # - Tests getting wider apart
        strengthening_signals = 0
        if recent_hold_rate >= overall_hold_rate:
            strengthening_signals += 1
        if volume_analysis['increasing']:
            strengthening_signals += 1
        if spacing_analysis['trend'] == 'GETTING_WIDER':
            strengthening_signals += 1

        # Weakening conditions:
        weakening_signals = 0
        if recent_hold_rate < overall_hold_rate:
            weakening_signals += 1
        if volume_analysis['decreasing']:
            weakening_signals += 1
        if spacing_analysis['getting_closer']:
            weakening_signals += 1

        if strengthening_signals >= 2:
            return 'STRENGTHENING'
        elif weakening_signals >= 2:
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
            'times_held': 0,
            'times_broken': 0,
            'hold_rate': 0.0,
            'avg_test_volume': 0.0,
            'last_test_volume': 0,
            'volume_trend': 'INSUFFICIENT_DATA',
            'avg_test_spacing_hours': 0,
            'spacing_trend': 'INSUFFICIENT_DATA',
            'is_strengthening': False,
            'is_weakening': False,
            'trend': 'INSUFFICIENT_DATA',
            'confidence': 'LOW',
            'last_test_time': None
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
