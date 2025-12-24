"""
Support/Resistance Strength Tracker with ML-based Trend Analysis

Tracks S/R strength over time and detects:
1. Whether support/resistance is getting stronger or weaker
2. Support-to-Resistance transitions (and vice versa)
3. Historical strength patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class SRStrengthPoint:
    """Single observation of S/R strength at a point in time"""
    timestamp: datetime
    price_level: float
    level_type: str  # 'support' or 'resistance'
    strength: float  # 0-100
    status: str  # 'BUILDING', 'TESTING', 'BREAKING', 'NEUTRAL'
    factors: List[str]  # Reasons for the strength
    volume: float = 0.0
    touch_count: int = 0  # Number of times price touched this level


@dataclass
class SRTransition:
    """Detected support-to-resistance (or vice versa) transition"""
    transition_type: str  # 'SUPPORT_TO_RESISTANCE' or 'RESISTANCE_TO_SUPPORT'
    price_level: float
    start_time: datetime
    end_time: datetime
    confidence: float  # 0-100
    strength_before: float
    strength_after: float
    reason: str


@dataclass
class SRStrengthTrend:
    """Trend analysis for a S/R level"""
    level_type: str  # 'support' or 'resistance'
    price_level: float
    current_strength: float
    trend: str  # 'STRENGTHENING', 'WEAKENING', 'STABLE', 'TRANSITIONING'
    trend_confidence: float  # 0-100
    strength_change_rate: float  # Points per hour
    prediction_1h: float  # Predicted strength in 1 hour
    prediction_4h: float  # Predicted strength in 4 hours
    reasons: List[str]


class SRStrengthTracker:
    """
    ML-based tracker for Support/Resistance strength over time

    Features:
    - Maintains historical strength observations
    - Detects strengthening/weakening trends
    - Identifies support-to-resistance transitions
    - Provides ML-based predictions
    """

    def __init__(self, max_history_hours: int = 24, transition_threshold: float = 0.6):
        """
        Initialize S/R Strength Tracker

        Args:
            max_history_hours: How many hours of history to keep
            transition_threshold: Minimum confidence to detect transitions (0-1)
        """
        self.max_history_hours = max_history_hours
        self.transition_threshold = transition_threshold

        # Historical data storage
        self.history: deque = deque(maxlen=500)  # Last 500 observations
        self.transitions: List[SRTransition] = []

        # Level tracking (price -> list of observations)
        self.level_observations: Dict[float, List[SRStrengthPoint]] = {}

    def add_observation(
        self,
        price_level: float,
        level_type: str,
        strength: float,
        status: str,
        factors: List[str],
        current_price: float = 0.0,
        volume: float = 0.0
    ):
        """
        Add a new S/R strength observation

        Args:
            price_level: The S/R price level
            level_type: 'support' or 'resistance'
            strength: Strength score (0-100)
            status: 'BUILDING', 'TESTING', 'BREAKING', 'NEUTRAL'
            factors: List of reasons for this strength
            current_price: Current market price
            volume: Volume at this level
        """
        observation = SRStrengthPoint(
            timestamp=datetime.now(),
            price_level=price_level,
            level_type=level_type,
            strength=strength,
            status=status,
            factors=factors,
            volume=volume,
            touch_count=self._count_touches(price_level, current_price)
        )

        # Add to history
        self.history.append(observation)

        # Add to level-specific tracking
        rounded_level = round(price_level / 50) * 50  # Round to nearest 50
        if rounded_level not in self.level_observations:
            self.level_observations[rounded_level] = []
        self.level_observations[rounded_level].append(observation)

        # Clean old data
        self._clean_old_observations()

        # Check for transitions
        self._detect_transitions(rounded_level)

    def _count_touches(self, price_level: float, current_price: float) -> int:
        """Count how many times price has touched this level recently"""
        if current_price == 0:
            return 0

        rounded_level = round(price_level / 50) * 50
        if rounded_level not in self.level_observations:
            return 0

        # Count observations within last 4 hours
        cutoff_time = datetime.now() - timedelta(hours=4)
        recent_obs = [
            obs for obs in self.level_observations[rounded_level]
            if obs.timestamp >= cutoff_time
        ]

        return len(recent_obs)

    def _clean_old_observations(self):
        """Remove observations older than max_history_hours"""
        cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)

        # Clean level observations
        for level in list(self.level_observations.keys()):
            self.level_observations[level] = [
                obs for obs in self.level_observations[level]
                if obs.timestamp >= cutoff_time
            ]

            # Remove level if no recent observations
            if not self.level_observations[level]:
                del self.level_observations[level]

    def _detect_transitions(self, price_level: float):
        """
        Detect if a level has transitioned from support to resistance or vice versa

        A transition is detected when:
        1. A level was acting as support, then price broke above it, and now it's resistance
        2. A level was acting as resistance, then price broke below it, and now it's support
        """
        if price_level not in self.level_observations:
            return

        observations = self.level_observations[price_level]
        if len(observations) < 3:
            return

        # Get recent observations (last 6 hours)
        cutoff = datetime.now() - timedelta(hours=6)
        recent = [obs for obs in observations if obs.timestamp >= cutoff]

        if len(recent) < 3:
            return

        # Check for type change
        old_obs = recent[:len(recent)//2]
        new_obs = recent[len(recent)//2:]

        old_type = self._get_dominant_type(old_obs)
        new_type = self._get_dominant_type(new_obs)

        # Detect transition
        if old_type != new_type and old_type in ['support', 'resistance'] and new_type in ['support', 'resistance']:
            # Calculate confidence based on strength changes
            old_strength = np.mean([obs.strength for obs in old_obs])
            new_strength = np.mean([obs.strength for obs in new_obs])

            # Confidence increases if new role has higher strength
            confidence = min(100, (new_strength / max(old_strength, 1)) * 60)

            if confidence >= self.transition_threshold * 100:
                transition_type = f"{old_type.upper()}_TO_{new_type.upper()}"

                transition = SRTransition(
                    transition_type=transition_type,
                    price_level=price_level,
                    start_time=old_obs[0].timestamp,
                    end_time=new_obs[-1].timestamp,
                    confidence=confidence,
                    strength_before=old_strength,
                    strength_after=new_strength,
                    reason=f"Level ₹{price_level:.0f} changed from {old_type} to {new_type} (confidence: {confidence:.0f}%)"
                )

                # Add if not duplicate
                if not self._is_duplicate_transition(transition):
                    self.transitions.append(transition)
                    logger.info(f"Detected transition: {transition.reason}")

    def _get_dominant_type(self, observations: List[SRStrengthPoint]) -> str:
        """Get dominant level type from observations"""
        if not observations:
            return 'unknown'

        type_counts = {}
        for obs in observations:
            type_counts[obs.level_type] = type_counts.get(obs.level_type, 0) + 1

        return max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'unknown'

    def _is_duplicate_transition(self, new_transition: SRTransition) -> bool:
        """Check if this transition was already detected"""
        for existing in self.transitions[-5:]:  # Check last 5 transitions
            if (existing.price_level == new_transition.price_level and
                existing.transition_type == new_transition.transition_type and
                abs((existing.end_time - new_transition.end_time).total_seconds()) < 3600):
                return True
        return False

    def analyze_trend(
        self,
        price_level: float,
        level_type: str
    ) -> Optional[SRStrengthTrend]:
        """
        Analyze strength trend for a specific S/R level

        Returns:
            SRStrengthTrend with trend analysis and predictions
        """
        rounded_level = round(price_level / 50) * 50

        if rounded_level not in self.level_observations:
            return None

        observations = self.level_observations[rounded_level]

        # Filter by type
        type_obs = [obs for obs in observations if obs.level_type == level_type]

        if len(type_obs) < 2:
            return None

        # Sort by time
        type_obs.sort(key=lambda x: x.timestamp)

        # Calculate trend using linear regression on strength values
        strengths = np.array([obs.strength for obs in type_obs])
        times = np.array([(obs.timestamp - type_obs[0].timestamp).total_seconds() / 3600 for obs in type_obs])

        # Simple linear regression
        if len(times) > 1:
            slope, intercept = np.polyfit(times, strengths, 1)
        else:
            slope, intercept = 0, strengths[0]

        # Determine trend
        current_strength = type_obs[-1].strength
        strength_change_rate = slope  # Points per hour

        if abs(strength_change_rate) < 2:
            trend = 'STABLE'
        elif strength_change_rate > 5:
            trend = 'STRENGTHENING'
        elif strength_change_rate < -5:
            trend = 'WEAKENING'
        else:
            trend = 'STABLE'

        # Check for transitions
        recent_transitions = [
            t for t in self.transitions
            if t.price_level == rounded_level and
            (datetime.now() - t.end_time).total_seconds() < 3600
        ]
        if recent_transitions:
            trend = 'TRANSITIONING'

        # Calculate trend confidence based on R² and data points
        residuals = strengths - (slope * times + intercept)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((strengths - np.mean(strengths))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        trend_confidence = min(100, max(0, r_squared * 100 * (len(type_obs) / 10)))

        # Predictions
        last_time = times[-1]
        prediction_1h = slope * (last_time + 1) + intercept
        prediction_4h = slope * (last_time + 4) + intercept

        # Clamp predictions
        prediction_1h = np.clip(prediction_1h, 0, 100)
        prediction_4h = np.clip(prediction_4h, 0, 100)

        # Generate reasons
        reasons = []
        if trend == 'STRENGTHENING':
            reasons.append(f"📈 Strength increasing at {strength_change_rate:.1f} pts/hour")
        elif trend == 'WEAKENING':
            reasons.append(f"📉 Strength decreasing at {abs(strength_change_rate):.1f} pts/hour")
        elif trend == 'TRANSITIONING':
            reasons.append(f"🔄 Level transitioning roles (see transitions)")
        else:
            reasons.append(f"➡️ Stable strength around {current_strength:.0f}%")

        if len(type_obs) >= 5:
            reasons.append(f"✅ {len(type_obs)} observations in last {self.max_history_hours}h")

        if type_obs[-1].touch_count > 3:
            reasons.append(f"🎯 Price tested {type_obs[-1].touch_count} times recently")

        return SRStrengthTrend(
            level_type=level_type,
            price_level=price_level,
            current_strength=current_strength,
            trend=trend,
            trend_confidence=trend_confidence,
            strength_change_rate=strength_change_rate,
            prediction_1h=prediction_1h,
            prediction_4h=prediction_4h,
            reasons=reasons
        )

    def get_recent_transitions(self, hours: int = 6) -> List[SRTransition]:
        """Get transitions detected in the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [t for t in self.transitions if t.end_time >= cutoff]

    def get_all_levels_analysis(self) -> List[SRStrengthTrend]:
        """Get trend analysis for all tracked levels"""
        results = []

        for level in self.level_observations.keys():
            # Try both support and resistance
            for level_type in ['support', 'resistance']:
                trend = self.analyze_trend(level, level_type)
                if trend:
                    results.append(trend)

        # Sort by confidence
        results.sort(key=lambda x: x.trend_confidence, reverse=True)

        return results

    def get_summary_stats(self) -> Dict:
        """Get summary statistics about tracked levels"""
        total_obs = len(self.history)
        total_levels = len(self.level_observations)
        total_transitions = len(self.transitions)

        if total_obs == 0:
            return {
                'total_observations': 0,
                'total_levels': 0,
                'total_transitions': 0,
                'avg_strength': 0,
                'strengthening_levels': 0,
                'weakening_levels': 0
            }

        avg_strength = np.mean([obs.strength for obs in self.history])

        # Count strengthening/weakening levels
        all_trends = self.get_all_levels_analysis()
        strengthening = sum(1 for t in all_trends if t.trend == 'STRENGTHENING')
        weakening = sum(1 for t in all_trends if t.trend == 'WEAKENING')

        return {
            'total_observations': total_obs,
            'total_levels': total_levels,
            'total_transitions': total_transitions,
            'avg_strength': avg_strength,
            'strengthening_levels': strengthening,
            'weakening_levels': weakening,
            'observations_last_hour': len([
                obs for obs in self.history
                if (datetime.now() - obs.timestamp).total_seconds() < 3600
            ])
        }


# Example usage
if __name__ == "__main__":
    tracker = SRStrengthTracker()

    # Simulate some observations
    tracker.add_observation(26100, 'support', 70, 'BUILDING', ['Volume increasing'], 26150, 1000000)
    tracker.add_observation(26100, 'support', 75, 'BUILDING', ['Volume increasing', 'OI buildup'], 26140, 1200000)
    tracker.add_observation(26100, 'support', 80, 'BUILDING', ['Volume increasing', 'OI buildup', 'Delta absorption'], 26130, 1500000)

    # Analyze trend
    trend = tracker.analyze_trend(26100, 'support')
    if trend:
        print(f"Trend: {trend.trend}")
        print(f"Confidence: {trend.trend_confidence:.1f}%")
        print(f"Current Strength: {trend.current_strength:.1f}%")
        print(f"Predicted 1h: {trend.prediction_1h:.1f}%")
        print(f"Reasons: {trend.reasons}")
