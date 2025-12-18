"""
HTF Support/Resistance Signal Generator

Generates trading signals based on:
1. Overall market sentiment (BULLISH/BEARISH)
2. Spot price proximity to HTF support/resistance levels (within 5 points)
3. Multiple timeframes: 10min, 15min
4. Automatic entry and stop loss calculation
5. HTF S/R level strength analysis
"""

from datetime import datetime
import pytz
from config import IST, get_current_time_ist
from typing import Dict, List, Optional
import pandas as pd
from indicators.htf_sr_strength_tracker import HTFSRStrengthTracker, get_emoji_for_strength, get_description_for_strength


class HTFSRSignalGenerator:
    """
    Generates trading signals based on HTF Support/Resistance levels and market sentiment
    """

    def __init__(self, proximity_threshold: float = 5.0):
        """
        Initialize HTF S/R Signal Generator

        Args:
            proximity_threshold: Distance from S/R level to trigger signal (default 5 points)
        """
        self.proximity_threshold = proximity_threshold
        self.last_signal = None
        self.signal_history = []
        # Timeframes to monitor (only 10min and 15min)
        self.monitored_timeframes = ['10T', '15T']
        self.strength_tracker = HTFSRStrengthTracker(touch_distance=10.0)

    def check_for_signal(self,
                        spot_price: float,
                        market_sentiment: str,
                        htf_levels: List[Dict],
                        index: str = "NIFTY",
                        df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """
        Check if conditions are met for a trading signal based on HTF S/R levels

        Args:
            spot_price: Current spot price
            market_sentiment: Overall market sentiment ("BULLISH", "BEARISH", "NEUTRAL")
            htf_levels: List of HTF S/R levels from multiple timeframes
            index: Index name (NIFTY or SENSEX)
            df: Price dataframe for strength analysis (optional)

        Returns:
            Signal dictionary if conditions met, None otherwise
        """
        signal = None

        # Filter to only monitored timeframes (10min, 15min)
        filtered_levels = [
            level for level in htf_levels
            if level.get('timeframe') in self.monitored_timeframes
        ]

        # Check for BULLISH signal (price above and near support)
        if market_sentiment == "BULLISH":
            signal = self._check_bullish_signal(spot_price, filtered_levels, index, df)

        # Check for BEARISH signal (price below and near resistance)
        elif market_sentiment == "BEARISH":
            signal = self._check_bearish_signal(spot_price, filtered_levels, index, df)

        # Store signal if generated
        if signal:
            self.last_signal = signal
            self.signal_history.append(signal)
            # Keep only last 50 signals
            if len(self.signal_history) > 50:
                self.signal_history = self.signal_history[-50:]

        return signal

    def _check_bullish_signal(self, spot_price: float, htf_levels: List[Dict], index: str, df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """
        Check for bullish entry signal

        Conditions:
        - Spot price is above and within proximity threshold of HTF support (pivot_low)
        - Market sentiment is BULLISH

        Returns:
            Signal dict with entry, stop loss, and target levels
        """
        for level in reversed(htf_levels):  # Check most recent timeframes first
            pivot_low = level.get('pivot_low')

            if pivot_low is None or pivot_low == 0:
                continue

            # Check if price is above support and within proximity threshold
            distance_from_support = spot_price - pivot_low

            # Signal condition: Price is above support and within proximity threshold
            if 0 <= distance_from_support <= self.proximity_threshold:
                # Calculate levels
                entry_price = spot_price
                stop_loss = pivot_low - self.proximity_threshold

                # Target is 1.5x the risk (simple R:R of 1:1.5)
                risk = entry_price - stop_loss
                target = entry_price + (risk * 1.5)

                # Calculate strength if dataframe provided
                strength_data = None
                if df is not None and len(df) > 0:
                    strength_data = self.strength_tracker.calculate_strength(
                        level=pivot_low,
                        level_type='SUPPORT',
                        df=df
                    )

                signal = {
                    'index': index,
                    'direction': 'CALL',
                    'signal_type': 'HTF_SR_BULL_ENTRY',
                    'entry_price': round(entry_price, 2),
                    'stop_loss': round(stop_loss, 2),
                    'target': round(target, 2),
                    'support_level': round(pivot_low, 2),
                    'resistance_level': round(level.get('pivot_high', 0), 2) if level.get('pivot_high') else None,
                    'timeframe': level.get('timeframe'),
                    'distance_from_level': round(distance_from_support, 2),
                    'risk_reward': '1:1.5',
                    'market_sentiment': 'BULLISH',
                    'timestamp': get_current_time_ist(),
                    'status': 'ACTIVE',
                    'strength': strength_data  # Add strength analysis
                }

                return signal

        return None

    def _check_bearish_signal(self, spot_price: float, htf_levels: List[Dict], index: str, df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """
        Check for bearish entry signal

        Conditions:
        - Spot price is below and within proximity threshold of HTF resistance (pivot_high)
        - Market sentiment is BEARISH

        Returns:
            Signal dict with entry, stop loss, and target levels
        """
        for level in reversed(htf_levels):  # Check most recent timeframes first
            pivot_high = level.get('pivot_high')

            if pivot_high is None or pivot_high == 0:
                continue

            # Check if price is below resistance and within proximity threshold
            distance_from_resistance = pivot_high - spot_price

            # Signal condition: Price is below resistance and within proximity threshold
            if 0 <= distance_from_resistance <= self.proximity_threshold:
                # Calculate levels
                entry_price = spot_price
                stop_loss = pivot_high + self.proximity_threshold

                # Target is 1.5x the risk (simple R:R of 1:1.5)
                risk = stop_loss - entry_price
                target = entry_price - (risk * 1.5)

                # Calculate strength if dataframe provided
                strength_data = None
                if df is not None and len(df) > 0:
                    strength_data = self.strength_tracker.calculate_strength(
                        level=pivot_high,
                        level_type='RESISTANCE',
                        df=df
                    )

                signal = {
                    'index': index,
                    'direction': 'PUT',
                    'signal_type': 'HTF_SR_BEAR_ENTRY',
                    'entry_price': round(entry_price, 2),
                    'stop_loss': round(stop_loss, 2),
                    'target': round(target, 2),
                    'resistance_level': round(pivot_high, 2),
                    'support_level': round(level.get('pivot_low', 0), 2) if level.get('pivot_low') else None,
                    'timeframe': level.get('timeframe'),
                    'distance_from_level': round(distance_from_resistance, 2),
                    'risk_reward': '1:1.5',
                    'market_sentiment': 'BEARISH',
                    'timestamp': get_current_time_ist(),
                    'status': 'ACTIVE',
                    'strength': strength_data  # Add strength analysis
                }

                return signal

        return None

    def get_last_signal(self) -> Optional[Dict]:
        """Get the last generated signal"""
        return self.last_signal

    def get_signal_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent signal history

        Args:
            limit: Number of recent signals to return

        Returns:
            List of recent signals
        """
        return self.signal_history[-limit:]

    def clear_history(self):
        """Clear signal history"""
        self.signal_history = []
        self.last_signal = None
