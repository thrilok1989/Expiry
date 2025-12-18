"""
VOB-Based Signal Generator

Generates trading signals based on:
1. Overall market sentiment (BULL/BEAR)
2. Spot price proximity to Volume Order Blocks (within 7 points)
3. Automatic entry and stop loss calculation
4. Volume Order Block strength analysis
"""

from datetime import datetime
import pytz
from config import IST, get_current_time_ist
from typing import Dict, List, Optional
import pandas as pd
from indicators.vob_strength_tracker import VOBStrengthTracker, get_emoji_for_strength, get_description_for_strength


class VOBSignalGenerator:
    """
    Generates trading signals based on VOB levels and market sentiment
    """

    def __init__(self, proximity_threshold: float = 7.0):
        """
        Initialize VOB Signal Generator

        Args:
            proximity_threshold: Distance from VOB to trigger signal (default 7 points)
        """
        self.proximity_threshold = proximity_threshold
        self.last_signal = None
        self.signal_history = []
        self.strength_tracker = VOBStrengthTracker(respect_distance=5.0)

    def check_for_signal(self,
                        spot_price: float,
                        market_sentiment: str,
                        bullish_blocks: List[Dict],
                        bearish_blocks: List[Dict],
                        index: str = "NIFTY",
                        df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """
        Check if conditions are met for a trading signal

        Args:
            spot_price: Current spot price
            market_sentiment: Overall market sentiment ("BULLISH", "BEARISH", "NEUTRAL")
            bullish_blocks: List of bullish VOB blocks
            bearish_blocks: List of bearish VOB blocks
            index: Index name (NIFTY or SENSEX)
            df: Price dataframe for strength analysis (optional)

        Returns:
            Signal dictionary if conditions met, None otherwise
        """
        signal = None

        # Check for BULLISH signal
        if market_sentiment == "BULLISH" and bullish_blocks:
            signal = self._check_bullish_signal(spot_price, bullish_blocks, index, df)

        # Check for BEARISH signal
        elif market_sentiment == "BEARISH" and bearish_blocks:
            signal = self._check_bearish_signal(spot_price, bearish_blocks, index, df)

        # Store signal if generated
        if signal:
            self.last_signal = signal
            self.signal_history.append(signal)
            # Keep only last 50 signals
            if len(self.signal_history) > 50:
                self.signal_history = self.signal_history[-50:]

        return signal

    def _check_bullish_signal(self, spot_price: float, bullish_blocks: List[Dict], index: str, df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """
        Check for bullish entry signal

        Conditions:
        - Spot price is above and within proximity threshold of a bullish VOB
        - Market sentiment is BULLISH

        Returns:
            Signal dict with entry, stop loss, and target levels
        """
        for block in reversed(bullish_blocks):  # Check most recent blocks first
            vob_upper = block['upper']
            vob_lower = block['lower']
            vob_mid = block['mid']

            # Check if price is near the VOB (within proximity threshold)
            # Price should be above VOB upper or within the block
            distance_from_upper = spot_price - vob_upper

            # Signal condition: Price is above VOB and within proximity threshold of upper level
            if 0 <= distance_from_upper <= self.proximity_threshold:
                # Calculate levels
                entry_price = spot_price
                stop_loss = vob_lower - self.proximity_threshold

                # Target is 1.5x the risk (simple R:R of 1:1.5)
                risk = entry_price - stop_loss
                target = entry_price + (risk * 1.5)

                # Calculate strength if dataframe provided
                strength_data = None
                if df is not None and len(df) > 0:
                    strength_data = self.strength_tracker.calculate_strength(block, df)

                signal = {
                    'index': index,
                    'direction': 'CALL',
                    'signal_type': 'VOB_BULL_ENTRY',
                    'entry_price': round(entry_price, 2),
                    'stop_loss': round(stop_loss, 2),
                    'target': round(target, 2),
                    'vob_level': round(vob_upper, 2),
                    'vob_lower': round(vob_lower, 2),
                    'vob_upper': round(vob_upper, 2),
                    'vob_volume': block['volume'],
                    'distance_from_vob': round(distance_from_upper, 2),
                    'risk_reward': '1:1.5',
                    'market_sentiment': 'BULLISH',
                    'timestamp': get_current_time_ist(),
                    'status': 'ACTIVE',
                    'strength': strength_data  # Add strength analysis
                }

                return signal

        return None

    def _check_bearish_signal(self, spot_price: float, bearish_blocks: List[Dict], index: str, df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """
        Check for bearish entry signal

        Conditions:
        - Spot price is below and within proximity threshold of a bearish VOB
        - Market sentiment is BEARISH

        Returns:
            Signal dict with entry, stop loss, and target levels
        """
        for block in reversed(bearish_blocks):  # Check most recent blocks first
            vob_upper = block['upper']
            vob_lower = block['lower']
            vob_mid = block['mid']

            # Check if price is near the VOB (within proximity threshold)
            # Price should be below VOB lower or within the block
            distance_from_lower = vob_lower - spot_price

            # Signal condition: Price is below VOB and within proximity threshold of lower level
            if 0 <= distance_from_lower <= self.proximity_threshold:
                # Calculate levels
                entry_price = spot_price
                stop_loss = vob_upper + self.proximity_threshold

                # Target is 1.5x the risk (simple R:R of 1:1.5)
                risk = stop_loss - entry_price
                target = entry_price - (risk * 1.5)

                # Calculate strength if dataframe provided
                strength_data = None
                if df is not None and len(df) > 0:
                    strength_data = self.strength_tracker.calculate_strength(block, df)

                signal = {
                    'index': index,
                    'direction': 'PUT',
                    'signal_type': 'VOB_BEAR_ENTRY',
                    'entry_price': round(entry_price, 2),
                    'stop_loss': round(stop_loss, 2),
                    'target': round(target, 2),
                    'vob_level': round(vob_lower, 2),
                    'vob_lower': round(vob_lower, 2),
                    'vob_upper': round(vob_upper, 2),
                    'vob_volume': block['volume'],
                    'distance_from_vob': round(distance_from_lower, 2),
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
