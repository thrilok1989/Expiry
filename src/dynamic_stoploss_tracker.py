"""
Dynamic Stop-Loss Tracker with Support/Resistance Monitoring

Tracks stop-loss levels based on actual support/resistance levels.
Generates EXIT signals when support/resistance levels change significantly,
indicating that the original trade scenario has changed.
"""

import streamlit as st
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SupportResistanceLevels:
    """Support/Resistance levels at signal generation"""
    support_price: float
    support_type: str  # "VOB Support", "HTF Support", etc.
    support_lower: float
    support_upper: float
    resistance_price: float
    resistance_type: str
    resistance_lower: float
    resistance_upper: float
    timestamp: datetime


@dataclass
class ExitSignalResult:
    """Exit signal result from stop-loss tracker"""
    should_exit: bool
    reason: str
    confidence: float  # 0-100
    scenario_changed: bool
    support_change_pct: float
    resistance_change_pct: float
    current_support: Optional[float] = None
    current_resistance: Optional[float] = None
    original_support: Optional[float] = None
    original_resistance: Optional[float] = None


class DynamicStopLossTracker:
    """
    Dynamic Stop-Loss Tracker

    Monitors support/resistance levels and generates exit signals
    when the original trade scenario changes significantly.
    """

    def __init__(self, change_threshold_pct: float = 0.5):
        """
        Initialize Dynamic Stop-Loss Tracker

        Args:
            change_threshold_pct: Percentage change in S/R that triggers exit signal
        """
        self.change_threshold_pct = change_threshold_pct

    def store_signal_levels(
        self,
        signal_id: str,
        direction: str,
        support_data: Dict,
        resistance_data: Dict
    ) -> None:
        """
        Store support/resistance levels when signal is generated

        Args:
            signal_id: Unique signal identifier
            direction: "LONG" or "SHORT"
            support_data: Dict with support level info
            resistance_data: Dict with resistance level info
        """
        levels = SupportResistanceLevels(
            support_price=support_data.get('price', 0),
            support_type=support_data.get('type', 'Unknown'),
            support_lower=support_data.get('lower', support_data.get('price', 0)),
            support_upper=support_data.get('upper', support_data.get('price', 0)),
            resistance_price=resistance_data.get('price', 0),
            resistance_type=resistance_data.get('type', 'Unknown'),
            resistance_lower=resistance_data.get('lower', resistance_data.get('price', 0)),
            resistance_upper=resistance_data.get('upper', resistance_data.get('price', 0)),
            timestamp=datetime.now()
        )

        # Store in session state
        if 'signal_sr_levels' not in st.session_state:
            st.session_state.signal_sr_levels = {}

        st.session_state.signal_sr_levels[signal_id] = {
            'levels': levels,
            'direction': direction,
            'active': True
        }

        logger.info(f"Stored S/R levels for signal {signal_id}: Support={levels.support_price:.2f}, Resistance={levels.resistance_price:.2f}")

    def check_for_exit_signal(
        self,
        signal_id: str,
        current_support_data: Optional[Dict],
        current_resistance_data: Optional[Dict],
        current_price: float
    ) -> ExitSignalResult:
        """
        Check if support/resistance has changed and exit signal should be generated

        Args:
            signal_id: Signal identifier to check
            current_support_data: Current support level data
            current_resistance_data: Current resistance level data
            current_price: Current spot price

        Returns:
            ExitSignalResult with exit decision and details
        """
        # Check if signal exists
        if 'signal_sr_levels' not in st.session_state or signal_id not in st.session_state.signal_sr_levels:
            return ExitSignalResult(
                should_exit=False,
                reason="Signal not found in tracker",
                confidence=0,
                scenario_changed=False,
                support_change_pct=0,
                resistance_change_pct=0
            )

        signal_data = st.session_state.signal_sr_levels[signal_id]

        # Check if signal is still active
        if not signal_data.get('active', False):
            return ExitSignalResult(
                should_exit=False,
                reason="Signal already closed",
                confidence=0,
                scenario_changed=False,
                support_change_pct=0,
                resistance_change_pct=0
            )

        original_levels = signal_data['levels']
        direction = signal_data['direction']

        # Get current levels
        current_support = current_support_data.get('price', 0) if current_support_data else 0
        current_resistance = current_resistance_data.get('price', 0) if current_resistance_data else 0

        # Calculate changes
        support_change_pct = 0
        resistance_change_pct = 0

        if original_levels.support_price > 0 and current_support > 0:
            support_change_pct = abs((current_support - original_levels.support_price) / original_levels.support_price * 100)

        if original_levels.resistance_price > 0 and current_resistance > 0:
            resistance_change_pct = abs((current_resistance - original_levels.resistance_price) / original_levels.resistance_price * 100)

        # Determine if exit signal should be generated
        should_exit = False
        scenario_changed = False
        reasons = []
        confidence = 0

        # For LONG positions: Monitor support level (stop-loss)
        if direction == "LONG":
            if support_change_pct > self.change_threshold_pct:
                should_exit = True
                scenario_changed = True
                reasons.append(f"Support level changed by {support_change_pct:.2f}% (from {original_levels.support_price:.2f} to {current_support:.2f})")
                confidence += 40

            # Check if support broke
            if current_price < current_support:
                should_exit = True
                reasons.append(f"Price broke below support level {current_support:.2f}")
                confidence += 30

            # Check if resistance changed significantly (scenario shift)
            if resistance_change_pct > self.change_threshold_pct * 2:  # 2x threshold for resistance
                should_exit = True
                scenario_changed = True
                reasons.append(f"Resistance level changed by {resistance_change_pct:.2f}% (from {original_levels.resistance_price:.2f} to {current_resistance:.2f})")
                confidence += 30

        # For SHORT positions: Monitor resistance level (stop-loss)
        elif direction == "SHORT":
            if resistance_change_pct > self.change_threshold_pct:
                should_exit = True
                scenario_changed = True
                reasons.append(f"Resistance level changed by {resistance_change_pct:.2f}% (from {original_levels.resistance_price:.2f} to {current_resistance:.2f})")
                confidence += 40

            # Check if resistance broke
            if current_price > current_resistance:
                should_exit = True
                reasons.append(f"Price broke above resistance level {current_resistance:.2f}")
                confidence += 30

            # Check if support changed significantly (scenario shift)
            if support_change_pct > self.change_threshold_pct * 2:  # 2x threshold for support
                should_exit = True
                scenario_changed = True
                reasons.append(f"Support level changed by {support_change_pct:.2f}% (from {original_levels.support_price:.2f} to {current_support:.2f})")
                confidence += 30

        # Cap confidence at 100
        confidence = min(confidence, 100)

        reason = " | ".join(reasons) if reasons else "No exit conditions met"

        result = ExitSignalResult(
            should_exit=should_exit,
            reason=reason,
            confidence=confidence,
            scenario_changed=scenario_changed,
            support_change_pct=support_change_pct,
            resistance_change_pct=resistance_change_pct,
            current_support=current_support,
            current_resistance=current_resistance,
            original_support=original_levels.support_price,
            original_resistance=original_levels.resistance_price
        )

        if should_exit:
            logger.warning(f"EXIT SIGNAL generated for {signal_id}: {reason}")

        return result

    def close_signal(self, signal_id: str) -> None:
        """
        Mark signal as closed (inactive)

        Args:
            signal_id: Signal identifier to close
        """
        if 'signal_sr_levels' in st.session_state and signal_id in st.session_state.signal_sr_levels:
            st.session_state.signal_sr_levels[signal_id]['active'] = False
            logger.info(f"Signal {signal_id} marked as closed")

    def get_active_signals(self) -> Dict:
        """
        Get all active signals being tracked

        Returns:
            Dict of active signals
        """
        if 'signal_sr_levels' not in st.session_state:
            return {}

        return {
            signal_id: data
            for signal_id, data in st.session_state.signal_sr_levels.items()
            if data.get('active', False)
        }

    def calculate_sr_based_stoploss(
        self,
        direction: str,
        support_data: Dict,
        resistance_data: Dict,
        current_price: float,
        buffer_points: float = 20.0
    ) -> Tuple[float, str]:
        """
        Calculate stop-loss based on support/resistance levels (not percentage)

        Args:
            direction: "LONG" or "SHORT"
            support_data: Support level data
            resistance_data: Resistance level data
            current_price: Current spot price
            buffer_points: Buffer points below support / above resistance

        Returns:
            (stop_loss_price, stop_loss_reason) tuple
        """
        if direction == "LONG":
            # For LONG: Stop-loss below support
            support_lower = support_data.get('lower', support_data.get('price', 0))
            stop_loss = support_lower - buffer_points
            reason = f"Below {support_data.get('type', 'Support')} ({support_lower:.2f} - {buffer_points} pts buffer)"

        elif direction == "SHORT":
            # For SHORT: Stop-loss above resistance
            resistance_upper = resistance_data.get('upper', resistance_data.get('price', 0))
            stop_loss = resistance_upper + buffer_points
            reason = f"Above {resistance_data.get('type', 'Resistance')} ({resistance_upper:.2f} + {buffer_points} pts buffer)"

        else:
            # Fallback to percentage-based (shouldn't happen)
            stop_loss = current_price * 0.97  # 3% default
            reason = "Percentage-based (3%)"

        return round(stop_loss, 2), reason


def get_nearest_support_resistance(
    vob_data: Optional[Dict],
    htf_sr_data: Optional[Dict],
    current_price: float
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Get nearest support and resistance from VOB and HTF S/R data

    Args:
        vob_data: Volume Order Block data
        htf_sr_data: HTF Support/Resistance data
        current_price: Current spot price

    Returns:
        (nearest_support_dict, nearest_resistance_dict) tuple
    """
    nearest_support = None
    nearest_resistance = None
    min_dist_sup = float('inf')
    min_dist_res = float('inf')

    # Check VOB data first (higher priority)
    if vob_data:
        # Find nearest bullish VOB below current price (support)
        bullish_blocks = vob_data.get('bullish_blocks', [])
        for block in bullish_blocks:
            if isinstance(block, dict):
                block_mid = (block.get('upper', 0) + block.get('lower', 0)) / 2
                if block_mid < current_price:
                    dist = current_price - block_mid
                    if dist < min_dist_sup:
                        min_dist_sup = dist
                        nearest_support = {
                            'price': block_mid,
                            'type': 'VOB Support',
                            'lower': block.get('lower', block_mid),
                            'upper': block.get('upper', block_mid),
                            'strength': block.get('strength', 'Medium')
                        }

        # Find nearest bearish VOB above current price (resistance)
        bearish_blocks = vob_data.get('bearish_blocks', [])
        for block in bearish_blocks:
            if isinstance(block, dict):
                block_mid = (block.get('upper', 0) + block.get('lower', 0)) / 2
                if block_mid > current_price:
                    dist = block_mid - current_price
                    if dist < min_dist_res:
                        min_dist_res = dist
                        nearest_resistance = {
                            'price': block_mid,
                            'type': 'VOB Resistance',
                            'lower': block.get('lower', block_mid),
                            'upper': block.get('upper', block_mid),
                            'strength': block.get('strength', 'Medium')
                        }

    # If no VOB data, check HTF S/R data
    if htf_sr_data and (nearest_support is None or nearest_resistance is None):
        htf_support_list = htf_sr_data.get('support', [])
        htf_resistance_list = htf_sr_data.get('resistance', [])

        # Find nearest HTF support
        if nearest_support is None:
            for level in htf_support_list:
                if isinstance(level, dict):
                    level_price = level.get('price', 0)
                else:
                    level_price = float(level) if level else 0

                if level_price < current_price:
                    dist = current_price - level_price
                    if dist < min_dist_sup:
                        min_dist_sup = dist
                        nearest_support = {
                            'price': level_price,
                            'type': 'HTF Support',
                            'lower': level_price - 5,  # Approx zone
                            'upper': level_price + 5,
                            'strength': 'Medium'
                        }

        # Find nearest HTF resistance
        if nearest_resistance is None:
            for level in htf_resistance_list:
                if isinstance(level, dict):
                    level_price = level.get('price', 0)
                else:
                    level_price = float(level) if level else 0

                if level_price > current_price:
                    dist = level_price - current_price
                    if dist < min_dist_res:
                        min_dist_res = dist
                        nearest_resistance = {
                            'price': level_price,
                            'type': 'HTF Resistance',
                            'lower': level_price - 5,
                            'upper': level_price + 5,
                            'strength': 'Medium'
                        }

    return nearest_support, nearest_resistance
