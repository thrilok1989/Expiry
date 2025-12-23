"""
MASTER DECISION ENGINE - Single Source of Truth for All Trading Decisions

This is the ONLY authority that decides:
- TRADE / WAIT / SCAN
- Which strategy is active
- Entry permissions
- Position sizing
- All other modules are ADVISORY ONLY - this engine makes the final call

NO CHERRY PICKING. NO CONTRADICTIONS. ONE VERDICT.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TradeState(Enum):
    """Master trade states - ONLY these 3 options"""
    TRADE = "TRADE"      # Full execution allowed
    WAIT = "WAIT"        # All entries locked - wait for better conditions
    SCAN = "SCAN"        # Paper trading / observation only


class ActiveStrategy(Enum):
    """Only ONE strategy can be active at a time"""
    RANGE_SCALPING = "RANGE_SCALPING"
    TREND_FOLLOWING = "TREND_FOLLOWING"
    REVERSAL_TRADING = "REVERSAL_TRADING"
    BREAKOUT_TRADING = "BREAKOUT_TRADING"
    NONE = "NONE"  # No strategy active


@dataclass
class MasterDecision:
    """
    THE SINGLE SOURCE OF TRUTH
    All other displays must defer to this
    """
    # MASTER STATE (NON-NEGOTIABLE)
    state: TradeState

    # Active strategy (only one)
    active_strategy: ActiveStrategy

    # Confidence (capped at 90%)
    confidence: float  # 0-90

    # Entry permission
    entries_enabled: bool
    option_buying_allowed: bool

    # Position sizing
    position_multiplier: float  # 0.25x, 0.5x, 1x, 1.5x
    max_risk_per_trade: float  # % of capital

    # Primary setup (ONLY ONE)
    primary_setup: Optional[Dict]

    # Reason (WHY this decision)
    reason: str
    lock_reason: str  # If locked, WHY

    # System locks
    hard_blocks: List[str]  # List of blocking conditions

    # Alerts
    send_telegram: bool
    telegram_message: str


class MasterDecisionEngine:
    """
    THE SINGLE AUTHORITY FOR ALL TRADING DECISIONS

    All analysis is input.
    This engine makes the FINAL call.
    No contradictions allowed.
    """

    def __init__(self):
        self.decision_version = "v1.0_authority"

    def make_decision(
        self,
        current_price: float,
        confidence_score: float,
        nearest_support: Optional[Dict],
        nearest_resistance: Optional[Dict],
        regime: Optional[str],
        atm_verdict: Optional[str],
        volume_available: bool,
        flow_available: bool,
        market_depth_available: bool,
        vob_levels: Optional[Dict],
        in_entry_zone: bool,
        distance_to_support: float,
        distance_to_resistance: float,
        session_time: str,
        losing_trades_today: int = 0
    ) -> MasterDecision:
        """
        MAKE THE FINAL DECISION

        This is the ONLY function that decides what to do.
        All contradictions resolved here.
        """

        # === HARD BLOCKS (NON-NEGOTIABLE) ===
        hard_blocks = []

        # BLOCK 1: Volume/Flow missing = NO OPTION BUYING
        if not volume_available or not flow_available:
            hard_blocks.append("Volume/Flow data unavailable - Option buying disabled")

        # BLOCK 2: 2+ losing trades today = STOP
        if losing_trades_today >= 2:
            hard_blocks.append(f"2+ losing trades today ({losing_trades_today}) - Trading stopped for the day")

        # BLOCK 3: Market depth unavailable = NO SCALPING
        if not market_depth_available:
            hard_blocks.append("Market depth unavailable - Scalping disabled")

        # BLOCK 4: Mid-session low momentum = WAIT
        if session_time == "MID-SESSION":
            hard_blocks.append("Mid-session low momentum period - Wait for power hours")

        # === CONFIDENCE-BASED PERMISSIONS ===
        # Cap confidence at 90% (never show 100%)
        capped_confidence = min(confidence_score, 90.0)

        # Confidence-based state
        if capped_confidence < 60:
            base_state = TradeState.SCAN
            position_multiplier = 0.0  # Paper only
            option_buying_allowed = False
        elif 60 <= capped_confidence < 75:
            base_state = TradeState.TRADE
            position_multiplier = 0.5  # Half size - SCALP ONLY
            option_buying_allowed = True if not hard_blocks else False
        else:  # 75+
            base_state = TradeState.TRADE
            position_multiplier = 1.0  # Full size
            option_buying_allowed = True if not hard_blocks else False

        # === MID-ZONE CHECK (OVERRIDE) ===
        # If not in entry zone, FORCE WAIT
        if not in_entry_zone:
            if distance_to_support > 5 and distance_to_resistance > 5:
                hard_blocks.append(f"Mid-Zone: {distance_to_support:.0f}pts from support, {distance_to_resistance:.0f}pts from resistance")
                base_state = TradeState.WAIT
                option_buying_allowed = False
                position_multiplier = 0.0

        # === DETERMINE ACTIVE STRATEGY ===
        active_strategy = self._determine_active_strategy(
            regime=regime,
            distance_to_support=distance_to_support,
            distance_to_resistance=distance_to_resistance,
            in_entry_zone=in_entry_zone,
            confidence=capped_confidence
        )

        # If no valid strategy, FORCE WAIT
        if active_strategy == ActiveStrategy.NONE:
            hard_blocks.append("No valid strategy detected - Wait for clear setup")
            base_state = TradeState.WAIT
            option_buying_allowed = False
            position_multiplier = 0.0

        # === FINAL STATE (APPLY ALL BLOCKS) ===
        if hard_blocks:
            final_state = TradeState.WAIT
            entries_enabled = False
            option_buying_allowed = False
            position_multiplier = 0.0
            lock_reason = " | ".join(hard_blocks)
        else:
            final_state = base_state
            entries_enabled = base_state == TradeState.TRADE
            lock_reason = ""

        # === DETERMINE PRIMARY SETUP (ONLY ONE) ===
        primary_setup = self._select_primary_setup(
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            distance_to_support=distance_to_support,
            distance_to_resistance=distance_to_resistance,
            active_strategy=active_strategy,
            in_entry_zone=in_entry_zone
        )

        # === GENERATE REASON ===
        reason = self._generate_reason(
            state=final_state,
            confidence=capped_confidence,
            active_strategy=active_strategy,
            primary_setup=primary_setup,
            hard_blocks=hard_blocks
        )

        # === TELEGRAM ALERT ===
        send_telegram = False
        telegram_message = ""

        if final_state == TradeState.TRADE and in_entry_zone and primary_setup:
            send_telegram = True
            setup_type = primary_setup.get('type', 'UNKNOWN')
            setup_price = primary_setup.get('price', 0)
            telegram_message = f"🚨 ENTRY SIGNAL: {setup_type} at ₹{setup_price:,.0f} | Confidence: {capped_confidence:.0f}% | Strategy: {active_strategy.value}"

        # === RISK MANAGEMENT ===
        if capped_confidence < 60:
            max_risk = 0.0  # Paper only
        elif capped_confidence < 75:
            max_risk = 0.25  # 0.25% per trade
        else:
            max_risk = 0.5  # 0.5% per trade

        # === CONSTRUCT DECISION ===
        return MasterDecision(
            state=final_state,
            active_strategy=active_strategy,
            confidence=capped_confidence,
            entries_enabled=entries_enabled,
            option_buying_allowed=option_buying_allowed,
            position_multiplier=position_multiplier,
            max_risk_per_trade=max_risk,
            primary_setup=primary_setup,
            reason=reason,
            lock_reason=lock_reason,
            hard_blocks=hard_blocks,
            send_telegram=send_telegram,
            telegram_message=telegram_message
        )

    def _determine_active_strategy(
        self,
        regime: Optional[str],
        distance_to_support: float,
        distance_to_resistance: float,
        in_entry_zone: bool,
        confidence: float
    ) -> ActiveStrategy:
        """
        Determine which ONE strategy should be active
        ONLY ONE at a time
        """

        if not regime:
            return ActiveStrategy.NONE

        # RANGING market = Range Scalping
        if "RANGING" in regime.upper() or "RANGE" in regime.upper():
            if in_entry_zone:
                return ActiveStrategy.RANGE_SCALPING
            else:
                return ActiveStrategy.NONE  # Wait for entry zone

        # TRENDING market = Trend Following
        elif "TRENDING" in regime.upper() or "TREND" in regime.upper():
            if confidence >= 70:
                return ActiveStrategy.TREND_FOLLOWING
            else:
                return ActiveStrategy.NONE  # Confidence too low

        # BREAKOUT market = Breakout Trading
        elif "BREAKOUT" in regime.upper():
            if confidence >= 75:
                return ActiveStrategy.BREAKOUT_TRADING
            else:
                return ActiveStrategy.NONE

        # Default: None
        return ActiveStrategy.NONE

    def _select_primary_setup(
        self,
        nearest_support: Optional[Dict],
        nearest_resistance: Optional[Dict],
        distance_to_support: float,
        distance_to_resistance: float,
        active_strategy: ActiveStrategy,
        in_entry_zone: bool
    ) -> Optional[Dict]:
        """
        Select THE PRIMARY SETUP (only one)
        All others are secondary/ignored
        """

        if not in_entry_zone:
            return None

        # Pick the nearest level
        if distance_to_support < distance_to_resistance:
            if nearest_support and distance_to_support <= 5:
                return {
                    'direction': 'LONG',
                    'type': nearest_support.get('type', 'Support'),
                    'price': nearest_support.get('price', 0),
                    'entry_zone_low': nearest_support.get('lower', 0),
                    'entry_zone_high': nearest_support.get('upper', 0),
                    'distance': distance_to_support
                }
        else:
            if nearest_resistance and distance_to_resistance <= 5:
                return {
                    'direction': 'SHORT',
                    'type': nearest_resistance.get('type', 'Resistance'),
                    'price': nearest_resistance.get('price', 0),
                    'entry_zone_low': nearest_resistance.get('lower', 0),
                    'entry_zone_high': nearest_resistance.get('upper', 0),
                    'distance': distance_to_resistance
                }

        return None

    def _generate_reason(
        self,
        state: TradeState,
        confidence: float,
        active_strategy: ActiveStrategy,
        primary_setup: Optional[Dict],
        hard_blocks: List[str]
    ) -> str:
        """Generate clear reason for the decision"""

        if state == TradeState.WAIT:
            if hard_blocks:
                return f"System locked: {hard_blocks[0]}"
            else:
                return "Confidence below threshold - Wait for better setup"

        elif state == TradeState.SCAN:
            return f"Low confidence ({confidence:.0f}%) - Paper trading mode only"

        elif state == TradeState.TRADE:
            if primary_setup:
                direction = primary_setup.get('direction', 'UNKNOWN')
                level_type = primary_setup.get('type', 'Unknown')
                price = primary_setup.get('price', 0)
                return f"{direction} setup active at {level_type} ₹{price:,.0f} | Strategy: {active_strategy.value} | Confidence: {confidence:.0f}%"
            else:
                return f"Trade mode active | Strategy: {active_strategy.value} | Confidence: {confidence:.0f}%"

        return "Unknown state"


def format_master_decision(decision: MasterDecision) -> str:
    """
    Format the master decision for display
    THIS IS THE ONLY DISPLAY THAT MATTERS
    """

    # State color
    if decision.state == TradeState.TRADE:
        state_color = "🟢"
        state_bg = "#1a4d1a"
    elif decision.state == TradeState.SCAN:
        state_color = "🟡"
        state_bg = "#4d4d1a"
    else:  # WAIT
        state_color = "🔴"
        state_bg = "#4d1a1a"

    output = f"""
<div style='background: {state_bg}; padding: 20px; border-radius: 10px; margin: 20px 0; border: 3px solid {"#00ff00" if decision.state == TradeState.TRADE else "#ffaa00" if decision.state == TradeState.SCAN else "#ff0000"}'>
    <h1 style='margin: 0 0 15px 0; color: white;'>{state_color} SYSTEM VERDICT: {decision.state.value}</h1>

    <div style='background: #0a0a0a; padding: 15px; border-radius: 8px; margin: 10px 0;'>
        <p style='margin: 5px 0; color: #ffffff; font-size: 18px;'><strong>📊 Confidence:</strong> {decision.confidence:.0f}% (Capped at 90%)</p>
        <p style='margin: 5px 0; color: #ffffff; font-size: 18px;'><strong>⚙️ Active Strategy:</strong> {decision.active_strategy.value}</p>
        <p style='margin: 5px 0; color: #ffffff; font-size: 18px;'><strong>🎯 Position Size:</strong> {decision.position_multiplier}x | Max Risk: {decision.max_risk_per_trade:.2f}%</p>
    </div>
"""

    # Show locks if WAIT
    if decision.state == TradeState.WAIT:
        output += f"""
    <div style='background: #4d1a1a; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 5px solid #ff0000;'>
        <p style='margin: 0; color: #ffffff; font-size: 16px;'><strong>🔒 SYSTEM LOCK:</strong> {decision.lock_reason}</p>
    </div>

    <div style='background: #2a2a2a; padding: 15px; border-radius: 8px; margin: 10px 0;'>
        <p style='margin: 0; color: #ff6666; font-size: 16px;'><strong>❌ ENTRIES DISABLED</strong></p>
        <p style='margin: 5px 0; color: #cccccc;'>Reason: {decision.reason}</p>
"""
        if decision.hard_blocks:
            output += "<ul style='margin: 10px 0; color: #cccccc;'>"
            for block in decision.hard_blocks:
                output += f"<li>{block}</li>"
            output += "</ul>"
        output += "</div>"

    # Show primary setup if TRADE
    elif decision.state == TradeState.TRADE and decision.primary_setup:
        setup = decision.primary_setup
        output += f"""
    <div style='background: #1a4d1a; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 5px solid #00ff00;'>
        <p style='margin: 0; color: #ffffff; font-size: 18px;'><strong>⭐ PRIMARY SETUP TODAY:</strong></p>
        <p style='margin: 5px 0; color: #00ff88; font-size: 20px;'><strong>{setup['direction']} at {setup['type']}</strong></p>
        <p style='margin: 5px 0; color: #ffffff;'>Entry Zone: ₹{setup['entry_zone_low']:,.0f} - ₹{setup['entry_zone_high']:,.0f}</p>
        <p style='margin: 5px 0; color: #ffffff;'>Distance: {setup['distance']:.0f} points ({setup['direction']} setup)</p>
    </div>

    <div style='background: #2a2a2a; padding: 15px; border-radius: 8px; margin: 10px 0;'>
        <p style='margin: 0; color: #00ff88; font-size: 16px;'><strong>✅ ENTRIES ENABLED</strong></p>
        <p style='margin: 5px 0; color: #cccccc;'>{decision.reason}</p>
        <p style='margin: 5px 0; color: #ffaa00;'>⚠️ All other setups: SECONDARY / IGNORE</p>
    </div>
"""

    # Show SCAN mode
    elif decision.state == TradeState.SCAN:
        output += f"""
    <div style='background: #4d4d1a; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 5px solid #ffaa00;'>
        <p style='margin: 0; color: #ffffff; font-size: 16px;'><strong>📊 PAPER TRADING MODE</strong></p>
        <p style='margin: 5px 0; color: #cccccc;'>Reason: {decision.reason}</p>
        <p style='margin: 5px 0; color: #ffaa00;'>💡 Use this time to observe and refine your edge</p>
    </div>
"""

    output += """
    <div style='background: #1a1a2e; padding: 10px; border-radius: 8px; margin: 10px 0;'>
        <p style='margin: 0; color: #888888; font-size: 14px;'>🧠 This is the SINGLE AUTHORITY. All other displays are advisory only.</p>
    </div>
</div>
"""

    return output
