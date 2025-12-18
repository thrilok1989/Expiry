"""
Advanced Risk Management AI Module
Comprehensive risk management with trailing stops, dynamic adjustments
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskManagementResult:
    """Risk management recommendation"""
    entry_approved: bool
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float]
    break_even_trigger: float
    partial_profit_levels: List[Tuple[float, float]]  # [(price, percent_to_exit)]
    max_holding_time: int  # minutes
    avoid_reasons: List[str]
    risk_score: float  # 0-100 (lower is better)
    recommendation: str
    dynamic_adjustments: Dict


class RiskManagementAI:
    """
    Advanced Risk Management AI

    Features:
    1. Dynamic stop-loss (ATR-based, support-based)
    2. Trailing stop logic
    3. Partial profit taking
    4. Break-even move
    5. Time-based exits
    6. News/event avoidance
    7. Trap zone detection
    8. Risk scoring
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        trailing_atr_multiplier: float = 3.0,
        break_even_rr: float = 1.0
    ):
        """Initialize Risk Management AI"""
        self.atr_multiplier = atr_multiplier
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.break_even_rr = break_even_rr

        # Risk thresholds
        self.max_acceptable_risk_score = 70

        # Market hours (IST)
        self.market_open = time(9, 15)
        self.market_close = time(15, 30)
        self.avoid_first_minutes = 15
        self.avoid_last_minutes = 15

    def evaluate_trade_risk(
        self,
        df: pd.DataFrame,
        entry_price: float,
        direction: str,  # "CALL" or "PUT"
        support_level: Optional[float] = None,
        resistance_level: Optional[float] = None,
        trap_probability: float = 0,
        volatility_regime: str = "Normal",
        news_event_nearby: bool = False,
        time_until_expiry: int = 5,  # days
        current_time: Optional[datetime] = None
    ) -> RiskManagementResult:
        """
        Comprehensive risk evaluation for trade

        Returns approval/rejection with complete risk management plan
        """
        avoid_reasons = []
        dynamic_adjustments = {}

        # 1. Time-based checks
        time_ok, time_reasons = self._check_trading_time(current_time)
        if not time_ok:
            avoid_reasons.extend(time_reasons)

        # 2. News event check
        if news_event_nearby:
            avoid_reasons.append("⚠️ Major news event nearby - High volatility risk")

        # 3. Trap zone check
        if trap_probability > 60:
            avoid_reasons.append(f"🚫 High trap probability ({trap_probability:.0f}%)")

        # 4. Expiry week check
        if time_until_expiry <= 2:
            avoid_reasons.append(f"⏰ Expiry in {time_until_expiry} days - Gamma risk")

        # 5. Volatility regime check
        if volatility_regime == "Extreme Volatility":
            avoid_reasons.append("⚠️ Extreme volatility regime - Unpredictable moves")

        # Calculate risk score
        risk_score = self._calculate_risk_score(
            trap_probability, volatility_regime, news_event_nearby,
            time_until_expiry, df
        )

        # Entry approval
        entry_approved = len(avoid_reasons) == 0 and risk_score < self.max_acceptable_risk_score

        # Calculate Stop Loss
        stop_loss = self._calculate_stop_loss(
            df, entry_price, direction, support_level, resistance_level
        )

        # Calculate Take Profit
        take_profit = self._calculate_take_profit(
            entry_price, stop_loss, direction, resistance_level, support_level
        )

        # Calculate Trailing Stop
        trailing_stop = self._calculate_trailing_stop(df, entry_price, direction)

        # Break-even trigger
        risk_amount = abs(entry_price - stop_loss)
        if direction == "CALL":
            break_even_trigger = entry_price + (risk_amount * self.break_even_rr)
        else:
            break_even_trigger = entry_price - (risk_amount * self.break_even_rr)

        # Partial profit levels
        partial_levels = self._calculate_partial_profit_levels(
            entry_price, take_profit, direction
        )

        # Max holding time
        max_holding_time = self._calculate_max_holding_time(
            volatility_regime, time_until_expiry
        )

        # Dynamic adjustments
        if trap_probability > 40:
            dynamic_adjustments['tighter_stop'] = True
            dynamic_adjustments['smaller_position'] = True

        if volatility_regime == "High Volatility":
            dynamic_adjustments['wider_stop'] = True
            dynamic_adjustments['earlier_profit_taking'] = True

        # Generate recommendation
        recommendation = self._generate_recommendation(
            entry_approved, risk_score, avoid_reasons, volatility_regime
        )

        return RiskManagementResult(
            entry_approved=entry_approved,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            break_even_trigger=break_even_trigger,
            partial_profit_levels=partial_levels,
            max_holding_time=max_holding_time,
            avoid_reasons=avoid_reasons,
            risk_score=risk_score,
            recommendation=recommendation,
            dynamic_adjustments=dynamic_adjustments
        )

    def _check_trading_time(
        self,
        current_time: Optional[datetime]
    ) -> Tuple[bool, List[str]]:
        """Check if current time is good for trading"""
        if current_time is None:
            return True, []

        reasons = []
        current_time_only = current_time.time()

        # Before market open
        if current_time_only < self.market_open:
            reasons.append("⏰ Market not open yet")
            return False, reasons

        # After market close
        if current_time_only > self.market_close:
            reasons.append("⏰ Market closed")
            return False, reasons

        # First 15 minutes (volatile)
        open_minutes = (current_time_only.hour - 9) * 60 + (current_time_only.minute - 15)
        if 0 <= open_minutes < self.avoid_first_minutes:
            reasons.append("⚠️ First 15 minutes - High volatility")
            return False, reasons

        # Last 15 minutes (closing volatility)
        close_minutes = (15 - current_time_only.hour) * 60 + (30 - current_time_only.minute)
        if 0 <= close_minutes < self.avoid_last_minutes:
            reasons.append("⚠️ Last 15 minutes - Closing volatility")
            return False, reasons

        return True, []

    def _calculate_risk_score(
        self,
        trap_probability: float,
        volatility_regime: str,
        news_event: bool,
        days_to_expiry: int,
        df: pd.DataFrame
    ) -> float:
        """Calculate overall risk score (0-100, lower is better)"""
        score = 0

        # Trap risk
        score += trap_probability * 0.3

        # Volatility regime
        vol_risk = {
            "Low Volatility": 10,
            "Normal Volatility": 20,
            "High Volatility": 40,
            "Extreme Volatility": 60,
            "Regime Change": 50
        }
        score += vol_risk.get(volatility_regime, 20)

        # News event risk
        if news_event:
            score += 25

        # Expiry risk
        if days_to_expiry <= 1:
            score += 20
        elif days_to_expiry <= 3:
            score += 10

        # Market choppiness
        if len(df) >= 10:
            price_changes = abs(df['close'].tail(10).diff()).sum()
            net_change = abs(df['close'].iloc[-1] - df['close'].iloc[-10])
            choppiness = price_changes / net_change if net_change > 0 else 10

            if choppiness > 5:
                score += 15

        return np.clip(score, 0, 100)

    def _calculate_stop_loss(
        self,
        df: pd.DataFrame,
        entry_price: float,
        direction: str,
        support: Optional[float],
        resistance: Optional[float]
    ) -> float:
        """Calculate dynamic stop loss"""
        # ATR-based stop
        if 'atr' in df.columns and len(df) > 0:
            atr = df['atr'].iloc[-1]
            atr_stop_distance = atr * self.atr_multiplier
        else:
            atr_stop_distance = entry_price * 0.01  # 1% fallback

        if direction == "CALL":
            atr_stop = entry_price - atr_stop_distance
            # Use support if closer
            if support and support > atr_stop and support < entry_price:
                return support - 10  # 10 points below support
            return atr_stop
        else:  # PUT
            atr_stop = entry_price + atr_stop_distance
            # Use resistance if closer
            if resistance and resistance < atr_stop and resistance > entry_price:
                return resistance + 10  # 10 points above resistance
            return atr_stop

    def _calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str,
        resistance: Optional[float],
        support: Optional[float]
    ) -> float:
        """Calculate take profit target"""
        risk = abs(entry_price - stop_loss)

        # Default R:R = 2:1
        rr_ratio = 2.0

        if direction == "CALL":
            rr_target = entry_price + (risk * rr_ratio)
            # Adjust if resistance nearby
            if resistance and resistance < rr_target:
                return resistance - 10  # 10 points before resistance
            return rr_target
        else:  # PUT
            rr_target = entry_price - (risk * rr_ratio)
            # Adjust if support nearby
            if support and support > rr_target:
                return support + 10  # 10 points above support
            return rr_target

    def _calculate_trailing_stop(
        self,
        df: pd.DataFrame,
        entry_price: float,
        direction: str
    ) -> Optional[float]:
        """Calculate trailing stop distance"""
        if 'atr' not in df.columns or len(df) == 0:
            return None

        atr = df['atr'].iloc[-1]
        trailing_distance = atr * self.trailing_atr_multiplier

        return trailing_distance

    def _calculate_partial_profit_levels(
        self,
        entry_price: float,
        take_profit: float,
        direction: str
    ) -> List[Tuple[float, float]]:
        """Calculate partial profit taking levels"""
        total_distance = abs(take_profit - entry_price)

        levels = []

        # Take 30% at 50% of target
        if direction == "CALL":
            level_1 = entry_price + (total_distance * 0.5)
        else:
            level_1 = entry_price - (total_distance * 0.5)
        levels.append((level_1, 0.3))

        # Take 40% at 75% of target
        if direction == "CALL":
            level_2 = entry_price + (total_distance * 0.75)
        else:
            level_2 = entry_price - (total_distance * 0.75)
        levels.append((level_2, 0.4))

        # Take remaining 30% at full target
        levels.append((take_profit, 0.3))

        return levels

    def _calculate_max_holding_time(
        self,
        volatility_regime: str,
        days_to_expiry: int
    ) -> int:
        """Calculate maximum holding time in minutes"""
        # Base holding time
        if days_to_expiry <= 1:
            base_time = 60  # 1 hour max on expiry day
        elif days_to_expiry <= 3:
            base_time = 180  # 3 hours in expiry week
        else:
            base_time = 360  # 6 hours normal

        # Adjust for volatility
        if volatility_regime == "Extreme Volatility":
            base_time = int(base_time * 0.5)
        elif volatility_regime == "High Volatility":
            base_time = int(base_time * 0.7)

        return base_time

    def _generate_recommendation(
        self,
        entry_approved: bool,
        risk_score: float,
        avoid_reasons: List[str],
        volatility_regime: str
    ) -> str:
        """Generate trading recommendation"""
        if not entry_approved:
            return f"🚫 AVOID TRADE - Risk Score: {risk_score:.0f}/100\nReasons: " + ", ".join(avoid_reasons)

        if risk_score > 50:
            return f"⚠️ HIGH RISK TRADE - Score: {risk_score:.0f}/100\nProceed with caution, reduce position size"

        if risk_score > 30:
            return f"✅ MODERATE RISK - Score: {risk_score:.0f}/100\nAcceptable risk, normal position size"

        return f"✅ LOW RISK TRADE - Score: {risk_score:.0f}/100\nFavorable conditions"


def format_risk_management_report(result: RiskManagementResult) -> str:
    """Format risk management as readable report"""
    report = f"""
╔══════════════════════════════════════════════════════════╗
║          RISK MANAGEMENT ANALYSIS                        ║
╚══════════════════════════════════════════════════════════╝

🎯 ENTRY DECISION: {'APPROVED ✅' if result.entry_approved else 'REJECTED 🚫'}
⚠️  RISK SCORE: {result.risk_score:.1f}/100

📊 TRADE LEVELS:
  • Entry SL: {result.stop_loss:.2f}
  • Take Profit: {result.take_profit:.2f}
  • Trailing Stop: {result.trailing_stop:.2f if result.trailing_stop else 'N/A'}
  • Break-Even Trigger: {result.break_even_trigger:.2f}

💰 PARTIAL PROFIT PLAN:
"""
    for i, (price, pct) in enumerate(result.partial_profit_levels, 1):
        report += f"  {i}. Exit {pct*100:.0f}% at {price:.2f}\n"

    report += f"\n⏱️  MAX HOLDING TIME: {result.max_holding_time} minutes\n"

    if result.avoid_reasons:
        report += "\n🚫 AVOIDANCE REASONS:\n"
        for reason in result.avoid_reasons:
            report += f"  • {reason}\n"

    if result.dynamic_adjustments:
        report += "\n🔧 DYNAMIC ADJUSTMENTS:\n"
        for adj, value in result.dynamic_adjustments.items():
            report += f"  • {adj.replace('_', ' ').title()}: {value}\n"

    report += f"\n─────────────────────────────────────────────────────────\n"
    report += f"💡 RECOMMENDATION:\n{result.recommendation}\n"

    return report
