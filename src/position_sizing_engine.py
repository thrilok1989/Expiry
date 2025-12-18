"""
Dynamic Position Sizing Engine
Professional position sizing based on volatility, risk, and edge
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Position sizing result"""
    recommended_lots: int
    recommended_contracts: int
    position_value: float
    risk_per_trade: float
    risk_percentage: float
    kelly_fraction: float
    volatility_adjustment: float
    rr_adjustment: float
    sizing_method: str
    confidence: float
    warnings: list[str]
    explanation: str


class PositionSizingEngine:
    """
    Dynamic Position Sizing Engine

    Methods:
    1. Fixed Fractional (% of capital)
    2. Volatility-Adjusted (ATR-based)
    3. Kelly Criterion (edge-based)
    4. Risk-Reward Adjusted
    5. Hybrid (combines all)
    """

    def __init__(
        self,
        account_size: float = 100000,
        max_risk_per_trade: float = 2.0,
        max_position_size: float = 10.0,
        kelly_divisor: float = 4.0
    ):
        """
        Initialize Position Sizing Engine

        Args:
            account_size: Total account capital
            max_risk_per_trade: Maximum % of capital to risk per trade
            max_position_size: Maximum % of capital in single position
            kelly_divisor: Divisor for Kelly Criterion (conservative)
        """
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.kelly_divisor = kelly_divisor

        # Instrument specifications (India)
        self.instrument_specs = {
            'NIFTY': {'lot_size': 75, 'margin_per_lot': 25000},
            'BANKNIFTY': {'lot_size': 25, 'margin_per_lot': 35000},
            'SENSEX': {'lot_size': 30, 'margin_per_lot': 30000},
            'FINNIFTY': {'lot_size': 40, 'margin_per_lot': 20000}
        }

    def calculate_position_size(
        self,
        instrument: str,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        df: pd.DataFrame,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        volatility_regime: str = "Normal",
        trap_risk: float = 0
    ) -> PositionSizeResult:
        """
        Calculate optimal position size

        Args:
            instrument: Trading instrument (NIFTY, BANKNIFTY, etc.)
            entry_price: Entry price
            stop_loss: Stop loss price
            target_price: Target price
            df: Historical price data with ATR
            win_rate: Historical win rate (0-100)
            avg_win: Average win amount
            avg_loss: Average loss amount
            volatility_regime: Current volatility regime
            trap_risk: Trap probability (0-100)

        Returns:
            PositionSizeResult with sizing recommendation
        """
        warnings = []

        # Get instrument specs
        if instrument not in self.instrument_specs:
            instrument = 'NIFTY'  # Default
            warnings.append(f"Unknown instrument, using NIFTY defaults")

        lot_size = self.instrument_specs[instrument]['lot_size']
        margin_per_lot = self.instrument_specs[instrument]['margin_per_lot']

        # Calculate risk per contract
        risk_per_point = abs(entry_price - stop_loss)
        risk_per_contract = risk_per_point * lot_size

        if risk_per_contract == 0:
            warnings.append("Zero risk - using 1 lot")
            return self._create_result(1, lot_size, warnings, "Default")

        # Calculate Risk-Reward ratio
        reward_per_point = abs(target_price - entry_price)
        rr_ratio = reward_per_point / risk_per_point if risk_per_point > 0 else 1

        # Method 1: Fixed Fractional
        fixed_lots = self._calculate_fixed_fractional(risk_per_contract)

        # Method 2: Volatility-Adjusted
        vol_lots = self._calculate_volatility_adjusted(
            df, risk_per_contract, volatility_regime
        )

        # Method 3: Kelly Criterion
        kelly_lots = self._calculate_kelly_criterion(
            risk_per_contract, win_rate, avg_win, avg_loss
        )

        # Method 4: Risk-Reward Adjusted
        rr_lots = self._calculate_rr_adjusted(fixed_lots, rr_ratio)

        # Method 5: Trap-Risk Adjusted
        trap_adjusted_lots = self._apply_trap_adjustment(
            [fixed_lots, vol_lots, kelly_lots, rr_lots], trap_risk
        )

        # Hybrid: Weighted average
        recommended_lots = self._calculate_hybrid(
            fixed_lots, vol_lots, kelly_lots, rr_lots, trap_adjusted_lots,
            win_rate is not None
        )

        # Apply position limits
        max_lots_by_capital = int((self.account_size * self.max_position_size / 100) / margin_per_lot)
        recommended_lots = min(recommended_lots, max_lots_by_capital)
        recommended_lots = max(1, recommended_lots)  # Minimum 1 lot

        # Calculate metrics
        recommended_contracts = recommended_lots * lot_size
        position_value = recommended_lots * margin_per_lot
        risk_per_trade = recommended_lots * risk_per_contract
        risk_percentage = (risk_per_trade / self.account_size) * 100

        # Validation warnings
        if risk_percentage > self.max_risk_per_trade:
            warnings.append(f"⚠️ Risk {risk_percentage:.2f}% exceeds max {self.max_risk_per_trade}%")

        if position_value > self.account_size * 0.5:
            warnings.append(f"⚠️ Large position: {position_value/self.account_size*100:.1f}% of capital")

        if rr_ratio < 1.5:
            warnings.append(f"⚠️ Poor R:R ratio: {rr_ratio:.2f}")

        # Calculate confidence
        confidence = self._calculate_confidence(
            risk_percentage, rr_ratio, volatility_regime, trap_risk
        )

        # Generate explanation
        explanation = self._generate_explanation(
            fixed_lots, vol_lots, kelly_lots, rr_lots, recommended_lots,
            risk_percentage, rr_ratio, volatility_regime
        )

        # Calculate Kelly fraction for reference
        kelly_fraction = 0
        if win_rate and avg_win and avg_loss and avg_loss > 0:
            kelly_fraction = ((win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss) / avg_loss

        return PositionSizeResult(
            recommended_lots=recommended_lots,
            recommended_contracts=recommended_contracts,
            position_value=position_value,
            risk_per_trade=risk_per_trade,
            risk_percentage=risk_percentage,
            kelly_fraction=kelly_fraction / self.kelly_divisor if kelly_fraction > 0 else 0,
            volatility_adjustment=vol_lots / fixed_lots if fixed_lots > 0 else 1.0,
            rr_adjustment=rr_lots / fixed_lots if fixed_lots > 0 else 1.0,
            sizing_method="Hybrid (Volatility + RR + Kelly)",
            confidence=confidence,
            warnings=warnings,
            explanation=explanation
        )

    def _calculate_fixed_fractional(self, risk_per_contract: float) -> int:
        """Fixed fractional position sizing"""
        max_risk_dollars = self.account_size * (self.max_risk_per_trade / 100)
        lots = int(max_risk_dollars / risk_per_contract)
        return max(1, lots)

    def _calculate_volatility_adjusted(
        self,
        df: pd.DataFrame,
        risk_per_contract: float,
        volatility_regime: str
    ) -> int:
        """Volatility-adjusted position sizing"""
        base_lots = self._calculate_fixed_fractional(risk_per_contract)

        # Get ATR percentile
        if 'atr' in df.columns and len(df) >= 20:
            current_atr = df['atr'].iloc[-1]
            atr_history = df['atr'].tail(50)
            atr_percentile = (atr_history <= current_atr).sum() / len(atr_history) * 100
        else:
            atr_percentile = 50

        # Adjust based on volatility regime
        if volatility_regime == "Low Volatility" or atr_percentile < 30:
            # Low volatility = Larger position
            multiplier = 1.3
        elif volatility_regime == "High Volatility" or atr_percentile > 70:
            # High volatility = Smaller position
            multiplier = 0.7
        elif volatility_regime == "Extreme Volatility" or atr_percentile > 90:
            # Extreme volatility = Much smaller position
            multiplier = 0.5
        else:
            multiplier = 1.0

        return max(1, int(base_lots * multiplier))

    def _calculate_kelly_criterion(
        self,
        risk_per_contract: float,
        win_rate: Optional[float],
        avg_win: Optional[float],
        avg_loss: Optional[float]
    ) -> int:
        """Kelly Criterion position sizing"""
        if not win_rate or not avg_win or not avg_loss or avg_loss == 0:
            return self._calculate_fixed_fractional(risk_per_contract)

        # Kelly formula: f* = (p * b - q) / b
        # where p = win rate, q = 1 - p, b = win/loss ratio
        p = win_rate / 100
        q = 1 - p
        b = avg_win / avg_loss

        kelly_fraction = (p * b - q) / b

        # Conservative Kelly (divide by 4 for safety)
        kelly_fraction = max(0, kelly_fraction / self.kelly_divisor)

        # Calculate lots
        kelly_capital = self.account_size * kelly_fraction
        lots = int(kelly_capital / (risk_per_contract * 2))  # Assume 2x risk buffer

        return max(1, lots)

    def _calculate_rr_adjusted(self, base_lots: int, rr_ratio: float) -> int:
        """Risk-Reward adjusted position sizing"""
        # Better R:R = Larger position
        if rr_ratio >= 3:
            multiplier = 1.4
        elif rr_ratio >= 2:
            multiplier = 1.2
        elif rr_ratio >= 1.5:
            multiplier = 1.0
        else:
            multiplier = 0.7  # Poor R:R = smaller position

        return max(1, int(base_lots * multiplier))

    def _apply_trap_adjustment(self, lot_sizes: list, trap_risk: float) -> list:
        """Reduce position size if trap risk is high"""
        if trap_risk > 70:
            multiplier = 0.5  # Reduce by 50%
        elif trap_risk > 50:
            multiplier = 0.7  # Reduce by 30%
        elif trap_risk > 30:
            multiplier = 0.85  # Reduce by 15%
        else:
            multiplier = 1.0

        return [max(1, int(lots * multiplier)) for lots in lot_sizes]

    def _calculate_hybrid(
        self,
        fixed: int,
        vol: int,
        kelly: int,
        rr: int,
        trap_adjusted: list,
        has_historical_data: bool
    ) -> int:
        """Hybrid position sizing (weighted average)"""
        if has_historical_data:
            # Use Kelly if historical data available
            weights = [0.2, 0.3, 0.3, 0.2]  # Fixed, Vol, Kelly, RR
            weighted = (fixed * weights[0] + vol * weights[1] +
                       kelly * weights[2] + rr * weights[3])
        else:
            # No Kelly data
            weights = [0.25, 0.5, 0.25]  # Fixed, Vol, RR
            weighted = (fixed * weights[0] + vol * weights[1] + rr * weights[2])

        # Apply trap adjustment to final number
        trap_multiplier = sum(trap_adjusted) / sum([fixed, vol, kelly, rr]) if sum([fixed, vol, kelly, rr]) > 0 else 1
        weighted = weighted * trap_multiplier

        return max(1, int(weighted))

    def _calculate_confidence(
        self,
        risk_pct: float,
        rr_ratio: float,
        vol_regime: str,
        trap_risk: float
    ) -> float:
        """Calculate confidence in position size (0-1)"""
        confidence = 0.8  # Base

        # Good R:R increases confidence
        if rr_ratio >= 2:
            confidence += 0.1
        elif rr_ratio < 1.5:
            confidence -= 0.15

        # Extreme volatility reduces confidence
        if vol_regime == "Extreme Volatility":
            confidence -= 0.2
        elif vol_regime == "High Volatility":
            confidence -= 0.1

        # Trap risk reduces confidence
        if trap_risk > 60:
            confidence -= 0.25
        elif trap_risk > 40:
            confidence -= 0.15

        # Risk within limits increases confidence
        if risk_pct <= self.max_risk_per_trade:
            confidence += 0.05

        return np.clip(confidence, 0.3, 1.0)

    def _generate_explanation(
        self,
        fixed: int,
        vol: int,
        kelly: int,
        rr: int,
        final: int,
        risk_pct: float,
        rr_ratio: float,
        vol_regime: str
    ) -> str:
        """Generate human-readable explanation"""
        explanation = f"""
Position Sizing Breakdown:
  • Fixed Fractional: {fixed} lots ({risk_pct:.2f}% risk)
  • Volatility Adjusted: {vol} lots ({vol_regime})
  • Kelly Criterion: {kelly} lots
  • R:R Adjusted: {rr} lots (R:R = {rr_ratio:.2f})

  → Final Recommendation: {final} lots

Rationale:
"""
        if vol < fixed:
            explanation += f"  • Reduced for high volatility ({vol_regime})\n"
        elif vol > fixed:
            explanation += f"  • Increased for low volatility ({vol_regime})\n"

        if rr_ratio >= 2:
            explanation += f"  • Increased for excellent R:R ({rr_ratio:.2f})\n"
        elif rr_ratio < 1.5:
            explanation += f"  • Reduced for poor R:R ({rr_ratio:.2f})\n"

        return explanation

    def _create_result(
        self,
        lots: int,
        lot_size: int,
        warnings: list,
        method: str
    ) -> PositionSizeResult:
        """Create default result"""
        return PositionSizeResult(
            recommended_lots=lots,
            recommended_contracts=lots * lot_size,
            position_value=lots * 25000,  # Estimate
            risk_per_trade=0,
            risk_percentage=0,
            kelly_fraction=0,
            volatility_adjustment=1.0,
            rr_adjustment=1.0,
            sizing_method=method,
            confidence=0.5,
            warnings=warnings,
            explanation="Insufficient data for full calculation"
        )


def format_position_size_report(result: PositionSizeResult) -> str:
    """Format position sizing as readable report"""
    report = f"""
╔══════════════════════════════════════════════════════════╗
║          POSITION SIZING RECOMMENDATION                  ║
╚══════════════════════════════════════════════════════════╝

🎯 RECOMMENDED SIZE:
  • Lots: {result.recommended_lots}
  • Contracts: {result.recommended_contracts}
  • Position Value: ₹{result.position_value:,.0f}

💰 RISK METRICS:
  • Risk per Trade: ₹{result.risk_per_trade:,.0f}
  • Risk Percentage: {result.risk_percentage:.2f}%
  • Kelly Fraction: {result.kelly_fraction:.3f}

📊 ADJUSTMENTS:
  • Volatility Adjustment: {result.volatility_adjustment:.2f}x
  • R:R Adjustment: {result.rr_adjustment:.2f}x

✅ CONFIDENCE: {result.confidence*100:.1f}%
📝 METHOD: {result.sizing_method}

─────────────────────────────────────────────────────────
{result.explanation}
"""

    if result.warnings:
        report += "\n⚠️  WARNINGS:\n"
        for warning in result.warnings:
            report += f"  • {warning}\n"

    return report
