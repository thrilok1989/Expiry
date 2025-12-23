"""
Dynamic Strike Defense Zone Calculator

Calculates strike defense zone width based on:
1. ATM ±1 Strike OI Distribution (PRIMARY)
2. Implied Volatility (Zone Expander)
3. Time to Expiry (Pinning Effect)
4. Flow/Delta Pressure (Confirmation)

KEY INSIGHT: Strike zone width = Seller comfort
- Tight zone (±10-15) = Sellers confident
- Medium zone (±20) = Moderate defense
- Wide zone (±25-35) = Sellers nervous
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrikeDefenseZone:
    """Strike defense zone result"""
    center_strike: float
    lower_bound: float
    upper_bound: float
    zone_width: float
    zone_type: str  # "TIGHT", "MEDIUM", "WIDE"

    # Reasoning
    oi_spread: float
    iv_regime: str  # "FALLING", "STABLE", "RISING"
    days_to_expiry: int
    delta_pressure: str  # "STRONG", "WEAK", "NEUTRAL"

    # Multipliers used
    iv_multiplier: float
    dte_multiplier: float

    # Confidence
    confidence: float  # 0-100
    reason: str


class DynamicStrikeDefenseZoneCalculator:
    """
    Calculate dynamic strike defense zone width

    Zone width = Seller comfort level
    """

    def __init__(self):
        """Initialize calculator"""
        self.min_zone = 10.0
        self.max_zone = 35.0

    def calculate_defense_zone(
        self,
        option_chain: pd.DataFrame,
        atm_strike: float,
        current_price: float,
        days_to_expiry: int,
        iv_current: float = None,
        iv_history: pd.Series = None,
        delta_net: float = 0.0
    ) -> StrikeDefenseZone:
        """
        Calculate dynamic strike defense zone

        Args:
            option_chain: Option chain DataFrame with OI_CE, OI_PE columns
            atm_strike: ATM strike price
            current_price: Current spot price
            days_to_expiry: Days until expiry
            iv_current: Current implied volatility (optional)
            iv_history: Historical IV series (optional)
            delta_net: Net delta pressure (positive = bullish, negative = bearish)

        Returns:
            StrikeDefenseZone with complete details
        """

        # STEP 1: Find OI center and spread (PRIMARY FACTOR)
        oi_center, oi_spread, oi_confidence = self._calculate_oi_distribution(
            option_chain, atm_strike
        )

        # STEP 2: Determine base zone from OI spread
        base_zone = self._determine_base_zone(oi_spread)

        # STEP 3: Calculate IV multiplier
        iv_multiplier, iv_regime = self._calculate_iv_multiplier(
            iv_current, iv_history
        )

        # STEP 4: Calculate DTE multiplier (pinning effect)
        dte_multiplier = self._calculate_dte_multiplier(days_to_expiry)

        # STEP 5: Determine delta pressure
        delta_pressure = self._classify_delta_pressure(delta_net)

        # STEP 6: Calculate final zone width
        zone_width = base_zone * iv_multiplier * dte_multiplier
        zone_width = np.clip(zone_width, self.min_zone, self.max_zone)

        # STEP 7: Determine zone type
        if zone_width <= 15:
            zone_type = "TIGHT"
        elif zone_width <= 22:
            zone_type = "MEDIUM"
        else:
            zone_type = "WIDE"

        # STEP 8: Calculate bounds
        lower_bound = oi_center - zone_width
        upper_bound = oi_center + zone_width

        # STEP 9: Generate reason
        reason = self._generate_reason(
            oi_spread, zone_width, iv_regime, days_to_expiry, delta_pressure
        )

        # STEP 10: Calculate confidence
        confidence = self._calculate_confidence(
            oi_confidence, iv_regime, days_to_expiry
        )

        return StrikeDefenseZone(
            center_strike=oi_center,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            zone_width=zone_width,
            zone_type=zone_type,
            oi_spread=oi_spread,
            iv_regime=iv_regime,
            days_to_expiry=days_to_expiry,
            delta_pressure=delta_pressure,
            iv_multiplier=iv_multiplier,
            dte_multiplier=dte_multiplier,
            confidence=confidence,
            reason=reason
        )

    def _calculate_oi_distribution(
        self,
        option_chain: pd.DataFrame,
        atm_strike: float
    ) -> Tuple[float, float, float]:
        """
        Calculate OI distribution around ATM

        Returns:
            (oi_center, oi_spread, confidence)
        """
        if option_chain.empty:
            return atm_strike, 100.0, 0.0

        # Calculate total OI per strike
        option_chain = option_chain.copy()
        option_chain['total_oi'] = option_chain['OI_CE'] + option_chain['OI_PE']

        # Find top 2 OI peaks
        top_oi_strikes = option_chain.nlargest(2, 'total_oi')

        if len(top_oi_strikes) < 2:
            return atm_strike, 100.0, 50.0

        # OI center = strike with max OI
        oi_center = top_oi_strikes.iloc[0]['strikePrice']

        # OI spread = distance between top 2 OI peaks
        oi_spread = abs(
            top_oi_strikes.iloc[0]['strikePrice'] -
            top_oi_strikes.iloc[1]['strikePrice']
        )

        # Confidence based on OI concentration
        top_oi = top_oi_strikes.iloc[0]['total_oi']
        total_oi = option_chain['total_oi'].sum()

        if total_oi > 0:
            concentration = (top_oi / total_oi) * 100
            confidence = min(concentration * 2, 100)  # Scale to 0-100
        else:
            confidence = 50.0

        return oi_center, oi_spread, confidence

    def _determine_base_zone(self, oi_spread: float) -> float:
        """
        Determine base zone width from OI spread

        Logic:
        - Spread ≤ 50 pts → base = 15 (TIGHT - sellers stacked close)
        - Spread 50-100 pts → base = 20 (MEDIUM - moderate defense)
        - Spread > 100 pts → base = 30 (WIDE - sellers nervous)
        """
        if oi_spread <= 50:
            return 15.0
        elif oi_spread <= 100:
            return 20.0
        else:
            return 30.0

    def _calculate_iv_multiplier(
        self,
        iv_current: Optional[float],
        iv_history: Optional[pd.Series]
    ) -> Tuple[float, str]:
        """
        Calculate IV multiplier (zone expander/contractor)

        Returns:
            (iv_multiplier, iv_regime)
        """
        if iv_current is None or iv_history is None or len(iv_history) < 5:
            return 1.0, "STABLE"

        # Calculate IV trend (last 5 periods)
        iv_recent = iv_history.tail(5)
        iv_mean = iv_recent.mean()

        # Determine IV regime
        if iv_current < iv_mean * 0.95:
            # IV falling → tighten zone
            return 0.7, "FALLING"
        elif iv_current > iv_mean * 1.05:
            # IV rising → expand zone
            return 1.3, "RISING"
        else:
            # IV stable → normal zone
            return 1.0, "STABLE"

    def _calculate_dte_multiplier(self, days_to_expiry: int) -> float:
        """
        Calculate DTE multiplier (pinning effect)

        Logic:
        - 0-2 days: Very tight (×0.7) - strong pinning
        - 3-5 days: Tight (×0.85)
        - 6-10 days: Normal (×1.0)
        - 11+ days: Wide (×1.2)
        """
        if days_to_expiry <= 2:
            return 0.7  # Very tight - strong pinning
        elif days_to_expiry <= 5:
            return 0.85  # Tight
        elif days_to_expiry <= 10:
            return 1.0  # Normal
        else:
            return 1.2  # Wide

    def _classify_delta_pressure(self, delta_net: float) -> str:
        """
        Classify delta pressure strength

        Returns:
            "STRONG", "WEAK", "NEUTRAL"
        """
        abs_delta = abs(delta_net)

        if abs_delta > 500000:  # Strong delta pressure
            return "STRONG"
        elif abs_delta < 100000:  # Weak delta pressure
            return "WEAK"
        else:
            return "NEUTRAL"

    def _generate_reason(
        self,
        oi_spread: float,
        zone_width: float,
        iv_regime: str,
        days_to_expiry: int,
        delta_pressure: str
    ) -> str:
        """Generate human-readable reason for zone width"""

        reasons = []

        # OI distribution
        if oi_spread <= 50:
            reasons.append("High ATM OI concentration")
        elif oi_spread <= 100:
            reasons.append("Moderate OI spread")
        else:
            reasons.append("Wide OI distribution")

        # IV regime
        if iv_regime == "FALLING":
            reasons.append("IV falling (tightening)")
        elif iv_regime == "RISING":
            reasons.append("IV rising (expanding)")
        else:
            reasons.append("IV stable")

        # DTE
        if days_to_expiry <= 2:
            reasons.append(f"{days_to_expiry} DTE (strong pinning)")
        elif days_to_expiry <= 5:
            reasons.append(f"{days_to_expiry} DTE (moderate pinning)")
        else:
            reasons.append(f"{days_to_expiry} DTE")

        # Delta pressure
        if delta_pressure == "STRONG":
            reasons.append("Strong delta pressure")
        elif delta_pressure == "WEAK":
            reasons.append("Weak delta pressure")

        return " | ".join(reasons)

    def _calculate_confidence(
        self,
        oi_confidence: float,
        iv_regime: str,
        days_to_expiry: int
    ) -> float:
        """Calculate overall confidence in zone calculation"""

        confidence = oi_confidence

        # Boost confidence if IV regime is clear
        if iv_regime in ["FALLING", "RISING"]:
            confidence = min(confidence + 10, 100)

        # Boost confidence near expiry (pinning more reliable)
        if days_to_expiry <= 3:
            confidence = min(confidence + 15, 100)

        return confidence


def format_zone_display(zone: StrikeDefenseZone) -> str:
    """
    Format zone for display in signal

    Returns formatted string for HTML/Streamlit display
    """

    display = f"""
🎯 STRIKE DEFENSE ZONE ({zone.zone_type})

Center: ₹{zone.center_strike:,.0f}
Lower: ₹{zone.lower_bound:,.0f}
Upper: ₹{zone.upper_bound:,.0f}
Zone Width: {zone.zone_width:.0f} pts

Reason: {zone.reason}
Confidence: {zone.confidence:.0f}%

📊 Breakdown:
- OI Spread: {zone.oi_spread:.0f} pts
- IV Regime: {zone.iv_regime}
- Days to Expiry: {zone.days_to_expiry}
- Delta Pressure: {zone.delta_pressure}
"""

    return display


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Example option chain
    option_chain = pd.DataFrame({
        'strikePrice': [25900, 25950, 26000, 26050, 26100],
        'OI_CE': [500000, 2000000, 1800000, 300000, 200000],
        'OI_PE': [200000, 1500000, 1600000, 800000, 500000]
    })

    # Calculate zone
    calculator = DynamicStrikeDefenseZoneCalculator()
    zone = calculator.calculate_defense_zone(
        option_chain=option_chain,
        atm_strike=25950,
        current_price=25975,
        days_to_expiry=4,
        iv_current=18.5,
        iv_history=pd.Series([17.0, 17.5, 18.0, 18.2, 18.5]),
        delta_net=250000
    )

    print("=== DYNAMIC STRIKE DEFENSE ZONE ===")
    print(format_zone_display(zone))

    # Example 2: Tight zone (expiry day)
    zone_tight = calculator.calculate_defense_zone(
        option_chain=option_chain,
        atm_strike=25950,
        current_price=25955,
        days_to_expiry=1,  # Expiry day
        iv_current=15.0,
        iv_history=pd.Series([18.0, 17.0, 16.0, 15.5, 15.0]),  # IV falling
        delta_net=50000  # Weak delta
    )

    print("\n\n=== TIGHT ZONE EXAMPLE (Expiry Day) ===")
    print(format_zone_display(zone_tight))
