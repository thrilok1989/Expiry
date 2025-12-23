"""
ADVANCED ENTRY LOGIC - Institutional Grade Entry System

Combines:
1. Zone Width Confirmation (Option Chain + Gamma + Liquidity + Price Action)
2. Premium Divergence Detection (Entry when spot moves but option premium doesn't)
3. Strike Defense Zone Calculator (Dynamic, not fixed)
4. Major vs Minor VOB Classification
5. NIFTY Futures Bias Analysis

This is the EXECUTION LAYER - where real money is made.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ZoneType(Enum):
    """Zone classification"""
    TIGHT = "TIGHT"      # ±10-15 pts
    MEDIUM = "MEDIUM"    # ±20-25 pts
    WIDE = "WIDE"        # ±30-40 pts


class VOBStrength(Enum):
    """VOB classification"""
    MAJOR = "MAJOR"      # High OI + Strong price action
    MINOR = "MINOR"      # Lower OI + Weak price action
    INVALID = "INVALID"  # Failed VOB


@dataclass
class ZoneWidthConfirmation:
    """Multi-source zone width confirmation"""
    base_zone_width: float  # From option chain
    gamma_adjusted_width: float  # After gamma modification
    liquidity_pivot: float  # Exact turning point
    price_action_confirmed: bool
    final_zone_lower: float
    final_zone_upper: float
    confidence: float  # 0-100
    sources_aligned: int  # Number of sources confirming


@dataclass
class PremiumDivergence:
    """Premium divergence signal - THE REAL ENTRY"""
    detected: bool
    direction: str  # LONG or SHORT
    spot_price: float
    spot_move: str  # "Higher High" or "Lower Low"
    option_strike: int
    option_premium: float
    premium_move: str  # "No Higher High" or "Flat" or "Declining"
    divergence_strength: float  # 0-100
    entry_signal: bool
    reason: str


@dataclass
class StrikeDefenseZone:
    """Dynamic strike defense zone"""
    atm_strike: int
    center_price: float
    zone_lower: float
    zone_upper: float
    zone_width: float
    zone_type: ZoneType
    oi_confidence: float
    gamma_regime: str
    liquidity_depth: str
    defense_strength: float  # 0-100


@dataclass
class VOBClassification:
    """VOB strength classification"""
    vob_price: float
    vob_type: str  # Support or Resistance
    strength: VOBStrength
    oi_concentration: float
    price_action_score: float
    option_chain_score: float
    combined_score: float
    is_tradeable: bool
    reason: str


class AdvancedEntryLogic:
    """
    Advanced Entry Logic - Institutional Grade

    This is where price actually reverses, not where indicators say it should.
    """

    def __init__(self):
        self.version = "v1.0_institutional"

    # =========================================================================
    # 1. ZONE WIDTH CONFIRMATION ENGINE
    # =========================================================================

    def calculate_zone_width(
        self,
        atm_strike: int,
        atm_oi: float,
        atm_minus_1_oi: float,
        atm_plus_1_oi: float,
        iv_current: float,
        iv_baseline: float = 15.0,
        dte: int = 5,
        gamma_exposure: float = 0.0,
        current_price: float = 0.0
    ) -> ZoneWidthConfirmation:
        """
        Calculate dynamic zone width using 4 sources:
        1. Option Chain (OI distribution)
        2. Gamma (GEX regime)
        3. Liquidity (will be added with depth data)
        4. Price Action (historical reactions)
        """

        # === SOURCE 1: OPTION CHAIN → BASE ZONE ===
        # Measure OI concentration
        total_oi = atm_oi + atm_minus_1_oi + atm_plus_1_oi
        atm_concentration = (atm_oi / total_oi * 100) if total_oi > 0 else 0

        # Base zone from OI distribution
        if atm_concentration > 60:
            # Very high ATM concentration = tight zone
            base_zone = 12
            zone_type = ZoneType.TIGHT
        elif atm_concentration > 40:
            # Moderate concentration = medium zone
            base_zone = 20
            zone_type = ZoneType.MEDIUM
        else:
            # Low concentration = wide zone
            base_zone = 30
            zone_type = ZoneType.WIDE

        oi_confidence = atm_concentration

        # === SOURCE 2: GAMMA ADJUSTMENT ===
        # Positive gamma = dealers stabilize price → tighter zone
        # Negative gamma = dealers chase price → wider zone
        if gamma_exposure > 0:
            gamma_modifier = 0.6  # Compress zone
            gamma_regime = "Positive Gamma (Stabilizing)"
        elif gamma_exposure < 0:
            gamma_modifier = 1.4  # Expand zone
            gamma_regime = "Negative Gamma (Chasing)"
        else:
            gamma_modifier = 1.0
            gamma_regime = "Neutral Gamma"

        gamma_adjusted = base_zone * gamma_modifier

        # === SOURCE 3: IV ADJUSTMENT ===
        # Rising IV = expand zone, Falling IV = compress zone
        iv_ratio = iv_current / iv_baseline if iv_baseline > 0 else 1.0
        if iv_ratio > 1.2:
            iv_modifier = 1.3
        elif iv_ratio < 0.8:
            iv_modifier = 0.7
        else:
            iv_modifier = 1.0

        # === SOURCE 4: TIME TO EXPIRY ===
        # Near expiry = tight pinning
        if dte <= 2:
            dte_modifier = 0.7
        elif dte <= 7:
            dte_modifier = 1.0
        else:
            dte_modifier = 1.2

        # === FINAL ZONE WIDTH ===
        final_width = gamma_adjusted * iv_modifier * dte_modifier
        final_width = max(10, min(40, final_width))  # Clamp between 10-40

        # Zone boundaries
        zone_center = atm_strike
        zone_lower = zone_center - final_width
        zone_upper = zone_center + final_width

        # Liquidity pivot (center of zone for now, will be refined with depth)
        liquidity_pivot = zone_center

        # Price action confirmation (check if current price shows reaction)
        price_action_confirmed = False
        if current_price > 0:
            # If price is near zone boundaries and showing rejection
            dist_to_lower = abs(current_price - zone_lower)
            dist_to_upper = abs(current_price - zone_upper)
            if dist_to_lower < 5 or dist_to_upper < 5:
                price_action_confirmed = True

        # Sources aligned
        sources_aligned = sum([
            atm_concentration > 40,  # OI confirms
            abs(gamma_modifier - 1.0) > 0.1,  # Gamma has opinion
            abs(iv_modifier - 1.0) > 0.1,  # IV has opinion
            price_action_confirmed
        ])

        # Overall confidence
        confidence = (oi_confidence + (sources_aligned * 20)) / 2
        confidence = min(90, confidence)

        return ZoneWidthConfirmation(
            base_zone_width=base_zone,
            gamma_adjusted_width=gamma_adjusted,
            liquidity_pivot=liquidity_pivot,
            price_action_confirmed=price_action_confirmed,
            final_zone_lower=zone_lower,
            final_zone_upper=zone_upper,
            confidence=confidence,
            sources_aligned=sources_aligned
        )

    # =========================================================================
    # 2. PREMIUM DIVERGENCE DETECTION (THE REAL ENTRY SIGNAL)
    # =========================================================================

    def detect_premium_divergence(
        self,
        spot_df: pd.DataFrame,  # Last 3-5 candles
        option_df: pd.DataFrame,  # Last 3-5 candles (ATM option)
        direction: str = "SHORT"  # SHORT (resistance) or LONG (support)
    ) -> PremiumDivergence:
        """
        Detect when spot makes new extreme but option premium doesn't follow

        THIS IS THE REAL ENTRY SIGNAL - where money stops flowing
        """

        if len(spot_df) < 2 or len(option_df) < 2:
            return PremiumDivergence(
                detected=False,
                direction=direction,
                spot_price=0,
                spot_move="Unknown",
                option_strike=0,
                option_premium=0,
                premium_move="Unknown",
                divergence_strength=0,
                entry_signal=False,
                reason="Insufficient data"
            )

        # Get current and previous values
        spot_current = spot_df['close'].iloc[-1]
        spot_prev = spot_df['close'].iloc[-2]
        spot_prev2 = spot_df['close'].iloc[-3] if len(spot_df) >= 3 else spot_prev

        option_current = option_df['close'].iloc[-1]
        option_prev = option_df['close'].iloc[-2]
        option_prev2 = option_df['close'].iloc[-3] if len(option_df) >= 3 else option_prev

        # === CHECK FOR DIVERGENCE ===
        divergence_detected = False
        spot_move = ""
        premium_move = ""
        divergence_strength = 0

        if direction == "SHORT":
            # Check for resistance failure
            # Spot makes higher high, but CE premium doesn't
            spot_higher_high = (spot_current > spot_prev) and (spot_prev > spot_prev2)

            if spot_higher_high:
                spot_move = "Higher High"

                # Check if premium follows
                premium_higher_high = (option_current > option_prev) and (option_prev > option_prev2)

                if not premium_higher_high:
                    # Divergence detected!
                    divergence_detected = True

                    # Measure strength
                    spot_gain = ((spot_current - spot_prev2) / spot_prev2) * 100
                    premium_gain = ((option_current - option_prev2) / option_prev2) * 100 if option_prev2 > 0 else 0

                    # Strong divergence if spot up but premium flat/down
                    if premium_gain < 0:
                        premium_move = "Declining (Strong Divergence)"
                        divergence_strength = 90
                    elif premium_gain < 0.5:
                        premium_move = "Flat (Moderate Divergence)"
                        divergence_strength = 70
                    else:
                        premium_move = "Weak Gain (Mild Divergence)"
                        divergence_strength = 50

        elif direction == "LONG":
            # Check for support failure
            # Spot makes lower low, but PE premium doesn't
            spot_lower_low = (spot_current < spot_prev) and (spot_prev < spot_prev2)

            if spot_lower_low:
                spot_move = "Lower Low"

                # Check if premium follows
                premium_higher_high = (option_current > option_prev) and (option_prev > option_prev2)

                if not premium_higher_high:
                    # Divergence detected!
                    divergence_detected = True

                    # Measure strength
                    spot_loss = ((spot_prev2 - spot_current) / spot_prev2) * 100
                    premium_gain = ((option_current - option_prev2) / option_prev2) * 100 if option_prev2 > 0 else 0

                    if premium_gain < 0:
                        premium_move = "Declining (Strong Divergence)"
                        divergence_strength = 90
                    elif premium_gain < 0.5:
                        premium_move = "Flat (Moderate Divergence)"
                        divergence_strength = 70
                    else:
                        premium_move = "Weak Gain (Mild Divergence)"
                        divergence_strength = 50

        # Entry signal if strong divergence
        entry_signal = divergence_detected and divergence_strength >= 70

        # Reason
        if entry_signal:
            reason = f"Premium exhaustion detected: Spot making {spot_move} but option premium {premium_move}"
        elif divergence_detected:
            reason = f"Weak divergence: Monitor for confirmation"
        else:
            reason = "No divergence detected - Money still flowing"

        return PremiumDivergence(
            detected=divergence_detected,
            direction=direction,
            spot_price=spot_current,
            spot_move=spot_move,
            option_strike=0,  # Will be filled by caller
            option_premium=option_current,
            premium_move=premium_move,
            divergence_strength=divergence_strength,
            entry_signal=entry_signal,
            reason=reason
        )

    # =========================================================================
    # 3. VOB CLASSIFICATION (MAJOR VS MINOR)
    # =========================================================================

    def classify_vob(
        self,
        vob_price: float,
        vob_type: str,
        vob_volume: float,
        nearby_oi: float,  # OI at nearest strike
        atm_oi: float,  # ATM OI for reference
        price_reactions: int,  # How many times price reacted at this VOB
        last_reaction_strength: float = 0.0  # Size of last bounce/rejection
    ) -> VOBClassification:
        """
        Classify VOB as MAJOR or MINOR based on:
        1. Option chain data (OI concentration)
        2. Price action (reactions, bounces)
        """

        # === 1. OPTION CHAIN SCORE ===
        # Compare VOB-level OI to ATM OI
        oi_ratio = (nearby_oi / atm_oi * 100) if atm_oi > 0 else 0

        if oi_ratio > 80:
            oi_score = 90
        elif oi_ratio > 50:
            oi_score = 70
        elif oi_ratio > 30:
            oi_score = 50
        else:
            oi_score = 30

        # === 2. PRICE ACTION SCORE ===
        # Multiple reactions = stronger VOB
        reaction_score = min(100, price_reactions * 25)  # 4+ reactions = 100

        # Last reaction strength
        if last_reaction_strength > 30:
            reaction_strength_score = 90
        elif last_reaction_strength > 15:
            reaction_strength_score = 70
        elif last_reaction_strength > 5:
            reaction_strength_score = 50
        else:
            reaction_strength_score = 30

        price_action_score = (reaction_score + reaction_strength_score) / 2

        # === 3. COMBINED SCORE ===
        combined_score = (oi_score * 0.6) + (price_action_score * 0.4)

        # === 4. CLASSIFICATION ===
        if combined_score >= 75:
            strength = VOBStrength.MAJOR
            is_tradeable = True
            reason = f"High OI ({oi_ratio:.0f}% of ATM) + Strong price reactions ({price_reactions}x)"
        elif combined_score >= 50:
            strength = VOBStrength.MINOR
            is_tradeable = True
            reason = f"Moderate OI + Some price reactions - Use with caution"
        else:
            strength = VOBStrength.INVALID
            is_tradeable = False
            reason = f"Weak OI + Poor price reactions - Avoid trading"

        return VOBClassification(
            vob_price=vob_price,
            vob_type=vob_type,
            strength=strength,
            oi_concentration=oi_ratio,
            price_action_score=price_action_score,
            option_chain_score=oi_score,
            combined_score=combined_score,
            is_tradeable=is_tradeable,
            reason=reason
        )

    # =========================================================================
    # 4. NIFTY FUTURES BIAS ANALYSIS
    # =========================================================================

    def analyze_futures_bias(
        self,
        spot_price: float,
        futures_price: float,
        futures_oi: float = 0,
        futures_oi_change: float = 0,
        futures_volume: float = 0
    ) -> Dict:
        """
        Analyze NIFTY Futures for market regime confirmation

        Futures lead spot by 5-10 seconds typically
        Futures OI shows institutional positioning
        """

        # Premium/Discount
        premium = futures_price - spot_price
        premium_pct = (premium / spot_price) * 100 if spot_price > 0 else 0

        # Bias from premium
        if premium_pct > 0.15:
            premium_bias = "BULLISH"
            premium_strength = min(100, premium_pct * 200)
        elif premium_pct < -0.15:
            premium_bias = "BEARISH"
            premium_strength = min(100, abs(premium_pct) * 200)
        else:
            premium_bias = "NEUTRAL"
            premium_strength = 50

        # OI bias
        if futures_oi_change > 0:
            if premium > 0:
                oi_bias = "BULLISH"  # Long buildup
            else:
                oi_bias = "BEARISH"  # Short buildup
        elif futures_oi_change < 0:
            if premium > 0:
                oi_bias = "BEARISH"  # Long unwinding
            else:
                oi_bias = "BULLISH"  # Short covering
        else:
            oi_bias = "NEUTRAL"

        # Combined futures bias
        if premium_bias == oi_bias:
            combined_bias = premium_bias
            confidence = 80
        elif premium_bias != "NEUTRAL":
            combined_bias = premium_bias
            confidence = 60
        else:
            combined_bias = oi_bias if oi_bias != "NEUTRAL" else "NEUTRAL"
            confidence = 50

        return {
            'futures_price': futures_price,
            'spot_price': spot_price,
            'premium': premium,
            'premium_pct': premium_pct,
            'premium_bias': premium_bias,
            'premium_strength': premium_strength,
            'oi_change': futures_oi_change,
            'oi_bias': oi_bias,
            'combined_bias': combined_bias,
            'confidence': confidence,
            'interpretation': self._interpret_futures_bias(combined_bias, premium_pct, confidence)
        }

    def _interpret_futures_bias(self, bias: str, premium_pct: float, confidence: float) -> str:
        """Generate interpretation of futures bias"""

        if bias == "BULLISH":
            if confidence > 75:
                return f"Strong institutional buying | Futures leading spot by {premium_pct:.2f}% | High confidence"
            else:
                return f"Moderate bullish bias | Watch for confirmation"
        elif bias == "BEARISH":
            if confidence > 75:
                return f"Strong institutional selling | Futures discount of {abs(premium_pct):.2f}% | High confidence"
            else:
                return f"Moderate bearish bias | Watch for confirmation"
        else:
            return "Neutral - No clear directional bias from futures"


# =========================================================================
# DISPLAY FORMATTING FUNCTIONS
# =========================================================================

def format_zone_width_display(zone: ZoneWidthConfirmation, atm_strike: int) -> str:
    """Format zone width confirmation for display"""

    zone_type_colors = {
        "TIGHT": "#00ff00",
        "MEDIUM": "#ffaa00",
        "WIDE": "#ff6600"
    }

    # Determine zone type
    if zone.final_zone_upper - zone.final_zone_lower <= 20:
        zone_type = "TIGHT"
    elif zone.final_zone_upper - zone.final_zone_lower <= 30:
        zone_type = "MEDIUM"
    else:
        zone_type = "WIDE"

    color = zone_type_colors.get(zone_type, "#ffaa00")

    return f"""
**🎯 STRIKE DEFENSE ZONE (Multi-Source Confirmed)**

**Center:** {atm_strike} | **Zone Type:** {zone_type} ({zone.final_zone_upper - zone.final_zone_lower:.0f} pts width)

**Zone Boundaries:**
- **Lower:** ₹{zone.final_zone_lower:,.0f}
- **Upper:** ₹{zone.final_zone_upper:,.0f}
- **Pivot:** ₹{zone.liquidity_pivot:,.0f}

**Calculation:**
- Base Zone: ±{zone.base_zone_width:.0f} pts (Option Chain)
- Gamma Adjusted: ±{zone.gamma_adjusted_width:.0f} pts
- Final Width: ±{(zone.final_zone_upper - zone.final_zone_lower) / 2:.0f} pts

**Confidence:** {zone.confidence:.0f}% | **Sources Aligned:** {zone.sources_aligned}/4

**Status:** {"✅ CONFIRMED" if zone.price_action_confirmed else "⏳ MONITORING"}
"""


def format_premium_divergence_display(div: PremiumDivergence) -> str:
    """Format premium divergence for display"""

    if not div.detected:
        return f"""
**💹 PREMIUM FLOW STATUS:** NORMAL
Money still flowing - No divergence detected
"""

    color = "#00ff00" if div.entry_signal else "#ffaa00"
    emoji = "🚨" if div.entry_signal else "⚠️"

    return f"""
**{emoji} PREMIUM DIVERGENCE DETECTED** {"- ENTRY SIGNAL!" if div.entry_signal else ""}

**Setup:** {div.direction}
**Spot:** {div.spot_move} at ₹{div.spot_price:,.2f}
**Option Premium:** {div.premium_move}
**Divergence Strength:** {div.divergence_strength:.0f}%

**{div.reason}**

{"**✅ ENTRY CONDITION MET - Consider taking trade**" if div.entry_signal else "**⏳ MONITOR - Wait for stronger confirmation**"}
"""
