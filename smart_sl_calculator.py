"""
Smart Stop Loss Calculator - Dynamic SL Based on Market Context
NOT fixed 20 points - adapts to pattern invalidation, regime flip, market mood changes
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SmartSL:
    """Smart Stop Loss with invalidation triggers"""
    price: float
    buffer_points: float
    invalidation_triggers: list
    reasoning: str
    atr_multiplier: float = 1.5


class SmartSLCalculator:
    """Calculate dynamic stop loss based on multiple factors"""

    def __init__(
        self,
        current_price: float,
        atr: float,
        position_type: str  # "LONG" or "SHORT"
    ):
        self.current_price = current_price
        self.atr = atr
        self.position_type = position_type

    def calculate_smart_sl(
        self,
        entry_level: Dict,
        pattern_data: Optional[Dict] = None,
        regime: Optional[str] = None,
        atm_bias: Optional[Dict] = None,
        sr_levels: Optional[list] = None
    ) -> SmartSL:
        """
        Calculate smart SL based on:
        1. Pattern invalidation points
        2. ATR-adjusted buffer
        3. Next S/R level breach
        4. Regime flip levels
        5. ATM bias flip triggers
        """

        invalidation_triggers = []
        sl_candidates = []

        # 1. Pattern-based invalidation
        if pattern_data:
            pattern_sl = self._get_pattern_invalidation_sl(pattern_data)
            if pattern_sl:
                sl_candidates.append({
                    'price': pattern_sl,
                    'reason': f"Pattern invalidation ({pattern_data.get('type', 'Unknown')})",
                    'priority': 1
                })
                invalidation_triggers.append({
                    'type': 'pattern_break',
                    'description': f"{pattern_data.get('type', 'Pattern')} invalidated"
                })

        # 2. Entry level + ATR buffer
        entry_price = entry_level.get('price', self.current_price)
        entry_lower = entry_level.get('lower', entry_price - 10)
        entry_upper = entry_level.get('upper', entry_price + 10)

        atr_buffer = self.atr * 1.5  # 1.5x ATR for breathing room

        if self.position_type == "LONG":
            atr_sl = entry_lower - atr_buffer
        else:  # SHORT
            atr_sl = entry_upper + atr_buffer

        sl_candidates.append({
            'price': atr_sl,
            'reason': f"Entry zone breach + 1.5x ATR buffer ({atr_buffer:.1f} pts)",
            'priority': 2
        })

        # 3. Next S/R level breach
        if sr_levels:
            sr_sl = self._get_sr_breach_sl(sr_levels)
            if sr_sl:
                sl_candidates.append({
                    'price': sr_sl,
                    'reason': "Next S/R level breach",
                    'priority': 2
                })
                invalidation_triggers.append({
                    'type': 'sr_breach',
                    'description': "Price breaks next S/R level"
                })

        # 4. Regime flip level
        if regime:
            regime_flip_trigger = {
                'type': 'regime_flip',
                'description': f"Market regime flips from {regime}"
            }
            invalidation_triggers.append(regime_flip_trigger)

        # 5. ATM Bias flip
        if atm_bias:
            current_verdict = atm_bias.get('verdict', 'NEUTRAL')
            bias_flip_trigger = {
                'type': 'atm_bias_flip',
                'description': f"ATM bias flips from {current_verdict}"
            }
            invalidation_triggers.append(bias_flip_trigger)

        # 6. OI unwinding trigger
        invalidation_triggers.append({
            'type': 'oi_unwinding',
            'description': "OI unwinding >20% at entry strike"
        })

        # 7. Volume spike trigger
        invalidation_triggers.append({
            'type': 'volume_spike',
            'description': "3x+ volume spike against position"
        })

        # Select the SL that gives best risk/reward while protecting capital
        # Use tightest SL from priority 1 sources, or ATR-based if no pattern
        priority_1_sls = [sl for sl in sl_candidates if sl['priority'] == 1]

        if priority_1_sls:
            # Use pattern-based SL
            selected_sl = min(priority_1_sls, key=lambda x: abs(x['price'] - self.current_price))
        else:
            # Use ATR-based SL
            selected_sl = next((sl for sl in sl_candidates if 'ATR' in sl['reason']), sl_candidates[0])

        sl_price = selected_sl['price']
        buffer_points = abs(sl_price - entry_price)

        # Build reasoning
        reasoning = f"SL: ₹{sl_price:,.0f} ({buffer_points:.0f} pts from entry)\n"
        reasoning += f"Primary: {selected_sl['reason']}\n"
        reasoning += f"Invalidation triggers: {len(invalidation_triggers)} factors monitored\n"
        reasoning += "Exit if: " + ", ".join([t['description'] for t in invalidation_triggers[:3]])

        return SmartSL(
            price=sl_price,
            buffer_points=buffer_points,
            invalidation_triggers=invalidation_triggers,
            reasoning=reasoning,
            atr_multiplier=1.5
        )

    def _get_pattern_invalidation_sl(self, pattern_data: Dict) -> Optional[float]:
        """Get SL based on pattern invalidation point"""
        pattern_type = pattern_data.get('type', '')

        if pattern_type == 'Head & Shoulders':
            # SL above right shoulder high for SHORT
            if self.position_type == "SHORT":
                right_shoulder_high = pattern_data.get('right_shoulder_high', 0)
                if right_shoulder_high > 0:
                    return right_shoulder_high + 10  # +10pts buffer

        elif pattern_type == 'Inverse H&S':
            # SL below right shoulder low for LONG
            if self.position_type == "LONG":
                right_shoulder_low = pattern_data.get('right_shoulder_low', 0)
                if right_shoulder_low > 0:
                    return right_shoulder_low - 10  # -10pts buffer

        elif 'Triangle' in pattern_type:
            # SL beyond triangle boundary
            boundary = pattern_data.get('boundary', 0)
            if boundary > 0:
                if self.position_type == "LONG":
                    return boundary - 15
                else:
                    return boundary + 15

        elif 'Flag' in pattern_type:
            # SL beyond flag pole low/high
            if 'Bull' in pattern_type and self.position_type == "LONG":
                pole_low = pattern_data.get('pole_low', 0)
                if pole_low > 0:
                    return pole_low - 10

            elif 'Bear' in pattern_type and self.position_type == "SHORT":
                pole_high = pattern_data.get('pole_high', 0)
                if pole_high > 0:
                    return pole_high + 10

        return None

    def _get_sr_breach_sl(self, sr_levels: list) -> Optional[float]:
        """Get SL based on next S/R level breach"""
        if not sr_levels:
            return None

        if self.position_type == "LONG":
            # Find nearest support below current price
            supports_below = [
                level for level in sr_levels
                if level.get('price', 0) < self.current_price
            ]
            if supports_below:
                nearest_support = max(supports_below, key=lambda x: x['price'])
                # SL below this support
                return nearest_support.get('lower', nearest_support['price']) - 10

        else:  # SHORT
            # Find nearest resistance above current price
            resistances_above = [
                level for level in sr_levels
                if level.get('price', 0) > self.current_price
            ]
            if resistances_above:
                nearest_resistance = min(resistances_above, key=lambda x: x['price'])
                # SL above this resistance
                return nearest_resistance.get('upper', nearest_resistance['price']) + 10

        return None

    def tighten_sl(
        self,
        current_sl: float,
        factors_triggered: int,
        current_profit_points: float
    ) -> Tuple[float, str]:
        """
        Tighten SL when exit factors trigger but not critical
        Returns: (new_sl, reasoning)
        """

        if self.position_type == "LONG":
            # Tighten by moving SL up
            if factors_triggered >= 2:
                # Move to breakeven + 5pts
                new_sl = self.current_price - (current_profit_points / 2)
                reasoning = f"2+ factors triggered → Trailing SL to ₹{new_sl:,.0f} (protect 50% profit)"
            else:
                # Move SL to breakeven
                new_sl = max(current_sl, self.current_price - 5)
                reasoning = f"1 factor triggered → Move SL to breakeven ₹{new_sl:,.0f}"

        else:  # SHORT
            # Tighten by moving SL down
            if factors_triggered >= 2:
                # Move to breakeven + 5pts
                new_sl = self.current_price + (current_profit_points / 2)
                reasoning = f"2+ factors triggered → Trailing SL to ₹{new_sl:,.0f} (protect 50% profit)"
            else:
                # Move SL to breakeven
                new_sl = min(current_sl, self.current_price + 5)
                reasoning = f"1 factor triggered → Move SL to breakeven ₹{new_sl:,.0f}"

        return new_sl, reasoning


def calculate_atr(df, period: int = 14) -> float:
    """Calculate ATR from DataFrame"""
    try:
        if df is None or len(df) < period:
            return 50.0  # Default fallback

        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]

        return float(atr) if atr > 0 else 50.0

    except Exception as e:
        logger.warning(f"ATR calculation failed: {e}")
        return 50.0  # Fallback for NIFTY


# Example usage
if __name__ == "__main__":
    # Example: LONG position at Head & Shoulders neckline
    calculator = SmartSLCalculator(
        current_price=24500,
        atr=55,
        position_type="LONG"
    )

    pattern = {
        'type': 'Inverse H&S',
        'right_shoulder_low': 24420,
        'neckline': 24520
    }

    entry_level = {
        'price': 24480,
        'upper': 24495,
        'lower': 24465
    }

    smart_sl = calculator.calculate_smart_sl(
        entry_level=entry_level,
        pattern_data=pattern,
        regime='BULLISH',
        atm_bias={'verdict': 'BULLISH', 'score': 8}
    )

    print(smart_sl.reasoning)
    print(f"Invalidation triggers: {len(smart_sl.invalidation_triggers)}")
    for trigger in smart_sl.invalidation_triggers:
        print(f"  - {trigger['type']}: {trigger['description']}")
