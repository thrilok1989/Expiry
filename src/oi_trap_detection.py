"""
OI Trap Detection Module
Detects retail traps in option chain through OI manipulation patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TrapType(Enum):
    """Types of OI traps"""
    FAKE_BREAKOUT = "Fake Breakout Trap"
    FALSE_OI_BUILDUP = "False OI Buildup"
    SUDDEN_UNWINDING = "Sudden OI Unwinding"
    SMART_MONEY_TRAP = "Smart Money Trapping Retail"
    SQUEEZE_TRAP = "Squeeze Trap"
    NO_TRAP = "No Trap Detected"


@dataclass
class OITrapResult:
    """OI Trap Detection Result"""
    trap_detected: bool
    trap_type: TrapType
    trap_probability: float  # 0-100
    retail_trap_score: float  # 0-100 (higher = more retail trapped)
    smart_money_signal: str  # "Accumulating", "Distributing", "Neutral"
    trapped_direction: str  # "CALL_BUYERS", "PUT_BUYERS", "BOTH", "NONE"
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "EXTREME"
    oi_manipulation_score: float  # 0-100
    recommendation: str
    signals: List[str]
    trap_strikes: List[int]


class OITrapDetector:
    """
    Open Interest Trap Detection System

    Detects when retail traders are being trapped by:
    - Fake OI buildup (false signals)
    - Sudden OI unwinding (stop hunting)
    - Smart money distribution/accumulation
    - False breakout patterns
    """

    def __init__(self):
        """Initialize OI Trap Detector"""
        # Thresholds for trap detection
        self.oi_change_threshold = 10  # % change considered significant
        self.fake_buildup_threshold = 25  # % OI increase without price follow-through
        self.unwinding_threshold = -20  # % OI decrease (sudden)
        self.retail_concentration_threshold = 70  # % OI in ATM ±1 strikes

    def detect_traps(
        self,
        option_chain: Dict,
        price_current: float,
        price_history: pd.DataFrame,
        previous_oi_data: Optional[Dict] = None
    ) -> OITrapResult:
        """
        Comprehensive OI trap detection

        Args:
            option_chain: Current option chain data
            price_current: Current underlying price
            price_history: Historical price data
            previous_oi_data: Previous option chain snapshot for comparison

        Returns:
            OITrapResult with complete analysis
        """
        signals = []
        trap_strikes = []

        # Extract option chain data
        ce_data = option_chain.get('CE', {})
        pe_data = option_chain.get('PE', {})

        ce_strikes = ce_data.get('strikePrice', [])
        pe_strikes = pe_data.get('strikePrice', [])
        ce_oi = ce_data.get('openInterest', [])
        pe_oi = pe_data.get('openInterest', [])
        ce_oi_change = ce_data.get('changeinOpenInterest', [])
        pe_oi_change = pe_data.get('changeinOpenInterest', [])
        ce_volume = ce_data.get('totalTradedVolume', [])
        pe_volume = pe_data.get('totalTradedVolume', [])

        if not ce_strikes or not pe_strikes:
            return self._no_trap_result()

        # Find ATM strike
        atm_strike = self._find_atm_strike(ce_strikes, price_current)

        # 1. Detect Fake OI Buildup
        fake_buildup_detected, fake_signals, fake_strikes = self._detect_fake_buildup(
            ce_strikes, pe_strikes, ce_oi, pe_oi, ce_oi_change, pe_oi_change,
            atm_strike, price_history
        )
        signals.extend(fake_signals)
        trap_strikes.extend(fake_strikes)

        # 2. Detect Sudden OI Unwinding
        unwinding_detected, unwind_signals, unwind_strikes = self._detect_oi_unwinding(
            ce_strikes, pe_strikes, ce_oi_change, pe_oi_change, atm_strike
        )
        signals.extend(unwind_signals)
        trap_strikes.extend(unwind_strikes)

        # 3. Detect Retail Concentration
        retail_trap_score, retail_signals = self._calculate_retail_concentration(
            ce_strikes, pe_strikes, ce_oi, pe_oi, ce_volume, pe_volume, atm_strike
        )
        signals.extend(retail_signals)

        # 4. Detect Smart Money Behavior
        smart_money_signal, smart_signals = self._detect_smart_money(
            ce_strikes, pe_strikes, ce_oi, pe_oi, ce_oi_change, pe_oi_change,
            ce_volume, pe_volume, atm_strike
        )
        signals.extend(smart_signals)

        # 5. Detect OI Manipulation
        oi_manipulation_score, manip_signals = self._detect_oi_manipulation(
            ce_oi, pe_oi, ce_oi_change, pe_oi_change, ce_volume, pe_volume
        )
        signals.extend(manip_signals)

        # 6. Determine Trapped Direction
        trapped_direction = self._determine_trapped_direction(
            ce_oi_change, pe_oi_change, ce_volume, pe_volume
        )

        # 7. Calculate Overall Trap Probability
        trap_probability = self._calculate_trap_probability(
            fake_buildup_detected,
            unwinding_detected,
            retail_trap_score,
            oi_manipulation_score,
            smart_money_signal
        )

        # 8. Classify Trap Type
        trap_type = self._classify_trap_type(
            fake_buildup_detected,
            unwinding_detected,
            retail_trap_score,
            smart_money_signal
        )

        # 9. Assess Risk Level
        risk_level = self._assess_risk_level(trap_probability, retail_trap_score)

        # 10. Generate Recommendation
        recommendation = self._generate_recommendation(
            trap_type, trap_probability, risk_level, trapped_direction, smart_money_signal
        )

        trap_detected = trap_probability > 40

        return OITrapResult(
            trap_detected=trap_detected,
            trap_type=trap_type,
            trap_probability=trap_probability,
            retail_trap_score=retail_trap_score,
            smart_money_signal=smart_money_signal,
            trapped_direction=trapped_direction,
            risk_level=risk_level,
            oi_manipulation_score=oi_manipulation_score,
            recommendation=recommendation,
            signals=signals,
            trap_strikes=sorted(set(trap_strikes))
        )

    def _find_atm_strike(self, strikes: List[float], price: float) -> float:
        """Find ATM strike closest to current price"""
        if not strikes:
            return price
        return min(strikes, key=lambda x: abs(x - price))

    def _detect_fake_buildup(
        self,
        ce_strikes: List[float],
        pe_strikes: List[float],
        ce_oi: List[int],
        pe_oi: List[int],
        ce_oi_change: List[int],
        pe_oi_change: List[int],
        atm_strike: float,
        price_history: pd.DataFrame
    ) -> Tuple[bool, List[str], List[int]]:
        """
        Detect fake OI buildup patterns

        Fake buildup = Large OI increase but price doesn't follow
        """
        signals = []
        fake_strikes = []

        # Get ATM ±2 strikes indices
        ce_atm_idx = self._get_strike_index(ce_strikes, atm_strike)
        pe_atm_idx = self._get_strike_index(pe_strikes, atm_strike)

        if ce_atm_idx is None or pe_atm_idx is None:
            return False, signals, fake_strikes

        # Check recent price movement
        if len(price_history) >= 10:
            recent_change_pct = (
                (price_history['close'].iloc[-1] - price_history['close'].iloc[-10]) /
                price_history['close'].iloc[-10] * 100
            )
        else:
            recent_change_pct = 0

        fake_buildup_detected = False

        # Check Call OI buildup without upside movement
        ce_range = slice(max(0, ce_atm_idx - 2), min(len(ce_oi), ce_atm_idx + 3))
        ce_total_change = sum(ce_oi_change[ce_range]) if ce_oi_change else 0
        ce_total_oi = sum(ce_oi[ce_range]) if ce_oi else 1

        if ce_total_oi > 0:
            ce_change_pct = (ce_total_change / ce_total_oi) * 100

            if ce_change_pct > self.fake_buildup_threshold and recent_change_pct < 0.5:
                signals.append(f"⚠️ Fake CALL Buildup: {ce_change_pct:.1f}% OI increase, price flat")
                fake_buildup_detected = True
                fake_strikes.extend([int(ce_strikes[i]) for i in range(ce_range.start, ce_range.stop)])

        # Check Put OI buildup without downside movement
        pe_range = slice(max(0, pe_atm_idx - 2), min(len(pe_oi), pe_atm_idx + 3))
        pe_total_change = sum(pe_oi_change[pe_range]) if pe_oi_change else 0
        pe_total_oi = sum(pe_oi[pe_range]) if pe_oi else 1

        if pe_total_oi > 0:
            pe_change_pct = (pe_total_change / pe_total_oi) * 100

            if pe_change_pct > self.fake_buildup_threshold and recent_change_pct > -0.5:
                signals.append(f"⚠️ Fake PUT Buildup: {pe_change_pct:.1f}% OI increase, price flat")
                fake_buildup_detected = True
                fake_strikes.extend([int(pe_strikes[i]) for i in range(pe_range.start, pe_range.stop)])

        return fake_buildup_detected, signals, fake_strikes

    def _detect_oi_unwinding(
        self,
        ce_strikes: List[float],
        pe_strikes: List[float],
        ce_oi_change: List[int],
        pe_oi_change: List[int],
        atm_strike: float
    ) -> Tuple[bool, List[str], List[int]]:
        """Detect sudden OI unwinding (trap sprung)"""
        signals = []
        unwind_strikes = []

        ce_atm_idx = self._get_strike_index(ce_strikes, atm_strike)
        pe_atm_idx = self._get_strike_index(pe_strikes, atm_strike)

        if ce_atm_idx is None or pe_atm_idx is None:
            return False, signals, unwind_strikes

        unwinding_detected = False

        # Check Call unwinding
        if ce_oi_change and ce_atm_idx < len(ce_oi_change):
            ce_change = ce_oi_change[ce_atm_idx]
            if ce_change < 0:  # Negative = unwinding
                ce_change_pct = ce_change  # Already percentage or absolute

                if ce_change_pct < self.unwinding_threshold * 1000:  # Scale check
                    signals.append(f"🚨 CALL Unwinding: {ce_change:,} contracts at {int(atm_strike)}")
                    unwinding_detected = True
                    unwind_strikes.append(int(atm_strike))

        # Check Put unwinding
        if pe_oi_change and pe_atm_idx < len(pe_oi_change):
            pe_change = pe_oi_change[pe_atm_idx]
            if pe_change < 0:
                if pe_change < self.unwinding_threshold * 1000:
                    signals.append(f"🚨 PUT Unwinding: {pe_change:,} contracts at {int(atm_strike)}")
                    unwinding_detected = True
                    unwind_strikes.append(int(atm_strike))

        return unwinding_detected, signals, unwind_strikes

    def _calculate_retail_concentration(
        self,
        ce_strikes: List[float],
        pe_strikes: List[float],
        ce_oi: List[int],
        pe_oi: List[int],
        ce_volume: List[int],
        pe_volume: List[int],
        atm_strike: float
    ) -> Tuple[float, List[str]]:
        """
        Calculate retail concentration score

        High retail activity = ATM ±1 strikes have disproportionate OI/volume
        """
        signals = []

        ce_atm_idx = self._get_strike_index(ce_strikes, atm_strike)
        pe_atm_idx = self._get_strike_index(pe_strikes, atm_strike)

        if ce_atm_idx is None or pe_atm_idx is None:
            return 0.0, signals

        # ATM ±1 range
        ce_atm_range = slice(max(0, ce_atm_idx - 1), min(len(ce_oi), ce_atm_idx + 2))
        pe_atm_range = slice(max(0, pe_atm_idx - 1), min(len(pe_oi), pe_atm_idx + 2))

        # Calculate concentration
        ce_atm_oi = sum(ce_oi[ce_atm_range]) if ce_oi else 0
        ce_total_oi = sum(ce_oi) if ce_oi else 1
        ce_concentration = (ce_atm_oi / ce_total_oi * 100) if ce_total_oi > 0 else 0

        pe_atm_oi = sum(pe_oi[pe_atm_range]) if pe_oi else 0
        pe_total_oi = sum(pe_oi) if pe_oi else 1
        pe_concentration = (pe_atm_oi / pe_total_oi * 100) if pe_total_oi > 0 else 0

        avg_concentration = (ce_concentration + pe_concentration) / 2

        # Volume concentration
        ce_atm_vol = sum(ce_volume[ce_atm_range]) if ce_volume else 0
        ce_total_vol = sum(ce_volume) if ce_volume else 1
        ce_vol_concentration = (ce_atm_vol / ce_total_vol * 100) if ce_total_vol > 0 else 0

        pe_atm_vol = sum(pe_volume[pe_atm_range]) if pe_volume else 0
        pe_total_vol = sum(pe_volume) if pe_volume else 1
        pe_vol_concentration = (pe_atm_vol / pe_total_vol * 100) if pe_total_vol > 0 else 0

        avg_vol_concentration = (ce_vol_concentration + pe_vol_concentration) / 2

        # Retail trap score (higher concentration = more retail)
        retail_score = (avg_concentration * 0.6 + avg_vol_concentration * 0.4)

        if retail_score > 70:
            signals.append(f"🎯 HIGH Retail Concentration: {retail_score:.1f}% at ATM")
        elif retail_score > 50:
            signals.append(f"⚠️ Moderate Retail Activity: {retail_score:.1f}%")

        return retail_score, signals

    def _detect_smart_money(
        self,
        ce_strikes: List[float],
        pe_strikes: List[float],
        ce_oi: List[int],
        pe_oi: List[int],
        ce_oi_change: List[int],
        pe_oi_change: List[int],
        ce_volume: List[int],
        pe_volume: List[int],
        atm_strike: float
    ) -> Tuple[str, List[str]]:
        """
        Detect smart money behavior

        Smart money signatures:
        - Deep ITM/OTM activity (institutions hedge)
        - Large OI with low volume (patient positioning)
        - OTM put protection (institutions buying insurance)
        """
        signals = []

        ce_atm_idx = self._get_strike_index(ce_strikes, atm_strike)
        pe_atm_idx = self._get_strike_index(pe_strikes, atm_strike)

        if ce_atm_idx is None or pe_atm_idx is None:
            return "Neutral", signals

        # Check deep ITM activity (smart money hedging)
        deep_itm_ce_idx = max(0, ce_atm_idx - 5) if ce_atm_idx >= 5 else 0
        deep_itm_pe_idx = min(len(pe_strikes) - 1, pe_atm_idx + 5) if pe_atm_idx + 5 < len(pe_strikes) else len(pe_strikes) - 1

        deep_ce_oi = ce_oi[deep_itm_ce_idx] if ce_oi and deep_itm_ce_idx < len(ce_oi) else 0
        deep_pe_oi = pe_oi[deep_itm_pe_idx] if pe_oi and deep_itm_pe_idx < len(pe_oi) else 0

        # Check OTM put buying (protection)
        otm_put_idx = min(len(pe_strikes) - 1, pe_atm_idx - 3) if pe_atm_idx >= 3 else 0
        otm_put_oi_change = pe_oi_change[otm_put_idx] if pe_oi_change and otm_put_idx < len(pe_oi_change) else 0

        smart_money_score = 0

        # Deep ITM presence
        if deep_ce_oi > 10000 or deep_pe_oi > 10000:
            smart_money_score += 30
            signals.append(f"🏦 Deep ITM Activity: Institutional hedging detected")

        # OTM put protection
        if otm_put_oi_change > 5000:
            smart_money_score += 40
            signals.append(f"🛡️ OTM Put Buying: Smart money protection at {int(pe_strikes[otm_put_idx])}")

        # Large OI, low volume (patient positioning)
        total_ce_oi = sum(ce_oi) if ce_oi else 1
        total_ce_vol = sum(ce_volume) if ce_volume else 1
        oi_to_vol_ratio = total_ce_oi / total_ce_vol if total_ce_vol > 0 else 0

        if oi_to_vol_ratio > 5:
            smart_money_score += 30
            signals.append(f"📊 High OI/Volume Ratio: Smart money positioning")

        # Classify behavior
        if smart_money_score >= 60:
            return "Accumulating", signals
        elif smart_money_score >= 30:
            return "Distributing", signals
        else:
            return "Neutral", signals

    def _detect_oi_manipulation(
        self,
        ce_oi: List[int],
        pe_oi: List[int],
        ce_oi_change: List[int],
        pe_oi_change: List[int],
        ce_volume: List[int],
        pe_volume: List[int]
    ) -> Tuple[float, List[str]]:
        """Detect OI manipulation patterns"""
        signals = []
        manipulation_score = 0.0

        # 1. Sudden large OI changes
        if ce_oi_change:
            max_ce_change = max(ce_oi_change, default=0)
            avg_ce_change = np.mean(ce_oi_change) if ce_oi_change else 0

            if max_ce_change > avg_ce_change * 3:
                manipulation_score += 35
                signals.append(f"⚠️ Abnormal CALL OI spike detected")

        if pe_oi_change:
            max_pe_change = max(pe_oi_change, default=0)
            avg_pe_change = np.mean(pe_oi_change) if pe_oi_change else 0

            if max_pe_change > avg_pe_change * 3:
                manipulation_score += 35
                signals.append(f"⚠️ Abnormal PUT OI spike detected")

        # 2. OI increase with very low volume (spoofing?)
        total_ce_oi_change = sum(ce_oi_change) if ce_oi_change else 0
        total_ce_volume = sum(ce_volume) if ce_volume else 1

        if total_ce_oi_change > 0 and total_ce_volume > 0:
            change_to_vol = total_ce_oi_change / total_ce_volume
            if change_to_vol > 10:  # OI increase 10x volume
                manipulation_score += 30
                signals.append(f"🚩 OI increase without proportional volume")

        return np.clip(manipulation_score, 0, 100), signals

    def _determine_trapped_direction(
        self,
        ce_oi_change: List[int],
        pe_oi_change: List[int],
        ce_volume: List[int],
        pe_volume: List[int]
    ) -> str:
        """Determine which side is trapped"""
        ce_activity = (sum(ce_oi_change) if ce_oi_change else 0) + (sum(ce_volume) if ce_volume else 0)
        pe_activity = (sum(pe_oi_change) if pe_oi_change else 0) + (sum(pe_volume) if pe_volume else 0)

        if ce_activity > pe_activity * 1.5:
            return "CALL_BUYERS"
        elif pe_activity > ce_activity * 1.5:
            return "PUT_BUYERS"
        elif ce_activity > 0 and pe_activity > 0:
            return "BOTH"
        else:
            return "NONE"

    def _calculate_trap_probability(
        self,
        fake_buildup: bool,
        unwinding: bool,
        retail_score: float,
        manipulation_score: float,
        smart_money_signal: str
    ) -> float:
        """Calculate overall trap probability (0-100)"""
        probability = 0.0

        if fake_buildup:
            probability += 30

        if unwinding:
            probability += 35

        probability += retail_score * 0.2  # Max 20 from retail score

        probability += manipulation_score * 0.15  # Max 15 from manipulation

        if smart_money_signal == "Distributing":
            probability += 10

        return np.clip(probability, 0, 100)

    def _classify_trap_type(
        self,
        fake_buildup: bool,
        unwinding: bool,
        retail_score: float,
        smart_money_signal: str
    ) -> TrapType:
        """Classify the type of trap"""
        if unwinding:
            return TrapType.SUDDEN_UNWINDING

        if fake_buildup:
            return TrapType.FALSE_OI_BUILDUP

        if smart_money_signal == "Distributing" and retail_score > 60:
            return TrapType.SMART_MONEY_TRAP

        if retail_score > 70:
            return TrapType.SQUEEZE_TRAP

        return TrapType.NO_TRAP

    def _assess_risk_level(self, trap_probability: float, retail_score: float) -> str:
        """Assess overall risk level"""
        combined_risk = (trap_probability + retail_score) / 2

        if combined_risk >= 75:
            return "EXTREME"
        elif combined_risk >= 60:
            return "HIGH"
        elif combined_risk >= 40:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_recommendation(
        self,
        trap_type: TrapType,
        trap_probability: float,
        risk_level: str,
        trapped_direction: str,
        smart_money_signal: str
    ) -> str:
        """Generate trading recommendation"""
        if risk_level == "EXTREME":
            return "🚨 AVOID TRADING - Extreme trap risk detected"

        if trap_type == TrapType.SUDDEN_UNWINDING:
            return "⚠️ OI Unwinding in progress - Wait for stabilization"

        if trap_type == TrapType.FALSE_OI_BUILDUP:
            return "🚫 Fake OI pattern - Don't follow the crowd"

        if trap_type == TrapType.SMART_MONEY_TRAP:
            if smart_money_signal == "Accumulating":
                return "✅ Follow smart money accumulation"
            else:
                return "⚠️ Smart money distributing - Consider opposite direction"

        if risk_level == "HIGH":
            return "⚠️ High trap risk - Reduce position size"

        if trapped_direction == "CALL_BUYERS":
            return "📉 Call buyers likely trapped - Bearish edge"
        elif trapped_direction == "PUT_BUYERS":
            return "📈 Put buyers likely trapped - Bullish edge"

        return "✅ No significant trap detected - Normal conditions"

    def _get_strike_index(self, strikes: List[float], target_strike: float) -> Optional[int]:
        """Get index of strike in list"""
        try:
            return strikes.index(target_strike)
        except ValueError:
            # Find closest
            if not strikes:
                return None
            closest = min(strikes, key=lambda x: abs(x - target_strike))
            try:
                return strikes.index(closest)
            except ValueError:
                return None

    def _no_trap_result(self) -> OITrapResult:
        """Return default no-trap result"""
        return OITrapResult(
            trap_detected=False,
            trap_type=TrapType.NO_TRAP,
            trap_probability=0.0,
            retail_trap_score=0.0,
            smart_money_signal="Neutral",
            trapped_direction="NONE",
            risk_level="LOW",
            oi_manipulation_score=0.0,
            recommendation="Insufficient data for trap analysis",
            signals=[],
            trap_strikes=[]
        )


def format_oi_trap_report(result: OITrapResult) -> str:
    """Format OI trap analysis as readable report"""
    report = f"""
╔══════════════════════════════════════════════════════════╗
║          OI TRAP DETECTION ANALYSIS                      ║
╚══════════════════════════════════════════════════════════╝

🎯 TRAP STATUS: {'DETECTED ⚠️' if result.trap_detected else 'Clear ✅'}
📊 TRAP TYPE: {result.trap_type.value}
🎲 TRAP PROBABILITY: {result.trap_probability:.1f}%
⚠️  RISK LEVEL: {result.risk_level}

─────────────────────────────────────────────────────────
RETAIL VS SMART MONEY:
  • Retail Trap Score: {result.retail_trap_score:.1f}/100
  • Smart Money Signal: {result.smart_money_signal}
  • Trapped Direction: {result.trapped_direction}

MANIPULATION ANALYSIS:
  • OI Manipulation Score: {result.oi_manipulation_score:.1f}/100
  • Trap Strikes: {', '.join(map(str, result.trap_strikes)) if result.trap_strikes else 'None'}

─────────────────────────────────────────────────────────
💡 RECOMMENDATION:
{result.recommendation}

─────────────────────────────────────────────────────────
📌 TRAP SIGNALS:
"""
    for signal in result.signals:
        report += f"  • {signal}\n"

    if not result.signals:
        report += "  • No trap signals detected\n"

    return report
