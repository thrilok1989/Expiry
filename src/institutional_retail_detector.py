"""
Institutional vs Retail Detection Module
Detects institutional and retail entry patterns with high accuracy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ParticipantType(Enum):
    """Market participant classification"""
    INSTITUTIONAL = "Institutional"
    RETAIL = "Retail"
    MIXED = "Mixed"
    UNKNOWN = "Unknown"


class EntryType(Enum):
    """Entry type classification"""
    INST_ACCUMULATION = "Institutional Accumulation"
    INST_DISTRIBUTION = "Institutional Distribution"
    RETAIL_FOMO = "Retail FOMO"
    RETAIL_PANIC = "Retail Panic"
    TRAP_SETUP = "Trap Setup"
    NEUTRAL = "Neutral"


@dataclass
class MarketParticipantResult:
    """Market participant detection result"""
    dominant_participant: ParticipantType
    entry_type: EntryType
    institutional_confidence: float  # 0-100
    retail_confidence: float  # 0-100
    smart_money_detected: bool
    dumb_money_detected: bool
    orderflow_signature: str
    volume_profile: str
    oi_classification: str
    price_action_quality: str
    recommendation: str
    signals: List[str]


class InstitutionalRetailDetector:
    """
    Institutional vs Retail Detection System

    Detects market participants using:
    1. Volume signatures (size, aggression, follow-through)
    2. Option chain behavior (OI patterns, ITM/OTM activity)
    3. Price action quality (smooth vs choppy)
    4. Liquidity zone interactions
    """

    def __init__(self):
        """Initialize detector"""
        # Thresholds
        self.large_volume_threshold = 2.0  # 2x average
        self.institutional_candle_body_pct = 70  # % body vs total range
        self.retail_wick_threshold = 40  # % wick size
        self.smooth_trend_threshold = 0.7  # R² for trend line

    def detect_participant(
        self,
        df: pd.DataFrame,
        option_chain: Dict,
        price_current: float,
        liquidity_zones: Optional[List[float]] = None
    ) -> MarketParticipantResult:
        """
        Detect institutional vs retail participation

        Args:
            df: OHLCV dataframe
            option_chain: Option chain data
            price_current: Current price
            liquidity_zones: Key liquidity levels

        Returns:
            MarketParticipantResult with classification
        """
        signals = []

        if len(df) < 10:
            return self._default_result()

        # 1. Volume Signature Analysis
        volume_sig, vol_signals = self._analyze_volume_signature(df)
        signals.extend(vol_signals)

        # 2. Option Chain Behavior
        oi_classification, oi_signals = self._analyze_option_behavior(
            option_chain, price_current
        )
        signals.extend(oi_signals)

        # 3. Price Action Quality
        price_quality, price_signals = self._analyze_price_action_quality(df)
        signals.extend(price_signals)

        # 4. Liquidity Zone Interaction
        liquidity_behavior, liq_signals = self._analyze_liquidity_interaction(
            df, liquidity_zones
        )
        signals.extend(liq_signals)

        # 5. Calculate Institutional Confidence
        inst_confidence = self._calculate_institutional_confidence(
            volume_sig, oi_classification, price_quality, liquidity_behavior
        )

        # 6. Calculate Retail Confidence
        retail_confidence = 100 - inst_confidence  # Inverse relationship

        # 7. Determine Dominant Participant
        dominant_participant = self._classify_participant(inst_confidence, retail_confidence)

        # 8. Classify Entry Type
        entry_type = self._classify_entry_type(
            dominant_participant, volume_sig, price_quality, df
        )

        # 9. Smart Money / Dumb Money Detection
        smart_money = inst_confidence > 65
        dumb_money = retail_confidence > 70

        # 10. Generate Recommendation
        recommendation = self._generate_recommendation(
            dominant_participant, entry_type, inst_confidence
        )

        return MarketParticipantResult(
            dominant_participant=dominant_participant,
            entry_type=entry_type,
            institutional_confidence=inst_confidence,
            retail_confidence=retail_confidence,
            smart_money_detected=smart_money,
            dumb_money_detected=dumb_money,
            orderflow_signature=volume_sig,
            volume_profile=self._describe_volume_profile(df),
            oi_classification=oi_classification,
            price_action_quality=price_quality,
            recommendation=recommendation,
            signals=signals
        )

    def _analyze_volume_signature(self, df: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Analyze volume signature

        Institutional signatures:
        - Large volume spikes (2x+ average)
        - Big body candles (clean displacement)
        - Low wicks (aggressive execution)
        - Follow-through (continuation after spike)

        Retail signatures:
        - Small frequent spikes
        - High wick candles (indecision)
        - No follow-through
        - Choppy movement
        """
        signals = []

        if 'volume' not in df.columns:
            return "Unknown", signals

        recent = df.tail(10)
        avg_volume = df['volume'].tail(20).mean()
        current_volume = recent['volume'].iloc[-1]

        # Volume spike detection
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Candle body analysis
        recent_candles = recent.tail(5)
        body_sizes = abs(recent_candles['close'] - recent_candles['open'])
        total_ranges = recent_candles['high'] - recent_candles['low']
        body_pct = (body_sizes / total_ranges * 100).mean() if (total_ranges != 0).all() else 50

        # Wick analysis
        wick_sizes = total_ranges - body_sizes
        wick_pct = (wick_sizes / total_ranges * 100).mean() if (total_ranges != 0).all() else 50

        # Follow-through analysis
        price_change_5 = (recent['close'].iloc[-1] - recent['close'].iloc[-5]) / recent['close'].iloc[-5] * 100 if len(recent) >= 5 else 0

        # Classification
        inst_score = 0

        if volume_ratio > self.large_volume_threshold:
            inst_score += 30
            signals.append(f"🏦 Large Volume: {volume_ratio:.1f}x average")

        if body_pct > self.institutional_candle_body_pct:
            inst_score += 25
            signals.append(f"✅ Clean Candles: {body_pct:.1f}% body")

        if wick_pct < 30:
            inst_score += 20
            signals.append(f"⚡ Aggressive Execution: Low wicks")

        if abs(price_change_5) > 0.5:
            inst_score += 25
            signals.append(f"📈 Follow-Through: {price_change_5:+.2f}%")

        if inst_score >= 60:
            return "Institutional", signals
        elif inst_score >= 30:
            return "Mixed", signals
        else:
            return "Retail", signals

    def _analyze_option_behavior(
        self,
        option_chain: Dict,
        price_current: float
    ) -> Tuple[str, List[str]]:
        """
        Analyze option chain behavior

        Institutional signatures:
        - Deep ITM activity (hedging)
        - OTM ±3 strikes buildup (positioning)
        - Low IV after OI build (patient)
        - Spread activity

        Retail signatures:
        - ATM ±1 hyper-activity
        - OTM far strikes (lottery tickets)
        - High IV (FOMO)
        - Single-leg trades
        """
        signals = []

        ce_data = option_chain.get('CE', {})
        pe_data = option_chain.get('PE', {})

        ce_strikes = ce_data.get('strikePrice', [])
        pe_strikes = pe_data.get('strikePrice', [])
        ce_oi = ce_data.get('openInterest', [])
        pe_oi = pe_data.get('openInterest', [])

        if not ce_strikes or not pe_strikes:
            return "Unknown", signals

        # Find ATM strike
        atm_strike = min(ce_strikes, key=lambda x: abs(x - price_current))
        try:
            atm_idx_ce = ce_strikes.index(atm_strike)
            atm_idx_pe = pe_strikes.index(atm_strike)
        except ValueError:
            return "Unknown", signals

        # ATM ±1 activity (retail signature)
        atm_range_ce = slice(max(0, atm_idx_ce - 1), min(len(ce_oi), atm_idx_ce + 2))
        atm_range_pe = slice(max(0, atm_idx_pe - 1), min(len(pe_oi), atm_idx_pe + 2))

        atm_oi_ce = sum(ce_oi[atm_range_ce]) if ce_oi else 0
        atm_oi_pe = sum(pe_oi[atm_range_pe]) if pe_oi else 0
        total_ce_oi = sum(ce_oi) if ce_oi else 1
        total_pe_oi = sum(pe_oi) if pe_oi else 1

        atm_concentration = ((atm_oi_ce + atm_oi_pe) / (total_ce_oi + total_pe_oi)) * 100

        # Deep ITM activity (institutional signature)
        deep_itm_idx_ce = max(0, atm_idx_ce - 5)
        deep_itm_idx_pe = min(len(pe_oi) - 1, atm_idx_pe + 5)

        deep_itm_oi_ce = ce_oi[deep_itm_idx_ce] if ce_oi and deep_itm_idx_ce < len(ce_oi) else 0
        deep_itm_oi_pe = pe_oi[deep_itm_idx_pe] if pe_oi and deep_itm_idx_pe < len(pe_oi) else 0

        # Classification
        inst_score = 0

        if deep_itm_oi_ce > 5000 or deep_itm_oi_pe > 5000:
            inst_score += 40
            signals.append(f"🏦 Deep ITM Activity: Institutional hedging")

        if atm_concentration < 40:
            inst_score += 30
            signals.append(f"✅ Distributed OI: {atm_concentration:.1f}% ATM")
        else:
            signals.append(f"⚠️ ATM Concentration: {atm_concentration:.1f}% (Retail)")

        # OTM ±3 buildup (institutional positioning)
        otm_range_ce = slice(min(len(ce_oi) - 1, atm_idx_ce + 2), min(len(ce_oi), atm_idx_ce + 5))
        otm_range_pe = slice(max(0, atm_idx_pe - 4), max(0, atm_idx_pe - 1))

        otm_oi_ce = sum(ce_oi[otm_range_ce]) if ce_oi and otm_range_ce.start < len(ce_oi) else 0
        otm_oi_pe = sum(pe_oi[otm_range_pe]) if pe_oi and otm_range_pe.start < len(pe_oi) else 0

        if otm_oi_ce > 10000 or otm_oi_pe > 10000:
            inst_score += 30
            signals.append(f"📊 OTM Positioning: Institutional buildup")

        if inst_score >= 60:
            return "Institutional", signals
        elif inst_score >= 30:
            return "Mixed", signals
        else:
            return "Retail", signals

    def _analyze_price_action_quality(self, df: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Analyze price action quality

        Institutional = Smooth, controlled trends
        Retail = Choppy, noisy movement
        """
        signals = []

        if len(df) < 10:
            return "Unknown", signals

        recent = df.tail(10)

        # Calculate R² for trend line
        closes = recent['close'].values
        x = np.arange(len(closes))

        if len(x) < 2:
            return "Unknown", signals

        try:
            slope, intercept = np.polyfit(x, closes, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((closes - y_pred) ** 2)
            ss_tot = np.sum((closes - np.mean(closes)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        except:
            r_squared = 0

        # Choppiness
        price_changes = abs(recent['close'].diff()).sum()
        net_change = abs(recent['close'].iloc[-1] - recent['close'].iloc[0])
        choppiness = price_changes / net_change if net_change > 0 else 10

        # Classification
        if r_squared > self.smooth_trend_threshold and choppiness < 3:
            signals.append(f"✅ Smooth Trend: R²={r_squared:.2f} (Institutional)")
            return "Institutional", signals
        elif r_squared < 0.4 or choppiness > 5:
            signals.append(f"⚠️ Choppy Movement: Chop={choppiness:.1f} (Retail)")
            return "Retail", signals
        else:
            signals.append(f"📊 Mixed Quality: R²={r_squared:.2f}")
            return "Mixed", signals

    def _analyze_liquidity_interaction(
        self,
        df: pd.DataFrame,
        liquidity_zones: Optional[List[float]]
    ) -> Tuple[str, List[str]]:
        """
        Analyze how price interacts with liquidity zones

        Institutional = Grabs liquidity, then reverses
        Retail = Breakout above liquidity, then trapped
        """
        signals = []

        if liquidity_zones is None or len(df) < 5:
            return "Unknown", signals

        recent_price = df['close'].tail(5)
        current_price = recent_price.iloc[-1]

        # Check if recently swept liquidity
        swept_liquidity = False
        for level in liquidity_zones:
            if abs(current_price - level) < (level * 0.01):  # Within 1%
                # Check if swept and reversed
                if len(recent_price) >= 2:
                    prev_price = recent_price.iloc[-2]
                    if (prev_price > level and current_price < level) or \
                       (prev_price < level and current_price > level):
                        swept_liquidity = True
                        signals.append(f"🏦 Liquidity Sweep: ${level:.0f} (Institutional)")
                        break

        if swept_liquidity:
            return "Institutional", signals
        else:
            return "Neutral", signals

    def _calculate_institutional_confidence(
        self,
        volume_sig: str,
        oi_classification: str,
        price_quality: str,
        liquidity_behavior: str
    ) -> float:
        """Calculate institutional confidence score (0-100)"""
        score = 0

        # Volume signature
        if volume_sig == "Institutional":
            score += 35
        elif volume_sig == "Mixed":
            score += 15

        # OI classification
        if oi_classification == "Institutional":
            score += 35
        elif oi_classification == "Mixed":
            score += 15

        # Price quality
        if price_quality == "Institutional":
            score += 20
        elif price_quality == "Mixed":
            score += 10

        # Liquidity behavior
        if liquidity_behavior == "Institutional":
            score += 10

        return np.clip(score, 0, 100)

    def _classify_participant(
        self,
        inst_confidence: float,
        retail_confidence: float
    ) -> ParticipantType:
        """Classify dominant participant"""
        if inst_confidence >= 65:
            return ParticipantType.INSTITUTIONAL
        elif retail_confidence >= 65:
            return ParticipantType.RETAIL
        elif inst_confidence >= 40:
            return ParticipantType.MIXED
        else:
            return ParticipantType.UNKNOWN

    def _classify_entry_type(
        self,
        participant: ParticipantType,
        volume_sig: str,
        price_quality: str,
        df: pd.DataFrame
    ) -> EntryType:
        """Classify type of entry"""
        if participant == ParticipantType.INSTITUTIONAL:
            # Check if accumulation or distribution
            price_trend = df['close'].tail(10).diff().mean()
            if price_trend > 0:
                return EntryType.INST_ACCUMULATION
            else:
                return EntryType.INST_DISTRIBUTION

        elif participant == ParticipantType.RETAIL:
            # Check if FOMO or panic
            price_momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100 if len(df) >= 5 else 0

            if abs(price_momentum) > 2:
                if price_momentum > 0:
                    return EntryType.RETAIL_FOMO
                else:
                    return EntryType.RETAIL_PANIC
            else:
                return EntryType.TRAP_SETUP

        return EntryType.NEUTRAL

    def _describe_volume_profile(self, df: pd.DataFrame) -> str:
        """Describe volume profile"""
        if 'volume' not in df.columns or len(df) < 10:
            return "Unknown"

        recent_volume = df['volume'].tail(10).mean()
        older_volume = df['volume'].tail(20).head(10).mean() if len(df) >= 20 else recent_volume

        if recent_volume > older_volume * 1.5:
            return "Increasing (Strong)"
        elif recent_volume < older_volume * 0.7:
            return "Decreasing (Weak)"
        else:
            return "Stable"

    def _generate_recommendation(
        self,
        participant: ParticipantType,
        entry_type: EntryType,
        inst_confidence: float
    ) -> str:
        """Generate trading recommendation"""
        if participant == ParticipantType.INSTITUTIONAL:
            if entry_type == EntryType.INST_ACCUMULATION:
                return "✅ FOLLOW Smart Money - Institutional accumulation detected"
            else:
                return "⚠️ CAUTION - Institutional distribution, consider opposite side"

        elif participant == ParticipantType.RETAIL:
            if entry_type == EntryType.RETAIL_FOMO:
                return "🚫 FADE the Move - Retail FOMO, likely to reverse"
            elif entry_type == EntryType.RETAIL_PANIC:
                return "🚫 FADE the Move - Retail panic selling"
            else:
                return "⚠️ TRAP SETUP - Retail likely to be trapped"

        elif participant == ParticipantType.MIXED:
            if inst_confidence > 40:
                return "⚖️ NEUTRAL with Institutional lean"
            else:
                return "⚖️ NEUTRAL - Wait for clearer signal"

        return "⏳ WAIT - Insufficient participant clarity"

    def _default_result(self) -> MarketParticipantResult:
        """Default result for insufficient data"""
        return MarketParticipantResult(
            dominant_participant=ParticipantType.UNKNOWN,
            entry_type=EntryType.NEUTRAL,
            institutional_confidence=50.0,
            retail_confidence=50.0,
            smart_money_detected=False,
            dumb_money_detected=False,
            orderflow_signature="Unknown",
            volume_profile="Unknown",
            oi_classification="Unknown",
            price_action_quality="Unknown",
            recommendation="Insufficient data for participant detection",
            signals=[]
        )


def format_participant_report(result: MarketParticipantResult) -> str:
    """Format participant detection as readable report"""
    report = f"""
╔══════════════════════════════════════════════════════════╗
║      INSTITUTIONAL vs RETAIL DETECTION                   ║
╚══════════════════════════════════════════════════════════╝

👔 DOMINANT PARTICIPANT: {result.dominant_participant.value}
🎯 ENTRY TYPE: {result.entry_type.value}

📊 CONFIDENCE SCORES:
  • Institutional: {result.institutional_confidence:.1f}%
  • Retail: {result.retail_confidence:.1f}%

⚠️  MONEY CLASSIFICATION:
  • Smart Money: {'DETECTED 🏦' if result.smart_money_detected else 'Not Detected'}
  • Dumb Money: {'DETECTED 🎯' if result.dumb_money_detected else 'Not Detected'}

─────────────────────────────────────────────────────────
ANALYSIS BREAKDOWN:
  • Orderflow Signature: {result.orderflow_signature}
  • Volume Profile: {result.volume_profile}
  • OI Classification: {result.oi_classification}
  • Price Action Quality: {result.price_action_quality}

─────────────────────────────────────────────────────────
💡 RECOMMENDATION:
{result.recommendation}

─────────────────────────────────────────────────────────
📌 DETECTION SIGNALS:
"""
    for signal in result.signals:
        report += f"  • {signal}\n"

    return report
