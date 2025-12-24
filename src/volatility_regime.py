"""
Volatility Regime Detection Module
Detects market volatility regimes for better strategy selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Volatility regime classifications"""
    LOW_VOLATILITY = "Low Volatility"
    NORMAL_VOLATILITY = "Normal Volatility"
    HIGH_VOLATILITY = "High Volatility"
    EXTREME_VOLATILITY = "Extreme Volatility"
    REGIME_CHANGE = "Regime Change"


class VolRegimeTrend(Enum):
    """Volatility regime trend direction"""
    COMPRESSING = "Compressing"
    STABLE = "Stable"
    EXPANDING = "Expanding"


@dataclass
class VolatilityRegimeResult:
    """Complete volatility regime analysis result"""
    regime: VolatilityRegime
    trend: VolRegimeTrend
    vix_level: float
    vix_percentile: float
    atr_regime: str
    atr_percentile: float
    iv_rv_ratio: float
    gamma_flip_detected: bool
    is_expiry_week: bool
    regime_strength: float  # 0-100
    regime_duration_bars: int
    compression_score: float  # -100 (compression) to +100 (expansion)
    recommended_strategy: str
    confidence: float  # 0-1
    signals: List[str]


class VolatilityRegimeDetector:
    """
    Advanced Volatility Regime Detection System

    Detects and classifies market volatility regimes using:
    - India VIX analysis
    - ATR-based regime detection
    - Implied vs Realized Volatility
    - Gamma flip detection
    - Expiry week behavior
    """

    def __init__(self, lookback_period: int = 252):
        """
        Initialize volatility regime detector

        Args:
            lookback_period: Historical lookback for percentile calculations (default 252 = 1 year)
        """
        self.lookback_period = lookback_period

        # VIX regime thresholds
        self.vix_thresholds = {
            'extreme': 30,
            'high': 20,
            'normal': 15,
            'low': 12
        }

        # ATR percentile thresholds
        self.atr_thresholds = {
            'extreme': 90,
            'high': 70,
            'normal': 50,
            'low': 30
        }

    def analyze_regime(
        self,
        df: pd.DataFrame,
        vix_current: float,
        vix_history: pd.Series,
        option_chain: Optional[Dict] = None,
        days_to_expiry: int = 0
    ) -> VolatilityRegimeResult:
        """
        Complete volatility regime analysis

        Args:
            df: OHLCV dataframe with ATR
            vix_current: Current India VIX value
            vix_history: Historical VIX series
            option_chain: Option chain data for IV calculation
            days_to_expiry: Days until weekly/monthly expiry

        Returns:
            VolatilityRegimeResult with complete analysis
        """
        signals = []

        # 1. VIX Analysis
        vix_regime, vix_percentile = self._analyze_vix(vix_current, vix_history)
        signals.append(f"VIX: {vix_current:.2f} ({vix_percentile:.1f}%ile)")

        # 2. ATR Analysis
        atr_regime, atr_percentile = self._analyze_atr(df)
        signals.append(f"ATR Regime: {atr_regime} ({atr_percentile:.1f}%ile)")

        # 3. IV vs RV Analysis
        iv_rv_ratio = self._calculate_iv_rv_ratio(df, option_chain)
        signals.append(f"IV/RV: {iv_rv_ratio:.2f}")

        # 4. Gamma Flip Detection
        gamma_flip_detected = self._detect_gamma_flip(option_chain)
        if gamma_flip_detected:
            signals.append("⚠️ GAMMA FLIP DETECTED")

        # 5. Expiry Week Analysis
        is_expiry_week = days_to_expiry <= 3
        if is_expiry_week:
            signals.append(f"⚡ Expiry Week (T-{days_to_expiry})")

        # 6. Regime Trend (Compression/Expansion)
        trend, compression_score = self._detect_regime_trend(
            df, vix_history, atr_percentile
        )

        # 7. Regime Duration
        regime_duration = self._calculate_regime_duration(df, atr_regime)

        # 8. Determine Overall Regime
        regime = self._classify_regime(
            vix_percentile,
            atr_percentile,
            iv_rv_ratio,
            gamma_flip_detected,
            is_expiry_week
        )

        # 9. Calculate Regime Strength
        regime_strength = self._calculate_regime_strength(
            vix_percentile,
            atr_percentile,
            iv_rv_ratio,
            regime_duration
        )

        # 10. Strategy Recommendation
        recommended_strategy = self._recommend_strategy(
            regime, trend, is_expiry_week, gamma_flip_detected
        )

        # 11. Confidence Score
        confidence = self._calculate_confidence(
            vix_percentile, atr_percentile, regime_duration
        )

        return VolatilityRegimeResult(
            regime=regime,
            trend=trend,
            vix_level=vix_current,
            vix_percentile=vix_percentile,
            atr_regime=atr_regime,
            atr_percentile=atr_percentile,
            iv_rv_ratio=iv_rv_ratio,
            gamma_flip_detected=gamma_flip_detected,
            is_expiry_week=is_expiry_week,
            regime_strength=regime_strength,
            regime_duration_bars=regime_duration,
            compression_score=compression_score,
            recommended_strategy=recommended_strategy,
            confidence=confidence,
            signals=signals
        )

    def _analyze_vix(
        self,
        vix_current: float,
        vix_history: pd.Series
    ) -> Tuple[str, float]:
        """Analyze VIX level and calculate percentile"""
        if len(vix_history) < 20:
            return "Unknown", 50.0

        # Calculate percentile
        percentile = (vix_history <= vix_current).sum() / len(vix_history) * 100

        # Classify regime
        if vix_current >= self.vix_thresholds['extreme']:
            regime = "Extreme"
        elif vix_current >= self.vix_thresholds['high']:
            regime = "High"
        elif vix_current >= self.vix_thresholds['normal']:
            regime = "Normal"
        elif vix_current >= self.vix_thresholds['low']:
            regime = "Moderate"
        else:
            regime = "Low"

        return regime, percentile

    def _analyze_atr(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Analyze ATR regime and calculate percentile"""
        if 'atr' not in df.columns or len(df) < 20:
            return "Unknown", 50.0

        current_atr = df['atr'].iloc[-1]
        atr_history = df['atr'].dropna()

        if len(atr_history) == 0:
            return "Unknown", 50.0

        # Calculate percentile
        percentile = (atr_history <= current_atr).sum() / len(atr_history) * 100

        # Classify ATR regime
        if percentile >= self.atr_thresholds['extreme']:
            regime = "Extreme"
        elif percentile >= self.atr_thresholds['high']:
            regime = "High"
        elif percentile >= self.atr_thresholds['normal']:
            regime = "Normal"
        else:
            regime = "Low"

        return regime, percentile

    def _calculate_iv_rv_ratio(
        self,
        df: pd.DataFrame,
        option_chain: Optional[Dict]
    ) -> float:
        """
        Calculate Implied Volatility vs Realized Volatility ratio

        IV > RV: Options are expensive (sell premium)
        IV < RV: Options are cheap (buy premium)
        """
        # Calculate Realized Volatility (20-day)
        if 'close' not in df.columns or len(df) < 20:
            return 1.0

        returns = df['close'].pct_change().dropna()
        realized_vol = returns.tail(20).std() * np.sqrt(252) * 100  # Annualized

        # Get Implied Volatility from option chain
        implied_vol = None
        if option_chain:
            # Try to extract ATM IV
            try:
                ce_data = option_chain.get('CE', {})
                pe_data = option_chain.get('PE', {})

                # Get ATM strikes IV
                atm_ce_iv = ce_data.get('IV', [])
                atm_pe_iv = pe_data.get('IV', [])

                if atm_ce_iv and atm_pe_iv:
                    # Average of ATM CE and PE IV
                    implied_vol = (np.mean(atm_ce_iv[:3]) + np.mean(atm_pe_iv[:3])) / 2
            except Exception as e:
                logger.debug(f"Could not extract IV from option chain: {e}")

        if implied_vol is None or realized_vol == 0:
            return 1.0

        return implied_vol / realized_vol

    def _detect_gamma_flip(self, option_chain: Optional[Dict]) -> bool:
        """
        Detect gamma flip conditions

        Gamma flip occurs when dealers switch from negative to positive gamma,
        causing explosive moves or sudden volatility changes
        """
        if not option_chain:
            return False

        try:
            ce_data = option_chain.get('CE', {})
            pe_data = option_chain.get('PE', {})

            # Get OI at ATM strikes
            ce_oi = ce_data.get('openInterest', [])
            pe_oi = pe_data.get('openInterest', [])

            if not ce_oi or not pe_oi:
                return False

            # Simple gamma flip heuristic:
            # Large imbalance in ATM OI + high total OI
            atm_ce_oi = sum(ce_oi[:3])
            atm_pe_oi = sum(pe_oi[:3])

            total_oi = atm_ce_oi + atm_pe_oi
            imbalance_ratio = abs(atm_ce_oi - atm_pe_oi) / total_oi if total_oi > 0 else 0

            # Gamma flip detected if large imbalance
            return imbalance_ratio > 0.6

        except Exception as e:
            logger.debug(f"Gamma flip detection error: {e}")
            return False

    def _detect_regime_trend(
        self,
        df: pd.DataFrame,
        vix_history: pd.Series,
        atr_percentile: float
    ) -> Tuple[VolRegimeTrend, float]:
        """
        Detect if volatility is compressing or expanding

        Returns:
            (trend, compression_score)
            compression_score: -100 (max compression) to +100 (max expansion)
        """
        if len(df) < 10 or len(vix_history) < 10:
            return VolRegimeTrend.STABLE, 0.0

        # ATR trend
        atr_recent = df['atr'].tail(5).mean() if 'atr' in df.columns else None
        atr_older = df['atr'].tail(20).head(5).mean() if 'atr' in df.columns and len(df) >= 20 else None

        # VIX trend
        vix_recent = vix_history.tail(5).mean()
        vix_older = vix_history.tail(20).head(5).mean() if len(vix_history) >= 20 else vix_recent

        # Calculate compression score
        compression_score = 0.0

        if atr_recent and atr_older and atr_older > 0:
            atr_change = ((atr_recent - atr_older) / atr_older) * 100
            compression_score += atr_change * 0.6  # 60% weight

        if vix_older > 0:
            vix_change = ((vix_recent - vix_older) / vix_older) * 100
            compression_score += vix_change * 0.4  # 40% weight

        # Classify trend
        if compression_score < -5:
            trend = VolRegimeTrend.COMPRESSING
        elif compression_score > 5:
            trend = VolRegimeTrend.EXPANDING
        else:
            trend = VolRegimeTrend.STABLE

        # Normalize compression score to -100 to +100
        compression_score = np.clip(compression_score * 2, -100, 100)

        return trend, compression_score

    def _calculate_regime_duration(self, df: pd.DataFrame, current_regime: str) -> int:
        """Calculate how many bars the current regime has persisted"""
        if 'atr' not in df.columns or len(df) < 20:
            return 0

        atr_history = df['atr'].tail(100)

        # Calculate regime for each bar
        duration = 0
        for atr in reversed(atr_history.values):
            percentile = (atr_history <= atr).sum() / len(atr_history) * 100

            if percentile >= 90:
                regime = "Extreme"
            elif percentile >= 70:
                regime = "High"
            elif percentile >= 50:
                regime = "Normal"
            else:
                regime = "Low"

            if regime == current_regime:
                duration += 1
            else:
                break

        return duration

    def _classify_regime(
        self,
        vix_percentile: float,
        atr_percentile: float,
        iv_rv_ratio: float,
        gamma_flip: bool,
        is_expiry_week: bool
    ) -> VolatilityRegime:
        """Classify overall volatility regime"""
        # Weight different factors
        avg_percentile = (vix_percentile * 0.5 + atr_percentile * 0.5)

        # Extreme conditions
        if gamma_flip or (is_expiry_week and avg_percentile > 80):
            return VolatilityRegime.EXTREME_VOLATILITY

        # Regime change detection
        if abs(iv_rv_ratio - 1.0) > 0.3:  # IV/RV divergence
            return VolatilityRegime.REGIME_CHANGE

        # Normal classification
        if avg_percentile >= 80:
            return VolatilityRegime.EXTREME_VOLATILITY
        elif avg_percentile >= 60:
            return VolatilityRegime.HIGH_VOLATILITY
        elif avg_percentile >= 35:
            return VolatilityRegime.NORMAL_VOLATILITY
        else:
            return VolatilityRegime.LOW_VOLATILITY

    def _calculate_regime_strength(
        self,
        vix_percentile: float,
        atr_percentile: float,
        iv_rv_ratio: float,
        regime_duration: int
    ) -> float:
        """
        Calculate regime strength (0-100)

        Higher strength = more confident in regime classification
        """
        # Alignment of VIX and ATR
        alignment = 100 - abs(vix_percentile - atr_percentile)

        # Duration factor (longer = stronger)
        duration_factor = min(regime_duration * 2, 50)

        # IV/RV confirmation (closer to extremes = stronger)
        iv_factor = abs(iv_rv_ratio - 1.0) * 30

        strength = (alignment * 0.5 + duration_factor * 0.3 + iv_factor * 0.2)
        return np.clip(strength, 0, 100)

    def _recommend_strategy(
        self,
        regime: VolatilityRegime,
        trend: VolRegimeTrend,
        is_expiry_week: bool,
        gamma_flip: bool
    ) -> str:
        """Recommend trading strategy based on regime"""
        if gamma_flip:
            return "⚠️ AVOID NEW POSITIONS - Gamma Flip Risk"

        if is_expiry_week:
            if regime == VolatilityRegime.LOW_VOLATILITY:
                return "Expiry Week: Sell ATM Straddles/Strangles"
            else:
                return "Expiry Week: Buy protection, avoid selling premium"

        if regime == VolatilityRegime.LOW_VOLATILITY:
            if trend == VolRegimeTrend.COMPRESSING:
                return "🎯 BREAKOUT Trades, Trend Following, Sell Premium"
            else:
                return "Range Trading, Mean Reversion, Sell Premium"

        elif regime == VolatilityRegime.NORMAL_VOLATILITY:
            return "Balanced approach, All strategies viable"

        elif regime == VolatilityRegime.HIGH_VOLATILITY:
            if trend == VolRegimeTrend.EXPANDING:
                return "🔥 MOMENTUM Trades, Buy volatility, Trend continuation"
            else:
                return "Buy Dips, Fade extremes, Reversal trades"

        elif regime == VolatilityRegime.EXTREME_VOLATILITY:
            return "⚠️ DEFENSIVE: Reduce size, Buy protection, Wait for calm"

        else:  # REGIME_CHANGE
            return "🔄 REGIME CHANGE: Wait for confirmation, reduce exposure"

    def _calculate_confidence(
        self,
        vix_percentile: float,
        atr_percentile: float,
        regime_duration: int
    ) -> float:
        """Calculate confidence in regime classification (0-1)"""
        # Alignment confidence
        alignment = 1.0 - (abs(vix_percentile - atr_percentile) / 100)

        # Duration confidence (longer = more confident)
        duration_confidence = min(regime_duration / 20, 1.0)

        # Combined confidence
        confidence = (alignment * 0.6 + duration_confidence * 0.4)
        return np.clip(confidence, 0.3, 1.0)  # Min 30% confidence


# Utility functions for integration
def format_regime_report(result: VolatilityRegimeResult) -> str:
    """Format volatility regime analysis as readable report"""
    report = f"""
╔══════════════════════════════════════════════════════════╗
║          VOLATILITY REGIME ANALYSIS                      ║
╚══════════════════════════════════════════════════════════╝

📊 CURRENT REGIME: {result.regime.value}
📈 TREND: {result.trend.value}
💪 STRENGTH: {result.regime_strength:.1f}/100
✅ CONFIDENCE: {result.confidence*100:.1f}%

─────────────────────────────────────────────────────────
VIX ANALYSIS:
  • Current: {result.vix_level:.2f}
  • Percentile: {result.vix_percentile:.1f}%ile

ATR ANALYSIS:
  • Regime: {result.atr_regime}
  • Percentile: {result.atr_percentile:.1f}%ile

VOLATILITY DYNAMICS:
  • IV/RV Ratio: {result.iv_rv_ratio:.2f}
  • Compression Score: {result.compression_score:+.1f}
  • Regime Duration: {result.regime_duration_bars} bars

⚠️  SPECIAL CONDITIONS:
  • Gamma Flip: {'YES ⚡' if result.gamma_flip_detected else 'No'}
  • Expiry Week: {'YES ⏰' if result.is_expiry_week else 'No'}

─────────────────────────────────────────────────────────
🎯 RECOMMENDED STRATEGY:
{result.recommended_strategy}

─────────────────────────────────────────────────────────
📌 KEY SIGNALS:
"""
    for signal in result.signals:
        report += f"  • {signal}\n"

    return report


def get_regime_bias(result: VolatilityRegimeResult) -> str:
    """Get simple bias: Bullish/Bearish/Neutral based on regime"""
    if result.regime == VolatilityRegime.LOW_VOLATILITY:
        if result.trend == VolRegimeTrend.COMPRESSING:
            return "Bullish"  # Breakout imminent
        else:
            return "Neutral"  # Range-bound

    elif result.regime == VolatilityRegime.HIGH_VOLATILITY:
        if result.trend == VolRegimeTrend.EXPANDING:
            return "Bearish"  # Fear increasing
        else:
            return "Neutral"  # Reverting

    elif result.regime == VolatilityRegime.EXTREME_VOLATILITY:
        return "Bearish"  # High risk

    else:
        return "Neutral"
