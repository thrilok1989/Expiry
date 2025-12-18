"""
Cumulative Volume Delta (CVD) & Delta Imbalance Module
Professional orderflow analysis using delta metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CVDResult:
    """CVD Analysis Result"""
    cvd: float
    cvd_trend: str  # "Bullish", "Bearish", "Neutral"
    delta_imbalance: float  # -100 to +100
    delta_divergence_detected: bool
    delta_absorption_detected: bool
    delta_spike_detected: bool
    institutional_sweep: bool
    orderflow_strength: float  # 0-100
    bias: str  # "Bullish", "Bearish", "Neutral"
    confidence: float  # 0-1
    signals: List[str]


class CVDAnalyzer:
    """
    Cumulative Volume Delta (CVD) Analyzer

    Professional orderflow analysis using:
    - CVD tracking (cumulative buying vs selling pressure)
    - Delta divergence (price vs CVD divergence)
    - Delta absorption (large volume, small price movement)
    - Delta spikes (institutional sweeps)
    """

    def __init__(self):
        """Initialize CVD Analyzer"""
        self.spike_threshold = 3  # Standard deviations for spike detection
        self.absorption_ratio = 2.5  # Volume/Price movement ratio
        self.divergence_lookback = 20  # Bars for divergence detection

    def analyze_cvd(
        self,
        df: pd.DataFrame,
        volume_profile: Optional[Dict] = None
    ) -> CVDResult:
        """
        Complete CVD and delta imbalance analysis

        Args:
            df: OHLCV dataframe with volume data
            volume_profile: Optional volume profile data

        Returns:
            CVDResult with complete analysis
        """
        signals = []

        # Calculate volume delta and CVD
        df = self._calculate_volume_delta(df)
        df = self._calculate_cvd(df)

        if len(df) < 10:
            return self._default_result()

        current_cvd = df['cvd'].iloc[-1]
        current_delta = df['volume_delta'].iloc[-1]

        # 1. CVD Trend Analysis
        cvd_trend = self._analyze_cvd_trend(df)
        signals.append(f"CVD Trend: {cvd_trend}")

        # 2. Delta Imbalance
        delta_imbalance = self._calculate_delta_imbalance(df)
        signals.append(f"Delta Imbalance: {delta_imbalance:+.1f}%")

        # 3. Detect Delta Divergence
        delta_divergence = self._detect_delta_divergence(df)
        if delta_divergence:
            signals.append("⚠️ DELTA DIVERGENCE - Price/CVD mismatch")

        # 4. Detect Delta Absorption
        delta_absorption = self._detect_delta_absorption(df)
        if delta_absorption:
            signals.append("🛡️ DELTA ABSORPTION - Large orders absorbed")

        # 5. Detect Delta Spikes (Institutional Sweeps)
        delta_spike, institutional_sweep = self._detect_delta_spike(df)
        if delta_spike:
            signals.append("⚡ DELTA SPIKE - Institutional activity detected")

        # 6. Calculate Orderflow Strength
        orderflow_strength = self._calculate_orderflow_strength(
            df, delta_imbalance, delta_divergence, delta_absorption
        )

        # 7. Determine Bias
        bias = self._determine_bias(cvd_trend, delta_imbalance, delta_divergence)

        # 8. Calculate Confidence
        confidence = self._calculate_confidence(
            df, delta_imbalance, orderflow_strength
        )

        return CVDResult(
            cvd=current_cvd,
            cvd_trend=cvd_trend,
            delta_imbalance=delta_imbalance,
            delta_divergence_detected=delta_divergence,
            delta_absorption_detected=delta_absorption,
            delta_spike_detected=delta_spike,
            institutional_sweep=institutional_sweep,
            orderflow_strength=orderflow_strength,
            bias=bias,
            confidence=confidence,
            signals=signals
        )

    def _calculate_volume_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume delta (buying volume - selling volume)

        Heuristic: Compare close to open
        - If close > open: Buying volume = volume
        - If close < open: Selling volume = volume
        """
        df = df.copy()

        if 'volume' not in df.columns:
            df['volume_delta'] = 0
            return df

        # Calculate up/down volume
        df['up_volume'] = np.where(df['close'] >= df['open'], df['volume'], 0)
        df['down_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)

        # Better heuristic using range
        df['range'] = df['high'] - df['low']
        df['close_position'] = np.where(
            df['range'] > 0,
            (df['close'] - df['low']) / df['range'],
            0.5
        )

        # Distribute volume based on close position
        df['buying_volume'] = df['volume'] * df['close_position']
        df['selling_volume'] = df['volume'] * (1 - df['close_position'])

        # Volume delta (selling volume - buying volume)
        df['volume_delta'] = df['selling_volume'] - df['buying_volume']

        return df

    def _calculate_cvd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Cumulative Volume Delta"""
        df = df.copy()

        if 'volume_delta' not in df.columns:
            df['cvd'] = 0
            return df

        # Cumulative sum of volume delta
        df['cvd'] = df['volume_delta'].cumsum()

        return df

    def _analyze_cvd_trend(self, df: pd.DataFrame) -> str:
        """Analyze CVD trend direction"""
        if 'cvd' not in df.columns or len(df) < 10:
            return "Neutral"

        cvd_recent = df['cvd'].tail(5).mean()
        cvd_older = df['cvd'].tail(20).head(5).mean() if len(df) >= 20 else cvd_recent

        if cvd_recent > cvd_older * 1.02:
            return "Bullish"
        elif cvd_recent < cvd_older * 0.98:
            return "Bearish"
        else:
            return "Neutral"

    def _calculate_delta_imbalance(self, df: pd.DataFrame) -> float:
        """
        Calculate current delta imbalance (-100 to +100)

        Positive = More buying pressure
        Negative = More selling pressure
        """
        if 'volume_delta' not in df.columns or len(df) < 5:
            return 0.0

        recent_delta = df['volume_delta'].tail(5).sum()
        recent_volume = df['volume'].tail(5).sum()

        if recent_volume == 0:
            return 0.0

        imbalance = (recent_delta / recent_volume) * 100
        return np.clip(imbalance, -100, 100)

    def _detect_delta_divergence(self, df: pd.DataFrame) -> bool:
        """
        Detect delta divergence

        Bullish divergence: Price making lower lows, CVD making higher lows
        Bearish divergence: Price making higher highs, CVD making lower highs
        """
        if 'cvd' not in df.columns or len(df) < self.divergence_lookback:
            return False

        recent_df = df.tail(self.divergence_lookback)

        # Price trend
        price_start = recent_df['close'].iloc[0]
        price_end = recent_df['close'].iloc[-1]
        price_change = price_end - price_start

        # CVD trend
        cvd_start = recent_df['cvd'].iloc[0]
        cvd_end = recent_df['cvd'].iloc[-1]
        cvd_change = cvd_end - cvd_start

        # Divergence detection
        # Price up, CVD down = Bearish divergence
        if price_change > 0 and cvd_change < 0:
            return True

        # Price down, CVD up = Bullish divergence
        if price_change < 0 and cvd_change > 0:
            return True

        return False

    def _detect_delta_absorption(self, df: pd.DataFrame) -> bool:
        """
        Detect delta absorption

        Absorption = Large volume but small price movement
        Indicates strong support/resistance
        """
        if len(df) < 5:
            return False

        recent = df.tail(5)

        # Calculate average volume and price movement
        avg_volume = recent['volume'].mean()
        recent_range = (recent['high'] - recent['low']).mean()
        recent_close_change = abs(recent['close'].iloc[-1] - recent['close'].iloc[0])

        # Check if volume is high but price movement is small
        current_volume = recent['volume'].iloc[-1]

        if current_volume > avg_volume * 1.5:  # High volume
            if recent_close_change < recent_range * 0.3:  # Small price movement
                return True

        return False

    def _detect_delta_spike(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Detect delta spikes (institutional sweeps)

        Delta spike = Volume delta significantly higher than normal
        """
        if 'volume_delta' not in df.columns or len(df) < 20:
            return False, False

        # Calculate rolling mean and std
        delta_series = df['volume_delta'].tail(20)
        delta_mean = delta_series.mean()
        delta_std = delta_series.std()

        if delta_std == 0:
            return False, False

        current_delta = df['volume_delta'].iloc[-1]

        # Spike detection (3 standard deviations)
        z_score = abs((current_delta - delta_mean) / delta_std)

        spike_detected = z_score > self.spike_threshold

        # Institutional sweep = large positive delta spike
        institutional_sweep = spike_detected and current_delta > delta_mean + (3 * delta_std)

        return spike_detected, institutional_sweep

    def _calculate_orderflow_strength(
        self,
        df: pd.DataFrame,
        delta_imbalance: float,
        delta_divergence: bool,
        delta_absorption: bool
    ) -> float:
        """Calculate overall orderflow strength (0-100)"""
        strength = 0.0

        # Delta imbalance contribution (0-40)
        strength += abs(delta_imbalance) * 0.4

        # CVD trend strength (0-30)
        if 'cvd' not in df.columns or len(df) < 10:
            cvd_strength = 0
        else:
            cvd_recent = df['cvd'].tail(5).values
            cvd_slope = np.polyfit(range(5), cvd_recent, 1)[0]
            cvd_strength = min(abs(cvd_slope) / 1000, 30)

        strength += cvd_strength

        # Divergence adds strength (20)
        if delta_divergence:
            strength += 20

        # Absorption adds strength (10)
        if delta_absorption:
            strength += 10

        return np.clip(strength, 0, 100)

    def _determine_bias(
        self,
        cvd_trend: str,
        delta_imbalance: float,
        delta_divergence: bool
    ) -> str:
        """Determine overall bias from CVD analysis"""
        # If divergence, reverse the bias
        if delta_divergence:
            if cvd_trend == "Bullish":
                return "Bearish"  # Bearish divergence
            elif cvd_trend == "Bearish":
                return "Bullish"  # Bullish divergence

        # Normal bias
        if cvd_trend == "Bullish" and delta_imbalance > 10:
            return "Bullish"
        elif cvd_trend == "Bearish" and delta_imbalance < -10:
            return "Bearish"
        else:
            return "Neutral"

    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        delta_imbalance: float,
        orderflow_strength: float
    ) -> float:
        """Calculate confidence in CVD analysis (0-1)"""
        # Volume consistency
        if len(df) < 10:
            volume_consistency = 0.5
        else:
            recent_volume = df['volume'].tail(10)
            volume_cv = recent_volume.std() / recent_volume.mean() if recent_volume.mean() > 0 else 1
            volume_consistency = max(0.3, 1 - volume_cv)

        # Delta imbalance strength
        imbalance_confidence = abs(delta_imbalance) / 100

        # Orderflow strength
        orderflow_confidence = orderflow_strength / 100

        # Combined confidence
        confidence = (
            volume_consistency * 0.4 +
            imbalance_confidence * 0.3 +
            orderflow_confidence * 0.3
        )

        return np.clip(confidence, 0.3, 1.0)

    def _default_result(self) -> CVDResult:
        """Return default result for insufficient data"""
        return CVDResult(
            cvd=0.0,
            cvd_trend="Neutral",
            delta_imbalance=0.0,
            delta_divergence_detected=False,
            delta_absorption_detected=False,
            delta_spike_detected=False,
            institutional_sweep=False,
            orderflow_strength=0.0,
            bias="Neutral",
            confidence=0.3,
            signals=["Insufficient data for CVD analysis"]
        )


def format_cvd_report(result: CVDResult) -> str:
    """Format CVD analysis as readable report"""
    report = f"""
╔══════════════════════════════════════════════════════════╗
║      CUMULATIVE VOLUME DELTA (CVD) ANALYSIS              ║
╚══════════════════════════════════════════════════════════╝

📊 CVD: {result.cvd:,.0f}
📈 CVD TREND: {result.cvd_trend}
⚖️  DELTA IMBALANCE: {result.delta_imbalance:+.1f}%

🎯 BIAS: {result.bias}
💪 ORDERFLOW STRENGTH: {result.orderflow_strength:.1f}/100
✅ CONFIDENCE: {result.confidence*100:.1f}%

─────────────────────────────────────────────────────────
⚠️  SPECIAL DETECTIONS:
  • Delta Divergence: {'YES ⚠️' if result.delta_divergence_detected else 'No'}
  • Delta Absorption: {'YES 🛡️' if result.delta_absorption_detected else 'No'}
  • Delta Spike: {'YES ⚡' if result.delta_spike_detected else 'No'}
  • Institutional Sweep: {'YES 🏦' if result.institutional_sweep else 'No'}

─────────────────────────────────────────────────────────
📌 ORDERFLOW SIGNALS:
"""
    for signal in result.signals:
        report += f"  • {signal}\n"

    return report


def get_cvd_bias(result: CVDResult) -> str:
    """Get simple bias from CVD analysis"""
    return result.bias
