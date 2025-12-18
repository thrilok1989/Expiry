"""
ML-Powered Market Regime Detection & Summary
Uses XGBoost/LightGBM for intelligent regime classification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MLMarketRegimeResult:
    """ML Market Regime Detection Result"""
    regime: str  # "Trending Up", "Trending Down", "Range Bound", "Volatile Breakout", "Consolidation"
    confidence: float  # 0-100
    regime_probabilities: Dict[str, float]  # Probability for each regime
    trend_strength: float  # 0-100
    volatility_state: str  # "Low", "Normal", "High", "Extreme"
    market_phase: str  # "Accumulation", "Markup", "Distribution", "Markdown"
    recommended_strategy: str
    optimal_timeframe: str  # "Scalp", "Intraday", "Swing", "Position"
    feature_importance: Dict[str, float]
    signals: List[str]


@dataclass
class MarketSummary:
    """Comprehensive Market Summary"""
    overall_bias: str  # "Bullish", "Bearish", "Neutral"
    bias_confidence: float  # 0-100
    regime: str
    volatility: str
    trend_quality: str  # "Strong", "Weak", "No Trend"
    momentum: str  # "Accelerating", "Decelerating", "Stable"
    support_level: float
    resistance_level: float
    key_target: float
    risk_level: str  # "Low", "Medium", "High", "Extreme"
    trade_signal: str  # "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"
    conviction_score: float  # 0-100
    market_health_score: float  # 0-100
    summary_text: str
    actionable_insights: List[str]


class MLMarketRegimeDetector:
    """
    ML-Powered Market Regime Detection

    Uses engineered features + rule-based classification (lightweight)
    Can be upgraded to XGBoost/LightGBM with training data
    """

    def __init__(self):
        """Initialize ML Market Regime Detector"""
        self.regime_classes = [
            "Trending Up",
            "Trending Down",
            "Range Bound",
            "Volatile Breakout",
            "Consolidation"
        ]

    def detect_regime(
        self,
        df: pd.DataFrame,
        cvd_result: Optional[any] = None,
        volatility_result: Optional[any] = None,
        oi_trap_result: Optional[any] = None
    ) -> MLMarketRegimeResult:
        """
        Detect market regime using ML-style feature engineering

        Args:
            df: OHLCV dataframe with indicators
            cvd_result: CVD analysis result
            volatility_result: Volatility regime result
            oi_trap_result: OI trap detection result

        Returns:
            MLMarketRegimeResult with regime classification
        """
        signals = []

        if len(df) < 50:
            return self._default_result()

        # Feature Engineering
        features = self._engineer_features(df)

        # Calculate regime probabilities using features
        regime_probs = self._calculate_regime_probabilities(features, df)

        # Determine primary regime
        regime = max(regime_probs, key=regime_probs.get)
        confidence = regime_probs[regime]

        # Incorporate external signals
        if cvd_result:
            if cvd_result.bias == "Bullish" and regime == "Trending Up":
                confidence = min(confidence + 10, 100)
                signals.append("✅ CVD confirms uptrend")
            elif cvd_result.bias == "Bearish" and regime == "Trending Down":
                confidence = min(confidence + 10, 100)
                signals.append("✅ CVD confirms downtrend")

        if volatility_result:
            features['volatility_regime'] = volatility_result.regime.value

        if oi_trap_result and oi_trap_result.trap_detected:
            signals.append(f"⚠️ {oi_trap_result.trap_type.value} detected")

        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(df, features)

        # Classify volatility state
        volatility_state = self._classify_volatility_state(df, volatility_result)

        # Determine market phase (Wyckoff)
        market_phase = self._determine_market_phase(df, features, regime)

        # Recommend strategy
        recommended_strategy = self._recommend_strategy(
            regime, trend_strength, volatility_state, market_phase
        )

        # Optimal timeframe
        optimal_timeframe = self._determine_optimal_timeframe(
            regime, volatility_state, trend_strength
        )

        # Feature importance (simulated)
        feature_importance = self._calculate_feature_importance(features, regime)

        # Generate signals
        signals.extend(self._generate_regime_signals(regime, confidence, features))

        return MLMarketRegimeResult(
            regime=regime,
            confidence=confidence,
            regime_probabilities=regime_probs,
            trend_strength=trend_strength,
            volatility_state=volatility_state,
            market_phase=market_phase,
            recommended_strategy=recommended_strategy,
            optimal_timeframe=optimal_timeframe,
            feature_importance=feature_importance,
            signals=signals
        )

    def _engineer_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Engineer ML features from price data"""
        features = {}

        # Price momentum features
        returns_5 = df['close'].pct_change(5).iloc[-1] * 100
        returns_20 = df['close'].pct_change(20).iloc[-1] * 100
        features['momentum_5'] = returns_5
        features['momentum_20'] = returns_20

        # Trend features
        recent = df.tail(20)
        x = np.arange(len(recent))
        if len(x) >= 2:
            slope = np.polyfit(x, recent['close'].values, 1)[0]
            features['trend_slope'] = slope
        else:
            features['trend_slope'] = 0

        # Volatility features
        if 'atr' in df.columns:
            atr_current = df['atr'].iloc[-1]
            atr_ma = df['atr'].tail(20).mean()
            features['atr_ratio'] = atr_current / atr_ma if atr_ma > 0 else 1
        else:
            features['atr_ratio'] = 1

        # Volume features
        if 'volume' in df.columns:
            vol_current = df['volume'].iloc[-1]
            vol_ma = df['volume'].tail(20).mean()
            features['volume_ratio'] = vol_current / vol_ma if vol_ma > 0 else 1
        else:
            features['volume_ratio'] = 1

        # Range features
        recent_range = (recent['high'] - recent['low']).mean()
        close_position = (recent['close'].iloc[-1] - recent['low'].min()) / (recent['high'].max() - recent['low'].min()) if (recent['high'].max() - recent['low'].min()) > 0 else 0.5
        features['range_position'] = close_position

        # RSI
        if 'rsi' in df.columns:
            features['rsi'] = df['rsi'].iloc[-1]
        else:
            # Calculate simple RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss if (loss != 0).all() else pd.Series([1]*len(gain))
            rsi = 100 - (100 / (1 + rs))
            features['rsi'] = rsi.iloc[-1] if len(rsi) > 0 else 50

        # ADX (trend strength)
        features['adx'] = self._calculate_adx(df)

        return features

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)"""
        if len(df) < period + 1:
            return 25  # Neutral

        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up = high - high.shift()
        down = low.shift() - low

        pos_dm = np.where((up > down) & (up > 0), up, 0)
        neg_dm = np.where((down > up) & (down > 0), down, 0)

        # Smoothed indicators
        atr = tr.rolling(period).mean()
        pos_di = 100 * pd.Series(pos_dm).rolling(period).mean() / atr
        neg_di = 100 * pd.Series(neg_dm).rolling(period).mean() / atr

        # ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean()

        return adx.iloc[-1] if len(adx) > 0 and not np.isnan(adx.iloc[-1]) else 25

    def _calculate_regime_probabilities(
        self,
        features: Dict[str, float],
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate probability of each regime using features"""
        probs = {regime: 0.0 for regime in self.regime_classes}

        # Trending Up indicators
        if features['momentum_20'] > 2 and features['trend_slope'] > 0:
            probs["Trending Up"] += 40
        if features['adx'] > 25 and features['momentum_5'] > 0:
            probs["Trending Up"] += 30
        if features['rsi'] > 55 and features['range_position'] > 0.6:
            probs["Trending Up"] += 20
        if features['volume_ratio'] > 1.2 and features['momentum_5'] > 0:
            probs["Trending Up"] += 10

        # Trending Down indicators
        if features['momentum_20'] < -2 and features['trend_slope'] < 0:
            probs["Trending Down"] += 40
        if features['adx'] > 25 and features['momentum_5'] < 0:
            probs["Trending Down"] += 30
        if features['rsi'] < 45 and features['range_position'] < 0.4:
            probs["Trending Down"] += 20
        if features['volume_ratio'] > 1.2 and features['momentum_5'] < 0:
            probs["Trending Down"] += 10

        # Range Bound indicators
        if features['adx'] < 20:
            probs["Range Bound"] += 40
        if abs(features['momentum_20']) < 1:
            probs["Range Bound"] += 30
        if 40 < features['rsi'] < 60:
            probs["Range Bound"] += 20
        if features['atr_ratio'] < 0.8:
            probs["Range Bound"] += 10

        # Volatile Breakout indicators
        if features['volume_ratio'] > 2.0:
            probs["Volatile Breakout"] += 40
        if features['atr_ratio'] > 1.5:
            probs["Volatile Breakout"] += 30
        if abs(features['momentum_5']) > 2:
            probs["Volatile Breakout"] += 20
        if features['adx'] > 30:
            probs["Volatile Breakout"] += 10

        # Consolidation indicators
        if features['atr_ratio'] < 0.7:
            probs["Consolidation"] += 40
        if features['volume_ratio'] < 0.8:
            probs["Consolidation"] += 30
        if abs(features['momentum_5']) < 0.5:
            probs["Consolidation"] += 20
        if features['adx'] < 15:
            probs["Consolidation"] += 10

        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {k: (v / total * 100) for k, v in probs.items()}

        return probs

    def _calculate_trend_strength(
        self,
        df: pd.DataFrame,
        features: Dict[str, float]
    ) -> float:
        """Calculate trend strength (0-100)"""
        strength = 0

        # ADX contribution
        adx = features.get('adx', 25)
        strength += min(adx, 50)

        # Momentum contribution
        momentum = abs(features.get('momentum_20', 0))
        strength += min(momentum * 5, 25)

        # Consistency contribution
        recent_returns = df['close'].pct_change().tail(10)
        consistency = (recent_returns > 0).sum() / len(recent_returns) * 25
        strength += consistency if features.get('momentum_5', 0) > 0 else (25 - consistency)

        return min(strength, 100)

    def _classify_volatility_state(
        self,
        df: pd.DataFrame,
        volatility_result: Optional[any]
    ) -> str:
        """Classify volatility state"""
        if volatility_result:
            regime = volatility_result.regime.value
            if "Extreme" in regime:
                return "Extreme"
            elif "High" in regime:
                return "High"
            elif "Low" in regime:
                return "Low"
            else:
                return "Normal"

        # Fallback: use ATR
        if 'atr' in df.columns and len(df) >= 20:
            atr_current = df['atr'].iloc[-1]
            atr_history = df['atr'].tail(50)
            percentile = (atr_history <= atr_current).sum() / len(atr_history) * 100

            if percentile > 90:
                return "Extreme"
            elif percentile > 70:
                return "High"
            elif percentile < 30:
                return "Low"

        return "Normal"

    def _determine_market_phase(
        self,
        df: pd.DataFrame,
        features: Dict[str, float],
        regime: str
    ) -> str:
        """Determine Wyckoff market phase"""
        momentum = features.get('momentum_20', 0)
        volume_ratio = features.get('volume_ratio', 1)
        range_position = features.get('range_position', 0.5)

        if regime == "Consolidation":
            if volume_ratio > 1.2:
                return "Accumulation" if range_position > 0.5 else "Distribution"
            return "Consolidation"

        if regime == "Trending Up":
            return "Markup"

        if regime == "Trending Down":
            return "Markdown"

        if regime == "Volatile Breakout":
            return "Markup" if momentum > 0 else "Markdown"

        return "Range Bound"

    def _recommend_strategy(
        self,
        regime: str,
        trend_strength: float,
        volatility_state: str,
        market_phase: str
    ) -> str:
        """Recommend trading strategy based on regime"""
        if regime == "Trending Up":
            if trend_strength > 70:
                return "🚀 Strong Trend Following - Buy dips, hold winners"
            else:
                return "📈 Trend Following - Enter on pullbacks"

        elif regime == "Trending Down":
            if trend_strength > 70:
                return "🔻 Short Trend - Sell rallies, hold shorts"
            else:
                return "📉 Bearish Bias - Fade pumps"

        elif regime == "Range Bound":
            return "↔️ Range Trading - Buy support, sell resistance"

        elif regime == "Volatile Breakout":
            if volatility_state == "Extreme":
                return "⚠️ WAIT - Too volatile, reduce exposure"
            else:
                return "⚡ Breakout Trading - Follow momentum with tight stops"

        elif regime == "Consolidation":
            if market_phase == "Accumulation":
                return "🎯 Position for breakout - Accumulate quality setups"
            else:
                return "⏳ WAIT - Consolidation, avoid low-quality trades"

        return "⏸️ NEUTRAL - Wait for clearer regime"

    def _determine_optimal_timeframe(
        self,
        regime: str,
        volatility_state: str,
        trend_strength: float
    ) -> str:
        """Determine optimal trading timeframe"""
        if regime in ["Trending Up", "Trending Down"] and trend_strength > 60:
            return "Swing (Hold multiple days)"

        if volatility_state == "Extreme":
            return "Scalp (Quick in/out)"

        if regime == "Volatile Breakout":
            return "Intraday (Same day exit)"

        if regime == "Range Bound":
            return "Intraday (Scalp swings)"

        return "Intraday (Standard)"

    def _calculate_feature_importance(
        self,
        features: Dict[str, float],
        regime: str
    ) -> Dict[str, float]:
        """Calculate feature importance (simulated)"""
        # This would come from trained model
        # For now, return heuristic importance
        importance = {
            'adx': 0.25,
            'momentum_20': 0.20,
            'trend_slope': 0.15,
            'volume_ratio': 0.15,
            'atr_ratio': 0.10,
            'rsi': 0.08,
            'range_position': 0.07
        }
        return importance

    def _generate_regime_signals(
        self,
        regime: str,
        confidence: float,
        features: Dict[str, float]
    ) -> List[str]:
        """Generate signals based on regime"""
        signals = []

        signals.append(f"📊 Regime: {regime} (Confidence: {confidence:.0f}%)")

        if features.get('adx', 0) > 30:
            signals.append(f"💪 Strong trend (ADX: {features['adx']:.1f})")
        elif features.get('adx', 0) < 20:
            signals.append(f"📊 Weak trend (ADX: {features['adx']:.1f})")

        if features.get('volume_ratio', 1) > 1.5:
            signals.append(f"📈 High volume confirmation")

        rsi = features.get('rsi', 50)
        if rsi > 70:
            signals.append(f"⚠️ Overbought (RSI: {rsi:.0f})")
        elif rsi < 30:
            signals.append(f"⚠️ Oversold (RSI: {rsi:.0f})")

        return signals

    def _default_result(self) -> MLMarketRegimeResult:
        """Default result for insufficient data"""
        return MLMarketRegimeResult(
            regime="Unknown",
            confidence=0,
            regime_probabilities={},
            trend_strength=0,
            volatility_state="Normal",
            market_phase="Unknown",
            recommended_strategy="Insufficient data",
            optimal_timeframe="Intraday",
            feature_importance={},
            signals=["Insufficient data for regime detection"]
        )


def generate_market_summary(
    ml_regime: MLMarketRegimeResult,
    cvd_result: Optional[any] = None,
    volatility_result: Optional[any] = None,
    oi_trap_result: Optional[any] = None,
    participant_result: Optional[any] = None,
    liquidity_result: Optional[any] = None,
    risk_result: Optional[any] = None,
    current_price: float = 0
) -> MarketSummary:
    """
    Generate comprehensive market summary combining all analyses

    This is the MASTER SUMMARY that combines everything
    """
    insights = []

    # Overall bias
    bias_score = 0
    bias_signals = []

    # ML Regime contribution
    if ml_regime.regime == "Trending Up":
        bias_score += 30
        bias_signals.append("ML: Trending Up")
    elif ml_regime.regime == "Trending Down":
        bias_score -= 30
        bias_signals.append("ML: Trending Down")

    # CVD contribution
    if cvd_result:
        if cvd_result.bias == "Bullish":
            bias_score += 20
            bias_signals.append("CVD: Bullish")
        elif cvd_result.bias == "Bearish":
            bias_score -= 20
            bias_signals.append("CVD: Bearish")

    # Institutional vs Retail
    if participant_result:
        if participant_result.smart_money_detected:
            if participant_result.entry_type.value == "Institutional Accumulation":
                bias_score += 25
                bias_signals.append("Smart Money Accumulating")
            elif participant_result.entry_type.value == "Institutional Distribution":
                bias_score -= 25
                bias_signals.append("Smart Money Distributing")

    # Overall bias classification
    if bias_score > 40:
        overall_bias = "Bullish"
        bias_confidence = min(bias_score, 100)
    elif bias_score < -40:
        overall_bias = "Bearish"
        bias_confidence = min(abs(bias_score), 100)
    else:
        overall_bias = "Neutral"
        bias_confidence = 100 - abs(bias_score)

    # Trend quality
    if ml_regime.trend_strength > 70:
        trend_quality = "Strong"
    elif ml_regime.trend_strength > 40:
        trend_quality = "Moderate"
    else:
        trend_quality = "Weak"

    # Momentum
    if ml_regime.regime == "Volatile Breakout":
        momentum = "Accelerating"
    elif ml_regime.regime == "Consolidation":
        momentum = "Decelerating"
    else:
        momentum = "Stable"

    # Support/Resistance
    support_level = liquidity_result.support_zones[0].price if liquidity_result and liquidity_result.support_zones else current_price * 0.98
    resistance_level = liquidity_result.resistance_zones[0].price if liquidity_result and liquidity_result.resistance_zones else current_price * 1.02
    key_target = liquidity_result.primary_target if liquidity_result else current_price

    # Risk level
    risk_level = risk_result.risk_level if risk_result else "Medium"

    # Trade signal
    if bias_score > 60 and ml_regime.confidence > 70:
        trade_signal = "Strong Buy"
    elif bias_score > 30:
        trade_signal = "Buy"
    elif bias_score < -60 and ml_regime.confidence > 70:
        trade_signal = "Strong Sell"
    elif bias_score < -30:
        trade_signal = "Sell"
    else:
        trade_signal = "Hold"

    # Conviction score
    conviction_score = ml_regime.confidence * 0.5 + bias_confidence * 0.5

    # Market health score
    health_score = 50  # Base
    if ml_regime.trend_strength > 60:
        health_score += 20
    if ml_regime.volatility_state in ["Normal", "Low"]:
        health_score += 15
    if not (oi_trap_result and oi_trap_result.trap_detected):
        health_score += 15
    health_score = min(health_score, 100)

    # Summary text
    summary_text = f"""
Market is in {ml_regime.regime} regime with {ml_regime.confidence:.0f}% confidence.
Overall bias is {overall_bias} with {bias_confidence:.0f}% conviction.
Trend quality: {trend_quality} | Volatility: {ml_regime.volatility_state}
Strategy: {ml_regime.recommended_strategy}
"""

    # Actionable insights
    insights.append(f"🎯 {trade_signal}: {ml_regime.recommended_strategy}")
    insights.append(f"📊 Target: {key_target:.2f} | Support: {support_level:.2f} | Resistance: {resistance_level:.2f}")
    insights.append(f"⚠️ Risk Level: {risk_level} | Timeframe: {ml_regime.optimal_timeframe}")

    if oi_trap_result and oi_trap_result.trap_detected:
        insights.append(f"🚨 {oi_trap_result.trap_type.value} - Use caution!")

    if participant_result and participant_result.smart_money_detected:
        insights.append(f"🏦 {participant_result.recommendation}")

    return MarketSummary(
        overall_bias=overall_bias,
        bias_confidence=bias_confidence,
        regime=ml_regime.regime,
        volatility=ml_regime.volatility_state,
        trend_quality=trend_quality,
        momentum=momentum,
        support_level=support_level,
        resistance_level=resistance_level,
        key_target=key_target,
        risk_level=risk_level,
        trade_signal=trade_signal,
        conviction_score=conviction_score,
        market_health_score=health_score,
        summary_text=summary_text.strip(),
        actionable_insights=insights
    )


def format_market_summary(summary: MarketSummary) -> str:
    """Format market summary as readable report"""
    return f"""
╔══════════════════════════════════════════════════════════╗
║              MASTER MARKET SUMMARY                       ║
╚══════════════════════════════════════════════════════════╝

🎯 TRADE SIGNAL: {summary.trade_signal}
📊 OVERALL BIAS: {summary.overall_bias} ({summary.bias_confidence:.0f}% confidence)
💪 CONVICTION: {summary.conviction_score:.0f}/100
🏥 MARKET HEALTH: {summary.market_health_score:.0f}/100

─────────────────────────────────────────────────────────
MARKET STATE:
  • Regime: {summary.regime}
  • Volatility: {summary.volatility}
  • Trend Quality: {summary.trend_quality}
  • Momentum: {summary.momentum}
  • Risk Level: {summary.risk_level}

KEY LEVELS:
  • Target: {summary.key_target:.2f}
  • Resistance: {summary.resistance_level:.2f}
  • Support: {summary.support_level:.2f}

─────────────────────────────────────────────────────────
💡 ACTIONABLE INSIGHTS:
"""
    + "\n".join(f"  • {insight}" for insight in summary.actionable_insights) + "\n"
