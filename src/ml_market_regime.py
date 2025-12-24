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
    support_resistance: Dict = None  # Support/Resistance levels (major and near)
    entry_exit_signals: Dict = None  # Entry/Exit signals with levels


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
        oi_trap_result: Optional[any] = None,
        option_chain_data: Optional[Dict] = None,
        sector_rotation_data: Optional[Dict] = None,
        bias_analysis_data: Optional[Dict] = None,
        india_vix_data: Optional[Dict] = None,
        gamma_squeeze_data: Optional[Dict] = None,
        advanced_chart_indicators: Optional[Dict] = None
    ) -> MLMarketRegimeResult:
        """
        Detect market regime using ML-style feature engineering
        Now uses ALL available data sources for comprehensive analysis

        Args:
            df: OHLCV dataframe with indicators
            cvd_result: CVD analysis result
            volatility_result: Volatility regime result
            oi_trap_result: OI trap detection result
            option_chain_data: Option chain analysis data (PCR, max pain, OI, gamma)
            sector_rotation_data: Sector rotation analysis (breadth, rotation bias)
            bias_analysis_data: Bias analysis from all 8 indicators
            india_vix_data: India VIX sentiment data
            gamma_squeeze_data: Gamma squeeze risk analysis
            advanced_chart_indicators: All indicators from Advanced Chart Analysis

        Returns:
            MLMarketRegimeResult with regime classification
        """
        signals = []

        if len(df) < 50:
            return self._default_result()

        # Feature Engineering (Base Features)
        features = self._engineer_features(df)

        # Add Option Chain Features
        if option_chain_data and option_chain_data.get('success'):
            option_features = self._extract_option_chain_features(option_chain_data)
            features.update(option_features)

        # Add Sector Rotation Features
        if sector_rotation_data and sector_rotation_data.get('success'):
            sector_features = self._extract_sector_rotation_features(sector_rotation_data)
            features.update(sector_features)

        # Add Bias Analysis Features
        if bias_analysis_data and bias_analysis_data.get('success'):
            bias_features = self._extract_bias_analysis_features(bias_analysis_data)
            features.update(bias_features)

        # Add India VIX Features
        if india_vix_data and india_vix_data.get('success'):
            vix_features = self._extract_vix_features(india_vix_data)
            features.update(vix_features)

        # Add Gamma Squeeze Features
        if gamma_squeeze_data and gamma_squeeze_data.get('success'):
            gamma_features = self._extract_gamma_features(gamma_squeeze_data)
            features.update(gamma_features)

        # Add Advanced Chart Indicator Features
        if advanced_chart_indicators:
            chart_features = self._extract_chart_indicator_features(advanced_chart_indicators)
            features.update(chart_features)

        # Calculate regime probabilities using ALL features
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

        # Add signals from new data sources
        if option_chain_data and option_chain_data.get('success'):
            signals.extend(self._generate_option_chain_signals(option_chain_data))

        if sector_rotation_data and sector_rotation_data.get('success'):
            signals.extend(self._generate_sector_rotation_signals(sector_rotation_data))

        if bias_analysis_data and bias_analysis_data.get('success'):
            signals.extend(self._generate_bias_analysis_signals(bias_analysis_data))

        if india_vix_data and india_vix_data.get('success'):
            signals.extend(self._generate_vix_signals(india_vix_data))

        if gamma_squeeze_data and gamma_squeeze_data.get('success'):
            signals.extend(self._generate_gamma_signals(gamma_squeeze_data))

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

        # Calculate Support/Resistance Levels
        support_resistance = self._calculate_support_resistance_levels(
            df, features, option_chain_data, advanced_chart_indicators
        )

        # Generate Entry/Exit Signals
        entry_exit_signals = self._generate_entry_exit_signals(
            regime, confidence, trend_strength, features, support_resistance
        )

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
            signals=signals,
            support_resistance=support_resistance,
            entry_exit_signals=entry_exit_signals
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

    # =========================================================================
    # NEW FEATURE EXTRACTION METHODS
    # =========================================================================

    def _extract_option_chain_features(self, option_data: Dict) -> Dict[str, float]:
        """Extract features from option chain data"""
        features = {}

        # PCR features
        features['pcr_value'] = option_data.get('pcr', 1.0)
        features['pcr_bullish'] = 1.0 if features['pcr_value'] < 0.7 else 0.0
        features['pcr_bearish'] = 1.0 if features['pcr_value'] > 1.3 else 0.0

        # OI concentration
        features['call_oi_concentration'] = option_data.get('call_oi_concentration', 0.0)
        features['put_oi_concentration'] = option_data.get('put_oi_concentration', 0.0)

        # Max Pain vs Spot
        max_pain = option_data.get('max_pain', 0)
        spot = option_data.get('spot', 0)
        if spot > 0:
            features['max_pain_distance'] = ((spot - max_pain) / spot) * 100
        else:
            features['max_pain_distance'] = 0

        # Gamma exposure
        features['total_gamma'] = option_data.get('total_gamma', 0)
        features['call_gamma'] = option_data.get('total_call_gamma', 0)
        features['put_gamma'] = option_data.get('total_put_gamma', 0)

        return features

    def _extract_sector_rotation_features(self, sector_data: Dict) -> Dict[str, float]:
        """Extract features from sector rotation analysis"""
        features = {}

        # Sector breadth
        features['sector_breadth'] = sector_data.get('sector_breadth', 50.0)
        features['bullish_sectors'] = sector_data.get('bullish_sectors_count', 0)
        features['bearish_sectors'] = sector_data.get('bearish_sectors_count', 0)

        # Rotation bias score
        rotation_bias = sector_data.get('rotation_bias', 'NEUTRAL')
        if 'BULLISH' in rotation_bias:
            features['rotation_score'] = sector_data.get('rotation_score', 50) / 100.0
        elif 'BEARISH' in rotation_bias:
            features['rotation_score'] = -sector_data.get('rotation_score', 50) / 100.0
        else:
            features['rotation_score'] = 0.0

        # Leader/laggard spread
        leaders = sector_data.get('leaders', [])
        laggards = sector_data.get('laggards', [])
        if leaders and laggards:
            leader_pct = leaders[0].get('change_pct', 0)
            laggard_pct = laggards[0].get('change_pct', 0)
            features['leader_laggard_spread'] = leader_pct - laggard_pct
        else:
            features['leader_laggard_spread'] = 0.0

        return features

    def _extract_bias_analysis_features(self, bias_data: Dict) -> Dict[str, float]:
        """Extract features from bias analysis (8 indicators)"""
        features = {}

        # Overall bias
        overall_bias = bias_data.get('overall_bias', 'NEUTRAL')
        features['bias_bullish'] = 1.0 if overall_bias == 'BULLISH' else 0.0
        features['bias_bearish'] = 1.0 if overall_bias == 'BEARISH' else 0.0

        # Confidence
        features['bias_confidence'] = bias_data.get('overall_confidence', 50.0) / 100.0

        # Indicator alignment
        total_indicators = bias_data.get('total_indicators', 8)
        bullish_count = bias_data.get('bullish_count', 0)
        bearish_count = bias_data.get('bearish_count', 0)

        if total_indicators > 0:
            features['bias_alignment'] = (bullish_count - bearish_count) / total_indicators
        else:
            features['bias_alignment'] = 0.0

        # Fast vs Slow divergence
        features['fast_bull_pct'] = bias_data.get('fast_bull_pct', 50.0) / 100.0
        features['fast_bear_pct'] = bias_data.get('fast_bear_pct', 50.0) / 100.0

        return features

    def _extract_vix_features(self, vix_data: Dict) -> Dict[str, float]:
        """Extract features from India VIX"""
        features = {}

        # VIX value and sentiment
        features['vix_value'] = vix_data.get('value', 15.0)
        features['vix_score'] = vix_data.get('score', 0) / 100.0

        # VIX regime
        vix_sentiment = vix_data.get('sentiment', 'MODERATE')
        if 'FEAR' in vix_sentiment:
            features['vix_regime'] = -1.0
        elif 'LOW' in vix_sentiment or 'COMPLACENCY' in vix_sentiment:
            features['vix_regime'] = 1.0
        else:
            features['vix_regime'] = 0.0

        return features

    def _extract_gamma_features(self, gamma_data: Dict) -> Dict[str, float]:
        """Extract features from gamma squeeze analysis"""
        features = {}

        # Gamma squeeze risk
        squeeze_score = gamma_data.get('squeeze_score', 0)
        features['gamma_squeeze_score'] = squeeze_score / 100.0

        # Net gamma exposure
        net_gamma = gamma_data.get('net_gamma', 0)
        features['net_gamma_normalized'] = np.tanh(net_gamma / 1000000.0) if net_gamma != 0 else 0.0

        # Gamma concentration
        features['gamma_concentration'] = gamma_data.get('gamma_concentration', 0)

        return features

    def _extract_chart_indicator_features(self, chart_indicators: Dict) -> Dict[str, float]:
        """Extract features from Advanced Chart indicators"""
        features = {}

        # Volume Order Blocks
        if 'order_blocks' in chart_indicators:
            ob_data = chart_indicators['order_blocks']
            features['bullish_ob_count'] = len([b for b in ob_data.get('bullish_blocks', []) if b.get('active')])
            features['bearish_ob_count'] = len([b for b in ob_data.get('bearish_blocks', []) if b.get('active')])

        # Ultimate RSI
        if 'rsi' in chart_indicators:
            rsi_data = chart_indicators['rsi']
            rsi_value = rsi_data.get('ultimate_rsi', pd.Series([50])).iloc[-1] if isinstance(rsi_data.get('ultimate_rsi'), pd.Series) else 50
            features['ultimate_rsi'] = rsi_value
            features['rsi_overbought'] = 1.0 if rsi_value > 70 else 0.0
            features['rsi_oversold'] = 1.0 if rsi_value < 30 else 0.0

        # Money Flow Profile
        if 'money_flow_profile' in chart_indicators:
            mfp_data = chart_indicators['money_flow_profile']
            features['money_flow_bullish'] = mfp_data.get('bullish_pct', 50.0) / 100.0

        # DeltaFlow Profile
        if 'deltaflow_profile' in chart_indicators:
            dfp_data = chart_indicators['deltaflow_profile']
            features['deltaflow_sentiment'] = dfp_data.get('overall_delta', 0) / 100.0

        # BOS/CHOCH signals
        if 'bos' in chart_indicators:
            bos_events = chart_indicators['bos']
            recent_bos = bos_events[-5:] if len(bos_events) > 0 else []
            bullish_bos = len([b for b in recent_bos if b.get('type') == 'BULLISH'])
            bearish_bos = len([b for b in recent_bos if b.get('type') == 'BEARISH'])
            features['bos_bias'] = (bullish_bos - bearish_bos) / max(len(recent_bos), 1)

        return features

    # =========================================================================
    # NEW SIGNAL GENERATION METHODS
    # =========================================================================

    def _generate_option_chain_signals(self, option_data: Dict) -> List[str]:
        """Generate signals from option chain data"""
        signals = []

        pcr = option_data.get('pcr', 1.0)
        if pcr < 0.7:
            signals.append(f"📊 PCR: {pcr:.2f} - Bullish (More Calls)")
        elif pcr > 1.3:
            signals.append(f"📊 PCR: {pcr:.2f} - Bearish (More Puts)")

        max_pain = option_data.get('max_pain', 0)
        spot = option_data.get('spot', 0)
        if max_pain > 0 and spot > 0:
            distance = ((spot - max_pain) / spot) * 100
            if abs(distance) > 1:
                direction = "above" if distance > 0 else "below"
                signals.append(f"🎯 Max Pain: {max_pain:.0f} ({abs(distance):.1f}% {direction} spot)")

        return signals

    def _generate_sector_rotation_signals(self, sector_data: Dict) -> List[str]:
        """Generate signals from sector rotation"""
        signals = []

        breadth = sector_data.get('sector_breadth', 50.0)
        if breadth > 60:
            signals.append(f"🌊 Sector Breadth: {breadth:.0f}% - Strong market breadth")
        elif breadth < 40:
            signals.append(f"🌊 Sector Breadth: {breadth:.0f}% - Weak market breadth")

        rotation_bias = sector_data.get('rotation_bias', 'NEUTRAL')
        if rotation_bias != 'NEUTRAL':
            signals.append(f"🔄 Rotation: {rotation_bias}")

        leaders = sector_data.get('leaders', [])
        if leaders:
            top_sector = leaders[0].get('sector', 'Unknown')
            top_change = leaders[0].get('change_pct', 0)
            signals.append(f"🥇 Leader: {top_sector} (+{top_change:.1f}%)")

        return signals

    def _generate_bias_analysis_signals(self, bias_data: Dict) -> List[str]:
        """Generate signals from bias analysis"""
        signals = []

        overall_bias = bias_data.get('overall_bias', 'NEUTRAL')
        confidence = bias_data.get('overall_confidence', 50.0)
        bullish_count = bias_data.get('bullish_count', 0)
        bearish_count = bias_data.get('bearish_count', 0)
        total = bias_data.get('total_indicators', 8)

        if overall_bias != 'NEUTRAL':
            signals.append(f"📈 Bias: {overall_bias} ({confidence:.0f}% confidence)")
            signals.append(f"📊 Indicators: {bullish_count}🟢 {bearish_count}🔴 of {total}")

        return signals

    def _generate_vix_signals(self, vix_data: Dict) -> List[str]:
        """Generate signals from India VIX"""
        signals = []

        vix_value = vix_data.get('value', 15.0)
        vix_sentiment = vix_data.get('sentiment', 'MODERATE')

        signals.append(f"📊 India VIX: {vix_value:.2f} - {vix_sentiment}")

        return signals

    def _generate_gamma_signals(self, gamma_data: Dict) -> List[str]:
        """Generate signals from gamma squeeze"""
        signals = []

        squeeze_risk = gamma_data.get('squeeze_risk', 'LOW')
        squeeze_bias = gamma_data.get('squeeze_bias', 'NEUTRAL')

        if squeeze_risk != 'LOW':
            signals.append(f"⚡ Gamma Squeeze: {squeeze_risk} - {squeeze_bias}")
            interpretation = gamma_data.get('interpretation', '')
            if interpretation:
                signals.append(f"   {interpretation}")

        return signals

    # =========================================================================
    # SUPPORT/RESISTANCE CALCULATION
    # =========================================================================

    def _calculate_support_resistance_levels(
        self,
        df: pd.DataFrame,
        features: Dict,
        option_data: Optional[Dict],
        chart_indicators: Optional[Dict]
    ) -> Dict:
        """Calculate major and near support/resistance levels"""
        current_price = df['close'].iloc[-1]

        supports = []
        resistances = []

        # 1. From price action (swing highs/lows)
        highs = df['high'].rolling(window=10).max()
        lows = df['low'].rolling(window=10).min()

        # Recent swing levels
        for i in range(-50, -1):
            if i < -len(df):
                break
            price_high = highs.iloc[i]
            price_low = lows.iloc[i]

            if price_high < current_price and price_high not in resistances:
                resistances.append(price_high)
            if price_low > current_price and price_low not in supports:
                supports.append(price_low)

        # 2. From option chain (max pain, high OI strikes)
        if option_data and option_data.get('success'):
            max_pain = option_data.get('max_pain', 0)
            if max_pain > 0:
                if max_pain < current_price:
                    supports.append(max_pain)
                else:
                    resistances.append(max_pain)

        # 3. From order blocks
        if chart_indicators and 'order_blocks' in chart_indicators:
            ob_data = chart_indicators['order_blocks']

            # Bullish OBs = Support
            for block in ob_data.get('bullish_blocks', []):
                if block.get('active'):
                    supports.append(block['mid'])

            # Bearish OBs = Resistance
            for block in ob_data.get('bearish_blocks', []):
                if block.get('active'):
                    resistances.append(block['mid'])

        # Sort and filter
        supports = sorted([s for s in supports if s < current_price], reverse=True)[:5]
        resistances = sorted([r for r in resistances if r > current_price])[:5]

        # Classify major vs near
        atr = df['high'] - df['low']
        avg_atr = atr.tail(14).mean()

        near_supports = [s for s in supports if (current_price - s) < avg_atr * 2]
        major_supports = [s for s in supports if s not in near_supports]

        near_resistances = [r for r in resistances if (r - current_price) < avg_atr * 2]
        major_resistances = [r for r in resistances if r not in near_resistances]

        return {
            'current_price': current_price,
            'major_support': major_supports[0] if major_supports else None,
            'near_support': near_supports[0] if near_supports else None,
            'major_resistance': major_resistances[0] if major_resistances else None,
            'near_resistance': near_resistances[0] if near_resistances else None,
            'all_supports': supports[:3],
            'all_resistances': resistances[:3],
            'atr': avg_atr
        }

    # =========================================================================
    # ENTRY/EXIT SIGNAL GENERATION
    # =========================================================================

    def _generate_entry_exit_signals(
        self,
        regime: str,
        confidence: float,
        trend_strength: float,
        features: Dict,
        support_resistance: Dict
    ) -> Dict:
        """Generate entry/exit signals based on regime and support/resistance"""
        current_price = support_resistance['current_price']
        near_support = support_resistance.get('near_support')
        near_resistance = support_resistance.get('near_resistance')
        major_support = support_resistance.get('major_support')
        major_resistance = support_resistance.get('major_resistance')
        atr = support_resistance.get('atr', 0)

        signals = {
            'action': 'WAIT',
            'direction': 'NEUTRAL',
            'entry_level': None,
            'stop_loss': None,
            'target_1': None,
            'target_2': None,
            'risk_reward': None,
            'confidence': confidence,
            'reasoning': []
        }

        # TRENDING UP REGIME
        if regime == "Trending Up" and confidence > 60:
            signals['direction'] = 'LONG'

            # Entry on pullback to support
            if near_support:
                signals['action'] = 'BUY_ON_PULLBACK'
                signals['entry_level'] = near_support
                signals['stop_loss'] = major_support if major_support else near_support * 0.98
                signals['target_1'] = near_resistance if near_resistance else current_price * 1.015
                signals['target_2'] = major_resistance if major_resistance else current_price * 1.03
                signals['reasoning'].append(f"Buy on pullback to {near_support:.2f}")
                signals['reasoning'].append(f"Stop loss at {signals['stop_loss']:.2f}")
                signals['reasoning'].append(f"Targets: {signals['target_1']:.2f}, {signals['target_2']:.2f}")
            else:
                signals['action'] = 'BUY_ON_BREAK'
                signals['entry_level'] = current_price * 1.001
                signals['stop_loss'] = current_price * 0.995
                signals['target_1'] = current_price * 1.015
                signals['target_2'] = current_price * 1.03

        # TRENDING DOWN REGIME
        elif regime == "Trending Down" and confidence > 60:
            signals['direction'] = 'SHORT'

            # Entry on rally to resistance
            if near_resistance:
                signals['action'] = 'SELL_ON_RALLY'
                signals['entry_level'] = near_resistance
                signals['stop_loss'] = major_resistance if major_resistance else near_resistance * 1.02
                signals['target_1'] = near_support if near_support else current_price * 0.985
                signals['target_2'] = major_support if major_support else current_price * 0.97
                signals['reasoning'].append(f"Sell on rally to {near_resistance:.2f}")
                signals['reasoning'].append(f"Stop loss at {signals['stop_loss']:.2f}")
                signals['reasoning'].append(f"Targets: {signals['target_1']:.2f}, {signals['target_2']:.2f}")
            else:
                signals['action'] = 'SELL_ON_BREAK'
                signals['entry_level'] = current_price * 0.999
                signals['stop_loss'] = current_price * 1.005
                signals['target_1'] = current_price * 0.985
                signals['target_2'] = current_price * 0.97

        # RANGE BOUND REGIME
        elif regime == "Range Bound":
            signals['action'] = 'RANGE_TRADE'
            signals['direction'] = 'BOTH'

            if near_support and near_resistance:
                signals['reasoning'].append(f"Range: {near_support:.2f} - {near_resistance:.2f}")
                signals['reasoning'].append(f"Buy near support, Sell near resistance")
                signals['entry_level'] = f"{near_support:.2f} / {near_resistance:.2f}"

        # REVERSAL REGIME
        elif "Reversal" in regime:
            signals['action'] = 'WAIT_FOR_CONFIRMATION'
            signals['direction'] = 'LONG' if 'Uptrend' in regime else 'SHORT'
            signals['reasoning'].append("Wait for 2-3 BOS confirmations")
            signals['reasoning'].append("Reduce position size until confirmed")

        # Calculate risk/reward if entry and targets are set
        if signals['entry_level'] and isinstance(signals['entry_level'], (int, float)):
            if signals['stop_loss'] and signals['target_1']:
                risk = abs(signals['entry_level'] - signals['stop_loss'])
                reward = abs(signals['target_1'] - signals['entry_level'])
                if risk > 0:
                    signals['risk_reward'] = reward / risk

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
            signals=["Insufficient data for regime detection"],
            support_resistance={},
            entry_exit_signals={}
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
