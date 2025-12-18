"""
Market Regime Detector
Identifies current market regime using indicator data and ML
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum


class MarketRegime(Enum):
    """Market regime types"""
    STRONG_UPTREND = "STRONG_UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    RANGING = "RANGING"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    REVERSAL_TO_UPTREND = "REVERSAL_TO_UPTREND"
    REVERSAL_TO_DOWNTREND = "REVERSAL_TO_DOWNTREND"
    UNCERTAIN = "UNCERTAIN"


class VolatilityRegime(Enum):
    """Volatility regime types"""
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    NORMAL_VOLATILITY = "NORMAL_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


class MarketRegimeDetector:
    """
    Detects market regime using rule-based logic and indicator data
    """

    def __init__(self):
        """Initialize regime detector"""
        pass

    def detect_regime(self, df: pd.DataFrame, indicator_data: Dict) -> Dict:
        """
        Detect current market regime

        Args:
            df: DataFrame with OHLCV data
            indicator_data: Dict with all indicator outputs

        Returns:
            Dict with regime, confidence, volatility, and recommendations
        """
        # Detect trend direction and strength
        trend_info = self._analyze_trend(df, indicator_data)

        # Detect ranging conditions
        is_ranging = self._detect_ranging(df, indicator_data)

        # Detect reversals
        reversal_info = self._detect_reversal(indicator_data)

        # Detect volatility
        volatility_regime = self._detect_volatility(df)

        # Combine signals to determine regime
        regime_result = self._combine_signals(
            trend_info, is_ranging, reversal_info, volatility_regime
        )

        # Generate trading recommendations
        recommendations = self._generate_recommendations(regime_result)

        return {
            'regime': regime_result['regime'],
            'confidence': regime_result['confidence'],
            'volatility': volatility_regime,
            'trend_strength': trend_info['strength'],
            'trend_direction': trend_info['direction'],
            'is_ranging': is_ranging,
            'reversal_signal': reversal_info['detected'],
            'recommendations': recommendations,
            'details': regime_result.get('details', {})
        }

    def _analyze_trend(self, df: pd.DataFrame, indicator_data: Dict) -> Dict:
        """Analyze trend using BOS and price action"""
        bos_events = indicator_data.get('bos', [])
        recent_bos = bos_events[-10:] if len(bos_events) > 0 else []

        bullish_bos = [b for b in recent_bos if b.get('type') == 'BULLISH']
        bearish_bos = [b for b in recent_bos if b.get('type') == 'BEARISH']

        # Calculate trend score
        if len(bullish_bos) + len(bearish_bos) > 0:
            trend_score = (len(bullish_bos) - len(bearish_bos)) / (len(bullish_bos) + len(bearish_bos))
        else:
            trend_score = 0

        # Determine direction
        if trend_score > 0.3:
            direction = 'BULLISH'
        elif trend_score < -0.3:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'

        # Calculate strength
        strength = abs(trend_score)

        return {
            'direction': direction,
            'strength': strength,
            'score': trend_score,
            'bullish_bos_count': len(bullish_bos),
            'bearish_bos_count': len(bearish_bos)
        }

    def _detect_ranging(self, df: pd.DataFrame, indicator_data: Dict) -> bool:
        """Detect if market is ranging"""
        htf_levels = indicator_data.get('htf_sr', [])

        if len(htf_levels) < 2:
            return False

        current_price = df['close'].iloc[-1]

        # Count support bounces and resistance rejections
        support_bounces = 0
        resistance_rejections = 0

        for level in htf_levels:
            if level['type'] == 'support':
                # Check if price bounced from this level recently
                if self._check_bounce(df, level['price'], direction='up'):
                    support_bounces += 1
            elif level['type'] == 'resistance':
                # Check if price rejected at this level recently
                if self._check_bounce(df, level['price'], direction='down'):
                    resistance_rejections += 1

        # Ranging if both supports and resistances are being respected
        return support_bounces >= 1 and resistance_rejections >= 1

    def _check_bounce(self, df: pd.DataFrame, level_price: float, direction: str, tolerance: float = 0.01) -> bool:
        """Check if price bounced from a level recently"""
        recent_data = df.tail(20)

        for i in range(len(recent_data)):
            price_low = recent_data['low'].iloc[i]
            price_high = recent_data['high'].iloc[i]
            price_close = recent_data['close'].iloc[i]

            # Check if price touched the level
            touched = (price_low <= level_price * (1 + tolerance) and
                      price_high >= level_price * (1 - tolerance))

            if touched:
                # Check bounce direction
                if direction == 'up' and i < len(recent_data) - 1:
                    # Price should move up after touching
                    next_close = recent_data['close'].iloc[i + 1]
                    if next_close > level_price:
                        return True
                elif direction == 'down' and i < len(recent_data) - 1:
                    # Price should move down after touching
                    next_close = recent_data['close'].iloc[i + 1]
                    if next_close < level_price:
                        return True

        return False

    def _detect_reversal(self, indicator_data: Dict) -> Dict:
        """Detect potential reversals using CHOCH and divergences"""
        choch_events = indicator_data.get('choch', [])
        recent_choch = choch_events[-3:] if len(choch_events) > 0 else []

        rsi_data = indicator_data.get('rsi', {})
        patterns = indicator_data.get('patterns', [])

        reversal_detected = False
        reversal_direction = None
        confidence = 0.0

        # CHOCH is primary reversal signal
        if recent_choch:
            reversal_detected = True
            last_choch = recent_choch[-1]
            reversal_direction = 'BULLISH' if last_choch.get('type') == 'BULLISH' else 'BEARISH'
            confidence = 0.6

            # Increase confidence with RSI divergence
            if rsi_data and 'divergence' in rsi_data:
                confidence += 0.2

            # Increase confidence with reversal patterns
            reversal_patterns = ['Head and Shoulders', 'Inverse Head and Shoulders']
            if any(p['name'] in reversal_patterns for p in patterns):
                confidence += 0.2

        return {
            'detected': reversal_detected,
            'direction': reversal_direction,
            'confidence': min(confidence, 1.0)
        }

    def _detect_volatility(self, df: pd.DataFrame, atr_period: int = 14) -> str:
        """Detect volatility regime"""
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(atr_period).mean()

        current_atr = atr.iloc[-1]
        avg_atr = atr.rolling(20).mean().iloc[-1]

        if current_atr > avg_atr * 1.5:
            return VolatilityRegime.HIGH_VOLATILITY.value
        elif current_atr < avg_atr * 0.6:
            return VolatilityRegime.LOW_VOLATILITY.value
        else:
            return VolatilityRegime.NORMAL_VOLATILITY.value

    def _combine_signals(self, trend_info: Dict, is_ranging: bool,
                        reversal_info: Dict, volatility: str) -> Dict:
        """Combine all signals to determine final regime"""

        # Reversal takes priority
        if reversal_info['detected']:
            if reversal_info['direction'] == 'BULLISH':
                regime = MarketRegime.REVERSAL_TO_UPTREND.value
            else:
                regime = MarketRegime.REVERSAL_TO_DOWNTREND.value

            return {
                'regime': regime,
                'confidence': reversal_info['confidence'],
                'details': {
                    'reversal': True,
                    'trend': trend_info
                }
            }

        # Ranging market
        if is_ranging:
            return {
                'regime': MarketRegime.RANGING.value,
                'confidence': 0.75,
                'details': {
                    'ranging': True
                }
            }

        # Trending market
        direction = trend_info['direction']
        strength = trend_info['strength']

        if direction == 'BULLISH':
            if strength > 0.6:
                regime = MarketRegime.STRONG_UPTREND.value
                confidence = strength
            elif strength > 0.3:
                regime = MarketRegime.WEAK_UPTREND.value
                confidence = strength
            else:
                regime = MarketRegime.UNCERTAIN.value
                confidence = 0.4
        elif direction == 'BEARISH':
            if strength > 0.6:
                regime = MarketRegime.STRONG_DOWNTREND.value
                confidence = strength
            elif strength > 0.3:
                regime = MarketRegime.WEAK_DOWNTREND.value
                confidence = strength
            else:
                regime = MarketRegime.UNCERTAIN.value
                confidence = 0.4
        else:
            regime = MarketRegime.UNCERTAIN.value
            confidence = 0.3

        return {
            'regime': regime,
            'confidence': confidence,
            'details': {
                'trend': trend_info
            }
        }

    def _generate_recommendations(self, regime_result: Dict) -> Dict:
        """Generate trading recommendations based on regime"""
        regime = regime_result['regime']
        confidence = regime_result['confidence']

        recommendations = {
            'position_bias': 'NEUTRAL',
            'strategy': 'WAIT',
            'risk_adjustment': 1.0,
            'stop_loss_multiplier': 1.0,
            'position_size_multiplier': 1.0,
            'allowed_setups': []
        }

        if regime == MarketRegime.STRONG_UPTREND.value:
            recommendations.update({
                'position_bias': 'LONG_ONLY',
                'strategy': 'BUY_PULLBACKS',
                'risk_adjustment': 0.8,
                'stop_loss_multiplier': 0.9,
                'position_size_multiplier': 1.2,
                'allowed_setups': ['Bullish Order Block', 'Fibonacci 61.8%', 'Support Bounce']
            })
        elif regime == MarketRegime.WEAK_UPTREND.value:
            recommendations.update({
                'position_bias': 'LONG_PREFERRED',
                'strategy': 'BUY_PULLBACKS',
                'risk_adjustment': 1.0,
                'stop_loss_multiplier': 1.0,
                'position_size_multiplier': 1.0,
                'allowed_setups': ['Bullish Order Block', 'Support']
            })
        elif regime == MarketRegime.STRONG_DOWNTREND.value:
            recommendations.update({
                'position_bias': 'SHORT_ONLY',
                'strategy': 'SELL_RALLIES',
                'risk_adjustment': 0.8,
                'stop_loss_multiplier': 0.9,
                'position_size_multiplier': 1.2,
                'allowed_setups': ['Bearish Order Block', 'Fibonacci 61.8%', 'Resistance Rejection']
            })
        elif regime == MarketRegime.WEAK_DOWNTREND.value:
            recommendations.update({
                'position_bias': 'SHORT_PREFERRED',
                'strategy': 'SELL_RALLIES',
                'risk_adjustment': 1.0,
                'stop_loss_multiplier': 1.0,
                'position_size_multiplier': 1.0,
                'allowed_setups': ['Bearish Order Block', 'Resistance']
            })
        elif regime == MarketRegime.RANGING.value:
            recommendations.update({
                'position_bias': 'NEUTRAL',
                'strategy': 'MEAN_REVERSION',
                'risk_adjustment': 1.2,
                'stop_loss_multiplier': 0.8,
                'position_size_multiplier': 0.8,
                'allowed_setups': ['Buy Support / Sell Resistance', 'Order Blocks']
            })
        elif 'REVERSAL' in regime:
            recommendations.update({
                'position_bias': 'LONG' if 'UPTREND' in regime else 'SHORT',
                'strategy': 'WAIT_FOR_CONFIRMATION',
                'risk_adjustment': 1.5,
                'stop_loss_multiplier': 1.2,
                'position_size_multiplier': 0.5,
                'allowed_setups': ['Wait for 2-3 BOS confirmations']
            })
        else:  # UNCERTAIN
            recommendations.update({
                'position_bias': 'NEUTRAL',
                'strategy': 'WAIT',
                'risk_adjustment': 2.0,
                'stop_loss_multiplier': 1.5,
                'position_size_multiplier': 0.3,
                'allowed_setups': ['High confidence setups only']
            })

        return recommendations
