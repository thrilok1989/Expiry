"""
Feature Extractor for Machine Learning
Converts indicator data into ML features for regime classification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class FeatureExtractor:
    """
    Extracts features from indicator data for ML models
    """

    def __init__(self):
        """Initialize feature extractor"""
        pass

    def extract_features(self, df: pd.DataFrame, indicator_data: Dict) -> pd.DataFrame:
        """
        Extract all features from OHLCV data and indicator outputs

        Args:
            df: DataFrame with OHLCV data
            indicator_data: Dict containing all indicator outputs

        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features = self._add_price_features(features, df)

        # Volume features
        features = self._add_volume_features(features, df)

        # Order Block features
        if 'order_blocks' in indicator_data:
            features = self._add_order_block_features(features, df, indicator_data['order_blocks'])

        # HTF Support/Resistance features
        if 'htf_sr' in indicator_data:
            features = self._add_htf_sr_features(features, df, indicator_data['htf_sr'])

        # BOS/CHOCH features
        if 'bos' in indicator_data:
            features = self._add_bos_features(features, df, indicator_data['bos'])

        if 'choch' in indicator_data:
            features = self._add_choch_features(features, df, indicator_data['choch'])

        # Volume Footprint features
        if 'footprint' in indicator_data:
            features = self._add_footprint_features(features, df, indicator_data['footprint'])

        # RSI features
        if 'rsi' in indicator_data:
            features = self._add_rsi_features(features, indicator_data['rsi'])

        # Pattern features
        if 'patterns' in indicator_data:
            features = self._add_pattern_features(features, indicator_data['patterns'])

        return features

    def _add_price_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        features['close'] = df['close']
        features['price_change_1'] = df['close'].pct_change(1)
        features['price_change_5'] = df['close'].pct_change(5)
        features['price_change_15'] = df['close'].pct_change(15)
        features['price_change_30'] = df['close'].pct_change(30)

        # Moving averages
        features['sma_20'] = df['close'].rolling(20).mean()
        features['sma_50'] = df['close'].rolling(50).mean()
        features['price_above_sma20'] = (df['close'] > features['sma_20']).astype(int)
        features['price_above_sma50'] = (df['close'] > features['sma_50']).astype(int)

        # ATR (volatility)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        features['atr'] = true_range.rolling(14).mean()
        features['atr_ratio'] = features['atr'] / features['atr'].rolling(20).mean()

        # Price range
        features['price_range_20'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']

        # Higher highs / Lower lows
        features['consecutive_higher_highs'] = self._count_consecutive_higher_highs(df['high'])
        features['consecutive_lower_lows'] = self._count_consecutive_lower_lows(df['low'])

        return features

    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        features['volume'] = df['volume']
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']

        # Volume trend
        features['volume_increasing'] = (df['volume'] > df['volume'].shift(1)).astype(int)

        return features

    def _add_order_block_features(self, features: pd.DataFrame, df: pd.DataFrame, ob_data: Dict) -> pd.DataFrame:
        """Add Order Block features"""
        current_price = df['close'].iloc[-1]

        # Bullish blocks
        bullish_blocks = ob_data.get('bullish_blocks', [])
        active_bullish = [b for b in bullish_blocks if b.get('active', False)]

        features['num_bullish_blocks'] = len(active_bullish)
        features['num_bearish_blocks'] = len([b for b in ob_data.get('bearish_blocks', []) if b.get('active', False)])

        if active_bullish:
            nearest_bull = min(active_bullish, key=lambda x: abs(x['mid'] - current_price))
            features['dist_to_bull_block'] = (current_price - nearest_bull['mid']) / current_price
            features['in_bullish_block'] = int(nearest_bull['lower'] <= current_price <= nearest_bull['upper'])
        else:
            features['dist_to_bull_block'] = np.nan
            features['in_bullish_block'] = 0

        # Bearish blocks
        bearish_blocks = ob_data.get('bearish_blocks', [])
        active_bearish = [b for b in bearish_blocks if b.get('active', False)]

        if active_bearish:
            nearest_bear = min(active_bearish, key=lambda x: abs(x['mid'] - current_price))
            features['dist_to_bear_block'] = (nearest_bear['mid'] - current_price) / current_price
            features['in_bearish_block'] = int(nearest_bear['lower'] <= current_price <= nearest_bear['upper'])
        else:
            features['dist_to_bear_block'] = np.nan
            features['in_bearish_block'] = 0

        return features

    def _add_htf_sr_features(self, features: pd.DataFrame, df: pd.DataFrame, htf_levels: List[Dict]) -> pd.DataFrame:
        """Add HTF Support/Resistance features"""
        current_price = df['close'].iloc[-1]

        supports = [l for l in htf_levels if l['type'] == 'support']
        resistances = [l for l in htf_levels if l['type'] == 'resistance']

        # Distance to nearest support/resistance
        if supports:
            nearest_support = max(supports, key=lambda x: x['price'])
            features['dist_to_support'] = (current_price - nearest_support['price']) / current_price
            features['near_support'] = int(abs(features['dist_to_support'].iloc[-1]) < 0.005)  # Within 0.5%
        else:
            features['dist_to_support'] = np.nan
            features['near_support'] = 0

        if resistances:
            nearest_resistance = min(resistances, key=lambda x: x['price'])
            features['dist_to_resistance'] = (nearest_resistance['price'] - current_price) / current_price
            features['near_resistance'] = int(abs(features['dist_to_resistance'].iloc[-1]) < 0.005)
        else:
            features['dist_to_resistance'] = np.nan
            features['near_resistance'] = 0

        # Confluence (multiple timeframes aligned)
        features['sr_confluence_count'] = len(htf_levels)

        return features

    def _add_bos_features(self, features: pd.DataFrame, df: pd.DataFrame, bos_events: List[Dict]) -> pd.DataFrame:
        """Add Break of Structure features"""
        # Count recent BOS events
        recent_bos = bos_events[-10:] if len(bos_events) > 0 else []

        bullish_bos = [b for b in recent_bos if b.get('type') == 'BULLISH']
        bearish_bos = [b for b in recent_bos if b.get('type') == 'BEARISH']

        features['bullish_bos_count_5'] = len([b for b in bos_events[-5:] if b.get('type') == 'BULLISH'])
        features['bearish_bos_count_5'] = len([b for b in bos_events[-5:] if b.get('type') == 'BEARISH'])

        features['bullish_bos_count_10'] = len(bullish_bos)
        features['bearish_bos_count_10'] = len(bearish_bos)

        # BOS trend strength
        if len(bullish_bos) + len(bearish_bos) > 0:
            features['bos_trend_strength'] = (len(bullish_bos) - len(bearish_bos)) / (len(bullish_bos) + len(bearish_bos))
        else:
            features['bos_trend_strength'] = 0

        # Last BOS type
        if recent_bos:
            features['last_bos_bullish'] = int(recent_bos[-1].get('type') == 'BULLISH')
        else:
            features['last_bos_bullish'] = 0

        return features

    def _add_choch_features(self, features: pd.DataFrame, df: pd.DataFrame, choch_events: List[Dict]) -> pd.DataFrame:
        """Add Change of Character features"""
        recent_choch = choch_events[-3:] if len(choch_events) > 0 else []

        features['choch_count_recent'] = len(recent_choch)

        if recent_choch:
            features['last_choch_bullish'] = int(recent_choch[-1].get('type') == 'BULLISH')
            features['choch_detected'] = 1
        else:
            features['last_choch_bullish'] = 0
            features['choch_detected'] = 0

        return features

    def _add_footprint_features(self, features: pd.DataFrame, footprint_data: Dict) -> pd.DataFrame:
        """Add Volume Footprint features"""
        if footprint_data:
            features['buy_volume'] = footprint_data.get('buy_volume', 0)
            features['sell_volume'] = footprint_data.get('sell_volume', 0)
            features['delta'] = footprint_data.get('delta', 0)

            total_vol = features['buy_volume'] + features['sell_volume']
            if total_vol > 0:
                features['buy_sell_ratio'] = features['buy_volume'] / total_vol
            else:
                features['buy_sell_ratio'] = 0.5

        return features

    def _add_rsi_features(self, features: pd.DataFrame, rsi_data: Dict) -> pd.DataFrame:
        """Add RSI features"""
        if 'ultimate_rsi' in rsi_data:
            rsi_values = rsi_data['ultimate_rsi']
            features['rsi'] = rsi_values[-1] if len(rsi_values) > 0 else 50

            features['rsi_oversold'] = int(features['rsi'].iloc[-1] < 30)
            features['rsi_overbought'] = int(features['rsi'].iloc[-1] > 70)

        if 'signal' in rsi_data:
            signal_map = {'BUY': 1, 'NEUTRAL': 0, 'SELL': -1}
            features['rsi_signal'] = signal_map.get(rsi_data['signal'][-1], 0)

        return features

    def _add_pattern_features(self, features: pd.DataFrame, patterns: List[Dict]) -> pd.DataFrame:
        """Add geometric pattern features"""
        bullish_patterns = ['Inverse Head and Shoulders', 'Ascending Triangle', 'Bull Flag', 'Bull Pennant']
        bearish_patterns = ['Head and Shoulders', 'Descending Triangle', 'Bear Flag', 'Bear Pennant']

        features['has_bullish_pattern'] = int(any(p['type'] in bullish_patterns for p in patterns))
        features['has_bearish_pattern'] = int(any(p['type'] in bearish_patterns for p in patterns))

        return features

    def _count_consecutive_higher_highs(self, highs: pd.Series, window: int = 5) -> pd.Series:
        """Count consecutive higher highs"""
        count = pd.Series(0, index=highs.index)
        for i in range(1, len(highs)):
            if highs.iloc[i] > highs.iloc[i-1]:
                count.iloc[i] = count.iloc[i-1] + 1
            else:
                count.iloc[i] = 0
        return count

    def _count_consecutive_lower_lows(self, lows: pd.Series, window: int = 5) -> pd.Series:
        """Count consecutive lower lows"""
        count = pd.Series(0, index=lows.index)
        for i in range(1, len(lows)):
            if lows.iloc[i] < lows.iloc[i-1]:
                count.iloc[i] = count.iloc[i-1] + 1
            else:
                count.iloc[i] = 0
        return count
