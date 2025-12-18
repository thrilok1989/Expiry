"""
Volume Order Blocks Indicator
Converted from Pine Script by BigBeluga
"""

import pandas as pd
import numpy as np


class VolumeOrderBlocks:
    """
    Volume Order Blocks indicator that detects institutional order blocks
    based on volume and EMA crossovers
    """

    def __init__(self, sensitivity=5, mid_line=True, trend_shadow=True,
                 color1='#26ba9f', color2='#6626ba'):
        """
        Initialize Volume Order Blocks indicator

        Args:
            sensitivity: Detection sensitivity (default 5)
            mid_line: Show mid line (default True)
            trend_shadow: Show trend shadow (default True)
            color1: Bullish color
            color2: Bearish color
        """
        self.length1 = sensitivity
        self.length2 = sensitivity + 13
        self.mid_line = mid_line
        self.trend_shadow = trend_shadow
        self.color1 = color1
        self.color2 = color2

    def calculate(self, df):
        """
        Calculate Volume Order Blocks

        Args:
            df: DataFrame with OHLCV data

        Returns:
            dict: Dictionary containing bullish and bearish order blocks
        """
        df = df.copy()

        # Calculate EMAs
        df['ema1'] = df['close'].ewm(span=self.length1, adjust=False).mean()
        df['ema2'] = df['ema1'].ewm(span=self.length2, adjust=False).mean()

        # Detect crossovers
        df['cross_up'] = (df['ema1'] > df['ema2']) & (df['ema1'].shift(1) <= df['ema2'].shift(1))
        df['cross_dn'] = (df['ema1'] < df['ema2']) & (df['ema1'].shift(1) >= df['ema2'].shift(1))

        # Calculate ATR for filtering
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=200).mean()
        df['atr_high'] = df['atr'].rolling(window=200).max() * 3
        df['atr_mid'] = df['atr'].rolling(window=200).max() * 2

        # Detect bullish and bearish order blocks
        bullish_blocks = []
        bearish_blocks = []

        # Process bullish order blocks (cross_up)
        for i in range(len(df)):
            if df['cross_up'].iloc[i]:
                # Look back for lowest low
                lookback_data = df.iloc[max(0, i-self.length2):i]
                if len(lookback_data) > 0:
                    lowest = lookback_data['low'].min()
                    lowest_idx = lookback_data['low'].idxmin()

                    # Calculate volume for the block
                    vol = df.iloc[max(0, i-self.length2):i+1]['volume'].sum()

                    # Get candle at lowest point
                    lowest_candle = df.loc[lowest_idx]
                    src = min(lowest_candle['open'], lowest_candle['close'])

                    # Adjust if too close
                    if (src - lowest) < df['atr_mid'].iloc[i] * 0.5:
                        src = lowest + df['atr_mid'].iloc[i] * 0.5

                    mid = (src + lowest) / 2

                    bullish_blocks.append({
                        'index': i,
                        'upper': src,
                        'lower': lowest,
                        'mid': mid,
                        'volume': vol,
                        'active': True
                    })

        # Process bearish order blocks (cross_dn)
        for i in range(len(df)):
            if df['cross_dn'].iloc[i]:
                # Look back for highest high
                lookback_data = df.iloc[max(0, i-self.length2):i]
                if len(lookback_data) > 0:
                    highest = lookback_data['high'].max()
                    highest_idx = lookback_data['high'].idxmax()

                    # Calculate volume for the block
                    vol = df.iloc[max(0, i-self.length2):i+1]['volume'].sum()

                    # Get candle at highest point
                    highest_candle = df.loc[highest_idx]
                    src = max(highest_candle['open'], highest_candle['close'])

                    # Adjust if too close
                    if (highest - src) < df['atr_mid'].iloc[i] * 0.5:
                        src = highest - df['atr_mid'].iloc[i] * 0.5

                    mid = (src + highest) / 2

                    bearish_blocks.append({
                        'index': i,
                        'upper': highest,
                        'lower': src,
                        'mid': mid,
                        'volume': vol,
                        'active': True
                    })

        # Filter overlapping blocks and crossed blocks
        bullish_blocks = self._filter_blocks(bullish_blocks, df, 'bullish')
        bearish_blocks = self._filter_blocks(bearish_blocks, df, 'bearish')

        # Keep only last 15 blocks
        bullish_blocks = bullish_blocks[-15:]
        bearish_blocks = bearish_blocks[-15:]

        return {
            'bullish_blocks': bullish_blocks,
            'bearish_blocks': bearish_blocks,
            'ema1': df['ema1'].values,
            'ema2': df['ema2'].values
        }

    def _filter_blocks(self, blocks, df, block_type):
        """Filter overlapping and crossed order blocks"""
        if len(blocks) == 0:
            return blocks

        filtered_blocks = []

        for i, block in enumerate(blocks):
            # Check if price crossed the block
            idx = block['index']
            if idx < len(df) - 1:
                future_data = df.iloc[idx+1:]

                if block_type == 'bullish':
                    # Check if price went below lower level
                    if (future_data['close'] < block['lower']).any():
                        continue
                else:
                    # Check if price went above upper level
                    if (future_data['close'] > block['upper']).any():
                        continue

            # Check for overlaps with previous block
            if len(filtered_blocks) > 0:
                prev_block = filtered_blocks[-1]
                if abs(block['mid'] - prev_block['mid']) < df['atr'].iloc[block['index']]:
                    continue

            filtered_blocks.append(block)

        return filtered_blocks
