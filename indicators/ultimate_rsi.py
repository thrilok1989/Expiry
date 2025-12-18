"""
Ultimate RSI Indicator
Converted from Pine Script by LuxAlgo
"""

import pandas as pd
import numpy as np


class UltimateRSI:
    """
    Ultimate RSI indicator - An augmented version of RSI
    that uses price range instead of just price changes
    """

    def __init__(self, length=14, smooth=14, method='RMA', signal_method='EMA',
                 ob_level=80, os_level=20):
        """
        Initialize Ultimate RSI indicator

        Args:
            length: RSI calculation length (default 14)
            smooth: Signal line smoothing (default 14)
            method: Calculation method - 'EMA', 'SMA', 'RMA', 'TMA' (default 'RMA')
            signal_method: Signal line method (default 'EMA')
            ob_level: Overbought level (default 80)
            os_level: Oversold level (default 20)
        """
        self.length = length
        self.smooth = smooth
        self.method = method
        self.signal_method = signal_method
        self.ob_level = ob_level
        self.os_level = os_level

    def calculate(self, df):
        """
        Calculate Ultimate RSI

        Args:
            df: DataFrame with OHLC data

        Returns:
            dict: Dictionary containing Ultimate RSI and signal line
        """
        df = df.copy()

        # Calculate highest high and lowest low
        df['upper'] = df['high'].rolling(window=self.length).max()
        df['lower'] = df['low'].rolling(window=self.length).min()
        df['range'] = df['upper'] - df['lower']

        # Calculate previous values for comparison
        df['upper_prev'] = df['upper'].shift(1)
        df['lower_prev'] = df['lower'].shift(1)

        # Calculate price change
        df['price_change'] = df['close'].diff()

        # Calculate diff based on range changes (vectorized approach)
        df['diff'] = df['price_change'].copy()  # Default to price change

        # Where upper increased
        upper_increased = df['upper'] > df['upper_prev']
        df.loc[upper_increased, 'diff'] = df.loc[upper_increased, 'range']

        # Where lower decreased
        lower_decreased = df['lower'] < df['lower_prev']
        df.loc[lower_decreased, 'diff'] = -df.loc[lower_decreased, 'range']

        # Calculate numerator (smoothed diff)
        df['num'] = self._apply_smoothing(df['diff'], self.length, self.method)

        # Calculate denominator (smoothed absolute diff)
        df['den'] = self._apply_smoothing(df['diff'].abs(), self.length, self.method)

        # Calculate Ultimate RSI
        df['ultimate_rsi'] = (df['num'] / df['den']) * 50 + 50

        # Replace inf and nan
        df['ultimate_rsi'] = df['ultimate_rsi'].replace([np.inf, -np.inf], np.nan)
        df['ultimate_rsi'] = df['ultimate_rsi'].fillna(50)

        # Calculate signal line
        df['signal'] = self._apply_smoothing(df['ultimate_rsi'], self.smooth, self.signal_method)

        # Determine RSI state
        df['rsi_state'] = df['ultimate_rsi'].apply(
            lambda x: 'overbought' if x > self.ob_level else 'oversold' if x < self.os_level else 'neutral'
        )

        return {
            'ultimate_rsi': df['ultimate_rsi'].values,
            'signal': df['signal'].values,
            'rsi_state': df['rsi_state'].values,
            'ob_level': self.ob_level,
            'os_level': self.os_level
        }

    def _apply_smoothing(self, series, length, method):
        """Apply smoothing based on method"""
        if method == 'EMA':
            return series.ewm(span=length, adjust=False).mean()
        elif method == 'SMA':
            return series.rolling(window=length).mean()
        elif method == 'RMA':
            # RMA is the same as EMA with alpha = 1/length
            return series.ewm(alpha=1/length, adjust=False).mean()
        elif method == 'TMA':
            # Triangular Moving Average (SMA of SMA)
            sma1 = series.rolling(window=length).mean()
            return sma1.rolling(window=length).mean()
        else:
            return series.ewm(span=length, adjust=False).mean()

    def get_signals(self, df):
        """
        Get trading signals based on Ultimate RSI

        Args:
            df: DataFrame with OHLC data

        Returns:
            dict: Dictionary with buy/sell signals
        """
        results = self.calculate(df)

        ultimate_rsi = pd.Series(results['ultimate_rsi'])
        signal = pd.Series(results['signal'])

        # Crossover signals
        buy_signals = (ultimate_rsi > signal) & (ultimate_rsi.shift(1) <= signal.shift(1))
        sell_signals = (ultimate_rsi < signal) & (ultimate_rsi.shift(1) >= signal.shift(1))

        # Oversold/Overbought signals
        oversold_buy = (ultimate_rsi < self.os_level) & (ultimate_rsi > ultimate_rsi.shift(1))
        overbought_sell = (ultimate_rsi > self.ob_level) & (ultimate_rsi < ultimate_rsi.shift(1))

        return {
            'ultimate_rsi': results['ultimate_rsi'],
            'signal': results['signal'],
            'buy_signals': buy_signals.values,
            'sell_signals': sell_signals.values,
            'oversold_buy': oversold_buy.values,
            'overbought_sell': overbought_sell.values,
            'rsi_state': results['rsi_state']
        }
