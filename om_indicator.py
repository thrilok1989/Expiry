"""
OM (Order Flow & Momentum) Indicator
Combines multiple advanced indicators: VOB, HVP, Delta, VWAP, VIDYA, LTP Trap
Converted from Pine Script v6
"""

import pandas as pd
import numpy as np


class OMIndicator:
    """
    OM (Order Flow & Momentum) Indicator
    Comprehensive indicator combining volume, momentum, and order flow analysis
    """

    def __init__(self,
                 vob_sensitivity=5,
                 hvp_left_bars=15,
                 hvp_right_bars=15,
                 hvp_volume_filter=2.0,
                 delta_length=10,
                 delta_threshold=1.5,
                 vidya_length=10,
                 vidya_momentum=20,
                 band_distance=2.0,
                 show_hvp=True,
                 color_bull='#26ba9f',
                 color_bear='#ba2646'):
        """
        Initialize OM Indicator

        Args:
            vob_sensitivity: VOB detection sensitivity
            hvp_left_bars: HVP pivot left bars
            hvp_right_bars: HVP pivot right bars
            hvp_volume_filter: HVP volume filter multiplier
            delta_length: Delta MA length
            delta_threshold: Delta spike threshold
            vidya_length: VIDYA length
            vidya_momentum: VIDYA momentum period
            band_distance: VIDYA band distance factor
            show_hvp: Show HVP zones
            color_bull: Bullish zone color
            color_bear: Bearish zone color
        """
        self.vob_sensitivity = vob_sensitivity
        self.hvp_left_bars = hvp_left_bars
        self.hvp_right_bars = hvp_right_bars
        self.hvp_volume_filter = hvp_volume_filter
        self.delta_length = delta_length
        self.delta_threshold = delta_threshold
        self.vidya_length = vidya_length
        self.vidya_momentum = vidya_momentum
        self.band_distance = band_distance
        self.show_hvp = show_hvp
        self.color_bull = color_bull
        self.color_bear = color_bear

    def calculate(self, df):
        """
        Calculate all OM indicator components

        Args:
            df: DataFrame with OHLCV data

        Returns:
            dict: All indicator results
        """
        df = df.copy()

        # Calculate VWAP
        vwap = self._calculate_vwap(df)

        # Calculate VOB (Volume Order Blocks)
        vob_data = self._calculate_vob(df)

        # Calculate HVP (High Volume Pivots)
        hvp_data = self._calculate_hvp(df)

        # Calculate Delta Module
        delta_data = self._calculate_delta(df, vwap)

        # Calculate VIDYA
        vidya_data = self._calculate_vidya(df)

        # Calculate LTP Trap signals
        ltp_trap = self._calculate_ltp_trap(df, vwap, delta_data)

        return {
            'vwap': vwap,
            'vob_data': vob_data,
            'hvp_data': hvp_data,
            'delta_data': delta_data,
            'vidya_data': vidya_data,
            'ltp_trap': ltp_trap
        }

    def _calculate_vwap(self, df):
        """Calculate VWAP (Volume Weighted Average Price)"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tpv = (typical_price * df['volume']).cumsum()
        cumulative_volume = df['volume'].cumsum()
        vwap = cumulative_tpv / cumulative_volume
        return vwap.values

    def _calculate_vob(self, df):
        """Calculate Volume Order Blocks (Stable Version)"""
        length1 = self.vob_sensitivity
        length2 = length1 + 13

        # Calculate EMAs
        ema1 = df['close'].ewm(span=length1, adjust=False).mean()
        ema2 = ema1.ewm(span=length2, adjust=False).mean()

        # Detect crossovers
        cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        cross_dn = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))

        # Calculate ATR
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = tr.rolling(window=200).mean() * 3

        # Detect order blocks
        bullish_blocks = []
        bearish_blocks = []

        # Process bullish blocks
        for i in range(len(df)):
            if cross_up.iloc[i] and i >= length2:
                lookback = df.iloc[i-length2:i]
                if len(lookback) > 0:
                    lowest = lookback['low'].min()
                    lowest_idx = lookback['low'].idxmin()

                    # Get candle at lowest point
                    base = min(df.loc[lowest_idx, 'open'], df.loc[lowest_idx, 'close'])

                    # Adjust if too close
                    if (base - lowest) < atr.iloc[i] * 0.5:
                        base = lowest + atr.iloc[i] * 0.5

                    bullish_blocks.append({
                        'index': i,
                        'upper': base,
                        'lower': lowest,
                        'start_time': df.index[i]
                    })

        # Process bearish blocks
        for i in range(len(df)):
            if cross_dn.iloc[i] and i >= length2:
                lookback = df.iloc[i-length2:i]
                if len(lookback) > 0:
                    highest = lookback['high'].max()
                    highest_idx = lookback['high'].idxmax()

                    # Get candle at highest point
                    base = max(df.loc[highest_idx, 'open'], df.loc[highest_idx, 'close'])

                    # Adjust if too close
                    if (highest - base) < atr.iloc[i] * 0.5:
                        base = highest - atr.iloc[i] * 0.5

                    bearish_blocks.append({
                        'index': i,
                        'upper': highest,
                        'lower': base,
                        'start_time': df.index[i]
                    })

        return {
            'ema1': ema1.values,
            'ema2': ema2.values,
            'bullish_blocks': bullish_blocks[-10:],  # Keep last 10
            'bearish_blocks': bearish_blocks[-10:]   # Keep last 10
        }

    def _calculate_hvp(self, df):
        """Calculate High Volume Pivots"""
        if not self.show_hvp:
            return {'pivot_highs': [], 'pivot_lows': []}

        # Calculate pivot highs and lows
        pivot_highs = []
        pivot_lows = []

        left = self.hvp_left_bars
        right = self.hvp_right_bars
        window = left + right

        # Calculate volume metrics
        volume_sum = df['volume'].rolling(window=window).sum()
        ref_vol = df['volume'].rolling(window=1000, min_periods=1).quantile(0.95)
        norm_vol = volume_sum / ref_vol * 5

        for i in range(left, len(df) - right):
            # Check for pivot high
            is_pivot_high = True
            for j in range(-left, right + 1):
                if j != 0 and df['high'].iloc[i + j] >= df['high'].iloc[i]:
                    is_pivot_high = False
                    break

            if is_pivot_high and norm_vol.iloc[i] > self.hvp_volume_filter:
                pivot_highs.append({
                    'index': i,
                    'price': df['high'].iloc[i],
                    'time': df.index[i],
                    'volume': norm_vol.iloc[i]
                })

            # Check for pivot low
            is_pivot_low = True
            for j in range(-left, right + 1):
                if j != 0 and df['low'].iloc[i + j] <= df['low'].iloc[i]:
                    is_pivot_low = False
                    break

            if is_pivot_low and norm_vol.iloc[i] > self.hvp_volume_filter:
                pivot_lows.append({
                    'index': i,
                    'price': df['low'].iloc[i],
                    'time': df.index[i],
                    'volume': norm_vol.iloc[i]
                })

        return {
            'pivot_highs': pivot_highs[-10:],  # Keep last 10
            'pivot_lows': pivot_lows[-10:]     # Keep last 10
        }

    def _calculate_delta(self, df, vwap):
        """Calculate Delta Module (Buy/Sell Pressure)"""
        delta = df['close'] - df['open']
        delta_ma = delta.ewm(span=self.delta_length, adjust=False).mean()

        buy_spike = delta_ma > self.delta_threshold
        sell_spike = delta_ma < -self.delta_threshold

        return {
            'delta': delta.values,
            'delta_ma': delta_ma.values,
            'buy_spike': buy_spike.values,
            'sell_spike': sell_spike.values
        }

    def _calculate_vidya(self, df):
        """Calculate VIDYA (Variable Index Dynamic Average)"""
        source = df['close']

        # Calculate CMO (Chande Momentum Oscillator)
        def vidya_calc(src, length, momentum):
            m = src.diff()
            p = m.where(m >= 0, 0).rolling(window=momentum).sum()
            n = (-m).where(m < 0, 0).rolling(window=momentum).sum()

            abs_cmo = abs(100 * (p - n) / (p + n))
            alpha = 2 / (length + 1)

            # Calculate VIDYA
            vidya = pd.Series(index=src.index, dtype=float)
            vidya.iloc[0] = src.iloc[0]

            for i in range(1, len(src)):
                if pd.isna(abs_cmo.iloc[i]):
                    vidya.iloc[i] = vidya.iloc[i-1]
                else:
                    vidya.iloc[i] = (alpha * abs_cmo.iloc[i] / 100 * src.iloc[i] +
                                    (1 - alpha * abs_cmo.iloc[i] / 100) * vidya.iloc[i-1])

            # Smooth with SMA
            return vidya.rolling(window=15, min_periods=1).mean()

        vidya_val = vidya_calc(source, self.vidya_length, self.vidya_momentum)

        # Calculate ATR for bands
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr_val = tr.rolling(window=200).mean()

        upper_band = vidya_val + atr_val * self.band_distance
        lower_band = vidya_val - atr_val * self.band_distance

        # Determine trend
        is_trend_up = pd.Series(False, index=df.index)
        for i in range(1, len(df)):
            if source.iloc[i] > upper_band.iloc[i]:
                is_trend_up.iloc[i] = True
            elif source.iloc[i] < lower_band.iloc[i]:
                is_trend_up.iloc[i] = False
            else:
                is_trend_up.iloc[i] = is_trend_up.iloc[i-1]

        # Calculate smoothed line
        smoothed = pd.Series(index=df.index, dtype=float)
        for i in range(len(df)):
            if is_trend_up.iloc[i]:
                smoothed.iloc[i] = lower_band.iloc[i]
            else:
                smoothed.iloc[i] = upper_band.iloc[i]

        # Detect trend changes
        trend_cross_up = (~is_trend_up.shift(1).fillna(False)) & is_trend_up
        trend_cross_down = is_trend_up.shift(1).fillna(False) & (~is_trend_up)

        # Calculate volume delta
        up_volume = []
        down_volume = []
        delta_volume_pct = []

        cumulative_up = 0
        cumulative_down = 0

        for i in range(len(df)):
            if i > 0 and (trend_cross_up.iloc[i] or trend_cross_down.iloc[i]):
                cumulative_up = 0
                cumulative_down = 0

            if df['close'].iloc[i] > df['open'].iloc[i]:
                cumulative_up += df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['open'].iloc[i]:
                cumulative_down += df['volume'].iloc[i]

            up_volume.append(cumulative_up)
            down_volume.append(cumulative_down)

            avg_vol = (cumulative_up + cumulative_down) / 2
            if avg_vol > 0:
                delta_pct = (cumulative_up - cumulative_down) / avg_vol * 100
            else:
                delta_pct = 0
            delta_volume_pct.append(delta_pct)

        return {
            'vidya': vidya_val.values,
            'upper_band': upper_band.values,
            'lower_band': lower_band.values,
            'smoothed': smoothed.values,
            'is_trend_up': is_trend_up.values,
            'trend_cross_up': trend_cross_up.values,
            'trend_cross_down': trend_cross_down.values,
            'up_volume': np.array(up_volume),
            'down_volume': np.array(down_volume),
            'delta_volume_pct': np.array(delta_volume_pct)
        }

    def _calculate_ltp_trap(self, df, vwap, delta_data):
        """Calculate LTP Trap signals"""
        delta_ma = delta_data['delta_ma']

        ltp_trap_buy = (
            (df['close'] < df['open']) &
            (df['close'] > vwap) &
            (delta_ma > self.delta_threshold)
        )

        ltp_trap_sell = (
            (df['close'] > df['open']) &
            (df['close'] < vwap) &
            (delta_ma < -self.delta_threshold)
        )

        return {
            'ltp_trap_buy': ltp_trap_buy.values,
            'ltp_trap_sell': ltp_trap_sell.values
        }
