"""
Real-Time HTF Volume Footprint Indicator
Converted from Pine Script by BigBeluga
"""

import pandas as pd
import numpy as np


class HTFVolumeFootprint:
    """
    Higher Time Frame Volume Footprint indicator that displays
    volume distribution across price levels
    """

    def __init__(self, bins=20, timeframe='W', dynamic_poc=False):
        """
        Initialize HTF Volume Footprint indicator

        Args:
            bins: Number of bins for volume distribution
            timeframe: Timeframe for analysis ('D', 'W', '2W', 'M')
            dynamic_poc: Show dynamic Point of Control
        """
        self.bins = bins
        self.timeframe = timeframe
        self.dynamic_poc = dynamic_poc

    def calculate(self, df):
        """
        Calculate Volume Footprint

        Args:
            df: DataFrame with OHLCV data (1-minute or higher)

        Returns:
            dict: Dictionary containing volume footprint data
        """
        df = df.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif 'datetime' in df.columns:
                df = df.set_index('datetime')

        # Resample to target timeframe
        df_resampled = self._resample_to_htf(df)

        # Calculate volume footprint for each period
        footprints = []
        historical_pocs = []  # Track historical POC lines for completed periods

        current_time = df.index[-1]

        for i, period_start in enumerate(df_resampled.index):
            # Ensure period_start is a Timestamp
            period_start = pd.Timestamp(period_start)

            # Get data for this period
            if self.timeframe == 'W':
                period_end = period_start + pd.Timedelta(weeks=1)
            elif self.timeframe == '2W':
                period_end = period_start + pd.Timedelta(weeks=2)
            elif self.timeframe == 'M':
                period_end = period_start + pd.DateOffset(months=1)
            elif self.timeframe == 'D':
                period_end = period_start + pd.Timedelta(days=1)
            else:
                period_end = period_start + pd.Timedelta(days=1)

            period_data = df[(df.index >= period_start) & (df.index < period_end)]

            if len(period_data) > 0:
                footprint = self._calculate_period_footprint(period_data, period_start)
                footprint['period_end'] = period_end
                footprint['is_current'] = (current_time >= period_start and current_time < period_end)
                footprints.append(footprint)

                # Add to historical POCs if this is a completed period (not current)
                if not footprint['is_current']:
                    historical_pocs.append({
                        'period_start': period_start,
                        'period_end': min(period_end, current_time),
                        'poc_price': footprint['poc']
                    })

        # Get the most recent footprint
        current_footprint = footprints[-1] if len(footprints) > 0 else None

        return {
            'footprints': footprints,
            'current_footprint': current_footprint,
            'historical_pocs': historical_pocs,
            'timeframe': self.timeframe,
            'dynamic_poc': self.dynamic_poc
        }

    def _resample_to_htf(self, df):
        """Resample dataframe to higher timeframe"""
        timeframe_map = {
            'D': 'D',
            '2D': '2D',
            '3D': '3D',
            '4D': '4D',
            '5D': '5D',
            'W': 'W',
            '2W': '2W',
            '3W': '3W',
            'M': 'M',
            '2M': '2M',
            '3M': '3M'
        }

        resample_freq = timeframe_map.get(self.timeframe, 'W')

        df_resampled = df.resample(resample_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return df_resampled

    def _calculate_period_footprint(self, period_data, period_start):
        """Calculate volume footprint for a specific period"""
        # Get high and low for the period
        period_high = period_data['high'].max()
        period_low = period_data['low'].min()
        period_range = period_high - period_low

        if period_range == 0:
            return None

        # Calculate step size
        step = period_range / self.bins

        # Initialize volume distribution
        volume_distribution = np.zeros(self.bins)

        # Normalize volume by standard deviation
        vol_std = period_data['volume'].std()
        if vol_std == 0:
            vol_std = 1

        # Distribute volume across bins
        for idx, row in period_data.iterrows():
            close_price = row['close']
            volume_val = row['volume'] / vol_std

            # Find which bin this price belongs to
            bin_idx = int((close_price - period_low) / step)
            bin_idx = min(bin_idx, self.bins - 1)  # Ensure within bounds

            if bin_idx >= 0:
                volume_distribution[bin_idx] += volume_val

        # Find Point of Control (POC) - bin with highest volume
        poc_bin = np.argmax(volume_distribution)
        poc_price = period_low + (poc_bin + 0.5) * step

        # Calculate value area (70% of volume)
        total_volume = np.sum(volume_distribution)
        value_area_volume = total_volume * 0.7

        # Find value area high and low
        sorted_bins = np.argsort(volume_distribution)[::-1]
        cumulative_volume = 0
        value_area_bins = []

        for bin_idx in sorted_bins:
            cumulative_volume += volume_distribution[bin_idx]
            value_area_bins.append(bin_idx)
            if cumulative_volume >= value_area_volume:
                break

        value_area_low = period_low + min(value_area_bins) * step
        value_area_high = period_low + (max(value_area_bins) + 1) * step

        # Create bin data
        bins_data = []
        for i in range(self.bins):
            bin_lower = period_low + i * step
            bin_upper = bin_lower + step
            bins_data.append({
                'lower': bin_lower,
                'upper': bin_upper,
                'mid': (bin_lower + bin_upper) / 2,
                'volume': volume_distribution[i],
                'is_poc': (i == poc_bin)
            })

        return {
            'period_start': period_start,
            'period_high': period_high,
            'period_low': period_low,
            'period_mid': (period_high + period_low) / 2,
            'poc': poc_price,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'bins': bins_data,
            'total_volume': total_volume
        }
