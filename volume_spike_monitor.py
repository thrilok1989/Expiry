"""
Volume Spike Monitor - Real-time Volume Analysis for Dynamic Exits
Detects institutional volume spikes and absorption using Volume Footprint data
"""

from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VolumeSpikeMonitor:
    """
    Monitor volume patterns to detect:
    1. Volume spikes (institutional moves)
    2. Volume absorption (S/R defense)
    3. Buy/Sell imbalance
    """

    def __init__(self, position_type: str, lookback_periods: int = 20):
        """
        Initialize volume spike monitor

        Args:
            position_type: "LONG" or "SHORT"
            lookback_periods: Number of candles for average calculation (default: 20)
        """
        self.position_type = position_type
        self.lookback_periods = lookback_periods
        self.volume_history = []
        self.avg_volume = 0
        self.avg_buy_volume = 0
        self.avg_sell_volume = 0

        logger.info(f"Volume Monitor initialized: {position_type} position, lookback: {lookback_periods}")

    def update_volume_baseline(self, df: pd.DataFrame):
        """
        Calculate rolling average volume baseline from historical data

        Args:
            df: DataFrame with columns: Volume, BuyVolume, SellVolume
        """
        try:
            if len(df) >= self.lookback_periods:
                self.avg_volume = df['Volume'].rolling(window=self.lookback_periods).mean().iloc[-1]

                if 'BuyVolume' in df.columns:
                    self.avg_buy_volume = df['BuyVolume'].rolling(window=self.lookback_periods).mean().iloc[-1]
                if 'SellVolume' in df.columns:
                    self.avg_sell_volume = df['SellVolume'].rolling(window=self.lookback_periods).mean().iloc[-1]
            else:
                self.avg_volume = df['Volume'].mean()
                if 'BuyVolume' in df.columns:
                    self.avg_buy_volume = df['BuyVolume'].mean()
                if 'SellVolume' in df.columns:
                    self.avg_sell_volume = df['SellVolume'].mean()

            logger.debug(f"Volume baseline updated: Avg Volume={self.avg_volume:.0f}, Buy={self.avg_buy_volume:.0f}, Sell={self.avg_sell_volume:.0f}")

        except Exception as e:
            logger.error(f"Error updating volume baseline: {e}")
            self.avg_volume = 50_000  # Default fallback

    def check_volume_spike(self, current_candle: Dict) -> Dict:
        """
        Check if current candle has abnormal volume spike

        Args:
            current_candle: {
                'volume': int,
                'buy_volume': int,
                'sell_volume': int,
                'delta': int (optional),
                'timestamp': datetime,
                'price': float,
                'high': float,
                'low': float
            }

        Returns:
            {
                'spike_detected': bool,
                'volume_ratio': float,
                'action': str,
                'severity': str,
                'reason': str,
                'alert_priority': str,
                'buy_volume': int,
                'sell_volume': int,
                'delta': int,
                'buy_pct': float,
                'sell_pct': float
            }
        """
        volume = current_candle.get('volume', 0)
        buy_volume = current_candle.get('buy_volume', 0)
        sell_volume = current_candle.get('sell_volume', 0)
        delta = current_candle.get('delta', buy_volume - sell_volume)
        price = current_candle.get('price', 0)

        # Store in history
        self.volume_history.append({
            'timestamp': current_candle.get('timestamp', datetime.now()),
            'volume': volume,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'delta': delta,
            'price': price
        })

        # Calculate volume ratio
        if self.avg_volume > 0:
            volume_ratio = volume / self.avg_volume
        else:
            volume_ratio = 1.0

        # Calculate buy/sell percentages
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            buy_pct = (buy_volume / total_volume) * 100
            sell_pct = (sell_volume / total_volume) * 100
        else:
            buy_pct = 50
            sell_pct = 50

        # Check volume spike magnitude
        spike_check = self._check_spike_magnitude(volume_ratio)

        # Check buy/sell imbalance
        imbalance_check = self._check_volume_imbalance(buy_volume, sell_volume, buy_pct, sell_pct)

        # Combine checks - use most severe
        if spike_check['action'] == 'EXIT_ALL' or imbalance_check['action'] == 'EXIT_ALL':
            return {
                'spike_detected': True,
                'volume_ratio': volume_ratio,
                'action': 'EXIT_ALL',
                'severity': 'CRITICAL',
                'reason': f"{spike_check['reason']} | {imbalance_check['reason']}",
                'alert_priority': 'CRITICAL',
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'delta': delta,
                'buy_pct': buy_pct,
                'sell_pct': sell_pct
            }

        elif spike_check['action'] == 'EXIT_PARTIAL' or imbalance_check['action'] == 'EXIT_PARTIAL':
            return {
                'spike_detected': True,
                'volume_ratio': volume_ratio,
                'action': 'EXIT_PARTIAL',
                'severity': 'HIGH',
                'reason': f"{spike_check['reason']} | {imbalance_check['reason']}",
                'alert_priority': 'HIGH',
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'delta': delta,
                'buy_pct': buy_pct,
                'sell_pct': sell_pct
            }

        elif spike_check['action'] == 'TIGHTEN_SL' or imbalance_check['action'] == 'TIGHTEN_SL':
            return {
                'spike_detected': True,
                'volume_ratio': volume_ratio,
                'action': 'TIGHTEN_SL',
                'severity': 'MEDIUM',
                'reason': f"{spike_check['reason']} | {imbalance_check['reason']}",
                'alert_priority': 'MEDIUM',
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'delta': delta,
                'buy_pct': buy_pct,
                'sell_pct': sell_pct
            }

        else:
            return {
                'spike_detected': False,
                'volume_ratio': volume_ratio,
                'action': 'HOLD',
                'severity': 'NORMAL',
                'reason': 'Volume normal',
                'alert_priority': 'LOW',
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'delta': delta,
                'buy_pct': buy_pct,
                'sell_pct': sell_pct
            }

    def _check_spike_magnitude(self, volume_ratio: float) -> Dict:
        """
        Check if volume spike magnitude is significant

        Args:
            volume_ratio: Current volume / average volume

        Returns:
            Action and reason
        """
        if volume_ratio >= 5.0:
            return {
                'action': 'EXIT_ALL',
                'reason': f'MASSIVE volume spike ({volume_ratio:.1f}x avg)'
            }
        elif volume_ratio >= 3.0:
            return {
                'action': 'EXIT_PARTIAL',
                'reason': f'Strong volume spike ({volume_ratio:.1f}x avg)'
            }
        elif volume_ratio >= 2.0:
            return {
                'action': 'TIGHTEN_SL',
                'reason': f'Volume spike ({volume_ratio:.1f}x avg)'
            }
        else:
            return {
                'action': 'HOLD',
                'reason': f'Volume normal ({volume_ratio:.1f}x avg)'
            }

    def _check_volume_imbalance(self, buy_volume: int, sell_volume: int,
                                buy_pct: float, sell_pct: float) -> Dict:
        """
        Check if buy/sell volume is heavily imbalanced

        Args:
            buy_volume: Buying volume
            sell_volume: Selling volume
            buy_pct: Buy percentage
            sell_pct: Sell percentage

        Returns:
            Action and reason
        """
        delta = buy_volume - sell_volume

        # LONG position - watch for heavy selling
        if self.position_type == "LONG":
            if sell_pct >= 75:
                return {
                    'action': 'EXIT_ALL',
                    'reason': f'CRITICAL: {sell_pct:.0f}% SELL volume (Δ {delta:,})'
                }
            elif sell_pct >= 65:
                return {
                    'action': 'EXIT_PARTIAL',
                    'reason': f'Heavy selling: {sell_pct:.0f}% (Δ {delta:,})'
                }
            elif sell_pct >= 55:
                return {
                    'action': 'TIGHTEN_SL',
                    'reason': f'Selling pressure: {sell_pct:.0f}%'
                }

        # SHORT position - watch for heavy buying
        elif self.position_type == "SHORT":
            if buy_pct >= 75:
                return {
                    'action': 'EXIT_ALL',
                    'reason': f'CRITICAL: {buy_pct:.0f}% BUY volume (Δ {delta:,})'
                }
            elif buy_pct >= 65:
                return {
                    'action': 'EXIT_PARTIAL',
                    'reason': f'Heavy buying: {buy_pct:.0f}% (Δ {delta:,})'
                }
            elif buy_pct >= 55:
                return {
                    'action': 'TIGHTEN_SL',
                    'reason': f'Buying pressure: {buy_pct:.0f}%'
                }

        return {
            'action': 'HOLD',
            'reason': f'Balanced (Buy: {buy_pct:.0f}%, Sell: {sell_pct:.0f}%)'
        }

    def detect_absorption(self, recent_candles: List[Dict]) -> Dict:
        """
        Detect volume absorption (high volume but minimal price movement)
        Indicates strong S/R defense by institutional players

        Args:
            recent_candles: Last 3-5 candles with volume and price data

        Returns:
            {
                'absorption_detected': bool,
                'total_volume': int,
                'price_change_pct': float,
                'action': str,
                'reason': str,
                'alert_priority': str
            }
        """
        if len(recent_candles) < 3:
            return {
                'absorption_detected': False,
                'action': 'HOLD',
                'reason': 'Insufficient data for absorption check'
            }

        # Calculate total volume and price change
        total_volume = sum(c.get('volume', 0) for c in recent_candles)
        price_start = recent_candles[0].get('price', 0)
        price_end = recent_candles[-1].get('price', 0)

        if price_start == 0:
            return {'absorption_detected': False}

        price_change_pct = abs((price_end - price_start) / price_start) * 100

        # Expected volume for this many candles
        expected_volume = self.avg_volume * len(recent_candles)

        # High volume but minimal price movement = absorption
        if total_volume > (expected_volume * 3) and price_change_pct < 0.3:
            # Determine what this means for our position
            if self.position_type == "LONG" and price_end >= price_start:
                # LONG position hitting resistance absorption
                return {
                    'absorption_detected': True,
                    'total_volume': total_volume,
                    'price_change_pct': price_change_pct,
                    'action': 'EXIT_PARTIAL',
                    'reason': f'Resistance absorption: {total_volume:,} volume, only {price_change_pct:.2f}% move',
                    'alert_priority': 'HIGH'
                }
            elif self.position_type == "SHORT" and price_end <= price_start:
                # SHORT position hitting support absorption
                return {
                    'absorption_detected': True,
                    'total_volume': total_volume,
                    'price_change_pct': price_change_pct,
                    'action': 'EXIT_PARTIAL',
                    'reason': f'Support absorption: {total_volume:,} volume, only {price_change_pct:.2f}% move',
                    'alert_priority': 'HIGH'
                }

        return {
            'absorption_detected': False,
            'action': 'HOLD',
            'reason': 'No absorption detected'
        }

    def get_volume_trend(self, lookback_candles: int = 10) -> Dict:
        """
        Analyze volume trend over recent history

        Args:
            lookback_candles: Number of candles to analyze

        Returns:
            {
                'trend': str,
                'avg_volume_ratio': float,
                'increasing_count': int
            }
        """
        if len(self.volume_history) < lookback_candles:
            return {
                'trend': 'STABLE',
                'avg_volume_ratio': 1.0,
                'increasing_count': 0
            }

        recent = self.volume_history[-lookback_candles:]

        # Calculate volume ratios
        if self.avg_volume > 0:
            ratios = [h['volume'] / self.avg_volume for h in recent]
            avg_ratio = sum(ratios) / len(ratios)
            increasing_count = sum(1 for r in ratios if r > 1.5)

            if avg_ratio > 2.0:
                trend = 'SURGING'
            elif avg_ratio > 1.5:
                trend = 'INCREASING'
            elif avg_ratio < 0.7:
                trend = 'DECLINING'
            else:
                trend = 'STABLE'

            return {
                'trend': trend,
                'avg_volume_ratio': avg_ratio,
                'increasing_count': increasing_count
            }

        return {
            'trend': 'UNKNOWN',
            'avg_volume_ratio': 1.0,
            'increasing_count': 0
        }
