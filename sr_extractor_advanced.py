"""
Advanced S/R Extractor - Comprehensive Multi-Source Support/Resistance
Extracts S/R levels from all 14 sources for Advanced signal system
"""

import streamlit as st
from typing import List, Dict, Optional, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class AdvancedSRExtractor:
    """Extract S/R levels from all 14 data sources with context-dependent strength scoring"""

    def __init__(self, current_price: float, atm_strike: int):
        self.current_price = current_price
        self.atm_strike = atm_strike
        self.support_levels = []
        self.resistance_levels = []

    def extract_all_sources(
        self,
        intraday_levels: List[Dict],
        market_depth: Dict,
        option_chain: Dict,
        volume_footprint_data: Dict,
        ultimate_rsi_data: Dict,
        om_indicator_data: Dict,
        money_flow_data: Dict,
        deltaflow_data: Dict,
        geometric_patterns: List[Dict],
        bos_choch_data: Dict,
        reversal_zones: List[Dict],
        liquidity_sentiment: Dict,
        dte: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract S/R from all 14 sources
        Returns: (support_levels, resistance_levels)
        """

        # Source 1: HTF Support/Resistance (75-90% strength)
        self._extract_htf_sr(intraday_levels)

        # Source 2: Market Depth S/R (85% strength)
        self._extract_depth_sr(market_depth)

        # Source 3: Fibonacci Levels (70-80% strength)
        self._extract_fibonacci_sr()

        # Source 4: Structural Levels (70% strength)
        self._extract_structural_sr()

        # Source 5: Volume Footprint HVN/POC (88% strength)
        self._extract_volume_footprint_sr(volume_footprint_data)

        # Source 6: Ultimate RSI Divergences (82% strength)
        self._extract_ultimate_rsi_sr(ultimate_rsi_data)

        # Source 7: OM Indicator Peaks/Troughs (80% strength)
        self._extract_om_indicator_sr(om_indicator_data)

        # Source 8: Money Flow Profile (85% strength)
        self._extract_money_flow_sr(money_flow_data)

        # Source 9: DeltaFlow Flip Zones (83% strength)
        self._extract_deltaflow_sr(deltaflow_data)

        # Source 10: Geometric Patterns (90-95% strength)
        self._extract_geometric_patterns_sr(geometric_patterns)

        # Source 11: BOS/CHOCH Levels (87% strength)
        self._extract_bos_choch_sr(bos_choch_data)

        # Source 12: Reversal Probability Zones (78% strength)
        self._extract_reversal_zones_sr(reversal_zones)

        # Source 13: Liquidity Sentiment Pools (81% strength)
        self._extract_liquidity_sentiment_sr(liquidity_sentiment)

        # Source 14: Option Chain OI Walls + Max Pain + GEX (Context-dependent)
        self._extract_option_chain_sr(option_chain, dte)

        # Sort by priority and distance from current price
        self.support_levels = sorted(
            self.support_levels,
            key=lambda x: (x['priority'], self.current_price - x['price'])
        )
        self.resistance_levels = sorted(
            self.resistance_levels,
            key=lambda x: (x['priority'], x['price'] - self.current_price)
        )

        return self.support_levels, self.resistance_levels

    def _extract_htf_sr(self, intraday_levels: List[Dict]):
        """Source 1: HTF S/R (75-90% strength based on timeframe)"""
        if not intraday_levels:
            return

        for level in intraday_levels:
            level_price = level['price']
            level_source = level.get('source', 'HTF')
            level_type = level.get('type', 'support')

            # Higher TF = higher strength
            if '30min' in level_source or '30m' in level_source:
                strength = 90
                priority = 1
            elif '15min' in level_source or '15m' in level_source:
                strength = 85
                priority = 1
            elif '5min' in level_source or '5m' in level_source:
                strength = 80
                priority = 2
            else:
                strength = 75
                priority = 2

            level_data = {
                'price': level_price,
                'upper': level_price + 5,
                'lower': level_price - 5,
                'type': f'HTF {level_source}',
                'source': level_source,
                'strength': strength,
                'priority': priority,
                'touches': level.get('touches', 1)
            }

            if level_type == 'support' or level_price < self.current_price:
                self.support_levels.append(level_data)
            else:
                self.resistance_levels.append(level_data)

    def _extract_depth_sr(self, market_depth: Dict):
        """Source 2: Market Depth S/R (85% strength)"""
        if not market_depth:
            return

        depth_support = market_depth.get('support_level', 0)
        depth_resistance = market_depth.get('resistance_level', 0)

        if depth_support > 0:
            self.support_levels.append({
                'price': depth_support,
                'upper': depth_support + 10,
                'lower': depth_support - 10,
                'type': 'Depth Support',
                'source': 'Option Chain Depth',
                'strength': 85,
                'priority': 2,
                'poc': market_depth.get('poc', depth_support)
            })

        if depth_resistance > 0:
            self.resistance_levels.append({
                'price': depth_resistance,
                'upper': depth_resistance + 10,
                'lower': depth_resistance - 10,
                'type': 'Depth Resistance',
                'source': 'Option Chain Depth',
                'strength': 85,
                'priority': 2,
                'poc': market_depth.get('poc', depth_resistance)
            })

    def _extract_fibonacci_sr(self):
        """Source 3: Fibonacci Levels (70-80% strength)"""
        if 'fibonacci_levels' not in st.session_state:
            return

        fib_levels = st.session_state.fibonacci_levels
        for fib in fib_levels:
            fib_price = fib.get('price', 0)
            fib_ratio = fib.get('ratio', 0)

            # Key ratios have higher strength
            if fib_ratio in [0.382, 0.5, 0.618, 0.786]:
                strength = 80
                priority = 3
            else:
                strength = 70
                priority = 4

            level_data = {
                'price': fib_price,
                'upper': fib_price + 5,
                'lower': fib_price - 5,
                'type': f'Fibonacci {fib_ratio:.3f}',
                'source': 'Fibonacci Retracement',
                'strength': strength,
                'priority': priority,
                'ratio': fib_ratio
            }

            if fib_price < self.current_price:
                self.support_levels.append(level_data)
            else:
                self.resistance_levels.append(level_data)

    def _extract_structural_sr(self):
        """Source 4: Structural Levels (70% strength)"""
        if 'key_levels' not in st.session_state:
            return

        key_levels = st.session_state.key_levels
        for level in key_levels:
            if level['type'] in ['ATM Strike', 'Max Pain']:
                continue  # Skip strikes

            level_data = {
                'price': level['price'],
                'upper': level['price'] + 10,
                'lower': level['price'] - 10,
                'type': level['type'],
                'source': level.get('source', 'Structural'),
                'strength': level.get('strength', 70),
                'priority': 4
            }

            if level['price'] < self.current_price:
                self.support_levels.append(level_data)
            else:
                self.resistance_levels.append(level_data)

    def _extract_volume_footprint_sr(self, volume_footprint_data: Dict):
        """Source 5: Volume Footprint HVN/POC (88% strength)"""
        if not volume_footprint_data:
            return

        # Extract HVN (High Volume Nodes)
        hvn_levels = volume_footprint_data.get('hvn_levels', [])
        for hvn in hvn_levels:
            price = hvn.get('price', 0)
            if price <= 0:
                continue

            level_data = {
                'price': price,
                'upper': price + 8,
                'lower': price - 8,
                'type': 'Volume HVN',
                'source': 'Volume Footprint',
                'strength': 88,
                'priority': 1,
                'volume': hvn.get('volume', 0),
                'imbalance': hvn.get('buy_sell_ratio', 1.0)
            }

            if price < self.current_price:
                self.support_levels.append(level_data)
            else:
                self.resistance_levels.append(level_data)

        # Extract POC (Point of Control)
        poc = volume_footprint_data.get('poc', 0)
        if poc > 0:
            level_data = {
                'price': poc,
                'upper': poc + 8,
                'lower': poc - 8,
                'type': 'Volume POC',
                'source': 'Volume Footprint',
                'strength': 88,
                'priority': 1,
                'volume': volume_footprint_data.get('poc_volume', 0)
            }

            if poc < self.current_price:
                self.support_levels.append(level_data)
            else:
                self.resistance_levels.append(level_data)

    def _extract_ultimate_rsi_sr(self, ultimate_rsi_data: Dict):
        """Source 6: Ultimate RSI Divergences (82% strength)"""
        if not ultimate_rsi_data:
            return

        # Bullish divergence = support
        bullish_divs = ultimate_rsi_data.get('bullish_divergences', [])
        for div in bullish_divs:
            price = div.get('price', 0)
            if price <= 0:
                continue

            self.support_levels.append({
                'price': price,
                'upper': price + 7,
                'lower': price - 7,
                'type': 'RSI Bullish Div',
                'source': 'Ultimate RSI',
                'strength': 82,
                'priority': 2,
                'rsi_value': div.get('rsi', 0),
                'divergence_type': div.get('type', 'regular')
            })

        # Bearish divergence = resistance
        bearish_divs = ultimate_rsi_data.get('bearish_divergences', [])
        for div in bearish_divs:
            price = div.get('price', 0)
            if price <= 0:
                continue

            self.resistance_levels.append({
                'price': price,
                'upper': price + 7,
                'lower': price - 7,
                'type': 'RSI Bearish Div',
                'source': 'Ultimate RSI',
                'strength': 82,
                'priority': 2,
                'rsi_value': div.get('rsi', 0),
                'divergence_type': div.get('type', 'regular')
            })

    def _extract_om_indicator_sr(self, om_indicator_data: Dict):
        """Source 7: OM Indicator Peaks/Troughs (80% strength)"""
        if not om_indicator_data:
            return

        # Troughs = support
        troughs = om_indicator_data.get('troughs', [])
        for trough in troughs:
            price = trough.get('price', 0)
            if price <= 0:
                continue

            self.support_levels.append({
                'price': price,
                'upper': price + 6,
                'lower': price - 6,
                'type': 'OM Trough',
                'source': 'OM Indicator',
                'strength': 80,
                'priority': 3,
                'om_value': trough.get('value', 0)
            })

        # Peaks = resistance
        peaks = om_indicator_data.get('peaks', [])
        for peak in peaks:
            price = peak.get('price', 0)
            if price <= 0:
                continue

            self.resistance_levels.append({
                'price': price,
                'upper': price + 6,
                'lower': price - 6,
                'type': 'OM Peak',
                'source': 'OM Indicator',
                'strength': 80,
                'priority': 3,
                'om_value': peak.get('value', 0)
            })

    def _extract_money_flow_sr(self, money_flow_data: Dict):
        """Source 8: Money Flow Profile POC/VAH/VAL (85% strength)"""
        if not money_flow_data:
            return

        # POC (Point of Control) - strongest
        poc = money_flow_data.get('poc', 0)
        if poc > 0:
            level_data = {
                'price': poc,
                'upper': poc + 8,
                'lower': poc - 8,
                'type': 'Money Flow POC',
                'source': 'Money Flow Profile',
                'strength': 85,
                'priority': 2,
                'volume': money_flow_data.get('poc_volume', 0)
            }

            if poc < self.current_price:
                self.support_levels.append(level_data)
            else:
                self.resistance_levels.append(level_data)

        # VAH (Value Area High) - resistance
        vah = money_flow_data.get('vah', 0)
        if vah > 0:
            self.resistance_levels.append({
                'price': vah,
                'upper': vah + 8,
                'lower': vah - 8,
                'type': 'Money Flow VAH',
                'source': 'Money Flow Profile',
                'strength': 83,
                'priority': 2
            })

        # VAL (Value Area Low) - support
        val = money_flow_data.get('val', 0)
        if val > 0:
            self.support_levels.append({
                'price': val,
                'upper': val + 8,
                'lower': val - 8,
                'type': 'Money Flow VAL',
                'source': 'Money Flow Profile',
                'strength': 83,
                'priority': 2
            })

    def _extract_deltaflow_sr(self, deltaflow_data: Dict):
        """Source 9: DeltaFlow Flip Zones (83% strength)"""
        if not deltaflow_data:
            return

        # Delta flip zones where cumulative delta changes direction
        flip_zones = deltaflow_data.get('flip_zones', [])
        for zone in flip_zones:
            price = zone.get('price', 0)
            delta_direction = zone.get('direction', 'neutral')

            if price <= 0:
                continue

            level_data = {
                'price': price,
                'upper': price + 7,
                'lower': price - 7,
                'type': f'Delta Flip ({delta_direction})',
                'source': 'DeltaFlow',
                'strength': 83,
                'priority': 2,
                'delta_value': zone.get('delta', 0)
            }

            # Bullish flip = support, Bearish flip = resistance
            if delta_direction == 'bullish' or price < self.current_price:
                self.support_levels.append(level_data)
            else:
                self.resistance_levels.append(level_data)

    def _extract_geometric_patterns_sr(self, geometric_patterns: List[Dict]):
        """Source 10: Geometric Patterns H&S/Triangles/Flags (90-95% strength)"""
        if not geometric_patterns:
            return

        for pattern in geometric_patterns:
            pattern_type = pattern.get('type', 'Unknown')

            # Head & Shoulders
            if pattern_type == 'Head & Shoulders':
                neckline = pattern.get('neckline', 0)
                if neckline > 0:
                    self.resistance_levels.append({
                        'price': neckline,
                        'upper': neckline + 5,
                        'lower': neckline - 5,
                        'type': 'H&S Neckline',
                        'source': 'Geometric Pattern',
                        'strength': 95,
                        'priority': 1,
                        'pattern_details': pattern,
                        'measured_move': pattern.get('target', 0)
                    })

            # Inverse Head & Shoulders
            elif pattern_type == 'Inverse H&S':
                neckline = pattern.get('neckline', 0)
                if neckline > 0:
                    self.support_levels.append({
                        'price': neckline,
                        'upper': neckline + 5,
                        'lower': neckline - 5,
                        'type': 'Inv H&S Neckline',
                        'source': 'Geometric Pattern',
                        'strength': 95,
                        'priority': 1,
                        'pattern_details': pattern,
                        'measured_move': pattern.get('target', 0)
                    })

            # Ascending/Descending Triangles
            elif 'Triangle' in pattern_type:
                breakout_level = pattern.get('breakout_level', 0)
                if breakout_level > 0:
                    level_data = {
                        'price': breakout_level,
                        'upper': breakout_level + 5,
                        'lower': breakout_level - 5,
                        'type': f'{pattern_type} Breakout',
                        'source': 'Geometric Pattern',
                        'strength': 92,
                        'priority': 1,
                        'pattern_details': pattern,
                        'measured_move': pattern.get('target', 0)
                    }

                    if 'Ascending' in pattern_type:
                        self.support_levels.append(level_data)
                    else:
                        self.resistance_levels.append(level_data)

            # Bull/Bear Flags
            elif 'Flag' in pattern_type:
                flag_boundary = pattern.get('flag_boundary', 0)
                if flag_boundary > 0:
                    level_data = {
                        'price': flag_boundary,
                        'upper': flag_boundary + 5,
                        'lower': flag_boundary - 5,
                        'type': f'{pattern_type} Boundary',
                        'source': 'Geometric Pattern',
                        'strength': 90,
                        'priority': 1,
                        'pattern_details': pattern,
                        'measured_move': pattern.get('target', 0)
                    }

                    if 'Bull' in pattern_type:
                        self.support_levels.append(level_data)
                    else:
                        self.resistance_levels.append(level_data)

    def _extract_bos_choch_sr(self, bos_choch_data: Dict):
        """Source 11: BOS/CHOCH Levels (87% strength)"""
        if not bos_choch_data:
            return

        # BOS (Break of Structure) levels
        bos_levels = bos_choch_data.get('bos_levels', [])
        for bos in bos_levels:
            price = bos.get('price', 0)
            direction = bos.get('direction', 'neutral')

            if price <= 0:
                continue

            level_data = {
                'price': price,
                'upper': price + 6,
                'lower': price - 6,
                'type': f'BOS ({direction})',
                'source': 'Break of Structure',
                'strength': 87,
                'priority': 2,
                'structure_type': 'BOS'
            }

            if direction == 'bullish':
                self.support_levels.append(level_data)
            else:
                self.resistance_levels.append(level_data)

        # CHOCH (Change of Character) levels
        choch_levels = bos_choch_data.get('choch_levels', [])
        for choch in choch_levels:
            price = choch.get('price', 0)
            direction = choch.get('direction', 'neutral')

            if price <= 0:
                continue

            level_data = {
                'price': price,
                'upper': price + 6,
                'lower': price - 6,
                'type': f'CHOCH ({direction})',
                'source': 'Change of Character',
                'strength': 85,
                'priority': 2,
                'structure_type': 'CHOCH'
            }

            if direction == 'bullish':
                self.support_levels.append(level_data)
            else:
                self.resistance_levels.append(level_data)

    def _extract_reversal_zones_sr(self, reversal_zones: List[Dict]):
        """Source 12: Reversal Probability Zones (78% strength)"""
        if not reversal_zones:
            return

        for zone in reversal_zones:
            price = zone.get('price', 0)
            zone_type = zone.get('type', 'neutral')
            probability = zone.get('probability', 0)

            if price <= 0:
                continue

            # Adjust strength by probability
            strength = int(78 * (probability / 100))

            level_data = {
                'price': price,
                'upper': price + 8,
                'lower': price - 8,
                'type': f'Reversal Zone ({probability}%)',
                'source': 'Reversal Probability',
                'strength': strength,
                'priority': 3,
                'probability': probability
            }

            if zone_type == 'bullish' or price < self.current_price:
                self.support_levels.append(level_data)
            else:
                self.resistance_levels.append(level_data)

    def _extract_liquidity_sentiment_sr(self, liquidity_sentiment: Dict):
        """Source 13: Liquidity Sentiment Pools (81% strength)"""
        if not liquidity_sentiment:
            return

        # Liquidity pools where large orders sit
        buy_pools = liquidity_sentiment.get('buy_pools', [])
        for pool in buy_pools:
            price = pool.get('price', 0)
            if price <= 0:
                continue

            self.support_levels.append({
                'price': price,
                'upper': price + 7,
                'lower': price - 7,
                'type': 'Buy Liquidity Pool',
                'source': 'Liquidity Sentiment',
                'strength': 81,
                'priority': 2,
                'liquidity_size': pool.get('size', 0)
            })

        sell_pools = liquidity_sentiment.get('sell_pools', [])
        for pool in sell_pools:
            price = pool.get('price', 0)
            if price <= 0:
                continue

            self.resistance_levels.append({
                'price': price,
                'upper': price + 7,
                'lower': price - 7,
                'type': 'Sell Liquidity Pool',
                'source': 'Liquidity Sentiment',
                'strength': 81,
                'priority': 2,
                'liquidity_size': pool.get('size', 0)
            })

    def _extract_option_chain_sr(self, option_chain: Dict, dte: int):
        """
        Source 14: Option Chain Analysis (Context-dependent strength)
        - OI Walls: 50-90% (based on buildup vs unwinding)
        - Max Pain: 60-92% (based on DTE)
        - GEX: 75-90% (based on concentration)
        """
        if not option_chain:
            return

        # Max Pain (strength varies by DTE)
        max_pain = option_chain.get('max_pain', 0)
        if max_pain > 0:
            if dte <= 2:
                max_pain_strength = 92
            elif dte <= 5:
                max_pain_strength = 80
            elif dte <= 10:
                max_pain_strength = 70
            else:
                max_pain_strength = 60

            level_data = {
                'price': max_pain,
                'upper': max_pain + 10,
                'lower': max_pain - 10,
                'type': f'Max Pain (DTE: {dte})',
                'source': 'Option Chain',
                'strength': max_pain_strength,
                'priority': 2 if dte <= 2 else 3,
                'dte': dte
            }

            if max_pain < self.current_price:
                self.support_levels.append(level_data)
            else:
                self.resistance_levels.append(level_data)

        # OI Walls (PUT walls = support, CALL walls = resistance)
        oi_walls = option_chain.get('oi_walls', {})

        # PUT OI walls (support)
        put_walls = oi_walls.get('put_walls', [])
        for wall in put_walls:
            strike = wall.get('strike', 0)
            oi = wall.get('oi', 0)
            is_fresh = wall.get('is_fresh', False)

            if strike <= 0:
                continue

            # Fresh buildup = stronger, unwinding = weaker
            if is_fresh and oi > 2000000:
                oi_strength = 90
                priority = 1
            elif is_fresh:
                oi_strength = 80
                priority = 2
            elif oi > 1000000:
                oi_strength = 70
                priority = 2
            else:
                oi_strength = 50
                priority = 3

            self.support_levels.append({
                'price': strike,
                'upper': strike + 15,
                'lower': strike - 15,
                'type': f'PUT OI Wall ({oi/1000000:.1f}M)',
                'source': 'Option Chain OI',
                'strength': oi_strength,
                'priority': priority,
                'oi': oi,
                'is_fresh': is_fresh
            })

        # CALL OI walls (resistance)
        call_walls = oi_walls.get('call_walls', [])
        for wall in call_walls:
            strike = wall.get('strike', 0)
            oi = wall.get('oi', 0)
            is_fresh = wall.get('is_fresh', False)

            if strike <= 0:
                continue

            # Fresh buildup = stronger, unwinding = weaker
            if is_fresh and oi > 2000000:
                oi_strength = 90
                priority = 1
            elif is_fresh:
                oi_strength = 80
                priority = 2
            elif oi > 1000000:
                oi_strength = 70
                priority = 2
            else:
                oi_strength = 50
                priority = 3

            self.resistance_levels.append({
                'price': strike,
                'upper': strike + 15,
                'lower': strike - 15,
                'type': f'CALL OI Wall ({oi/1000000:.1f}M)',
                'source': 'Option Chain OI',
                'strength': oi_strength,
                'priority': priority,
                'oi': oi,
                'is_fresh': is_fresh
            })

        # GEX levels (gamma exposure)
        gex_levels = option_chain.get('gex_levels', [])
        for gex in gex_levels:
            price = gex.get('price', 0)
            gex_value = gex.get('gex', 0)
            concentration = gex.get('concentration', 0)

            if price <= 0:
                continue

            # Higher concentration = higher strength
            if concentration > 80:
                gex_strength = 90
                priority = 1
            elif concentration > 60:
                gex_strength = 85
                priority = 2
            else:
                gex_strength = 75
                priority = 3

            level_data = {
                'price': price,
                'upper': price + 12,
                'lower': price - 12,
                'type': f'GEX Level ({concentration}%)',
                'source': 'Gamma Exposure',
                'strength': gex_strength,
                'priority': priority,
                'gex_value': gex_value,
                'concentration': concentration
            }

            if price < self.current_price:
                self.support_levels.append(level_data)
            else:
                self.resistance_levels.append(level_data)

    def find_confluence_clusters(self, threshold_distance: int = 15) -> Dict[str, List[Dict]]:
        """
        Find clusters where multiple S/R sources agree (within threshold_distance points)
        Returns: {'support_clusters': [...], 'resistance_clusters': [...]}
        """

        def cluster_levels(levels: List[Dict], threshold: int) -> List[Dict]:
            if not levels:
                return []

            clusters = []
            sorted_levels = sorted(levels, key=lambda x: x['price'])

            current_cluster = [sorted_levels[0]]

            for level in sorted_levels[1:]:
                if abs(level['price'] - current_cluster[-1]['price']) <= threshold:
                    current_cluster.append(level)
                else:
                    if len(current_cluster) >= 2:  # Only keep clusters with 2+ sources
                        clusters.append({
                            'price': sum(l['price'] for l in current_cluster) / len(current_cluster),
                            'sources': current_cluster,
                            'confluence_count': len(current_cluster),
                            'avg_strength': sum(l['strength'] for l in current_cluster) / len(current_cluster),
                            'source_list': [l['source'] for l in current_cluster]
                        })
                    current_cluster = [level]

            # Don't forget last cluster
            if len(current_cluster) >= 2:
                clusters.append({
                    'price': sum(l['price'] for l in current_cluster) / len(current_cluster),
                    'sources': current_cluster,
                    'confluence_count': len(current_cluster),
                    'avg_strength': sum(l['strength'] for l in current_cluster) / len(current_cluster),
                    'source_list': [l['source'] for l in current_cluster]
                })

            return sorted(clusters, key=lambda x: -x['confluence_count'])

        return {
            'support_clusters': cluster_levels(self.support_levels, threshold_distance),
            'resistance_clusters': cluster_levels(self.resistance_levels, threshold_distance)
        }
