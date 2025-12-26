"""
Smart Target Calculator - Confluence-Based Targets
Based on OI walls, GEX, Fibonacci, geometric patterns, market depth, and all S/R sources
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SmartTarget:
    """Smart target with confluence data"""
    price: float
    distance_points: float
    confluence_count: int
    sources: list
    confidence: int  # 1-100%
    reasoning: str


class SmartTargetCalculator:
    """Calculate targets based on confluence from all S/R sources"""

    def __init__(
        self,
        entry_price: float,
        current_price: float,
        position_type: str  # "LONG" or "SHORT"
    ):
        self.entry_price = entry_price
        self.current_price = current_price
        self.position_type = position_type

    def calculate_smart_targets(
        self,
        resistance_levels: List[Dict],  # For LONG
        support_levels: List[Dict],  # For SHORT
        pattern_data: Optional[Dict] = None,
        option_chain: Optional[Dict] = None,
        confluence_clusters: Optional[List[Dict]] = None
    ) -> List[SmartTarget]:
        """
        Calculate multiple targets based on confluence
        Returns: List of SmartTarget objects (T1, T2, T3)
        """

        target_candidates = []

        # 1. Pattern measured move (highest priority)
        if pattern_data:
            pattern_targets = self._get_pattern_targets(pattern_data)
            target_candidates.extend(pattern_targets)

        # 2. Option chain levels (OI walls, Max Pain, GEX)
        if option_chain:
            oi_targets = self._get_option_chain_targets(option_chain)
            target_candidates.extend(oi_targets)

        # 3. Confluence clusters (multiple S/R sources agreeing)
        if confluence_clusters:
            cluster_targets = self._get_confluence_targets(confluence_clusters)
            target_candidates.extend(cluster_targets)

        # 4. Individual S/R levels (if no confluence)
        if self.position_type == "LONG":
            sr_targets = self._get_sr_targets(resistance_levels, "resistance")
        else:
            sr_targets = self._get_sr_targets(support_levels, "support")

        target_candidates.extend(sr_targets)

        # Remove duplicates (within 10 points)
        target_candidates = self._deduplicate_targets(target_candidates)

        # Filter targets in the right direction
        if self.position_type == "LONG":
            valid_targets = [t for t in target_candidates if t['price'] > self.current_price]
        else:
            valid_targets = [t for t in target_candidates if t['price'] < self.current_price]

        # Sort by distance and select top 3
        valid_targets = sorted(valid_targets, key=lambda x: abs(x['price'] - self.current_price))

        # Create SmartTarget objects for T1, T2, T3
        smart_targets = []

        for i, target in enumerate(valid_targets[:3]):
            distance = abs(target['price'] - self.entry_price)
            risk_reward = distance / 50  # Assuming avg SL of 50 pts

            target_num = i + 1
            reasoning = f"T{target_num}: ₹{target['price']:,.0f} (+{distance:.0f} pts, R:R {risk_reward:.1f}:1)\n"
            reasoning += f"Confluence: {target['confluence_count']} sources\n"
            reasoning += f"Sources: {', '.join(target['sources'][:3])}"
            if len(target['sources']) > 3:
                reasoning += f" +{len(target['sources']) - 3} more"

            smart_targets.append(SmartTarget(
                price=target['price'],
                distance_points=distance,
                confluence_count=target['confluence_count'],
                sources=target['sources'],
                confidence=target['confidence'],
                reasoning=reasoning
            ))

        # Ensure we have at least 3 targets (fill with simple R:R if needed)
        while len(smart_targets) < 3:
            fallback_target = self._get_fallback_target(len(smart_targets) + 1, smart_targets)
            if fallback_target:
                smart_targets.append(fallback_target)
            else:
                break

        return smart_targets

    def _get_pattern_targets(self, pattern_data: Dict) -> List[Dict]:
        """Get targets from geometric pattern measured moves"""
        targets = []
        pattern_type = pattern_data.get('type', '')

        # Measured move target
        measured_move = pattern_data.get('target', 0)
        if measured_move > 0:
            targets.append({
                'price': measured_move,
                'confluence_count': 5,  # Patterns are high priority
                'sources': [f'{pattern_type} Measured Move'],
                'confidence': 95,
                'priority': 1
            })

        # Extension targets (1.618, 2.618 Fibonacci extensions)
        if pattern_type in ['Head & Shoulders', 'Inverse H&S']:
            neckline = pattern_data.get('neckline', 0)
            head = pattern_data.get('head', 0)

            if neckline > 0 and head > 0:
                pattern_height = abs(head - neckline)

                if self.position_type == "LONG":
                    # Inverse H&S
                    extension_1618 = neckline + (pattern_height * 1.618)
                    extension_2618 = neckline + (pattern_height * 2.618)

                    targets.append({
                        'price': extension_1618,
                        'confluence_count': 4,
                        'sources': ['Inv H&S 1.618 Extension'],
                        'confidence': 85,
                        'priority': 2
                    })

                    targets.append({
                        'price': extension_2618,
                        'confluence_count': 3,
                        'sources': ['Inv H&S 2.618 Extension'],
                        'confidence': 70,
                        'priority': 3
                    })

                else:
                    # Head & Shoulders
                    extension_1618 = neckline - (pattern_height * 1.618)
                    extension_2618 = neckline - (pattern_height * 2.618)

                    targets.append({
                        'price': extension_1618,
                        'confluence_count': 4,
                        'sources': ['H&S 1.618 Extension'],
                        'confidence': 85,
                        'priority': 2
                    })

                    targets.append({
                        'price': extension_2618,
                        'confluence_count': 3,
                        'sources': ['H&S 2.618 Extension'],
                        'confidence': 70,
                        'priority': 3
                    })

        return targets

    def _get_option_chain_targets(self, option_chain: Dict) -> List[Dict]:
        """Get targets from OI walls, Max Pain, GEX levels"""
        targets = []

        # Max Pain (magnet for price)
        max_pain = option_chain.get('max_pain', 0)
        if max_pain > 0:
            targets.append({
                'price': max_pain,
                'confluence_count': 3,
                'sources': ['Max Pain'],
                'confidence': 75,
                'priority': 2
            })

        # OI Walls
        oi_walls = option_chain.get('oi_walls', {})

        if self.position_type == "LONG":
            # Target CALL OI walls (resistance)
            call_walls = oi_walls.get('call_walls', [])
            for wall in call_walls[:3]:  # Top 3 walls
                strike = wall.get('strike', 0)
                oi = wall.get('oi', 0)

                if strike > self.current_price:
                    targets.append({
                        'price': strike,
                        'confluence_count': 3,
                        'sources': [f'CALL OI Wall ({oi/1000000:.1f}M)'],
                        'confidence': 80,
                        'priority': 2
                    })

        else:  # SHORT
            # Target PUT OI walls (support)
            put_walls = oi_walls.get('put_walls', [])
            for wall in put_walls[:3]:  # Top 3 walls
                strike = wall.get('strike', 0)
                oi = wall.get('oi', 0)

                if strike < self.current_price:
                    targets.append({
                        'price': strike,
                        'confluence_count': 3,
                        'sources': [f'PUT OI Wall ({oi/1000000:.1f}M)'],
                        'confidence': 80,
                        'priority': 2
                    })

        # GEX levels
        gex_levels = option_chain.get('gex_levels', [])
        for gex in gex_levels[:2]:  # Top 2 GEX levels
            price = gex.get('price', 0)
            concentration = gex.get('concentration', 0)

            if (self.position_type == "LONG" and price > self.current_price) or \
               (self.position_type == "SHORT" and price < self.current_price):
                targets.append({
                    'price': price,
                    'confluence_count': 2,
                    'sources': [f'GEX Level ({concentration}%)'],
                    'confidence': 75,
                    'priority': 3
                })

        return targets

    def _get_confluence_targets(self, confluence_clusters: List[Dict]) -> List[Dict]:
        """Get targets from confluence clusters (multiple sources agreeing)"""
        targets = []

        for cluster in confluence_clusters:
            price = cluster.get('price', 0)
            confluence_count = cluster.get('confluence_count', 0)
            sources = cluster.get('source_list', [])
            avg_strength = cluster.get('avg_strength', 0)

            # Only use clusters with 3+ sources
            if confluence_count >= 3:
                confidence = min(95, int(avg_strength))

                targets.append({
                    'price': price,
                    'confluence_count': confluence_count,
                    'sources': sources,
                    'confidence': confidence,
                    'priority': 1 if confluence_count >= 5 else 2
                })

        return targets

    def _get_sr_targets(self, sr_levels: List[Dict], level_type: str) -> List[Dict]:
        """Get targets from individual S/R levels"""
        targets = []

        for level in sr_levels[:5]:  # Top 5 levels
            targets.append({
                'price': level.get('price', 0),
                'confluence_count': 1,
                'sources': [level.get('source', 'Unknown')],
                'confidence': level.get('strength', 70),
                'priority': level.get('priority', 3)
            })

        return targets

    def _deduplicate_targets(self, targets: List[Dict], threshold: int = 10) -> List[Dict]:
        """Remove duplicate targets within threshold points"""
        if not targets:
            return []

        # Sort by price
        sorted_targets = sorted(targets, key=lambda x: x['price'])

        deduplicated = []
        current = sorted_targets[0]

        for target in sorted_targets[1:]:
            if abs(target['price'] - current['price']) <= threshold:
                # Merge: keep the one with higher confluence
                if target['confluence_count'] > current['confluence_count']:
                    # Merge sources
                    current = {
                        'price': (current['price'] + target['price']) / 2,
                        'confluence_count': current['confluence_count'] + target['confluence_count'],
                        'sources': current['sources'] + target['sources'],
                        'confidence': max(current['confidence'], target['confidence']),
                        'priority': min(current['priority'], target['priority'])
                    }
                else:
                    # Keep current, add sources
                    current['sources'].extend(target['sources'])
                    current['confluence_count'] += 1
            else:
                deduplicated.append(current)
                current = target

        deduplicated.append(current)

        # Sort by priority and confluence
        deduplicated = sorted(
            deduplicated,
            key=lambda x: (-x['priority'], -x['confluence_count'])
        )

        return deduplicated

    def _get_fallback_target(self, target_num: int, existing_targets: List[SmartTarget]) -> Optional[SmartTarget]:
        """Get fallback target based on simple R:R if not enough confluence targets"""

        # T1: 1:1 R:R (50 pts)
        # T2: 2:1 R:R (100 pts)
        # T3: 3:1 R:R (150 pts)

        if target_num == 1:
            distance = 50
        elif target_num == 2:
            distance = 100
        else:
            distance = 150

        if self.position_type == "LONG":
            target_price = self.entry_price + distance
        else:
            target_price = self.entry_price - distance

        reasoning = f"T{target_num}: ₹{target_price:,.0f} (+{distance:.0f} pts, R:R {target_num}:1)\n"
        reasoning += f"Fallback target (no confluence)\n"
        reasoning += f"Simple {target_num}:1 risk/reward ratio"

        return SmartTarget(
            price=target_price,
            distance_points=distance,
            confluence_count=0,
            sources=['Fallback R:R'],
            confidence=50,
            reasoning=reasoning
        )


# Example usage
if __name__ == "__main__":
    # Example: LONG position targeting resistance levels
    calculator = SmartTargetCalculator(
        entry_price=24480,
        current_price=24490,
        position_type="LONG"
    )

    pattern = {
        'type': 'Inverse H&S',
        'neckline': 24520,
        'head': 24420,
        'target': 24620  # Measured move
    }

    option_chain = {
        'max_pain': 24600,
        'oi_walls': {
            'call_walls': [
                {'strike': 24550, 'oi': 1500000},
                {'strike': 24600, 'oi': 2000000},
                {'strike': 24650, 'oi': 1200000}
            ]
        },
        'gex_levels': [
            {'price': 24580, 'concentration': 85}
        ]
    }

    resistance_levels = [
        {'price': 24550, 'source': 'HTF S/R', 'strength': 85, 'priority': 1},
        {'price': 24600, 'source': 'Volume POC', 'strength': 88, 'priority': 1},
        {'price': 24620, 'source': 'Fibonacci 0.618', 'strength': 80, 'priority': 3}
    ]

    confluence_clusters = [
        {
            'price': 24605,
            'confluence_count': 4,
            'source_list': ['Max Pain', 'CALL OI Wall', 'Volume POC', 'HTF S/R'],
            'avg_strength': 85
        }
    ]

    targets = calculator.calculate_smart_targets(
        resistance_levels=resistance_levels,
        support_levels=[],
        pattern_data=pattern,
        option_chain=option_chain,
        confluence_clusters=confluence_clusters
    )

    print("Smart Targets:")
    for target in targets:
        print(f"\n{target.reasoning}")
        print(f"Confidence: {target.confidence}%")
