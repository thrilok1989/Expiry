"""
Liquidity Gravity & Oasis Module
Predicts where price is magnetically attracted based on liquidity analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LiquidityZone:
    """Represents a liquidity zone"""
    price: float
    strength: float  # 0-100
    type: str  # "HVN", "LVN", "Gap", "Gamma Wall", "POC"
    direction: str  # "Support", "Resistance", "Magnet"
    volume: int
    distance_pct: float  # % from current price


@dataclass
class LiquidityGravityResult:
    """Liquidity gravity analysis result"""
    primary_target: float
    secondary_targets: List[float]
    support_zones: List[LiquidityZone]
    resistance_zones: List[LiquidityZone]
    hvn_zones: List[LiquidityZone]  # High Volume Nodes
    lvn_zones: List[LiquidityZone]  # Low Volume Nodes (gaps)
    fair_value_gaps: List[Tuple[float, float]]  # (gap_low, gap_high)
    gamma_walls: List[Tuple[float, str]]  # (strike, type)
    poc: float  # Point of Control (highest volume)
    vwap_bands: Dict[str, float]  # Upper, Lower, VWAP
    gravity_strength: float  # 0-100
    recommendation: str
    signals: List[str]


class LiquidityGravityAnalyzer:
    """
    Liquidity Gravity & Oasis Analyzer

    Identifies where price is magnetically pulled:
    1. High Volume Nodes (HVN) - Areas of accumulation
    2. Low Volume Nodes (LVN) - Gaps that attract price
    3. Fair Value Gaps (FVG) - Unfilled price gaps
    4. Gamma Walls - Option strikes with massive OI
    5. VWAP bands - Institutional reference points
    """

    def __init__(self):
        """Initialize Liquidity Gravity Analyzer"""
        self.hvn_threshold = 1.5  # Volume 1.5x average = HVN
        self.lvn_threshold = 0.5  # Volume 0.5x average = LVN
        self.gap_min_size = 0.005  # 0.5% minimum gap size
        self.gamma_wall_threshold = 50000  # OI threshold

    def analyze_liquidity_gravity(
        self,
        df: pd.DataFrame,
        option_chain: Optional[Dict] = None,
        volume_profile: Optional[Dict] = None
    ) -> LiquidityGravityResult:
        """
        Complete liquidity gravity analysis

        Args:
            df: OHLCV dataframe
            option_chain: Option chain for gamma wall detection
            volume_profile: Volume profile data

        Returns:
            LiquidityGravityResult with price targets
        """
        signals = []

        if len(df) < 20:
            return self._default_result(df['close'].iloc[-1] if len(df) > 0 else 0)

        current_price = df['close'].iloc[-1]

        # 1. Identify High Volume Nodes (HVN)
        hvn_zones = self._identify_hvn_zones(df, current_price)
        if hvn_zones:
            signals.append(f"Found {len(hvn_zones)} HVN zones (liquidity pools)")

        # 2. Identify Low Volume Nodes (LVN - gaps)
        lvn_zones = self._identify_lvn_zones(df, current_price)
        if lvn_zones:
            signals.append(f"Found {len(lvn_zones)} LVN zones (gaps to fill)")

        # 3. Detect Fair Value Gaps
        fair_value_gaps = self._detect_fair_value_gaps(df, current_price)
        if fair_value_gaps:
            signals.append(f"Found {len(fair_value_gaps)} Fair Value Gaps")

        # 4. Calculate VWAP bands
        vwap_bands = self._calculate_vwap_bands(df)
        signals.append(f"VWAP: {vwap_bands['vwap']:.2f}")

        # 5. Detect Gamma Walls
        gamma_walls = []
        if option_chain:
            gamma_walls = self._detect_gamma_walls(option_chain, current_price)
            if gamma_walls:
                signals.append(f"Found {len(gamma_walls)} Gamma Walls")

        # 6. Calculate Point of Control (POC)
        poc = self._calculate_poc(df)

        # 7. Separate Support and Resistance zones
        support_zones = [z for z in hvn_zones if z.price < current_price]
        resistance_zones = [z for z in hvn_zones if z.price > current_price]

        # 8. Calculate Primary Target (strongest gravity)
        primary_target, gravity_strength = self._calculate_primary_target(
            current_price, hvn_zones, lvn_zones, fair_value_gaps,
            gamma_walls, vwap_bands, poc
        )
        signals.append(f"🎯 Primary Target: {primary_target:.2f} (Strength: {gravity_strength:.0f}%)")

        # 9. Calculate Secondary Targets
        secondary_targets = self._calculate_secondary_targets(
            current_price, hvn_zones, lvn_zones, gamma_walls, primary_target
        )

        # 10. Generate Recommendation
        recommendation = self._generate_recommendation(
            current_price, primary_target, gravity_strength,
            support_zones, resistance_zones, fair_value_gaps
        )

        return LiquidityGravityResult(
            primary_target=primary_target,
            secondary_targets=secondary_targets,
            support_zones=support_zones,
            resistance_zones=resistance_zones,
            hvn_zones=hvn_zones,
            lvn_zones=lvn_zones,
            fair_value_gaps=fair_value_gaps,
            gamma_walls=gamma_walls,
            poc=poc,
            vwap_bands=vwap_bands,
            gravity_strength=gravity_strength,
            recommendation=recommendation,
            signals=signals
        )

    def _identify_hvn_zones(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> List[LiquidityZone]:
        """Identify High Volume Nodes (areas of accumulation)"""
        zones = []

        if 'volume' not in df.columns or len(df) < 10:
            return zones

        # Calculate volume moving average
        df['vol_ma'] = df['volume'].rolling(window=20).mean()
        avg_volume = df['vol_ma'].mean()

        # Find high volume bars
        high_vol_bars = df[df['volume'] > avg_volume * self.hvn_threshold].tail(50)

        if len(high_vol_bars) == 0:
            return zones

        # Cluster nearby HVN zones
        prices = high_vol_bars['close'].values
        volumes = high_vol_bars['volume'].values

        # Simple clustering by rounding to price levels
        price_clusters = {}
        for price, vol in zip(prices, volumes):
            # Round to nearest 50 points for clustering
            cluster_price = round(price / 50) * 50
            if cluster_price not in price_clusters:
                price_clusters[cluster_price] = {'volume': 0, 'count': 0}
            price_clusters[cluster_price]['volume'] += vol
            price_clusters[cluster_price]['count'] += 1

        # Create zones from clusters
        for cluster_price, data in price_clusters.items():
            if data['count'] >= 2:  # At least 2 bars
                distance_pct = ((cluster_price - current_price) / current_price) * 100
                strength = min((data['volume'] / avg_volume / data['count']) * 20, 100)

                zone_type = "Support" if cluster_price < current_price else "Resistance"

                zones.append(LiquidityZone(
                    price=cluster_price,
                    strength=strength,
                    type="HVN",
                    direction=zone_type,
                    volume=int(data['volume']),
                    distance_pct=distance_pct
                ))

        # Sort by strength
        zones.sort(key=lambda x: x.strength, reverse=True)
        return zones[:10]  # Top 10

    def _identify_lvn_zones(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> List[LiquidityZone]:
        """Identify Low Volume Nodes (gaps that attract price)"""
        zones = []

        if 'volume' not in df.columns or len(df) < 10:
            return zones

        avg_volume = df['volume'].tail(50).mean()

        # Find low volume bars
        low_vol_bars = df[df['volume'] < avg_volume * self.lvn_threshold].tail(30)

        if len(low_vol_bars) == 0:
            return zones

        # Look for consecutive low volume areas (gaps)
        consecutive_low = []
        for i in range(len(low_vol_bars) - 2):
            window = low_vol_bars.iloc[i:i+3]
            if len(window) == 3:
                avg_price = window['close'].mean()
                total_vol = window['volume'].sum()

                distance_pct = ((avg_price - current_price) / current_price) * 100

                # Only consider gaps within reasonable range
                if abs(distance_pct) < 10:
                    zones.append(LiquidityZone(
                        price=avg_price,
                        strength=70,  # LVNs are strong magnets
                        type="LVN",
                        direction="Magnet",
                        volume=int(total_vol),
                        distance_pct=distance_pct
                    ))

        return zones[:5]  # Top 5

    def _detect_fair_value_gaps(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> List[Tuple[float, float]]:
        """
        Detect Fair Value Gaps (FVG)

        FVG = 3-candle pattern where middle candle doesn't overlap with outer candles
        """
        gaps = []

        if len(df) < 3:
            return gaps

        recent = df.tail(50)

        for i in range(len(recent) - 2):
            candle_1 = recent.iloc[i]
            candle_2 = recent.iloc[i + 1]
            candle_3 = recent.iloc[i + 2]

            # Bullish FVG: Gap between candle 1 high and candle 3 low
            if candle_1['high'] < candle_3['low']:
                gap_size_pct = ((candle_3['low'] - candle_1['high']) / current_price) * 100
                if gap_size_pct > self.gap_min_size * 100:
                    gap_mid = (candle_1['high'] + candle_3['low']) / 2
                    distance_pct = ((gap_mid - current_price) / current_price) * 100

                    if abs(distance_pct) < 5:  # Within 5%
                        gaps.append((candle_1['high'], candle_3['low']))

            # Bearish FVG: Gap between candle 1 low and candle 3 high
            elif candle_1['low'] > candle_3['high']:
                gap_size_pct = ((candle_1['low'] - candle_3['high']) / current_price) * 100
                if gap_size_pct > self.gap_min_size * 100:
                    gap_mid = (candle_1['low'] + candle_3['high']) / 2
                    distance_pct = ((gap_mid - current_price) / current_price) * 100

                    if abs(distance_pct) < 5:
                        gaps.append((candle_3['high'], candle_1['low']))

        return gaps[-5:]  # Most recent 5

    def _detect_gamma_walls(
        self,
        option_chain: Dict,
        current_price: float
    ) -> List[Tuple[float, str]]:
        """Detect gamma walls (strikes with massive OI)"""
        walls = []

        ce_data = option_chain.get('CE', {})
        pe_data = option_chain.get('PE', {})

        ce_strikes = ce_data.get('strikePrice', [])
        pe_strikes = pe_data.get('strikePrice', [])
        ce_oi = ce_data.get('openInterest', [])
        pe_oi = pe_data.get('openInterest', [])

        # Find strikes with massive OI
        for strike, oi in zip(ce_strikes, ce_oi):
            if oi > self.gamma_wall_threshold:
                distance_pct = abs((strike - current_price) / current_price) * 100
                if distance_pct < 5:  # Within 5%
                    walls.append((strike, "CALL"))

        for strike, oi in zip(pe_strikes, pe_oi):
            if oi > self.gamma_wall_threshold:
                distance_pct = abs((strike - current_price) / current_price) * 100
                if distance_pct < 5:
                    walls.append((strike, "PUT"))

        return walls[:10]

    def _calculate_vwap_bands(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate VWAP and bands"""
        if 'volume' not in df.columns or len(df) < 10:
            close = df['close'].iloc[-1] if len(df) > 0 else 0
            return {'vwap': close, 'upper': close, 'lower': close}

        # VWAP calculation
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']

        recent = df.tail(50)
        vwap = recent['tp_volume'].sum() / recent['volume'].sum() if recent['volume'].sum() > 0 else recent['close'].mean()

        # VWAP bands (1 standard deviation)
        vwap_distances = (recent['typical_price'] - vwap) ** 2 * recent['volume']
        variance = vwap_distances.sum() / recent['volume'].sum() if recent['volume'].sum() > 0 else 0
        std_dev = np.sqrt(variance)

        return {
            'vwap': vwap,
            'upper': vwap + std_dev,
            'lower': vwap - std_dev
        }

    def _calculate_poc(self, df: pd.DataFrame) -> float:
        """Calculate Point of Control (price level with highest volume)"""
        if 'volume' not in df.columns or len(df) < 10:
            return df['close'].iloc[-1] if len(df) > 0 else 0

        recent = df.tail(100)

        # Create price bins
        price_volume = {}
        for _, row in recent.iterrows():
            price_bin = round(row['close'] / 50) * 50
            if price_bin not in price_volume:
                price_volume[price_bin] = 0
            price_volume[price_bin] += row['volume']

        # Find price with max volume
        if not price_volume:
            return recent['close'].mean()

        poc = max(price_volume, key=price_volume.get)
        return poc

    def _calculate_primary_target(
        self,
        current_price: float,
        hvn_zones: List[LiquidityZone],
        lvn_zones: List[LiquidityZone],
        fair_value_gaps: List[Tuple[float, float]],
        gamma_walls: List[Tuple[float, str]],
        vwap_bands: Dict[str, float],
        poc: float
    ) -> Tuple[float, float]:
        """Calculate primary price target with highest gravity"""
        candidates = []

        # HVN zones (weight by strength)
        for zone in hvn_zones[:5]:
            if abs(zone.distance_pct) < 3:  # Within 3%
                candidates.append({
                    'price': zone.price,
                    'strength': zone.strength * 0.8,
                    'type': 'HVN'
                })

        # LVN zones (strong magnets)
        for zone in lvn_zones[:3]:
            candidates.append({
                'price': zone.price,
                'strength': 85,
                'type': 'LVN'
            })

        # Fair Value Gaps (strong magnets)
        for gap_low, gap_high in fair_value_gaps[:3]:
            gap_mid = (gap_low + gap_high) / 2
            candidates.append({
                'price': gap_mid,
                'strength': 80,
                'type': 'FVG'
            })

        # Gamma Walls
        for strike, wall_type in gamma_walls[:5]:
            candidates.append({
                'price': strike,
                'strength': 75,
                'type': 'Gamma Wall'
            })

        # VWAP
        candidates.append({
            'price': vwap_bands['vwap'],
            'strength': 70,
            'type': 'VWAP'
        })

        # POC
        candidates.append({
            'price': poc,
            'strength': 65,
            'type': 'POC'
        })

        # Find strongest candidate
        if not candidates:
            return current_price, 0

        # Adjust strength by distance (closer = stronger)
        for candidate in candidates:
            distance_pct = abs((candidate['price'] - current_price) / current_price) * 100
            distance_penalty = min(distance_pct * 5, 30)
            candidate['adjusted_strength'] = candidate['strength'] - distance_penalty

        # Sort by adjusted strength
        candidates.sort(key=lambda x: x['adjusted_strength'], reverse=True)

        best = candidates[0]
        return best['price'], best['adjusted_strength']

    def _calculate_secondary_targets(
        self,
        current_price: float,
        hvn_zones: List[LiquidityZone],
        lvn_zones: List[LiquidityZone],
        gamma_walls: List[Tuple[float, str]],
        primary_target: float
    ) -> List[float]:
        """Calculate secondary targets"""
        targets = []

        # Add strong HVN zones
        for zone in hvn_zones[:3]:
            if zone.price != primary_target:
                targets.append(zone.price)

        # Add LVN zones
        for zone in lvn_zones[:2]:
            if zone.price != primary_target:
                targets.append(zone.price)

        # Add gamma walls
        for strike, _ in gamma_walls[:3]:
            if strike != primary_target and strike not in targets:
                targets.append(strike)

        # Sort by distance from current price
        targets.sort(key=lambda x: abs(x - current_price))

        return targets[:5]

    def _generate_recommendation(
        self,
        current_price: float,
        primary_target: float,
        gravity_strength: float,
        support_zones: List[LiquidityZone],
        resistance_zones: List[LiquidityZone],
        fair_value_gaps: List[Tuple[float, float]]
    ) -> str:
        """Generate trading recommendation"""
        direction = "UP" if primary_target > current_price else "DOWN"
        distance = abs(primary_target - current_price)
        distance_pct = (distance / current_price) * 100

        if gravity_strength > 70:
            strength_desc = "STRONG"
        elif gravity_strength > 50:
            strength_desc = "MODERATE"
        else:
            strength_desc = "WEAK"

        rec = f"🎯 {strength_desc} Gravity {direction} to {primary_target:.2f} ({distance_pct:+.2f}%)\n"

        if direction == "UP":
            if support_zones:
                nearest_support = support_zones[0]
                rec += f"✅ Support at {nearest_support.price:.2f}\n"
        else:
            if resistance_zones:
                nearest_resistance = resistance_zones[0]
                rec += f"⚠️ Resistance at {nearest_resistance.price:.2f}\n"

        if fair_value_gaps:
            rec += f"📊 {len(fair_value_gaps)} Fair Value Gaps to fill\n"

        return rec

    def _default_result(self, current_price: float) -> LiquidityGravityResult:
        """Default result for insufficient data"""
        return LiquidityGravityResult(
            primary_target=current_price,
            secondary_targets=[],
            support_zones=[],
            resistance_zones=[],
            hvn_zones=[],
            lvn_zones=[],
            fair_value_gaps=[],
            gamma_walls=[],
            poc=current_price,
            vwap_bands={'vwap': current_price, 'upper': current_price, 'lower': current_price},
            gravity_strength=0,
            recommendation="Insufficient data for liquidity analysis",
            signals=[]
        )


def format_liquidity_report(result: LiquidityGravityResult) -> str:
    """Format liquidity gravity analysis as readable report"""
    report = f"""
╔══════════════════════════════════════════════════════════╗
║          LIQUIDITY GRAVITY ANALYSIS                      ║
╚══════════════════════════════════════════════════════════╝

🎯 PRIMARY TARGET: {result.primary_target:.2f}
💪 GRAVITY STRENGTH: {result.gravity_strength:.1f}/100

📊 SECONDARY TARGETS:
"""
    for i, target in enumerate(result.secondary_targets[:3], 1):
        report += f"  {i}. {target:.2f}\n"

    report += f"\n─────────────────────────────────────────────────────────\n"
    report += f"🛡️  SUPPORT ZONES ({len(result.support_zones)}):\n"
    for zone in result.support_zones[:3]:
        report += f"  • {zone.price:.2f} - {zone.type} (Strength: {zone.strength:.0f})\n"

    report += f"\n⚡ RESISTANCE ZONES ({len(result.resistance_zones)}):\n"
    for zone in result.resistance_zones[:3]:
        report += f"  • {zone.price:.2f} - {zone.type} (Strength: {zone.strength:.0f})\n"

    if result.fair_value_gaps:
        report += f"\n📈 FAIR VALUE GAPS ({len(result.fair_value_gaps)}):\n"
        for gap_low, gap_high in result.fair_value_gaps[:3]:
            report += f"  • {gap_low:.2f} - {gap_high:.2f}\n"

    if result.gamma_walls:
        report += f"\n🧱 GAMMA WALLS ({len(result.gamma_walls)}):\n"
        for strike, wall_type in result.gamma_walls[:5]:
            report += f"  • {strike:.0f} ({wall_type})\n"

    report += f"\n📊 KEY LEVELS:\n"
    report += f"  • POC: {result.poc:.2f}\n"
    report += f"  • VWAP: {result.vwap_bands['vwap']:.2f}\n"
    report += f"  • VWAP Upper: {result.vwap_bands['upper']:.2f}\n"
    report += f"  • VWAP Lower: {result.vwap_bands['lower']:.2f}\n"

    report += f"\n─────────────────────────────────────────────────────────\n"
    report += f"💡 RECOMMENDATION:\n{result.recommendation}\n"

    return report
