"""
ML-Based Entry Finder
Uses Machine Learning to find optimal entry points based on comprehensive S/R analysis

Analyzes:
- OI Walls (Max PUT/CALL OI)
- GEX Walls (Gamma exposure)
- HTF S/R (Multi-timeframe pivots)
- VOB levels (Volume Order Blocks)
- Price proximity to levels
- Market sentiment
- VIX and volatility

Outputs:
- Best entry price zone
- Entry confidence score
- Stop loss level
- Target levels (conservative, moderate, aggressive)
- Risk-reward ratio
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MLEntryFinder:
    """
    ML-powered entry finder using comprehensive S/R analysis
    """

    def __init__(self):
        self.logger = logger

    def analyze_entry_opportunity(
        self,
        current_price: float,
        support_levels: List[Dict],
        resistance_levels: List[Dict],
        market_sentiment: Dict,
        vix: float = 15.0,
        pcr: float = 1.0,
        chart_indicators: Optional[Dict] = None,
        futures_analysis: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze and find optimal entry based on all S/R levels

        Args:
            current_price: Current spot price
            support_levels: List of support levels with type, source, strength
            resistance_levels: List of resistance levels
            market_sentiment: Overall market sentiment dict
            vix: India VIX value
            pcr: Put-Call Ratio

        Returns:
            Dict with entry analysis including:
            - entry_zone: (lower, upper) price range
            - direction: 'LONG', 'SHORT', or 'NO_TRADE'
            - confidence: 0-100%
            - stop_loss: Stop loss price
            - targets: {conservative, moderate, aggressive}
            - risk_reward: Ratio
            - reasoning: Why this entry
        """

        # ==========================================
        # FILTER 1: MAJOR + NEAREST LEVELS ONLY
        # ==========================================
        # Filter support levels: Keep only HIGH strength (MAJOR) or within 50 pts (NEAREST)
        major_support = [l for l in support_levels if l.get('strength') == 'HIGH']
        nearest_support_candidates = [
            l for l in support_levels
            if abs(current_price - l['price']) <= 50 and l['price'] < current_price
        ]
        # Combine major + nearest, remove duplicates
        filtered_supports = {l['price']: l for l in major_support + nearest_support_candidates}.values()
        filtered_supports = list(filtered_supports)

        # Filter resistance levels: Keep only HIGH strength (MAJOR) or within 50 pts (NEAREST)
        major_resistance = [l for l in resistance_levels if l.get('strength') == 'HIGH']
        nearest_resistance_candidates = [
            l for l in resistance_levels
            if abs(l['price'] - current_price) <= 50 and l['price'] > current_price
        ]
        # Combine major + nearest, remove duplicates
        filtered_resistances = {l['price']: l for l in major_resistance + nearest_resistance_candidates}.values()
        filtered_resistances = list(filtered_resistances)

        # Score each level based on strength and distance
        support_scores = self._score_levels(filtered_supports, current_price, is_support=True)
        resistance_scores = self._score_levels(filtered_resistances, current_price, is_support=False)

        # Find nearest and strongest levels
        nearest_support = self._find_nearest_level(filtered_supports, current_price, is_support=True)
        nearest_resistance = self._find_nearest_level(filtered_resistances, current_price, is_support=False)
        strongest_support = self._find_strongest_level(support_scores, filtered_supports)
        strongest_resistance = self._find_strongest_level(resistance_scores, filtered_resistances)

        # Calculate distance to key levels
        support_distance = abs(current_price - nearest_support['price']) if nearest_support else 999
        resistance_distance = abs(nearest_resistance['price'] - current_price) if nearest_resistance else 999

        # Determine trade direction based on ML analysis + chart indicators + futures
        direction, direction_confidence = self._determine_direction(
            current_price=current_price,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            support_distance=support_distance,
            resistance_distance=resistance_distance,
            market_sentiment=market_sentiment,
            vix=vix,
            pcr=pcr,
            support_scores=support_scores,
            resistance_scores=resistance_scores,
            chart_indicators=chart_indicators,
            futures_analysis=futures_analysis  # ADD FUTURES ANALYSIS
        )

        # If NO_TRADE, return early
        if direction == 'NO_TRADE':
            return {
                'direction': 'NO_TRADE',
                'confidence': 0,
                'entry_zone': None,
                'stop_loss': None,
                'targets': None,
                'risk_reward': 0,
                'reasoning': 'Price not near any significant S/R level or market conditions unclear'
            }

        # Calculate entry zone
        entry_zone = self._calculate_entry_zone(
            direction=direction,
            current_price=current_price,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            vix=vix
        )

        # Calculate stop loss
        stop_loss = self._calculate_stop_loss(
            direction=direction,
            entry_zone=entry_zone,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            vix=vix
        )

        # Calculate targets
        targets = self._calculate_targets(
            direction=direction,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            strongest_support=strongest_support,
            strongest_resistance=strongest_resistance,
            support_levels=support_levels,
            resistance_levels=resistance_levels
        )

        # Calculate risk-reward ratio
        risk = abs(entry_zone['mid'] - stop_loss)
        reward = abs(targets['moderate'] - entry_zone['mid'])
        risk_reward = reward / risk if risk > 0 else 0

        # Generate reasoning
        reasoning = self._generate_reasoning(
            direction=direction,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            market_sentiment=market_sentiment,
            support_distance=support_distance,
            resistance_distance=resistance_distance,
            vix=vix,
            pcr=pcr
        )

        return {
            'direction': direction,
            'confidence': direction_confidence,
            'entry_zone': entry_zone,
            'stop_loss': stop_loss,
            'targets': targets,
            'risk_reward': risk_reward,
            'reasoning': reasoning,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance
        }

    def _score_levels(
        self,
        levels: List[Dict],
        current_price: float,
        is_support: bool
    ) -> List[float]:
        """
        Score each level based on strength and distance

        Higher score = more important level
        """
        scores = []

        for level in levels:
            score = 0

            # Base score from strength
            strength = level.get('strength', 'LOW')
            if strength == 'HIGH':
                score += 100
            elif strength == 'MEDIUM':
                score += 70
            else:  # LOW
                score += 40

            # Bonus for level type (institutional > technical)
            level_type = level.get('type', '')
            if 'OI Wall' in level_type:
                score += 50  # Highest priority
            elif 'GEX Wall' in level_type:
                score += 40
            elif 'HTF' in level_type:
                score += 30
            elif 'VOB' in level_type:
                score += 20

            # Distance penalty (closer = higher score)
            distance = abs(current_price - level['price'])
            distance_pct = (distance / current_price) * 100

            if distance_pct < 0.5:  # Within 0.5%
                score += 50
            elif distance_pct < 1.0:  # Within 1%
                score += 30
            elif distance_pct < 2.0:  # Within 2%
                score += 10
            else:  # Far away
                score -= 20

            scores.append(max(0, score))

        return scores

    def _find_nearest_level(
        self,
        levels: List[Dict],
        current_price: float,
        is_support: bool
    ) -> Optional[Dict]:
        """Find nearest support or resistance level"""
        if not levels:
            return None

        valid_levels = []
        for level in levels:
            price = level.get('price', 0)
            if is_support and price < current_price:
                valid_levels.append(level)
            elif not is_support and price > current_price:
                valid_levels.append(level)

        if not valid_levels:
            return None

        if is_support:
            # Nearest support = highest price below current
            return max(valid_levels, key=lambda x: x['price'])
        else:
            # Nearest resistance = lowest price above current
            return min(valid_levels, key=lambda x: x['price'])

    def _find_strongest_level(
        self,
        scores: List[float],
        levels: List[Dict]
    ) -> Optional[Dict]:
        """Find level with highest score"""
        if not scores or not levels:
            return None

        max_idx = np.argmax(scores)
        return levels[max_idx]

    def _determine_direction(
        self,
        current_price: float,
        nearest_support: Optional[Dict],
        nearest_resistance: Optional[Dict],
        support_distance: float,
        resistance_distance: float,
        market_sentiment: Dict,
        vix: float,
        pcr: float,
        support_scores: List[float],
        resistance_scores: List[float],
        chart_indicators: Optional[Dict] = None,
        futures_analysis: Optional[Dict] = None
    ) -> Tuple[str, float]:
        """
        Use ML logic to determine trade direction and confidence

        Now integrates:
        - NIFTY Futures analysis (premium/discount, OI bias)
        - ALL chart indicators (RSI, OM, Money Flow, DeltaFlow, BOS, CHOCH, Fibonacci)
        - Support/Resistance levels (OI walls, GEX walls, HTF S/R)
        - Market sentiment
        - VIX, PCR

        Returns:
            (direction, confidence)
        """

        # Initialize ML score
        long_score = 0
        short_score = 0
        confidence = 50  # Base confidence

        # ============================================
        # FACTOR 1: Proximity to MAJOR S/R Levels (25%)
        # ============================================
        if support_distance < 20:  # Within 20 points of support
            long_score += 25
            confidence += 10
        if resistance_distance < 20:  # Within 20 points of resistance
            short_score += 25
            confidence += 10

        # ============================================
        # FACTOR 2: Level Strength (15%)
        # ============================================
        avg_support_score = np.mean(support_scores) if support_scores else 0
        avg_resistance_score = np.mean(resistance_scores) if resistance_scores else 0

        if avg_support_score > avg_resistance_score:
            long_score += 15
            confidence += 6
        elif avg_resistance_score > avg_support_score:
            short_score += 15
            confidence += 6

        # ============================================
        # FACTOR 3: NIFTY FUTURES ANALYSIS (20% - NEW!)
        # Futures premium/discount shows institutional positioning
        # ============================================
        if futures_analysis:
            # Futures Premium Bias (10%)
            futures_bias = futures_analysis.get('premium_bias', 'NEUTRAL')
            if 'BULL' in futures_bias:
                long_score += 10
                confidence += 5
            elif 'BEAR' in futures_bias:
                short_score += 10
                confidence += 5

            # Futures OI Bias (5%)
            futures_oi_bias = futures_analysis.get('oi_bias', 'NEUTRAL')
            if 'BULL' in futures_oi_bias:
                long_score += 5
                confidence += 3
            elif 'BEAR' in futures_oi_bias:
                short_score += 5
                confidence += 3

            # Combined Futures Bias (5%)
            combined_bias = futures_analysis.get('combined_bias', 'NEUTRAL')
            if 'BULL' in combined_bias:
                long_score += 5
            elif 'BEAR' in combined_bias:
                short_score += 5

            # Premium Strength Bonus (higher premium = stronger conviction)
            premium_pct = abs(futures_analysis.get('premium_pct', 0))
            if premium_pct > 0.5:  # >0.5% premium/discount is significant
                confidence += 3

        # ============================================
        # FACTOR 4: Market Sentiment (10%)
        # ============================================
        sentiment_overall = market_sentiment.get('overall', 'NEUTRAL')
        if 'BULL' in sentiment_overall:
            long_score += 10
            confidence += 4
        elif 'BEAR' in sentiment_overall:
            short_score += 10
            confidence += 4

        # ============================================
        # FACTOR 5: CHART INDICATORS (20%)
        # ============================================
        if chart_indicators:
            # RSI Signal (5%)
            rsi_signal = chart_indicators.get('rsi_signal')
            if rsi_signal == 'BULLISH' or rsi_signal == 'BUY':
                long_score += 5
                confidence += 3
            elif rsi_signal == 'BEARISH' or rsi_signal == 'SELL':
                short_score += 5
                confidence += 3

            # OM Indicator (Order Flow & Momentum) (5%)
            om_signal = chart_indicators.get('om_signal')
            if om_signal == 'BULLISH':
                long_score += 5
                confidence += 3
            elif om_signal == 'BEARISH':
                short_score += 5
                confidence += 3

            # Money Flow Profile POC Sentiment (5%)
            mfp_sentiment = chart_indicators.get('money_flow_sentiment')
            if mfp_sentiment == 'BULLISH':
                long_score += 5
                confidence += 2
            elif mfp_sentiment == 'BEARISH':
                short_score += 5
                confidence += 2

            # DeltaFlow Profile Delta Bias (5%)
            delta_bias = chart_indicators.get('deltaflow_bias')
            if delta_bias == 'BULLISH':
                long_score += 5
                confidence += 2
            elif delta_bias == 'BEARISH':
                short_score += 5
                confidence += 2

            # BOS/CHOCH Price Action (3%)
            if chart_indicators.get('bos_bullish'):
                long_score += 3
                confidence += 2
            if chart_indicators.get('bos_bearish'):
                short_score += 3
                confidence += 2
            if chart_indicators.get('choch_bullish'):
                long_score += 2
            if chart_indicators.get('choch_bearish'):
                short_score += 2

            # Fibonacci Level Proximity (2%)
            if chart_indicators.get('near_fibonacci_support'):
                long_score += 2
            if chart_indicators.get('near_fibonacci_resistance'):
                short_score += 2

        # ============================================
        # FACTOR 5: PCR (5%)
        # ============================================
        if pcr > 1.2:  # Bullish
            long_score += 5
        elif pcr < 0.8:  # Bearish
            short_score += 5

        # ============================================
        # FACTOR 6: VIX (5%)
        # ============================================
        if vix < 12:  # Low volatility = trend continuation
            confidence += 5
        elif vix > 20:  # High volatility = risky
            confidence -= 10

        # ============================================
        # Determine final direction
        # ============================================
        confidence = min(90, max(0, confidence))  # Cap 0-90%

        # Require minimum confidence for trade
        if confidence < 55:
            return ('NO_TRADE', 0)

        # Determine direction
        if long_score > short_score + 20:  # Clear long bias
            return ('LONG', confidence)
        elif short_score > long_score + 20:  # Clear short bias
            return ('SHORT', confidence)
        else:  # Too close to call
            return ('NO_TRADE', 0)

    def _calculate_entry_zone(
        self,
        direction: str,
        current_price: float,
        nearest_support: Optional[Dict],
        nearest_resistance: Optional[Dict],
        vix: float
    ) -> Dict:
        """Calculate entry zone with lower, mid, upper"""

        # Zone width based on VIX
        zone_width = 10 if vix < 12 else (15 if vix < 18 else 20)

        if direction == 'LONG':
            # Entry zone around support
            support_price = nearest_support['price'] if nearest_support else current_price
            lower = support_price - (zone_width / 2)
            upper = support_price + (zone_width / 2)
            mid = support_price
        elif direction == 'SHORT':
            # Entry zone around resistance
            resistance_price = nearest_resistance['price'] if nearest_resistance else current_price
            lower = resistance_price - (zone_width / 2)
            upper = resistance_price + (zone_width / 2)
            mid = resistance_price
        else:
            return None

        return {
            'lower': lower,
            'mid': mid,
            'upper': upper,
            'width': zone_width
        }

    def _calculate_stop_loss(
        self,
        direction: str,
        entry_zone: Dict,
        nearest_support: Optional[Dict],
        nearest_resistance: Optional[Dict],
        vix: float
    ) -> float:
        """Calculate stop loss level"""

        # Buffer based on VIX
        buffer = 10 if vix < 12 else (15 if vix < 18 else 20)

        if direction == 'LONG':
            # Stop below support
            support_price = nearest_support['price'] if nearest_support else entry_zone['lower']
            return support_price - buffer
        elif direction == 'SHORT':
            # Stop above resistance
            resistance_price = nearest_resistance['price'] if nearest_resistance else entry_zone['upper']
            return resistance_price + buffer
        else:
            return 0

    def _calculate_targets(
        self,
        direction: str,
        entry_zone: Dict,
        stop_loss: float,
        nearest_support: Optional[Dict],
        nearest_resistance: Optional[Dict],
        strongest_support: Optional[Dict],
        strongest_resistance: Optional[Dict],
        support_levels: List[Dict],
        resistance_levels: List[Dict]
    ) -> Dict:
        """Calculate conservative, moderate, aggressive targets"""

        risk = abs(entry_zone['mid'] - stop_loss)

        if direction == 'LONG':
            # Target at nearest resistance or 1:1.5 RR
            nearest_res = nearest_resistance['price'] if nearest_resistance else (entry_zone['mid'] + risk * 1.5)

            targets = {
                'conservative': entry_zone['mid'] + (risk * 1.0),  # 1:1 RR
                'moderate': entry_zone['mid'] + (risk * 1.5),      # 1:1.5 RR
                'aggressive': nearest_res                          # Resistance target
            }

        elif direction == 'SHORT':
            # Target at nearest support or 1:1.5 RR
            nearest_sup = nearest_support['price'] if nearest_support else (entry_zone['mid'] - risk * 1.5)

            targets = {
                'conservative': entry_zone['mid'] - (risk * 1.0),  # 1:1 RR
                'moderate': entry_zone['mid'] - (risk * 1.5),      # 1:1.5 RR
                'aggressive': nearest_sup                          # Support target
            }
        else:
            return None

        return targets

    def _generate_reasoning(
        self,
        direction: str,
        nearest_support: Optional[Dict],
        nearest_resistance: Optional[Dict],
        market_sentiment: Dict,
        support_distance: float,
        resistance_distance: float,
        vix: float,
        pcr: float
    ) -> str:
        """Generate human-readable reasoning for the trade"""

        reasons = []

        # Direction
        reasons.append(f"**Direction:** {direction}")

        # Proximity
        if direction == 'LONG' and nearest_support:
            reasons.append(
                f"- Price near **{nearest_support['type']}** at ₹{nearest_support['price']:,.0f} "
                f"({support_distance:.0f} pts away)"
            )
        elif direction == 'SHORT' and nearest_resistance:
            reasons.append(
                f"- Price near **{nearest_resistance['type']}** at ₹{nearest_resistance['price']:,.0f} "
                f"({resistance_distance:.0f} pts away)"
            )

        # Market sentiment
        sentiment = market_sentiment.get('overall', 'NEUTRAL')
        reasons.append(f"- Market sentiment: **{sentiment}**")

        # PCR
        if pcr > 1.2:
            reasons.append(f"- PCR {pcr:.2f} suggests **bullish** positioning")
        elif pcr < 0.8:
            reasons.append(f"- PCR {pcr:.2f} suggests **bearish** positioning")

        # VIX
        if vix < 12:
            reasons.append(f"- Low VIX ({vix:.1f}) = stable conditions")
        elif vix > 20:
            reasons.append(f"- High VIX ({vix:.1f}) = elevated risk")

        return "\n".join(reasons)


# ==========================================
# Helper Functions
# ==========================================

def find_best_entry(comprehensive_params: Dict, chart_indicators: Optional[Dict] = None) -> Dict:
    """
    Find best entry using comprehensive tab data

    Args:
        comprehensive_params: Output from ComprehensiveChartIntegrator
        chart_indicators: Optional dict with chart indicator signals (RSI, OM, etc.)

    Returns:
        Entry analysis dict
    """
    finder = MLEntryFinder()

    institutional_levels = comprehensive_params.get('institutional_levels', {})
    market_sentiment = comprehensive_params.get('market_sentiment', {})
    vix = comprehensive_params.get('vix', 15.0)
    pcr = comprehensive_params.get('pcr', 1.0)
    current_price = comprehensive_params.get('current_price', 0)

    # Extract futures analysis from raw data
    raw_data = comprehensive_params.get('raw_data', {})
    futures_analysis = raw_data.get('futures_analysis')

    if current_price == 0:
        return {
            'direction': 'NO_TRADE',
            'confidence': 0,
            'reasoning': 'Current price not available'
        }

    entry_analysis = finder.analyze_entry_opportunity(
        current_price=current_price,
        support_levels=institutional_levels.get('support', []),
        resistance_levels=institutional_levels.get('resistance', []),
        market_sentiment=market_sentiment,
        vix=vix,
        pcr=pcr,
        chart_indicators=chart_indicators,
        futures_analysis=futures_analysis
    )

    return entry_analysis
