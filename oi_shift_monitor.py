"""
OI Shift Monitor - Real-time OI Change Detection for Dynamic Exits
Monitors OI unwinding and fresh OI buildup on opposite side
"""

from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class OIShiftMonitor:
    """
    Monitor Option Open Interest changes to detect:
    1. Entry level OI unwinding (wall collapsing)
    2. Fresh OI buildup on opposite side (new barrier forming)
    """

    def __init__(self, entry_strike: int, option_type: str, entry_oi: int, position_type: str):
        """
        Initialize OI shift monitor

        Args:
            entry_strike: Strike price where position was entered (e.g., 24400)
            option_type: "PE" (PUT) or "CE" (CALL)
            entry_oi: Open Interest at entry time (e.g., 5_000_000)
            position_type: "LONG" or "SHORT"
        """
        self.entry_strike = entry_strike
        self.option_type = option_type
        self.entry_oi = entry_oi
        self.position_type = position_type
        self.oi_history = []

        logger.info(f"OI Monitor initialized: {position_type} position at {entry_strike} {option_type}, Entry OI: {entry_oi:,}")

    def check_oi_shift(self, current_oi: int, current_price: float) -> Dict:
        """
        Check if OI at entry level has shifted significantly

        Args:
            current_oi: Current Open Interest at entry strike
            current_price: Current market price

        Returns:
            {
                'action': 'HOLD' | 'WARNING' | 'EXIT_PARTIAL' | 'EXIT_ALL',
                'reason': str,
                'oi_change': int,
                'oi_change_pct': float,
                'alert_priority': 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
                'details': str
            }
        """
        # Calculate OI change
        oi_change = current_oi - self.entry_oi
        oi_change_pct = (oi_change / self.entry_oi) * 100 if self.entry_oi > 0 else 0

        # Store history
        self.oi_history.append({
            'timestamp': datetime.now(),
            'oi': current_oi,
            'change': oi_change,
            'change_pct': oi_change_pct,
            'price': current_price
        })

        # Check for OI unwinding (wall disappearing)
        if oi_change_pct < -30:
            # CRITICAL: 30%+ OI unwound
            return {
                'action': 'EXIT_ALL',
                'reason': f'CRITICAL: {abs(oi_change_pct):.1f}% OI unwinding - Wall collapsing!',
                'oi_change': oi_change,
                'oi_change_pct': oi_change_pct,
                'alert_priority': 'CRITICAL',
                'details': f'Entry OI: {self.entry_oi:,} → Current: {current_oi:,} ({oi_change:,})'
            }

        elif oi_change_pct < -20:
            # MAJOR: 20-30% OI unwound
            return {
                'action': 'EXIT_PARTIAL',
                'reason': f'MAJOR WARNING: {abs(oi_change_pct):.1f}% OI unwinding - Exit 50% position',
                'oi_change': oi_change,
                'oi_change_pct': oi_change_pct,
                'alert_priority': 'HIGH',
                'details': f'Entry OI: {self.entry_oi:,} → Current: {current_oi:,} ({oi_change:,})'
            }

        elif oi_change_pct < -10:
            # WARNING: 10-20% OI unwound
            return {
                'action': 'WARNING',
                'reason': f'WARNING: {abs(oi_change_pct):.1f}% OI unwinding - Move SL to breakeven',
                'oi_change': oi_change,
                'oi_change_pct': oi_change_pct,
                'alert_priority': 'MEDIUM',
                'details': f'Entry OI: {self.entry_oi:,} → Current: {current_oi:,} ({oi_change:,})'
            }

        # Check for OI buildup (strengthening)
        elif oi_change_pct > 20:
            # POSITIVE: OI increasing significantly
            return {
                'action': 'HOLD',
                'reason': f'POSITIVE: {oi_change_pct:.1f}% OI increase - Wall strengthening!',
                'oi_change': oi_change,
                'oi_change_pct': oi_change_pct,
                'alert_priority': 'LOW',
                'details': f'Entry OI: {self.entry_oi:,} → Current: {current_oi:,} ({oi_change:+,})'
            }

        else:
            # STABLE: OI change within ±10%
            return {
                'action': 'HOLD',
                'reason': f'OI stable ({oi_change_pct:+.1f}%)',
                'oi_change': oi_change,
                'oi_change_pct': oi_change_pct,
                'alert_priority': 'LOW',
                'details': f'Current OI: {current_oi:,}'
            }

    def check_opposite_side_buildup(self, option_chain: Dict, current_price: float) -> Dict:
        """
        Check if fresh OI is building on the opposite side (creating new barrier)

        Args:
            option_chain: Dictionary with strikes as keys, CE/PE data
            current_price: Current market price

        Returns:
            {
                'barrier_detected': bool,
                'barriers': List[Dict],
                'action': str,
                'reason': str,
                'alert_priority': str
            }
        """
        barriers_detected = []

        if self.position_type == "LONG":
            # For LONG positions, check CALL OI buildup above current price
            resistance_strikes = sorted([s for s in option_chain.keys() if s > current_price])[:10]

            for strike in resistance_strikes:
                if strike not in option_chain or 'CE' not in option_chain[strike]:
                    continue

                call_data = option_chain[strike]['CE']
                call_oi = call_data.get('openInterest', 0)
                call_oi_change = call_data.get('changeinOpenInterest', 0)

                # Detect significant fresh CALL OI buildup
                if call_oi_change > 500_000:  # 500K+ contracts added
                    distance = strike - current_price

                    barrier_info = {
                        'strike': strike,
                        'oi': call_oi,
                        'oi_change': call_oi_change,
                        'distance': distance,
                        'type': 'CALL Wall (Resistance)'
                    }

                    if distance <= 30:
                        barrier_info['severity'] = 'CRITICAL'
                        barrier_info['action'] = 'EXIT_PARTIAL'
                    elif distance <= 50:
                        barrier_info['severity'] = 'HIGH'
                        barrier_info['action'] = 'WARNING'
                    else:
                        barrier_info['severity'] = 'MEDIUM'
                        barrier_info['action'] = 'MONITOR'

                    barriers_detected.append(barrier_info)

        elif self.position_type == "SHORT":
            # For SHORT positions, check PUT OI buildup below current price
            support_strikes = sorted([s for s in option_chain.keys() if s < current_price], reverse=True)[:10]

            for strike in support_strikes:
                if strike not in option_chain or 'PE' not in option_chain[strike]:
                    continue

                put_data = option_chain[strike]['PE']
                put_oi = put_data.get('openInterest', 0)
                put_oi_change = put_data.get('changeinOpenInterest', 0)

                # Detect significant fresh PUT OI buildup
                if put_oi_change > 500_000:  # 500K+ contracts added
                    distance = current_price - strike

                    barrier_info = {
                        'strike': strike,
                        'oi': put_oi,
                        'oi_change': put_oi_change,
                        'distance': distance,
                        'type': 'PUT Wall (Support)'
                    }

                    if distance <= 30:
                        barrier_info['severity'] = 'CRITICAL'
                        barrier_info['action'] = 'EXIT_PARTIAL'
                    elif distance <= 50:
                        barrier_info['severity'] = 'HIGH'
                        barrier_info['action'] = 'WARNING'
                    else:
                        barrier_info['severity'] = 'MEDIUM'
                        barrier_info['action'] = 'MONITOR'

                    barriers_detected.append(barrier_info)

        # Determine overall action
        if not barriers_detected:
            return {
                'barrier_detected': False,
                'barriers': [],
                'action': 'HOLD',
                'reason': 'No significant opposite OI buildup detected',
                'alert_priority': 'LOW'
            }

        # Find most critical barrier
        critical_barriers = [b for b in barriers_detected if b['severity'] == 'CRITICAL']
        high_barriers = [b for b in barriers_detected if b['severity'] == 'HIGH']

        if critical_barriers:
            nearest = critical_barriers[0]
            return {
                'barrier_detected': True,
                'barriers': barriers_detected,
                'action': 'EXIT_PARTIAL',
                'reason': f"CRITICAL: Fresh {nearest['type']} at ₹{nearest['strike']} (+{nearest['oi_change']:,} OI, {nearest['distance']}pts away)",
                'alert_priority': 'CRITICAL'
            }
        elif high_barriers:
            nearest = high_barriers[0]
            return {
                'barrier_detected': True,
                'barriers': barriers_detected,
                'action': 'WARNING',
                'reason': f"WARNING: Fresh {nearest['type']} at ₹{nearest['strike']} (+{nearest['oi_change']:,} OI, {nearest['distance']}pts away)",
                'alert_priority': 'HIGH'
            }
        else:
            return {
                'barrier_detected': True,
                'barriers': barriers_detected,
                'action': 'MONITOR',
                'reason': f"Monitoring {len(barriers_detected)} new OI walls",
                'alert_priority': 'MEDIUM'
            }

    def get_oi_trend(self, lookback_minutes: int = 15) -> Dict:
        """
        Analyze OI trend over recent history

        Args:
            lookback_minutes: Minutes to look back (default: 15)

        Returns:
            {
                'trend': 'UNWINDING' | 'STABLE' | 'BUILDING',
                'avg_change_pct': float,
                'consistency': str
            }
        """
        if len(self.oi_history) < 3:
            return {
                'trend': 'STABLE',
                'avg_change_pct': 0,
                'consistency': 'INSUFFICIENT_DATA'
            }

        # Get recent history
        recent = self.oi_history[-10:]  # Last 10 data points

        # Calculate average change
        changes = [h['change_pct'] for h in recent]
        avg_change = sum(changes) / len(changes)

        # Determine trend
        if avg_change < -5:
            trend = 'UNWINDING'
        elif avg_change > 5:
            trend = 'BUILDING'
        else:
            trend = 'STABLE'

        # Check consistency
        positive_count = sum(1 for c in changes if c > 0)
        negative_count = sum(1 for c in changes if c < 0)

        if positive_count >= 7:
            consistency = 'CONSISTENTLY_BUILDING'
        elif negative_count >= 7:
            consistency = 'CONSISTENTLY_UNWINDING'
        else:
            consistency = 'MIXED'

        return {
            'trend': trend,
            'avg_change_pct': avg_change,
            'consistency': consistency,
            'data_points': len(recent)
        }
