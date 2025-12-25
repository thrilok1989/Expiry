"""
Unified Exit Coordinator - Master exit decision system
Monitors all 9 factors and triggers appropriate exit actions
"""

from datetime import datetime
from typing import Dict, List, Optional
import logging
from oi_shift_monitor import OIShiftMonitor
from volume_spike_monitor import VolumeSpikeMonitor

logger = logging.getLogger(__name__)


class ExitCoordinator:
    """
    Coordinates all exit monitoring systems
    Combines 9 factors to make intelligent exit decisions
    """

    def __init__(self, position_type: str, entry_price: float, entry_strike: int,
                 entry_oi: int, option_type: str):
        """
        Initialize exit coordinator

        Args:
            position_type: "LONG" or "SHORT"
            entry_price: Entry price
            entry_strike: Entry strike (for OI monitoring)
            entry_oi: Entry OI (for OI monitoring)
            option_type: "PE" or "CE"
        """
        self.position_type = position_type
        self.entry_price = entry_price
        self.entry_time = datetime.now()

        # Initialize monitoring systems
        self.oi_monitor = OIShiftMonitor(
            entry_strike=entry_strike,
            option_type=option_type,
            entry_oi=entry_oi,
            position_type=position_type
        )

        self.volume_monitor = VolumeSpikeMonitor(
            position_type=position_type,
            lookback_periods=20
        )

        # Track entry conditions for comparison
        self.entry_conditions = {
            'regime': None,
            'atm_bias': None,
            'rsi_divergence': None,
            'money_flow': None,
            'delta': None
        }

        # Exit factor history
        self.exit_checks = []

        logger.info(f"Exit Coordinator initialized for {position_type} at ₹{entry_price}")

    def set_entry_conditions(self, regime: str = None, atm_bias: str = None,
                            rsi_divergence: bool = None, money_flow: str = None,
                            delta: float = None):
        """
        Set entry conditions for comparison

        Args:
            regime: Entry regime (e.g., "WEAK_DOWNTREND")
            atm_bias: Entry ATM bias (e.g., "CALL SELLERS")
            rsi_divergence: RSI divergence present at entry
            money_flow: Money flow direction at entry
            delta: Delta value at entry
        """
        self.entry_conditions = {
            'regime': regime,
            'atm_bias': atm_bias,
            'rsi_divergence': rsi_divergence,
            'money_flow': money_flow,
            'delta': delta
        }

    def check_all_exit_factors(self, current_data: Dict) -> Dict:
        """
        Check all 9 exit factors and return comprehensive result

        Args:
            current_data: {
                'price': float,
                'option_chain': dict,
                'current_oi': int,
                'regime': str,
                'atm_bias': str,
                'volume_candle': dict,
                'recent_candles': list,
                'rsi_data': dict,
                'money_flow': dict,
                'deltaflow': dict,
                'df': DataFrame (for volume baseline)
            }

        Returns:
            {
                'exit_action': 'HOLD' | 'TIGHTEN_SL' | 'EXIT_PARTIAL' | 'EXIT_ALL',
                'exit_priority': 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
                'factors_triggered': int,
                'critical_factors': list,
                'factor_details': dict,
                'telegram_alerts': list,
                'reason': str
            }
        """
        current_price = current_data['price']
        exit_factors = {}
        critical_factors = []
        telegram_alerts = []

        # ===================================================================
        # FACTOR 1: OI UNWINDING (CRITICAL)
        # ===================================================================
        try:
            current_oi = current_data.get('current_oi', 0)
            oi_check = self.oi_monitor.check_oi_shift(current_oi, current_price)

            exit_factors['oi_unwinding'] = {
                'triggered': oi_check['action'] in ['EXIT_PARTIAL', 'EXIT_ALL'],
                'action': oi_check['action'],
                'severity': oi_check['alert_priority'],
                'details': oi_check
            }

            if oi_check['action'] == 'EXIT_ALL':
                critical_factors.append('OI_UNWINDING_CRITICAL')
                telegram_alerts.append({
                    'type': 'oi_unwinding',
                    'data': oi_check
                })

        except Exception as e:
            logger.error(f"Error checking OI unwinding: {e}")
            exit_factors['oi_unwinding'] = {'triggered': False, 'error': str(e)}

        # ===================================================================
        # FACTOR 2: OPPOSITE OI BUILDUP
        # ===================================================================
        try:
            option_chain = current_data.get('option_chain', {})
            if option_chain:
                opposite_oi_check = self.oi_monitor.check_opposite_side_buildup(
                    option_chain, current_price
                )

                exit_factors['opposite_oi_buildup'] = {
                    'triggered': opposite_oi_check['barrier_detected'],
                    'action': opposite_oi_check['action'],
                    'severity': opposite_oi_check['alert_priority'],
                    'details': opposite_oi_check
                }

                if opposite_oi_check['alert_priority'] == 'CRITICAL':
                    telegram_alerts.append({
                        'type': 'opposite_oi_buildup',
                        'data': opposite_oi_check
                    })
            else:
                exit_factors['opposite_oi_buildup'] = {'triggered': False}

        except Exception as e:
            logger.error(f"Error checking opposite OI buildup: {e}")
            exit_factors['opposite_oi_buildup'] = {'triggered': False, 'error': str(e)}

        # ===================================================================
        # FACTOR 3: VOLUME SPIKE (CRITICAL)
        # ===================================================================
        try:
            # Update volume baseline
            df = current_data.get('df')
            if df is not None and len(df) > 0:
                self.volume_monitor.update_volume_baseline(df)

            # Check current candle
            volume_candle = current_data.get('volume_candle', {})
            if volume_candle:
                volume_check = self.volume_monitor.check_volume_spike(volume_candle)

                exit_factors['volume_spike'] = {
                    'triggered': volume_check['spike_detected'],
                    'action': volume_check['action'],
                    'severity': volume_check['severity'],
                    'details': volume_check
                }

                if volume_check['action'] == 'EXIT_ALL':
                    critical_factors.append('VOLUME_SPIKE_CRITICAL')
                    telegram_alerts.append({
                        'type': 'volume_spike',
                        'data': volume_check
                    })
                elif volume_check['action'] == 'EXIT_PARTIAL':
                    telegram_alerts.append({
                        'type': 'volume_spike',
                        'data': volume_check
                    })
            else:
                exit_factors['volume_spike'] = {'triggered': False}

        except Exception as e:
            logger.error(f"Error checking volume spike: {e}")
            exit_factors['volume_spike'] = {'triggered': False, 'error': str(e)}

        # ===================================================================
        # FACTOR 4: VOLUME ABSORPTION
        # ===================================================================
        try:
            recent_candles = current_data.get('recent_candles', [])
            if len(recent_candles) >= 3:
                absorption_check = self.volume_monitor.detect_absorption(recent_candles)

                exit_factors['volume_absorption'] = {
                    'triggered': absorption_check.get('absorption_detected', False),
                    'action': absorption_check.get('action', 'HOLD'),
                    'details': absorption_check
                }

                if absorption_check.get('absorption_detected'):
                    telegram_alerts.append({
                        'type': 'volume_absorption',
                        'data': absorption_check
                    })
            else:
                exit_factors['volume_absorption'] = {'triggered': False}

        except Exception as e:
            logger.error(f"Error checking volume absorption: {e}")
            exit_factors['volume_absorption'] = {'triggered': False, 'error': str(e)}

        # ===================================================================
        # FACTOR 5: REGIME FLIP
        # ===================================================================
        try:
            current_regime = current_data.get('regime')
            entry_regime = self.entry_conditions['regime']

            regime_flipped = False
            if entry_regime and current_regime:
                # Check for significant regime change
                if self.position_type == "LONG":
                    # LONG position - exit if regime turns bearish
                    if 'DOWNTREND' in current_regime and 'UPTREND' in entry_regime:
                        regime_flipped = True
                    elif 'RANGING' in current_regime and 'UPTREND' in entry_regime:
                        regime_flipped = True
                elif self.position_type == "SHORT":
                    # SHORT position - exit if regime turns bullish
                    if 'UPTREND' in current_regime and 'DOWNTREND' in entry_regime:
                        regime_flipped = True
                    elif 'RANGING' in current_regime and 'DOWNTREND' in entry_regime:
                        regime_flipped = True

            exit_factors['regime_flip'] = {
                'triggered': regime_flipped,
                'entry_regime': entry_regime,
                'current_regime': current_regime,
                'action': 'EXIT_PARTIAL' if regime_flipped else 'HOLD'
            }

        except Exception as e:
            logger.error(f"Error checking regime flip: {e}")
            exit_factors['regime_flip'] = {'triggered': False, 'error': str(e)}

        # ===================================================================
        # FACTOR 6: ATM BIAS FLIP
        # ===================================================================
        try:
            current_atm_bias = current_data.get('atm_bias')
            entry_atm_bias = self.entry_conditions['atm_bias']

            atm_flipped = False
            if entry_atm_bias and current_atm_bias:
                # Check for ATM bias flip
                if self.position_type == "LONG":
                    # LONG - exit if flips to CALL SELLERS
                    if 'CALL SELLERS' in current_atm_bias and 'PUT SELLERS' in entry_atm_bias:
                        atm_flipped = True
                elif self.position_type == "SHORT":
                    # SHORT - exit if flips to PUT SELLERS
                    if 'PUT SELLERS' in current_atm_bias and 'CALL SELLERS' in entry_atm_bias:
                        atm_flipped = True

            exit_factors['atm_bias_flip'] = {
                'triggered': atm_flipped,
                'entry_bias': entry_atm_bias,
                'current_bias': current_atm_bias,
                'action': 'EXIT_PARTIAL' if atm_flipped else 'HOLD'
            }

        except Exception as e:
            logger.error(f"Error checking ATM bias flip: {e}")
            exit_factors['atm_bias_flip'] = {'triggered': False, 'error': str(e)}

        # ===================================================================
        # FACTOR 7: DELTA FLIP
        # ===================================================================
        try:
            deltaflow = current_data.get('deltaflow', {})
            current_delta = deltaflow.get('current_delta', 0)
            entry_delta = self.entry_conditions.get('delta', 0)

            delta_flipped = False
            if entry_delta is not None:
                # Check for significant delta reversal
                if self.position_type == "LONG":
                    if current_delta < -3000 and entry_delta > 0:
                        delta_flipped = True
                elif self.position_type == "SHORT":
                    if current_delta > 3000 and entry_delta < 0:
                        delta_flipped = True

            exit_factors['delta_flip'] = {
                'triggered': delta_flipped,
                'entry_delta': entry_delta,
                'current_delta': current_delta,
                'action': 'TIGHTEN_SL' if delta_flipped else 'HOLD'
            }

        except Exception as e:
            logger.error(f"Error checking delta flip: {e}")
            exit_factors['delta_flip'] = {'triggered': False, 'error': str(e)}

        # ===================================================================
        # FACTOR 8: MONEY FLOW FLIP
        # ===================================================================
        try:
            money_flow = current_data.get('money_flow', {})
            current_flow = money_flow.get('direction', 'NEUTRAL')
            entry_flow = self.entry_conditions.get('money_flow')

            flow_flipped = False
            if entry_flow:
                if self.position_type == "LONG":
                    if current_flow == 'SELLING' and entry_flow == 'BUYING':
                        flow_flipped = True
                elif self.position_type == "SHORT":
                    if current_flow == 'BUYING' and entry_flow == 'SELLING':
                        flow_flipped = True

            exit_factors['money_flow_flip'] = {
                'triggered': flow_flipped,
                'entry_flow': entry_flow,
                'current_flow': current_flow,
                'action': 'TIGHTEN_SL' if flow_flipped else 'HOLD'
            }

        except Exception as e:
            logger.error(f"Error checking money flow flip: {e}")
            exit_factors['money_flow_flip'] = {'triggered': False, 'error': str(e)}

        # ===================================================================
        # FACTOR 9: RSI DIVERGENCE INVALID
        # ===================================================================
        try:
            rsi_data = current_data.get('rsi_data', {})
            entry_had_divergence = self.entry_conditions.get('rsi_divergence', False)

            divergence_invalid = False
            if entry_had_divergence:
                # Check if divergence still valid
                current_divergence = rsi_data.get('divergence_active', False)
                if not current_divergence:
                    divergence_invalid = True

            exit_factors['rsi_divergence_invalid'] = {
                'triggered': divergence_invalid,
                'entry_had_divergence': entry_had_divergence,
                'action': 'TIGHTEN_SL' if divergence_invalid else 'HOLD'
            }

        except Exception as e:
            logger.error(f"Error checking RSI divergence: {e}")
            exit_factors['rsi_divergence_invalid'] = {'triggered': False, 'error': str(e)}

        # ===================================================================
        # AGGREGATE EXIT DECISION
        # ===================================================================
        result = self._aggregate_exit_decision(exit_factors, critical_factors, telegram_alerts)

        # Store check in history
        self.exit_checks.append({
            'timestamp': datetime.now(),
            'price': current_price,
            'result': result
        })

        return result

    def _aggregate_exit_decision(self, exit_factors: Dict, critical_factors: List,
                                 telegram_alerts: List) -> Dict:
        """
        Aggregate all exit factors into final decision

        Args:
            exit_factors: Dictionary of all factor checks
            critical_factors: List of critical factors triggered
            telegram_alerts: List of alerts to send

        Returns:
            Final exit decision
        """
        # Count triggered factors
        factors_triggered = sum(1 for f in exit_factors.values()
                               if isinstance(f, dict) and f.get('triggered', False))

        # Check for critical factors (immediate exit)
        if critical_factors:
            return {
                'exit_action': 'EXIT_ALL',
                'exit_priority': 'CRITICAL',
                'factors_triggered': factors_triggered,
                'critical_factors': critical_factors,
                'factor_details': exit_factors,
                'telegram_alerts': telegram_alerts,
                'reason': f"CRITICAL: {', '.join(critical_factors)}"
            }

        # Check for EXIT_ALL actions
        exit_all_count = sum(1 for f in exit_factors.values()
                            if isinstance(f, dict) and f.get('action') == 'EXIT_ALL')
        if exit_all_count > 0:
            return {
                'exit_action': 'EXIT_ALL',
                'exit_priority': 'CRITICAL',
                'factors_triggered': factors_triggered,
                'critical_factors': critical_factors,
                'factor_details': exit_factors,
                'telegram_alerts': telegram_alerts,
                'reason': f"{exit_all_count} factor(s) triggered EXIT_ALL"
            }

        # Check for 4+ factors (major shift)
        if factors_triggered >= 4:
            return {
                'exit_action': 'EXIT_ALL',
                'exit_priority': 'HIGH',
                'factors_triggered': factors_triggered,
                'critical_factors': critical_factors,
                'factor_details': exit_factors,
                'telegram_alerts': telegram_alerts,
                'reason': f"Major market shift: {factors_triggered} factors triggered"
            }

        # Check for 3 factors (warning - partial exit)
        if factors_triggered >= 3:
            return {
                'exit_action': 'EXIT_PARTIAL',
                'exit_priority': 'HIGH',
                'factors_triggered': factors_triggered,
                'critical_factors': critical_factors,
                'factor_details': exit_factors,
                'telegram_alerts': telegram_alerts,
                'reason': f"Warning: {factors_triggered} factors triggered"
            }

        # Check for 2 factors (tighten SL)
        if factors_triggered >= 2:
            return {
                'exit_action': 'TIGHTEN_SL',
                'exit_priority': 'MEDIUM',
                'factors_triggered': factors_triggered,
                'critical_factors': critical_factors,
                'factor_details': exit_factors,
                'telegram_alerts': telegram_alerts,
                'reason': f"{factors_triggered} factors triggered - tighten stop"
            }

        # All clear - hold position
        return {
            'exit_action': 'HOLD',
            'exit_priority': 'LOW',
            'factors_triggered': factors_triggered,
            'critical_factors': critical_factors,
            'factor_details': exit_factors,
            'telegram_alerts': telegram_alerts,
            'reason': 'All systems normal - hold position'
        }

    def get_exit_summary(self) -> Dict:
        """
        Get summary of exit monitoring status

        Returns:
            Summary dict
        """
        if not self.exit_checks:
            return {
                'total_checks': 0,
                'last_check': None,
                'current_status': 'NOT_STARTED'
            }

        last_check = self.exit_checks[-1]

        return {
            'total_checks': len(self.exit_checks),
            'last_check': last_check['timestamp'],
            'current_status': last_check['result']['exit_action'],
            'factors_triggered': last_check['result']['factors_triggered'],
            'position_age_minutes': (datetime.now() - self.entry_time).total_seconds() / 60
        }
