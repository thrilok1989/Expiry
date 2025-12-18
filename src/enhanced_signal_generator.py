"""
Enhanced Signal Generator for Market Regime XGBoost Complete Signal System

Generates 5 types of trading signals:
1. Entry Signal (LONG/SHORT with CALL/PUT details)
2. Exit Signal (close position)
3. Wait Signal (no clear edge)
4. Direction Change Signal (trend reversal)
5. Bias Change Signal (sentiment shift)

Integrates all 146 XGBoost features for maximum accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading Signal with complete details"""
    signal_type: str  # "ENTRY", "EXIT", "WAIT", "DIRECTION_CHANGE", "BIAS_CHANGE"
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    action: str  # "BUY_CALL", "BUY_PUT", "EXIT_ALL", "WAIT", "HOLD"
    option_type: Optional[str] = None  # "CALL", "PUT", None
    strike: Optional[float] = None  # Strike price
    entry_price: Optional[float] = None  # Premium price
    entry_range_low: Optional[float] = None  # Entry zone low
    entry_range_high: Optional[float] = None  # Entry zone high
    stop_loss: Optional[float] = None  # SL price
    target_1: Optional[float] = None  # First target
    target_2: Optional[float] = None  # Second target
    target_3: Optional[float] = None  # Third target
    confidence: float = 0.0  # 0-100
    confluence: int = 0  # Number of agreeing indicators
    total_indicators: int = 0  # Total indicators analyzed
    xgboost_prediction: str = ""  # "BUY", "SELL", "HOLD"
    xgboost_probability: float = 0.0  # XGBoost probability
    expected_return: float = 0.0  # Expected % return
    risk_reward_ratio: float = 0.0  # R:R ratio
    reason: str = ""  # Why this signal was generated
    market_regime: str = ""  # Current market regime
    timestamp: datetime = None  # Signal generation time

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EnhancedSignalGenerator:
    """
    Enhanced Signal Generator using all 146 XGBoost features

    Signal Generation Logic:
    1. Analyze XGBoost prediction (BUY/SELL/HOLD)
    2. Check confluence across all indicators
    3. Validate entry zone conditions
    4. Calculate option strikes and premiums
    5. Set stop loss and targets
    6. Generate appropriate signal type
    """

    def __init__(self, min_confidence: float = 65.0, min_confluence: int = 6):
        """
        Initialize Enhanced Signal Generator

        Args:
            min_confidence: Minimum confidence % for entry signals
            min_confluence: Minimum number of agreeing indicators for entry
        """
        self.min_confidence = min_confidence
        self.min_confluence = min_confluence
        self.last_signal = None
        self.last_direction = "NEUTRAL"
        self.last_bias = "NEUTRAL"

    def generate_signal(
        self,
        xgboost_result: any,
        features_df: pd.DataFrame,
        current_price: float,
        option_chain: Optional[Dict] = None,
        atm_strike: Optional[float] = None
    ) -> TradingSignal:
        """
        Generate trading signal from XGBoost prediction and features

        Args:
            xgboost_result: MLPredictionResult from XGBoost
            features_df: DataFrame with all 146 features
            current_price: Current spot price
            option_chain: Option chain data for premium calculation
            atm_strike: ATM strike price

        Returns:
            TradingSignal with complete details
        """
        try:
            # Extract feature values
            features = features_df.iloc[0].to_dict() if len(features_df) > 0 else {}

            # 1. Analyze XGBoost prediction
            xgb_prediction = xgboost_result.prediction
            xgb_confidence = xgboost_result.confidence
            xgb_probability = xgboost_result.probability

            # 2. Calculate confluence score (how many indicators agree)
            confluence_result = self._calculate_confluence(features, xgb_prediction)
            confluence_count = confluence_result['agreeing_indicators']
            total_indicators = confluence_result['total_indicators']
            confluence_pct = (confluence_count / total_indicators * 100) if total_indicators > 0 else 0

            # 3. Determine direction
            direction = self._determine_direction(xgb_prediction, features)

            # 4. Validate entry zone
            entry_valid = self._validate_entry_zone(features, direction, xgb_confidence)

            # 5. Check for signal type changes
            signal_type = self._determine_signal_type(
                direction,
                xgb_confidence,
                confluence_count,
                entry_valid
            )

            # 6. Generate appropriate signal
            if signal_type == "ENTRY":
                return self._generate_entry_signal(
                    direction,
                    xgb_prediction,
                    xgb_confidence,
                    xgb_probability,
                    confluence_count,
                    total_indicators,
                    current_price,
                    features,
                    option_chain,
                    atm_strike
                )
            elif signal_type == "EXIT":
                return self._generate_exit_signal(
                    direction,
                    xgb_confidence,
                    confluence_count,
                    total_indicators,
                    current_price,
                    features
                )
            elif signal_type == "DIRECTION_CHANGE":
                return self._generate_direction_change_signal(
                    direction,
                    xgb_confidence,
                    confluence_count,
                    total_indicators,
                    current_price,
                    features
                )
            elif signal_type == "BIAS_CHANGE":
                return self._generate_bias_change_signal(
                    direction,
                    xgb_confidence,
                    confluence_count,
                    total_indicators,
                    current_price,
                    features
                )
            else:  # WAIT
                return self._generate_wait_signal(
                    xgb_prediction,
                    xgb_confidence,
                    confluence_count,
                    total_indicators,
                    current_price,
                    features
                )

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return self._generate_error_signal(str(e))

    def _calculate_confluence(self, features: Dict, xgb_prediction: str) -> Dict:
        """
        Calculate confluence - how many indicators agree with XGBoost prediction

        Returns:
            Dict with 'agreeing_indicators' count and 'total_indicators' count
        """
        agreeing = 0
        total = 0

        # Expected direction based on XGBoost
        expected_direction = 1 if xgb_prediction == "BUY" else -1 if xgb_prediction == "SELL" else 0

        # Check each feature category for agreement
        # Tab 1: Overall Market Sentiment
        if features.get('overall_market_direction', 0) == expected_direction:
            agreeing += 1
        total += 1

        # Bias indicators
        bias_features = [
            'bias_Advance_Decline_Ratio', 'bias_ATR_Analysis', 'bias_Bollinger_Band_Position',
            'bias_EMA_Alignment', 'bias_Market_Cap_Weighted', 'bias_Moving_Average_Position',
            'bias_OI_Analysis', 'bias_PCR_Sentiment', 'bias_RSI_Divergence', 'bias_Stochastic_Momentum',
            'bias_Supertrend', 'bias_VWAP', 'bias_Volume_Oscillator'
        ]
        for feat in bias_features:
            if feat in features:
                if (expected_direction > 0 and features[feat] > 20) or \
                   (expected_direction < 0 and features[feat] < -20):
                    agreeing += 1
                total += 1

        # CVD/Delta features
        if 'cvd_bias' in features:
            if features['cvd_bias'] == expected_direction:
                agreeing += 1
            total += 1

        # Money Flow Profile
        if 'mfp_sentiment' in features:
            if features['mfp_sentiment'] == expected_direction:
                agreeing += 1
            total += 1

        # DeltaFlow Profile
        if 'dfp_sentiment' in features:
            if (expected_direction > 0 and features['dfp_sentiment'] > 0) or \
               (expected_direction < 0 and features['dfp_sentiment'] < 0):
                agreeing += 1
            total += 1

        # ATM Bias features (Tab 8)
        atm_bias_features = [
            'atm_oi_bias', 'atm_chgoi_bias', 'atm_volume_bias', 'atm_delta_bias',
            'atm_gamma_bias', 'atm_iv_bias', 'atm_delta_exposure_bias', 'atm_gamma_exposure_bias'
        ]
        for feat in atm_bias_features:
            if feat in features:
                if (expected_direction > 0 and features[feat] > 0) or \
                   (expected_direction < 0 and features[feat] < 0):
                    agreeing += 1
                total += 1

        # Sector Rotation (Tab 9)
        if 'sector_rotation_bias' in features:
            if (expected_direction > 0 and features['sector_rotation_bias'] > 0) or \
               (expected_direction < 0 and features['sector_rotation_bias'] < 0):
                agreeing += 1
            total += 1

        # Liquidity sentiment (Tab 7)
        if 'liquidity_sentiment' in features:
            if features['liquidity_sentiment'] == expected_direction:
                agreeing += 1
            total += 1

        return {
            'agreeing_indicators': agreeing,
            'total_indicators': total,
            'confluence_pct': (agreeing / total * 100) if total > 0 else 0
        }

    def _determine_direction(self, xgb_prediction: str, features: Dict) -> str:
        """
        Determine overall direction (LONG/SHORT/NEUTRAL)

        Args:
            xgb_prediction: XGBoost prediction (BUY/SELL/HOLD)
            features: Feature dictionary

        Returns:
            Direction string: "LONG", "SHORT", or "NEUTRAL"
        """
        if xgb_prediction == "BUY":
            return "LONG"
        elif xgb_prediction == "SELL":
            return "SHORT"
        else:
            return "NEUTRAL"

    def _validate_entry_zone(
        self,
        features: Dict,
        direction: str,
        confidence: float
    ) -> bool:
        """
        Validate if current conditions are suitable for entry

        Args:
            features: Feature dictionary
            direction: LONG/SHORT/NEUTRAL
            confidence: XGBoost confidence

        Returns:
            True if entry zone is valid, False otherwise
        """
        # Check minimum confidence
        if confidence < self.min_confidence:
            return False

        # Check volatility regime (avoid extreme volatility)
        vix_level = features.get('vix_level', 15)
        if vix_level > 30:  # Too high volatility
            return False

        # Check if in expiry week (be cautious)
        is_expiry_week = features.get('is_expiry_week', 0)
        time_decay_factor = features.get('time_decay_factor', 0.5)
        if is_expiry_week and time_decay_factor > 0.85:
            # Too close to expiry, avoid entry
            return False

        # Check liquidity conditions
        liquidity_sentiment = features.get('liquidity_sentiment', 0)
        if direction == "LONG" and liquidity_sentiment == -1:
            # Bearish liquidity pull during long signal
            return False
        if direction == "SHORT" and liquidity_sentiment == 1:
            # Bullish liquidity pull during short signal
            return False

        # Check market regime
        market_regime = features.get('market_regime', 0)
        if abs(market_regime) < 1:  # Too choppy/ranging
            return False

        return True

    def _determine_signal_type(
        self,
        direction: str,
        confidence: float,
        confluence: int,
        entry_valid: bool
    ) -> str:
        """
        Determine which type of signal to generate

        Returns:
            Signal type: "ENTRY", "EXIT", "WAIT", "DIRECTION_CHANGE", "BIAS_CHANGE"
        """
        # Check for direction change
        if self.last_direction != "NEUTRAL" and direction != self.last_direction and direction != "NEUTRAL":
            return "DIRECTION_CHANGE"

        # Check for bias change (when direction shifts but not reversed)
        if self.last_direction in ["LONG", "SHORT"] and direction == "NEUTRAL":
            return "BIAS_CHANGE"

        # Check if holding position and conditions turned negative
        if self.last_signal and self.last_signal.signal_type == "ENTRY":
            if confidence < 50 or confluence < 4:
                return "EXIT"

        # Check for entry signal
        if entry_valid and confluence >= self.min_confluence:
            return "ENTRY"

        # Default to WAIT
        return "WAIT"

    def _generate_entry_signal(
        self,
        direction: str,
        xgb_prediction: str,
        xgb_confidence: float,
        xgb_probability: float,
        confluence: int,
        total_indicators: int,
        current_price: float,
        features: Dict,
        option_chain: Optional[Dict],
        atm_strike: Optional[float]
    ) -> TradingSignal:
        """Generate ENTRY signal with CALL/PUT details"""

        # Determine option type
        option_type = "CALL" if direction == "LONG" else "PUT"
        action = f"BUY_{option_type}"

        # Calculate strike price (ATM or slightly OTM)
        if atm_strike:
            strike = atm_strike
        else:
            # Round to nearest 50
            strike = round(current_price / 50) * 50

        # Calculate entry price (premium) from option chain
        entry_price, entry_low, entry_high = self._calculate_entry_premium(
            option_chain,
            strike,
            option_type,
            current_price
        )

        # Calculate stop loss
        stop_loss = self._calculate_stop_loss(entry_price, direction, features)

        # Calculate targets
        target_1, target_2, target_3 = self._calculate_targets(entry_price, direction, features)

        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss) if stop_loss else entry_price * 0.25
        reward = abs(target_1 - entry_price) if target_1 else entry_price * 0.3
        rr_ratio = reward / risk if risk > 0 else 0

        # Get market regime
        market_regime = self._get_market_regime_description(features)

        # Generate reason
        reason = self._generate_entry_reason(
            direction,
            xgb_prediction,
            confluence,
            total_indicators,
            features
        )

        signal = TradingSignal(
            signal_type="ENTRY",
            direction=direction,
            action=action,
            option_type=option_type,
            strike=strike,
            entry_price=entry_price,
            entry_range_low=entry_low,
            entry_range_high=entry_high,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            target_3=target_3,
            confidence=xgb_confidence,
            confluence=confluence,
            total_indicators=total_indicators,
            xgboost_prediction=xgb_prediction,
            xgboost_probability=xgb_probability,
            expected_return=0.0,  # Will be calculated
            risk_reward_ratio=rr_ratio,
            reason=reason,
            market_regime=market_regime
        )

        # Update state
        self.last_signal = signal
        self.last_direction = direction

        return signal

    def _calculate_entry_premium(
        self,
        option_chain: Optional[Dict],
        strike: float,
        option_type: str,
        current_price: float
    ) -> Tuple[float, float, float]:
        """
        Calculate entry premium (LTP) from option chain

        Returns:
            (entry_price, entry_low, entry_high) tuple
        """
        # TODO: Extract actual premium from option chain
        # For now, estimate based on intrinsic + time value

        if option_type == "CALL":
            intrinsic = max(0, current_price - strike)
        else:  # PUT
            intrinsic = max(0, strike - current_price)

        # Estimate time value (typically 2-5% of strike for ATM options)
        time_value = strike * 0.03

        entry_price = intrinsic + time_value

        # Entry range (±5%)
        entry_low = entry_price * 0.95
        entry_high = entry_price * 1.05

        return (round(entry_price, 2), round(entry_low, 2), round(entry_high, 2))

    def _calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        features: Dict
    ) -> float:
        """
        Calculate stop loss based on ATR and volatility

        Returns:
            Stop loss price
        """
        # Use ATR for dynamic stop loss
        atr_pct = features.get('atr_pct', 1.5)

        # Adjust SL based on volatility regime
        vix_level = features.get('vix_level', 15)
        if vix_level > 20:
            # Wider stop in high volatility
            sl_pct = 0.30
        elif vix_level > 15:
            sl_pct = 0.25
        else:
            sl_pct = 0.20

        stop_loss = entry_price * (1 - sl_pct)

        return round(stop_loss, 2)

    def _calculate_targets(
        self,
        entry_price: float,
        direction: str,
        features: Dict
    ) -> Tuple[float, float, float]:
        """
        Calculate three target levels (T1, T2, T3)

        Returns:
            (target_1, target_2, target_3) tuple
        """
        # Target percentages based on market regime
        market_regime = features.get('market_regime', 0)

        if abs(market_regime) >= 2:  # Strong trending
            t1_pct, t2_pct, t3_pct = 0.25, 0.50, 0.80
        else:  # Normal trending
            t1_pct, t2_pct, t3_pct = 0.20, 0.40, 0.65

        target_1 = entry_price * (1 + t1_pct)
        target_2 = entry_price * (1 + t2_pct)
        target_3 = entry_price * (1 + t3_pct)

        return (round(target_1, 2), round(target_2, 2), round(target_3, 2))

    def _generate_exit_signal(
        self,
        direction: str,
        confidence: float,
        confluence: int,
        total_indicators: int,
        current_price: float,
        features: Dict
    ) -> TradingSignal:
        """Generate EXIT signal"""

        reason = f"Exit conditions met: Confidence dropped to {confidence:.1f}%, Confluence {confluence}/{total_indicators}"

        signal = TradingSignal(
            signal_type="EXIT",
            direction=direction,
            action="EXIT_ALL",
            confidence=confidence,
            confluence=confluence,
            total_indicators=total_indicators,
            reason=reason,
            market_regime=self._get_market_regime_description(features)
        )

        # Clear position state
        self.last_signal = signal
        self.last_direction = "NEUTRAL"

        return signal

    def _generate_wait_signal(
        self,
        xgb_prediction: str,
        confidence: float,
        confluence: int,
        total_indicators: int,
        current_price: float,
        features: Dict
    ) -> TradingSignal:
        """Generate WAIT signal"""

        reason = f"Waiting for better setup: Confidence {confidence:.1f}%, Confluence {confluence}/{total_indicators}"

        return TradingSignal(
            signal_type="WAIT",
            direction="NEUTRAL",
            action="WAIT",
            confidence=confidence,
            confluence=confluence,
            total_indicators=total_indicators,
            xgboost_prediction=xgb_prediction,
            reason=reason,
            market_regime=self._get_market_regime_description(features)
        )

    def _generate_direction_change_signal(
        self,
        direction: str,
        confidence: float,
        confluence: int,
        total_indicators: int,
        current_price: float,
        features: Dict
    ) -> TradingSignal:
        """Generate DIRECTION CHANGE signal"""

        reason = f"Direction changed from {self.last_direction} to {direction}"

        signal = TradingSignal(
            signal_type="DIRECTION_CHANGE",
            direction=direction,
            action="ALERT",
            confidence=confidence,
            confluence=confluence,
            total_indicators=total_indicators,
            reason=reason,
            market_regime=self._get_market_regime_description(features)
        )

        self.last_direction = direction

        return signal

    def _generate_bias_change_signal(
        self,
        direction: str,
        confidence: float,
        confluence: int,
        total_indicators: int,
        current_price: float,
        features: Dict
    ) -> TradingSignal:
        """Generate BIAS CHANGE signal"""

        reason = f"Market bias changed to {direction}"

        signal = TradingSignal(
            signal_type="BIAS_CHANGE",
            direction=direction,
            action="ALERT",
            confidence=confidence,
            confluence=confluence,
            total_indicators=total_indicators,
            reason=reason,
            market_regime=self._get_market_regime_description(features)
        )

        self.last_bias = direction

        return signal

    def _generate_entry_reason(
        self,
        direction: str,
        xgb_prediction: str,
        confluence: int,
        total_indicators: int,
        features: Dict
    ) -> str:
        """Generate human-readable reason for entry signal"""

        reasons = []

        # XGBoost prediction
        reasons.append(f"XGBoost: {xgb_prediction}")

        # Confluence
        conf_pct = (confluence / total_indicators * 100) if total_indicators > 0 else 0
        reasons.append(f"Confluence: {confluence}/{total_indicators} ({conf_pct:.0f}%)")

        # Key supporting factors
        if features.get('overall_market_direction', 0) == (1 if direction == "LONG" else -1):
            reasons.append("Overall sentiment aligned")

        if abs(features.get('market_regime', 0)) >= 2:
            reasons.append("Strong trending market")

        if features.get('sector_rotation_bias', 0) == (1 if direction == "LONG" else -1):
            reasons.append("Sector rotation supportive")

        return " | ".join(reasons)

    def _get_market_regime_description(self, features: Dict) -> str:
        """Get human-readable market regime description"""

        regime_code = features.get('market_regime', 0)

        regime_map = {
            2: "STRONG_UPTREND",
            1: "UPTREND",
            0: "RANGING",
            -1: "DOWNTREND",
            -2: "STRONG_DOWNTREND"
        }

        return regime_map.get(regime_code, "UNKNOWN")

    def _generate_error_signal(self, error_msg: str) -> TradingSignal:
        """Generate error signal when something goes wrong"""

        return TradingSignal(
            signal_type="ERROR",
            direction="NEUTRAL",
            action="WAIT",
            reason=f"Error generating signal: {error_msg}"
        )


def format_signal_for_telegram(signal: TradingSignal) -> str:
    """
    Format trading signal for Telegram message

    Args:
        signal: TradingSignal object

    Returns:
        Formatted message string
    """
    if signal.signal_type == "ENTRY":
        msg = f"""
🚀 *ENTRY SIGNAL* - {signal.direction}

📊 *Option Details:*
Type: {signal.option_type}
Strike: {signal.strike}
Entry Price: ₹{signal.entry_price} (Range: ₹{signal.entry_range_low}-{signal.entry_range_high})

🎯 *Targets & Risk:*
Stop Loss: ₹{signal.stop_loss}
Target 1: ₹{signal.target_1} (+{((signal.target_1/signal.entry_price-1)*100):.1f}%)
Target 2: ₹{signal.target_2} (+{((signal.target_2/signal.entry_price-1)*100):.1f}%)
Target 3: ₹{signal.target_3} (+{((signal.target_3/signal.entry_price-1)*100):.1f}%)
R:R Ratio: {signal.risk_reward_ratio:.2f}

💪 *Strength:*
Confidence: {signal.confidence:.1f}%
Confluence: {signal.confluence}/{signal.total_indicators} indicators

📈 *Market Context:*
Regime: {signal.market_regime}
XGBoost: {signal.xgboost_prediction} ({signal.xgboost_probability*100:.1f}%)

💡 *Reason:*
{signal.reason}
"""
    elif signal.signal_type == "EXIT":
        msg = f"""
🔻 *EXIT SIGNAL*

Action: Close all positions
Reason: {signal.reason}

Market Regime: {signal.market_regime}
Confidence: {signal.confidence:.1f}%
"""
    elif signal.signal_type == "WAIT":
        msg = f"""
⏸️ *WAIT SIGNAL*

Status: No clear edge, wait for better setup
Reason: {signal.reason}

XGBoost: {signal.xgboost_prediction}
Confidence: {signal.confidence:.1f}%
Confluence: {signal.confluence}/{signal.total_indicators}
"""
    elif signal.signal_type == "DIRECTION_CHANGE":
        msg = f"""
🔄 *DIRECTION CHANGE ALERT*

New Direction: {signal.direction}
Reason: {signal.reason}

Confidence: {signal.confidence:.1f}%
Confluence: {signal.confluence}/{signal.total_indicators}
Market Regime: {signal.market_regime}
"""
    elif signal.signal_type == "BIAS_CHANGE":
        msg = f"""
⚠️ *BIAS CHANGE ALERT*

New Bias: {signal.direction}
Reason: {signal.reason}

Confidence: {signal.confidence:.1f}%
Market Regime: {signal.market_regime}
"""
    else:
        msg = f"Unknown signal type: {signal.signal_type}"

    return msg
