"""
XGBoost ML Analyzer
Uses XGBoost to analyze ALL data from all tabs and make predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MLPredictionResult:
    """ML Prediction Result"""
    prediction: str  # "BUY", "SELL", "HOLD"
    probability: float  # 0-1
    confidence: float  # 0-100
    feature_importance: Dict[str, float]
    all_probabilities: Dict[str, float]  # Probabilities for all classes
    expected_return: float  # Expected % return
    risk_score: float  # 0-100
    recommendation: str
    model_version: str


class XGBoostMLAnalyzer:
    """
    XGBoost ML Analyzer

    Analyzes ALL features from ALL tabs/modules and makes ML-based predictions

    Features used:
    - Technical indicators (13 from Bias Analysis)
    - Price action (BOS, CHOCH, Fibonacci)
    - Volatility metrics (VIX, ATR, regime)
    - Option chain (OI, PCR, IV, Greeks)
    - CVD & Delta metrics
    - Institutional flow signatures
    - Liquidity levels
    - Sentiment scores
    - Market regime features
    - Money Flow Profile (POC, volume distribution, sentiment)
    - DeltaFlow Profile (delta per price level, strong buy/sell levels)
    """

    def __init__(self):
        """Initialize XGBoost ML Analyzer"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.model_version = "v1.0_production"

        # XGBoost parameters (optimized for trading)
        self.params = {
            'objective': 'multi:softprob',
            'num_class': 3,  # BUY, SELL, HOLD
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }

    def extract_features_from_all_tabs(
        self,
        df: pd.DataFrame,
        bias_results: Optional[Dict] = None,
        option_chain: Optional[Dict] = None,
        volatility_result: Optional[any] = None,
        oi_trap_result: Optional[any] = None,
        cvd_result: Optional[any] = None,
        participant_result: Optional[any] = None,
        liquidity_result: Optional[any] = None,
        ml_regime_result: Optional[any] = None,
        sentiment_score: float = 0.0,
        option_screener_data: Optional[Dict] = None,
        money_flow_signals: Optional[Dict] = None,
        deltaflow_signals: Optional[Dict] = None,
        overall_sentiment_data: Optional[Dict] = None,
        enhanced_market_data: Optional[Dict] = None,
        nifty_screener_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Extract ALL features from ALL modules into a single feature vector

        Returns a DataFrame with 1 row containing all features
        """
        features = {}

        # ========== PRICE FEATURES ==========
        if len(df) > 0:
            current_price = df['close'].iloc[-1]
            features['price_current'] = current_price

            # Price momentum
            if len(df) >= 5:
                features['price_change_1'] = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
                features['price_change_5'] = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100

            if len(df) >= 20:
                features['price_change_20'] = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100

            # Volatility
            if 'atr' in df.columns:
                features['atr'] = df['atr'].iloc[-1]
                features['atr_pct'] = (df['atr'].iloc[-1] / current_price) * 100

        # ========== BIAS ANALYSIS FEATURES (13 indicators) ==========
        if bias_results:
            for indicator, data in bias_results.items():
                if isinstance(data, dict) and 'bias_score' in data:
                    features[f'bias_{indicator}'] = data['bias_score']

        # ========== VOLATILITY REGIME FEATURES ==========
        if volatility_result:
            features['vix_level'] = volatility_result.vix_level
            features['vix_percentile'] = volatility_result.vix_percentile
            features['atr_percentile'] = volatility_result.atr_percentile
            features['iv_rv_ratio'] = volatility_result.iv_rv_ratio
            features['regime_strength'] = volatility_result.regime_strength
            features['compression_score'] = volatility_result.compression_score
            features['gamma_flip'] = 1 if volatility_result.gamma_flip_detected else 0
            features['expiry_week'] = 1 if volatility_result.is_expiry_week else 0

            # One-hot encode regime
            regime_map = {
                "Low Volatility": 1,
                "Normal Volatility": 2,
                "High Volatility": 3,
                "Extreme Volatility": 4,
                "Regime Change": 5
            }
            features['volatility_regime'] = regime_map.get(volatility_result.regime.value, 2)

        # ========== OI TRAP FEATURES ==========
        if oi_trap_result:
            features['trap_detected'] = 1 if oi_trap_result.trap_detected else 0
            features['trap_probability'] = oi_trap_result.trap_probability
            features['retail_trap_score'] = oi_trap_result.retail_trap_score
            features['oi_manipulation_score'] = oi_trap_result.oi_manipulation_score

            # Encode trapped direction
            direction_map = {"CALL_BUYERS": 1, "PUT_BUYERS": -1, "BOTH": 0, "NONE": 0}
            features['trapped_direction'] = direction_map.get(oi_trap_result.trapped_direction, 0)

        # ========== CVD FEATURES ==========
        if cvd_result:
            features['cvd_value'] = cvd_result.cvd
            features['delta_imbalance'] = cvd_result.delta_imbalance
            features['orderflow_strength'] = cvd_result.orderflow_strength
            features['delta_divergence'] = 1 if cvd_result.delta_divergence_detected else 0
            features['delta_absorption'] = 1 if cvd_result.delta_absorption_detected else 0
            features['delta_spike'] = 1 if cvd_result.delta_spike_detected else 0
            features['institutional_sweep'] = 1 if cvd_result.institutional_sweep else 0

            # Encode bias
            bias_map = {"Bullish": 1, "Bearish": -1, "Neutral": 0}
            features['cvd_bias'] = bias_map.get(cvd_result.bias, 0)

        # ========== INSTITUTIONAL/RETAIL FEATURES ==========
        if participant_result:
            features['institutional_confidence'] = participant_result.institutional_confidence
            features['retail_confidence'] = participant_result.retail_confidence
            features['smart_money'] = 1 if participant_result.smart_money_detected else 0
            features['dumb_money'] = 1 if participant_result.dumb_money_detected else 0

            # Encode participant type
            part_map = {"Institutional": 1, "Retail": -1, "Mixed": 0, "Unknown": 0}
            participant_val = part_map.get(str(participant_result.dominant_participant.value), 0)
            features['dominant_participant'] = participant_val

        # ========== LIQUIDITY FEATURES ==========
        if liquidity_result:
            features['primary_target'] = liquidity_result.primary_target
            features['gravity_strength'] = liquidity_result.gravity_strength
            features['num_support_zones'] = len(liquidity_result.support_zones)
            features['num_resistance_zones'] = len(liquidity_result.resistance_zones)
            features['num_hvn_zones'] = len(liquidity_result.hvn_zones)
            features['num_fvg'] = len(liquidity_result.fair_value_gaps)
            features['num_gamma_walls'] = len(liquidity_result.gamma_walls)

            # Distance to target
            if 'price_current' in features and features['primary_target'] != 0:
                features['target_distance_pct'] = (liquidity_result.primary_target - features['price_current']) / features['price_current'] * 100

        # ========== MONEY FLOW PROFILE FEATURES ==========
        if money_flow_signals and money_flow_signals.get('success'):
            features['mfp_poc_price'] = money_flow_signals['poc_price']
            features['mfp_bullish_pct'] = money_flow_signals['bullish_volume_pct']
            features['mfp_bearish_pct'] = money_flow_signals['bearish_volume_pct']
            features['mfp_distance_from_poc_pct'] = money_flow_signals['distance_from_poc_pct']
            features['mfp_num_hv_levels'] = len(money_flow_signals['high_volume_levels'])
            features['mfp_num_lv_levels'] = len(money_flow_signals['low_volume_levels'])

            # Encode sentiment
            sentiment_map = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}
            features['mfp_sentiment'] = sentiment_map.get(money_flow_signals['sentiment'], 0)

            # Encode price position
            position_map = {"Above POC": 1, "At POC": 0, "Below POC": -1}
            features['mfp_price_position'] = position_map.get(money_flow_signals['price_position'], 0)

        # ========== DELTAFLOW PROFILE FEATURES ==========
        if deltaflow_signals and deltaflow_signals.get('success'):
            features['dfp_overall_delta'] = deltaflow_signals['overall_delta']
            features['dfp_bull_pct'] = deltaflow_signals['overall_bull_pct']
            features['dfp_bear_pct'] = deltaflow_signals['overall_bear_pct']
            features['dfp_poc_price'] = deltaflow_signals['poc_price']
            features['dfp_distance_from_poc_pct'] = deltaflow_signals['distance_from_poc_pct']
            features['dfp_num_strong_buy'] = len(deltaflow_signals['strong_buy_levels'])
            features['dfp_num_strong_sell'] = len(deltaflow_signals['strong_sell_levels'])
            features['dfp_num_absorption'] = len(deltaflow_signals['absorption_zones'])

            # Encode sentiment
            sentiment_map = {
                "STRONG BULLISH": 2,
                "BULLISH": 1,
                "NEUTRAL": 0,
                "BEARISH": -1,
                "STRONG BEARISH": -2
            }
            features['dfp_sentiment'] = sentiment_map.get(deltaflow_signals['sentiment'], 0)

            # Encode price position
            position_map = {"Above POC": 1, "At POC": 0, "Below POC": -1}
            features['dfp_price_position'] = position_map.get(deltaflow_signals['price_position'], 0)

        # ========== ML REGIME FEATURES ==========
        if ml_regime_result:
            features['trend_strength'] = ml_regime_result.trend_strength
            features['regime_confidence'] = ml_regime_result.confidence

            # Encode regime
            regime_map = {
                "Trending Up": 2,
                "Trending Down": -2,
                "Range Bound": 0,
                "Volatile Breakout": 1,
                "Consolidation": -1
            }
            features['market_regime'] = regime_map.get(ml_regime_result.regime, 0)

            # Encode volatility state
            vol_map = {"Low": 1, "Normal": 2, "High": 3, "Extreme": 4}
            features['volatility_state'] = vol_map.get(ml_regime_result.volatility_state, 2)

        # ========== OPTION CHAIN FEATURES ==========
        if option_chain:
            ce_data = option_chain.get('CE', {})
            pe_data = option_chain.get('PE', {})

            ce_oi = ce_data.get('openInterest', [])
            pe_oi = pe_data.get('openInterest', [])

            if ce_oi and pe_oi:
                total_ce_oi = sum(ce_oi[:10]) if len(ce_oi) >= 10 else sum(ce_oi)
                total_pe_oi = sum(pe_oi[:10]) if len(pe_oi) >= 10 else sum(pe_oi)

                features['total_ce_oi'] = total_ce_oi
                features['total_pe_oi'] = total_pe_oi
                features['pcr'] = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0

        # ========== SENTIMENT FEATURES ==========
        features['overall_sentiment'] = sentiment_score

        # ========== OPTION SCREENER FEATURES ==========
        if option_screener_data:
            features['momentum_burst'] = option_screener_data.get('momentum_burst', 0)
            features['orderbook_pressure'] = option_screener_data.get('orderbook_pressure', 0)
            features['gamma_cluster_concentration'] = option_screener_data.get('gamma_cluster', 0)
            features['oi_acceleration'] = option_screener_data.get('oi_acceleration', 0)
            features['expiry_spike_detected'] = 1 if option_screener_data.get('expiry_spike', False) else 0
            features['net_vega_exposure'] = option_screener_data.get('net_vega_exposure', 0)
            features['skew_ratio'] = option_screener_data.get('skew_ratio', 0)
            features['atm_vol_premium'] = option_screener_data.get('atm_vol_premium', 0)
        else:
            features['momentum_burst'] = 0
            features['orderbook_pressure'] = 0
            features['gamma_cluster_concentration'] = 0
            features['oi_acceleration'] = 0
            features['expiry_spike_detected'] = 0
            features['net_vega_exposure'] = 0
            features['skew_ratio'] = 0
            features['atm_vol_premium'] = 0

        # ========== TAB 1: OVERALL MARKET SENTIMENT FEATURES ==========
        if overall_sentiment_data and overall_sentiment_data.get('data_available'):
            # 1. Overall Market Direction (BULLISH/BEARISH/NEUTRAL)
            direction_map = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}
            overall_direction = overall_sentiment_data.get('overall_sentiment', 'NEUTRAL')
            features['overall_market_direction'] = direction_map.get(overall_direction, 0)

            # 2. Confluence Score (0-100%)
            features['confluence_score'] = overall_sentiment_data.get('confidence', 0)

            # 3. Number of Bullish Indicators
            features['num_bullish_indicators'] = overall_sentiment_data.get('bullish_sources', 0)

            # 4. Number of Bearish Indicators
            features['num_bearish_indicators'] = overall_sentiment_data.get('bearish_sources', 0)

            # 5. Number of Neutral Indicators
            features['num_neutral_indicators'] = overall_sentiment_data.get('neutral_sources', 0)
        else:
            features['overall_market_direction'] = 0
            features['confluence_score'] = 0
            features['num_bullish_indicators'] = 0
            features['num_bearish_indicators'] = 0
            features['num_neutral_indicators'] = 0

        # ========== TAB 9: ENHANCED MARKET DATA FEATURES ==========
        if enhanced_market_data:
            # --- India VIX Features (4 features) ---
            india_vix = enhanced_market_data.get('india_vix', {})
            if india_vix.get('success'):
                features['vix_level'] = india_vix.get('value', 0)

                # VIX percentile (approximation based on value)
                vix_val = india_vix.get('value', 15)
                if vix_val > 25:
                    features['vix_percentile'] = 95
                elif vix_val > 20:
                    features['vix_percentile'] = 80
                elif vix_val > 15:
                    features['vix_percentile'] = 50
                elif vix_val > 12:
                    features['vix_percentile'] = 25
                else:
                    features['vix_percentile'] = 10

                # VIX interpretation (encoded)
                vix_interp_map = {
                    "HIGH FEAR": 4,
                    "ELEVATED FEAR": 3,
                    "MODERATE": 2,
                    "LOW VOLATILITY": 1,
                    "COMPLACENCY": 0
                }
                features['vix_interpretation'] = vix_interp_map.get(india_vix.get('sentiment', 'MODERATE'), 2)

                # VIX trend (from score: positive score = bullish = VIX down)
                vix_score = india_vix.get('score', 0)
                if vix_score > 20:
                    features['vix_trend'] = -1  # VIX down (STABLE/DOWN)
                elif vix_score < -20:
                    features['vix_trend'] = 1   # VIX up (UP)
                else:
                    features['vix_trend'] = 0   # VIX stable
            else:
                features['vix_level'] = 15
                features['vix_percentile'] = 50
                features['vix_interpretation'] = 2
                features['vix_trend'] = 0

            # --- Sector Rotation Features (4 features) ---
            sector_rotation = enhanced_market_data.get('sector_rotation', {})
            if sector_rotation.get('success'):
                # Sector rotation bias (encoded)
                rotation_bias_map = {
                    "STRONG BULLISH": 2,
                    "BULLISH": 1,
                    "NEUTRAL": 0,
                    "BEARISH": -1,
                    "STRONG BEARISH": -2
                }
                rot_bias = sector_rotation.get('rotation_bias', 'NEUTRAL')
                features['sector_rotation_bias'] = rotation_bias_map.get(rot_bias, 0)

                features['num_sectors_bullish'] = sector_rotation.get('bullish_sectors_count', 0)
                features['num_sectors_bearish'] = sector_rotation.get('bearish_sectors_count', 0)
                features['num_sectors_neutral'] = sector_rotation.get('neutral_sectors_count', 0)
            else:
                features['sector_rotation_bias'] = 0
                features['num_sectors_bullish'] = 0
                features['num_sectors_bearish'] = 0
                features['num_sectors_neutral'] = 0

            # --- Global Markets Features (3 features) ---
            global_markets = enhanced_market_data.get('global_markets', [])
            if global_markets:
                num_up = sum(1 for m in global_markets if m.get('change_pct', 0) > 0)
                num_down = sum(1 for m in global_markets if m.get('change_pct', 0) < 0)

                features['num_global_markets_up'] = num_up
                features['num_global_markets_down'] = num_down

                # Global sentiment (based on majority)
                if num_up > num_down:
                    features['global_markets_sentiment'] = 1  # BULLISH
                elif num_down > num_up:
                    features['global_markets_sentiment'] = -1  # BEARISH
                else:
                    features['global_markets_sentiment'] = 0  # NEUTRAL
            else:
                features['num_global_markets_up'] = 0
                features['num_global_markets_down'] = 0
                features['global_markets_sentiment'] = 0

            # --- Intermarket Features (2 features) ---
            intermarket_data = enhanced_market_data.get('intermarket', [])
            if intermarket_data:
                # Calculate intermarket sentiment (RISK_ON/RISK_OFF)
                risk_on_count = sum(1 for asset in intermarket_data if 'RISK ON' in asset.get('bias', '') or 'BULLISH' in asset.get('bias', ''))
                risk_off_count = sum(1 for asset in intermarket_data if 'RISK OFF' in asset.get('bias', '') or 'BEARISH' in asset.get('bias', ''))

                if risk_on_count > risk_off_count:
                    features['intermarket_sentiment'] = 1  # RISK_ON
                elif risk_off_count > risk_on_count:
                    features['intermarket_sentiment'] = -1  # RISK_OFF
                else:
                    features['intermarket_sentiment'] = 0  # NEUTRAL

                # USD Index trend (extract from intermarket data)
                usd_asset = next((a for a in intermarket_data if 'DOLLAR' in a.get('asset', '')), None)
                if usd_asset:
                    usd_change = usd_asset.get('change_pct', 0)
                    if usd_change > 0.3:
                        features['usd_index_trend'] = 1  # UP
                    elif usd_change < -0.3:
                        features['usd_index_trend'] = -1  # DOWN
                    else:
                        features['usd_index_trend'] = 0  # STABLE
                else:
                    features['usd_index_trend'] = 0
            else:
                features['intermarket_sentiment'] = 0
                features['usd_index_trend'] = 0

            # --- Gamma Squeeze Feature (1 feature) ---
            gamma_squeeze = enhanced_market_data.get('gamma_squeeze', {})
            if gamma_squeeze.get('success'):
                # Convert squeeze risk to probability score
                squeeze_risk = gamma_squeeze.get('squeeze_risk', 'LOW')
                if 'HIGH' in squeeze_risk:
                    features['gamma_squeeze_probability'] = 80
                elif 'MODERATE' in squeeze_risk:
                    features['gamma_squeeze_probability'] = 50
                else:
                    features['gamma_squeeze_probability'] = 10
            else:
                features['gamma_squeeze_probability'] = 0

            # --- Intraday Seasonality Feature (1 feature) ---
            intraday_seasonality = enhanced_market_data.get('intraday_seasonality', {})
            if intraday_seasonality.get('success'):
                # Encode session bias
                session_map = {
                    "HIGH VOLATILITY": 0,
                    "TREND FORMATION": 1,
                    "TRENDING": 2,
                    "CONSOLIDATION": -1,
                    "MOMENTUM": 1,
                    "CLOSING": 0,
                    "NEUTRAL": 0
                }
                session_bias = intraday_seasonality.get('session_bias', 'NEUTRAL')
                features['intraday_session_bias'] = session_map.get(session_bias, 0)
            else:
                features['intraday_session_bias'] = 0
        else:
            # Default values if enhanced_market_data not available
            features['vix_level'] = 15
            features['vix_percentile'] = 50
            features['vix_interpretation'] = 2
            features['vix_trend'] = 0
            features['sector_rotation_bias'] = 0
            features['num_sectors_bullish'] = 0
            features['num_sectors_bearish'] = 0
            features['num_sectors_neutral'] = 0
            features['global_markets_sentiment'] = 0
            features['num_global_markets_up'] = 0
            features['num_global_markets_down'] = 0
            features['intermarket_sentiment'] = 0
            features['usd_index_trend'] = 0
            features['gamma_squeeze_probability'] = 0
            features['intraday_session_bias'] = 0

        # ========== TAB 8: NIFTY OPTION SCREENER FEATURES ==========
        if nifty_screener_data:
            # --- ATM Bias Features (13 features from 11 bias metrics) ---
            atm_bias = nifty_screener_data.get('atm_bias', {})
            if atm_bias:
                bias_scores = atm_bias.get('bias_scores', {})

                # Extract all 13 ATM bias metrics
                features['atm_oi_bias'] = bias_scores.get('OI_Bias', 0)
                features['atm_chgoi_bias'] = bias_scores.get('ChgOI_Bias', 0)
                features['atm_volume_bias'] = bias_scores.get('Volume_Bias', 0)
                features['atm_delta_bias'] = bias_scores.get('Delta_Bias', 0)
                features['atm_gamma_bias'] = bias_scores.get('Gamma_Bias', 0)
                features['atm_premium_bias'] = bias_scores.get('Premium_Bias', 0)
                features['atm_askqty_bias'] = bias_scores.get('AskQty_Bias', 0) if 'AskQty_Bias' in bias_scores else 0
                features['atm_bidqty_bias'] = bias_scores.get('BidQty_Bias', 0) if 'BidQty_Bias' in bias_scores else 0
                features['atm_iv_bias'] = bias_scores.get('IV_Bias', 0)
                features['atm_dvp_bias'] = bias_scores.get('DVP_Bias', 0) if 'DVP_Bias' in bias_scores else 0
                features['atm_delta_exposure_bias'] = bias_scores.get('Delta_Exposure_Bias', 0)
                features['atm_gamma_exposure_bias'] = bias_scores.get('Gamma_Exposure_Bias', 0)
                features['atm_iv_skew_bias'] = bias_scores.get('IV_Skew_Bias', 0)
            else:
                features['atm_oi_bias'] = 0
                features['atm_chgoi_bias'] = 0
                features['atm_volume_bias'] = 0
                features['atm_delta_bias'] = 0
                features['atm_gamma_bias'] = 0
                features['atm_premium_bias'] = 0
                features['atm_askqty_bias'] = 0
                features['atm_bidqty_bias'] = 0
                features['atm_iv_bias'] = 0
                features['atm_dvp_bias'] = 0
                features['atm_delta_exposure_bias'] = 0
                features['atm_gamma_exposure_bias'] = 0
                features['atm_iv_skew_bias'] = 0

            # --- Market Depth Features (5 features from orderbook) ---
            # Note: Market depth data from option_screener_data parameter (moment_metrics)
            if option_screener_data and 'orderbook_pressure' in option_screener_data:
                features['market_depth_pressure'] = option_screener_data.get('orderbook_pressure', 0)
            else:
                features['market_depth_pressure'] = 0

            # Additional market depth features (if available from session state orderbook)
            features['market_depth_bid_qty'] = 0  # Will be populated from orderbook if available
            features['market_depth_ask_qty'] = 0
            features['market_depth_order_imbalance'] = 0
            features['market_depth_spread'] = 0

            # --- Expiry Context Features (4 features) ---
            expiry_spike_data = nifty_screener_data.get('expiry_spike_data', {})
            if expiry_spike_data:
                features['days_to_expiry'] = expiry_spike_data.get('days_to_expiry', 0)
                features['is_expiry_week'] = 1 if expiry_spike_data.get('is_expiry_week', False) else 0
                features['is_monthly_expiry'] = 1 if expiry_spike_data.get('is_monthly_expiry', False) else 0

                # Time decay factor (higher closer to expiry)
                days_left = expiry_spike_data.get('days_to_expiry', 7)
                if days_left <= 0:
                    features['time_decay_factor'] = 1.0
                elif days_left <= 2:
                    features['time_decay_factor'] = 0.9
                elif days_left <= 5:
                    features['time_decay_factor'] = 0.7
                else:
                    features['time_decay_factor'] = 0.4
            else:
                features['days_to_expiry'] = 7
                features['is_expiry_week'] = 0
                features['is_monthly_expiry'] = 0
                features['time_decay_factor'] = 0.5

            # --- OI/PCR Advanced Features (3 features) ---
            oi_pcr_metrics = nifty_screener_data.get('oi_pcr_metrics', {})
            if oi_pcr_metrics:
                features['pcr_value'] = oi_pcr_metrics.get('pcr_value', 1.0)

                # OI buildup pattern (encoded)
                buildup_pattern = oi_pcr_metrics.get('buildup_pattern', 'NEUTRAL')
                buildup_map = {
                    "LONG BUILDUP": -1,
                    "SHORT BUILDUP": 1,
                    "LONG UNWINDING": 1,
                    "SHORT UNWINDING": -1,
                    "NEUTRAL": 0,
                    "CONSOLIDATION": 0
                }
                features['oi_buildup_pattern'] = buildup_map.get(buildup_pattern, 0)
            else:
                features['pcr_value'] = 1.0
                features['oi_buildup_pattern'] = 0

            # Max Pain Distance (from atm_bias or seller_max_pain)
            seller_max_pain = nifty_screener_data.get('seller_max_pain', 0)
            if seller_max_pain and len(df) > 0:
                current_price = df['close'].iloc[-1]
                max_pain_distance = ((seller_max_pain - current_price) / current_price) * 100
                features['max_pain_distance'] = max_pain_distance
            else:
                features['max_pain_distance'] = 0
        else:
            # Default values for all Tab 8 features
            features['atm_oi_bias'] = 0
            features['atm_chgoi_bias'] = 0
            features['atm_volume_bias'] = 0
            features['atm_delta_bias'] = 0
            features['atm_gamma_bias'] = 0
            features['atm_premium_bias'] = 0
            features['atm_askqty_bias'] = 0
            features['atm_bidqty_bias'] = 0
            features['atm_iv_bias'] = 0
            features['atm_dvp_bias'] = 0
            features['atm_delta_exposure_bias'] = 0
            features['atm_gamma_exposure_bias'] = 0
            features['atm_iv_skew_bias'] = 0
            features['market_depth_pressure'] = 0
            features['market_depth_bid_qty'] = 0
            features['market_depth_ask_qty'] = 0
            features['market_depth_order_imbalance'] = 0
            features['market_depth_spread'] = 0
            features['days_to_expiry'] = 7
            features['is_expiry_week'] = 0
            features['is_monthly_expiry'] = 0
            features['time_decay_factor'] = 0.5
            features['pcr_value'] = 1.0
            features['oi_buildup_pattern'] = 0
            features['max_pain_distance'] = 0

        # ========== TAB 7: ADVANCED CHART ANALYSIS FEATURES ==========
        # These features will be populated from liquidity_result and chart analysis

        # --- HTF Support/Resistance Features (4 features) ---
        if liquidity_result:
            # Distance to nearest HTF resistance
            resistance_zones = liquidity_result.resistance_zones if hasattr(liquidity_result, 'resistance_zones') else []
            support_zones = liquidity_result.support_zones if hasattr(liquidity_result, 'support_zones') else []

            if len(df) > 0:
                current_price = df['close'].iloc[-1]

                if resistance_zones:
                    nearest_resistance = min(resistance_zones, key=lambda x: abs(x - current_price))
                    features['htf_resistance_distance'] = ((nearest_resistance - current_price) / current_price) * 100
                    features['htf_resistance_strength'] = len([r for r in resistance_zones if abs(r - nearest_resistance) < 50])
                else:
                    features['htf_resistance_distance'] = 5.0
                    features['htf_resistance_strength'] = 0

                if support_zones:
                    nearest_support = min(support_zones, key=lambda x: abs(x - current_price))
                    features['htf_support_distance'] = ((current_price - nearest_support) / current_price) * 100
                    features['htf_support_strength'] = len([s for s in support_zones if abs(s - nearest_support) < 50])
                else:
                    features['htf_support_distance'] = 5.0
                    features['htf_support_strength'] = 0
            else:
                features['htf_resistance_distance'] = 5.0
                features['htf_resistance_strength'] = 0
                features['htf_support_distance'] = 5.0
                features['htf_support_strength'] = 0
        else:
            features['htf_resistance_distance'] = 5.0
            features['htf_resistance_strength'] = 0
            features['htf_support_distance'] = 5.0
            features['htf_support_strength'] = 0

        # --- Volume Footprint Features (4 features) ---
        # Volume footprint shows buying vs selling pressure at each price level
        if len(df) >= 20:
            # Calculate volume-weighted price momentum
            df_recent = df.tail(20)
            volume_price_trend = df_recent[['close', 'volume']].corr().iloc[0, 1] if 'volume' in df.columns else 0
            features['volume_footprint_trend'] = volume_price_trend

            # Calculate buying/selling pressure from volume
            up_bars = df_recent[df_recent['close'] > df_recent['open']]
            down_bars = df_recent[df_recent['close'] < df_recent['open']]

            buy_volume = up_bars['volume'].sum() if 'volume' in df.columns and len(up_bars) > 0 else 0
            sell_volume = down_bars['volume'].sum() if 'volume' in df.columns and len(down_bars) > 0 else 0
            total_vol = buy_volume + sell_volume

            if total_vol > 0:
                features['volume_buy_sell_ratio'] = (buy_volume - sell_volume) / total_vol
                features['volume_imbalance'] = abs(buy_volume - sell_volume) / total_vol
            else:
                features['volume_buy_sell_ratio'] = 0
                features['volume_imbalance'] = 0

            # Volume concentration at current level
            recent_vol_avg = df_recent['volume'].mean() if 'volume' in df.columns else 0
            current_vol = df['volume'].iloc[-1] if 'volume' in df.columns else 0
            features['volume_concentration'] = (current_vol / recent_vol_avg) if recent_vol_avg > 0 else 1.0
        else:
            features['volume_footprint_trend'] = 0
            features['volume_buy_sell_ratio'] = 0
            features['volume_imbalance'] = 0
            features['volume_concentration'] = 1.0

        # --- Liquidity Sentiment Features (3 features) ---
        if liquidity_result:
            # Gravity strength towards liquidity target
            features['liquidity_gravity_strength'] = liquidity_result.gravity_strength if hasattr(liquidity_result, 'gravity_strength') else 0

            # Number of nearby liquidity zones
            hvn_count = len(liquidity_result.hvn_zones) if hasattr(liquidity_result, 'hvn_zones') else 0
            features['liquidity_hvn_count'] = hvn_count

            # Liquidity sentiment (encoded: positive = bullish liquidity pull)
            if len(df) > 0 and hasattr(liquidity_result, 'primary_target'):
                current_price = df['close'].iloc[-1]
                target_price = liquidity_result.primary_target

                if target_price > current_price:
                    features['liquidity_sentiment'] = 1  # Bullish liquidity pull
                elif target_price < current_price:
                    features['liquidity_sentiment'] = -1  # Bearish liquidity pull
                else:
                    features['liquidity_sentiment'] = 0
            else:
                features['liquidity_sentiment'] = 0
        else:
            features['liquidity_gravity_strength'] = 0
            features['liquidity_hvn_count'] = 0
            features['liquidity_sentiment'] = 0

        # --- Fibonacci Level Features (4 features) ---
        # Fibonacci retracement levels (calculated from recent swing high/low)
        if len(df) >= 50:
            df_fib = df.tail(50)
            swing_high = df_fib['high'].max()
            swing_low = df_fib['low'].min()
            current_price = df['close'].iloc[-1]

            # Calculate Fibonacci retracement levels
            fib_range = swing_high - swing_low
            fib_618 = swing_low + (fib_range * 0.618)
            fib_50 = swing_low + (fib_range * 0.5)
            fib_382 = swing_low + (fib_range * 0.382)

            # Distance to key Fib levels
            fib_levels = [fib_618, fib_50, fib_382]
            nearest_fib = min(fib_levels, key=lambda x: abs(x - current_price))
            features['fib_nearest_level_distance'] = ((nearest_fib - current_price) / current_price) * 100

            # Price position relative to Fib range (0 = at swing low, 1 = at swing high)
            if fib_range > 0:
                features['fib_position_in_range'] = (current_price - swing_low) / fib_range
            else:
                features['fib_position_in_range'] = 0.5

            # Golden pocket zone (0.618-0.65 retracement) - binary feature
            golden_pocket_low = swing_low + (fib_range * 0.618)
            golden_pocket_high = swing_low + (fib_range * 0.65)
            features['fib_in_golden_pocket'] = 1 if golden_pocket_low <= current_price <= golden_pocket_high else 0

            # Fib extension level (above swing high for uptrends)
            fib_ext_1618 = swing_high + (fib_range * 0.618)
            features['fib_extension_distance'] = ((fib_ext_1618 - current_price) / current_price) * 100
        else:
            features['fib_nearest_level_distance'] = 0
            features['fib_position_in_range'] = 0.5
            features['fib_in_golden_pocket'] = 0
            features['fib_extension_distance'] = 0

        # Convert to DataFrame
        feature_df = pd.DataFrame([features])

        # Fill missing values with 0
        feature_df = feature_df.fillna(0)

        return feature_df

    def train_model_with_simulated_data(self, n_samples: int = 1000):
        """
        Train XGBoost model with simulated training data

        In production, replace this with actual historical trade data
        """
        logger.info("Generating simulated training data...")

        # Generate random features (simulate historical data)
        np.random.seed(42)

        n_features = 50
        X = np.random.randn(n_samples, n_features)

        # Generate labels based on feature combinations (simulate profitable patterns)
        # BUY signals: positive momentum + institutional buying + high liquidity gravity
        buy_score = (
            X[:, 0] +  # Price momentum
            X[:, 10] +  # Institutional confidence
            X[:, 20] -  # Inverse trap probability
            X[:, 5]    # Volatility factor
        )

        # SELL signals: negative momentum + retail activity + OI traps
        sell_score = -(
            X[:, 0] +
            X[:, 11] +  # Retail activity
            X[:, 21]    # Trap detection
        )

        # Create labels
        y = np.zeros(n_samples)
        y[buy_score > 1.5] = 0  # BUY
        y[sell_score > 1.5] = 1  # SELL
        y[(buy_score <= 1.5) & (sell_score <= 1.5)] = 2  # HOLD

        # Train XGBoost
        logger.info("Training XGBoost model...")

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)

        # Save feature names
        self.feature_names = [f'feature_{i}' for i in range(n_features)]
        self.is_trained = True

        logger.info("✅ Model training complete!")

        return self.model

    def predict(
        self,
        features_df: pd.DataFrame
    ) -> MLPredictionResult:
        """
        Make prediction using XGBoost model

        Args:
            features_df: DataFrame with extracted features

        Returns:
            MLPredictionResult with prediction and probabilities
        """
        if not self.is_trained:
            # Train with simulated data if not trained
            self.train_model_with_simulated_data()

        # Ensure features match training
        missing_features = set(self.feature_names) - set(features_df.columns)
        for feat in missing_features:
            features_df[feat] = 0

        # Reorder to match training
        features_df = features_df[self.feature_names]

        # Make prediction
        y_pred = self.model.predict(features_df)[0]
        y_proba = self.model.predict_proba(features_df)[0]

        # Map prediction to label
        label_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
        prediction = label_map[y_pred]
        probability = y_proba[y_pred]

        # Get all probabilities
        all_probs = {
            "BUY": y_proba[0],
            "SELL": y_proba[1],
            "HOLD": y_proba[2]
        }

        # Calculate confidence (0-100)
        confidence = probability * 100

        # Calculate expected return (based on prediction probabilities)
        expected_return = (y_proba[0] * 2.0) + (y_proba[1] * -2.0) + (y_proba[2] * 0.0)

        # Calculate risk score
        risk_score = (1 - probability) * 100

        # Feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            importance_values = self.model.feature_importances_
            # Get top 10 important features
            top_indices = np.argsort(importance_values)[-10:]
            for idx in top_indices:
                feature_importance[self.feature_names[idx]] = float(importance_values[idx])

        # Generate recommendation
        recommendation = self._generate_ml_recommendation(
            prediction, confidence, expected_return, risk_score
        )

        return MLPredictionResult(
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            feature_importance=feature_importance,
            all_probabilities=all_probs,
            expected_return=expected_return,
            risk_score=risk_score,
            recommendation=recommendation,
            model_version=self.model_version
        )

    def _generate_ml_recommendation(
        self,
        prediction: str,
        confidence: float,
        expected_return: float,
        risk_score: float
    ) -> str:
        """Generate trading recommendation from ML prediction"""
        if prediction == "BUY":
            if confidence > 80:
                return f"🚀 STRONG BUY - High confidence ({confidence:.1f}%), Expected: +{expected_return:.2f}%"
            elif confidence > 65:
                return f"✅ BUY - Good confidence ({confidence:.1f}%)"
            else:
                return f"⚠️ WEAK BUY - Low confidence ({confidence:.1f}%), be cautious"

        elif prediction == "SELL":
            if confidence > 80:
                return f"🔻 STRONG SELL - High confidence ({confidence:.1f}%), Expected: {expected_return:.2f}%"
            elif confidence > 65:
                return f"⚠️ SELL - Good confidence ({confidence:.1f}%)"
            else:
                return f"⚠️ WEAK SELL - Low confidence ({confidence:.1f}%), be cautious"

        else:  # HOLD
            if risk_score > 60:
                return f"⏸️ HOLD - High risk ({risk_score:.0f}%), wait for better setup"
            else:
                return f"⏸️ HOLD - Neutral conditions, no clear edge"

    def analyze_complete_market(
        self,
        df: pd.DataFrame,
        bias_results: Optional[Dict] = None,
        option_chain: Optional[Dict] = None,
        volatility_result: Optional[any] = None,
        oi_trap_result: Optional[any] = None,
        cvd_result: Optional[any] = None,
        participant_result: Optional[any] = None,
        liquidity_result: Optional[any] = None,
        ml_regime_result: Optional[any] = None,
        sentiment_score: float = 0.0,
        option_screener_data: Optional[Dict] = None
    ) -> MLPredictionResult:
        """
        Complete XGBoost ML analysis of ALL market data

        This is the main entry point that:
        1. Extracts features from ALL modules
        2. Runs XGBoost prediction
        3. Returns ML-based trading signal
        """
        # Extract all features
        features_df = self.extract_features_from_all_tabs(
            df=df,
            bias_results=bias_results,
            option_chain=option_chain,
            volatility_result=volatility_result,
            oi_trap_result=oi_trap_result,
            cvd_result=cvd_result,
            participant_result=participant_result,
            liquidity_result=liquidity_result,
            ml_regime_result=ml_regime_result,
            sentiment_score=sentiment_score,
            option_screener_data=option_screener_data
        )

        # Make prediction
        result = self.predict(features_df)

        return result


def format_ml_result(result: MLPredictionResult) -> str:
    """Format ML prediction result as readable report"""
    report = f"""
╔══════════════════════════════════════════════════════════╗
║          XGBOOST ML PREDICTION                           ║
╚══════════════════════════════════════════════════════════╝

🎯 PREDICTION: {result.prediction}
💪 CONFIDENCE: {result.confidence:.1f}%
📊 PROBABILITY: {result.probability:.3f}

─────────────────────────────────────────────────────────
PREDICTION PROBABILITIES:
  • BUY:  {result.all_probabilities['BUY']*100:.1f}%
  • SELL: {result.all_probabilities['SELL']*100:.1f}%
  • HOLD: {result.all_probabilities['HOLD']*100:.1f}%

EXPECTED METRICS:
  • Expected Return: {result.expected_return:+.2f}%
  • Risk Score: {result.risk_score:.1f}/100

─────────────────────────────────────────────────────────
💡 RECOMMENDATION:
{result.recommendation}

─────────────────────────────────────────────────────────
TOP FEATURE IMPORTANCE:
"""

    # Sort features by importance
    sorted_features = sorted(
        result.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for feat, importance in sorted_features[:10]:
        report += f"  • {feat}: {importance:.4f}\n"

    report += f"\n📦 Model Version: {result.model_version}\n"

    return report
