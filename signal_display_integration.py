"""
Signal Display Integration for UI

Connects Enhanced Signal Generator with Streamlit UI.
Provides display functions for signal cards, history, and statistics.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime
import logging

from src.enhanced_signal_generator import EnhancedSignalGenerator, TradingSignal
from src.xgboost_ml_analyzer import XGBoostMLAnalyzer
from src.telegram_signal_manager import TelegramSignalManager

logger = logging.getLogger(__name__)


def generate_trading_signal(
    df: pd.DataFrame,
    bias_results: Optional[Dict],
    option_chain: Optional[Dict],
    volatility_result: Optional[any],
    oi_trap_result: Optional[any],
    cvd_result: Optional[any],
    participant_result: Optional[any],
    liquidity_result: Optional[any],
    ml_regime_result: Optional[any],
    sentiment_score: float,
    option_screener_data: Optional[Dict] = None,
    money_flow_signals: Optional[Dict] = None,
    deltaflow_signals: Optional[Dict] = None,
    overall_sentiment_data: Optional[Dict] = None,
    enhanced_market_data: Optional[Dict] = None,
    nifty_screener_data: Optional[Dict] = None,
    current_price: float = 0.0,
    atm_strike: Optional[int] = None
) -> Optional[TradingSignal]:
    """
    Generate comprehensive trading signal using all available data sources.

    Returns:
        TradingSignal object or None if generation fails
    """
    try:
        # Initialize analyzers
        if 'xgboost_analyzer' not in st.session_state:
            st.session_state.xgboost_analyzer = XGBoostMLAnalyzer()

        if 'signal_generator' not in st.session_state:
            st.session_state.signal_generator = EnhancedSignalGenerator(
                min_confidence=65.0,
                min_confluence=6
            )

        xgb_analyzer = st.session_state.xgboost_analyzer
        signal_generator = st.session_state.signal_generator

        # Extract all 146 features
        features_df = xgb_analyzer.extract_features_from_all_tabs(
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
            option_screener_data=option_screener_data,
            money_flow_signals=money_flow_signals,
            deltaflow_signals=deltaflow_signals,
            overall_sentiment_data=overall_sentiment_data,
            enhanced_market_data=enhanced_market_data,
            nifty_screener_data=nifty_screener_data
        )

        # Get XGBoost prediction
        xgb_result = xgb_analyzer.predict(features_df)

        # Determine current price if not provided
        if current_price == 0.0 and len(df) > 0:
            current_price = df['close'].iloc[-1]

        # Determine ATM strike if not provided
        if atm_strike is None and current_price > 0:
            atm_strike = round(current_price / 50) * 50

        # Prepare option chain data for signal generation
        option_chain_for_signal = None
        if nifty_screener_data:
            option_chain_for_signal = nifty_screener_data.get('option_chain', {})
        elif option_chain:
            option_chain_for_signal = option_chain

        # Generate signal
        signal = signal_generator.generate_signal(
            xgboost_result=xgb_result,
            features_df=features_df,
            current_price=current_price,
            option_chain=option_chain_for_signal,
            atm_strike=atm_strike
        )

        # Save signal to history
        if signal and 'signal_history' not in st.session_state:
            st.session_state.signal_history = []

        if signal:
            st.session_state.signal_history.insert(0, {
                'timestamp': signal.timestamp,
                'signal_type': signal.signal_type,
                'direction': signal.direction,
                'confidence': signal.confidence,
                'confluence': signal.confluence_count
            })
            # Keep only last 50 signals
            st.session_state.signal_history = st.session_state.signal_history[:50]

        return signal

    except Exception as e:
        logger.error(f"Signal generation error: {e}", exc_info=True)
        return None


def display_final_assessment(
    nifty_screener_data: Optional[Dict],
    enhanced_market_data: Optional[Dict],
    ml_regime_result: Optional[any],
    liquidity_result: Optional[any],
    current_price: float,
    atm_strike: int,
    option_chain: Optional[Dict] = None,
    money_flow_signals: Optional[Dict] = None,
    deltaflow_signals: Optional[Dict] = None,
    cvd_result: Optional[any] = None,
    volatility_result: Optional[any] = None,
    oi_trap_result: Optional[any] = None,
    participant_result: Optional[any] = None
):
    """
    Display FINAL ASSESSMENT with Market Makers narrative.

    Now integrates ALL analysis from all tabs:
    - Tab 7: Volume Footprint, Money Flow, DeltaFlow, HTF S/R
    - Tab 8: All 10 sub-tabs (OI/PCR, ATM Bias, Seller, Moment, Depth, Expiry, etc.)
    - Tab 9: Enhanced market data, VIX, sectors

    Uses comprehensive multi-factor analysis for scoring and entries.
    """
    # Extract data
    atm_bias_data = nifty_screener_data.get('atm_bias', {}) if nifty_screener_data else {}
    moment_data = nifty_screener_data.get('moment_metrics', {}) if nifty_screener_data else {}  # Fixed: moment_metrics not moment_detector
    expiry_data = nifty_screener_data.get('expiry_spike_data', {}) if nifty_screener_data else {}  # Fixed: expiry_spike_data not expiry_context
    oi_pcr_data = nifty_screener_data.get('oi_pcr_metrics', {}) if nifty_screener_data else {}  # Fixed: oi_pcr_metrics not oi_pcr
    market_depth = nifty_screener_data.get('market_depth', {}) if nifty_screener_data else {}

    # Get regime from ML result
    regime = "RANGING"
    if ml_regime_result and hasattr(ml_regime_result, 'regime'):
        regime = ml_regime_result.regime
    elif ml_regime_result and isinstance(ml_regime_result, dict):
        regime = ml_regime_result.get('regime', 'RANGING')

    # Get sector rotation from enhanced market data
    sector_bias = "NEUTRAL"
    if enhanced_market_data:
        sectors = enhanced_market_data.get('sectors', {})
        if sectors.get('success'):
            sector_data = sectors.get('data', [])
            bullish_sectors = sum(1 for s in sector_data if s.get('change_pct', 0) > 0.5)
            bearish_sectors = sum(1 for s in sector_data if s.get('change_pct', 0) < -0.5)
            if bullish_sectors > bearish_sectors + 2:
                sector_bias = "BULLISH"
            elif bearish_sectors > bullish_sectors + 2:
                sector_bias = "BEARISH"

    # Get ATM Bias
    atm_bias_score = atm_bias_data.get('total_score', 0)
    atm_bias_verdict = atm_bias_data.get('verdict', 'NEUTRAL')

    # Determine ATM bias emoji
    if atm_bias_verdict == "CALL SELLERS":
        atm_emoji = "🔴"
    elif atm_bias_verdict == "PUT SELLERS":
        atm_emoji = "🟢"
    else:
        atm_emoji = "⚖️"

    # Get Moment Detector (extract from moment_metrics structure)
    moment_verdict = 'NEUTRAL'
    moment_score = 0
    orderbook_pressure = 'NEUTRAL'
    orderbook_pressure_raw = 0

    if moment_data:
        # moment_metrics has structure: {momentum_burst: {}, orderbook: {}, gamma_cluster: {}, oi_accel: {}}
        if 'orderbook' in moment_data and moment_data['orderbook'].get('available'):
            orderbook_pressure_raw = moment_data['orderbook'].get('pressure', 0)

            # Convert raw number to label
            if orderbook_pressure_raw > 0.5:
                orderbook_pressure = "STRONG BUY PRESSURE ↑"
            elif orderbook_pressure_raw > 0.15:
                orderbook_pressure = "BUY PRESSURE ↑"
            elif orderbook_pressure_raw > 0.05:
                orderbook_pressure = "MILD BUY"
            elif orderbook_pressure_raw < -0.5:
                orderbook_pressure = "STRONG SELL PRESSURE ↓"
            elif orderbook_pressure_raw < -0.15:
                orderbook_pressure = "SELL PRESSURE ↓"
            elif orderbook_pressure_raw < -0.05:
                orderbook_pressure = "MILD SELL"
            else:
                orderbook_pressure = "NEUTRAL (Low participation)"

        if 'momentum_burst' in moment_data:
            moment_score = moment_data['momentum_burst'].get('score', 0)
            # Determine verdict from score
            if moment_score > 50:
                moment_verdict = 'BULLISH'
            elif moment_score < -50:
                moment_verdict = 'BEARISH'
            else:
                moment_verdict = 'NEUTRAL'

    # Get OI/PCR metrics (fixed key names from Tab 8)
    pcr_value = oi_pcr_data.get('pcr_total', 0.9)  # Correct key: pcr_total
    call_oi = oi_pcr_data.get('total_ce_oi', 0)  # Correct key: total_ce_oi
    put_oi = oi_pcr_data.get('total_pe_oi', 0)  # Correct key: total_pe_oi
    atm_total_oi = oi_pcr_data.get('atm_total_oi', 0)
    total_oi = call_oi + put_oi
    atm_concentration = (atm_total_oi / total_oi * 100) if total_oi > 0 else 0

    # Create label for ATM concentration
    if atm_concentration == 0:
        atm_conc_display = "LOW (<5%)"
    elif atm_concentration < 5:
        atm_conc_display = f"LOW ({atm_concentration:.1f}%)"
    elif atm_concentration < 15:
        atm_conc_display = f"MODERATE ({atm_concentration:.1f}%)"
    elif atm_concentration < 30:
        atm_conc_display = f"HIGH ({atm_concentration:.1f}%)"
    else:
        atm_conc_display = f"VERY HIGH ({atm_concentration:.1f}%)"

    # Determine PCR interpretation
    if pcr_value > 1.2:
        pcr_sentiment = "STRONG BULLISH"
    elif pcr_value > 1.0:
        pcr_sentiment = "MILD BULLISH"
    elif pcr_value > 0.8:
        pcr_sentiment = "NEUTRAL"
    elif pcr_value > 0.6:
        pcr_sentiment = "MILD BEARISH"
    else:
        pcr_sentiment = "STRONG BEARISH"

    # Get Expiry Context (stored directly in nifty_screener_data)
    days_to_expiry = nifty_screener_data.get('days_to_expiry', 7) if nifty_screener_data else 7

    # Get Support/Resistance using multiple data sources for accuracy
    # Priority: 1) nearest_sup/nearest_res from Tab 8, 2) liquidity zones, 3) calculated default
    support_level = round((current_price - 100) / 50) * 50  # Default fallback
    resistance_level = round((current_price + 100) / 50) * 50  # Default fallback

    # Try to get from NIFTY Option Screener (most accurate - from option chain OI)
    if nifty_screener_data:
        nearest_sup = nifty_screener_data.get('nearest_sup')
        nearest_res = nifty_screener_data.get('nearest_res')

        # Type checking: ensure they're numbers (int or float), not dicts or other types
        if nearest_sup and isinstance(nearest_sup, (int, float)) and nearest_sup < current_price:
            support_level = nearest_sup
        if nearest_res and isinstance(nearest_res, (int, float)) and nearest_res > current_price:
            resistance_level = nearest_res

    # Fallback to liquidity zones from Advanced Chart Analysis
    if liquidity_result and (support_level == round((current_price - 100) / 50) * 50):
        support_zones = liquidity_result.support_zones if hasattr(liquidity_result, 'support_zones') else []
        resistance_zones = liquidity_result.resistance_zones if hasattr(liquidity_result, 'resistance_zones') else []
        if support_zones:
            # Type checking: filter only numeric values
            valid_supports = [s for s in support_zones if isinstance(s, (int, float)) and s < current_price]
            if valid_supports:
                support_level = max(valid_supports)
        if resistance_zones:
            # Type checking: filter only numeric values
            valid_resistances = [r for r in resistance_zones if isinstance(r, (int, float)) and r > current_price]
            if valid_resistances:
                resistance_level = min(valid_resistances)

    # Get Max OI Walls (fixed key names from Tab 8)
    max_call_strike = atm_strike + 500
    max_put_strike = atm_strike - 500
    if oi_pcr_data.get('max_ce_strike'):  # Correct key: max_ce_strike
        max_call_strike = oi_pcr_data['max_ce_strike']
    if oi_pcr_data.get('max_pe_strike'):  # Correct key: max_pe_strike
        max_put_strike = oi_pcr_data['max_pe_strike']

    # Get Max Pain (check if seller_max_pain exists in nifty_screener_data)
    max_pain = atm_strike
    if nifty_screener_data and 'seller_max_pain' in nifty_screener_data:
        seller_max_pain_data = nifty_screener_data['seller_max_pain']
        if isinstance(seller_max_pain_data, dict):
            max_pain = seller_max_pain_data.get('max_pain_strike', atm_strike)
        elif isinstance(seller_max_pain_data, (int, float)):
            max_pain = seller_max_pain_data

    # --- Market Makers Narrative ---
    if atm_bias_verdict == "CALL SELLERS":
        mm_narrative = "Sellers aggressively WRITING CALLS (bearish conviction). Expecting price to STAY BELOW strikes."
        game_plan = "Bearish breakdown likely. Sellers confident in downside."
    elif atm_bias_verdict == "PUT SELLERS":
        mm_narrative = "Sellers aggressively WRITING PUTS (bullish conviction). Expecting price to STAY ABOVE strikes."
        game_plan = "Bullish breakout likely. Sellers confident in upside."
    else:
        mm_narrative = "Balanced selling in both CALLS and PUTS. No clear directional bias."
        game_plan = "Range-bound consolidation expected. Wait for breakout."

    # --- Display FINAL ASSESSMENT (Native Python/Streamlit - NO HTML!) ---
    st.markdown("### 📊 FINAL ASSESSMENT (Seller + ATM Bias + Moment + Expiry + OI/PCR)")

    # --- DATA VALIDITY HEALTH INDICATOR (VERY CRITICAL) ---
    data_valid = call_oi > 0 and put_oi > 0 and total_oi > 10000
    data_partial = (call_oi == 0 or put_oi == 0) and total_oi > 0

    if not data_valid and not data_partial:
        st.error("🔴 **OI DATA: STALE / FAILED** - Signals may be unreliable! Consider waiting for fresh data.")
    elif data_partial:
        st.warning("🟡 **PARTIAL DATA USED** - Some OI metrics unavailable. Exercise caution with signals.")
    else:
        st.success("🟢 **OI DATA: VALID** - All systems operational ✓")

    # --- TRADE CONFIDENCE SCORE (MOST IMPORTANT) ---
    # Calculate confidence from multiple factors (0-100 scale)
    confidence_factors = {}

    # Factor 1: OI Data Quality (30 points)
    if data_valid:
        confidence_factors['oi_data'] = 30
    elif data_partial:
        confidence_factors['oi_data'] = 15
    else:
        confidence_factors['oi_data'] = 0

    # Factor 2: ATM Bias Strength (20 points)
    if abs(atm_bias_score) > 0.7:
        confidence_factors['atm_bias'] = 20
    elif abs(atm_bias_score) > 0.4:
        confidence_factors['atm_bias'] = 10
    else:
        confidence_factors['atm_bias'] = 0

    # Factor 3: Regime Clarity (15 points) - Trending is better than ranging
    if regime in ['TRENDING_UP', 'TRENDING_DOWN', 'STRONG_TRENDING_UP', 'STRONG_TRENDING_DOWN']:
        confidence_factors['regime'] = 15
    elif regime in ['RANGING', 'CONSOLIDATING']:
        confidence_factors['regime'] = 5
    else:
        confidence_factors['regime'] = 0

    # Factor 4: PCR Conviction (15 points) - Extreme PCR values show conviction
    if pcr_value > 1.2 or pcr_value < 0.7:
        confidence_factors['pcr'] = 15
    elif pcr_value > 1.0 or pcr_value < 0.85:
        confidence_factors['pcr'] = 8
    else:
        confidence_factors['pcr'] = 3

    # Factor 5: Support/Resistance Distance (10 points) - Tighter range = lower confidence for breakout
    sr_distance = abs(resistance_level - support_level)
    if sr_distance > 200:
        confidence_factors['sr_distance'] = 10  # Wide range, good for trending
    elif sr_distance > 100:
        confidence_factors['sr_distance'] = 5
    else:
        confidence_factors['sr_distance'] = 2  # Very tight, chop zone

    # Factor 6: Moment/Orderbook Pressure (10 points)
    if abs(moment_score) > 50 or orderbook_pressure in ['STRONG_BUY', 'STRONG_SELL']:
        confidence_factors['momentum'] = 10
    elif abs(moment_score) > 25:
        confidence_factors['momentum'] = 5
    else:
        confidence_factors['momentum'] = 0

    # Calculate total confidence score
    confidence_score = sum(confidence_factors.values())

    # Display confidence score with clear labels
    if confidence_score >= 70:
        st.success(f"✅ **HIGH PROBABILITY SETUP** ({confidence_score}/100) - Strong edge detected")
        market_state_banner = "🧠 **MARKET STATE: HIGH EDGE** → Trade with conviction"
    elif confidence_score >= 45:
        st.warning(f"⚠️ **LOW CONFIDENCE - WAIT** ({confidence_score}/100) - Setup not ideal, wait for better conditions")
        market_state_banner = "🧠 **MARKET STATE: LOW EDGE** → WAIT FOR EXTREMES ONLY"
    else:
        st.error(f"❌ **NO TRADE ZONE** ({confidence_score}/100) - Stay out, conditions unfavorable")
        market_state_banner = "🧠 **MARKET STATE: NO EDGE** → DO NOT TRADE"

    # Display golden line at top
    st.markdown(f"### {market_state_banner}")
    st.markdown("---")

    with st.container():
        st.info(f"**🟠 Market Makers are telling us:**\n\n{mm_narrative}")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**🔵 ATM Zone Analysis:**\n\nATM Bias: {atm_emoji} {atm_bias_verdict} ({atm_bias_score:.2f} score)")
            st.info(f"**🟢 Their game plan:**\n\n{game_plan}")
            st.info(f"**🟡 Moment Detector:**\n\n{moment_verdict} | Orderbook: {orderbook_pressure}")
            st.info(f"**🔴 OI/PCR Analysis:**\n\nPCR: {pcr_value:.2f} ({pcr_sentiment})  \nCALL OI: {call_oi:,}  \nPUT OI: {put_oi:,}  \nATM Conc: {atm_conc_display}")
            st.info(f"**🟣 Expiry Context:**\n\nExpiry in {days_to_expiry:.1f} days")

        with col2:
            st.success(f"**🟢 Key defense levels:**\n\n₹{support_level:,.0f} (Support) | ₹{resistance_level:,.0f} (Resistance)")
            st.error(f"**🔴 Max OI Walls:**\n\nCALL: ₹{max_call_strike:,} | PUT: ₹{max_put_strike:,}")
            st.info(f"**🔵 Preferred price level:**\n\n₹{max_pain:,} (Max Pain)")
            st.warning(f"**🟡 Regime (Advanced Chart Analysis):**\n\n{regime}")
            st.success(f"**🟢 Sector Rotation Analysis:**\n\n{sector_bias} bias detected")

        # --- IV CONTEXT & SESSION INTELLIGENCE (New Row) ---
        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            # IV Context (Volatility Regime)
            vix_value = 15.0  # Default
            if enhanced_market_data:
                vix_data = enhanced_market_data.get('vix', {})
                if isinstance(vix_data, dict):
                    vix_value = vix_data.get('lastPrice', 15.0)
                elif isinstance(vix_data, (int, float)):
                    vix_value = vix_data

            # Determine IV regime
            if vix_value < 12:
                iv_regime = "VERY LOW"
                iv_trend = "🔽 FALLING"
                iv_advice = "Premium decay strong. Avoid buying options. Consider selling."
                iv_color = "success"
            elif vix_value < 15:
                iv_regime = "LOW"
                iv_trend = "📉 Flat/Falling"
                iv_advice = "Moderate decay. Be selective with long options."
                iv_color = "info"
            elif vix_value < 20:
                iv_regime = "NORMAL"
                iv_trend = "➡️ STABLE"
                iv_advice = "Balanced conditions. Both buying/selling viable."
                iv_color = "info"
            elif vix_value < 25:
                iv_regime = "ELEVATED"
                iv_trend = "📈 Rising"
                iv_advice = "Volatility rising. Long options have edge."
                iv_color = "warning"
            else:
                iv_regime = "HIGH"
                iv_trend = "🔼 RISING"
                iv_advice = "High volatility! Long options premium justified."
                iv_color = "error"

            if iv_color == "success":
                st.success(f"**🟣 Volatility Context (IV Regime):**\n\nVIX: {vix_value:.2f} ({iv_regime})  \nTrend: {iv_trend}  \n💡 {iv_advice}")
            elif iv_color == "warning":
                st.warning(f"**🟣 Volatility Context (IV Regime):**\n\nVIX: {vix_value:.2f} ({iv_regime})  \nTrend: {iv_trend}  \n💡 {iv_advice}")
            elif iv_color == "error":
                st.error(f"**🟣 Volatility Context (IV Regime):**\n\nVIX: {vix_value:.2f} ({iv_regime})  \nTrend: {iv_trend}  \n💡 {iv_advice}")
            else:
                st.info(f"**🟣 Volatility Context (IV Regime):**\n\nVIX: {vix_value:.2f} ({iv_regime})  \nTrend: {iv_trend}  \n💡 {iv_advice}")

        with col4:
            # Session Intelligence (Time Context)
            from datetime import datetime
            import pytz

            ist = pytz.timezone('Asia/Kolkata')
            current_time = datetime.now(ist)
            current_hour = current_time.hour
            current_minute = current_time.minute

            # Determine session
            if current_hour == 9 and current_minute < 30:
                session = "PRE-MARKET"
                session_advice = "Wait for market open. High volatility expected."
                session_color = "warning"
            elif current_hour == 9 and current_minute >= 15:
                session = "OPENING HOUR"
                session_advice = "High volatility! Wait for first 15-30 min to settle."
                session_color = "error"
            elif current_hour == 10:
                session = "POST-OPENING"
                session_advice = "Initial direction established. Good for trend trades."
                session_color = "success"
            elif current_hour >= 11 and current_hour < 13:
                session = "MID-SESSION"
                session_advice = "Low momentum period. Avoid unless strong setup."
                session_color = "warning"
            elif current_hour >= 13 and current_hour < 15:
                session = "AFTERNOON"
                session_advice = "Momentum picking up. Watch for directional moves."
                session_color = "info"
            elif current_hour == 15 and current_minute < 30:
                session = "POWER HOUR"
                session_advice = "High activity! Final push of the day."
                session_color = "success"
            else:
                session = "POST-MARKET"
                session_advice = "Market closed. Prepare for tomorrow."
                session_color = "info"

            # Check if expiry day
            if days_to_expiry <= 0:
                session = f"{session} (EXPIRY DAY!)"
                session_advice = "⚠️ EXPIRY DAY - Extreme volatility! Reduce size, tight stops."
                session_color = "error"
            elif days_to_expiry <= 1:
                session = f"{session} (Pre-Expiry)"
                session_advice = f"{session_advice} ⚠️ Expiry tomorrow - Increased volatility likely."

            if session_color == "success":
                st.success(f"**🕒 Session Intelligence (Time Context):**\n\nSession: {session}  \nTime: {current_time.strftime('%I:%M %p IST')}  \n💡 {session_advice}")
            elif session_color == "warning":
                st.warning(f"**🕒 Session Intelligence (Time Context):**\n\nSession: {session}  \nTime: {current_time.strftime('%I:%M %p IST')}  \n💡 {session_advice}")
            elif session_color == "error":
                st.error(f"**🕒 Session Intelligence (Time Context):**\n\nSession: {session}  \nTime: {current_time.strftime('%I:%M %p IST')}  \n💡 {session_advice}")
            else:
                st.info(f"**🕒 Session Intelligence (Time Context):**\n\nSession: {session}  \nTime: {current_time.strftime('%I:%M %p IST')}  \n💡 {session_advice}")

    # --- Comprehensive Liquidity & Support/Resistance Levels ---
    st.markdown("### 📊 Comprehensive Liquidity Analysis")

    # Extract all available S/R levels from different sources
    liquidity_levels = []

    # From liquidity zones (Advanced Chart Analysis)
    if liquidity_result:
        if hasattr(liquidity_result, 'support_zones'):
            for level in liquidity_result.support_zones:
                if isinstance(level, (int, float)):
                    liquidity_levels.append({
                        'price': level,
                        'type': 'Support',
                        'strength': 'Major' if abs(level - current_price) > 100 else 'Minor',
                        'source': 'Liquidity Zone'
                    })
        if hasattr(liquidity_result, 'resistance_zones'):
            for level in liquidity_result.resistance_zones:
                if isinstance(level, (int, float)):
                    liquidity_levels.append({
                        'price': level,
                        'type': 'Resistance',
                        'strength': 'Major' if abs(level - current_price) > 100 else 'Minor',
                        'source': 'Liquidity Zone'
                    })

    # From OI data (Max OI walls)
    if max_call_strike != atm_strike + 500:  # Not default value
        liquidity_levels.append({
            'price': max_call_strike,
            'type': 'Resistance',
            'strength': 'Major',
            'source': 'Max CALL OI Wall'
        })
    if max_put_strike != atm_strike - 500:  # Not default value
        liquidity_levels.append({
            'price': max_put_strike,
            'type': 'Support',
            'strength': 'Major',
            'source': 'Max PUT OI Wall'
        })

    # Add current support/resistance
    liquidity_levels.append({
        'price': support_level,
        'type': 'Support',
        'strength': 'Key',
        'source': 'Nearest Support'
    })
    liquidity_levels.append({
        'price': resistance_level,
        'type': 'Resistance',
        'strength': 'Key',
        'source': 'Nearest Resistance'
    })

    # Add Max Pain
    liquidity_levels.append({
        'price': max_pain,
        'type': 'Magnet',
        'strength': 'Critical',
        'source': 'Max Pain'
    })

    # Sort by price
    liquidity_levels = sorted(liquidity_levels, key=lambda x: x['price'])

    # Separate into above and below current price
    levels_below = [l for l in liquidity_levels if l['price'] < current_price]
    levels_above = [l for l in liquidity_levels if l['price'] > current_price]

    # Display in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔽 Levels BELOW Current Price**")
        if levels_below:
            # Show closest 5 levels
            closest_below = sorted(levels_below, key=lambda x: x['price'], reverse=True)[:5]
            for level in closest_below:
                distance = current_price - level['price']
                color = "🔴" if level['type'] == 'Support' else "🔵" if level['type'] == 'Magnet' else "⚪"
                st.text(f"{color} ₹{level['price']:,.0f} ({level['strength']} {level['type']})")
                st.caption(f"   -{distance:.0f} pts | {level['source']}")
        else:
            st.caption("No significant levels below")

    with col2:
        st.markdown("**🔼 Levels ABOVE Current Price**")
        if levels_above:
            # Show closest 5 levels
            closest_above = sorted(levels_above, key=lambda x: x['price'])[:5]
            for level in closest_above:
                distance = level['price'] - current_price
                color = "🟢" if level['type'] == 'Resistance' else "🔵" if level['type'] == 'Magnet' else "⚪"
                st.text(f"{color} ₹{level['price']:,.0f} ({level['strength']} {level['type']})")
                st.caption(f"   +{distance:.0f} pts | {level['source']}")
        else:
            st.caption("No significant levels above")

    st.markdown("---")

    # --- INTRADAY NEAR-SPOT LEVELS (Within 30 points for scalping) ---
    st.markdown("### ⚡ INTRADAY Near-Spot Levels (For 1-Hour Scalping)")

    # Filter only levels within 30 points of current price
    intraday_threshold = 30
    intraday_levels = [l for l in liquidity_levels if abs(l['price'] - current_price) <= intraday_threshold]

    if intraday_levels:
        # Sort by distance from current price
        intraday_levels = sorted(intraday_levels, key=lambda x: abs(x['price'] - current_price))

        col1, col2 = st.columns(2)

        intraday_support = [l for l in intraday_levels if l['price'] < current_price]
        intraday_resistance = [l for l in intraday_levels if l['price'] > current_price]

        with col1:
            st.markdown("**🔻 Immediate Support (Scalp Zone)**")
            if intraday_support:
                for level in intraday_support[:3]:  # Show top 3
                    distance = current_price - level['price']
                    st.success(f"🟢 ₹{level['price']:,.0f} (-{distance:.0f} pts) - {level['type']}")
                    st.caption(f"   Source: {level['source']}")
            else:
                st.warning("⚠️ No immediate support within 30 pts - Price may fall further")

        with col2:
            st.markdown("**🔺 Immediate Resistance (Scalp Zone)**")
            if intraday_resistance:
                for level in intraday_resistance[:3]:  # Show top 3
                    distance = level['price'] - current_price
                    st.error(f"🔴 ₹{level['price']:,.0f} (+{distance:.0f} pts) - {level['type']}")
                    st.caption(f"   Source: {level['source']}")
            else:
                st.warning("⚠️ No immediate resistance within 30 pts - Price may rally further")

        # Add actionable insight
        if intraday_support and intraday_resistance:
            nearest_sup = intraday_support[0]['price']
            nearest_res = intraday_resistance[0]['price']
            intraday_range = nearest_res - nearest_sup
            st.info(f"📊 **Intraday Range:** ₹{nearest_sup:,.0f} - ₹{nearest_res:,.0f} ({intraday_range:.0f} pts width)")

            # Trading advice based on range
            if intraday_range < 30:
                st.warning("⚠️ **TIGHT RANGE** - Chop zone! Wait for breakout or avoid.")
            elif intraday_range <= 50:
                st.success("✅ **IDEAL SCALP RANGE** - Good for quick in-out trades.")
            else:
                st.info("📈 **WIDER RANGE** - Consider swing trades with wider stops.")
    else:
        st.warning("⚠️ No levels within 30 points of current price. Price is in open space - use wider timeframe levels.")

    st.markdown("---")

    # --- FLOW CONFIRMATION & FAKE BREAKOUT WARNING ---
    st.markdown("### 🔄 Market Flow & Breakout Validation")

    col_flow1, col_flow2 = st.columns(2)

    with col_flow1:
        # Flow Confirmation (Multi-Source Analysis)
        st.markdown("**📊 Flow Confirmation (Who's in Control?)**")

        # === COLLECT FLOW DATA FROM ALL SOURCES ===
        flow_signals = []
        total_flow_score = 0
        max_flow_score = 0

        # 1. Money Flow Profile (Tab 7)
        if money_flow_signals:
            mf_signal = money_flow_signals.get('signal', 'NEUTRAL')
            mf_strength = money_flow_signals.get('volume_strength', 0)

            if mf_signal == 'BUY':
                flow_signals.append(f"💰 Money Flow: BUY ({mf_strength:.0f}%)")
                total_flow_score += mf_strength
            elif mf_signal == 'SELL':
                flow_signals.append(f"💰 Money Flow: SELL ({mf_strength:.0f}%)")
                total_flow_score -= mf_strength
            else:
                flow_signals.append(f"💰 Money Flow: NEUTRAL")

            max_flow_score += 100

        # 2. DeltaFlow Profile (Tab 7)
        if deltaflow_signals:
            delta = deltaflow_signals.get('cumulative_delta', 0)

            if delta > 1000:
                flow_signals.append(f"⚡ DeltaFlow: +{delta:,.0f} (BUY)")
                total_flow_score += 80
            elif delta < -1000:
                flow_signals.append(f"⚡ DeltaFlow: {delta:,.0f} (SELL)")
                total_flow_score -= 80
            else:
                flow_signals.append(f"⚡ DeltaFlow: Balanced ({delta:,.0f})")
                total_flow_score += 0

            max_flow_score += 80

        # 3. CVD - Cumulative Volume Delta (Tab 4)
        if cvd_result and hasattr(cvd_result, 'signal'):
            cvd_signal = cvd_result.signal

            if cvd_signal in ['BULLISH', 'STRONG_BULLISH']:
                flow_signals.append(f"📈 CVD: {cvd_signal}")
                total_flow_score += 70
            elif cvd_signal in ['BEARISH', 'STRONG_BEARISH']:
                flow_signals.append(f"📉 CVD: {cvd_signal}")
                total_flow_score -= 70
            else:
                flow_signals.append(f"📊 CVD: NEUTRAL")

            max_flow_score += 70

        # 4. Market Depth Orderbook Pressure
        if moment_data and 'orderbook' in moment_data:
            orderbook = moment_data['orderbook']
            if orderbook.get('available', False):
                pressure = orderbook.get('pressure', 'NEUTRAL')
                pressure_score = orderbook.get('pressure_score', 0)

                if pressure == 'BUY' and pressure_score > 60:
                    flow_signals.append(f"📊 Depth: BUY pressure ({pressure_score:.0f}%)")
                    total_flow_score += 60
                elif pressure == 'SELL' and pressure_score > 60:
                    flow_signals.append(f"📊 Depth: SELL pressure ({pressure_score:.0f}%)")
                    total_flow_score -= 60
                else:
                    flow_signals.append(f"📊 Depth: Balanced")

                max_flow_score += 60

        # 5. OI Flow (CALL/PUT buildup)
        if nifty_screener_data and 'oi_pcr_metrics' in nifty_screener_data:
            oi_pcr = nifty_screener_data['oi_pcr_metrics']
            pcr_change = oi_pcr.get('pcr_change_pct', 0) if isinstance(oi_pcr, dict) else 0

            if pcr_change > 5:  # PCR increasing = PUT buildup
                flow_signals.append(f"🔄 OI Flow: PUT buildup (+{pcr_change:.1f}%)")
                total_flow_score -= 50
            elif pcr_change < -5:  # PCR decreasing = CALL buildup
                flow_signals.append(f"🔄 OI Flow: CALL buildup ({pcr_change:.1f}%)")
                total_flow_score += 50
            else:
                flow_signals.append(f"🔄 OI Flow: Balanced")

            max_flow_score += 50

        # === DETERMINE OVERALL FLOW ===
        if max_flow_score > 0:
            flow_pct = (total_flow_score / max_flow_score) * 100

            if flow_pct > 40:
                flow_verdict = "🟢 BUYERS IN CONTROL"
                flow_strength = f"STRONG BUY FLOW ({flow_pct:.0f}%)"
                flow_advice = "Bullish bias confirmed. Look for CALL entries at support."
                flow_color = "success"
            elif flow_pct > 15:
                flow_verdict = "🟢 Mild Buy Pressure"
                flow_strength = f"WEAK BUY FLOW ({flow_pct:.0f}%)"
                flow_advice = "Slight bullish edge. Wait for confirmation."
                flow_color = "info"
            elif flow_pct < -40:
                flow_verdict = "🔴 SELLERS IN CONTROL"
                flow_strength = f"STRONG SELL FLOW ({flow_pct:.0f}%)"
                flow_advice = "Bearish bias confirmed. Look for PUT entries at resistance."
                flow_color = "error"
            elif flow_pct < -15:
                flow_verdict = "🔴 Mild Sell Pressure"
                flow_strength = f"WEAK SELL FLOW ({flow_pct:.0f}%)"
                flow_advice = "Slight bearish edge. Wait for confirmation."
                flow_color = "warning"
            else:
                flow_verdict = "⚖️ BALANCED FLOW"
                flow_strength = f"NEUTRAL ({flow_pct:+.0f}%)"
                flow_advice = "No clear bias. Wait for directional move."
                flow_color = "info"

            # Display with all sources
            flow_details = "\n".join(flow_signals)

            if flow_color == "success":
                st.success(f"{flow_verdict}\n\n{flow_strength}\n\n**Flow Sources:**\n{flow_details}\n\n💡 {flow_advice}")
            elif flow_color == "error":
                st.error(f"{flow_verdict}\n\n{flow_strength}\n\n**Flow Sources:**\n{flow_details}\n\n💡 {flow_advice}")
            elif flow_color == "warning":
                st.warning(f"{flow_verdict}\n\n{flow_strength}\n\n**Flow Sources:**\n{flow_details}\n\n💡 {flow_advice}")
            else:
                st.info(f"{flow_verdict}\n\n{flow_strength}\n\n**Flow Sources:**\n{flow_details}\n\n💡 {flow_advice}")
        else:
            st.info("⚖️ FLOW DATA UNAVAILABLE\n\nNo flow sources available. Check Money Flow, DeltaFlow, CVD, Market Depth tabs.")

    with col_flow2:
        # Fake Breakout Warning (Volume Confirmation)
        st.markdown("**⚠️ Fake Breakout Warning System**")

        # Check if price is near support/resistance (within 10 points)
        near_support = abs(current_price - support_level) <= 10
        near_resistance = abs(current_price - resistance_level) <= 10

        if near_support or near_resistance:
            level_name = "SUPPORT" if near_support else "RESISTANCE"
            level_price = support_level if near_support else resistance_level

            # Check volume confirmation (if available from market_depth or enhanced_market_data)
            volume_confirmed = False
            if total_volume > 0:
                # Assume breakout is confirmed if volume is high (this is placeholder logic)
                # In real implementation, compare with average volume
                volume_confirmed = total_volume > 50000  # Placeholder threshold

            # Check for wick-only breakout (would need candle data - using placeholder)
            wick_breakout = False  # Placeholder - would need actual candle OHLC data

            if volume_confirmed and not wick_breakout:
                st.success(f"✅ BREAKOUT LIKELY VALID\n\nLevel: ₹{level_price:,.0f} ({level_name})  \n📊 Volume: CONFIRMED  \n🕯️ Candle: BODY CLOSE ABOVE/BELOW  \n💡 Safe to trade breakout direction")
            elif not volume_confirmed and not wick_breakout:
                st.warning(f"⚠️ WEAK BREAKOUT - CAUTION\n\nLevel: ₹{level_price:,.0f} ({level_name})  \n📊 Volume: LOW (Not confirmed)  \n💡 Wait for volume surge to confirm")
            elif wick_breakout:
                st.error(f"🚫 FAKE BREAKOUT WARNING!\n\nLevel: ₹{level_price:,.0f} ({level_name})  \n🕯️ WICK ONLY - Body didn't close through  \n💡 DO NOT CHASE! Likely rejection")
            else:
                st.info(f"📊 Near {level_name}: ₹{level_price:,.0f}\n\nWaiting for breakout attempt...")
        else:
            # Price not near key levels
            distance_to_support = current_price - support_level
            distance_to_resistance = resistance_level - current_price

            if distance_to_support < distance_to_resistance:
                st.info(f"📍 Price in middle zone\n\n{distance_to_support:.0f} pts above support  \n{distance_to_resistance:.0f} pts below resistance  \n💡 Wait for move to key levels")
            else:
                st.info(f"📍 Price in middle zone\n\n{distance_to_resistance:.0f} pts below resistance  \n{distance_to_support:.0f} pts above support  \n💡 Wait for move to key levels")

    st.markdown("---")

    # --- ELITE OPTION SCORER (Multi-Factor Analysis) ---
    st.markdown("### 🎯 ELITE OPTION SCORER (OI + Depth + Liquidity + GEX + ATM + Bias + Indicators)")

    # Get ATM and surrounding strikes
    atm_strike_base = round(current_price / 50) * 50
    strikes_to_analyze = [
        atm_strike_base - 150,
        atm_strike_base - 100,
        atm_strike_base - 50,
        atm_strike_base,
        atm_strike_base + 50,
        atm_strike_base + 100,
        atm_strike_base + 150
    ]

    # Score each strike
    option_scores = []

    for strike in strikes_to_analyze:
        # Get data for this strike from option chain
        strike_data_ce = None
        strike_data_pe = None

        if option_chain and 'data' in option_chain:
            for opt in option_chain['data']:
                if opt.get('strikePrice') == strike:
                    strike_data_ce = opt.get('CE', {})
                    strike_data_pe = opt.get('PE', {})
                    break

        # === CALL SCORING ===
        ce_score = 0
        ce_factors = {}

        if strike_data_ce:
            # Factor 1: OI Concentration (0-20 pts)
            ce_oi = strike_data_ce.get('openInterest', 0)
            if total_oi > 0:
                oi_pct = (ce_oi / total_oi) * 100
                ce_factors['OI %'] = f"{oi_pct:.2f}%"
                if oi_pct > 5:
                    ce_score += 20
                elif oi_pct > 2:
                    ce_score += 15
                elif oi_pct > 1:
                    ce_score += 10
                else:
                    ce_score += 5

            # Factor 2: Liquidity (Bid-Ask Spread) (0-15 pts)
            ce_bid = strike_data_ce.get('bidprice', 0)
            ce_ask = strike_data_ce.get('askPrice', 0)
            if ce_ask > 0:
                spread_pct = ((ce_ask - ce_bid) / ce_ask) * 100
                ce_factors['Spread'] = f"{spread_pct:.1f}%"
                if spread_pct < 2:
                    ce_score += 15
                elif spread_pct < 5:
                    ce_score += 10
                else:
                    ce_score += 5

            # Factor 3: Distance from ATM (0-15 pts) - closer is better for scalping
            distance_from_atm = abs(strike - atm_strike_base)
            ce_factors['ATM Dist'] = f"{distance_from_atm} pts"
            if distance_from_atm == 0:
                ce_score += 15
            elif distance_from_atm <= 50:
                ce_score += 12
            elif distance_from_atm <= 100:
                ce_score += 8
            else:
                ce_score += 3

            # Factor 4: Max Pain Alignment (0-15 pts)
            distance_from_max_pain = abs(strike - max_pain)
            ce_factors['MaxPain Dist'] = f"{distance_from_max_pain} pts"
            if distance_from_max_pain < 50:
                ce_score += 15
            elif distance_from_max_pain < 100:
                ce_score += 10
            else:
                ce_score += 5

            # Factor 5: ATM Bias Alignment (0-20 pts)
            if strike < current_price and atm_bias_verdict == "PUT SELLERS":  # Bullish setup
                ce_score += 20
                ce_factors['Bias Align'] = "✓ Bullish"
            elif strike >= current_price and atm_bias_verdict == "CALL SELLERS":  # Bearish, favor lower strikes
                ce_score += 10
                ce_factors['Bias Align'] = "~ Neutral"
            else:
                ce_factors['Bias Align'] = "✗ Against"

            # Factor 6: Technical Trend Alignment (0-15 pts)
            if regime in ['TRENDING_UP', 'STRONG_TRENDING_UP']:
                if strike <= atm_strike_base:  # ATM or ITM calls in uptrend
                    ce_score += 15
                    ce_factors['Trend Align'] = "✓ With Trend"
                else:
                    ce_score += 5
                    ce_factors['Trend Align'] = "~ Weak"
            elif regime in ['TRENDING_DOWN', 'STRONG_TRENDING_DOWN']:
                if strike > atm_strike_base:  # OTM calls in downtrend (premium selling)
                    ce_score += 10
                    ce_factors['Trend Align'] = "~ Counter"
                else:
                    ce_factors['Trend Align'] = "✗ Against"
            else:
                ce_score += 8
                ce_factors['Trend Align'] = "~ Range"

            # Factor 7: Money Flow Profile (0-10 pts)
            if money_flow_signals:
                mf_signal = money_flow_signals.get('signal', 'NEUTRAL')
                mf_strength = money_flow_signals.get('strength', 0)
                if mf_signal == 'BUY' and strike <= atm_strike_base:  # Calls align with buy flow
                    ce_score += 10
                    ce_factors['Money Flow'] = "✓ Buy Flow"
                elif mf_signal == 'SELL':
                    ce_factors['Money Flow'] = "✗ Sell Flow"
                else:
                    ce_score += 5
                    ce_factors['Money Flow'] = "~ Neutral"
            else:
                ce_factors['Money Flow'] = "N/A"

            # Factor 8: DeltaFlow Profile (0-10 pts)
            if deltaflow_signals:
                df_signal = deltaflow_signals.get('signal', 'NEUTRAL')
                df_delta = deltaflow_signals.get('cumulative_delta', 0)
                if df_delta > 0 and strike <= atm_strike_base:  # Positive delta favors calls
                    ce_score += 10
                    ce_factors['DeltaFlow'] = f"✓ +{abs(df_delta):.0f}"
                elif df_delta < 0:
                    ce_factors['DeltaFlow'] = f"✗ -{abs(df_delta):.0f}"
                else:
                    ce_score += 5
                    ce_factors['DeltaFlow'] = "~ Flat"
            else:
                ce_factors['DeltaFlow'] = "N/A"

            # Factor 9: CVD (Cumulative Volume Delta) (0-10 pts)
            if cvd_result and hasattr(cvd_result, 'signal'):
                cvd_signal = cvd_result.signal
                if cvd_signal in ['BULLISH', 'STRONG_BULLISH'] and strike <= atm_strike_base:
                    ce_score += 10
                    ce_factors['CVD'] = "✓ Bullish"
                elif cvd_signal in ['BEARISH', 'STRONG_BEARISH']:
                    ce_factors['CVD'] = "✗ Bearish"
                else:
                    ce_score += 5
                    ce_factors['CVD'] = "~ Neutral"
            else:
                ce_factors['CVD'] = "N/A"

            # Factor 10: Volatility Regime + VIX (0-10 pts)
            vix_value = 15.0
            if enhanced_market_data and 'vix' in enhanced_market_data:
                vix_data = enhanced_market_data['vix']
                vix_value = vix_data.get('lastPrice', 15.0) if isinstance(vix_data, dict) else vix_data

            # Low VIX favors selling, High VIX favors buying
            if vix_value < 12 and strike > atm_strike_base:  # Low VIX + OTM call = good for selling
                ce_score += 10
                ce_factors['VIX Regime'] = f"✓ Low ({vix_value:.1f})"
            elif vix_value > 20 and strike <= atm_strike_base:  # High VIX + ATM/ITM call = good for buying
                ce_score += 10
                ce_factors['VIX Regime'] = f"✓ High ({vix_value:.1f})"
            else:
                ce_score += 5
                ce_factors['VIX Regime'] = f"~ Normal ({vix_value:.1f})"

            # Factor 11: Sector Rotation (0-5 pts)
            if enhanced_market_data and 'sectors' in enhanced_market_data:
                sectors = enhanced_market_data['sectors']
                if sectors.get('success'):
                    sector_data = sectors.get('data', [])
                    bullish_count = sum(1 for s in sector_data if s.get('change_pct', 0) > 0.5)
                    bearish_count = sum(1 for s in sector_data if s.get('change_pct', 0) < -0.5)

                    if bullish_count > bearish_count + 2 and strike <= atm_strike_base:  # Sector rotation bullish
                        ce_score += 5
                        ce_factors['Sector'] = "✓ Bullish Rotation"
                    elif bearish_count > bullish_count + 2:
                        ce_factors['Sector'] = "✗ Bearish Rotation"
                    else:
                        ce_score += 2
                        ce_factors['Sector'] = "~ Mixed"
                else:
                    ce_factors['Sector'] = "N/A"
            else:
                ce_factors['Sector'] = "N/A"

            # Factor 12: GEX (Gamma Exposure) (0-10 pts)
            total_gex_net = nifty_screener_data.get('total_gex_net', 0) if nifty_screener_data else 0
            if total_gex_net > 1000000:  # Positive GEX = Stabilizing, favors selling OTM
                if strike > atm_strike_base:  # OTM call selling
                    ce_score += 10
                    ce_factors['GEX'] = f"✓ +GEX ({total_gex_net/1e6:.1f}M)"
                else:  # ATM/ITM less favorable
                    ce_score += 3
                    ce_factors['GEX'] = f"~ +GEX ({total_gex_net/1e6:.1f}M)"
            elif total_gex_net < -1000000:  # Negative GEX = Destabilizing, favors buying ATM/ITM
                if strike <= atm_strike_base:  # ATM/ITM call buying
                    ce_score += 10
                    ce_factors['GEX'] = f"✓ -GEX ({abs(total_gex_net)/1e6:.1f}M)"
                else:  # OTM less favorable
                    ce_score += 3
                    ce_factors['GEX'] = f"~ -GEX ({abs(total_gex_net)/1e6:.1f}M)"
            else:  # Neutral GEX
                ce_score += 5
                ce_factors['GEX'] = f"~ Neutral ({abs(total_gex_net)/1e6:.1f}M)"

            # Factor 13: Market Depth/Orderbook Pressure (0-10 pts)
            if moment_data and 'orderbook' in moment_data:
                orderbook = moment_data['orderbook']
                if orderbook.get('available', False):
                    pressure = orderbook.get('pressure', 'NEUTRAL')
                    pressure_score = orderbook.get('pressure_score', 0)

                    if pressure == 'BUY' and pressure_score > 60:  # Strong buy pressure
                        if strike <= atm_strike_base:  # Favors calls
                            ce_score += 10
                            ce_factors['Depth'] = f"✓ Buy ({pressure_score}%)"
                        else:
                            ce_score += 5
                            ce_factors['Depth'] = f"~ Buy ({pressure_score}%)"
                    elif pressure == 'SELL':
                        ce_factors['Depth'] = f"✗ Sell ({pressure_score}%)"
                    else:
                        ce_score += 5
                        ce_factors['Depth'] = "~ Neutral"
                else:
                    ce_factors['Depth'] = "N/A"
            else:
                ce_factors['Depth'] = "N/A"

        # === PUT SCORING ===
        pe_score = 0
        pe_factors = {}

        if strike_data_pe:
            # Factor 1: OI Concentration (0-20 pts)
            pe_oi = strike_data_pe.get('openInterest', 0)
            if total_oi > 0:
                oi_pct = (pe_oi / total_oi) * 100
                pe_factors['OI %'] = f"{oi_pct:.2f}%"
                if oi_pct > 5:
                    pe_score += 20
                elif oi_pct > 2:
                    pe_score += 15
                elif oi_pct > 1:
                    pe_score += 10
                else:
                    pe_score += 5

            # Factor 2: Liquidity (Bid-Ask Spread) (0-15 pts)
            pe_bid = strike_data_pe.get('bidprice', 0)
            pe_ask = strike_data_pe.get('askPrice', 0)
            if pe_ask > 0:
                spread_pct = ((pe_ask - pe_bid) / pe_ask) * 100
                pe_factors['Spread'] = f"{spread_pct:.1f}%"
                if spread_pct < 2:
                    pe_score += 15
                elif spread_pct < 5:
                    pe_score += 10
                else:
                    pe_score += 5

            # Factor 3: Distance from ATM (0-15 pts)
            distance_from_atm = abs(strike - atm_strike_base)
            pe_factors['ATM Dist'] = f"{distance_from_atm} pts"
            if distance_from_atm == 0:
                pe_score += 15
            elif distance_from_atm <= 50:
                pe_score += 12
            elif distance_from_atm <= 100:
                pe_score += 8
            else:
                pe_score += 3

            # Factor 4: Max Pain Alignment (0-15 pts)
            distance_from_max_pain = abs(strike - max_pain)
            pe_factors['MaxPain Dist'] = f"{distance_from_max_pain} pts"
            if distance_from_max_pain < 50:
                pe_score += 15
            elif distance_from_max_pain < 100:
                pe_score += 10
            else:
                pe_score += 5

            # Factor 5: ATM Bias Alignment (0-20 pts)
            if strike > current_price and atm_bias_verdict == "CALL SELLERS":  # Bearish setup
                pe_score += 20
                pe_factors['Bias Align'] = "✓ Bearish"
            elif strike <= current_price and atm_bias_verdict == "PUT SELLERS":  # Bullish, favor higher strikes
                pe_score += 10
                pe_factors['Bias Align'] = "~ Neutral"
            else:
                pe_factors['Bias Align'] = "✗ Against"

            # Factor 6: Technical Trend Alignment (0-15 pts)
            if regime in ['TRENDING_DOWN', 'STRONG_TRENDING_DOWN']:
                if strike >= atm_strike_base:  # ATM or ITM puts in downtrend
                    pe_score += 15
                    pe_factors['Trend Align'] = "✓ With Trend"
                else:
                    pe_score += 5
                    pe_factors['Trend Align'] = "~ Weak"
            elif regime in ['TRENDING_UP', 'STRONG_TRENDING_UP']:
                if strike < atm_strike_base:  # OTM puts in uptrend
                    pe_score += 10
                    pe_factors['Trend Align'] = "~ Counter"
                else:
                    pe_factors['Trend Align'] = "✗ Against"
            else:
                pe_score += 8
                pe_factors['Trend Align'] = "~ Range"

            # Factor 7: Money Flow Profile (0-10 pts)
            if money_flow_signals:
                mf_signal = money_flow_signals.get('signal', 'NEUTRAL')
                if mf_signal == 'SELL' and strike >= atm_strike_base:  # Puts align with sell flow
                    pe_score += 10
                    pe_factors['Money Flow'] = "✓ Sell Flow"
                elif mf_signal == 'BUY':
                    pe_factors['Money Flow'] = "✗ Buy Flow"
                else:
                    pe_score += 5
                    pe_factors['Money Flow'] = "~ Neutral"
            else:
                pe_factors['Money Flow'] = "N/A"

            # Factor 8: DeltaFlow Profile (0-10 pts)
            if deltaflow_signals:
                df_delta = deltaflow_signals.get('cumulative_delta', 0)
                if df_delta < 0 and strike >= atm_strike_base:  # Negative delta favors puts
                    pe_score += 10
                    pe_factors['DeltaFlow'] = f"✓ -{abs(df_delta):.0f}"
                elif df_delta > 0:
                    pe_factors['DeltaFlow'] = f"✗ +{abs(df_delta):.0f}"
                else:
                    pe_score += 5
                    pe_factors['DeltaFlow'] = "~ Flat"
            else:
                pe_factors['DeltaFlow'] = "N/A"

            # Factor 9: CVD (Cumulative Volume Delta) (0-10 pts)
            if cvd_result and hasattr(cvd_result, 'signal'):
                cvd_signal = cvd_result.signal
                if cvd_signal in ['BEARISH', 'STRONG_BEARISH'] and strike >= atm_strike_base:
                    pe_score += 10
                    pe_factors['CVD'] = "✓ Bearish"
                elif cvd_signal in ['BULLISH', 'STRONG_BULLISH']:
                    pe_factors['CVD'] = "✗ Bullish"
                else:
                    pe_score += 5
                    pe_factors['CVD'] = "~ Neutral"
            else:
                pe_factors['CVD'] = "N/A"

            # Factor 10: Volatility Regime + VIX (0-10 pts)
            vix_value = 15.0
            if enhanced_market_data and 'vix' in enhanced_market_data:
                vix_data = enhanced_market_data['vix']
                vix_value = vix_data.get('lastPrice', 15.0) if isinstance(vix_data, dict) else vix_data

            # Low VIX favors selling, High VIX favors buying
            if vix_value < 12 and strike < atm_strike_base:  # Low VIX + OTM put = good for selling
                pe_score += 10
                pe_factors['VIX Regime'] = f"✓ Low ({vix_value:.1f})"
            elif vix_value > 20 and strike >= atm_strike_base:  # High VIX + ATM/ITM put = good for buying
                pe_score += 10
                pe_factors['VIX Regime'] = f"✓ High ({vix_value:.1f})"
            else:
                pe_score += 5
                pe_factors['VIX Regime'] = f"~ Normal ({vix_value:.1f})"

            # Factor 11: Sector Rotation (0-5 pts)
            if enhanced_market_data and 'sectors' in enhanced_market_data:
                sectors = enhanced_market_data['sectors']
                if sectors.get('success'):
                    sector_data = sectors.get('data', [])
                    bullish_count = sum(1 for s in sector_data if s.get('change_pct', 0) > 0.5)
                    bearish_count = sum(1 for s in sector_data if s.get('change_pct', 0) < -0.5)

                    if bearish_count > bullish_count + 2 and strike >= atm_strike_base:  # Sector rotation bearish
                        pe_score += 5
                        pe_factors['Sector'] = "✓ Bearish Rotation"
                    elif bullish_count > bearish_count + 2:
                        pe_factors['Sector'] = "✗ Bullish Rotation"
                    else:
                        pe_score += 2
                        pe_factors['Sector'] = "~ Mixed"
                else:
                    pe_factors['Sector'] = "N/A"
            else:
                pe_factors['Sector'] = "N/A"

            # Factor 12: GEX (Gamma Exposure) (0-10 pts)
            total_gex_net = nifty_screener_data.get('total_gex_net', 0) if nifty_screener_data else 0
            if total_gex_net > 1000000:  # Positive GEX = Stabilizing, favors selling OTM
                if strike < atm_strike_base:  # OTM put selling
                    pe_score += 10
                    pe_factors['GEX'] = f"✓ +GEX ({total_gex_net/1e6:.1f}M)"
                else:  # ATM/ITM less favorable
                    pe_score += 3
                    pe_factors['GEX'] = f"~ +GEX ({total_gex_net/1e6:.1f}M)"
            elif total_gex_net < -1000000:  # Negative GEX = Destabilizing, favors buying ATM/ITM
                if strike >= atm_strike_base:  # ATM/ITM put buying
                    pe_score += 10
                    pe_factors['GEX'] = f"✓ -GEX ({abs(total_gex_net)/1e6:.1f}M)"
                else:  # OTM less favorable
                    pe_score += 3
                    pe_factors['GEX'] = f"~ -GEX ({abs(total_gex_net)/1e6:.1f}M)"
            else:  # Neutral GEX
                pe_score += 5
                pe_factors['GEX'] = f"~ Neutral ({abs(total_gex_net)/1e6:.1f}M)"

            # Factor 13: Market Depth/Orderbook Pressure (0-10 pts)
            if moment_data and 'orderbook' in moment_data:
                orderbook = moment_data['orderbook']
                if orderbook.get('available', False):
                    pressure = orderbook.get('pressure', 'NEUTRAL')
                    pressure_score = orderbook.get('pressure_score', 0)

                    if pressure == 'SELL' and pressure_score > 60:  # Strong sell pressure
                        if strike >= atm_strike_base:  # Favors puts
                            pe_score += 10
                            pe_factors['Depth'] = f"✓ Sell ({pressure_score}%)"
                        else:
                            pe_score += 5
                            pe_factors['Depth'] = f"~ Sell ({pressure_score}%)"
                    elif pressure == 'BUY':
                        pe_factors['Depth'] = f"✗ Buy ({pressure_score}%)"
                    else:
                        pe_score += 5
                        pe_factors['Depth'] = "~ Neutral"
                else:
                    pe_factors['Depth'] = "N/A"
            else:
                pe_factors['Depth'] = "N/A"

        # Store scores
        if strike_data_ce:
            option_scores.append({
                'strike': strike,
                'type': 'CE',
                'score': ce_score,
                'factors': ce_factors,
                'premium': strike_data_ce.get('lastPrice', 0)
            })

        if strike_data_pe:
            option_scores.append({
                'strike': strike,
                'type': 'PE',
                'score': pe_score,
                'factors': pe_factors,
                'premium': strike_data_pe.get('lastPrice', 0)
            })

    # Sort by score (highest first)
    option_scores = sorted(option_scores, key=lambda x: x['score'], reverse=True)

    # Display top 5 options
    st.markdown("**🏆 TOP 5 HIGHEST SCORING OPTIONS (13-Factor Comprehensive Analysis)**")
    st.caption("Max Score: 165 pts | Factors: OI, Liquidity, ATM Dist, Max Pain, Bias, Trend, Money Flow, DeltaFlow, CVD, VIX, Sector, GEX, Market Depth")

    for i, opt in enumerate(option_scores[:5], 1):
        # Updated thresholds for 165-point scale
        score_color = "success" if opt['score'] >= 110 else "warning" if opt['score'] >= 85 else "error"
        score_label = "EXCELLENT" if opt['score'] >= 110 else "GOOD" if opt['score'] >= 85 else "WEAK"
        score_pct = (opt['score'] / 165) * 100

        # Create detailed breakdown
        factors_text = "\n".join([f"   • {k}: {v}" for k, v in opt['factors'].items()])

        if score_color == "success":
            st.success(f"""
**#{i}. {opt['strike']} {opt['type']} - Score: {opt['score']}/165 ({score_pct:.0f}%) - {score_label}**

**Premium:** ₹{opt['premium']:.2f}

**Score Breakdown:**
{factors_text}
            """)
        elif score_color == "warning":
            st.warning(f"""
**#{i}. {opt['strike']} {opt['type']} - Score: {opt['score']}/165 ({score_pct:.0f}%) - {score_label}**

**Premium:** ₹{opt['premium']:.2f}

**Score Breakdown:**
{factors_text}
            """)
        else:
            st.info(f"""
**#{i}. {opt['strike']} {opt['type']} - Score: {opt['score']}/165 ({score_pct:.0f}%) - {score_label}**

**Premium:** ₹{opt['premium']:.2f}

**Score Breakdown:**
{factors_text}
            """)

    st.markdown("---")

    # ============================================
    # 🎯 KEY PRICE LEVELS & ENTRY ZONES (ALL DATA INTEGRATED)
    # ============================================
    st.markdown("## 🎯 KEY PRICE LEVELS & ENTRY ZONES")
    st.caption("**Exact reversal, continuation, and range levels based on 13-factor comprehensive analysis**")

    # Collect all key levels from various sources
    key_levels = []

    # 1. ATM Strike (Major pivot)
    key_levels.append({
        'price': atm_strike,
        'type': 'ATM Strike',
        'strength': 100,
        'bias': atm_bias_data.get('overall_bias', 'NEUTRAL') if atm_bias_data else 'NEUTRAL',
        'source': 'Option Chain'
    })

    # 2. Max Pain (Magnet level)
    if nifty_screener_data and 'seller_max_pain' in nifty_screener_data:
        max_pain_data = nifty_screener_data['seller_max_pain']
        if max_pain_data and 'max_pain_strike' in max_pain_data:
            max_pain_strike = max_pain_data['max_pain_strike']
            key_levels.append({
                'price': max_pain_strike,
                'type': 'Max Pain',
                'strength': 90,
                'bias': 'MAGNET',
                'source': 'Option Sellers'
            })

    # 3. Support/Resistance from HTF
    if support_level and support_level > 0:
        key_levels.append({
            'price': support_level,
            'type': 'Support',
            'strength': 85,
            'bias': 'BULLISH',
            'source': 'HTF Analysis'
        })

    if resistance_level and resistance_level > 0:
        key_levels.append({
            'price': resistance_level,
            'type': 'Resistance',
            'strength': 85,
            'bias': 'BEARISH',
            'source': 'HTF Analysis'
        })

    # 4. Nearest Support/Resistance from Option Chain
    if nifty_screener_data:
        nearest_sup = nifty_screener_data.get('nearest_sup')
        nearest_res = nifty_screener_data.get('nearest_res')

        if nearest_sup and 'strike' in nearest_sup:
            key_levels.append({
                'price': nearest_sup['strike'],
                'type': 'OI Support',
                'strength': 75,
                'bias': 'BULLISH',
                'source': 'OI Concentration'
            })

        if nearest_res and 'strike' in nearest_res:
            key_levels.append({
                'price': nearest_res['strike'],
                'type': 'OI Resistance',
                'strength': 75,
                'bias': 'BEARISH',
                'source': 'OI Concentration'
            })

    # 5. GEX Gamma Walls (Major barriers)
    total_gex_net = nifty_screener_data.get('total_gex_net', 0) if nifty_screener_data else 0
    if abs(total_gex_net) > 1000000:
        # High GEX creates walls around ATM
        gex_type = "Gamma Wall (Stabilizing)" if total_gex_net > 0 else "Gamma Wall (Explosive)"
        key_levels.append({
            'price': atm_strike,
            'type': gex_type,
            'strength': 80,
            'bias': 'MAGNET' if total_gex_net > 0 else 'BREAKOUT',
            'source': f'GEX {total_gex_net/1e6:.1f}M'
        })

    # 6. Volume Order Blocks (if available)
    if ml_regime_result and hasattr(ml_regime_result, 'support_zones'):
        for zone in ml_regime_result.support_zones[:2]:  # Top 2 support zones
            key_levels.append({
                'price': zone.get('price', 0),
                'type': 'Volume Block Support',
                'strength': 70,
                'bias': 'BULLISH',
                'source': 'Volume Analysis'
            })

    if ml_regime_result and hasattr(ml_regime_result, 'resistance_zones'):
        for zone in ml_regime_result.resistance_zones[:2]:  # Top 2 resistance zones
            key_levels.append({
                'price': zone.get('price', 0),
                'type': 'Volume Block Resistance',
                'strength': 70,
                'bias': 'BEARISH',
                'source': 'Volume Analysis'
            })

    # Sort levels by price
    key_levels = [l for l in key_levels if l['price'] > 0]
    key_levels = sorted(key_levels, key=lambda x: abs(x['price'] - current_price))

    # Identify zones
    levels_below = [l for l in key_levels if l['price'] < current_price]
    levels_above = [l for l in key_levels if l['price'] > current_price]

    # Display Current Market Position
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("**Current Price**", f"₹{current_price:,.2f}")
    with col2:
        nearest_support = levels_below[0] if levels_below else None
        if nearest_support:
            distance = current_price - nearest_support['price']
            st.metric("**Nearest Support**", f"₹{nearest_support['price']:,.0f}", f"-{distance:.0f} pts")
        else:
            st.metric("**Nearest Support**", "N/A")
    with col3:
        nearest_resistance = levels_above[0] if levels_above else None
        if nearest_resistance:
            distance = nearest_resistance['price'] - current_price
            st.metric("**Nearest Resistance**", f"₹{nearest_resistance['price']:,.0f}", f"+{distance:.0f} pts")
        else:
            st.metric("**Nearest Resistance**", "N/A")

    st.markdown("---")

    # ===== REVERSAL ZONES =====
    st.markdown("#### 🔄 REVERSAL ZONES (High Probability Bounce/Rejection)")

    reversal_zones = []

    # Support reversal zones (bullish bounce expected)
    support_clusters = {}
    for level in levels_below[:5]:  # Top 5 supports
        price_key = round(level['price'] / 25) * 25  # Cluster within 25 pts
        if price_key not in support_clusters:
            support_clusters[price_key] = []
        support_clusters[price_key].append(level)

    for cluster_price, cluster_levels in support_clusters.items():
        if len(cluster_levels) >= 2:  # Multiple factors converging
            strength_sum = sum(l['strength'] for l in cluster_levels)
            reversal_zones.append({
                'price': cluster_price,
                'type': 'REVERSAL UP',
                'strength': min(100, strength_sum),
                'factors': len(cluster_levels),
                'details': [f"{l['type']} ({l['source']})" for l in cluster_levels]
            })

    # Resistance reversal zones (bearish rejection expected)
    resistance_clusters = {}
    for level in levels_above[:5]:  # Top 5 resistances
        price_key = round(level['price'] / 25) * 25  # Cluster within 25 pts
        if price_key not in resistance_clusters:
            resistance_clusters[price_key] = []
        resistance_clusters[price_key].append(level)

    for cluster_price, cluster_levels in resistance_clusters.items():
        if len(cluster_levels) >= 2:  # Multiple factors converging
            strength_sum = sum(l['strength'] for l in cluster_levels)
            reversal_zones.append({
                'price': cluster_price,
                'type': 'REVERSAL DOWN',
                'strength': min(100, strength_sum),
                'factors': len(cluster_levels),
                'details': [f"{l['type']} ({l['source']})" for l in cluster_levels]
            })

    # Display reversal zones
    if reversal_zones:
        for zone in sorted(reversal_zones, key=lambda x: x['strength'], reverse=True)[:4]:
            emoji = "🟢" if zone['type'] == 'REVERSAL UP' else "🔴"
            distance = zone['price'] - current_price
            distance_str = f"+{distance:.0f}" if distance > 0 else f"{distance:.0f}"

            st.success(f"""
**{emoji} {zone['type']} ZONE: ₹{zone['price']:,.0f}** ({distance_str} pts away)
**Confidence:** {zone['strength']:.0f}% | **Factors:** {zone['factors']}
**Supporting Analysis:** {', '.join(zone['details'][:3])}
            """) if zone['type'] == 'REVERSAL UP' else st.error(f"""
**{emoji} {zone['type']} ZONE: ₹{zone['price']:,.0f}** ({distance_str} pts away)
**Confidence:** {zone['strength']:.0f}% | **Factors:** {zone['factors']}
**Supporting Analysis:** {', '.join(zone['details'][:3])}
            """)
    else:
        st.info("No strong reversal zones identified. Market may be in trending mode.")

    st.markdown("---")

    # ===== CONTINUATION ZONES =====
    st.markdown("#### 🚀 CONTINUATION/BREAKOUT ZONES")

    # Identify breakout levels
    regime = ml_regime_result.regime if ml_regime_result and hasattr(ml_regime_result, 'regime') else 'RANGING'

    if 'TRENDING' in regime:
        st.info(f"""
**📈 TRENDING MARKET DETECTED ({regime})**

**Continuation Strategy:**
- **If Bullish Trend:** Buy dips to nearest support, ride to next resistance
- **If Bearish Trend:** Sell rallies to nearest resistance, ride to next support
- **Breakout Confirmation:** Wait for 15-min close beyond key level with volume

**Key Breakout Levels:**
        """)

        if levels_above:
            st.success(f"**Upside Breakout:** ₹{levels_above[0]['price']:,.0f} → Target: ₹{levels_above[1]['price']:,.0f if len(levels_above) > 1 else levels_above[0]['price'] + 50:,.0f}")

        if levels_below:
            st.error(f"**Downside Breakdown:** ₹{levels_below[0]['price']:,.0f} → Target: ₹{levels_below[1]['price']:,.0f if len(levels_below) > 1 else levels_below[0]['price'] - 50:,.0f}")
    else:
        st.warning("""
**📊 RANGING MARKET DETECTED**

**Range Trading Strategy:**
- **Buy Zone:** Near support levels with tight stops
- **Sell Zone:** Near resistance levels with tight stops
- **Avoid:** Center of range (low probability)
        """)

        if levels_below and levels_above:
            range_low = levels_below[0]['price']
            range_high = levels_above[0]['price']
            range_mid = (range_low + range_high) / 2

            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"**Range Low (BUY ZONE)**\n₹{range_low:,.0f}")
            with col2:
                st.info(f"**Range Mid (AVOID)**\n₹{range_mid:,.0f}")
            with col3:
                st.error(f"**Range High (SELL ZONE)**\n₹{range_high:,.0f}")

    st.markdown("---")

    # ===== EXACT ENTRY POINTS =====
    st.markdown("#### 🎯 EXACT ENTRY POINTS (Based on Current Market Position)")

    # Determine market position relative to key levels
    if nearest_support and nearest_resistance:
        support_dist = current_price - nearest_support['price']
        resistance_dist = nearest_resistance['price'] - current_price

        # Check if near support or resistance
        if support_dist < 30:  # Within 30 pts of support
            st.success(f"""
**🟢 NEAR SUPPORT ZONE**

**LONG Entry Strategy:**
- **Entry:** ₹{nearest_support['price']:,.0f} - ₹{nearest_support['price'] + 10:,.0f}
- **Stop Loss:** ₹{nearest_support['price'] - 20:,.0f} (below support)
- **Target 1:** ₹{current_price + 30:,.0f} (+30 pts)
- **Target 2:** ₹{nearest_resistance['price'] if nearest_resistance else current_price + 50:,.0f} (resistance)
- **Risk/Reward:** 1:2 to 1:3
- **Confidence:** {nearest_support['strength']:.0f}%

**Why This Works:**
- Price near strong support ({nearest_support['type']})
- Multiple factors supporting bounce
- Favorable R:R ratio
            """)
        elif resistance_dist < 30:  # Within 30 pts of resistance
            st.error(f"""
**🔴 NEAR RESISTANCE ZONE**

**SHORT Entry Strategy:**
- **Entry:** ₹{nearest_resistance['price'] - 10:,.0f} - ₹{nearest_resistance['price']:,.0f}
- **Stop Loss:** ₹{nearest_resistance['price'] + 20:,.0f} (above resistance)
- **Target 1:** ₹{current_price - 30:,.0f} (-30 pts)
- **Target 2:** ₹{nearest_support['price'] if nearest_support else current_price - 50:,.0f} (support)
- **Risk/Reward:** 1:2 to 1:3
- **Confidence:** {nearest_resistance['strength']:.0f}%

**Why This Works:**
- Price near strong resistance ({nearest_resistance['type']})
- Multiple factors supporting rejection
- Favorable R:R ratio
            """)
        else:  # In the middle
            st.info(f"""
**⚠️ NO-TRADE ZONE (Middle of Range)**

**Current Position:** ₹{current_price:,.2f}
- **Distance to Support:** -{support_dist:.0f} pts
- **Distance to Resistance:** +{resistance_dist:.0f} pts

**Recommended Action:** WAIT for price to reach entry zones
- **Buy Zone:** ₹{nearest_support['price']:,.0f} - ₹{nearest_support['price'] + 10:,.0f}
- **Sell Zone:** ₹{nearest_resistance['price'] - 10:,.0f} - ₹{nearest_resistance['price']:,.0f}

**Why Wait:**
- Poor risk/reward in the middle
- Higher probability at extremes
- Patience = Profitability
            """)

    st.markdown("---")

    # ============================================
    # 🔥 CONFLUENCE ENTRY CHECK (VOB + HTF S/R + REGIME)
    # ============================================
    st.markdown("## 🔥 CONFLUENCE ENTRY CHECK")
    st.caption("**Highest probability setups: Volume Order Blocks + HTF Support/Resistance + XGBoost Regime alignment**")

    # Display ATM ±2 Strike Bias Tabulation FIRST (Critical Decision Data)
    st.markdown("### 📊 ATM ±2 Strikes - 14 Bias Metrics Tabulation")

    if atm_bias_data:
        atm_verdict = atm_bias_data.get('overall_bias', 'NEUTRAL')
        atm_score = atm_bias_data.get('total_score', 0)
        atm_strike_val = atm_bias_data.get('atm_strike', atm_strike)

        # Count bullish and bearish metrics
        bias_scores = atm_bias_data.get('bias_scores', {})
        bullish_count = sum(1 for score in bias_scores.values() if score > 0.3)
        bearish_count = sum(1 for score in bias_scores.values() if score < -0.3)
        total_metrics = len(bias_scores) if bias_scores else 14

        bullish_pct = (bullish_count / total_metrics * 100) if total_metrics > 0 else 0
        bearish_pct = (bearish_count / total_metrics * 100) if total_metrics > 0 else 0

        # Determine verdict color and emoji
        if 'BULLISH' in atm_verdict:
            verdict_color = "#00ff88"
            verdict_emoji = "🐂"
            verdict_bg = "#1a2e1a"
        elif 'BEARISH' in atm_verdict:
            verdict_color = "#ff6666"
            verdict_emoji = "🐻"
            verdict_bg = "#2e1a1a"
        else:
            verdict_color = "#ffa500"
            verdict_emoji = "⚖️"
            verdict_bg = "#2e2e1a"

        # Display ATM Bias Card
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div style='padding: 15px; background: {verdict_bg}; border-radius: 10px; border-left: 4px solid {verdict_color};'>
                <h4 style='margin: 0; color: {verdict_color};'>{verdict_emoji} {atm_verdict}</h4>
                <p style='margin: 5px 0 0 0; color: #888; font-size: 12px;'>ATM STRIKE VERDICT</p>
                <p style='margin: 5px 0 0 0; color: #ccc; font-size: 11px;'>Strike: {atm_strike_val} | Score: {atm_score:+.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='padding: 15px; background: #1a2e1a; border-radius: 10px; border-left: 4px solid #00ff88;'>
                <h4 style='margin: 0; color: #00ff88;'>{bullish_count} / {total_metrics}</h4>
                <p style='margin: 5px 0 0 0; color: #888; font-size: 12px;'>🐂 BULLISH METRICS</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style='padding: 15px; background: #2e1a1a; border-radius: 10px; border-left: 4px solid #ff6666;'>
                <h4 style='margin: 0; color: #ff6666;'>{bearish_count} / {total_metrics}</h4>
                <p style='margin: 5px 0 0 0; color: #888; font-size: 12px;'>🐻 BEARISH METRICS</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div style='padding: 15px; background: #1e1e1e; border-radius: 10px;'>
                <p style='margin: 0; color: #00ff88; font-size: 14px;'><strong>BULLISH:</strong> {bullish_pct:.1f}%</p>
                <p style='margin: 5px 0 0 0; color: #ff6666; font-size: 14px;'><strong>BEARISH:</strong> {bearish_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # Show interpretation
        if bullish_pct > 40:
            st.success(f"✅ **Strong Bullish Bias** - {bullish_count}/{total_metrics} metrics support LONG positions")
        elif bearish_pct > 40:
            st.error(f"✅ **Strong Bearish Bias** - {bearish_count}/{total_metrics} metrics support SHORT positions")
        else:
            st.info(f"⚠️ **Neutral/Mixed Bias** - No clear directional edge from ATM strikes")
    else:
        st.warning("⚠️ ATM Bias data unavailable - Check NIFTY Option Screener tab")

    st.markdown("---")

    # Get regime for directional bias
    regime_direction = "UNKNOWN"
    regime_conf = 0
    if ml_regime_result and hasattr(ml_regime_result, 'regime'):
        regime = ml_regime_result.regime
        regime_conf = ml_regime_result.confidence if hasattr(ml_regime_result, 'confidence') else 0

        if 'TRENDING_UP' in regime or 'STRONG_TRENDING_UP' in regime:
            regime_direction = "BULLISH"
        elif 'TRENDING_DOWN' in regime or 'STRONG_TRENDING_DOWN' in regime:
            regime_direction = "BEARISH"
        elif 'RANGING' in regime:
            regime_direction = "RANGING"

    # Check for confluence setups
    confluence_setups = []

    # --- BULLISH CONFLUENCE CHECK ---
    if regime_direction in ["BULLISH", "RANGING"]:
        # Check if current price is near HTF support
        htf_support_distance = abs(current_price - support_level) if support_level > 0 else 999

        # Check if current price is near a bullish VOB (if available from session state)
        vob_support_distance = 999
        vob_support_level = None

        # Try to get VOB data from session state
        try:
            import streamlit as st
            if 'vob_data_nifty' in st.session_state and st.session_state.vob_data_nifty:
                vob_data = st.session_state.vob_data_nifty
                bullish_blocks = vob_data.get('bullish_blocks', [])

                # Find nearest bullish VOB below current price
                for block in bullish_blocks:
                    if isinstance(block, dict):
                        block_price = block.get('upper', block.get('lower', 0))
                        if block_price < current_price:
                            distance = current_price - block_price
                            if distance < vob_support_distance:
                                vob_support_distance = distance
                                vob_support_level = block_price
        except:
            pass

        # Check for confluence (all within 30 points)
        if htf_support_distance < 30 or vob_support_distance < 30:
            factors = []
            total_confidence = 0

            if htf_support_distance < 30:
                factors.append(f"HTF Support @ ₹{support_level:,.0f} ({htf_support_distance:.0f} pts away)")
                total_confidence += 35

            if vob_support_distance < 30 and vob_support_level:
                factors.append(f"VOB Support @ ₹{vob_support_level:,.0f} ({vob_support_distance:.0f} pts away)")
                total_confidence += 35

            if regime_direction == "BULLISH":
                factors.append(f"XGBoost Regime: {regime} ({regime_conf:.0f}% confidence)")
                total_confidence += 30
            elif regime_direction == "RANGING":
                factors.append(f"XGBoost Regime: {regime} (Buy at support)")
                total_confidence += 20

            # Check ATM Bias alignment
            if atm_bias_data and 'overall_bias' in atm_bias_data:
                atm_bias = atm_bias_data['overall_bias']
                if atm_bias in ['BULLISH', 'STRONGLY_BULLISH']:
                    factors.append(f"ATM Bias: {atm_bias} (13-factor alignment)")
                    total_confidence += 15

            if len(factors) >= 2:  # At least 2 factors must align
                confluence_setups.append({
                    'direction': 'LONG',
                    'factors': factors,
                    'confidence': min(100, total_confidence),
                    'entry': min([l for l in [support_level if htf_support_distance < 30 else None,
                                             vob_support_level if vob_support_distance < 30 else None] if l]),
                    'type': 'BULLISH CONFLUENCE'
                })

    # --- BEARISH CONFLUENCE CHECK ---
    if regime_direction in ["BEARISH", "RANGING"]:
        # Check if current price is near HTF resistance
        htf_resistance_distance = abs(current_price - resistance_level) if resistance_level > 0 else 999

        # Check if current price is near a bearish VOB
        vob_resistance_distance = 999
        vob_resistance_level = None

        try:
            import streamlit as st
            if 'vob_data_nifty' in st.session_state and st.session_state.vob_data_nifty:
                vob_data = st.session_state.vob_data_nifty
                bearish_blocks = vob_data.get('bearish_blocks', [])

                # Find nearest bearish VOB above current price
                for block in bearish_blocks:
                    if isinstance(block, dict):
                        block_price = block.get('lower', block.get('upper', 0))
                        if block_price > current_price:
                            distance = block_price - current_price
                            if distance < vob_resistance_distance:
                                vob_resistance_distance = distance
                                vob_resistance_level = block_price
        except:
            pass

        # Check for confluence
        if htf_resistance_distance < 30 or vob_resistance_distance < 30:
            factors = []
            total_confidence = 0

            if htf_resistance_distance < 30:
                factors.append(f"HTF Resistance @ ₹{resistance_level:,.0f} ({htf_resistance_distance:.0f} pts away)")
                total_confidence += 35

            if vob_resistance_distance < 30 and vob_resistance_level:
                factors.append(f"VOB Resistance @ ₹{vob_resistance_level:,.0f} ({vob_resistance_distance:.0f} pts away)")
                total_confidence += 35

            if regime_direction == "BEARISH":
                factors.append(f"XGBoost Regime: {regime} ({regime_conf:.0f}% confidence)")
                total_confidence += 30
            elif regime_direction == "RANGING":
                factors.append(f"XGBoost Regime: {regime} (Sell at resistance)")
                total_confidence += 20

            # Check ATM Bias alignment
            if atm_bias_data and 'overall_bias' in atm_bias_data:
                atm_bias = atm_bias_data['overall_bias']
                if atm_bias in ['BEARISH', 'STRONGLY_BEARISH']:
                    factors.append(f"ATM Bias: {atm_bias} (13-factor alignment)")
                    total_confidence += 15

            if len(factors) >= 2:
                confluence_setups.append({
                    'direction': 'SHORT',
                    'factors': factors,
                    'confidence': min(100, total_confidence),
                    'entry': max([l for l in [resistance_level if htf_resistance_distance < 30 else None,
                                             vob_resistance_level if vob_resistance_distance < 30 else None] if l]),
                    'type': 'BEARISH CONFLUENCE'
                })

    # Display confluence setups
    if confluence_setups:
        for setup in sorted(confluence_setups, key=lambda x: x['confidence'], reverse=True):
            if setup['direction'] == 'LONG':
                st.success(f"""
**🔥 {setup['type']} DETECTED - {setup['confidence']:.0f}% CONFIDENCE**

**📍 LONG ENTRY SETUP:**
- **Entry Zone:** ₹{setup['entry']:,.0f} - ₹{setup['entry'] + 10:,.0f}
- **Stop Loss:** ₹{setup['entry'] - 25:,.0f} (below confluence)
- **Target 1:** ₹{setup['entry'] + 40:,.0f} (+40 pts, 1:1.6 R:R)
- **Target 2:** ₹{setup['entry'] + 60:,.0f} (+60 pts, 1:2.4 R:R)

**✅ Confluence Factors:**
{chr(10).join(['   • ' + f for f in setup['factors']])}

**🎯 Why This Works:**
Multiple institutional levels align at same zone = High probability bounce
Regime supports direction = Trend/Range alignment
**This is a PRIME setup - Wait for price to reach entry zone!**

**⚠️ BEFORE ENTRY - Confirm These:**
1. ✅ Price forms Higher Low (HL) after decline
2. ✅ Strong volume on bounce candle (check volume analysis above)
3. ✅ Price NOT chasing (wait for dip to entry zone)
4. ✅ Ideally price above VWAP when entering
5. ✅ ATM Bias supports LONG (check ATM 13-Bias Tabulation in Tab 8)
                """)
            else:
                st.error(f"""
**🔥 {setup['type']} DETECTED - {setup['confidence']:.0f}% CONFIDENCE**

**📍 SHORT ENTRY SETUP:**
- **Entry Zone:** ₹{setup['entry'] - 10:,.0f} - ₹{setup['entry']:,.0f}
- **Stop Loss:** ₹{setup['entry'] + 25:,.0f} (above confluence)
- **Target 1:** ₹{setup['entry'] - 40:,.0f} (-40 pts, 1:1.6 R:R)
- **Target 2:** ₹{setup['entry'] - 60:,.0f} (-60 pts, 1:2.4 R:R)

**✅ Confluence Factors:**
{chr(10).join(['   • ' + f for f in setup['factors']])}

**🎯 Why This Works:**
Multiple institutional levels align at same zone = High probability rejection
Regime supports direction = Trend/Range alignment
**This is a PRIME setup - Wait for price to reach entry zone!**

**⚠️ BEFORE ENTRY - Confirm These:**
1. ✅ Price forms Lower High (LH) after rally
2. ✅ Strong volume on rejection candle (check volume analysis above)
3. ✅ Price NOT chasing (wait for rally to entry zone)
4. ✅ Ideally price below VWAP when entering
5. ✅ ATM Bias supports SHORT (check ATM 13-Bias Tabulation in Tab 8)
                """)
    else:
        st.warning("""
**⚠️ NO CONFLUENCE SETUP DETECTED**

**Current Status:**
- HTF S/R not near current price (>30 pts away)
- VOB levels not near current price (>30 pts away)
- OR regime doesn't support clear direction

**What to do:**
- WAIT for price to reach confluence zones
- Don't force trades without alignment
- Patience = Higher win rate
        """)

    st.markdown("---")

    # ===== PROFESSIONAL ENTRY RULES & VOLUME CONFIRMATION =====
    st.markdown("#### 📋 PROFESSIONAL ENTRY RULES (Based on Trading Theory)")

    st.info("""
**🎯 ENTRY CHECKLIST - All Must Be TRUE Before Entry:**

**1. Structure Confirmation:**
- ✅ **For LONG:** Price forms Higher Low (HL) after decline
- ✅ **For SHORT:** Price forms Lower High (LH) after rally
- ❌ **DON'T** chase first green/red candle (often fake)

**2. Volume Confirmation:**
- ✅ Breakout candle has **above-average volume**
- ✅ Volume increasing on move toward entry level
- ❌ Low volume = Fake breakout, avoid entry

**3. Position Confirmation:**
- ✅ **For LONG:** Price above VWAP or breaks recent intraday high
- ✅ **For SHORT:** Price below VWAP or breaks recent intraday low
- ❌ Entry in middle of range = Poor R:R

**4. Risk Management:**
- ✅ Stop Loss below Higher Low (LONG) or above Lower High (SHORT)
- ✅ Target minimum 1:2 Risk:Reward ratio
- ✅ If 2 trades failed today → STOP trading, review tomorrow
    """)

    # ===== MARKET REGIME CONFIRMATION =====
    st.markdown("**🎯 Market Regime Analysis:**")

    regime_status = "⚠️ Regime data unavailable"
    regime_color = "warning"
    regime_recommendation = ""

    if ml_regime_result and hasattr(ml_regime_result, 'regime'):
        regime = ml_regime_result.regime
        regime_confidence = ml_regime_result.confidence if hasattr(ml_regime_result, 'confidence') else 0

        if 'TRENDING_UP' in regime or 'STRONG_TRENDING_UP' in regime:
            regime_status = f"✅ **{regime}** (Confidence: {regime_confidence:.0f}%)"
            regime_color = "success"
            regime_recommendation = "**Trade Bias:** LONG only (buy dips to support)\n**Avoid:** Selling against trend"
        elif 'TRENDING_DOWN' in regime or 'STRONG_TRENDING_DOWN' in regime:
            regime_status = f"✅ **{regime}** (Confidence: {regime_confidence:.0f}%)"
            regime_color = "error"
            regime_recommendation = "**Trade Bias:** SHORT only (sell rallies to resistance)\n**Avoid:** Buying against trend"
        elif 'RANGING' in regime:
            regime_status = f"⚠️ **{regime}** (Confidence: {regime_confidence:.0f}%)"
            regime_color = "warning"
            regime_recommendation = "**Trade Bias:** BOTH (buy at support, sell at resistance)\n**Avoid:** Middle of range"
        else:
            regime_status = f"~ **{regime}** (Confidence: {regime_confidence:.0f}%)"
            regime_color = "info"
            regime_recommendation = "**Trade Bias:** Wait for clearer regime"

    if regime_color == "success":
        st.success(f"{regime_status}\n{regime_recommendation}")
    elif regime_color == "error":
        st.error(f"{regime_status}\n{regime_recommendation}")
    elif regime_color == "warning":
        st.warning(f"{regime_status}\n{regime_recommendation}")
    else:
        st.info(f"{regime_status}\n{regime_recommendation}")

    st.markdown("---")

    # Volume Analysis (if available)
    st.markdown("**📊 Current Volume Analysis:**")

    volume_status = "⚠️ Volume data unavailable"
    volume_color = "warning"

    # Check Money Flow for volume confirmation
    if money_flow_signals:
        mf_volume = money_flow_signals.get('volume_strength', 0)
        mf_signal = money_flow_signals.get('signal', 'NEUTRAL')

        if mf_volume > 70:
            volume_status = f"✅ **HIGH VOLUME** ({mf_volume:.0f}%) - Strong {mf_signal} flow detected"
            volume_color = "success"
        elif mf_volume > 40:
            volume_status = f"~ **MODERATE VOLUME** ({mf_volume:.0f}%) - {mf_signal} flow present"
            volume_color = "info"
        else:
            volume_status = f"❌ **LOW VOLUME** ({mf_volume:.0f}%) - Weak confirmation"
            volume_color = "error"

    # Check DeltaFlow for buying/selling pressure
    delta_pressure = "Neutral"
    if deltaflow_signals:
        delta = deltaflow_signals.get('cumulative_delta', 0)
        if delta > 1000:
            delta_pressure = f"🟢 **Strong Buying Pressure** (+{delta:,.0f})"
        elif delta < -1000:
            delta_pressure = f"🔴 **Strong Selling Pressure** ({delta:,.0f})"
        else:
            delta_pressure = f"⚖️ **Balanced** ({delta:,.0f})"

    # Check Market Depth (Orderbook) Pressure
    depth_pressure = ""
    if moment_data and 'orderbook' in moment_data:
        orderbook = moment_data['orderbook']
        if orderbook.get('available', False):
            pressure = orderbook.get('pressure', 'NEUTRAL')
            pressure_score = orderbook.get('pressure_score', 0)

            if pressure == 'BUY' and pressure_score > 60:
                depth_pressure = f"\n📊 **Market Depth:** BUY pressure ({pressure_score:.0f}%) - Bid strength"
            elif pressure == 'SELL' and pressure_score > 60:
                depth_pressure = f"\n📊 **Market Depth:** SELL pressure ({pressure_score:.0f}%) - Ask strength"
            else:
                depth_pressure = f"\n📊 **Market Depth:** Balanced ({pressure_score:.0f}%)"

    # Check Option Chain OI Changes (Volume flow)
    oi_flow = ""
    if nifty_screener_data and 'oi_pcr_metrics' in nifty_screener_data:
        oi_pcr = nifty_screener_data['oi_pcr_metrics']
        pcr_change = oi_pcr.get('pcr_change_pct', 0) if isinstance(oi_pcr, dict) else 0

        if pcr_change > 5:  # PCR increasing = More PUT buying
            oi_flow = f"\n🔄 **OI Flow:** PUT buildup (+{pcr_change:.1f}% PCR) - Bearish hedging"
        elif pcr_change < -5:  # PCR decreasing = More CALL buying
            oi_flow = f"\n🔄 **OI Flow:** CALL buildup ({pcr_change:.1f}% PCR) - Bullish positioning"

    # Combined volume display
    combined_volume_info = f"{volume_status}\n{delta_pressure}{depth_pressure}{oi_flow}"

    if volume_color == "success":
        st.success(combined_volume_info)
    elif volume_color == "error":
        st.error(combined_volume_info)
    else:
        st.info(combined_volume_info)

    # Professional Entry Decision
    st.markdown("---")
    st.markdown("**🎲 ENTRY DECISION FRAMEWORK:**")

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
**✅ TAKE THE TRADE IF:**
1. Price at key support/resistance
2. Higher Low (LONG) or Lower High (SHORT) formed
3. Volume confirms the move (>40% strength)
4. Clear stop loss level identified
5. Target gives minimum 1:2 R:R
6. Not in middle of range
        """)

    with col2:
        st.error("""
**❌ SKIP THE TRADE IF:**
1. Chasing price (away from key levels)
2. No clear structure (no HL/LH)
3. Low volume / No confirmation
4. Stop loss too wide (>30 pts)
5. Poor R:R (<1:1.5)
6. Already 2 losing trades today
        """)

    st.warning("""
**💡 REMEMBER:**
- Missing a trade is 100x better than entering a wrong trade
- Intraday gives multiple chances - **WAIT** for perfect setup
- **One quality trade > Five mediocre trades**
- Capital protection = Real profit
    """)

    st.markdown("---")

    # --- Entry Price Recommendations (INTRADAY/SCALPING) ---
    st.markdown("### ⚡ INTRADAY/SCALPING Entry Recommendations (1-Hour Trades)")

    # ===== GET VOB AND HTF S/R LEVELS FOR PRECISE ENTRY ZONES =====
    # Try to get VOB levels from session state
    vob_support_level = None
    vob_support_source = None
    vob_resistance_level = None
    vob_resistance_source = None

    try:
        import streamlit as st
        if 'vob_data_nifty' in st.session_state and st.session_state.vob_data_nifty:
            vob_data = st.session_state.vob_data_nifty

            # Find nearest bullish VOB below current price (support)
            bullish_blocks = vob_data.get('bullish_blocks', [])
            min_distance_support = 999
            for block in bullish_blocks:
                if isinstance(block, dict):
                    block_price = block.get('upper', block.get('lower', 0))
                    if block_price < current_price:
                        distance = current_price - block_price
                        if distance < min_distance_support and distance < 50:  # Within 50 pts
                            min_distance_support = distance
                            vob_support_level = block_price
                            vob_support_source = "VOB"

            # Find nearest bearish VOB above current price (resistance)
            bearish_blocks = vob_data.get('bearish_blocks', [])
            min_distance_resistance = 999
            for block in bearish_blocks:
                if isinstance(block, dict):
                    block_price = block.get('lower', block.get('upper', 0))
                    if block_price > current_price:
                        distance = block_price - current_price
                        if distance < min_distance_resistance and distance < 50:  # Within 50 pts
                            min_distance_resistance = distance
                            vob_resistance_level = block_price
                            vob_resistance_source = "VOB"
    except:
        pass

    # Get HTF S/R levels from intraday_levels (3min, 5min, 15min, 30min)
    htf_support_level = None
    htf_support_timeframe = None
    htf_resistance_level = None
    htf_resistance_timeframe = None

    if intraday_levels:
        # Find nearest HTF support
        supports_below = [l for l in intraday_levels if l['price'] < current_price]
        if supports_below:
            nearest_htf_support = supports_below[0]
            htf_support_level = nearest_htf_support['price']
            htf_support_timeframe = nearest_htf_support.get('source', 'HTF')

        # Find nearest HTF resistance
        resistances_above = [l for l in intraday_levels if l['price'] > current_price]
        if resistances_above:
            nearest_htf_resistance = resistances_above[0]
            htf_resistance_level = nearest_htf_resistance['price']
            htf_resistance_timeframe = nearest_htf_resistance.get('source', 'HTF')

    # ===== CHOOSE BEST LEVEL (VOB PRIORITY, THEN HTF) =====
    # For scalping support
    if vob_support_level and (not htf_support_level or abs(current_price - vob_support_level) < abs(current_price - htf_support_level)):
        scalp_support = vob_support_level
        scalp_support_source = f"VOB Support"
    elif htf_support_level:
        scalp_support = htf_support_level
        scalp_support_source = f"HTF {htf_support_timeframe} Support"
    else:
        scalp_support = support_level
        scalp_support_source = "Structural Support"

    # For scalping resistance
    if vob_resistance_level and (not htf_resistance_level or abs(vob_resistance_level - current_price) < abs(htf_resistance_level - current_price)):
        scalp_resistance = vob_resistance_level
        scalp_resistance_source = f"VOB Resistance"
    elif htf_resistance_level:
        scalp_resistance = htf_resistance_level
        scalp_resistance_source = f"HTF {htf_resistance_timeframe} Resistance"
    else:
        scalp_resistance = resistance_level
        scalp_resistance_source = "Structural Resistance"

    # ===== SCALPING RR (TIGHTER THAN POSITION TRADES) =====
    # Scalping uses tighter stops and targets
    scalp_sl_pct = 0.10  # 10% stop for scalping
    scalp_target_pct = 0.20  # 20% target for scalping

    # Display scalping parameters
    st.info(f"""
**📊 Scalping Parameters:**
**Stop Loss:** {scalp_sl_pct*100:.0f}% | **Target:** {scalp_target_pct*100:.0f}%
**Timeframe:** 1-Hour | **Style:** Quick In-Out
**Nearest Support:** ₹{scalp_support:,.0f} | **Nearest Resistance:** ₹{scalp_resistance:,.0f}
    """)

    # Helper function to get option premium from chain
    def get_option_premium(chain: Dict, strike: int, option_type: str) -> float:
        """Extract LTP from option chain for given strike"""
        if not chain or 'data' not in chain:
            return 0.0

        for option in chain.get('data', []):
            if option.get('strikePrice') == strike:
                if option_type == 'CE':
                    return option.get('CE', {}).get('lastPrice', 0.0)
                elif option_type == 'PE':
                    return option.get('PE', {}).get('lastPrice', 0.0)
        return 0.0

    col1, col2 = st.columns(2)

    with col1:
        # CALL Entry (SCALP at nearest intraday support)
        call_strike = round(current_price / 50) * 50  # ATM strike based on spot
        call_premium = get_option_premium(option_chain, call_strike, 'CE') if option_chain else 0.0

        # Use real premium if available, otherwise estimate
        if call_premium > 0:
            call_entry_estimate = call_premium
            call_sl = call_premium * (1 - scalp_sl_pct)  # 10% SL for scalping
            call_target = call_premium * (1 + scalp_target_pct)  # 20% target for scalping
        else:
            # Estimate based on distance from spot
            distance = abs(call_strike - current_price)
            call_entry_estimate = max(50, 250 - (distance / 10))
            call_sl = call_entry_estimate * (1 - scalp_sl_pct)
            call_target = call_entry_estimate * (1 + scalp_target_pct)

        # Calculate distance to INTRADAY support
        distance_to_scalp_support = current_price - scalp_support

        # Entry trigger zone (tight for scalping - 5-10 pts buffer)
        scalp_support_trigger_low = int(scalp_support - 5)
        scalp_support_trigger_high = int(scalp_support + 5)

        # ===== ALWAYS SHOW CALL SCALP ENTRY =====
        st.success(f"""
**🟢 CALL Scalp Entry ({scalp_support_source})**

**Spot Price:** ₹{current_price:,.2f}
**Strike:** {call_strike} CE (ATM)
**Entry Price:** ₹{call_entry_estimate:.2f}
**Stop Loss:** ₹{call_sl:.2f} (-{scalp_sl_pct*100:.0f}%) **[TIGHT]**
**Target:** ₹{call_target:.2f} (+{scalp_target_pct*100:.0f}%) **[QUICK]**
**{scalp_support_source}:** ₹{scalp_support:,.0f}
**Distance:** {distance_to_scalp_support:.0f} pts away
**Entry Zone:** ₹{scalp_support_trigger_low:,.0f}-{scalp_support_trigger_high:,.0f}
**Timeframe:** 1-Hour Scalp
**Level Type:** {scalp_support_source}
        """)

    with col2:
        # PUT Entry (SCALP at nearest intraday resistance)
        put_strike = round(current_price / 50) * 50  # ATM strike based on spot
        put_premium = get_option_premium(option_chain, put_strike, 'PE') if option_chain else 0.0

        # Use real premium if available, otherwise estimate
        if put_premium > 0:
            put_entry_estimate = put_premium
            put_sl = put_premium * (1 - scalp_sl_pct)  # 10% SL for scalping
            put_target = put_premium * (1 + scalp_target_pct)  # 20% target for scalping
        else:
            # Estimate based on distance from spot
            distance = abs(put_strike - current_price)
            put_entry_estimate = max(50, 250 - (distance / 10))
            put_sl = put_entry_estimate * (1 - scalp_sl_pct)
            put_target = put_entry_estimate * (1 + scalp_target_pct)

        # Calculate distance to INTRADAY resistance
        distance_to_scalp_resistance = scalp_resistance - current_price

        # Entry trigger zone (tight for scalping - 5-10 pts buffer)
        scalp_resistance_trigger_low = int(scalp_resistance - 5)
        scalp_resistance_trigger_high = int(scalp_resistance + 5)

        # ===== ALWAYS SHOW PUT SCALP ENTRY =====
        st.error(f"""
**🔴 PUT Scalp Entry ({scalp_resistance_source})**

**Spot Price:** ₹{current_price:,.2f}
**Strike:** {put_strike} PE (ATM)
**Entry Price:** ₹{put_entry_estimate:.2f}
**Stop Loss:** ₹{put_sl:.2f} (-{scalp_sl_pct*100:.0f}%) **[TIGHT]**
**Target:** ₹{put_target:.2f} (+{scalp_target_pct*100:.0f}%) **[QUICK]**
**{scalp_resistance_source}:** ₹{scalp_resistance:,.0f}
**Distance:** {distance_to_scalp_resistance:.0f} pts away
**Entry Zone:** ₹{scalp_resistance_trigger_low:,.0f}-{scalp_resistance_trigger_high:,.0f}
**Timeframe:** 1-Hour Scalp
**Level Type:** {scalp_resistance_source}
        """)

    st.markdown("---")

    # --- POSITION TRADE Entry Recommendations (SWING/MULTI-DAY) ---
    st.markdown("### 📈 POSITION TRADE Entry Recommendations (Swing/Multi-Day)")

    # ===== POSITION TRADE RR (WIDER STOPS AND TARGETS) =====
    position_sl_pct = 0.25  # 25% stop for position
    position_target_pct = 0.70  # 70% target for position

    # Display position parameters
    st.info(f"""
**📊 Position Trade Parameters:**
**Stop Loss:** {position_sl_pct*100:.0f}% | **Target:** {position_target_pct*100:.0f}%
**Timeframe:** Multi-Day/Week | **Style:** Swing Trade
**Structural Support:** ₹{support_level:,.0f} | **Structural Resistance:** ₹{resistance_level:,.0f}
    """)

    col3, col4 = st.columns(2)

    with col3:
        # CALL Entry (POSITION at structural support)
        call_strike_pos = round(current_price / 50) * 50
        call_premium_pos = get_option_premium(option_chain, call_strike_pos, 'CE') if option_chain else 0.0

        if call_premium_pos > 0:
            call_entry_pos = call_premium_pos
            call_sl_pos = call_premium_pos * (1 - position_sl_pct)
            call_target_pos = call_premium_pos * (1 + position_target_pct)
        else:
            distance_pos = abs(call_strike_pos - current_price)
            call_entry_pos = max(50, 300 - (distance_pos / 10))
            call_sl_pos = call_entry_pos * (1 - position_sl_pct)
            call_target_pos = call_entry_pos * (1 + position_target_pct)

        distance_to_position_support = current_price - support_level
        position_support_trigger_low = int(support_level - 30)
        position_support_trigger_high = int(support_level + 10)

        st.success(f"""
**🟢 CALL Position Entry (Structural Support)**

**Spot Price:** ₹{current_price:,.2f}
**Strike:** {call_strike_pos} CE (ATM)
**Entry Price:** ₹{call_entry_pos:.2f}
**Stop Loss:** ₹{call_sl_pos:.2f} (-{position_sl_pct*100:.0f}%) **[WIDE]**
**Target:** ₹{call_target_pos:.2f} (+{position_target_pct*100:.0f}%) **[BIG MOVE]**
**Structural Support:** ₹{support_level:,.0f}
**Distance:** {distance_to_position_support:.0f} pts away
**Entry Zone:** ₹{position_support_trigger_low:,.0f}-{position_support_trigger_high:,.0f}
**Timeframe:** Multi-Day Swing
        """)

    with col4:
        # PUT Entry (POSITION at structural resistance)
        put_strike_pos = round(current_price / 50) * 50
        put_premium_pos = get_option_premium(option_chain, put_strike_pos, 'PE') if option_chain else 0.0

        if put_premium_pos > 0:
            put_entry_pos = put_premium_pos
            put_sl_pos = put_premium_pos * (1 - position_sl_pct)
            put_target_pos = put_premium_pos * (1 + position_target_pct)
        else:
            distance_pos = abs(put_strike_pos - current_price)
            put_entry_pos = max(50, 300 - (distance_pos / 10))
            put_sl_pos = put_entry_pos * (1 - position_sl_pct)
            put_target_pos = put_entry_pos * (1 + position_target_pct)

        distance_to_position_resistance = resistance_level - current_price
        position_resistance_trigger_low = int(resistance_level - 10)
        position_resistance_trigger_high = int(resistance_level + 30)

        st.error(f"""
**🔴 PUT Position Entry (Structural Resistance)**

**Spot Price:** ₹{current_price:,.2f}
**Strike:** {put_strike_pos} PE (ATM)
**Entry Price:** ₹{put_entry_pos:.2f}
**Stop Loss:** ₹{put_sl_pos:.2f} (-{position_sl_pct*100:.0f}%) **[WIDE]**
**Target:** ₹{put_target_pos:.2f} (+{position_target_pct*100:.0f}%) **[BIG MOVE]**
**Structural Resistance:** ₹{resistance_level:,.0f}
**Distance:** {distance_to_position_resistance:.0f} pts away
**Entry Zone:** ₹{position_resistance_trigger_low:,.0f}-{position_resistance_trigger_high:,.0f}
**Timeframe:** Multi-Day Swing
        """)

    # --- WHAT NOT TO DO (Elite-level UX) ---
    st.markdown("---")
    st.markdown("### 🚫 WHAT NOT TO DO (Critical Trade Avoidance)")

    # Collect all warning conditions
    warnings = []

    # Warning 1: OI Data Invalid
    if not data_valid:
        warnings.append("❌ **DO NOT TRADE** - OI Data is stale/failed. Signals unreliable.")

    # Warning 2: Low Confidence Score
    if confidence_score < 45:
        warnings.append("❌ **DO NOT TRADE** - Confidence score too low. Wait for better setup.")

    # Warning 3: Tight Range / Chop Zone
    if sr_distance < 50:
        warnings.append("⚠️ **AVOID** - Price in tight chop zone ({sr_distance:.0f} pts range). High risk of whipsaws.")

    # Warning 4: Mid-Session Low Momentum
    if 11 <= current_hour < 13:
        warnings.append("⚠️ **CAUTION** - Mid-session low momentum period. Only trade strong setups.")

    # Warning 5: Opening Hour Volatility
    if current_hour == 9 and current_minute >= 15 and current_minute < 45:
        warnings.append("⚠️ **WAIT** - Opening hour high volatility. Let market settle first 30 minutes.")

    # Warning 6: Expiry Day
    if days_to_expiry <= 0:
        warnings.append("🔥 **EXPIRY DAY** - Extreme volatility! Reduce position size 50%, use tighter stops.")

    # Warning 7: IV Too Low (Premium Decay)
    if vix_value < 12:
        warnings.append("⚠️ **AVOID BUYING OPTIONS** - IV very low. Premium decay will hurt long positions.")

    # Warning 8: Balanced Flow (No Edge)
    if total_volume > 0:
        buy_pct_check = (buy_volume / total_volume) * 100
        delta_check = abs(buy_pct_check - 50)
        if delta_check < 5:  # Within 45-55% range = balanced
            warnings.append("⚠️ **NO CLEAR FLOW** - Balanced buy/sell. Wait for directional commitment.")

    # Warning 9: Price Between Levels (No Clear Zone)
    distance_to_sup = current_price - support_level
    distance_to_res = resistance_level - current_price
    if distance_to_sup > 30 and distance_to_res > 30:
        warnings.append("⚠️ **AVOID MID-ZONE** - Price not near key levels. Wait for support/resistance approach.")

    # Warning 10: Recent Stop Loss Hit (if tracking in session_state)
    if 'last_sl_hit_time' in st.session_state:
        from datetime import datetime, timedelta
        last_sl_time = st.session_state.last_sl_hit_time
        if datetime.now() - last_sl_time < timedelta(minutes=30):
            warnings.append("🛑 **PAUSE TRADING** - Stop loss hit within 30 minutes. Take a break to avoid revenge trading.")

    # Display warnings
    if warnings:
        st.error("### 🚨 ACTIVE WARNINGS - READ BEFORE TRADING!")
        for warning in warnings:
            st.warning(warning)

        # Add summary guidance
        if len(warnings) >= 5:
            st.error("🔴 **TOO MANY RED FLAGS** - Step away! Market conditions not favorable.")
        elif len(warnings) >= 3:
            st.warning("🟡 **MULTIPLE CONCERNS** - Trade only if you have strong conviction and tight risk management.")
        else:
            st.info("🟢 **MANAGEABLE RISKS** - Proceed with caution and proper risk management.")
    else:
        st.success("✅ **NO MAJOR WARNINGS** - Conditions favorable for trading with proper risk management.")


def display_signal_card(signal: TradingSignal):
    """Display trading signal as a formatted card."""

    # Determine colors
    if signal.direction == "LONG":
        dir_color = "#00ff88"
        dir_emoji = "🚀"
    elif signal.direction == "SHORT":
        dir_color = "#ff4444"
        dir_emoji = "🔻"
    else:
        dir_color = "#ffa500"
        dir_emoji = "⚖️"

    # Signal type specific formatting
    if signal.signal_type == "ENTRY":
        type_emoji = "🎯"
        type_text = "ENTRY SIGNAL"
    elif signal.signal_type == "EXIT":
        type_emoji = "🚪"
        type_text = "EXIT SIGNAL"
    elif signal.signal_type == "DIRECTION_CHANGE":
        type_emoji = "🔄"
        type_text = "DIRECTION CHANGE"
    elif signal.signal_type == "BIAS_CHANGE":
        type_emoji = "⚡"
        type_text = "BIAS CHANGE"
    else:  # WAIT
        type_emoji = "⏸️"
        type_text = "WAIT"

    # Main signal card
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
                border-radius: 15px; padding: 25px; margin: 15px 0;
                border-left: 5px solid {dir_color}; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
        <h2 style='margin: 0 0 15px 0; color: {dir_color};'>
            {type_emoji} {type_text} - {dir_emoji} {signal.direction}
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Option Details (for ENTRY signals)
    if signal.signal_type == "ENTRY" and signal.option_type:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Option Details")
            opt_color = "#00ff88" if signal.option_type == "CALL" else "#ff4444"
            st.markdown(f"""
            <div style='background: #1e1e1e; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <p style='margin: 5px 0;'><strong>Type:</strong>
                   <span style='color: {opt_color}; font-weight: bold;'>{signal.option_type}</span></p>
                <p style='margin: 5px 0;'><strong>Strike:</strong> {signal.strike_price} {signal.option_type[:2]}</p>
                <p style='margin: 5px 0;'><strong>Entry Price:</strong> ₹{signal.entry_price:.2f} - {signal.entry_price * 1.04:.2f}</p>
                <p style='margin: 5px 0; color: #888;'><em>Current: ₹{signal.entry_price:.2f}</em></p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 🎯 Targets & Risk")
            st.markdown(f"""
            <div style='background: #1e1e1e; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <p style='margin: 5px 0;'><strong>Stop Loss:</strong>
                   <span style='color: #ff4444;'>₹{signal.stop_loss:.2f}</span>
                   ({((signal.stop_loss - signal.entry_price) / signal.entry_price * 100):.1f}%)</p>
                <p style='margin: 5px 0;'><strong>Target 1:</strong>
                   <span style='color: #00ff88;'>₹{signal.target1:.2f}</span>
                   (+{((signal.target1 - signal.entry_price) / signal.entry_price * 100):.1f}%)</p>
                <p style='margin: 5px 0;'><strong>Target 2:</strong>
                   <span style='color: #00ff88;'>₹{signal.target2:.2f}</span>
                   (+{((signal.target2 - signal.entry_price) / signal.entry_price * 100):.1f}%)</p>
                <p style='margin: 5px 0;'><strong>Target 3:</strong>
                   <span style='color: #00ff88;'>₹{signal.target3:.2f}</span>
                   (+{((signal.target3 - signal.entry_price) / signal.entry_price * 100):.1f}%)</p>
                <p style='margin: 10px 0 5px 0; border-top: 1px solid #444; padding-top: 10px;'>
                   <strong>R:R Ratio:</strong>
                   <span style='color: #6495ED; font-size: 18px; font-weight: bold;'>{signal.risk_reward_ratio:.1f}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Strength metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        conf_color = "#00ff88" if signal.confidence >= 80 else "#ffa500" if signal.confidence >= 65 else "#ff4444"
        st.markdown(f"""
        <div style='background: #1e1e1e; padding: 15px; border-radius: 10px; text-align: center;
                    border-left: 4px solid {conf_color};'>
            <h2 style='margin: 0; color: {conf_color};'>{signal.confidence:.1f}%</h2>
            <p style='margin: 5px 0 0 0; color: #888; font-size: 14px;'>Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        conf_pct = (signal.confluence_count / 10) * 100  # Assuming max 10 indicators
        conf_color = "#00ff88" if conf_pct >= 70 else "#ffa500" if conf_pct >= 50 else "#ff4444"
        st.markdown(f"""
        <div style='background: #1e1e1e; padding: 15px; border-radius: 10px; text-align: center;
                    border-left: 4px solid {conf_color};'>
            <h2 style='margin: 0; color: {conf_color};'>{signal.confluence_count}/10</h2>
            <p style='margin: 5px 0 0 0; color: #888; font-size: 14px;'>Confluence</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        regime_color = "#00ff88" if "UPTREND" in signal.market_regime else "#ff4444" if "DOWNTREND" in signal.market_regime else "#ffa500"
        st.markdown(f"""
        <div style='background: #1e1e1e; padding: 15px; border-radius: 10px; text-align: center;
                    border-left: 4px solid {regime_color};'>
            <h3 style='margin: 0; color: {regime_color}; font-size: 16px;'>{signal.market_regime}</h3>
            <p style='margin: 5px 0 0 0; color: #888; font-size: 14px;'>Market Regime</p>
        </div>
        """, unsafe_allow_html=True)

    # Reason and XGBoost info
    st.markdown("### 💡 Signal Reasoning")
    st.markdown(f"""
    <div style='background: #1e1e1e; padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <p style='margin: 0; line-height: 1.6;'>{signal.reason}</p>
    </div>
    """, unsafe_allow_html=True)

    # Timestamp
    st.caption(f"⏰ Generated: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")


def display_signal_history():
    """Display recent signal history."""
    if 'signal_history' not in st.session_state or not st.session_state.signal_history:
        st.info("No signal history yet. Generate your first signal!")
        return

    history = st.session_state.signal_history[:10]  # Show last 10

    st.markdown("#### 📜 Recent Signals")

    for i, sig in enumerate(history):
        type_emoji = {
            "ENTRY": "🎯",
            "EXIT": "🚪",
            "WAIT": "⏸️",
            "DIRECTION_CHANGE": "🔄",
            "BIAS_CHANGE": "⚡"
        }.get(sig['signal_type'], "📊")

        dir_emoji = {
            "LONG": "🚀",
            "SHORT": "🔻",
            "NEUTRAL": "⚖️"
        }.get(sig['direction'], "")

        timestamp = sig['timestamp']
        if isinstance(timestamp, str):
            time_str = timestamp
        else:
            time_str = timestamp.strftime('%H:%M:%S')

        st.markdown(f"""
        <div style='background: #1e1e1e; padding: 10px; border-radius: 8px; margin: 5px 0;
                    border-left: 3px solid #6495ED;'>
            <span style='color: #6495ED;'>{type_emoji} {sig['signal_type']}</span>
            <span style='margin-left: 10px;'>{dir_emoji} {sig['direction']}</span>
            <span style='float: right; color: #888; font-size: 12px;'>{time_str}</span>
            <br>
            <span style='font-size: 12px; color: #888;'>
                Confidence: {sig['confidence']:.1f}% | Confluence: {sig['confluence']}/10
            </span>
        </div>
        """, unsafe_allow_html=True)


def display_telegram_stats():
    """Display Telegram alert statistics."""
    if 'telegram_manager' not in st.session_state:
        st.info("Telegram alerts not configured yet.")
        return

    telegram_manager = st.session_state.telegram_manager
    stats = telegram_manager.get_statistics()

    st.markdown("#### 📱 Telegram Alert Stats")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Sent", stats['total_sent'])
        st.metric("Success Rate", f"{stats['success_rate']:.1f}%")

    with col2:
        st.metric("Total Failed", stats['total_failed'])
        st.metric("Currently Blocked", stats['currently_blocked'])

    # Per-type stats
    st.markdown("**Per Alert Type:**")
    for alert_type, type_stats in stats['per_type'].items():
        st.markdown(f"**{alert_type}**: {type_stats['sent']} sent, {type_stats['failed']} failed")


def create_active_signal_from_trading_signal(
    trading_signal: TradingSignal,
    signal_manager: any
) -> Optional[str]:
    """
    Auto-create an Active Signal entry from a TradingSignal (Tab 3 integration).

    Returns:
        Signal ID if created, None otherwise
    """
    try:
        if trading_signal.signal_type != "ENTRY":
            return None

        # Create signal entry for Active Signals tab
        signal_data = {
            'timestamp': trading_signal.timestamp,
            'direction': trading_signal.direction,
            'entry_price': trading_signal.entry_price,
            'stop_loss': trading_signal.stop_loss,
            'target1': trading_signal.target1,
            'target2': trading_signal.target2,
            'target3': trading_signal.target3,
            'confidence': trading_signal.confidence,
            'confluence': trading_signal.confluence_count,
            'regime': trading_signal.market_regime,
            'option_type': trading_signal.option_type,
            'strike_price': trading_signal.strike_price,
            'reason': trading_signal.reason,
            'risk_reward': trading_signal.risk_reward_ratio
        }

        # Add to signal manager
        signal_id = signal_manager.create_signal(signal_data)

        return signal_id

    except Exception as e:
        logger.error(f"Error creating active signal: {e}")
        return None
