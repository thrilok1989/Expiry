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
    option_chain: Optional[Dict] = None
):
    """
    Display FINAL ASSESSMENT with Market Makers narrative.

    Format includes:
    - Seller Activity + ATM Bias + Moment + Expiry + OI/PCR
    - Market Makers interpretation
    - ATM Zone Analysis
    - Game plan
    - Key levels and entry prices
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
        # Flow Confirmation (Delta Imbalance, Candle Strength)
        st.markdown("**📊 Flow Confirmation (Who's in Control?)**")

        # Try to get volume data and delta imbalance from market_depth or enhanced_market_data
        buy_volume = 0
        sell_volume = 0
        total_volume = 0

        if market_depth and isinstance(market_depth, dict):
            buy_volume = market_depth.get('total_buy_qty', 0)
            sell_volume = market_depth.get('total_sell_qty', 0)
            total_volume = buy_volume + sell_volume

        # Calculate delta imbalance
        if total_volume > 0:
            buy_pct = (buy_volume / total_volume) * 100
            sell_pct = (sell_volume / total_volume) * 100
            delta_imbalance = buy_pct - sell_pct

            if delta_imbalance > 15:
                flow_verdict = "🟢 BUYERS IN CONTROL"
                flow_strength = "STRONG BUY FLOW"
                flow_advice = "Bullish bias confirmed. Look for CALL entries."
                flow_color = "success"
            elif delta_imbalance > 5:
                flow_verdict = "🟢 Mild Buy Pressure"
                flow_strength = "WEAK BUY FLOW"
                flow_advice = "Slight bullish edge. Wait for confirmation."
                flow_color = "info"
            elif delta_imbalance < -15:
                flow_verdict = "🔴 SELLERS IN CONTROL"
                flow_strength = "STRONG SELL FLOW"
                flow_advice = "Bearish bias confirmed. Look for PUT entries."
                flow_color = "error"
            elif delta_imbalance < -5:
                flow_verdict = "🔴 Mild Sell Pressure"
                flow_strength = "WEAK SELL FLOW"
                flow_advice = "Slight bearish edge. Wait for confirmation."
                flow_color = "warning"
            else:
                flow_verdict = "⚖️ BALANCED FLOW"
                flow_strength = "NEUTRAL"
                flow_advice = "No clear bias. Wait for directional move."
                flow_color = "info"

            if flow_color == "success":
                st.success(f"{flow_verdict}\n\nDelta: {delta_imbalance:+.1f}%  \nBuy: {buy_pct:.1f}% | Sell: {sell_pct:.1f}%  \n💡 {flow_advice}")
            elif flow_color == "error":
                st.error(f"{flow_verdict}\n\nDelta: {delta_imbalance:+.1f}%  \nBuy: {buy_pct:.1f}% | Sell: {sell_pct:.1f}%  \n💡 {flow_advice}")
            elif flow_color == "warning":
                st.warning(f"{flow_verdict}\n\nDelta: {delta_imbalance:+.1f}%  \nBuy: {buy_pct:.1f}% | Sell: {sell_pct:.1f}%  \n💡 {flow_advice}")
            else:
                st.info(f"{flow_verdict}\n\nDelta: {delta_imbalance:+.1f}%  \nBuy: {buy_pct:.1f}% | Sell: {sell_pct:.1f}%  \n💡 {flow_advice}")
        else:
            st.info("⚖️ FLOW DATA UNAVAILABLE\n\nCannot determine current flow. Use other indicators.")

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

    # --- Entry Price Recommendations ---
    st.markdown("### 🎯 Entry Price Recommendations")

    # ===== ENTRY PERMISSION LOGIC (CRITICAL) =====
    # Check multiple conditions before allowing entries

    # Condition 1: Confidence threshold
    confidence_ok = confidence_score >= 60

    # Condition 2: Distance to levels (not in mid-zone)
    distance_to_sup = current_price - support_level
    distance_to_res = resistance_level - current_price
    in_mid_zone = distance_to_sup > 50 and distance_to_res > 50

    # Condition 3: Flow data available
    flow_available = total_volume > 0

    # Condition 4: Market session (not post-market)
    from datetime import datetime
    import pytz
    ist = pytz.timezone('Asia/Kolkata')
    current_time_check = datetime.now(ist)
    is_market_hours = 9 <= current_time_check.hour < 16

    # Condition 5: Check if near-spot levels exist (within 50 pts)
    has_nearby_support = distance_to_sup <= 50
    has_nearby_resistance = distance_to_res <= 50

    # Master permission flags
    allow_call_entry = confidence_ok and has_nearby_support and (flow_available or confidence_score >= 70) and is_market_hours
    allow_put_entry = confidence_ok and has_nearby_resistance and (flow_available or confidence_score >= 70) and is_market_hours

    # ===== REGIME-BASED RR (DYNAMIC) =====
    # Adjust SL/Target based on market regime
    if regime in ['TRENDING_UP', 'TRENDING_DOWN', 'STRONG_TRENDING_UP', 'STRONG_TRENDING_DOWN']:
        sl_pct = 0.30  # 30% for trending
        target_pct = 0.60  # 60% for trending
        rr_mode = "BREAKOUT MODE"
    elif regime in ['RANGING', 'CONSOLIDATING']:
        sl_pct = 0.18  # 18% for ranging
        target_pct = 0.30  # 30% for ranging
        rr_mode = "RANGE SCALP MODE"
    else:
        sl_pct = 0.25  # 25% default
        target_pct = 0.45  # 45% default
        rr_mode = "BALANCED MODE"

    # Display RR mode
    st.info(f"🎯 **Risk/Reward Mode:** {rr_mode} (SL: {sl_pct*100:.0f}% | Target: {target_pct*100:.0f}%)")

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
        # CALL Entry (at support) - using current_price not support for strike
        call_strike = round(current_price / 50) * 50  # ATM strike based on spot
        call_premium = get_option_premium(option_chain, call_strike, 'CE') if option_chain else 0.0

        # Use real premium if available, otherwise estimate
        if call_premium > 0:
            call_entry_estimate = call_premium
            call_sl = call_premium * (1 - sl_pct)  # Dynamic SL
            call_target = call_premium * (1 + target_pct)  # Dynamic Target
        else:
            # Estimate based on distance from spot
            distance = abs(call_strike - current_price)
            call_entry_estimate = max(50, 300 - (distance / 10))
            call_sl = call_entry_estimate * (1 - sl_pct)
            call_target = call_entry_estimate * (1 + target_pct)

        # Calculate INTELLIGENT trigger zones using market data
        # Use orderbook pressure and distance to support for dynamic buffer
        orderbook_pressure_val = market_depth.get('pressure', 0) if market_depth else 0
        distance_to_support = current_price - support_level

        # Dynamic buffer: closer support = tighter buffer, strong selling pressure = wider buffer
        base_buffer = min(50, distance_to_support * 0.15)  # 15% of distance, max 50 points
        pressure_adjustment = abs(orderbook_pressure_val) * 10 if orderbook_pressure_val < 0 else 0  # Add buffer if selling pressure

        support_trigger_low = int(support_level - (base_buffer + pressure_adjustment))
        support_trigger_high = int(support_level - (base_buffer * 0.5))

        # Calculate distance to entry
        call_entry_distance = distance_to_support

        # ===== ALWAYS SHOW CALL ENTRY =====
        st.success(f"""
**🟢 CALL Entry (Support)**

**Spot Price:** ₹{current_price:,.2f}
**Strike:** {call_strike} CE (ATM)
**Entry Price:** ₹{call_entry_estimate:.2f}
**Stop Loss:** ₹{call_sl:.2f} (-{sl_pct*100:.0f}%)
**Target:** ₹{call_target:.2f} (+{target_pct*100:.0f}%)
**Support Zone:** ₹{support_level:,.0f}
**Distance to Support:** {call_entry_distance:.0f} pts
**Trigger:** Price dips to ₹{support_trigger_low:,.0f}-{support_trigger_high:,.0f} and bounces back
        """)

    with col2:
        # PUT Entry (at resistance) - using current_price not resistance for strike
        put_strike = round(current_price / 50) * 50  # ATM strike based on spot
        put_premium = get_option_premium(option_chain, put_strike, 'PE') if option_chain else 0.0

        # Use real premium if available, otherwise estimate
        if put_premium > 0:
            put_entry_estimate = put_premium
            put_sl = put_premium * (1 - sl_pct)  # Dynamic SL
            put_target = put_premium * (1 + target_pct)  # Dynamic Target
        else:
            # Estimate based on distance from spot
            distance = abs(put_strike - current_price)
            put_entry_estimate = max(50, 300 - (distance / 10))
            put_sl = put_entry_estimate * (1 - sl_pct)
            put_target = put_entry_estimate * (1 + target_pct)

        # Calculate INTELLIGENT trigger zones using market data
        # Use orderbook pressure and distance to resistance for dynamic buffer
        distance_to_resistance_calc = resistance_level - current_price

        # Dynamic buffer: closer resistance = tighter buffer, strong buying pressure = wider buffer
        base_buffer_res = min(50, distance_to_resistance_calc * 0.15)  # 15% of distance, max 50 points
        pressure_adjustment_res = abs(orderbook_pressure_val) * 10 if orderbook_pressure_val > 0 else 0  # Add buffer if buying pressure

        resistance_trigger_low = int(resistance_level + (base_buffer_res * 0.5))
        resistance_trigger_high = int(resistance_level + (base_buffer_res + pressure_adjustment_res))

        # Calculate distance to entry
        put_entry_distance = distance_to_res

        # ===== ALWAYS SHOW PUT ENTRY =====
        st.error(f"""
**🔴 PUT Entry (Resistance)**

**Spot Price:** ₹{current_price:,.2f}
**Strike:** {put_strike} PE (ATM)
**Entry Price:** ₹{put_entry_estimate:.2f}
**Stop Loss:** ₹{put_sl:.2f} (-{sl_pct*100:.0f}%)
**Target:** ₹{put_target:.2f} (+{target_pct*100:.0f}%)
**Resistance Zone:** ₹{resistance_level:,.0f}
**Distance to Resistance:** {put_entry_distance:.0f} pts
**Trigger:** Price spikes to ₹{resistance_trigger_low:,.0f}-{resistance_trigger_high:,.0f} and rejects
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
