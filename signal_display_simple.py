"""
SIMPLIFIED AI Trading Signal Display
Shows ONLY what's needed - clean and actionable
UPDATED: Now uses NATIVE Streamlit components (NO HTML)
"""

import streamlit as st
from typing import Optional, Dict


def display_simple_assessment(
    nifty_screener_data: Optional[Dict],
    enhanced_market_data: Optional[Dict],
    ml_regime_result: Optional[any],
    current_price: float,
    atm_strike: int,
    option_chain: Optional[Dict] = None,
    money_flow_signals: Optional[Dict] = None,
    deltaflow_signals: Optional[Dict] = None
):
    """
    SIMPLE AI TRADING SIGNAL - Only essentials
    Using NATIVE Streamlit Components (NO HTML)

    Shows:
    1. State (TRADE/WAIT/SCAN)
    2. Direction (LONG/SHORT/NEUTRAL)
    3. Confidence
    4. Primary Setup
    5. Entry Zone
    6. Stop Loss
    7. Target
    8. Reason
    """

    # Store current time for Telegram alerts
    from datetime import datetime
    from config import IST
    st.session_state.current_time_ist = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')

    # === EXTRACT ESSENTIAL DATA ===

    # Get ATM Bias
    atm_bias_data = nifty_screener_data.get('atm_bias', {}) if nifty_screener_data else {}
    atm_bias_score = atm_bias_data.get('total_score', 0)
    atm_bias_verdict = atm_bias_data.get('verdict', 'NEUTRAL')

    # Get Regime
    regime = "RANGING"
    if ml_regime_result and hasattr(ml_regime_result, 'regime'):
        regime = ml_regime_result.regime
    elif ml_regime_result and isinstance(ml_regime_result, dict):
        regime = ml_regime_result.get('regime', 'RANGING')

    # Get OI/PCR
    oi_pcr_data = nifty_screener_data.get('oi_pcr_metrics', {}) if nifty_screener_data else {}
    pcr = oi_pcr_data.get('pcr', 1.0)

    # === GET SUPPORT & RESISTANCE FROM MULTIPLE SOURCES ===
    # Priority: ML Analysis > VOB > HTF S/R > Calculated

    support = current_price - 100  # Fallback
    resistance = current_price + 100  # Fallback
    support_type = "Calculated"
    resistance_type = "Calculated"

    # Try to get from ML regime result first (uses 165+ features)
    if ml_regime_result:
        if hasattr(ml_regime_result, 'support_level') and ml_regime_result.support_level:
            support = ml_regime_result.support_level
            support_type = "ML Analysis (165+ features)"
        if hasattr(ml_regime_result, 'resistance_level') and ml_regime_result.resistance_level:
            resistance = ml_regime_result.resistance_level
            resistance_type = "ML Analysis (165+ features)"

        # Also check if it's a dict
        if isinstance(ml_regime_result, dict):
            if 'support_level' in ml_regime_result and ml_regime_result['support_level']:
                support = ml_regime_result['support_level']
                support_type = "ML Analysis"
            if 'resistance_level' in ml_regime_result and ml_regime_result['resistance_level']:
                resistance = ml_regime_result['resistance_level']
                resistance_type = "ML Analysis"

    # Get from nifty_screener_data
    if nifty_screener_data:
        # Priority 1: Try HTF S/R levels (from bias analysis results)
        if 'bias_analysis_results' in st.session_state:
            bias_results = st.session_state['bias_analysis_results']

            # Look for HTF support/resistance
            if 'nearest_support' in bias_results and bias_results['nearest_support']:
                nearest_sup = bias_results['nearest_support']
                if isinstance(nearest_sup, dict):
                    support = nearest_sup.get('price', support)
                    support_type = f"HTF {nearest_sup.get('type', 'S/R')} ({nearest_sup.get('timeframe', '1H')})"
                elif isinstance(nearest_sup, (int, float)):
                    support = nearest_sup
                    support_type = "HTF Support"

            if 'nearest_resistance' in bias_results and bias_results['nearest_resistance']:
                nearest_res = bias_results['nearest_resistance']
                if isinstance(nearest_res, dict):
                    resistance = nearest_res.get('price', resistance)
                    resistance_type = f"HTF {nearest_res.get('type', 'S/R')} ({nearest_res.get('timeframe', '1H')})"
                elif isinstance(nearest_res, (int, float)):
                    resistance = nearest_res
                    resistance_type = "HTF Resistance"

        # Priority 2: VOB levels (Volume Order Blocks)
        vob_levels = nifty_screener_data.get('vob_signals', [])
        if vob_levels and support_type == "Calculated":  # Only if not found from HTF
            support_levels = [v for v in vob_levels if v.get('price', 0) < current_price]
            resistance_levels = [v for v in vob_levels if v.get('price', 0) > current_price]

            if support_levels:
                # Get nearest VOB support
                nearest_support = max(support_levels, key=lambda x: x.get('price', 0))
                support = nearest_support.get('price', support)
                vob_strength = nearest_support.get('strength', 'Medium')
                support_type = f"VOB Support ({vob_strength})"

            if resistance_levels:
                # Get nearest VOB resistance
                nearest_resistance = min(resistance_levels, key=lambda x: x.get('price', 0))
                resistance = nearest_resistance.get('price', resistance)
                vob_strength = nearest_resistance.get('strength', 'Medium')
                resistance_type = f"VOB Resistance ({vob_strength})"

        # Priority 3: Fallback to simple nearest_support/resistance
        if support_type == "Calculated":
            if 'nearest_support' in nifty_screener_data:
                support = nifty_screener_data['nearest_support']
                support_type = "Key Support"
            if 'nearest_resistance' in nifty_screener_data:
                resistance = nifty_screener_data['nearest_resistance']
                resistance_type = "Key Resistance"

    # === GET ZONE WIDTH FROM COMPREHENSIVE S/R ANALYSIS ===
    # Use 10-factor analysis with GEX, Market Depth, Volume, Delta Flow, OI, etc.
    zone_width = resistance - support  # Default
    zone_quality = "NARROW" if zone_width < 75 else "MEDIUM" if zone_width < 150 else "WIDE"

    # Try to use comprehensive S/R analysis for precise zone widths
    try:
        from src.comprehensive_sr_analysis import analyze_sr_strength_comprehensive

        # Build features dict from available data
        features = {
            'close': current_price,
            'support_level': support,
            'resistance_level': resistance,
            'pcr': pcr,
            'vix': 15.0,  # Will update below
            'atm_bias_score': atm_bias_score,
        }

        # Add regime features if available
        if ml_regime_result:
            if hasattr(ml_regime_result, 'regime_confidence'):
                features['regime_confidence'] = ml_regime_result.regime_confidence
            if hasattr(ml_regime_result, 'trend_strength'):
                features['trend_strength'] = ml_regime_result.trend_strength
            elif isinstance(ml_regime_result, dict):
                features['regime_confidence'] = ml_regime_result.get('regime_confidence', 50.0)
                features['trend_strength'] = ml_regime_result.get('trend_strength', 0.0)

        # Add nifty_screener_data features
        if nifty_screener_data:
            # GEX features
            if 'gamma_exposure' in nifty_screener_data:
                gex_data = nifty_screener_data['gamma_exposure']
                features['gamma_squeeze_probability'] = gex_data.get('squeeze_probability', 0.0)

            # Market depth features
            if 'market_depth' in nifty_screener_data:
                depth_data = nifty_screener_data['market_depth']
                features['market_depth_order_imbalance'] = depth_data.get('order_imbalance', 0.0)

            # OI buildup
            if 'oi_buildup_pattern' in nifty_screener_data:
                features['oi_buildup_pattern'] = nifty_screener_data['oi_buildup_pattern']

        # Run comprehensive S/R analysis
        sr_analysis = analyze_sr_strength_comprehensive(features)

        # Get precise zone widths from analysis
        if 'zone_width_support' in sr_analysis:
            support_zone_width = sr_analysis['zone_width_support']
            resistance_zone_width = sr_analysis['zone_width_resistance']

            # Use average zone width
            zone_width = (support_zone_width + resistance_zone_width) / 2
            zone_quality = "NARROW" if zone_width < 20 else "MEDIUM" if zone_width < 35 else "WIDE"

    except Exception as e:
        # Fallback to simple calculation
        pass

    # === GET VOB MAJOR/MINOR ===
    vob_major_support = None
    vob_major_resistance = None
    vob_minor_support = None
    vob_minor_resistance = None

    if nifty_screener_data and 'vob_signals' in nifty_screener_data:
        vob_levels = nifty_screener_data['vob_signals']
        for vob in vob_levels:
            vob_price = vob.get('price', 0)
            vob_strength = vob.get('strength', 'Medium')

            if vob_price < current_price:
                # Support
                if vob_strength == "Major" and vob_major_support is None:
                    vob_major_support = vob_price
                elif vob_strength == "Minor" and vob_minor_support is None:
                    vob_minor_support = vob_price
            else:
                # Resistance
                if vob_strength == "Major" and vob_major_resistance is None:
                    vob_major_resistance = vob_price
                elif vob_strength == "Minor" and vob_minor_resistance is None:
                    vob_minor_resistance = vob_price

    # === GET EXPIRY SPIKE ===
    expiry_spike = "✅ Normal"
    expiry_days = 1
    if nifty_screener_data and 'expiry_spike_data' in nifty_screener_data:
        expiry_data = nifty_screener_data['expiry_spike_data']
        expiry_days = expiry_data.get('days_to_expiry', 1)
        if expiry_days == 0:
            expiry_spike = "⚠️ EXPIRY TODAY"
        elif expiry_days <= 0.5:
            expiry_spike = "⚠️ HIGH VOLATILITY"
        elif expiry_days <= 1:
            expiry_spike = "Elevated"

    # === GET GEX (Gamma Exposure) ===
    gex_level = "Neutral"
    max_gamma_strike = atm_strike
    if nifty_screener_data and 'gamma_exposure' in nifty_screener_data:
        gex_data = nifty_screener_data['gamma_exposure']
        max_gamma_strike = gex_data.get('max_gamma_strike', atm_strike)
        gex_level = gex_data.get('level', 'Neutral')

    # === GET MARKET BIAS ===
    market_bias = atm_bias_verdict  # From ATM Bias

    # Get VIX
    vix = 15.0
    if enhanced_market_data and 'vix' in enhanced_market_data:
        vix_data = enhanced_market_data['vix']
        if isinstance(vix_data, dict):
            vix = vix_data.get('current', 15.0)
        else:
            vix = vix_data

    # === CALCULATE CONFIDENCE ===

    confidence = 50  # Base

    # ATM Bias contribution (±15)
    if abs(atm_bias_score) > 0.5:
        confidence += 15
    elif abs(atm_bias_score) > 0.2:
        confidence += 10

    # Regime contribution (±15)
    if "TRENDING" in regime:
        confidence += 15
    elif "RANGING" in regime:
        confidence += 5

    # PCR contribution (±10)
    if pcr < 0.7 or pcr > 1.3:
        confidence += 10
    elif pcr < 0.85 or pcr > 1.15:
        confidence += 5

    # VIX contribution (±10)
    if vix < 12:
        confidence += 10  # Low volatility = high confidence
    elif vix < 15:
        confidence += 5
    elif vix > 20:
        confidence -= 10  # High volatility = low confidence

    # Cap at 90
    confidence = min(confidence, 90)

    # === DETERMINE DIRECTION ===

    bullish_signals = 0
    bearish_signals = 0

    # ATM Bias
    if "PUT" in atm_bias_verdict:
        bullish_signals += 1
    elif "CALL" in atm_bias_verdict:
        bearish_signals += 1

    # PCR
    if pcr > 1.15:
        bullish_signals += 1
    elif pcr < 0.85:
        bearish_signals += 1

    # Regime
    if "UP" in regime or "BULLISH" in regime:
        bullish_signals += 1
    elif "DOWN" in regime or "BEARISH" in regime:
        bearish_signals += 1

    # Money Flow
    if money_flow_signals:
        flow_signal = money_flow_signals.get('signal', 'NEUTRAL')
        if "BULLISH" in flow_signal or "BUY" in flow_signal:
            bullish_signals += 1
        elif "BEARISH" in flow_signal or "SELL" in flow_signal:
            bearish_signals += 1

    # Determine direction
    if bullish_signals > bearish_signals + 1:
        direction = "LONG"
        dir_emoji = "🚀"
    elif bearish_signals > bullish_signals + 1:
        direction = "SHORT"
        dir_emoji = "🔻"
    else:
        direction = "NEUTRAL"
        dir_emoji = "⚖️"

    # === DETERMINE STATE ===

    if confidence < 60:
        state = "WAIT"
        state_emoji = "🔴"
    elif confidence < 75:
        state = "SCAN"
        state_emoji = "🟡"
    else:
        state = "TRADE"
        state_emoji = "🟢"

    # Distance check - if too far from support/resistance, WAIT
    dist_to_support = abs(current_price - support)
    dist_to_resistance = abs(current_price - resistance)

    if dist_to_support > 50 and dist_to_resistance > 50:
        state = "WAIT"
        state_emoji = "🔴"

    # === DETERMINE SETUP ===

    if direction == "LONG":
        entry_zone = f"₹{support - 10:,.0f} - ₹{support + 10:,.0f}"
        stop_loss = f"₹{support - 30:,.0f}"
        target = f"₹{resistance:,.0f}"
        setup_type = "Support Bounce"
    elif direction == "SHORT":
        entry_zone = f"₹{resistance - 10:,.0f} - ₹{resistance + 10:,.0f}"
        stop_loss = f"₹{resistance + 30:,.0f}"
        target = f"₹{support:,.0f}"
        setup_type = "Resistance Rejection"
    else:
        entry_zone = "Wait for clear direction"
        stop_loss = "N/A"
        target = "N/A"
        setup_type = "No Setup"

    # === GENERATE REASON ===

    if state == "WAIT":
        reason = "Low confidence or price in mid-zone. Wait for better setup."
    elif state == "SCAN":
        reason = f"Moderate setup. {direction} bias detected but not strong enough for full position."
    else:
        reason = f"Strong {direction} setup with {confidence}% confidence. {setup_type} active."

    # === DISPLAY USING NATIVE STREAMLIT COMPONENTS ===

    # HEADER
    bias_emoji_map = {
        "BULLISH": "🟢",
        "BEARISH": "🔴",
        "NEUTRAL": "⚖️",
        "PUT SELLERS DOMINANT": "🐂",
        "CALL SELLERS DOMINANT": "🐻"
    }
    bias_display_emoji = bias_emoji_map.get(market_bias, "⚖️")

    st.markdown(f"## {state_emoji} **{state}** - {dir_emoji} {direction}")
    st.markdown("---")

    # CORE METRICS (4 columns)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Confidence",
            value=f"{confidence}%"
        )

    with col2:
        st.metric(
            label="XGBoost Regime",
            value=regime
        )

    with col3:
        st.metric(
            label="Market Bias",
            value=f"{bias_display_emoji} {market_bias}"
        )

    with col4:
        zone_indicator = "🟢" if zone_quality == "NARROW" else ("🟡" if zone_quality == "MEDIUM" else "🔴")
        st.metric(
            label="Zone Width",
            value=f"{zone_indicator} {zone_width:.0f} pts",
            delta=zone_quality
        )

    st.markdown("---")

    # SUPPORT & RESISTANCE (2 columns)
    st.markdown("### 📊 Support & Resistance")

    col_sup, col_res = st.columns(2)

    with col_sup:
        st.markdown("#### 🟢 SUPPORT")
        st.markdown(f"### ₹{support:,.0f}")
        st.caption(f"**Source:** {support_type}")
        st.caption(f"**Distance:** {abs(current_price - support):.0f} points away")

    with col_res:
        st.markdown("#### 🔴 RESISTANCE")
        st.markdown(f"### ₹{resistance:,.0f}")
        st.caption(f"**Source:** {resistance_type}")
        st.caption(f"**Distance:** {abs(resistance - current_price):.0f} points away")

    st.markdown("---")

    # VOB LEVELS
    st.markdown("### 📊 VOB LEVELS")

    vob_col1, vob_col2 = st.columns(2)

    with vob_col1:
        major_sup = f"₹{vob_major_support:,.0f}" if vob_major_support else "N/A"
        minor_sup = f"₹{vob_minor_support:,.0f}" if vob_minor_support else "N/A"
        st.markdown(f"**Major Support:** {major_sup}")
        st.markdown(f"**Minor Support:** {minor_sup}")

    with vob_col2:
        major_res = f"₹{vob_major_resistance:,.0f}" if vob_major_resistance else "N/A"
        minor_res = f"₹{vob_minor_resistance:,.0f}" if vob_minor_resistance else "N/A"
        st.markdown(f"**Major Resistance:** {major_res}")
        st.markdown(f"**Minor Resistance:** {minor_res}")

    st.markdown("---")

    # ENTRY SETUP
    st.markdown("### 🎯 ENTRY SETUP")

    # Choose container based on state and direction
    if state == "TRADE":
        if direction == "LONG":
            container = st.success
        elif direction == "SHORT":
            container = st.error
        else:
            container = st.info
    elif state == "SCAN":
        container = st.warning
    else:  # WAIT
        container = st.warning

    with container("Setup Details"):
        st.markdown(f"**Setup Type:** {setup_type}")
        st.markdown(f"**Entry Zone:** {entry_zone}")
        st.markdown(f"**Stop Loss:** {stop_loss}")
        st.markdown(f"**Target:** {target}")

    st.markdown("---")

    # ADDITIONAL INFO (3 columns)
    st.markdown("### 📈 Additional Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.metric(
            label="Expiry Status",
            value=expiry_spike
        )

    with info_col2:
        st.metric(
            label="GEX Level",
            value=gex_level
        )

    with info_col3:
        st.metric(
            label="VIX",
            value=f"{vix:.1f}"
        )

    st.markdown("---")

    # ANALYSIS REASON
    st.markdown("### 💡 Analysis")
    st.info(reason)

    st.markdown("---")

    # FOOTER
    footer_text = f"**Current:** ₹{current_price:,.2f} | **PCR:** {pcr:.2f} | **Bullish:** {bullish_signals} | **Bearish:** {bearish_signals}"
    st.caption(footer_text)

    # TRADE RECOMMENDATION
    if state == "WAIT":
        st.error("### 🔴 NO TRADE")
        st.warning("""
        **Wait for better setup**
        - Price in mid-zone or confidence too low
        - Be patient - missing a trade is better than a bad trade
        """)
    elif state == "SCAN":
        st.warning("### 🟡 SCAN MODE")
        st.info("""
        **Moderate Setup - Monitor Closely**
        - Some signals aligned but not all
        - Consider smaller position size
        - Wait for full confirmation
        """)
    else:  # TRADE
        # Check if price in entry zone
        in_entry_zone = False
        if direction == "LONG":
            if abs(current_price - support) <= 20:
                in_entry_zone = True
        elif direction == "SHORT":
            if abs(current_price - resistance) <= 20:
                in_entry_zone = True

        if in_entry_zone and direction != "NEUTRAL":
            st.success("### 🎯 PRICE IN ENTRY ZONE")
            st.balloons()

    # === TELEGRAM ALERT - When price in zone + signals align ===

    # Check if price is in entry zone
    in_entry_zone = False
    if direction == "LONG":
        # Check if price near support (within entry zone)
        if abs(current_price - support) <= 20:  # Within 20 points of support
            in_entry_zone = True
    elif direction == "SHORT":
        # Check if price near resistance (within entry zone)
        if abs(current_price - resistance) <= 20:  # Within 20 points of resistance
            in_entry_zone = True

    # Send Telegram if:
    # 1. State = TRADE (high confidence)
    # 2. Direction != NEUTRAL
    # 3. Price in entry zone
    # 4. Not sent recently (prevent spam)

    send_telegram = False
    if state == "TRADE" and direction != "NEUTRAL" and in_entry_zone:
        # Check if we sent recently
        last_telegram_time = st.session_state.get('last_telegram_signal_time', None)
        current_time = datetime.now(IST)

        if last_telegram_time is None:
            send_telegram = True
        else:
            # Only send if 15+ minutes since last signal
            time_diff = (current_time - last_telegram_time).total_seconds() / 60
            if time_diff >= 15:
                send_telegram = True

    if send_telegram:
        try:
            from telegram_integration import send_telegram_message

            telegram_message = f"""
🎯 **TRADING SIGNAL ALERT** 🎯

**State:** {state_emoji} {state}
**Direction:** {dir_emoji} {direction}
**Confidence:** {confidence}%

**Setup:** {setup_type}
**Entry Zone:** {entry_zone}
**Stop Loss:** {stop_loss}
**Target:** {target}

**Market Data:**
- Current Price: ₹{current_price:,.2f}
- Support: ₹{support:,.0f} ({support_type})
- Resistance: ₹{resistance:,.0f} ({resistance_type})
- PCR: {pcr:.2f}
- VIX: {vix:.1f}

**Analysis:** {reason}

*Time: {st.session_state.current_time_ist}*
            """

            send_telegram_message(telegram_message)
            st.session_state.last_telegram_signal_time = current_time
            st.toast(f"✅ Telegram alert sent: {direction} signal at {current_time.strftime('%H:%M:%S')}")

        except Exception as e:
            st.warning(f"⚠️ Telegram notification failed: {e}")
