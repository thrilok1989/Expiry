"""
SIMPLIFIED AI Trading Signal Display
Shows ONLY what's needed - clean and actionable
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

    # === GET SUPPORT & RESISTANCE ===
    support = current_price - 100
    resistance = current_price + 100
    support_type = "Calculated"
    resistance_type = "Calculated"

    if nifty_screener_data:
        # Try to get from VOB levels first
        vob_levels = nifty_screener_data.get('vob_signals', [])
        if vob_levels:
            support_levels = [v for v in vob_levels if v.get('price', 0) < current_price]
            resistance_levels = [v for v in vob_levels if v.get('price', 0) > current_price]

            if support_levels:
                nearest_support = max(support_levels, key=lambda x: x.get('price', 0))
                support = nearest_support.get('price', support)
                support_type = f"{nearest_support.get('type', 'VOB')} ({nearest_support.get('strength', 'Medium')})"

            if resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: x.get('price', 0))
                resistance = nearest_resistance.get('price', resistance)
                resistance_type = f"{nearest_resistance.get('type', 'VOB')} ({nearest_resistance.get('strength', 'Medium')})"

        # Fallback to nearest_support/resistance
        if 'nearest_support' in nifty_screener_data:
            support = nifty_screener_data['nearest_support']
            support_type = "Key Support"
        if 'nearest_resistance' in nifty_screener_data:
            resistance = nifty_screener_data['nearest_resistance']
            resistance_type = "Key Resistance"

    # === GET ZONE WIDTH ===
    zone_width = resistance - support
    zone_quality = "NARROW" if zone_width < 75 else "MEDIUM" if zone_width < 150 else "WIDE"

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
    expiry_spike = "Normal"
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
        dir_color = "#00ff88"
        dir_emoji = "🚀"
    elif bearish_signals > bullish_signals + 1:
        direction = "SHORT"
        dir_color = "#ff4444"
        dir_emoji = "🔻"
    else:
        direction = "NEUTRAL"
        dir_color = "#ffa500"
        dir_emoji = "⚖️"

    # === DETERMINE STATE ===

    if confidence < 60:
        state = "WAIT"
        state_color = "#ff4444"
        state_emoji = "🔴"
    elif confidence < 75:
        state = "SCAN"
        state_color = "#ffa500"
        state_emoji = "🟡"
    else:
        state = "TRADE"
        state_color = "#00ff88"
        state_emoji = "🟢"

    # Distance check - if too far from support/resistance, WAIT
    dist_to_support = abs(current_price - support)
    dist_to_resistance = abs(current_price - resistance)

    if dist_to_support > 50 and dist_to_resistance > 50:
        state = "WAIT"
        state_color = "#ff4444"
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

    # === DISPLAY ESSENTIAL DATA ===

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
                border-radius: 15px; padding: 30px; margin: 20px 0;
                border-left: 8px solid {state_color}; box-shadow: 0 8px 20px rgba(0,0,0,0.4);'>

        <h1 style='margin: 0 0 20px 0; color: {state_color}; font-size: 36px;'>
            {state_emoji} {state} - {dir_emoji} {direction}
        </h1>

        <!-- Core Metrics -->
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0;'>
            <div style='background: #0a0a0a; padding: 15px; border-radius: 10px; text-align: center;'>
                <div style='color: #6495ED; font-size: 32px; font-weight: bold;'>{confidence}%</div>
                <div style='color: #888; font-size: 12px;'>Confidence</div>
            </div>

            <div style='background: #0a0a0a; padding: 15px; border-radius: 10px; text-align: center;'>
                <div style='color: #ff9500; font-size: 20px; font-weight: bold;'>{regime}</div>
                <div style='color: #888; font-size: 12px;'>XGBoost Regime</div>
            </div>

            <div style='background: #0a0a0a; padding: 15px; border-radius: 10px; text-align: center;'>
                <div style='color: {"#00ff88" if "PUT" in market_bias else "#ff4444" if "CALL" in market_bias else "#ffa500"}; font-size: 18px; font-weight: bold;'>{market_bias}</div>
                <div style='color: #888; font-size: 12px;'>Market Bias</div>
            </div>

            <div style='background: #0a0a0a; padding: 15px; border-radius: 10px; text-align: center;'>
                <div style='color: {"#00ff88" if zone_quality == "NARROW" else "#ffa500" if zone_quality == "MEDIUM" else "#ff4444"}; font-size: 20px; font-weight: bold;'>{zone_width:.0f} pts</div>
                <div style='color: #888; font-size: 12px;'>Zone Width ({zone_quality})</div>
            </div>
        </div>

        <!-- Support & Resistance -->
        <div style='background: #0a0a0a; padding: 20px; border-radius: 10px; margin: 15px 0;'>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
                <div>
                    <div style='color: #00ff88; font-size: 16px; font-weight: bold; margin-bottom: 10px;'>🟢 SUPPORT</div>
                    <div style='color: #00ff88; font-size: 28px; font-weight: bold;'>₹{support:,.0f}</div>
                    <div style='color: #888; font-size: 13px; margin-top: 5px;'>{support_type}</div>
                    <div style='color: #666; font-size: 13px; margin-top: 5px;'>{abs(current_price - support):.0f} points away</div>
                </div>
                <div style='text-align: right;'>
                    <div style='color: #ff4444; font-size: 16px; font-weight: bold; margin-bottom: 10px;'>🔴 RESISTANCE</div>
                    <div style='color: #ff4444; font-size: 28px; font-weight: bold;'>₹{resistance:,.0f}</div>
                    <div style='color: #888; font-size: 13px; margin-top: 5px;'>{resistance_type}</div>
                    <div style='color: #666; font-size: 13px; margin-top: 5px;'>{abs(resistance - current_price):.0f} points away</div>
                </div>
            </div>
        </div>

        <!-- VOB Levels -->
        <div style='background: #0a0a0a; padding: 15px; border-radius: 10px; margin: 15px 0;'>
            <div style='color: #6495ED; font-size: 14px; font-weight: bold; margin-bottom: 10px;'>📊 VOB LEVELS</div>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 13px;'>
                <div style='color: #00ff88;'>Major Support: {f"₹{vob_major_support:,.0f}" if vob_major_support else "N/A"}</div>
                <div style='color: #ff4444; text-align: right;'>Major Resistance: {f"₹{vob_major_resistance:,.0f}" if vob_major_resistance else "N/A"}</div>
                <div style='color: #888;'>Minor Support: {f"₹{vob_minor_support:,.0f}" if vob_minor_support else "N/A"}</div>
                <div style='color: #888; text-align: right;'>Minor Resistance: {f"₹{vob_minor_resistance:,.0f}" if vob_minor_resistance else "N/A"}</div>
            </div>
        </div>

        <!-- Entry Setup -->
        <div style='background: #1a4d1a; padding: 20px; border-radius: 10px; margin: 15px 0; border: 2px solid {dir_color};'>
            <div style='color: #ffffff; font-size: 18px; font-weight: bold; margin-bottom: 15px;'>🎯 ENTRY SETUP</div>
            <div style='color: #ffffff; font-size: 15px; line-height: 2;'>
                <strong style='color: #6495ED;'>Setup Type:</strong> {setup_type}<br>
                <strong style='color: #00ff88;'>Entry Zone:</strong> {entry_zone}<br>
                <strong style='color: #ff4444;'>Stop Loss:</strong> {stop_loss}<br>
                <strong style='color: #00ff88;'>Target:</strong> {target}
            </div>
        </div>

        <!-- Additional Info -->
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 15px 0;'>
            <div style='background: #0a0a0a; padding: 10px; border-radius: 8px; text-align: center;'>
                <div style='color: #888; font-size: 11px;'>Expiry</div>
                <div style='color: {"#ff4444" if expiry_days <= 0.5 else "#ffa500" if expiry_days <= 1 else "#00ff88"}; font-size: 14px; font-weight: bold;'>{expiry_spike}</div>
            </div>
            <div style='background: #0a0a0a; padding: 10px; border-radius: 8px; text-align: center;'>
                <div style='color: #888; font-size: 11px;'>GEX Level</div>
                <div style='color: #ffa500; font-size: 14px; font-weight: bold;'>{gex_level}</div>
            </div>
            <div style='background: #0a0a0a; padding: 10px; border-radius: 8px; text-align: center;'>
                <div style='color: #888; font-size: 11px;'>VIX</div>
                <div style='color: {"#00ff88" if vix < 15 else "#ffa500" if vix < 20 else "#ff4444"}; font-size: 14px; font-weight: bold;'>{vix:.1f}</div>
            </div>
        </div>

        <!-- Reason -->
        <div style='background: #1a1a2e; padding: 15px; border-radius: 10px; margin: 15px 0;'>
            <div style='color: #cccccc; font-size: 14px; line-height: 1.6;'>
                💡 <strong>Analysis:</strong> {reason}
            </div>
        </div>

        <!-- Footer -->
        <div style='color: #666; font-size: 12px; margin-top: 15px; padding-top: 15px; border-top: 1px solid #333; text-align: center;'>
            Current: ₹{current_price:,.2f} | PCR: {pcr:.2f} | Bullish: {bullish_signals} | Bearish: {bearish_signals}
        </div>

    </div>
    """, unsafe_allow_html=True)

    # === ACTIONABLE GUIDANCE ===

    if state == "TRADE":
        st.success(f"""
        ✅ **TRADE SIGNAL ACTIVE**

        - Direction: {direction}
        - Entry: Wait for price to reach {entry_zone}
        - Stop: Exit if price breaks {stop_loss}
        - Target: Book profit at {target}
        - Position Size: Full size (confidence {confidence}%)
        """)
    elif state == "SCAN":
        st.warning(f"""
        ⚠️ **PAPER TRADE ONLY**

        - Direction: {direction} bias detected
        - Entry: {entry_zone}
        - Position Size: 50% or paper trade only
        - Confidence: {confidence}% (not high enough for full position)
        """)
    else:
        st.error("""
        🔴 **NO TRADE**

        - Wait for better setup
        - Price in mid-zone or confidence too low
        - Be patient - missing a trade is better than a bad trade
        """)
