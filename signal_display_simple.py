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
    # Priority: OI Walls > GEX > HTF S/R > VOB > ML Analysis > Calculated (±100)
    # OI and GEX are PRIMARY because they show institutional positioning

    support = current_price - 100  # Fallback (last resort)
    resistance = current_price + 100  # Fallback (last resort)
    support_type = "Calculated"
    resistance_type = "Calculated"

    # PRIORITY 1: OI Walls (Max PUT/CALL OI strikes) - Primary institutional levels
    if nifty_screener_data and 'oi_pcr_metrics' in nifty_screener_data:
        oi_metrics = nifty_screener_data['oi_pcr_metrics']
        # Max PUT OI = Support (where institutions defend)
        if 'max_pe_strike' in oi_metrics and oi_metrics['max_pe_strike']:
            max_pe = oi_metrics['max_pe_strike']
            if max_pe < current_price:
                support = max_pe
                support_type = "OI Wall (Max PUT OI)"
        # Max CALL OI = Resistance (where institutions defend)
        if 'max_ce_strike' in oi_metrics and oi_metrics['max_ce_strike']:
            max_ce = oi_metrics['max_ce_strike']
            if max_ce > current_price:
                resistance = max_ce
                resistance_type = "OI Wall (Max CALL OI)"

    # PRIORITY 2: GEX (Gamma Exposure) Walls - Market maker hedging levels
    if nifty_screener_data and support_type == "Calculated" and 'gamma_exposure' in nifty_screener_data:
        gex_data = nifty_screener_data['gamma_exposure']
        if 'gamma_walls' in gex_data and gex_data['gamma_walls']:
            for wall in gex_data['gamma_walls']:
                if isinstance(wall, dict):
                    wall_price = wall.get('strike', 0)
                    if wall_price < current_price and support_type == "Calculated":
                        support = wall_price
                        support_type = "GEX Wall (Gamma Support)"
                    elif wall_price > current_price and resistance_type == "Calculated":
                        resistance = wall_price
                        resistance_type = "GEX Wall (Gamma Resistance)"

    # PRIORITY 3: HTF S/R levels from session state (pivot-based technical levels)
    if support_type == "Calculated" and 'htf_nearest_support' in st.session_state and st.session_state.htf_nearest_support:
        htf_sup = st.session_state.htf_nearest_support
        if isinstance(htf_sup, dict):
            support = htf_sup.get('price', support)
            support_type = f"HTF {htf_sup.get('type', 'Support')} ({htf_sup.get('timeframe', '')})"
        elif isinstance(htf_sup, (int, float)):
            support = htf_sup
            support_type = "HTF Support"

    if resistance_type == "Calculated" and 'htf_nearest_resistance' in st.session_state and st.session_state.htf_nearest_resistance:
        htf_res = st.session_state.htf_nearest_resistance
        if isinstance(htf_res, dict):
            resistance = htf_res.get('price', resistance)
            resistance_type = f"HTF {htf_res.get('type', 'Resistance')} ({htf_res.get('timeframe', '')})"
        elif isinstance(htf_res, (int, float)):
            resistance = htf_res
            resistance_type = "HTF Resistance"

    # PRIORITY 4: Try to get from ML regime result (uses 165+ features)
    if ml_regime_result and support_type == "Calculated":  # Only if OI/GEX/HTF didn't provide
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

        # Priority 5: VOB levels (Volume Order Blocks)
        vob_levels = nifty_screener_data.get('vob_signals', [])
        if vob_levels and support_type == "Calculated":  # Only if not found from OI/GEX/HTF
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

        # Priority 6: Fallback to simple nearest_support/resistance
        if support_type == "Calculated":
            if 'nearest_support' in nifty_screener_data:
                support = nifty_screener_data['nearest_support']
                support_type = "Key Support"
            if 'nearest_resistance' in nifty_screener_data:
                resistance = nifty_screener_data['nearest_resistance']
                resistance_type = "Key Resistance"

    # === GET ZONE WIDTH FROM COMPREHENSIVE S/R ANALYSIS ===
    # Zone width = width of ZONE around the level (NOT distance between S/R)
    # Example: Support at 26,050 with zone_width=20 means entry zone is 26,040 to 26,060

    support_zone_width = 20.0  # Default: ±10 pts around support
    resistance_zone_width = 20.0  # Default: ±10 pts around resistance
    zone_quality_support = "MEDIUM"
    zone_quality_resistance = "MEDIUM"

    # Try to use comprehensive S/R analysis for precise zone widths
    try:
        from src.comprehensive_sr_analysis import analyze_sr_strength_comprehensive

        # Build features dict from available data
        features = {
            'close': current_price,
            'support_level': support,
            'resistance_level': resistance,
            'pcr': pcr,
            'vix': vix,  # REAL VIX from Enhanced Market Data
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

        # Get precise zone widths from analysis (separate for support and resistance)
        if 'zone_width_support' in sr_analysis:
            support_zone_width = sr_analysis['zone_width_support']
            zone_quality_support = "NARROW" if support_zone_width < 15 else "MEDIUM" if support_zone_width < 25 else "WIDE"

        if 'zone_width_resistance' in sr_analysis:
            resistance_zone_width = sr_analysis['zone_width_resistance']
            zone_quality_resistance = "NARROW" if resistance_zone_width < 15 else "MEDIUM" if resistance_zone_width < 25 else "WIDE"

    except Exception as e:
        # Fallback to default zone widths
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

    # Get VIX from Enhanced Market Data (REAL VIX, not hardcoded)
    vix = 15.0  # Fallback only
    if enhanced_market_data:
        # Try multiple locations for VIX
        if 'vix' in enhanced_market_data:
            vix_data = enhanced_market_data['vix']
            if isinstance(vix_data, dict):
                vix = vix_data.get('current', vix_data.get('value', 15.0))
            elif isinstance(vix_data, (int, float)):
                vix = vix_data
        elif 'india_vix' in enhanced_market_data:
            vix = enhanced_market_data['india_vix']
        elif 'VIX' in enhanced_market_data:
            vix = enhanced_market_data['VIX']

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

    # === DETERMINE SETUP WITH PRECISE ZONE WIDTHS ===

    if direction == "LONG":
        # Entry zone = support ± (zone_width / 2)
        entry_lower = support - (support_zone_width / 2)
        entry_upper = support + (support_zone_width / 2)
        entry_zone = f"₹{entry_lower:,.0f} - ₹{entry_upper:,.0f} (±{support_zone_width/2:.0f} pts)"

        # Stop loss below the support zone
        stop_loss = f"₹{entry_lower - 10:,.0f} (Below zone - 10 pts)"
        target = f"₹{resistance:,.0f}"
        setup_type = f"Support Bounce ({zone_quality_support} zone)"

    elif direction == "SHORT":
        # Entry zone = resistance ± (zone_width / 2)
        entry_lower = resistance - (resistance_zone_width / 2)
        entry_upper = resistance + (resistance_zone_width / 2)
        entry_zone = f"₹{entry_lower:,.0f} - ₹{entry_upper:,.0f} (±{resistance_zone_width/2:.0f} pts)"

        # Stop loss above the resistance zone
        stop_loss = f"₹{entry_upper + 10:,.0f} (Above zone + 10 pts)"
        target = f"₹{support:,.0f}"
        setup_type = f"Resistance Rejection ({zone_quality_resistance} zone)"

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
        # Show zone width based on direction
        if direction == "LONG":
            zone_width_display = support_zone_width
            zone_quality_display = zone_quality_support
            zone_label = "Support Zone Width"
        elif direction == "SHORT":
            zone_width_display = resistance_zone_width
            zone_quality_display = zone_quality_resistance
            zone_label = "Resistance Zone Width"
        else:
            # For NEUTRAL, show both
            zone_width_display = (support_zone_width + resistance_zone_width) / 2
            zone_quality_display = "MEDIUM"
            zone_label = "Avg Zone Width"

        zone_indicator = "🟢" if zone_quality_display == "NARROW" else ("🟡" if zone_quality_display == "MEDIUM" else "🔴")
        st.metric(
            label=zone_label,
            value=f"{zone_indicator} {zone_width_display:.0f} pts",
            delta=zone_quality_display
        )

    st.markdown("---")

    # === COMPREHENSIVE S/R LEVELS FROM ALL DATA SOURCES ===
    st.markdown("### 📊 Comprehensive Support & Resistance Levels")

    # Collect ALL S/R levels from multiple sources
    all_support_levels = []
    all_resistance_levels = []

    # 1. VOB Levels
    if nifty_screener_data and 'vob_signals' in nifty_screener_data:
        for vob in nifty_screener_data['vob_signals']:
            if isinstance(vob, dict):
                vob_price = vob.get('price', 0)
                vob_strength = vob.get('strength', 'Medium')
                if vob_price < current_price:
                    all_support_levels.append({
                        'price': vob_price,
                        'source': f"VOB ({vob_strength})",
                        'strength': 3 if vob_strength == 'Major' else 2
                    })
                elif vob_price > current_price:
                    all_resistance_levels.append({
                        'price': vob_price,
                        'source': f"VOB ({vob_strength})",
                        'strength': 3 if vob_strength == 'Major' else 2
                    })

    # 2. HTF S/R Levels
    if 'bias_analysis_results' in st.session_state:
        bias_res = st.session_state['bias_analysis_results']
        if 'nearest_support' in bias_res and bias_res['nearest_support']:
            sup_data = bias_res['nearest_support']
            if isinstance(sup_data, dict):
                all_support_levels.append({
                    'price': sup_data.get('price', 0),
                    'source': f"HTF {sup_data.get('type', 'S/R')}",
                    'strength': 3
                })
            elif isinstance(sup_data, (int, float)):
                all_support_levels.append({
                    'price': sup_data,
                    'source': "HTF Support",
                    'strength': 3
                })

        if 'nearest_resistance' in bias_res and bias_res['nearest_resistance']:
            res_data = bias_res['nearest_resistance']
            if isinstance(res_data, dict):
                all_resistance_levels.append({
                    'price': res_data.get('price', 0),
                    'source': f"HTF {res_data.get('type', 'S/R')}",
                    'strength': 3
                })
            elif isinstance(res_data, (int, float)):
                all_resistance_levels.append({
                    'price': res_data,
                    'source': "HTF Resistance",
                    'strength': 3
                })

    # 3. OI Strikes (Max OI levels)
    if nifty_screener_data:
        if 'oi_pcr_metrics' in nifty_screener_data:
            oi_metrics = nifty_screener_data['oi_pcr_metrics']
            if 'max_ce_strike' in oi_metrics and oi_metrics['max_ce_strike']:
                max_ce = oi_metrics['max_ce_strike']
                if max_ce > current_price:
                    all_resistance_levels.append({
                        'price': max_ce,
                        'source': "Max CALL OI",
                        'strength': 2
                    })
            if 'max_pe_strike' in oi_metrics and oi_metrics['max_pe_strike']:
                max_pe = oi_metrics['max_pe_strike']
                if max_pe < current_price:
                    all_support_levels.append({
                        'price': max_pe,
                        'source': "Max PUT OI",
                        'strength': 2
                    })

    # 4. GEX Walls (if available)
    if nifty_screener_data and 'gamma_exposure' in nifty_screener_data:
        gex = nifty_screener_data['gamma_exposure']
        if 'gamma_walls' in gex:
            for wall in gex['gamma_walls']:
                if isinstance(wall, dict):
                    wall_price = wall.get('strike', 0)
                    if wall_price < current_price:
                        all_support_levels.append({
                            'price': wall_price,
                            'source': "GEX Wall",
                            'strength': 2
                        })
                    elif wall_price > current_price:
                        all_resistance_levels.append({
                            'price': wall_price,
                            'source': "GEX Wall",
                            'strength': 2
                        })

    # Remove duplicates and sort
    def deduplicate_levels(levels):
        unique = {}
        for level in levels:
            price = level['price']
            if price not in unique or level['strength'] > unique[price]['strength']:
                unique[price] = level
        return sorted(unique.values(), key=lambda x: x['price'], reverse=True)

    all_support_levels = deduplicate_levels(all_support_levels)
    all_resistance_levels = deduplicate_levels(all_resistance_levels)

    # Find NEAREST and MAJOR levels (with safe fallbacks)
    if all_support_levels:
        # Filter valid levels (price must be a number)
        valid_supports = [x for x in all_support_levels if isinstance(x.get('price'), (int, float)) and x.get('price', 0) > 0]
        if valid_supports:
            nearest_support = min(valid_supports, key=lambda x: abs(x['price'] - current_price))
            major_support = max(valid_supports, key=lambda x: x.get('strength', 1))
        else:
            nearest_support = {'price': support, 'source': support_type, 'strength': 1}
            major_support = nearest_support
    else:
        nearest_support = {'price': support, 'source': support_type, 'strength': 1}
        major_support = nearest_support

    if all_resistance_levels:
        # Filter valid levels (price must be a number)
        valid_resistances = [x for x in all_resistance_levels if isinstance(x.get('price'), (int, float)) and x.get('price', 0) > 0]
        if valid_resistances:
            nearest_resistance = min(valid_resistances, key=lambda x: abs(x['price'] - current_price))
            major_resistance = max(valid_resistances, key=lambda x: x.get('strength', 1))
        else:
            nearest_resistance = {'price': resistance, 'source': resistance_type, 'strength': 1}
            major_resistance = nearest_resistance
    else:
        nearest_resistance = {'price': resistance, 'source': resistance_type, 'strength': 1}
        major_resistance = nearest_resistance

    # DISPLAY: 2 columns - Support | Resistance
    col_sup, col_res = st.columns(2)

    with col_sup:
        st.markdown("#### 🟢 SUPPORT LEVELS")

        # Nearest Support
        st.markdown(f"**📍 NEAREST:** ₹{nearest_support['price']:,.0f}")
        st.caption(f"Source: {nearest_support['source']}")
        st.caption(f"Distance: {abs(current_price - nearest_support['price']):.0f} pts away")

        # Zone
        sup_zone_lower = nearest_support['price'] - (support_zone_width / 2)
        sup_zone_upper = nearest_support['price'] + (support_zone_width / 2)
        zone_emoji = "🟢" if zone_quality_support == "NARROW" else ("🟡" if zone_quality_support == "MEDIUM" else "🔴")
        st.caption(f"Zone: {zone_emoji} ₹{sup_zone_lower:,.0f} - ₹{sup_zone_upper:,.0f}")

        st.markdown("---")

        # Major Support
        st.markdown(f"**💪 MAJOR:** ₹{major_support['price']:,.0f}")
        st.caption(f"Source: {major_support['source']}")
        st.caption(f"Distance: {abs(current_price - major_support['price']):.0f} pts away")

        # Show all supports
        if len(all_support_levels) > 2:
            with st.expander(f"📋 All {len(all_support_levels)} Support Levels"):
                for lvl in all_support_levels[:5]:  # Top 5
                    strength_emoji = "💪" if lvl['strength'] == 3 else "📊"
                    st.caption(f"{strength_emoji} ₹{lvl['price']:,.0f} - {lvl['source']}")

    with col_res:
        st.markdown("#### 🔴 RESISTANCE LEVELS")

        # Nearest Resistance
        st.markdown(f"**📍 NEAREST:** ₹{nearest_resistance['price']:,.0f}")
        st.caption(f"Source: {nearest_resistance['source']}")
        st.caption(f"Distance: {abs(nearest_resistance['price'] - current_price):.0f} pts away")

        # Zone
        res_zone_lower = nearest_resistance['price'] - (resistance_zone_width / 2)
        res_zone_upper = nearest_resistance['price'] + (resistance_zone_width / 2)
        zone_emoji = "🟢" if zone_quality_resistance == "NARROW" else ("🟡" if zone_quality_resistance == "MEDIUM" else "🔴")
        st.caption(f"Zone: {zone_emoji} ₹{res_zone_lower:,.0f} - ₹{res_zone_upper:,.0f}")

        st.markdown("---")

        # Major Resistance
        st.markdown(f"**💪 MAJOR:** ₹{major_resistance['price']:,.0f}")
        st.caption(f"Source: {major_resistance['source']}")
        st.caption(f"Distance: {abs(major_resistance['price'] - current_price):.0f} pts away")

        # Show all resistances
        if len(all_resistance_levels) > 2:
            with st.expander(f"📋 All {len(all_resistance_levels)} Resistance Levels"):
                for lvl in all_resistance_levels[:5]:  # Top 5
                    strength_emoji = "💪" if lvl['strength'] == 3 else "📊"
                    st.caption(f"{strength_emoji} ₹{lvl['price']:,.0f} - {lvl['source']}")

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
