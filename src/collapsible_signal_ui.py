"""
Collapsible Trading Signal UI Component

Displays trading signals with:
1. Main summary view (always visible)
2. Expandable detailed sections (collapsed by default)
3. Clean, organized presentation
"""

import streamlit as st
from typing import Dict, Optional, List
from datetime import datetime


def display_collapsible_trading_signal(
    signal_data: Dict,
    nifty_screener_data: Optional[Dict] = None,
    enhanced_market_data: Optional[Dict] = None,
    ml_regime_result: Optional[any] = None,
    liquidity_result: Optional[any] = None,
    money_flow_signals: Optional[Dict] = None,
    deltaflow_signals: Optional[Dict] = None,
    cvd_result: Optional[any] = None,
    volatility_result: Optional[any] = None,
    oi_trap_result: Optional[any] = None,
    sr_trend_analysis: Optional[Dict] = None,
    sr_transitions: Optional[List] = None
):
    """
    Display trading signal with collapsible sections

    Args:
        signal_data: Main signal information
        nifty_screener_data: Data from option screener
        enhanced_market_data: VIX, sectors, etc.
        ml_regime_result: ML regime prediction
        liquidity_result: Liquidity analysis
        money_flow_signals: Money flow data
        deltaflow_signals: Delta flow data
        cvd_result: CVD analysis
        volatility_result: Volatility analysis
        oi_trap_result: OI trap detection
        sr_trend_analysis: S/R strength trend analysis
        sr_transitions: Recent S/R transitions
    """

    # ============================================================
    # MAIN SUMMARY - Always Visible
    # ============================================================
    st.markdown("---")
    st.markdown("## 📊 AI TRADING SIGNAL")

    # Extract main data
    confidence = signal_data.get('confidence', 0)
    direction = signal_data.get('direction', 'NEUTRAL')
    state = signal_data.get('state', 'WAIT')
    atm_bias = signal_data.get('atm_bias', 'NEUTRAL')
    regime = signal_data.get('regime', 'RANGING')

    # Main metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        state_emoji = {'TRADE': '🟢', 'WAIT': '🔴', 'SCAN': '🟡'}.get(state, '⚪')
        st.metric("State", f"{state_emoji} {state}")

    with col2:
        dir_emoji = {'LONG': '🚀', 'SHORT': '🔻', 'NEUTRAL': '⚖️'}.get(direction, '⚪')
        st.metric("Direction", f"{dir_emoji} {direction}")

    with col3:
        conf_color = "🟢" if confidence > 70 else "🟡" if confidence > 50 else "🔴"
        st.metric("Confidence", f"{conf_color} {confidence:.0f}%")

    with col4:
        st.metric("Regime", regime)

    # Key levels
    st.markdown("### 🎯 Key Levels")
    level_col1, level_col2 = st.columns(2)

    with level_col1:
        support = signal_data.get('support', 0)
        support_dist = signal_data.get('support_distance', 0)
        st.markdown(f"**Support:** ₹{support:,.0f} ({support_dist:+.0f} pts)")

    with level_col2:
        resistance = signal_data.get('resistance', 0)
        resistance_dist = signal_data.get('resistance_distance', 0)
        st.markdown(f"**Resistance:** ₹{resistance:,.0f} ({resistance_dist:+.0f} pts)")

    # Entry recommendation (if applicable)
    if state == 'TRADE':
        st.success(f"✅ **{direction} Entry**: {signal_data.get('entry_reason', 'Setup detected')}")

        entry_col1, entry_col2, entry_col3 = st.columns(3)
        with entry_col1:
            st.markdown(f"**Entry Zone:** ₹{signal_data.get('entry_low', 0):,.0f} - ₹{signal_data.get('entry_high', 0):,.0f}")
        with entry_col2:
            st.markdown(f"**Stop Loss:** ₹{signal_data.get('stop_loss', 0):,.0f}")
        with entry_col3:
            st.markdown(f"**Target:** ₹{signal_data.get('target', 0):,.0f}")

    elif state == 'WAIT':
        st.warning(f"⏸️ **WAIT**: {signal_data.get('wait_reason', 'No clear setup')}")

    # ============================================================
    # DETAILED SECTIONS - Collapsible
    # ============================================================

    # Section 1: ATM Bias & Market Makers Analysis
    with st.expander("📊 ATM Bias & Market Makers Analysis", expanded=False):
        if nifty_screener_data:
            _display_atm_bias_section(nifty_screener_data, atm_bias)
        else:
            st.info("No ATM bias data available")

    # Section 2: OI/PCR Analysis
    with st.expander("📈 OI/PCR Analysis", expanded=False):
        if nifty_screener_data:
            _display_oi_pcr_section(nifty_screener_data)
        else:
            st.info("No OI/PCR data available")

    # Section 3: Support/Resistance Strength Trends (NEW!)
    with st.expander("🔍 S/R Strength Trends & Transitions (ML Analysis)", expanded=False):
        if sr_trend_analysis or sr_transitions:
            _display_sr_trends_section(sr_trend_analysis, sr_transitions)
        else:
            st.info("No S/R trend data available. Data will accumulate over time.")

    # Section 4: Volume & Flow Analysis
    with st.expander("📊 Volume & Flow Analysis", expanded=False):
        _display_volume_flow_section(money_flow_signals, deltaflow_signals, cvd_result)

    # Section 5: Volatility & Risk
    with st.expander("⚡ Volatility & Risk Analysis", expanded=False):
        _display_volatility_section(enhanced_market_data, volatility_result, oi_trap_result)

    # Section 6: ML Regime & Liquidity
    with st.expander("🤖 ML Regime & Liquidity Analysis", expanded=False):
        _display_regime_liquidity_section(ml_regime_result, liquidity_result)

    # Section 7: Market Context
    with st.expander("🌍 Market Context (Sectors, VIX, Time)", expanded=False):
        _display_market_context_section(enhanced_market_data)

    # Section 8: Complete Entry Rules
    with st.expander("📋 Professional Entry Rules & Checklist", expanded=False):
        _display_entry_rules_section()

    st.markdown("---")


# ============================================================
# HELPER FUNCTIONS FOR EACH SECTION
# ============================================================

def _display_atm_bias_section(nifty_screener_data: Dict, atm_bias: str):
    """Display ATM Bias and Market Makers analysis"""
    atm_bias_data = nifty_screener_data.get('atm_bias', {})

    if atm_bias_data:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ATM Bias")
            bias_score = atm_bias_data.get('score', 0)
            bias_emoji = "🐂" if bias_score > 0.1 else "🐻" if bias_score < -0.1 else "⚖️"
            st.markdown(f"**{bias_emoji} {atm_bias}** (Score: {bias_score:.2f})")

        with col2:
            st.markdown("#### Market Makers Verdict")
            verdict = atm_bias_data.get('verdict', 'Unknown')
            st.markdown(f"**{verdict}**")

        # Detailed metrics
        st.markdown("#### Detailed Metrics")
        metrics_data = atm_bias_data.get('metrics', {})
        if metrics_data:
            for key, value in metrics_data.items():
                st.markdown(f"- **{key}**: {value}")
    else:
        st.info("ATM bias data not available")


def _display_oi_pcr_section(nifty_screener_data: Dict):
    """Display OI/PCR analysis"""
    oi_pcr_data = nifty_screener_data.get('oi_pcr_metrics', {})

    if oi_pcr_data:
        col1, col2, col3 = st.columns(3)

        with col1:
            pcr = oi_pcr_data.get('pcr_ratio', 0)
            pcr_color = "🟢" if pcr > 1.1 else "🔴" if pcr < 0.9 else "🟡"
            st.metric("PCR Ratio", f"{pcr_color} {pcr:.2f}")

        with col2:
            call_oi = oi_pcr_data.get('total_call_oi', 0)
            st.metric("Call OI", f"{call_oi:,.0f}")

        with col3:
            put_oi = oi_pcr_data.get('total_put_oi', 0)
            st.metric("PUT OI", f"{put_oi:,.0f}")

        # Max OI strikes
        st.markdown("#### Max OI Walls")
        wall_col1, wall_col2 = st.columns(2)

        with wall_col1:
            max_call_strike = oi_pcr_data.get('max_call_strike', 0)
            max_call_oi = oi_pcr_data.get('max_call_oi', 0)
            st.markdown(f"**CALL Wall:** ₹{max_call_strike:,} ({max_call_oi:,.0f} OI)")

        with wall_col2:
            max_put_strike = oi_pcr_data.get('max_put_strike', 0)
            max_put_oi = oi_pcr_data.get('max_put_oi', 0)
            st.markdown(f"**PUT Wall:** ₹{max_put_strike:,} ({max_put_oi:,.0f} OI)")
    else:
        st.info("OI/PCR data not available")


def _display_sr_trends_section(sr_trend_analysis: Optional[Dict], sr_transitions: Optional[List]):
    """Display S/R strength trends and transitions (NEW!)"""
    st.markdown("### 📊 Support/Resistance Strength Trends")

    if sr_trend_analysis:
        trends = sr_trend_analysis.get('trends', [])

        if trends:
            for trend in trends[:5]:  # Show top 5 trends
                level_type = trend.get('level_type', 'unknown')
                price = trend.get('price_level', 0)
                current_strength = trend.get('current_strength', 0)
                trend_dir = trend.get('trend', 'STABLE')
                confidence = trend.get('trend_confidence', 0)
                change_rate = trend.get('strength_change_rate', 0)
                pred_1h = trend.get('prediction_1h', 0)
                pred_4h = trend.get('prediction_4h', 0)

                # Trend emoji
                trend_emoji = {
                    'STRENGTHENING': '📈🟢',
                    'WEAKENING': '📉🔴',
                    'STABLE': '➡️🟡',
                    'TRANSITIONING': '🔄🟣'
                }.get(trend_dir, '⚪')

                st.markdown(f"#### {level_type.upper()} @ ₹{price:,.0f} {trend_emoji}")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Current Strength", f"{current_strength:.0f}%")

                with col2:
                    st.metric("Trend", trend_dir)

                with col3:
                    st.metric("Confidence", f"{confidence:.0f}%")

                with col4:
                    st.metric("Change Rate", f"{change_rate:+.1f}/hr")

                # Predictions
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.markdown(f"**Predicted 1h:** {pred_1h:.0f}%")
                with pred_col2:
                    st.markdown(f"**Predicted 4h:** {pred_4h:.0f}%")

                # Reasons
                reasons = trend.get('reasons', [])
                if reasons:
                    st.markdown("**Analysis:**")
                    for reason in reasons:
                        st.markdown(f"- {reason}")

                st.markdown("---")
        else:
            st.info("No trend data available yet. Accumulating observations...")

    # Display transitions
    if sr_transitions:
        st.markdown("### 🔄 Recent S/R Transitions")

        for transition in sr_transitions[:3]:  # Show last 3 transitions
            trans_type = transition.get('transition_type', '')
            price = transition.get('price_level', 0)
            confidence = transition.get('confidence', 0)
            reason = transition.get('reason', '')
            time_str = transition.get('end_time', datetime.now()).strftime('%H:%M')

            # Color based on transition type
            if 'SUPPORT_TO_RESISTANCE' in trans_type:
                st.error(f"🔴 **{trans_type}** @ ₹{price:,.0f} (Confidence: {confidence:.0f}%) - {time_str}")
            else:
                st.success(f"🟢 **{trans_type}** @ ₹{price:,.0f} (Confidence: {confidence:.0f}%) - {time_str}")

            st.markdown(f"_{reason}_")
            st.markdown("---")


def _display_volume_flow_section(money_flow: Optional[Dict], delta_flow: Optional[Dict], cvd: Optional[any]):
    """Display volume and flow analysis"""
    if money_flow:
        st.markdown("#### Money Flow")
        flow_bias = money_flow.get('bias', 'NEUTRAL')
        flow_strength = money_flow.get('strength', 0)
        st.markdown(f"**Bias:** {flow_bias} | **Strength:** {flow_strength:.0f}%")

    if delta_flow:
        st.markdown("#### Delta Flow")
        delta_bias = delta_flow.get('bias', 'NEUTRAL')
        delta_strength = delta_flow.get('strength', 0)
        st.markdown(f"**Bias:** {delta_bias} | **Strength:** {delta_strength:.0f}%")

    if cvd and hasattr(cvd, 'cvd_bias'):
        st.markdown("#### CVD Analysis")
        st.markdown(f"**CVD Bias:** {cvd.cvd_bias}")
        st.markdown(f"**Strength:** {cvd.strength if hasattr(cvd, 'strength') else 'N/A'}")

    if not money_flow and not delta_flow and not cvd:
        st.info("Volume/Flow data not available")


def _display_volatility_section(enhanced_data: Optional[Dict], vol_result: Optional[any], oi_trap: Optional[any]):
    """Display volatility and risk analysis"""
    if enhanced_data:
        vix_data = enhanced_data.get('vix', {})
        if vix_data.get('success'):
            vix = vix_data.get('data', {}).get('last_price', 0)
            vix_change = vix_data.get('data', {}).get('change_pct', 0)

            vix_color = "🟢" if vix < 15 else "🟡" if vix < 20 else "🔴"
            st.metric("VIX", f"{vix_color} {vix:.2f}", f"{vix_change:+.2f}%")

    if vol_result and hasattr(vol_result, 'regime'):
        st.markdown(f"**Volatility Regime:** {vol_result.regime}")

    if oi_trap and hasattr(oi_trap, 'trap_detected'):
        if oi_trap.trap_detected:
            st.warning(f"⚠️ **OI Trap Detected:** {oi_trap.trap_type if hasattr(oi_trap, 'trap_type') else 'Unknown'}")
        else:
            st.success("✅ No OI trap detected")

    if not enhanced_data and not vol_result and not oi_trap:
        st.info("Volatility data not available")


def _display_regime_liquidity_section(ml_regime: Optional[any], liquidity: Optional[any]):
    """Display ML regime and liquidity analysis"""
    if ml_regime:
        if hasattr(ml_regime, 'regime'):
            st.markdown(f"**ML Regime:** {ml_regime.regime}")
        if hasattr(ml_regime, 'confidence'):
            st.markdown(f"**Confidence:** {ml_regime.confidence:.0f}%")
        if hasattr(ml_regime, 'trend_strength'):
            st.markdown(f"**Trend Strength:** {ml_regime.trend_strength:.2f}")

    if liquidity and hasattr(liquidity, 'gravity_zones'):
        st.markdown("#### Liquidity Gravity Zones")
        zones = liquidity.gravity_zones if hasattr(liquidity, 'gravity_zones') else []
        for zone in zones[:3]:
            st.markdown(f"- ₹{zone.get('price', 0):,.0f} (Strength: {zone.get('strength', 0):.0f}%)")

    if not ml_regime and not liquidity:
        st.info("Regime/Liquidity data not available")


def _display_market_context_section(enhanced_data: Optional[Dict]):
    """Display market context (sectors, time, etc.)"""
    if enhanced_data:
        # Sector rotation
        sectors = enhanced_data.get('sectors', {})
        if sectors.get('success'):
            sector_data = sectors.get('data', [])
            bullish = sum(1 for s in sector_data if s.get('change_pct', 0) > 0.5)
            bearish = sum(1 for s in sector_data if s.get('change_pct', 0) < -0.5)

            st.markdown(f"**Sector Rotation:** {bullish} bullish, {bearish} bearish")

            # Top performers
            sorted_sectors = sorted(sector_data, key=lambda x: x.get('change_pct', 0), reverse=True)
            st.markdown("**Top 3 Sectors:**")
            for sector in sorted_sectors[:3]:
                st.markdown(f"- {sector.get('name', 'Unknown')}: {sector.get('change_pct', 0):+.2f}%")

        # Time context
        current_time = datetime.now()
        hour = current_time.hour

        if 9 <= hour < 11:
            session = "OPENING (High volatility)"
        elif 11 <= hour < 13:
            session = "MID-MORNING (Trend establishment)"
        elif 13 <= hour < 15:
            session = "AFTERNOON (Momentum continuation)"
        elif 15 <= hour < 16:
            session = "CLOSING (Volatility spike expected)"
        else:
            session = "AFTER HOURS"

        st.markdown(f"**Session:** {session}")
        st.markdown(f"**Time:** {current_time.strftime('%I:%M %p IST')}")
    else:
        st.info("Market context data not available")


def _display_entry_rules_section():
    """Display professional entry rules and checklist"""
    st.markdown("""
    ### ✅ ENTRY CHECKLIST - All Must Be TRUE Before Entry

    #### 1. Structure Confirmation
    - ✅ For LONG: Price forms Higher Low (HL) after decline
    - ✅ For SHORT: Price forms Lower High (LH) after rally
    - ❌ DON'T chase first green/red candle (often fake)

    #### 2. Volume Confirmation
    - ✅ Breakout candle has above-average volume
    - ✅ Volume increasing on move toward entry level
    - ❌ Low volume = Fake breakout, avoid entry

    #### 3. Position Confirmation
    - ✅ For LONG: Price above VWAP or breaks recent intraday high
    - ✅ For SHORT: Price below VWAP or breaks recent intraday low
    - ❌ Entry in middle of range = Poor R:R

    #### 4. Risk Management
    - ✅ Stop Loss below Higher Low (LONG) or above Lower High (SHORT)
    - ✅ Target minimum 1:2 Risk:Reward ratio
    - ✅ If 2 trades failed today → STOP trading, review tomorrow

    ### 🚫 SKIP THE TRADE IF:
    - Chasing price (away from key levels)
    - No clear structure (no HL/LH)
    - Low volume / No confirmation
    - Stop loss too wide (>30 pts)
    - Poor R:R (<1:1.5)
    - Already 2 losing trades today

    ### 💡 REMEMBER:
    - Missing a trade is 100x better than entering a wrong trade
    - Intraday gives multiple chances - WAIT for perfect setup
    - One quality trade > Five mediocre trades
    - Capital protection = Real profit
    """)
