"""
Advanced AI Modules Integration for Streamlit App
Add this to your app.py to display all the new AI modules
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Import all new AI modules
try:
    from src.master_ai_orchestrator import MasterAIOrchestrator, format_master_report
    from src.volatility_regime import format_regime_report
    from src.oi_trap_detection import format_oi_trap_report
    from src.cvd_delta_imbalance import format_cvd_report
    from src.institutional_retail_detector import format_participant_report
    from src.liquidity_gravity import format_liquidity_report
    from src.position_sizing_engine import format_position_size_report
    from src.risk_management_ai import format_risk_management_report
    from src.expectancy_model import format_expectancy_report
    from src.ml_market_regime import format_market_summary

    AI_MODULES_AVAILABLE = True
except ImportError as e:
    AI_MODULES_AVAILABLE = False
    st.warning(f"Advanced AI modules not available: {e}")


def render_master_ai_analysis_tab(df, option_chain, vix_current, vix_history, instrument="NIFTY", days_to_expiry=5):
    """
    Render the Master AI Analysis tab
    This is the MAIN tab showing complete AI analysis
    """
    st.header("🤖 MASTER AI ORCHESTRATOR")
    st.markdown("**Institutional-Grade Trading Intelligence**")

    if not AI_MODULES_AVAILABLE:
        st.error("Advanced AI modules not found. Please check installation.")
        return

    # Sidebar configuration
    with st.sidebar:
        st.subheader("⚙️ AI Configuration")
        account_size = st.number_input(
            "Account Size (₹)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000
        )
        max_risk = st.slider(
            "Max Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1
        )

    # Initialize orchestrator
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = MasterAIOrchestrator(
            account_size=account_size,
            max_risk_per_trade=max_risk
        )

    # Run Analysis Button
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        analyze_button = st.button(
            "🔍 RUN COMPLETE AI ANALYSIS",
            type="primary",
            use_container_width=True
        )

    with col2:
        auto_refresh = st.checkbox("Auto-refresh (1min)", value=False)

    with col3:
        if auto_refresh:
            st.info("🔄 Auto-refreshing...")

    # Auto-run if data is available or button clicked
    if analyze_button or auto_refresh or 'ai_result' not in st.session_state:

        with st.spinner("🤖 AI analyzing market conditions..."):
            try:
                # Prepare historical trades (if available)
                historical_trades = None
                if 'trade_history' in st.session_state and len(st.session_state.trade_history) > 0:
                    historical_trades = st.session_state.trade_history

                # Run complete analysis
                result = st.session_state.orchestrator.analyze_complete_market(
                    df=df,
                    option_chain=option_chain,
                    vix_current=vix_current,
                    vix_history=vix_history,
                    instrument=instrument,
                    days_to_expiry=days_to_expiry,
                    historical_trades=historical_trades
                )

                st.session_state.ai_result = result
                st.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    # Display results if available
    if 'ai_result' in st.session_state:
        result = st.session_state.ai_result

        # ========== VERDICT SECTION ==========
        st.markdown("---")

        # Color-code verdict
        verdict_color = {
            "STRONG BUY": "#00FF00",
            "BUY": "#90EE90",
            "HOLD": "#FFD700",
            "SELL": "#FFA500",
            "STRONG SELL": "#FF0000",
            "NO TRADE": "#808080"
        }

        color = verdict_color.get(result.final_verdict, "#FFFFFF")

        st.markdown(f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
            <h1 style="color: black; margin: 0;">🎯 {result.final_verdict}</h1>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Confidence",
            f"{result.confidence:.1f}%",
            delta=f"{result.confidence - 50:.1f}%" if result.confidence > 50 else None
        )
        col2.metric(
            "Trade Quality",
            f"{result.trade_quality_score:.1f}/100",
            delta="High" if result.trade_quality_score > 70 else "Low"
        )
        col3.metric(
            "Win Probability",
            f"{result.expected_win_probability:.1f}%"
        )
        col4.metric(
            "Risk/Reward",
            f"{result.risk_reward_ratio:.2f}:1"
        )

        # ========== REASONING ==========
        st.markdown("---")
        st.subheader("🧠 AI Reasoning Chain")

        for i, reason in enumerate(result.reasoning, 1):
            st.markdown(f"**{i}.** {reason}")

        # ========== TABS FOR DETAILED ANALYSIS ==========
        st.markdown("---")

        analysis_tabs = st.tabs([
            "📊 Market Summary",
            "⚡ Volatility & Risk",
            "🏦 Institutional Flow",
            "🧲 Liquidity Gravity",
            "💰 Position & Risk",
            "📈 Full Report"
        ])

        # Tab 1: Market Summary
        with analysis_tabs[0]:
            st.subheader("📊 Market Summary")

            summary = result.market_summary

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Overall Bias", summary.overall_bias, delta=f"{summary.bias_confidence:.0f}% confident")
                st.metric("Market Regime", summary.regime)
                st.metric("Volatility State", summary.volatility)
                st.metric("Trend Quality", summary.trend_quality)

            with col2:
                st.metric("Momentum", summary.momentum)
                st.metric("Risk Level", summary.risk_level)
                st.metric("Market Health", f"{summary.market_health_score:.0f}/100")
                st.metric("Conviction", f"{summary.conviction_score:.0f}/100")

            st.markdown("---")
            st.subheader("🎯 Key Levels")

            col1, col2, col3 = st.columns(3)
            col1.metric("Target", f"{summary.key_target:.2f}")
            col2.metric("Resistance", f"{summary.resistance_level:.2f}")
            col3.metric("Support", f"{summary.support_level:.2f}")

            st.markdown("---")
            st.subheader("💡 Actionable Insights")

            for insight in summary.actionable_insights:
                st.info(insight)

        # Tab 2: Volatility & Risk
        with analysis_tabs[1]:
            st.subheader("⚡ Volatility Regime Analysis")

            vol = result.volatility_regime

            col1, col2, col3 = st.columns(3)
            col1.metric("Regime", vol.regime.value)
            col2.metric("VIX", f"{vol.vix_level:.2f}")
            col3.metric("Trend", vol.trend.value)

            st.progress(vol.regime_strength / 100, text=f"Regime Strength: {vol.regime_strength:.0f}/100")

            st.markdown(f"**Recommended Strategy:** {vol.recommended_strategy}")

            if vol.gamma_flip_detected:
                st.error("⚠️ GAMMA FLIP DETECTED - Expect volatile moves!")

            if vol.is_expiry_week:
                st.warning(f"⏰ Expiry Week - {result.market_summary.risk_level} risk")

            st.markdown("---")
            st.subheader("🚨 OI Trap Detection")

            oi = result.oi_trap

            if oi.trap_detected:
                st.error(f"⚠️ TRAP DETECTED: {oi.trap_type.value}")
                st.metric("Trap Probability", f"{oi.trap_probability:.1f}%")
                st.metric("Retail Trap Score", f"{oi.retail_trap_score:.1f}/100")
                st.metric("Smart Money", oi.smart_money_signal)
                st.metric("Trapped Direction", oi.trapped_direction)

                st.warning(f"**Recommendation:** {oi.recommendation}")
            else:
                st.success("✅ No trap detected - Market conditions appear clean")

        # Tab 3: Institutional Flow
        with analysis_tabs[2]:
            st.subheader("🏦 Institutional vs Retail Detection")

            participant = result.participant

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Dominant Participant", participant.dominant_participant.value)
                st.metric("Entry Type", participant.entry_type.value)

                if participant.smart_money_detected:
                    st.success("✅ Smart Money Detected")

                if participant.dumb_money_detected:
                    st.warning("⚠️ Retail Activity Detected")

            with col2:
                st.metric("Institutional Confidence", f"{participant.institutional_confidence:.1f}%")
                st.metric("Retail Confidence", f"{participant.retail_confidence:.1f}%")

            st.progress(participant.institutional_confidence / 100, text="Institutional Presence")

            st.markdown("---")
            st.info(f"**Recommendation:** {participant.recommendation}")

            st.markdown("---")
            st.subheader("📊 CVD & Delta Imbalance")

            cvd = result.cvd

            col1, col2, col3 = st.columns(3)
            col1.metric("CVD Bias", cvd.bias)
            col2.metric("Delta Imbalance", f"{cvd.delta_imbalance:+.1f}%")
            col3.metric("Orderflow Strength", f"{cvd.orderflow_strength:.0f}/100")

            if cvd.delta_divergence_detected:
                st.warning("⚠️ Delta Divergence - Price vs Orderflow mismatch")

            if cvd.institutional_sweep:
                st.error("⚡ INSTITUTIONAL SWEEP DETECTED")

        # Tab 4: Liquidity Gravity
        with analysis_tabs[3]:
            st.subheader("🧲 Liquidity Gravity Analysis")

            liq = result.liquidity

            st.metric("Primary Target", f"{liq.primary_target:.2f}")
            st.progress(liq.gravity_strength / 100, text=f"Gravity Strength: {liq.gravity_strength:.0f}/100")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🛡️ Support Zones**")
                if liq.support_zones:
                    for zone in liq.support_zones[:5]:
                        st.markdown(f"- **{zone.price:.2f}** ({zone.type}) - Strength: {zone.strength:.0f}%")
                else:
                    st.info("No significant support zones nearby")

            with col2:
                st.markdown("**⚡ Resistance Zones**")
                if liq.resistance_zones:
                    for zone in liq.resistance_zones[:5]:
                        st.markdown(f"- **{zone.price:.2f}** ({zone.type}) - Strength: {zone.strength:.0f}%")
                else:
                    st.info("No significant resistance zones nearby")

            if liq.fair_value_gaps:
                st.markdown("---")
                st.markdown("**📈 Fair Value Gaps (Unfilled)**")
                for gap_low, gap_high in liq.fair_value_gaps[:3]:
                    st.markdown(f"- Gap: **{gap_low:.2f}** to **{gap_high:.2f}**")

            if liq.gamma_walls:
                st.markdown("---")
                st.markdown("**🧱 Gamma Walls**")
                for strike, wall_type in liq.gamma_walls[:5]:
                    st.markdown(f"- **{strike:.0f}** ({wall_type})")

        # Tab 5: Position & Risk Management
        with analysis_tabs[4]:
            st.subheader("💰 Position Sizing")

            if result.position_size:
                pos = result.position_size

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Lots", pos.recommended_lots)
                col2.metric("Contracts", pos.recommended_contracts)
                col3.metric("Risk %", f"{pos.risk_percentage:.2f}%")
                col4.metric("Position Value", f"₹{pos.position_value:,.0f}")

                st.markdown("---")
                st.markdown(f"**Sizing Method:** {pos.sizing_method}")
                st.markdown(f"**Kelly Fraction:** {pos.kelly_fraction:.3f}")

                if pos.warnings:
                    st.markdown("**⚠️ Warnings:**")
                    for warning in pos.warnings:
                        st.warning(warning)
            else:
                st.info("Enter trade parameters in Trade Setup tab to calculate position size")

            st.markdown("---")
            st.subheader("🛡️ Risk Management")

            if result.risk_management:
                risk = result.risk_management

                col1, col2, col3 = st.columns(3)
                col1.metric("Stop Loss", f"{risk.stop_loss:.2f}")
                col2.metric("Take Profit", f"{risk.take_profit:.2f}")
                col3.metric("Risk Score", f"{risk.risk_score:.0f}/100")

                st.markdown(f"**Break-Even Trigger:** {risk.break_even_trigger:.2f}")

                if risk.trailing_stop:
                    st.markdown(f"**Trailing Stop:** {risk.trailing_stop:.2f} points")

                st.markdown("---")
                st.markdown("**📊 Partial Profit Plan:**")
                for i, (price, pct) in enumerate(risk.partial_profit_levels, 1):
                    st.markdown(f"{i}. Exit **{pct*100:.0f}%** at **{price:.2f}**")

                if risk.avoid_reasons:
                    st.markdown("---")
                    st.error("**🚫 Avoidance Reasons:**")
                    for reason in risk.avoid_reasons:
                        st.markdown(f"- {reason}")
                else:
                    st.success("✅ No risk issues detected")

            if result.expectancy:
                st.markdown("---")
                st.subheader("📈 Expectancy Model")

                exp = result.expectancy

                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Value", f"₹{exp.expected_value:.2f}/trade")
                col2.metric("Win Rate", f"{exp.win_rate:.1f}%")
                col3.metric("Profit Factor", f"{exp.profit_factor:.2f}")

                col1, col2 = st.columns(2)
                col1.metric("Avg Win", f"₹{exp.avg_win:.2f}")
                col2.metric("Avg Loss", f"₹{exp.avg_loss:.2f}")

                st.markdown(f"**Payoff Ratio:** {exp.payoff_ratio:.2f}:1")
                st.markdown(f"**Expected Edge:** {exp.expected_edge:+.2f}% per trade")

        # Tab 6: Full Report
        with analysis_tabs[5]:
            st.subheader("📋 Complete Master AI Report")

            report = format_master_report(result)
            st.code(report, language="text")

            # Download button
            st.download_button(
                label="📥 Download Full Report",
                data=report,
                file_name=f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )


def render_advanced_analytics_tab(df, option_chain, vix_current, vix_history):
    """
    Render individual module analysis tab
    Allows users to explore each AI module separately
    """
    st.header("🔬 Advanced Analytics")
    st.markdown("**Explore Individual AI Modules**")

    if not AI_MODULES_AVAILABLE:
        st.error("Advanced AI modules not found. Please check installation.")
        return

    # Module selector
    module = st.selectbox(
        "Select Module to Analyze",
        [
            "Volatility Regime Detection",
            "OI Trap Detection",
            "CVD & Delta Imbalance",
            "Institutional vs Retail",
            "Liquidity Gravity",
            "ML Market Regime"
        ]
    )

    if module == "Volatility Regime Detection":
        from src.volatility_regime import VolatilityRegimeDetector

        detector = VolatilityRegimeDetector()
        result = detector.analyze_regime(df, vix_current, vix_history, option_chain, days_to_expiry=5)

        report = format_regime_report(result)
        st.code(report, language="text")

    elif module == "OI Trap Detection":
        from src.oi_trap_detection import OITrapDetector

        detector = OITrapDetector()
        result = detector.detect_traps(option_chain, df['close'].iloc[-1], df)

        report = format_oi_trap_report(result)
        st.code(report, language="text")

    elif module == "CVD & Delta Imbalance":
        from src.cvd_delta_imbalance import CVDAnalyzer

        analyzer = CVDAnalyzer()
        result = analyzer.analyze_cvd(df)

        report = format_cvd_report(result)
        st.code(report, language="text")

    elif module == "Institutional vs Retail":
        from src.institutional_retail_detector import InstitutionalRetailDetector

        detector = InstitutionalRetailDetector()
        result = detector.detect_participant(df, option_chain, df['close'].iloc[-1])

        report = format_participant_report(result)
        st.code(report, language="text")

    elif module == "Liquidity Gravity":
        from src.liquidity_gravity import LiquidityGravityAnalyzer

        analyzer = LiquidityGravityAnalyzer()
        result = analyzer.analyze_liquidity_gravity(df, option_chain)

        report = format_liquidity_report(result)
        st.code(report, language="text")

    elif module == "ML Market Regime":
        from src.ml_market_regime import MLMarketRegimeDetector

        detector = MLMarketRegimeDetector()
        result = detector.detect_regime(df)

        st.markdown(f"**Regime:** {result.regime}")
        st.markdown(f"**Confidence:** {result.confidence:.1f}%")
        st.markdown(f"**Strategy:** {result.recommended_strategy}")


# ============================================================================
# INTEGRATION INSTRUCTIONS
# ============================================================================

"""
TO INTEGRATE INTO YOUR APP.PY:

1. Add these imports at the top of app.py:

   from ai_tab_integration import render_master_ai_analysis_tab, render_advanced_analytics_tab

2. Modify the tabs line (around line 1720) from:

   tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([...])

   TO:

   tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
       "🌟 Overall Market Sentiment",
       "🎯 Trade Setup",
       "📊 Active Signals",
       "📈 Positions",
       "🎲 Bias Analysis Pro",
       "🔍 Option Chain Analysis",
       "📉 Advanced Chart Analysis",
       "🎯 NIFTY Option Screener v7.0",
       "🌐 Enhanced Market Data",
       "🤖 MASTER AI ANALYSIS",      # NEW
       "🔬 Advanced Analytics"        # NEW
   ])

3. Add these new tab implementations at the end (around line 3400):

   # TAB 10: MASTER AI ANALYSIS
   with tab10:
       if 'ohlcv_data' in st.session_state and 'option_chain_data' in st.session_state:
           render_master_ai_analysis_tab(
               df=st.session_state.ohlcv_data,
               option_chain=st.session_state.option_chain_data,
               vix_current=st.session_state.get('india_vix', 15.0),
               vix_history=st.session_state.get('vix_history', pd.Series([15.0])),
               instrument=selected_index,
               days_to_expiry=calculate_days_to_expiry()
           )
       else:
           st.warning("Please fetch market data first")

   # TAB 11: ADVANCED ANALYTICS
   with tab11:
       if 'ohlcv_data' in st.session_state:
           render_advanced_analytics_tab(
               df=st.session_state.ohlcv_data,
               option_chain=st.session_state.get('option_chain_data', {}),
               vix_current=st.session_state.get('india_vix', 15.0),
               vix_history=st.session_state.get('vix_history', pd.Series([15.0]))
           )
       else:
           st.warning("Please fetch market data first")

4. Make sure your session_state has:
   - st.session_state.ohlcv_data (DataFrame with OHLCV)
   - st.session_state.option_chain_data (Dict with CE/PE data)
   - st.session_state.india_vix (float)
   - st.session_state.vix_history (Series)

THAT'S IT! Your advanced AI modules will now be displayed in 2 new tabs.
"""
