"""
INTEGRATION EXAMPLE
How to use the Master AI Orchestrator in your Streamlit app
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src.master_ai_orchestrator import MasterAIOrchestrator, format_master_report

# =============================================================================
# EXAMPLE 1: Basic Usage (Minimal)
# =============================================================================

def example_basic_analysis():
    """
    Minimal example showing basic market analysis
    """
    # Initialize orchestrator
    orchestrator = MasterAIOrchestrator(
        account_size=100000,  # Your capital
        max_risk_per_trade=2.0  # Max 2% risk per trade
    )

    # Prepare your data (you already have this from Dhan API)
    ohlcv_data = pd.DataFrame({
        'open': [...],  # Your OHLC data
        'high': [...],
        'low': [...],
        'close': [...],
        'volume': [...],
        'atr': [...]  # Add ATR if you have it
    })

    option_chain = {
        'CE': {
            'strikePrice': [...],
            'openInterest': [...],
            'changeinOpenInterest': [...],
            'totalTradedVolume': [...]
        },
        'PE': {
            'strikePrice': [...],
            'openInterest': [...],
            'changeinOpenInterest': [...],
            'totalTradedVolume': [...]
        }
    }

    # VIX data
    india_vix = 15.5  # Current VIX
    vix_history = pd.Series([...])  # Historical VIX

    # Technical bias data (from bias_analysis.py - 13 indicators)
    bias_data = {
        'volume_delta': 0.65,
        'hvp': 0.72,
        'vob': 0.58,
        'order_blocks': 0.68,
        'rsi': 0.45,
        'dmi': 0.71,
        'vidya': 0.63,
        'mfi': 0.55,
        'vwap': 0.69,
        'atr': 0.48,
        'ema': 0.67,
        'obv': 0.61,
        'force_index': 0.59
    }

    # Option screener data (from NiftyOptionScreener.py)
    screener_data = {
        'momentum_burst': 0.78,
        'orderbook_pressure': 0.82,
        'gamma_cluster': 0.65,
        'oi_acceleration': 0.71,
        'expiry_spike': True,
        'net_vega_exposure': 1250000,
        'skew_ratio': 1.15,
        'atm_vol_premium': 0.08
    }

    # Run analysis
    result = orchestrator.analyze_complete_market(
        df=ohlcv_data,
        option_chain=option_chain,
        vix_current=india_vix,
        vix_history=vix_history,
        instrument="NIFTY",
        days_to_expiry=5,
        bias_results=bias_data,  # Technical indicators
        sentiment_score=0.65,  # AI news sentiment (-1 to +1)
        option_screener_data=screener_data  # Option screener analysis
    )

    # Get verdict
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {result.final_verdict}")
    print(f"CONFIDENCE: {result.confidence:.1f}%")
    print(f"TRADE QUALITY: {result.trade_quality_score:.1f}/100")
    print(f"{'='*60}\n")

    # Access specific modules
    print(f"Volatility Regime: {result.volatility_regime.regime.value}")
    print(f"OI Trap Detected: {result.oi_trap.trap_detected}")
    print(f"CVD Bias: {result.cvd.bias}")
    print(f"Institutional Confidence: {result.participant.institutional_confidence:.1f}%")
    print(f"Liquidity Target: {result.liquidity.primary_target:.2f}")

    # Full report
    print(format_master_report(result))


# =============================================================================
# EXAMPLE 2: Complete Usage (With Trade Setup)
# =============================================================================

def example_complete_with_trade():
    """
    Complete example including position sizing and risk management
    """
    orchestrator = MasterAIOrchestrator(account_size=100000, max_risk_per_trade=2.0)

    # Your OHLCV data
    ohlcv_data = pd.DataFrame({...})
    option_chain = {...}
    india_vix = 15.5
    vix_history = pd.Series([...])

    # Technical bias and sentiment
    bias_data = {...}  # From bias_analysis.py
    sentiment = 0.65  # From AI news analysis
    screener_data = {...}  # From option screener

    # Your trade setup
    current_price = 22000
    entry_price = 22050
    stop_loss = 21950
    target_price = 22250
    direction = "CALL"  # or "PUT"

    # Historical trades (optional but recommended for expectancy)
    historical_trades = pd.DataFrame({
        'pnl': [100, -50, 150, -30, 200, -40, 180],  # Past P&Ls
    })

    # Run complete analysis
    result = orchestrator.analyze_complete_market(
        df=ohlcv_data,
        option_chain=option_chain,
        vix_current=india_vix,
        vix_history=vix_history,
        instrument="NIFTY",
        days_to_expiry=5,
        historical_trades=historical_trades,
        trade_params={
            'entry': entry_price,
            'stop': stop_loss,
            'target': target_price,
            'direction': direction
        },
        bias_results=bias_data,
        sentiment_score=sentiment,
        option_screener_data=screener_data
    )

    # Decision logic
    if result.final_verdict == "NO TRADE":
        print("❌ Trade rejected by AI")
        print("Reasons:")
        for reason in result.reasoning:
            print(f"  - {reason}")
        return

    if result.trade_quality_score < 60:
        print("⚠️ Low quality setup - SKIP")
        return

    if result.final_verdict in ["STRONG BUY", "BUY"]:
        # Position sizing
        if result.position_size:
            lots = result.position_size.recommended_lots
            risk_pct = result.position_size.risk_percentage

            print(f"✅ TRADE APPROVED")
            print(f"Signal: {result.final_verdict}")
            print(f"Position: {lots} lots")
            print(f"Risk: {risk_pct:.2f}%")

            # Risk management
            if result.risk_management:
                print(f"Stop Loss: {result.risk_management.stop_loss:.2f}")
                print(f"Target: {result.risk_management.take_profit:.2f}")
                print(f"Break-even at: {result.risk_management.break_even_trigger:.2f}")

            # Expectancy
            if result.expectancy:
                print(f"Expected Value: ₹{result.expectancy.expected_value:.2f} per trade")
                print(f"Win Rate: {result.expectancy.win_rate:.1f}%")

            # Execute trade (pseudo-code)
            # place_order(
            #     direction=direction,
            #     lots=lots,
            #     entry=entry_price,
            #     stop=result.risk_management.stop_loss,
            #     target=result.risk_management.take_profit
            # )


# =============================================================================
# EXAMPLE 3: Streamlit Integration
# =============================================================================

def streamlit_integration_example():
    """
    How to integrate with Streamlit UI
    """
    import streamlit as st

    st.title("🤖 AI Trading Orchestrator")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        account_size = st.number_input("Account Size (₹)", value=100000)
        max_risk = st.slider("Max Risk per Trade (%)", 0.5, 5.0, 2.0, 0.1)

    # Initialize orchestrator
    orchestrator = MasterAIOrchestrator(
        account_size=account_size,
        max_risk_per_trade=max_risk
    )

    # Fetch data (you already have this logic)
    # df = fetch_ohlcv_data()
    # option_chain = fetch_option_chain()
    # vix_current = fetch_vix()
    # vix_history = fetch_vix_history()
    # bias_data = calculate_bias_analysis(df)  # From bias_analysis.py
    # sentiment = get_ai_sentiment()  # From AI news engine
    # screener_data = get_option_screener_data()  # From option screener

    # Button to run analysis
    if st.button("🔍 Run Complete Analysis"):
        with st.spinner("Analyzing market..."):
            result = orchestrator.analyze_complete_market(
                df=df,
                option_chain=option_chain,
                vix_current=vix_current,
                vix_history=vix_history,
                instrument="NIFTY",
                days_to_expiry=5,
                bias_results=bias_data,
                sentiment_score=sentiment,
                option_screener_data=screener_data
            )

        # Display verdict prominently
        st.header(f"🎯 {result.final_verdict}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{result.confidence:.1f}%")
        col2.metric("Trade Quality", f"{result.trade_quality_score:.1f}/100")
        col3.metric("Win Probability", f"{result.expected_win_probability:.1f}%")

        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Market Summary",
            "⚠️ Risk Analysis",
            "🏦 Participant Detection",
            "🧲 Liquidity Gravity",
            "📈 Full Report"
        ])

        with tab1:
            st.subheader("Market Summary")
            st.write(result.market_summary.summary_text)
            for insight in result.market_summary.actionable_insights:
                st.info(insight)

        with tab2:
            st.subheader("Volatility & Risk")
            st.write(f"**Regime**: {result.volatility_regime.regime.value}")
            st.write(f"**VIX**: {result.volatility_regime.vix_level:.2f}")

            if result.oi_trap.trap_detected:
                st.error(f"⚠️ OI TRAP: {result.oi_trap.trap_type.value}")
                st.write(f"Trap Probability: {result.oi_trap.trap_probability:.1f}%")

        with tab3:
            st.subheader("Institutional vs Retail")
            st.write(f"**Dominant**: {result.participant.dominant_participant.value}")
            st.write(f"**Entry Type**: {result.participant.entry_type.value}")
            st.progress(result.participant.institutional_confidence / 100)
            st.caption(f"Institutional Confidence: {result.participant.institutional_confidence:.1f}%")

        with tab4:
            st.subheader("Liquidity Gravity")
            st.write(f"**Primary Target**: {result.liquidity.primary_target:.2f}")
            st.write(f"**Gravity Strength**: {result.liquidity.gravity_strength:.1f}/100")

            if result.liquidity.support_zones:
                st.write("**Support Zones**:")
                for zone in result.liquidity.support_zones[:3]:
                    st.write(f"  - {zone.price:.2f} ({zone.type})")

            if result.liquidity.resistance_zones:
                st.write("**Resistance Zones**:")
                for zone in result.liquidity.resistance_zones[:3]:
                    st.write(f"  - {zone.price:.2f} ({zone.type})")

        with tab5:
            st.subheader("Complete Analysis Report")
            st.code(format_master_report(result), language="text")


# =============================================================================
# EXAMPLE 4: Individual Module Usage
# =============================================================================

def example_individual_modules():
    """
    Use individual modules separately if needed
    """
    from src.volatility_regime import VolatilityRegimeDetector, format_regime_report
    from src.oi_trap_detection import OITrapDetector, format_oi_trap_report
    from src.cvd_delta_imbalance import CVDAnalyzer, format_cvd_report

    df = pd.DataFrame({...})
    option_chain = {...}

    # 1. Volatility Regime Only
    vol_detector = VolatilityRegimeDetector()
    vol_result = vol_detector.analyze_regime(
        df=df,
        vix_current=15.5,
        vix_history=pd.Series([...]),
        option_chain=option_chain,
        days_to_expiry=5
    )
    print(format_regime_report(vol_result))

    # 2. OI Trap Only
    trap_detector = OITrapDetector()
    trap_result = trap_detector.detect_traps(
        option_chain=option_chain,
        price_current=22000,
        price_history=df
    )
    print(format_oi_trap_report(trap_result))

    # 3. CVD Only
    cvd_analyzer = CVDAnalyzer()
    cvd_result = cvd_analyzer.analyze_cvd(df)
    print(format_cvd_report(cvd_result))


# =============================================================================
# EXAMPLE 5: Automated Trading Loop
# =============================================================================

def example_automated_trading_loop():
    """
    Automated trading loop (pseudo-code)
    """
    orchestrator = MasterAIOrchestrator(account_size=100000, max_risk_per_trade=2.0)

    while market_is_open():
        # Fetch latest data
        df = fetch_latest_ohlcv()
        option_chain = fetch_option_chain()
        vix_current = fetch_vix()
        vix_history = fetch_vix_history()
        bias_data = calculate_bias_analysis(df)
        sentiment = get_ai_sentiment()
        screener_data = get_option_screener_data()

        # Analyze
        result = orchestrator.analyze_complete_market(
            df=df,
            option_chain=option_chain,
            vix_current=vix_current,
            vix_history=vix_history,
            instrument="NIFTY",
            days_to_expiry=calculate_days_to_expiry(),
            bias_results=bias_data,
            sentiment_score=sentiment,
            option_screener_data=screener_data
        )

        # Decision logic
        if result.final_verdict in ["STRONG BUY", "BUY"]:
            if result.trade_quality_score > 70:
                # High quality buy signal
                execute_buy_trade(result)

        elif result.final_verdict in ["STRONG SELL", "SELL"]:
            if result.trade_quality_score > 70:
                # High quality sell signal
                execute_sell_trade(result)

        # Sleep for next candle
        wait_for_next_candle()


def execute_buy_trade(result):
    """Execute buy trade based on AI recommendation"""
    if result.position_size and result.risk_management:
        # Place order
        order = {
            'symbol': 'NIFTY',
            'direction': 'CALL',
            'lots': result.position_size.recommended_lots,
            'entry': result.risk_management.stop_loss + 100,  # Example
            'stop_loss': result.risk_management.stop_loss,
            'target': result.risk_management.take_profit
        }

        # Log
        print(f"✅ Executing BUY: {order}")

        # Place via Dhan API
        # dhan.place_order(order)


# =============================================================================
# RUN EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("Choose example to run:")
    print("1. Basic Analysis")
    print("2. Complete with Trade Setup")
    print("3. Streamlit Integration")
    print("4. Individual Modules")
    print("5. Automated Trading Loop")

    choice = input("Enter choice (1-5): ")

    if choice == "1":
        example_basic_analysis()
    elif choice == "2":
        example_complete_with_trade()
    elif choice == "3":
        streamlit_integration_example()
    elif choice == "4":
        example_individual_modules()
    elif choice == "5":
        example_automated_trading_loop()
