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
