"""
Native Streamlit Signal Display (NO HTML)
Uses only Streamlit components for clean rendering
"""

import streamlit as st
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class SignalData:
    """Signal data structure"""
    signal_type: str  # "WAIT", "ENTRY", "EXIT"
    direction: str  # "NEUTRAL", "LONG", "SHORT"
    confidence: float
    xgboost_regime: str
    market_bias: str
    zone_width: float
    zone_width_status: str
    support_price: float
    support_source: str
    support_distance: float
    resistance_price: float
    resistance_source: str
    resistance_distance: float
    vob_major_support: Optional[float] = None
    vob_major_resistance: Optional[float] = None
    vob_minor_support: Optional[float] = None
    vob_minor_resistance: Optional[float] = None
    setup_type: str = "No Setup"
    entry_zone: str = "Wait for clear direction"
    stop_loss: str = "N/A"
    target: str = "N/A"
    expiry_status: str = "⚠️ HIGH VOLATILITY"
    gex_level: str = "Neutral"
    vix: float = 15.0
    analysis_text: str = "Low confidence. Wait for better setup."
    current_price: float = 0.0
    pcr: float = 1.0
    bullish_count: int = 0
    bearish_count: int = 0


def display_signal_native(data: SignalData):
    """
    Display trading signal using ONLY Streamlit native components
    NO HTML - Pure Streamlit
    """

    # Determine colors based on signal type
    if data.signal_type == "ENTRY" and data.direction == "LONG":
        header_emoji = "🟢"
        header_color = "green"
    elif data.signal_type == "ENTRY" and data.direction == "SHORT":
        header_emoji = "🔴"
        header_color = "red"
    elif data.signal_type == "EXIT":
        header_emoji = "🚪"
        header_color = "red"
    else:  # WAIT
        header_emoji = "🔴"
        header_color = "orange"

    # ================================================================
    # HEADER
    # ================================================================
    bias_emoji_map = {
        "BULLISH": "🟢",
        "BEARISH": "🔴",
        "NEUTRAL": "⚖️",
        "Mild Bullish": "🐂",
        "Mild Bearish": "🐻"
    }
    bias_emoji = bias_emoji_map.get(data.market_bias, "⚖️")

    st.markdown(f"## {header_emoji} **{data.signal_type}** - {bias_emoji} {data.market_bias}")
    st.markdown("---")

    # ================================================================
    # CORE METRICS (4 columns)
    # ================================================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Confidence color
        if data.confidence >= 70:
            delta_color = "normal"
        elif data.confidence >= 50:
            delta_color = "off"
        else:
            delta_color = "inverse"

        st.metric(
            label="Confidence",
            value=f"{data.confidence:.0f}%",
            delta=None
        )

    with col2:
        st.metric(
            label="XGBoost Regime",
            value=data.xgboost_regime
        )

    with col3:
        st.metric(
            label="Market Bias",
            value=data.market_bias
        )

    with col4:
        # Zone width color indicator
        zone_indicator = "🔴" if data.zone_width_status == "WIDE" else ("🟡" if data.zone_width_status == "MODERATE" else "🟢")
        st.metric(
            label="Zone Width",
            value=f"{zone_indicator} {data.zone_width:.0f} pts",
            delta=data.zone_width_status
        )

    st.markdown("---")

    # ================================================================
    # SUPPORT & RESISTANCE (2 columns)
    # ================================================================
    st.markdown("### 📊 Support & Resistance")

    col_sup, col_res = st.columns(2)

    with col_sup:
        st.markdown("#### 🟢 SUPPORT")
        st.markdown(f"### ₹{data.support_price:,.0f}")
        st.caption(f"**Source:** {data.support_source}")
        st.caption(f"**Distance:** {data.support_distance:.0f} points away")

    with col_res:
        st.markdown("#### 🔴 RESISTANCE")
        st.markdown(f"### ₹{data.resistance_price:,.0f}")
        st.caption(f"**Source:** {data.resistance_source}")
        st.caption(f"**Distance:** {data.resistance_distance:.0f} points away")

    st.markdown("---")

    # ================================================================
    # VOB LEVELS
    # ================================================================
    st.markdown("### 📊 VOB LEVELS")

    vob_col1, vob_col2 = st.columns(2)

    with vob_col1:
        major_sup = f"₹{data.vob_major_support:,.0f}" if data.vob_major_support else "N/A"
        minor_sup = f"₹{data.vob_minor_support:,.0f}" if data.vob_minor_support else "N/A"
        st.markdown(f"**Major Support:** {major_sup}")
        st.markdown(f"**Minor Support:** {minor_sup}")

    with vob_col2:
        major_res = f"₹{data.vob_major_resistance:,.0f}" if data.vob_major_resistance else "N/A"
        minor_res = f"₹{data.vob_minor_resistance:,.0f}" if data.vob_minor_resistance else "N/A"
        st.markdown(f"**Major Resistance:** {major_res}")
        st.markdown(f"**Minor Resistance:** {minor_res}")

    st.markdown("---")

    # ================================================================
    # ENTRY SETUP
    # ================================================================
    st.markdown("### 🎯 ENTRY SETUP")

    # Choose container based on signal type
    if data.signal_type == "ENTRY":
        if data.direction == "LONG":
            container = st.success
        elif data.direction == "SHORT":
            container = st.error
        else:
            container = st.info
    elif data.signal_type == "EXIT":
        container = st.error
    else:  # WAIT
        container = st.warning

    with container("Setup Details"):
        st.markdown(f"**Setup Type:** {data.setup_type}")
        st.markdown(f"**Entry Zone:** {data.entry_zone}")
        st.markdown(f"**Stop Loss:** {data.stop_loss}")
        st.markdown(f"**Target:** {data.target}")

    st.markdown("---")

    # ================================================================
    # ADDITIONAL INFO (3 columns)
    # ================================================================
    st.markdown("### 📈 Additional Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.metric(
            label="Expiry Status",
            value=data.expiry_status
        )

    with info_col2:
        st.metric(
            label="GEX Level",
            value=data.gex_level
        )

    with info_col3:
        st.metric(
            label="VIX",
            value=f"{data.vix:.1f}"
        )

    st.markdown("---")

    # ================================================================
    # ANALYSIS REASON
    # ================================================================
    st.markdown("### 💡 Analysis")
    st.info(data.analysis_text)

    st.markdown("---")

    # ================================================================
    # FOOTER
    # ================================================================
    footer_text = f"**Current:** ₹{data.current_price:,.2f} | **PCR:** {data.pcr:.2f} | **Bullish:** {data.bullish_count} | **Bearish:** {data.bearish_count}"
    st.caption(footer_text)

    # ================================================================
    # TRADE RECOMMENDATION
    # ================================================================
    if data.signal_type == "WAIT":
        st.error("### 🔴 NO TRADE")
        st.warning("""
        **Wait for better setup**
        - Price in mid-zone or confidence too low
        - Be patient - missing a trade is better than a bad trade
        """)


def display_trading_signal(
    signal_type: str,
    direction: str,
    confidence: float,
    support: float,
    resistance: float,
    current_price: float,
    **kwargs
):
    """
    Simplified function to display trading signal

    Args:
        signal_type: "ENTRY", "EXIT", "WAIT"
        direction: "LONG", "SHORT", "NEUTRAL"
        confidence: 0-100
        support: Support price
        resistance: Resistance price
        current_price: Current spot price
        **kwargs: Additional optional parameters
    """

    data = SignalData(
        signal_type=signal_type,
        direction=direction,
        confidence=confidence,
        xgboost_regime=kwargs.get('xgboost_regime', 'RANGING'),
        market_bias=kwargs.get('market_bias', 'NEUTRAL'),
        zone_width=kwargs.get('zone_width', abs(resistance - support)),
        zone_width_status=kwargs.get('zone_width_status', 'MEDIUM'),
        support_price=support,
        support_source=kwargs.get('support_source', 'Calculated'),
        support_distance=abs(current_price - support),
        resistance_price=resistance,
        resistance_source=kwargs.get('resistance_source', 'Calculated'),
        resistance_distance=abs(resistance - current_price),
        vob_major_support=kwargs.get('vob_major_support'),
        vob_major_resistance=kwargs.get('vob_major_resistance'),
        vob_minor_support=kwargs.get('vob_minor_support'),
        vob_minor_resistance=kwargs.get('vob_minor_resistance'),
        setup_type=kwargs.get('setup_type', 'No Setup'),
        entry_zone=kwargs.get('entry_zone', 'Wait for clear direction'),
        stop_loss=kwargs.get('stop_loss', 'N/A'),
        target=kwargs.get('target', 'N/A'),
        expiry_status=kwargs.get('expiry_status', '⚠️ HIGH VOLATILITY'),
        gex_level=kwargs.get('gex_level', 'Neutral'),
        vix=kwargs.get('vix', 15.0),
        analysis_text=kwargs.get('analysis_text', 'Low confidence. Wait for better setup.'),
        current_price=current_price,
        pcr=kwargs.get('pcr', 1.0),
        bullish_count=kwargs.get('bullish_count', 0),
        bearish_count=kwargs.get('bearish_count', 0)
    )

    display_signal_native(data)


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_app():
    """Example Streamlit app"""

    st.title("🎯 Native Streamlit Signal Display")
    st.markdown("---")

    # Example 1: WAIT Signal
    st.subheader("Example 1: WAIT Signal (Current Display)")

    display_trading_signal(
        signal_type="WAIT",
        direction="NEUTRAL",
        confidence=55.0,
        support=26077.0,
        resistance=26277.0,
        current_price=26177.15,
        xgboost_regime="RANGING",
        market_bias="Mild Bearish",
        zone_width=200.0,
        zone_width_status="WIDE",
        support_source="Calculated",
        resistance_source="Calculated",
        analysis_text="Low confidence or price in mid-zone. Wait for better setup.",
        expiry_status="⚠️ HIGH VOLATILITY",
        gex_level="Neutral",
        vix=15.0,
        pcr=1.0,
        bullish_count=0,
        bearish_count=0
    )

    st.markdown("---")
    st.markdown("---")

    # Example 2: LONG Entry
    st.subheader("Example 2: LONG Entry Signal")

    display_trading_signal(
        signal_type="ENTRY",
        direction="LONG",
        confidence=75.0,
        support=26100.0,
        resistance=26300.0,
        current_price=26120.0,
        xgboost_regime="TRENDING_UP",
        market_bias="BULLISH",
        zone_width=50.0,
        zone_width_status="NARROW",
        support_source="VOB Support",
        resistance_source="HTF Resistance",
        vob_major_support=26050.0,
        vob_major_resistance=26350.0,
        vob_minor_support=26075.0,
        vob_minor_resistance=26325.0,
        setup_type="LONG at VOB Support",
        entry_zone="₹26,100 - ₹26,120",
        stop_loss="₹26,030 (Below VOB Support - 20 pts)",
        target="₹26,300 (HTF Resistance)",
        analysis_text="Strong bullish setup with 75% confidence. Price at major VOB support with trending regime. High gamma squeeze provides tight zone and strong defense.",
        expiry_status="🔥 SUPPORT SPIKE (85%)",
        gex_level="Bullish",
        vix=14.2,
        pcr=1.15,
        bullish_count=8,
        bearish_count=2
    )

    st.markdown("---")
    st.markdown("---")

    # Example 3: SHORT Entry
    st.subheader("Example 3: SHORT Entry Signal")

    display_trading_signal(
        signal_type="ENTRY",
        direction="SHORT",
        confidence=72.0,
        support=26050.0,
        resistance=26250.0,
        current_price=26240.0,
        xgboost_regime="TRENDING_DOWN",
        market_bias="BEARISH",
        zone_width=45.0,
        zone_width_status="NARROW",
        support_source="HTF Support",
        resistance_source="VOB Resistance",
        vob_major_support=26000.0,
        vob_major_resistance=26280.0,
        setup_type="SHORT at VOB Resistance",
        entry_zone="₹26,240 - ₹26,260",
        stop_loss="₹26,310 (Above VOB Resistance + 20 pts)",
        target="₹26,050 (HTF Support)",
        analysis_text="Bearish setup with 72% confidence. Price rejecting at major VOB resistance with strong selling volume. Market depth shows ask-heavy order book.",
        expiry_status="🔥 RESISTANCE SPIKE (78%)",
        gex_level="Bearish",
        vix=16.8,
        pcr=0.85,
        bullish_count=3,
        bearish_count=9
    )


if __name__ == "__main__":
    example_app()
