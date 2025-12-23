"""
Streamlit Signal Display Integration
Easy-to-use functions for displaying trading signals in Streamlit
"""

import streamlit as st
from typing import Optional, Dict
from src.signal_display_html_generator import (
    SignalDisplayData,
    generate_signal_html,
    generate_signal_html_from_dict
)
from src.enhanced_signal_generator import TradingSignal


def display_signal_in_streamlit(
    signal: TradingSignal,
    current_price: float,
    pcr: float = 1.0,
    bullish_count: int = 0,
    bearish_count: int = 0,
    vob_data: Optional[Dict] = None,
    regime_data: Optional[Dict] = None,
    zone_width: float = 200.0
) -> None:
    """
    Display TradingSignal in Streamlit using HTML

    Args:
        signal: TradingSignal object from enhanced_signal_generator
        current_price: Current spot price
        pcr: Put-Call Ratio
        bullish_count: Number of bullish indicators
        bearish_count: Number of bearish indicators
        vob_data: Volume Order Block data (optional)
        regime_data: Market regime data (optional)
        zone_width: Entry zone width in points
    """

    # Extract data from TradingSignal
    signal_type = signal.signal_type  # "ENTRY", "EXIT", "WAIT"
    direction = signal.direction  # "LONG", "SHORT", "NEUTRAL"
    confidence = signal.confidence
    market_regime = signal.market_regime if hasattr(signal, 'market_regime') else "RANGING"

    # Determine market bias
    if direction == "LONG":
        market_bias = "BULLISH"
    elif direction == "SHORT":
        market_bias = "BEARISH"
    else:
        market_bias = "NEUTRAL"

    # Get support/resistance from signal or VOB data
    support_price = signal.support_level if hasattr(signal, 'support_level') and signal.support_level else current_price - 100
    support_source = signal.support_type if hasattr(signal, 'support_type') and signal.support_type else "Calculated"
    resistance_price = signal.resistance_level if hasattr(signal, 'resistance_level') and signal.resistance_level else current_price + 100
    resistance_source = signal.resistance_type if hasattr(signal, 'resistance_type') and signal.resistance_type else "Calculated"

    support_distance = abs(current_price - support_price)
    resistance_distance = abs(resistance_price - current_price)

    # VOB levels
    vob_major_support = None
    vob_major_resistance = None
    vob_minor_support = None
    vob_minor_resistance = None

    if vob_data:
        # Extract VOB levels from vob_data
        bullish_blocks = vob_data.get('bullish_blocks', [])
        bearish_blocks = vob_data.get('bearish_blocks', [])

        # Find nearest major/minor support
        for block in bullish_blocks:
            if isinstance(block, dict):
                block_mid = (block.get('upper', 0) + block.get('lower', 0)) / 2
                if block_mid < current_price:
                    strength = block.get('strength', 'Medium')
                    if strength == 'Major':
                        if vob_major_support is None or block_mid > vob_major_support:
                            vob_major_support = block_mid
                    elif strength == 'Minor':
                        if vob_minor_support is None or block_mid > vob_minor_support:
                            vob_minor_support = block_mid

        # Find nearest major/minor resistance
        for block in bearish_blocks:
            if isinstance(block, dict):
                block_mid = (block.get('upper', 0) + block.get('lower', 0)) / 2
                if block_mid > current_price:
                    strength = block.get('strength', 'Medium')
                    if strength == 'Major':
                        if vob_major_resistance is None or block_mid < vob_major_resistance:
                            vob_major_resistance = block_mid
                    elif strength == 'Minor':
                        if vob_minor_resistance is None or block_mid < vob_minor_resistance:
                            vob_minor_resistance = block_mid

    # Entry setup details
    if signal_type == "ENTRY":
        setup_type = f"{direction} at {support_source if direction == 'LONG' else resistance_source}"
        entry_zone = f"₹{signal.entry_range_low:,.0f} - ₹{signal.entry_range_high:,.0f}" if signal.entry_range_low else "N/A"
        stop_loss_text = f"₹{signal.stop_loss:,.0f}"
        if hasattr(signal, 'stop_loss_reason') and signal.stop_loss_reason:
            stop_loss_text += f" ({signal.stop_loss_reason})"
        target = f"₹{signal.target_1:,.0f}" if signal.target_1 else "N/A"
    else:
        setup_type = "No Setup" if signal_type == "WAIT" else "Exit Position"
        entry_zone = "Wait for clear direction" if signal_type == "WAIT" else "Close all positions"
        stop_loss_text = "N/A"
        target = "N/A"

    # Determine zone width status
    if zone_width < 100:
        zone_width_status = "NARROW"
    elif zone_width < 200:
        zone_width_status = "MODERATE"
    else:
        zone_width_status = "WIDE"

    # Expiry status (get from regime_data if available)
    expiry_status = "✅ NORMAL"
    gex_level = "Neutral"
    vix = 15.0

    if regime_data:
        if regime_data.get('is_expiry_week', False):
            expiry_status = "⚠️ EXPIRY WEEK"
        if regime_data.get('volatility_regime', '') == 'HIGH_VOLATILITY':
            expiry_status = "⚠️ HIGH VOLATILITY"
        vix = regime_data.get('vix_level', 15.0)

        # Check for expiry spike
        if regime_data.get('expiry_spike_detected', False):
            spike_type = regime_data.get('expiry_spike_type', '')
            spike_prob = regime_data.get('expiry_spike_probability', 0)
            if spike_type == "SUPPORT SPIKE":
                expiry_status = f"🔥 SUPPORT SPIKE ({spike_prob:.0f}%)"
            elif spike_type == "RESISTANCE SPIKE":
                expiry_status = f"🔥 RESISTANCE SPIKE ({spike_prob:.0f}%)"

    # Analysis text
    analysis_text = signal.reason if signal.reason else "No clear setup detected."

    # Create SignalDisplayData
    display_data = SignalDisplayData(
        signal_type=signal_type,
        direction=direction,
        confidence=confidence,
        xgboost_regime=market_regime,
        market_bias=market_bias,
        zone_width=zone_width,
        zone_width_status=zone_width_status,
        support_price=support_price,
        support_source=support_source,
        support_distance=support_distance,
        resistance_price=resistance_price,
        resistance_source=resistance_source,
        resistance_distance=resistance_distance,
        vob_major_support=vob_major_support,
        vob_major_resistance=vob_major_resistance,
        vob_minor_support=vob_minor_support,
        vob_minor_resistance=vob_minor_resistance,
        setup_type=setup_type,
        entry_zone=entry_zone,
        stop_loss=stop_loss_text,
        target=target,
        expiry_status=expiry_status,
        gex_level=gex_level,
        vix=vix,
        analysis_text=analysis_text,
        current_price=current_price,
        pcr=pcr,
        bullish_count=bullish_count,
        bearish_count=bearish_count
    )

    # Generate and display HTML
    html = generate_signal_html(display_data)
    st.markdown(html, unsafe_allow_html=True)


def display_custom_signal(
    signal_type: str,
    direction: str,
    confidence: float,
    support: float,
    resistance: float,
    current_price: float,
    **kwargs
) -> None:
    """
    Display custom signal with minimal parameters

    Args:
        signal_type: "ENTRY", "EXIT", "WAIT"
        direction: "LONG", "SHORT", "NEUTRAL"
        confidence: 0-100
        support: Support price
        resistance: Resistance price
        current_price: Current spot price
        **kwargs: Additional optional parameters
    """

    signal_dict = {
        'signal_type': signal_type,
        'direction': direction,
        'confidence': confidence,
        'support_price': support,
        'resistance_price': resistance,
        'current_price': current_price,
        'support_distance': abs(current_price - support),
        'resistance_distance': abs(resistance - current_price),
        'zone_width': abs(resistance - support),
        **kwargs
    }

    html = generate_signal_html_from_dict(signal_dict)
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# EXAMPLE STREAMLIT APP
# ============================================================

def example_streamlit_app():
    """Example Streamlit app demonstrating signal display"""

    st.title("🎯 Trading Signal Display Examples")

    st.markdown("---")

    # Example 1: WAIT Signal (Neutral)
    st.subheader("Example 1: WAIT Signal (Neutral)")
    display_custom_signal(
        signal_type="WAIT",
        direction="NEUTRAL",
        confidence=55.0,
        support=26077.0,
        resistance=26277.0,
        current_price=26177.15,
        xgboost_regime="RANGING",
        market_bias="NEUTRAL",
        support_source="Calculated",
        resistance_source="Calculated",
        setup_type="No Setup",
        entry_zone="Wait for clear direction",
        analysis_text="Low confidence or price in mid-zone. Wait for better setup.",
        pcr=1.0,
        vix=15.0
    )

    st.markdown("---")

    # Example 2: ENTRY Signal (LONG)
    st.subheader("Example 2: ENTRY Signal (LONG)")
    display_custom_signal(
        signal_type="ENTRY",
        direction="LONG",
        confidence=75.0,
        support=26100.0,
        resistance=26300.0,
        current_price=26177.15,
        xgboost_regime="TRENDING_UP",
        market_bias="BULLISH",
        support_source="VOB Support",
        resistance_source="HTF Resistance",
        vob_major_support=26050.0,
        vob_major_resistance=26350.0,
        vob_minor_support=26075.0,
        vob_minor_resistance=26325.0,
        setup_type="LONG at VOB Support",
        entry_zone="₹26,100 - ₹26,120",
        stop_loss="₹26,030 (Below VOB Support)",
        target="₹26,300 (HTF Resistance)",
        analysis_text="Strong bullish setup with 75% confidence. Price at major VOB support with trending regime.",
        pcr=1.15,
        bullish_count=8,
        bearish_count=2,
        vix=14.2,
        expiry_status="🔥 SUPPORT SPIKE (85%)"
    )

    st.markdown("---")

    # Example 3: ENTRY Signal (SHORT)
    st.subheader("Example 3: ENTRY Signal (SHORT)")
    display_custom_signal(
        signal_type="ENTRY",
        direction="SHORT",
        confidence=72.0,
        support=26050.0,
        resistance=26250.0,
        current_price=26177.15,
        xgboost_regime="TRENDING_DOWN",
        market_bias="BEARISH",
        support_source="HTF Support",
        resistance_source="VOB Resistance",
        vob_major_support=26000.0,
        vob_major_resistance=26280.0,
        setup_type="SHORT at VOB Resistance",
        entry_zone="₹26,250 - ₹26,270",
        stop_loss="₹26,320 (Above VOB Resistance)",
        target="₹26,050 (HTF Support)",
        analysis_text="Bearish setup with 72% confidence. Price rejecting at major VOB resistance.",
        pcr=0.85,
        bullish_count=3,
        bearish_count=9,
        vix=16.8,
        expiry_status="🔥 RESISTANCE SPIKE (78%)"
    )

    st.markdown("---")

    # Example 4: EXIT Signal
    st.subheader("Example 4: EXIT Signal")
    display_custom_signal(
        signal_type="EXIT",
        direction="NEUTRAL",
        confidence=90.0,
        support=26100.0,
        resistance=26300.0,
        current_price=26177.15,
        xgboost_regime="REGIME_CHANGE",
        market_bias="NEUTRAL",
        setup_type="Exit Position",
        entry_zone="Close all positions",
        analysis_text="⚠️ SCENARIO CHANGED - Support level changed by 0.8% (from 26,100 to 26,000). Original trade thesis invalidated.",
        pcr=1.0,
        vix=18.5
    )


if __name__ == "__main__":
    example_streamlit_app()
