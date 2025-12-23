"""
Signal Display HTML Generator
Converts trading signal data to formatted HTML display
"""

from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class SignalDisplayData:
    """Data structure for signal display"""
    # Header
    signal_type: str  # "WAIT", "ENTRY", "EXIT"
    direction: str  # "NEUTRAL", "LONG", "SHORT"

    # Core Metrics
    confidence: float  # 0-100
    xgboost_regime: str  # "RANGING", "TRENDING_UP", "TRENDING_DOWN"
    market_bias: str  # "NEUTRAL", "BULLISH", "BEARISH"
    zone_width: float  # Points
    zone_width_status: str  # "NARROW", "MODERATE", "WIDE"

    # Support & Resistance
    support_price: float
    support_source: str  # "Calculated", "VOB Support", "HTF Support"
    support_distance: float  # Points away
    resistance_price: float
    resistance_source: str
    resistance_distance: float

    # VOB Levels
    vob_major_support: Optional[float] = None
    vob_major_resistance: Optional[float] = None
    vob_minor_support: Optional[float] = None
    vob_minor_resistance: Optional[float] = None

    # Entry Setup
    setup_type: str = "No Setup"
    entry_zone: str = "Wait for clear direction"
    stop_loss: str = "N/A"
    target: str = "N/A"

    # Additional Info
    expiry_status: str = "⚠️ HIGH VOLATILITY"
    gex_level: str = "Neutral"
    vix: float = 15.0

    # Analysis
    analysis_text: str = "Low confidence or price in mid-zone. Wait for better setup."

    # Footer
    current_price: float = 0.0
    pcr: float = 1.0
    bullish_count: int = 0
    bearish_count: int = 0


def generate_signal_html(data: SignalDisplayData) -> str:
    """
    Generate HTML for signal display

    Args:
        data: SignalDisplayData object with all signal information

    Returns:
        HTML string for display
    """

    # Determine signal color based on type and direction
    if data.signal_type == "ENTRY":
        if data.direction == "LONG":
            header_color = "#00ff88"  # Green
            signal_emoji = "🟢"
        elif data.direction == "SHORT":
            header_color = "#ff4444"  # Red
            signal_emoji = "🔴"
        else:
            header_color = "#ffa500"  # Orange
            signal_emoji = "⚖️"
    elif data.signal_type == "EXIT":
        header_color = "#ff4444"
        signal_emoji = "🚪"
    else:  # WAIT
        header_color = "#ff4444"
        signal_emoji = "🔴"

    # Confidence color
    if data.confidence >= 70:
        confidence_color = "#00ff88"
    elif data.confidence >= 50:
        confidence_color = "#6495ED"
    else:
        confidence_color = "#ff4444"

    # Zone width color
    zone_color = "#ff4444" if data.zone_width_status == "WIDE" else ("#ffa500" if data.zone_width_status == "MODERATE" else "#00ff88")

    # Market bias color and emoji
    if data.market_bias == "BULLISH":
        bias_color = "#00ff88"
        bias_emoji = "🟢"
    elif data.market_bias == "BEARISH":
        bias_color = "#ff4444"
        bias_emoji = "🔴"
    else:
        bias_color = "#ffa500"
        bias_emoji = "⚖️"

    # VOB level display
    vob_major_sup = f"₹{data.vob_major_support:,.0f}" if data.vob_major_support else "N/A"
    vob_major_res = f"₹{data.vob_major_resistance:,.0f}" if data.vob_major_resistance else "N/A"
    vob_minor_sup = f"₹{data.vob_minor_support:,.0f}" if data.vob_minor_support else "N/A"
    vob_minor_res = f"₹{data.vob_minor_resistance:,.0f}" if data.vob_minor_resistance else "N/A"

    # Entry setup background color
    if data.signal_type == "ENTRY":
        if data.direction == "LONG":
            setup_bg = "#1a4d1a"  # Dark green
            setup_border = "#00ff88"
        elif data.direction == "SHORT":
            setup_bg = "#4d1a1a"  # Dark red
            setup_border = "#ff4444"
        else:
            setup_bg = "#1a1a2e"  # Dark blue
            setup_border = "#ffa500"
    else:
        setup_bg = "#1a4d1a"
        setup_border = "#ffa500"

    html = f"""
<h1 style='margin: 0 0 20px 0; color: {header_color}; font-size: 36px;'>
    {signal_emoji} {data.signal_type} - {bias_emoji} {data.direction}
</h1>

<!-- Core Metrics -->
<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0;'>
    <div style='background: #0a0a0a; padding: 15px; border-radius: 10px; text-align: center;'>
        <div style='color: {confidence_color}; font-size: 32px; font-weight: bold;'>{data.confidence:.0f}%</div>
        <div style='color: #888; font-size: 12px;'>Confidence</div>
    </div>

    <div style='background: #0a0a0a; padding: 15px; border-radius: 10px; text-align: center;'>
        <div style='color: #ff9500; font-size: 20px; font-weight: bold;'>{data.xgboost_regime}</div>
        <div style='color: #888; font-size: 12px;'>XGBoost Regime</div>
    </div>

    <div style='background: #0a0a0a; padding: 15px; border-radius: 10px; text-align: center;'>
        <div style='color: {bias_color}; font-size: 18px; font-weight: bold;'>{bias_emoji} {data.market_bias}</div>
        <div style='color: #888; font-size: 12px;'>Market Bias</div>
    </div>

    <div style='background: #0a0a0a; padding: 15px; border-radius: 10px; text-align: center;'>
        <div style='color: {zone_color}; font-size: 20px; font-weight: bold;'>{data.zone_width:.0f} pts</div>
        <div style='color: #888; font-size: 12px;'>Zone Width ({data.zone_width_status})</div>
    </div>
</div>

<!-- Support & Resistance -->
<div style='background: #0a0a0a; padding: 20px; border-radius: 10px; margin: 15px 0;'>
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
        <div>
            <div style='color: #00ff88; font-size: 16px; font-weight: bold; margin-bottom: 10px;'>🟢 SUPPORT</div>
            <div style='color: #00ff88; font-size: 28px; font-weight: bold;'>₹{data.support_price:,.0f}</div>
            <div style='color: #888; font-size: 13px; margin-top: 5px;'>{data.support_source}</div>
            <div style='color: #666; font-size: 13px; margin-top: 5px;'>{data.support_distance:.0f} points away</div>
        </div>
        <div style='text-align: right;'>
            <div style='color: #ff4444; font-size: 16px; font-weight: bold; margin-bottom: 10px;'>🔴 RESISTANCE</div>
            <div style='color: #ff4444; font-size: 28px; font-weight: bold;'>₹{data.resistance_price:,.0f}</div>
            <div style='color: #888; font-size: 13px; margin-top: 5px;'>{data.resistance_source}</div>
            <div style='color: #666; font-size: 13px; margin-top: 5px;'>{data.resistance_distance:.0f} points away</div>
        </div>
    </div>
</div>

<!-- VOB Levels -->
<div style='background: #0a0a0a; padding: 15px; border-radius: 10px; margin: 15px 0;'>
    <div style='color: #6495ED; font-size: 14px; font-weight: bold; margin-bottom: 10px;'>📊 VOB LEVELS</div>
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 13px;'>
        <div style='color: #00ff88;'>Major Support: {vob_major_sup}</div>
        <div style='color: #ff4444; text-align: right;'>Major Resistance: {vob_major_res}</div>
        <div style='color: #888;'>Minor Support: {vob_minor_sup}</div>
        <div style='color: #888; text-align: right;'>Minor Resistance: {vob_minor_res}</div>
    </div>
</div>

<!-- Entry Setup -->
<div style='background: {setup_bg}; padding: 20px; border-radius: 10px; margin: 15px 0; border: 2px solid {setup_border};'>
    <div style='color: #ffffff; font-size: 18px; font-weight: bold; margin-bottom: 15px;'>🎯 ENTRY SETUP</div>
    <div style='color: #ffffff; font-size: 15px; line-height: 2;'>
        <strong style='color: #6495ED;'>Setup Type:</strong> {data.setup_type}<br>
        <strong style='color: #00ff88;'>Entry Zone:</strong> {data.entry_zone}<br>
        <strong style='color: #ff4444;'>Stop Loss:</strong> {data.stop_loss}<br>
        <strong style='color: #00ff88;'>Target:</strong> {data.target}
    </div>
</div>

<!-- Additional Info -->
<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 15px 0;'>
    <div style='background: #0a0a0a; padding: 10px; border-radius: 8px; text-align: center;'>
        <div style='color: #888; font-size: 11px;'>Expiry</div>
        <div style='color: #ff4444; font-size: 14px; font-weight: bold;'>{data.expiry_status}</div>
    </div>
    <div style='background: #0a0a0a; padding: 10px; border-radius: 8px; text-align: center;'>
        <div style='color: #888; font-size: 11px;'>GEX Level</div>
        <div style='color: #ffa500; font-size: 14px; font-weight: bold;'>{data.gex_level}</div>
    </div>
    <div style='background: #0a0a0a; padding: 10px; border-radius: 8px; text-align: center;'>
        <div style='color: #888; font-size: 11px;'>VIX</div>
        <div style='color: #ffa500; font-size: 14px; font-weight: bold;'>{data.vix:.1f}</div>
    </div>
</div>

<!-- Reason -->
<div style='background: #1a1a2e; padding: 15px; border-radius: 10px; margin: 15px 0;'>
    <div style='color: #cccccc; font-size: 14px; line-height: 1.6;'>
        💡 <strong>Analysis:</strong> {data.analysis_text}
    </div>
</div>

<!-- Footer -->
<div style='color: #666; font-size: 12px; margin-top: 15px; padding-top: 15px; border-top: 1px solid #333; text-align: center;'>
    Current: ₹{data.current_price:,.2f} | PCR: {data.pcr:.2f} | Bullish: {data.bullish_count} | Bearish: {data.bearish_count}
</div>
"""

    return html


def generate_signal_html_from_dict(signal_data: Dict) -> str:
    """
    Generate HTML from dictionary (convenience function)

    Args:
        signal_data: Dictionary with signal information

    Returns:
        HTML string for display
    """
    data = SignalDisplayData(
        signal_type=signal_data.get('signal_type', 'WAIT'),
        direction=signal_data.get('direction', 'NEUTRAL'),
        confidence=signal_data.get('confidence', 55.0),
        xgboost_regime=signal_data.get('xgboost_regime', 'RANGING'),
        market_bias=signal_data.get('market_bias', 'NEUTRAL'),
        zone_width=signal_data.get('zone_width', 200.0),
        zone_width_status=signal_data.get('zone_width_status', 'WIDE'),
        support_price=signal_data.get('support_price', 26077.0),
        support_source=signal_data.get('support_source', 'Calculated'),
        support_distance=signal_data.get('support_distance', 100.0),
        resistance_price=signal_data.get('resistance_price', 26277.0),
        resistance_source=signal_data.get('resistance_source', 'Calculated'),
        resistance_distance=signal_data.get('resistance_distance', 100.0),
        vob_major_support=signal_data.get('vob_major_support'),
        vob_major_resistance=signal_data.get('vob_major_resistance'),
        vob_minor_support=signal_data.get('vob_minor_support'),
        vob_minor_resistance=signal_data.get('vob_minor_resistance'),
        setup_type=signal_data.get('setup_type', 'No Setup'),
        entry_zone=signal_data.get('entry_zone', 'Wait for clear direction'),
        stop_loss=signal_data.get('stop_loss', 'N/A'),
        target=signal_data.get('target', 'N/A'),
        expiry_status=signal_data.get('expiry_status', '⚠️ HIGH VOLATILITY'),
        gex_level=signal_data.get('gex_level', 'Neutral'),
        vix=signal_data.get('vix', 15.0),
        analysis_text=signal_data.get('analysis_text', 'Low confidence or price in mid-zone. Wait for better setup.'),
        current_price=signal_data.get('current_price', 26177.15),
        pcr=signal_data.get('pcr', 1.0),
        bullish_count=signal_data.get('bullish_count', 0),
        bearish_count=signal_data.get('bearish_count', 0)
    )

    return generate_signal_html(data)


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Example 1: Using SignalDisplayData dataclass
    signal_data = SignalDisplayData(
        signal_type="WAIT",
        direction="NEUTRAL",
        confidence=55.0,
        xgboost_regime="RANGING",
        market_bias="NEUTRAL",
        zone_width=200.0,
        zone_width_status="WIDE",
        support_price=26077.0,
        support_source="Calculated",
        support_distance=100.0,
        resistance_price=26277.0,
        resistance_source="Calculated",
        resistance_distance=100.0,
        vob_major_support=None,
        vob_major_resistance=None,
        vob_minor_support=None,
        vob_minor_resistance=None,
        setup_type="No Setup",
        entry_zone="Wait for clear direction",
        stop_loss="N/A",
        target="N/A",
        expiry_status="⚠️ HIGH VOLATILITY",
        gex_level="Neutral",
        vix=15.0,
        analysis_text="Low confidence or price in mid-zone. Wait for better setup.",
        current_price=26177.15,
        pcr=1.0,
        bullish_count=0,
        bearish_count=0
    )

    html_output = generate_signal_html(signal_data)
    print(html_output)

    # Example 2: Using dictionary
    signal_dict = {
        'signal_type': 'ENTRY',
        'direction': 'LONG',
        'confidence': 75.0,
        'xgboost_regime': 'TRENDING_UP',
        'market_bias': 'BULLISH',
        'zone_width': 50.0,
        'zone_width_status': 'NARROW',
        'support_price': 26100.0,
        'support_source': 'VOB Support',
        'support_distance': 77.0,
        'resistance_price': 26300.0,
        'resistance_source': 'HTF Resistance',
        'resistance_distance': 123.0,
        'vob_major_support': 26050.0,
        'vob_major_resistance': 26350.0,
        'vob_minor_support': 26075.0,
        'vob_minor_resistance': 26325.0,
        'setup_type': 'LONG at Support',
        'entry_zone': '₹26,100 - ₹26,120',
        'stop_loss': '₹26,030 (Below VOB Support)',
        'target': '₹26,300 (HTF Resistance)',
        'expiry_status': '✅ NORMAL',
        'gex_level': 'Bullish',
        'vix': 14.2,
        'analysis_text': 'Strong bullish setup with 75% confidence. Price at major VOB support with trending regime.',
        'current_price': 26177.15,
        'pcr': 1.15,
        'bullish_count': 8,
        'bearish_count': 2
    }

    html_output_2 = generate_signal_html_from_dict(signal_dict)
    print("\n\n=== LONG ENTRY EXAMPLE ===\n")
    print(html_output_2)
