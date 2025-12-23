# 🎯 Signal Display HTML Generator - Usage Guide

## Overview
Clean, professional HTML-based trading signal display for Streamlit apps.

## Files
1. **`src/signal_display_html_generator.py`** - Core HTML generation logic
2. **`src/streamlit_signal_display.py`** - Streamlit integration wrapper

---

## Quick Start

### Option 1: Using TradingSignal Object (Recommended)

```python
import streamlit as st
from src.streamlit_signal_display import display_signal_in_streamlit
from src.enhanced_signal_generator import EnhancedSignalGenerator

# Generate signal
signal_generator = EnhancedSignalGenerator()
signal = signal_generator.generate_signal(
    xgboost_result,
    features_df,
    current_price,
    option_chain,
    atm_strike,
    vob_data=st.session_state.vob_data_nifty,
    htf_sr_data=st.session_state.htf_sr_levels
)

# Display in Streamlit
display_signal_in_streamlit(
    signal=signal,
    current_price=26177.15,
    pcr=1.15,
    bullish_count=8,
    bearish_count=2,
    vob_data=st.session_state.vob_data_nifty,
    regime_data={
        'is_expiry_week': True,
        'volatility_regime': 'HIGH_VOLATILITY',
        'vix_level': 18.5,
        'expiry_spike_detected': True,
        'expiry_spike_type': 'SUPPORT SPIKE',
        'expiry_spike_probability': 85
    },
    zone_width=200.0
)
```

---

### Option 2: Custom Display (Simple)

```python
from src.streamlit_signal_display import display_custom_signal

display_custom_signal(
    signal_type="ENTRY",  # "ENTRY", "EXIT", "WAIT"
    direction="LONG",     # "LONG", "SHORT", "NEUTRAL"
    confidence=75.0,
    support=26100.0,
    resistance=26300.0,
    current_price=26177.15,

    # Optional parameters
    xgboost_regime="TRENDING_UP",
    market_bias="BULLISH",
    support_source="VOB Support",
    resistance_source="HTF Resistance",
    vob_major_support=26050.0,
    vob_major_resistance=26350.0,
    setup_type="LONG at VOB Support",
    entry_zone="₹26,100 - ₹26,120",
    stop_loss="₹26,030 (Below VOB Support - 20 pts buffer)",
    target="₹26,300 (HTF Resistance)",
    analysis_text="Strong bullish setup. Price at major VOB support.",
    pcr=1.15,
    vix=14.2,
    expiry_status="🔥 SUPPORT SPIKE (85%)"
)
```

---

### Option 3: Using Dictionary

```python
from src.signal_display_html_generator import generate_signal_html_from_dict
import streamlit as st

signal_dict = {
    'signal_type': 'WAIT',
    'direction': 'NEUTRAL',
    'confidence': 55.0,
    'support_price': 26077.0,
    'resistance_price': 26277.0,
    'current_price': 26177.15,
    'xgboost_regime': 'RANGING',
    'market_bias': 'NEUTRAL',
    'zone_width': 200.0,
    'analysis_text': 'Low confidence. Wait for better setup.'
}

html = generate_signal_html_from_dict(signal_dict)
st.markdown(html, unsafe_allow_html=True)
```

---

## Signal Types

### 1. WAIT Signal (Neutral)
```python
display_custom_signal(
    signal_type="WAIT",
    direction="NEUTRAL",
    confidence=55.0,
    support=26077.0,
    resistance=26277.0,
    current_price=26177.15,
    analysis_text="Low confidence or price in mid-zone. Wait for better setup."
)
```

**Displays:**
- 🔴 WAIT - ⚖️ NEUTRAL
- Orange/Red color scheme
- "No Setup" entry type

---

### 2. ENTRY Signal (LONG)
```python
display_custom_signal(
    signal_type="ENTRY",
    direction="LONG",
    confidence=75.0,
    support=26100.0,
    resistance=26300.0,
    current_price=26177.15,
    stop_loss="₹26,030 (Below VOB Support - 20 pts buffer)",
    target="₹26,300 (HTF Resistance)",
    analysis_text="Strong bullish setup at VOB support."
)
```

**Displays:**
- 🟢 ENTRY - 🟢 LONG
- Green color scheme
- Entry zone, stop-loss, targets

---

### 3. ENTRY Signal (SHORT)
```python
display_custom_signal(
    signal_type="ENTRY",
    direction="SHORT",
    confidence=72.0,
    support=26050.0,
    resistance=26250.0,
    current_price=26177.15,
    stop_loss="₹26,320 (Above VOB Resistance + 20 pts buffer)",
    target="₹26,050 (HTF Support)",
    analysis_text="Bearish setup. Price rejecting at VOB resistance."
)
```

**Displays:**
- 🔴 ENTRY - 🔴 SHORT
- Red color scheme
- Entry zone, stop-loss, targets

---

### 4. EXIT Signal
```python
display_custom_signal(
    signal_type="EXIT",
    direction="NEUTRAL",
    confidence=90.0,
    support=26100.0,
    resistance=26300.0,
    current_price=26177.15,
    analysis_text="⚠️ SCENARIO CHANGED - Support level changed by 0.8%. Exit position."
)
```

**Displays:**
- 🚪 EXIT - ⚖️ NEUTRAL
- Red color scheme
- Exit reason prominently displayed

---

## Integration with Existing Code

### In `app.py` or main Streamlit file:

```python
import streamlit as st
from src.streamlit_signal_display import display_signal_in_streamlit
from src.enhanced_signal_generator import EnhancedSignalGenerator

# ... your existing code ...

# Generate XGBoost signal
signal_generator = EnhancedSignalGenerator()
signal = signal_generator.generate_signal(
    xgboost_result=ml_result,
    features_df=features_df,
    current_price=current_price,
    option_chain=option_chain_data,
    atm_strike=atm_strike,
    vob_data=st.session_state.get('vob_data_nifty'),  # VOB data for S/R
    htf_sr_data=st.session_state.get('htf_sr_levels')  # HTF S/R data
)

# Display signal with beautiful HTML
st.markdown("### 🎯 AI Trading Signal")
display_signal_in_streamlit(
    signal=signal,
    current_price=current_price,
    pcr=pcr_value,
    bullish_count=num_bullish_indicators,
    bearish_count=num_bearish_indicators,
    vob_data=st.session_state.get('vob_data_nifty'),
    regime_data=volatility_regime_result.__dict__ if volatility_regime_result else None,
    zone_width=abs(resistance - support)
)
```

---

## Key Features

### 1. **Support/Resistance Tracking**
- Shows exact support/resistance levels
- Indicates source (VOB, HTF, Calculated)
- Displays distance from current price

### 2. **VOB Level Display**
- Major/Minor support
- Major/Minor resistance
- Automatically extracted from `vob_data`

### 3. **Expiry Spike Integration**
- Shows expiry status
- Displays spike type (SUPPORT SPIKE / RESISTANCE SPIKE)
- Includes spike probability

### 4. **Stop-Loss Reasoning**
- Shows why stop-loss was calculated
- Examples:
  - "Below VOB Support (23,450 - 20 pts buffer)"
  - "Above HTF Resistance (23,580 + 20 pts buffer)"
  - "Percentage-based (25% of entry, VIX=18.5)"

### 5. **Dynamic Color Coding**
- LONG entries: Green theme
- SHORT entries: Red theme
- WAIT signals: Orange/Red theme
- EXIT signals: Red theme with warning

---

## Customization

### Color Schemes
Edit colors in `src/signal_display_html_generator.py`:

```python
# Header colors
header_color = "#00ff88"  # Green for LONG
header_color = "#ff4444"  # Red for SHORT/EXIT/WAIT

# Confidence colors
confidence_color = "#00ff88"  # High confidence (>70%)
confidence_color = "#6495ED"  # Medium confidence (50-70%)
confidence_color = "#ff4444"  # Low confidence (<50%)
```

### Zone Width Status
```python
if zone_width < 100:
    zone_width_status = "NARROW"  # Green
elif zone_width < 200:
    zone_width_status = "MODERATE"  # Orange
else:
    zone_width_status = "WIDE"  # Red
```

---

## Testing

Run the example app:

```bash
# Test HTML generator
python src/signal_display_html_generator.py

# Test Streamlit integration
streamlit run src/streamlit_signal_display.py
```

---

## Benefits

✅ **Professional Display** - Clean, dark theme UI
✅ **Dynamic Updates** - Real-time signal changes
✅ **Complete Info** - All trading details in one view
✅ **Stop-Loss Transparency** - Shows why SL was calculated
✅ **Expiry Spike Awareness** - Highlights support/resistance spikes
✅ **Easy Integration** - Drop-in replacement for existing signal display

---

## Example Output

### WAIT Signal:
```
🔴 WAIT - ⚖️ NEUTRAL

Confidence: 55%
XGBoost Regime: RANGING
Market Bias: ⚖️ NEUTRAL
Zone Width: 200 pts (WIDE)

Support: ₹26,077 (Calculated, 100 pts away)
Resistance: ₹26,277 (Calculated, 100 pts away)

VOB LEVELS
Major Support: N/A | Major Resistance: N/A

ENTRY SETUP
Setup Type: No Setup
Entry Zone: Wait for clear direction
Stop Loss: N/A
Target: N/A

Analysis: Low confidence or price in mid-zone. Wait for better setup.
```

### LONG ENTRY Signal:
```
🟢 ENTRY - 🟢 LONG

Confidence: 75%
XGBoost Regime: TRENDING_UP
Market Bias: 🟢 BULLISH
Zone Width: 50 pts (NARROW)

Support: ₹26,100 (VOB Support, 77 pts away)
Resistance: ₹26,300 (HTF Resistance, 123 pts away)

VOB LEVELS
Major Support: ₹26,050 | Major Resistance: ₹26,350

ENTRY SETUP
Setup Type: LONG at VOB Support
Entry Zone: ₹26,100 - ₹26,120
Stop Loss: ₹26,030 (Below VOB Support - 20 pts buffer)
Target: ₹26,300 (HTF Resistance)

Expiry: 🔥 SUPPORT SPIKE (85%)

Analysis: Strong bullish setup. Price at major VOB support with trending regime.
```

---

## Support

For issues or questions, refer to:
- `src/signal_display_html_generator.py` - Core logic
- `src/streamlit_signal_display.py` - Streamlit integration
- `src/enhanced_signal_generator.py` - Signal generation
- `src/dynamic_stoploss_tracker.py` - Stop-loss tracking
