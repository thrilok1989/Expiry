# Console Signal Display for NIFTY/BANKNIFTY Trading Signals

## Overview

This module provides Python-based console/terminal formatting for NIFTY and BANKNIFTY trading signals, replacing HTML-based displays with clean, colored terminal output.

## Features

✅ **No HTML** - Pure Python console output
✅ **Color-coded** - ANSI colors for better readability in terminals
✅ **Complete Signal Info** - Entry, Stop Loss, Target, Risk:Reward, VOB levels
✅ **Strength Analysis** - Order Block strength metrics (score, trend, respect rate)
✅ **Easy Integration** - Works with existing signal generators

## Files

- `console_signal_display.py` - Main module with formatting functions
- `simple_console_signal_demo.py` - Simple demo showing your exact signal
- `example_console_signal_usage.py` - Advanced integration examples

## Quick Start

### 1. Basic Usage

```python
from console_signal_display import print_signal

# Your signal data
signal = {
    'index': 'NIFTY',
    'direction': 'PUT',  # or 'CALL'
    'market_sentiment': 'BEARISH',  # or 'BULLISH'
    'entry_price': 26140.0,
    'stop_loss': 26157.85,
    'target': 26113.22,
    'risk_reward': '1:1.5',
    'vob_level': 26141.52,
    'distance_from_vob': 1.52,
    'timestamp': '14:55:11',
    'strength': {  # Optional
        'strength_score': 62.7,
        'strength_label': 'MODERATE',
        'trend': 'WEAKENING',
        'times_tested': 22,
        'respect_rate': 81.8
    }
}

# Print to console
print_signal(signal)
```

### 2. Simple Function (without creating dict)

```python
from console_signal_display import format_simple_signal_console

output = format_simple_signal_console(
    index="NIFTY",
    direction="PUT",
    sentiment="BEARISH",
    entry_price=26140.0,
    stop_loss=26157.85,
    target=26113.22,
    risk_reward="1:1.5",
    vob_level=26141.52,
    distance_from_vob=1.52,
    signal_time="14:55:11",
    # Optional strength parameters
    strength_score=62.7,
    strength_status="MODERATE",
    trend="WEAKENING",
    times_tested=22,
    respect_rate=81.8
)

print(output)
```

### 3. Integration with VOB Signal Generator

```python
from vob_signal_generator import VOBSignalGenerator
from console_signal_display import print_signal

# Generate signal
vob_gen = VOBSignalGenerator()
signal = vob_gen.check_for_signal(
    spot_price=26140.0,
    market_sentiment="BEARISH",
    bullish_blocks=[],
    bearish_blocks=bearish_blocks,
    df=df  # Pass dataframe for strength analysis
)

# Display in console
if signal:
    print_signal(signal)
```

## Output Format

The console display shows:

1. **Header** - Signal type (BULLISH/BEARISH) with color coding
2. **Entry Levels** - Entry price, Stop Loss, Target, Risk:Reward
3. **VOB Details** - VOB level and distance from current price
4. **Signal Time** - Time when signal was generated
5. **Strength Analysis** (if available):
   - Strength Score (0-100) with color coding
   - Status (WEAK/MODERATE/STRONG)
   - Trend (STRENGTHENING/WEAKENING/STABLE) with emoji
   - Tests and Respect Rate

## Color Coding

- **🟢 Green** - Bullish signals, targets, strong strength
- **🔴 Red** - Bearish signals, stop loss, weak strength
- **🟡 Yellow** - Moderate strength
- **🔵 Cyan** - VOB levels
- **Bold** - Headers and important values

## Example Output

```
======================================================================
🔴 NIFTY BEARISH ENTRY SIGNAL
Market Sentiment: BEARISH
======================================================================

Entry Price
  ₹26,140.00

Stop Loss
  ₹26,157.85

Target
  ₹26,113.22

Risk:Reward
  1:1.5

VOB Level
  ₹26,141.52

Distance from VOB
  1.52 points

Signal Time
  14:55:11

----------------------------------------------------------------------
📊 Order Block Strength Analysis

  Strength Score:      62.7/100
  Status:              MODERATE
  Trend:               🔻 WEAKENING
  Tests / Respect Rate: 22 / 81.8%

======================================================================
```

## Signal Dictionary Structure

### Required Fields

```python
{
    'index': str,              # "NIFTY", "BANKNIFTY", etc.
    'direction': str,          # "CALL" or "PUT"
    'market_sentiment': str,   # "BULLISH" or "BEARISH"
    'entry_price': float,      # Entry price in rupees
    'stop_loss': float,        # Stop loss price
    'target': float,           # Target price
    'risk_reward': str,        # e.g., "1:1.5"
    'vob_level': float,        # Volume Order Block level
    'distance_from_vob': float,# Distance in points
    'timestamp': str or datetime  # Signal time
}
```

### Optional Fields

```python
{
    'strength': {              # Order Block Strength Analysis
        'strength_score': float,     # 0-100
        'strength_label': str,       # "WEAK", "MODERATE", "STRONG"
        'trend': str,                # "STRENGTHENING", "WEAKENING", "STABLE"
        'times_tested': int,         # Number of times tested
        'respect_rate': float        # Respect rate percentage
    }
}
```

## Running Demos

```bash
# Simple demo (your exact signal)
python simple_console_signal_demo.py

# Full examples with all features
python console_signal_display.py
```

## Terminal Support

Works best in terminals that support ANSI color codes:
- Linux/Mac terminals (default support)
- Windows Terminal
- VS Code integrated terminal
- iTerm2 (Mac)
- Most modern terminal emulators

## Converting from HTML

**Before (HTML):**
```html
<hr style='margin: 10px 0;'>
<div style='background-color: rgba(0,0,0,0.05); padding: 10px; border-radius: 5px;'>
    <p style='margin: 0; font-size: 14px; font-weight: bold;'>📊 Order Block Strength Analysis</p>
    <!-- ... more HTML ... -->
</div>
```

**After (Python):**
```python
from console_signal_display import print_signal

print_signal(signal_dict)
```

Much cleaner and no HTML required! 🎉

## Notes

- ANSI color codes are automatically used - no configuration needed
- Colors will display in compatible terminals
- In non-compatible terminals, you'll see the text without colors (still readable)
- The module is completely standalone - no external dependencies beyond Python standard library

## License

Part of the Expiry trading system.
