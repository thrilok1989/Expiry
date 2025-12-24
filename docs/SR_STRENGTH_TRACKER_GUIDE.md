# S/R Strength Tracker & ML Analysis - User Guide

## Overview

The new **Support/Resistance Strength Tracker** with **ML-based Trend Analysis** provides real-time intelligence on how support and resistance levels are evolving over time. This helps you answer critical questions like:

- ✅ Is this support level getting **stronger** or **weaker**?
- ✅ Is a support level about to **transition** into resistance (or vice versa)?
- ✅ What will the strength be in **1 hour** or **4 hours**?
- ✅ Has this level been **tested multiple times**?

---

## Key Features

### 1. **S/R Strength Trends** 📈📉

The tracker monitors every support/resistance level and analyzes:

- **Current Strength**: 0-100% score based on 10 factors
- **Trend Direction**:
  - 🟢 **STRENGTHENING**: Strength increasing (getting more reliable)
  - 🔴 **WEAKENING**: Strength decreasing (about to break)
  - 🟡 **STABLE**: No significant change
  - 🟣 **TRANSITIONING**: Changing from support to resistance or vice versa

- **Predictions**:
  - **1-hour prediction**: Expected strength in 1 hour
  - **4-hour prediction**: Expected strength in 4 hours

- **Confidence Score**: How reliable the trend analysis is (based on data points and R²)

### 2. **Support-to-Resistance Transitions** 🔄

Detects when a level changes its role:

- **SUPPORT_TO_RESISTANCE**: Price broke above former support, now acting as resistance
- **RESISTANCE_TO_SUPPORT**: Price broke below former resistance, now acting as support

Each transition includes:
- Confidence score (0-100%)
- Strength before and after transition
- Timestamp of when it happened

### 3. **Historical Tracking**

The system maintains up to **24 hours** of observations for each level, allowing:
- Pattern recognition
- Trend detection
- Touch count (how many times price tested the level)

---

## How It Works

### Data Collection

Every time the "Re-run All Analyses" button is clicked, the system:

1. **Analyzes** current S/R strength using 10 factors:
   - Price action
   - Volume confirmation
   - Delta/Flow (CVD)
   - Gamma Exposure (GEX)
   - Market Depth
   - OI Buildup
   - Regime Strength
   - Participant Analysis
   - Liquidity Features
   - Expiry Context

2. **Records** the observation with:
   - Timestamp
   - Price level
   - Strength score
   - Status (BUILDING/TESTING/BREAKING/NEUTRAL)
   - Contributing factors

3. **Analyzes** trends using **linear regression** on historical strength values

4. **Detects** transitions by comparing old vs. new role of a level

### ML Algorithm

The trend analysis uses:

```
Simple Linear Regression on strength values over time:
  strength = slope × time + intercept

Where:
  - slope = strength change rate (pts/hour)
  - R² = confidence in the trend
  - predictions = extrapolate slope into future
```

Transition detection looks for:
- Level type change (support → resistance or vice versa)
- Sufficient confidence (strength after transition > 60% of strength before)
- Non-duplicate (not already detected in last hour)

---

## Where to Find It

### In Tab 1: Overall Market Sentiment

After the "Session Intelligence" section, you'll see:

```
🔍 S/R Strength Trends (ML Analysis)
───────────────────────────────────────

Support @ ₹26,100
📈🟢 STRENGTHENING
Strength: 75% → 80% (1h)

Resistance @ ₹26,300
📉🔴 WEAKENING
Strength: 70% → 65% (1h)

⚠️ 1 S/R transition(s) detected in last 2 hours!
```

### In Collapsible View (Future Enhancement)

The full detailed analysis is available in the expandable "S/R Strength Trends & Transitions" section.

---

## Interpreting the Signals

### Strength Scores

| Score | Meaning | Action |
|-------|---------|--------|
| 80-100% | **Very Strong** | High-confidence entry at this level |
| 60-79% | **Strong** | Good entry with proper stop loss |
| 40-59% | **Moderate** | Use caution, may break |
| 20-39% | **Weak** | Likely to break, avoid entry |
| 0-19% | **Very Weak** | Level breaking down |

### Trend Directions

- **STRENGTHENING (📈🟢)**:
  - More buyers defending support OR more sellers defending resistance
  - Strength increasing over time
  - **Action**: Prepare for bounce/rejection at this level

- **WEAKENING (📉🔴)**:
  - Level losing conviction
  - Strength decreasing over time
  - **Action**: Expect breakout if price reaches this level

- **STABLE (➡️🟡)**:
  - No significant change
  - Level holding steady
  - **Action**: Business as usual, respect the level

- **TRANSITIONING (🔄🟣)**:
  - Role reversal happening
  - Former support becoming resistance or vice versa
  - **Action**: Trade the new role, not the old one

### Transitions

When you see a transition alert:

```
🔴 SUPPORT_TO_RESISTANCE @ ₹26,150 (Confidence: 75%)
Level changed from support to resistance (confidence: 75%)
```

**What it means**:
- Price was bouncing off ₹26,150 as support
- Price broke above it
- Now ₹26,150 is acting as resistance on retests

**How to trade**:
- If price pulls back to ₹26,150, expect **rejection** (not bounce)
- Short entries near ₹26,150 have higher probability
- Don't try to buy support at ₹26,150 anymore

---

## Best Practices

### 1. Wait for Data Accumulation

- First few runs: Not enough data
- After 3-5 refreshes: Trends start appearing
- After 1 hour: Reliable predictions
- After 4 hours: High-confidence analysis

### 2. Combine with Other Signals

Don't use S/R trends in isolation. Combine with:
- ATM Bias
- OI/PCR Analysis
- Volume Confirmation
- ML Regime
- VIX levels

### 3. Trust High-Confidence Trends

- **Confidence > 70%**: Act on the trend
- **Confidence 50-70%**: Use as additional confirmation
- **Confidence < 50%**: Ignore, insufficient data

### 4. Watch for Transitions

Transitions are **critical**. They tell you when the market has **changed the rules**.

Former support becoming resistance is a **bearish** signal.
Former resistance becoming support is a **bullish** signal.

### 5. Use Predictions Wisely

The 1h and 4h predictions help you:
- Anticipate if a level will strengthen before price reaches it
- Plan exits if support is weakening
- Time entries when resistance is weakening

---

## Technical Details

### Files Added

1. **`src/sr_strength_tracker.py`**:
   - Core ML tracker
   - Classes: `SRStrengthPoint`, `SRTransition`, `SRStrengthTrend`, `SRStrengthTracker`
   - Functions: trend analysis, transition detection, historical tracking

2. **`src/sr_integration.py`**:
   - Integration with existing comprehensive S/R analysis
   - Session state management
   - Display helpers

3. **`src/collapsible_signal_ui.py`**:
   - New UI component for organized signal display
   - Collapsible sections for detailed analysis
   - Clean, structured presentation

### Session State

Data is stored in:
```python
st.session_state.sr_strength_tracker
```

Persists across page refreshes within the same session.

### Data Retention

- Observations: Last 500 entries
- Time window: 24 hours
- Automatic cleanup of old data

---

## Example Scenarios

### Scenario 1: Support Strengthening Before Price Reaches It

```
Current: ₹26,150
Support: ₹26,100 (50 pts below)

Support Analysis:
- Current Strength: 70%
- Trend: STRENGTHENING (+5 pts/hr)
- Prediction 1h: 75%

Action:
✅ Price likely to bounce at ₹26,100
✅ Prepare LONG entry at support
✅ High probability setup
```

### Scenario 2: Resistance Weakening (Breakout Coming)

```
Current: ₹26,280
Resistance: ₹26,300 (20 pts above)

Resistance Analysis:
- Current Strength: 65%
- Trend: WEAKENING (-8 pts/hr)
- Prediction 1h: 57%

Action:
⚠️ Resistance losing strength
✅ Breakout likely if price reaches ₹26,300
✅ Consider LONG breakout setup
```

### Scenario 3: Support-to-Resistance Transition

```
Previous: Support @ ₹26,200 (Strength: 75%)
Now: Resistance @ ₹26,200 (Strength: 70%)

Transition Detected:
🔴 SUPPORT_TO_RESISTANCE (Confidence: 80%)

Action:
❌ Don't buy at ₹26,200 expecting support
✅ SHORT entries at ₹26,200 on retest
✅ Former support = new resistance
```

---

## Troubleshooting

### "S/R trend analysis initializing..."

**Reason**: Not enough historical data yet

**Solution**: Run "Re-run All Analyses" 3-5 times over 15-30 minutes

### "No trend data available yet"

**Reason**: Price hasn't tested the S/R levels recently

**Solution**: Wait for price to approach support/resistance

### Low Confidence Scores

**Reason**: Inconsistent strength readings or limited data

**Solution**:
- Wait for more observations
- Check if market is choppy (low regime confidence)

### No Transitions Detected

**Reason**: Levels haven't broken and reversed roles

**Solution**: This is normal. Transitions are relatively rare events.

---

## Future Enhancements

Planned improvements:

1. **Volume-weighted strength**: Give more weight to high-volume tests
2. **Multi-level tracking**: Track up to 10 levels simultaneously
3. **Alert system**: Notify when transitions are imminent
4. **Strength heatmap**: Visual representation of all tracked levels
5. **Historical playback**: Review past transitions and their outcomes
6. **Export data**: Download S/R history for external analysis

---

## Summary

The S/R Strength Tracker transforms static support/resistance levels into **dynamic, intelligent zones** that evolve with market conditions. By tracking strength trends and detecting transitions, you gain a **significant edge** in timing entries and avoiding false setups.

**Key Takeaway**: Support and resistance are not fixed lines. They strengthen, weaken, and change roles. Now you can **see it happening** in real-time.

---

**Questions or Issues?**

Check the logs for detailed debugging:
```python
logger.info("S/R Strength Tracker logs")
```

Or examine session state:
```python
st.session_state.sr_strength_tracker.get_summary_stats()
```
