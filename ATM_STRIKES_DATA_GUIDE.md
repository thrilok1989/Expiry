# ATM ±2 Strikes Data - Complete Guide

## Overview
The ATM ±2 Strikes tabulation analyzes **5 strikes** (ATM strike ±2 strikes above and below) with **12 bias metrics** each to determine market sentiment.

## 12 Bias Metrics Explained

### 1. **OI (Open Interest Bias)**
- **Formula:** PE_OI / CE_OI ratio
- **Interpretation:**
  - Ratio > 1.3 = 🐂 BULLISH (More put writers, expecting upside)
  - Ratio < 0.77 = 🐻 BEARISH (More call writers, expecting downside)
  - Otherwise = ⚖️ NEUTRAL
- **Example:** "PE/CE OI: 1.45" means 45% more put OI

### 2. **ChgOI (Change in OI Bias)**
- **Formula:** Compares PE and CE OI changes
- **Interpretation:**
  - PE change > CE change (by 20%) = 🐂 BULLISH (Put writing increasing)
  - CE change > PE change (by 20%) = 🐻 BEARISH (Call writing increasing)
  - Similar changes = ⚖️ NEUTRAL
- **Example:** "CE:50,000 PE:75,000" shows put OI building faster

### 3. **Volume (Volume Bias)**
- **Formula:** PE_Vol / CE_Vol ratio
- **Interpretation:**
  - Ratio > 1.2 = 🐂 BULLISH (More put activity)
  - Ratio < 0.83 = 🐻 BEARISH (More call activity)
  - Otherwise = ⚖️ NEUTRAL
- **Example:** "PE/CE Vol: 1.35" means 35% more put volume

### 4. **Delta (Delta Bias)**
- **Formula:** Position-based delta analysis
- **Interpretation:**
  - Below ATM: PE OI > CE OI = 🐂 BULLISH (Support building)
  - Above ATM: CE OI > PE OI = 🐻 BEARISH (Resistance building)
  - ATM: Requires 20% difference for bias
- **Example:** "Position: ITM" indicates strike is near ATM

### 5. **Gamma (Gamma Bias)**
- **Formula:** OI concentration at ATM
- **Interpretation:**
  - At ATM with PE > CE = 🐂 BULLISH (Strong support)
  - At ATM with CE > PE = 🐻 BEARISH (Strong resistance)
  - Away from ATM = Weaker signal (±0.5)
- **Example:** "ATM Distance: 0" means this IS the ATM strike

### 6. **Premium (Premium Bias)**
- **Formula:** PE_LTP / CE_LTP ratio
- **Interpretation:**
  - Ratio > 1.5 = 🐂 BULLISH (Puts expensive, fear premium)
  - Ratio < 0.67 = 🐻 BEARISH (Calls expensive, greed premium)
  - Otherwise = ⚖️ NEUTRAL
- **Example:** "PE/CE Premium: 1.75" means puts are 75% more expensive

### 7. **IV (Implied Volatility Bias)**
- **Formula:** PE_IV - CE_IV difference
- **Interpretation:**
  - Difference > 2% = 🐂 BULLISH (Put volatility higher)
  - Difference < -2% = 🐻 BEARISH (Call volatility higher)
  - Otherwise = ⚖️ NEUTRAL
- **Example:** "PE-CE IV: 3.5%" shows put IV is 3.5% higher

### 8. **ΔExp (Delta Exposure)**
- **Formula:** Net delta = (CE_OI × 0.5) + (PE_OI × -0.5)
- **Interpretation:**
  - Net Delta > 0 = 🐂 BULLISH (Net long delta)
  - Net Delta < 0 = 🐻 BEARISH (Net short delta)
  - Net Delta = 0 = ⚖️ NEUTRAL
- **Example:** "Net ΔExp: 25,000" means net long delta exposure

### 9. **γExp (Gamma Exposure)**
- **Formula:** Net gamma = CE_GammaExp - PE_GammaExp
- **Interpretation:**
  - Net Gamma > 0 = 🐻 BEARISH (Dealers long gamma, will sell rallies)
  - Net Gamma < 0 = 🐂 BULLISH (Dealers short gamma, will buy dips)
  - Net Gamma = 0 = ⚖️ NEUTRAL
- **Example:** "Net γExp: -15,000" indicates negative gamma (bullish)

### 10. **IVSkew (IV Skew Bias)**
- **Formula:** Average IV level at strike
- **Interpretation:**
  - Avg IV > 18% = 🐻 BEARISH (High fear)
  - Avg IV < 12% = 🐂 BULLISH (Low fear, complacency)
  - Otherwise = ⚖️ NEUTRAL
- **Example:** "Avg IV: 15.5%" shows moderate volatility

### 11. **OIRate (OI Change Rate)**
- **Formula:** (Total OI Change / Total OI) × 100
- **Interpretation:**
  - Rate > 5% with PE > CE change = 🐂 BULLISH (Rapid put building)
  - Rate > 5% with CE > PE change = 🐻 BEARISH (Rapid call building)
  - Rate < 5% = ⚖️ NEUTRAL (Slow OI change)
- **Example:** "Chg Rate: 7.5%" shows high activity

### 12. **PCR (Put-Call Ratio at Strike)**
- **Formula:** PE_OI / CE_OI at specific strike
- **Interpretation:**
  - PCR > 1.5 = 🐂 BULLISH (Strong put base)
  - PCR < 0.67 = 🐻 BEARISH (Strong call base)
  - Otherwise = ⚖️ NEUTRAL
- **Example:** "Strike PCR: 1.85" shows put dominance

## Overall Verdict Calculation

The verdict for each strike is based on the **sum of all 12 metric scores**:

- **Total Score ≥ +3.0** = 🐂 STRONG BULLISH (Green #00FF00)
- **Total Score ≥ +1.0** = 🐂 Bullish (Light Green #90EE90)
- **Total Score ≤ -3.0** = 🐻 STRONG BEARISH (Red #FF0000)
- **Total Score ≤ -1.0** = 🐻 Bearish (Light Red #FFA07A)
- **Total Score between -1.0 and +1.0** = ⚖️ Neutral (Gold #FFD700)

## ATM Strike Summary

The **ATM Strike** gets special focus with:

1. **ATM Strike Verdict** - Overall bias based on total score
2. **Bullish Metrics Count** - How many of 12 metrics are bullish (score > 0)
3. **Bearish Metrics Count** - How many of 12 metrics are bearish (score < 0)
4. **Neutral Metrics Count** - How many of 12 metrics are neutral (score = 0)
5. **Bullish %** - Percentage of metrics that are bullish
6. **Bearish %** - Percentage of metrics that are bearish

## How to Use the Utility Script

### Method 1: Import in Your Streamlit App

```python
from get_atm_strikes_data import get_atm_strikes_data, display_atm_strikes_data_detailed

# Get raw data
data = get_atm_strikes_data()
if data['status'] == 'SUCCESS':
    print(f"ATM Strike: {data['atm_strike']}")
    print(f"Total Strikes: {len(data['strikes'])}")

    # Access specific strike data
    for strike in data['strikes']:
        print(f"Strike {strike['strike_price']}: {strike['verdict']}")
        print(f"  Total Score: {strike['total_bias_score']}")

        # Access individual metrics
        for metric_key, metric_info in strike['metrics'].items():
            print(f"  {metric_info['name']}: {metric_info['emoji']} {metric_info['score']:+.1f}")

# Or display formatted view
display_atm_strikes_data_detailed()
```

### Method 2: Run as Standalone App

```bash
streamlit run get_atm_strikes_data.py
```

This will open a viewer with 3 tabs:
1. **Detailed View** - Full breakdown of all strikes and metrics
2. **Summary** - Concise ATM strike summary
3. **Export JSON** - Download data in JSON format

### Method 3: Get Summary Only

```python
from get_atm_strikes_data import get_atm_strike_metrics_summary

summary = get_atm_strike_metrics_summary()
if summary:
    print(f"ATM Strike: {summary['atm_strike']}")
    print(f"Verdict: {summary['verdict']}")
    print(f"Total Score: {summary['total_score']:+.2f}")
    print(f"\nBullish Metrics ({summary['bullish_count']}):")
    for metric in summary['metrics_breakdown']['bullish']:
        print(f"  - {metric['name']}: {metric['score']:+.1f} ({metric['interpretation']})")
```

## Data Structure

```python
{
    'status': 'SUCCESS',
    'last_updated': datetime,
    'atm_strike': 24000,
    'atm_summary': {
        'verdict': '🐂 STRONG BULLISH',
        'total_bias_score': 5.5,
        'bullish_metrics': 9,
        'bearish_metrics': 2,
        'neutral_metrics': 1,
        'total_metrics': 12,
        'bullish_percentage': 75.0,
        'bearish_percentage': 16.7
    },
    'strikes': [
        {
            'strike_price': 23900,
            'is_atm': False,
            'total_bias_score': 3.5,
            'verdict': '🐂 STRONG BULLISH',
            'verdict_color': '#00FF00',
            'metrics': {
                'OI': {
                    'name': 'Open Interest Bias',
                    'score': 1.0,
                    'emoji': '🐂',
                    'interpretation': 'PE/CE OI: 1.45'
                },
                # ... 11 more metrics
            }
        },
        # ... 4 more strikes (ATM-1, ATM, ATM+1, ATM+2)
    ]
}
```

## Example: Reading All Metrics for ATM Strike

```python
from get_atm_strikes_data import get_atm_strikes_data

data = get_atm_strikes_data()

if data['status'] == 'SUCCESS':
    atm_strike = data['atm_strike']

    # Find ATM strike data
    atm_data = next((s for s in data['strikes'] if s['is_atm']), None)

    if atm_data:
        print(f"ATM Strike: {atm_strike}")
        print(f"Overall Verdict: {atm_data['verdict']}")
        print(f"Total Bias Score: {atm_data['total_bias_score']:+.2f}\n")

        print("=" * 80)
        print("12 BIAS METRICS BREAKDOWN:")
        print("=" * 80)

        for metric_key, metric_info in atm_data['metrics'].items():
            print(f"{metric_info['emoji']} {metric_info['name']:30} | Score: {metric_info['score']:+5.1f} | {metric_info['interpretation']}")
```

## Troubleshooting

### "Option chain data not yet loaded"
**Solution:** Navigate to the "🎯 NIFTY Option Screener v7.0" tab first to load the data, then return to your analysis.

### All scores showing 0 or N/A
**Solution:** This was caused by the missing `seller_bias_direction` function, which has now been fixed. Reload the option chain data.

### Data not updating
**Solution:** The data is cached in session state. To refresh, click the "Load Data" button again in the NIFTY Option Screener tab.

## Best Practices

1. **Always check data status** before processing
2. **Monitor the last_updated timestamp** to ensure fresh data
3. **Focus on the ATM strike** for strongest signals
4. **Look for metric consensus** - When 8+ metrics agree, the signal is stronger
5. **Watch for divergences** - When price and metrics disagree, reversals may be near
6. **Use multiple timeframes** - Compare current data with previous snapshots

## Integration with Trading Strategies

### Bullish Setup (Score ≥ +3)
- Entry: When ATM shows STRONG BULLISH
- Confirmation: 8+ bullish metrics out of 12
- Target: Next resistance level
- Stop: Below ATM strike - (2 × strike gap)

### Bearish Setup (Score ≤ -3)
- Entry: When ATM shows STRONG BEARISH
- Confirmation: 8+ bearish metrics out of 12
- Target: Next support level
- Stop: Above ATM strike + (2 × strike gap)

### Neutral Setup (-1 < Score < +1)
- Strategy: Range-bound trading
- Sell ATM straddles/strangles
- Profit from theta decay
- Manage gamma risk

---

**Note:** This data updates in real-time based on option chain data from Dhan API. Always verify with live market conditions before making trading decisions.
