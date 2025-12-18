# 📊 NIFTY & SENSEX Trading Guide

Complete guide for trading NIFTY 50 and SENSEX indices using the Ultimate Trading Application.

---

## 🎯 Index Specifications

### NIFTY 50

| Parameter | Value |
|-----------|-------|
| Security ID | `13` |
| Exchange | IDX_I (Index) |
| Symbol | NIFTY 50 |
| Typical Range | 100-200 points per day |
| Trading Hours | 9:15 AM - 3:30 PM IST |
| Lot Size (Options) | 50 |
| Tick Size | 0.05 |

### SENSEX

| Parameter | Value |
|-----------|-------|
| Security ID | `51` |
| Exchange | IDX_I (Index) |
| Symbol | SENSEX |
| Typical Range | 300-600 points per day |
| Trading Hours | 9:15 AM - 3:30 PM IST |
| Lot Size (Options) | 10 |
| Tick Size | 0.05 |

### BANK NIFTY (Bonus)

| Parameter | Value |
|-----------|-------|
| Security ID | `25` |
| Exchange | IDX_I (Index) |
| Symbol | BANK NIFTY |
| Typical Range | 200-400 points per day |
| Lot Size (Options) | 15 |
| High volatility ⚠️ |

---

## ⚙️ Recommended Settings

### For NIFTY 50

```toml
# In Streamlit sidebar
Security ID: 13
Symbol: NIFTY 50
Exchange: IDX_I
Alert Distance: 10 points
Auto Refresh: Enabled
```

**Why 10 points?**
- NIFTY moves faster than stocks
- 10 points = reasonable buffer
- Reduces false alerts
- Still catches significant levels

### For SENSEX

```toml
Security ID: 51
Symbol: SENSEX
Exchange: IDX_I
Alert Distance: 20 points  # SENSEX is ~3x NIFTY
Auto Refresh: Enabled
```

**Why 20 points?**
- SENSEX trades at higher absolute levels
- Larger point movements
- 20 points ≈ 0.03% (similar to NIFTY's 10 points)

---

## 📈 Index Trading Strategies

### Strategy 1: Intraday Scalping (NIFTY)

**Setup**:
- Timeframe: 1-minute
- Target: 20-30 points
- Stop Loss: 15-20 points
- Risk:Reward = 1:1.5

**Entry Rules**:
```
✅ Price at Bullish VOB + RSI < 40
✅ HTF 10T support nearby
✅ VIDYA trending up

Entry: Buy at VOB lower edge
Stop: 20 points below entry
Target: 30 points above entry
```

**Example Trade**:
```
Time: 10:30 AM
NIFTY: 24,485

Alert: 🟢 Bullish VOB at 24,470-24,500
Confirmation:
- RSI: 38 (approaching oversold)
- 10T Support: 24,480
- VIDYA: Trending up

Action:
- Buy NIFTY 24500 CE
- Entry: ₹120
- Stop: ₹90 (spot below 24,465)
- Target: ₹165 (spot at 24,515)
- Quantity: 50 (1 lot)

Result:
- Time: 11:15 AM
- NIFTY: 24,520
- Exit: ₹170
- Profit: ₹2,500 (20.8%)
```

### Strategy 2: Trend Following (SENSEX)

**Setup**:
- Timeframe: 5-minute
- Target: 100-150 points
- Stop Loss: 80-100 points
- Hold time: 1-3 hours

**Entry Rules**:
```
✅ Price breaks above HTF resistance
✅ VIDYA trending strongly
✅ Volume confirmation
✅ RSI > 50 (uptrend)

Entry: Above resistance with confirmation
Stop: Below nearest VOB
Target: Next HTF resistance
```

**Example Trade**:
```
Time: 11:00 AM
SENSEX: 81,250

Setup:
- 15T Resistance broken: 81,200
- VIDYA: Strong uptrend
- Volume: Above average
- RSI: 58

Action:
- Buy SENSEX 81500 CE (next weekly)
- Entry: ₹300
- Stop: ₹220 (spot below 81,100)
- Target 1: ₹420 (spot at 81,400)
- Target 2: ₹550 (spot at 81,600)
- Quantity: 10 (1 lot)

Management:
- Book 50% at Target 1
- Trail stop to breakeven
- Let rest run

Result:
- Time: 2:30 PM
- SENSEX: 81,550
- Exit: ₹590 (Target 2)
- Profit: ₹2,900 on full lot (96.7%)
```

### Strategy 3: Reversal Trading (NIFTY)

**Setup**:
- Timeframe: 1-minute
- Trade against extreme moves
- High win rate, small targets

**Entry Rules**:
```
✅ Price hits Bearish VOB
✅ RSI > 80 (overbought)
✅ Rejection candle forms
✅ HTF resistance nearby

Entry: Short at resistance
Stop: Above VOB upper edge
Target: Next support level
```

**Example Trade**:
```
Time: 1:45 PM
NIFTY: 24,565

Alert: 🔴 Bearish VOB at 24,550-24,580
Confirmation:
- RSI: 82 (overbought)
- 10T Resistance: 24,570
- Rejection candle formed

Action:
- Buy NIFTY 24550 PE
- Entry: ₹115
- Stop: ₹85 (spot above 24,585)
- Target: ₹155 (spot at 24,530)
- Quantity: 50 (1 lot)

Result:
- Time: 2:15 PM
- NIFTY: 24,525
- Exit: ₹162
- Profit: ₹2,350 (40.9%)
```

---

## 🔔 Alert Interpretation for Indices

### NIFTY Alerts

**Bullish VOB Alert**:
```
🟢 Bullish VOB Alert - NIFTY 50
💰 Current: 24,485
📊 VOB: 24,470-24,500
📍 Distance: 8 points
```

**What it means**:
- Strong demand zone detected
- Price approaching support
- High probability bounce area
- Look for long opportunities

**Action Plan**:
1. Check RSI (want < 50 for longs)
2. Verify VIDYA trend (want uptrend)
3. Look for rejection candle
4. Enter with stop below VOB

**HTF Resistance Alert**:
```
🔵 HTF Resistance Alert - NIFTY 50
💰 Current: 24,565
🚧 Resistance: 24,570
⏱ Timeframe: 15T
```

**What it means**:
- Price at significant resistance
- Multiple attempts to break
- Could reverse or breakout
- Decision point

**Action Plan**:
1. Wait for confirmation
2. If rejection → Look for shorts
3. If breakout → Wait for retest, then long
4. Don't chase breakouts

### SENSEX Alerts

**Similar interpretation, but**:
- Larger point movements
- More institutional
- Follows global cues more
- Less volatile than NIFTY

---

## ⏰ Best Trading Times

### High Probability Sessions

**Morning Session (9:15 AM - 11:30 AM)**
```
✅ BEST for: Trend following
✅ Volatility: High
✅ Volume: Highest
✅ Setup: Follow opening momentum

Strategy:
- Wait for 15 minutes after open
- Identify trend direction
- Trade in trend direction
- Use VOBs for entries
```

**Mid-Day (11:30 AM - 2:00 PM)**
```
⚠️ CAUTION: Range-bound
⚠️ Volatility: Low
⚠️ Volume: Moderate
⚠️ Setup: Scalp between levels

Strategy:
- Trade range-bound
- Buy at support, sell at resistance
- Smaller targets (20-30 points NIFTY)
- Quick in-and-out
```

**Closing Session (2:00 PM - 3:30 PM)**
```
✅ GOOD for: Breakouts
✅ Volatility: Moderate-High
✅ Volume: Increases
✅ Setup: Breakout/breakdown plays

Strategy:
- Watch for breakouts of day's range
- Follow closing momentum
- Exit all positions by 3:15 PM
- Don't carry overnight (unless planned)
```

### Avoid These Times

**9:15 AM - 9:30 AM**
- ❌ Extreme volatility
- ❌ Erratic price action
- ❌ Wide spreads
- Wait for market to settle

**12:00 PM - 1:00 PM**
- ❌ Low volume
- ❌ Choppy movement
- ❌ False signals
- Lunch hour effect

---

## 📊 Using Indicators for Index Trading

### Volume Order Blocks (VOB)

**NIFTY specific**:
- Strong VOBs form at round numbers (24,500, 24,550, etc.)
- Option strikes create natural VOBs
- Higher volume VOBs = stronger zones
- Watch for VOB clusters

**Entry technique**:
```
1. Wait for price to touch VOB edge
2. Look for rejection candle
3. Check RSI for confirmation
4. Enter with tight stop below VOB
5. Target: Next VOB or HTF level
```

### HTF Support/Resistance

**Timeframe selection**:
- **10T (10-minute)**: Intraday scalping
- **15T (15-minute)**: Swing trades (2-4 hours)

**Multiple timeframe confluence**:
```
Strongest Setup:
- 10T support at 24,480
- 15T support at 24,475
- Bullish VOB at 24,470-24,500

Action: Strong buy zone!
Target: Next resistance cluster
```

### VIDYA (Trend Indicator)

**For NIFTY**:
- Orange line = trend direction
- Above VIDYA = Bullish (look for longs)
- Below VIDYA = Bearish (look for shorts)
- VIDYA slope = trend strength

**Trading with VIDYA**:
```
Strong Uptrend (VIDYA steep up):
- Buy pullbacks to VIDYA
- Hold winners longer
- Trail stops aggressively

Weak Trend (VIDYA flat):
- Scalp only
- Quick profits
- Tight stops
```

### Ultimate RSI

**NIFTY thresholds**:
- **> 80**: Overbought (look for shorts)
- **70-80**: Strong uptrend (don't fight)
- **50-70**: Normal uptrend
- **30-50**: Normal downtrend
- **20-30**: Strong downtrend (don't fight)
- **< 20**: Oversold (look for longs)

**Divergence trading**:
```
Bullish Divergence:
- NIFTY making lower lows
- RSI making higher lows
- Signal: Reversal coming
- Action: Prepare to buy

Bearish Divergence:
- NIFTY making higher highs
- RSI making lower highs
- Signal: Reversal coming
- Action: Prepare to sell
```

---

## 💰 Position Sizing for Indices

### NIFTY 50

**Example: ₹1,00,000 Capital**

**Conservative (1% risk)**:
```
Risk per trade: ₹1,000
Stop loss: 20 points
Lot size: 50

Position size calculation:
₹1,000 / (20 points × 50) = 1 lot

Trade:
- Buy 1 lot (50 qty)
- Risk: ₹1,000
- If stop hit: -1% of capital
```

**Moderate (2% risk)**:
```
Risk per trade: ₹2,000
Stop loss: 20 points

Position size: 2 lots
Risk: ₹2,000 max
```

**Aggressive (3% risk)**:
```
Risk per trade: ₹3,000
Stop loss: 20 points

Position size: 3 lots
Risk: ₹3,000 max
```

### SENSEX

**Example: ₹1,00,000 Capital**

**Conservative**:
```
Risk per trade: ₹1,000
Stop loss: 100 points
Lot size: 10

Position size: 1 lot
Max risk: ₹1,000
```

**Note**: SENSEX options are costlier but smaller lot size (10 vs 50)

---

## 🎯 Daily Trading Routine

### Pre-Market (8:00 AM - 9:15 AM)

```
☐ Regenerate DhanHQ Access Token
☐ Start application
☐ Check global markets (US, Asia)
☐ Check SGX NIFTY
☐ Identify key levels from yesterday
☐ Plan trades based on gap up/down
☐ Set alerts
```

### Opening (9:15 AM - 9:30 AM)

```
☐ Observe opening price action
☐ DON'T trade first 15 minutes
☐ Let volatility settle
☐ Note opening range
☐ Watch volume
```

### Active Trading (9:30 AM - 3:00 PM)

```
☐ Monitor Telegram alerts
☐ Follow your trading plan
☐ Take trades at planned levels
☐ Manage positions actively
☐ Book profits at targets
☐ Cut losses quickly
```

### Closing (3:00 PM - 3:30 PM)

```
☐ Close all intraday positions by 3:15 PM
☐ Don't take new trades post 3:00 PM
☐ Review day's trades
☐ Update trading journal
☐ Plan for tomorrow
```

### Post-Market (3:30 PM - 4:00 PM)

```
☐ Analyze what worked/didn't work
☐ Check overnight news
☐ Review VOB and HTF levels for tomorrow
☐ Calculate P&L
☐ Prepare for next day
```

---

## 📱 Telegram Alert Workflow

### When Alert Received

```
1. READ the alert (don't react immediately)
2. OPEN the app and verify
3. CHECK all indicators align
4. CALCULATE risk and position size
5. PLACE order with stop loss
6. SET target and manage trade
```

### Alert Priority System

**Highest Priority** (Take immediately):
```
🟢 Bullish VOB + 10T Support + RSI < 30
🔴 Bearish VOB + 15T Resistance + RSI > 80
```

**Medium Priority** (Wait for confirmation):
```
🟢 Bullish VOB alone
🔵 HTF Resistance alone
```

**Low Priority** (Monitor only):
```
Single indicator signals
Against main trend
Low volume periods
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ DON'T:

1. **Trade during 9:15-9:30 AM**
   - Wait for market to settle
   - Avoid whipsaws

2. **Hold losses hoping for recovery**
   - Respect your stop loss
   - Cut losses quickly

3. **Trade against VIDYA trend**
   - Trend is your friend
   - Don't pick tops/bottoms

4. **Ignore RSI extremes**
   - Don't buy at RSI > 80
   - Don't short at RSI < 20

5. **Over-leverage**
   - Stick to 1-3% risk
   - Survive to trade another day

6. **Trade every alert**
   - Quality over quantity
   - Wait for best setups

7. **Hold overnight without plan**
   - Intraday = close by 3:15 PM
   - Overnight = different strategy

### ✅ DO:

1. **Wait for multiple confirmations**
   - VOB + HTF + RSI/VIDYA

2. **Always use stop losses**
   - Calculate before entry
   - Place immediately

3. **Book partial profits**
   - 50% at 1:1 R:R
   - Let rest run with trailing stop

4. **Follow your plan**
   - Don't deviate based on emotions
   - Stick to strategy

5. **Keep a trading journal**
   - Track all trades
   - Learn from mistakes

6. **Respect market hours**
   - Best trading: 9:30 AM - 11:30 AM
   - Close all by 3:15 PM

---

## 📊 Performance Tracking

### Daily Metrics

```
Date: ___________
Index: NIFTY/SENSEX

Trades Taken: ___
Winners: ___
Losers: ___
Win Rate: ___%

Gross P&L: ₹_____
Net P&L: ₹_____ (after charges)
ROI: ___%

Best Trade: ₹_____
Worst Trade: ₹_____
Average Win: ₹_____
Average Loss: ₹_____

Lessons Learned:
_________________
_________________
```

### Weekly Review

```
Week: ___________

Total Trades: ___
Win Rate: ___%
Net P&L: ₹_____
Capital Growth: ___%

Strategy Performance:
- Scalping: __% win rate, ₹____
- Trend: __% win rate, ₹____
- Reversal: __% win rate, ₹____

What worked:
_________________

What didn't:
_________________

Next week focus:
_________________
```

---

## 🎯 Quick Reference Card

```
┌─────────────────────────────────────┐
│   NIFTY/SENSEX TRADING QUICK CARD   │
├─────────────────────────────────────┤
│ NIFTY:                              │
│ - Security ID: 13                   │
│ - Alert Distance: 10 points         │
│ - Lot Size: 50                      │
│                                      │
│ SENSEX:                             │
│ - Security ID: 51                   │
│ - Alert Distance: 20 points         │
│ - Lot Size: 10                      │
├─────────────────────────────────────┤
│ BEST TIMES:                         │
│ - 9:30 AM - 11:30 AM ✅            │
│ - 2:00 PM - 3:15 PM ✅             │
│                                      │
│ AVOID:                              │
│ - 9:15 AM - 9:30 AM ❌             │
│ - 12:00 PM - 1:00 PM ❌            │
├─────────────────────────────────────┤
│ ENTRY RULES:                        │
│ 1. Alert received                   │
│ 2. Multiple confirmations           │
│ 3. Stop loss defined                │
│ 4. Position sized                   │
│ 5. Execute!                         │
├─────────────────────────────────────┤
│ RISK MANAGEMENT:                    │
│ - Max 1-2% per trade               │
│ - Always use stops                  │
│ - Book 50% at 1:1 R:R              │
│ - Close all by 3:15 PM             │
└─────────────────────────────────────┘
```

---

**Trade NIFTY & SENSEX like a pro! 📈🇮🇳**

*This guide is for educational purposes only. Always practice proper risk management.*
