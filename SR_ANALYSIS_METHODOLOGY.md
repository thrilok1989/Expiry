# 📊 Support/Resistance Analysis Methodology

## Complete S/R Analysis Pipeline in ML Entry Finder

---

## 📑 TAB REFERENCE GUIDE

**Where Data Comes From:**

```
Tab 1: 🌟 Overall Market Sentiment
Tab 2: 🎯 Trade Setup
Tab 3: 📊 Active Signals (ML Entry Finder displayed here)
Tab 4: 📈 Positions
Tab 5: 🎲 Bias Analysis Pro
Tab 6: 🔍 Option Chain Analysis
Tab 7: 📉 Advanced Chart Analysis
       └─ HTF S/R (Multi-timeframe pivots)
Tab 8: 🎯 NIFTY Option Screener v7.0 ⭐ PRIMARY DATA SOURCE
       ├─ OI Walls (Max PUT/CALL OI)
       ├─ GEX Walls (Gamma Exposure)
       ├─ OI PCR (Put-Call Ratio)
       ├─ Max Pain
       ├─ Depth Analysis
       ├─ VOB (Volume Order Blocks)
       └─ NIFTY Futures Analysis
Tab 9: 🌐 Enhanced Market Data
Tab 10: 🤖 MASTER AI ANALYSIS
Tab 11: 🔬 Advanced Analytics
Tab 12: 📜 Signal History & Performance
```

**IMPORTANT**: All option chain data (OI, GEX, depth, PCR, expiry data) is in **Tab 8: NIFTY Option Screener v7.0**, NOT Tab 6.

---

## 🔍 STEP 1: DATA COLLECTION FROM 4 INSTITUTIONAL SOURCES

⚠️ **IMPORTANT**: Tab 1 (Overall Market Sentiment) S/R data is **EXCLUDED** - not working properly.

The system collects S/R levels from ONLY these sources in **PRIORITY ORDER**:

### **SOURCE 1: OI WALLS (Max PUT/CALL OI)** - HIGHEST PRIORITY
```python
# Location: comprehensive_chart_integration.py:173-191

Max PUT OI Strike → Support (where institutions defend)
Max CALL OI Strike → Resistance (where institutions defend)

Strength: HIGH
Priority Score Bonus: +50 points
Color: Support=#FF6B6B, Resistance=#4ECDC4

Example:
- Max PUT OI at ₹24,450 → SUPPORT (institutions buying PUTs = defending this level)
- Max CALL OI at ₹24,550 → RESISTANCE (institutions buying CALLs = capping at this level)
```

### **SOURCE 2: GEX WALLS (Gamma Exposure)** - 2nd PRIORITY
```python
# Location: comprehensive_chart_integration.py:193-213

Gamma Walls → Where market makers must hedge (pin zones)

Strength: HIGH
Priority Score Bonus: +40 points
Color: Support=#FFB347, Resistance=#87CEEB

How it works:
- High Gamma at strike → Market makers must buy/sell to hedge
- Creates price "magnets" that attract spot price
- Acts as support if below spot, resistance if above
```

### **SOURCE 3: HTF S/R (Multi-Timeframe Pivots)** - 3rd PRIORITY
```python
# Location: comprehensive_chart_integration.py:215-236

Pivot Lows → Support
Pivot Highs → Resistance

Timeframes: 3min, 5min, 10min, 15min

Strength: MEDIUM
Priority Score Bonus: +30 points
Color: Support=#98D8C8, Resistance=#F7DC6F

How it works:
- Calculates swing highs/lows across multiple timeframes
- Higher timeframe pivots = stronger levels
- 15min pivot > 5min pivot in importance
```

### **SOURCE 4: VOB (Volume Order Blocks)** - 4th PRIORITY
```python
# Location: comprehensive_chart_integration.py:238-259

Volume spikes → Institutional order blocks

Strength: MEDIUM (if Major) or LOW (if Minor)
Priority Score Bonus: +20 points
Color: Support=#BB8FCE, Resistance=#85C1E2

How it works:
- Detects large volume candles (institutional footprints)
- "Major" VOB = 2x avg volume
- "Minor" VOB = 1.5x avg volume
- Assumes institutions will defend their entry zones
```

### **SOURCE 5: ML CALCULATED** - FALLBACK
```python
# Generic ±100 calculation if no institutional levels found
Support = Current Price - 100
Resistance = Current Price + 100

Strength: LOW
Priority Score Bonus: 0 points
```

### **❌ EXCLUDED: Tab 1 (Overall Market Sentiment) S/R**
```
NOT USED - Not working properly/not reliable
The system does NOT use S/R levels from Tab 1.
Only institutional sources (OI, GEX, HTF, VOB) are used.
```

---

## ⚖️ STEP 2: STRENGTH ASSIGNMENT

Each level is assigned a strength rating:

```
HIGH STRENGTH:
- OI Walls (Max PUT/CALL OI)
- GEX Walls (Gamma Exposure)
- Major VOB (if volume >= 2x average)

MEDIUM STRENGTH:
- HTF S/R (all timeframes)
- Minor VOB (if volume >= 1.5x average)

LOW STRENGTH:
- ML Calculated (±100 fallback)
- Any level with weak institutional footprint
```

---

## 🎯 STEP 3: SCORING METHODOLOGY

Each S/R level receives a **COMPOSITE SCORE** based on 3 factors:

### **Factor 1: Base Strength Score**
```python
# Location: ml_entry_finder.py:219-226

if strength == 'HIGH':
    score += 100  ✅ OI Walls, GEX Walls
elif strength == 'MEDIUM':
    score += 70   📊 HTF S/R, VOB
else:  # LOW
    score += 40   📉 ML Calculated
```

### **Factor 2: Level Type Bonus (Institutional Priority)**
```python
# Location: ml_entry_finder.py:228-237

if 'OI Wall' in level_type:
    score += 50  ⭐⭐⭐⭐⭐ HIGHEST
elif 'GEX Wall' in level_type:
    score += 40  ⭐⭐⭐⭐
elif 'HTF' in level_type:
    score += 30  ⭐⭐⭐
elif 'VOB' in level_type:
    score += 20  ⭐⭐
else:
    score += 0   ⭐
```

### **Factor 3: Distance Proximity Bonus/Penalty**
```python
# Location: ml_entry_finder.py:239-250

distance_pct = (distance / current_price) * 100

if distance_pct < 0.5%:    # Within ~120 pts for NIFTY
    score += 50  🎯 VERY CLOSE
elif distance_pct < 1.0%:  # Within ~240 pts
    score += 30  📍 CLOSE
elif distance_pct < 2.0%:  # Within ~480 pts
    score += 10  📌 MODERATE
else:
    score -= 20  ⛔ FAR AWAY (penalized)
```

### **Final Score Formula**
```
TOTAL SCORE = Strength Score + Type Bonus + Distance Score

Examples:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OI Wall (HIGH) + 30pts away:
100 (HIGH) + 50 (OI) + 50 (0.12% away) = 200 points ⭐⭐⭐⭐⭐

GEX Wall (HIGH) + 100pts away:
100 (HIGH) + 40 (GEX) + 30 (0.4% away) = 170 points ⭐⭐⭐⭐

HTF 15min (MEDIUM) + 50pts away:
70 (MEDIUM) + 30 (HTF) + 50 (0.2% away) = 150 points ⭐⭐⭐

VOB (MEDIUM) + 200pts away:
70 (MEDIUM) + 20 (VOB) + 10 (0.8% away) = 100 points ⭐⭐

ML Calculated (LOW) + 500pts away:
40 (LOW) + 0 (generic) - 20 (2% away) = 20 points ⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🔬 STEP 4: FILTERING (MAJOR + NEAR)

After scoring, levels are **FILTERED** to show only relevant levels:

### **MAJOR LEVELS Filter**
```python
# Location: ml_entry_finder.py:75-82

MAJOR = Only levels with strength == 'HIGH'

Includes:
✅ OI Walls (Max PUT/CALL OI)
✅ GEX Walls (Gamma Exposure)
✅ Major VOB (if 2x volume)

Excludes:
❌ HTF S/R (MEDIUM strength)
❌ Minor VOB (MEDIUM strength)
❌ ML Calculated (LOW strength)
```

### **NEAR LEVELS Filter**
```python
# Location: ml_entry_finder.py:76-92

NEAR = Only levels within 50 POINTS of spot price

For Support:
- Level price < current price
- Distance = current_price - level_price
- Keep if distance <= 50 pts

For Resistance:
- Level price > current price
- Distance = level_price - current_price
- Keep if distance <= 50 pts

Example at spot ₹24,500:
✅ Support at ₹24,460 → 40 pts below ✅ NEAR
✅ Resistance at ₹24,540 → 40 pts above ✅ NEAR
❌ Support at ₹24,400 → 100 pts below ❌ TOO FAR
```

### **Combined Filtering Logic**
```python
# Location: ml_entry_finder.py:81-92

filtered_supports = MAJOR supports + NEAR supports (no duplicates)
filtered_resistances = MAJOR resistances + NEAR resistances (no duplicates)

Result:
- Shows HIGH strength levels regardless of distance
- Shows levels within 50 pts regardless of strength
- Removes noise from far away + weak levels
```

---

## 📊 STEP 5: FINDING NEAREST & STRONGEST LEVELS

### **Nearest Level**
```python
# Location: ml_entry_finder.py:256-282

Nearest Support = Highest price BELOW current price
Nearest Resistance = Lowest price ABOVE current price

Example at spot ₹24,500:
Supports: [24,450, 24,400, 24,350]
→ Nearest = ₹24,450 (closest below)

Resistances: [24,550, 24,600, 24,650]
→ Nearest = ₹24,550 (closest above)
```

### **Strongest Level**
```python
# Location: ml_entry_finder.py:284-294

Strongest = Level with HIGHEST composite score

Example scores:
- OI Wall at ₹24,450: Score 200
- GEX at ₹24,400: Score 170
- HTF at ₹24,420: Score 150

→ Strongest Support = OI Wall ₹24,450
```

---

## 🎯 STEP 6: FINAL PRIORITIZATION & DISPLAY

Levels are displayed in **PRIORITY ORDER**:

### **Priority Hierarchy**
```
1. OI WALLS (Max PUT/CALL OI)        ⭐⭐⭐⭐⭐
2. GEX WALLS (Gamma Exposure)        ⭐⭐⭐⭐
3. HTF S/R (Multi-timeframe)         ⭐⭐⭐
4. VOB (Volume Order Blocks)         ⭐⭐
5. ML CALCULATED (±100 fallback)     ⭐
```

### **Display Structure in Tab 3**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💎 MAJOR Support/Resistance Levels (HIGH Strength ONLY)

Shows: Top 5 OI Walls, GEX Walls, Major VOB
Sorted: By price (support descending, resistance ascending)
Display: Price, Type, Distance from spot

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 NEAR Spot Support/Resistance (Within 50 Points)

Shows: All levels within 50 pts (any strength)
Sorted: By price (support descending, resistance ascending)
Display: Price, Type, Strength, Distance from spot

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Complete Breakdown (Expandable)

Shows: All filtered levels (up to 10 each)
Includes: MAJOR + NEAR combined
Display: Price, Type, Strength
```

---

## 🧮 DISTANCE CALCULATION

```python
# For Support (below spot):
distance = current_price - support_price

# For Resistance (above spot):
distance = resistance_price - current_price

# Display format:
Support at ₹24,450: "Distance: 50 pts below"
Resistance at ₹24,550: "Distance: 50 pts above"
```

---

## 📈 COMPLETE EXAMPLE

### **Current Price: ₹24,500**

### **Data Collection**
```
Tab 8: NIFTY Option Screener v7.0
→ Max PUT OI: ₹24,450 (10,000 OI)
→ Max CALL OI: ₹24,550 (12,000 OI)
→ GEX Wall: ₹24,400 (high gamma)
→ GEX Wall: ₹24,600 (high gamma)
→ OI PCR: 1.15
→ Max Pain: ₹24,500
→ Depth Analysis: Available
→ VOB: Major ₹24,430 (3x volume), Minor ₹24,350

Tab 7: Advanced Chart Analysis
→ HTF 15min pivot low: ₹24,420
→ HTF 15min pivot high: ₹24,580
→ HTF 5min pivot low: ₹24,470
```

### **Strength Assignment**
```
₹24,450 (OI Wall) → HIGH
₹24,550 (OI Wall) → HIGH
₹24,400 (GEX) → HIGH
₹24,600 (GEX) → HIGH
₹24,420 (HTF 15min) → MEDIUM
₹24,580 (HTF 15min) → MEDIUM
₹24,470 (HTF 5min) → MEDIUM
₹24,430 (Major VOB) → MEDIUM
₹24,350 (Minor VOB) → LOW
```

### **Scoring**
```
₹24,450 OI Wall: 100+50+50 = 200 ⭐⭐⭐⭐⭐
₹24,470 HTF 5min: 70+30+50 = 150 ⭐⭐⭐
₹24,420 HTF 15min: 70+30+30 = 130 ⭐⭐⭐
₹24,430 VOB: 70+20+30 = 120 ⭐⭐
₹24,400 GEX: 100+40-20 = 120 ⭐⭐
₹24,350 VOB: 40+20-20 = 40 ⭐
```

### **Filtering - MAJOR**
```
MAJOR Support (HIGH strength only):
✅ ₹24,450 (OI Wall)
✅ ₹24,400 (GEX Wall)
```

### **Filtering - NEAR**
```
NEAR Support (within 50 pts):
✅ ₹24,470 (HTF 5min) → 30 pts below
✅ ₹24,450 (OI Wall) → 50 pts below
```

### **Combined Filtered Support**
```
All Support Levels (MAJOR + NEAR, no duplicates):
1. ₹24,470 (HTF 5min) - MEDIUM - 30 pts below
2. ₹24,450 (OI Wall) - HIGH - 50 pts below
3. ₹24,400 (GEX Wall) - HIGH - 100 pts below

Nearest Support: ₹24,470 (closest)
Strongest Support: ₹24,450 (highest score: 200)
```

### **Display in Tab 3**
```
💎 MAJOR Support Levels:
• ₹24,450 (OI Wall - Max PUT OI) - Distance: 50 pts below
• ₹24,400 (GEX Wall) - Distance: 100 pts below

📍 NEAR Support Levels:
• ₹24,470 (HTF Support - 5min) - MEDIUM - Distance: 30 pts below
• ₹24,450 (OI Wall - Max PUT OI) - HIGH - Distance: 50 pts below
```

---

## 🔑 KEY TAKEAWAYS

1. **5 Data Sources**: OI Walls > GEX > HTF > VOB > ML
2. **3-Factor Scoring**: Strength + Type + Distance
3. **Smart Filtering**: MAJOR (HIGH strength) + NEAR (50 pts)
4. **Priority Display**: Institutional levels shown first
5. **Distance Aware**: Closer levels score higher
6. **No Noise**: Far away + weak levels filtered out

**Result**: Clean, actionable S/R levels based on institutional activity 🎯
