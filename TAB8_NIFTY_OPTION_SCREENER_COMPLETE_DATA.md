# 📊 TAB 8: NIFTY OPTION SCREENER V7.0 - COMPLETE DATA AVAILABLE

**File:** `NiftyOptionScreener.py`
**Version:** v7.0 - 100% Seller's Perspective

---

## 🎯 OVERVIEW - 5 MAJOR FEATURE SETS

1. **ATM Bias Analyzer** (12 metrics)
2. **Moment Detector** (4 components)
3. **Market Depth/Orderbook Pressure**
4. **Expiry Spike Detector**
5. **Enhanced OI/PCR Analytics**

---

## 1️⃣ ATM BIAS ANALYZER (12 METRICS)

**Function:** `analyze_atm_bias(merged_df, spot, atm_strike, strike_gap)`

**ATM Window:** ±2 strikes around ATM (5 strikes total)

### **12 ATM Bias Metrics:**

#### **Metric 1: OI_Bias**
- **Calculation:** PE OI / CE OI ratio at ATM
- **Values:**
  - +1.0: Ratio > 1.5 (Heavy PUT OI → Bullish sellers)
  - +0.5: Ratio > 1.0 (Moderate PUT OI → Mild bullish)
  - -0.5: Ratio < 1.0 (Moderate CALL OI → Mild bearish)
  - -1.0: Ratio < 0.7 (Heavy CALL OI → Bearish sellers)
  - 0: Balanced
- **Interpretation:** From seller's perspective

#### **Metric 2: ChgOI_Bias**
- **Calculation:** PE Change in OI vs CE Change in OI
- **Values:**
  - +1.0: Only PUT writing (Strong bullish buildup)
  - +0.5: More PUT writing (Bullish buildup)
  - -0.5: More CALL writing (Bearish buildup)
  - -1.0: Only CALL writing (Strong bearish)
  - 0: Both unwinding or balanced
- **Interpretation:** Active buildup direction

#### **Metric 3: Volume_Bias**
- **Calculation:** PE Volume / CE Volume ratio
- **Values:**
  - +1.0: Ratio > 1.3 (High PUT volume → Bullish activity)
  - +0.5: Ratio > 1.0 (More PUT volume → Mild bullish)
  - -0.5: Ratio < 1.0 (More CALL volume → Mild bearish)
  - -1.0: Ratio < 0.8 (High CALL volume → Bearish activity)
  - 0: Balanced volume
- **Interpretation:** Immediate trading activity

#### **Metric 4: Delta_Bias**
- **Calculation:** Net Delta = Sum(CE Delta) + Sum(PE Delta)
- **Values:**
  - +1.0: Net Delta > 0.3 (Positive delta → CALL heavy → Bullish)
  - +0.5: Net Delta > 0.1 (Mild positive delta → Slightly bullish)
  - -0.5: Net Delta < -0.1 (Mild negative delta → Slightly bearish)
  - -1.0: Net Delta < -0.3 (Negative delta → PUT heavy → Bearish)
  - 0: Neutral delta
- **Interpretation:** Directional exposure

#### **Metric 5: Gamma_Bias**
- **Calculation:** Net Gamma = Sum(CE Gamma) + Sum(PE Gamma)
- **Values:**
  - +1.0: Net Gamma > 0.1 (Positive → Stabilizing → Bullish)
  - +0.5: Net Gamma > 0 (Mild positive → Slightly stabilizing)
  - -0.5: Net Gamma < 0 (Mild negative → Slightly explosive)
  - -1.0: Net Gamma < -0.1 (Negative → Explosive → Bearish)
  - 0: Neutral gamma
- **Interpretation:** Volatility tendency (sellers prefer positive gamma)

#### **Metric 6: Premium_Bias**
- **Calculation:** Average PE Premium / Average CE Premium
- **Values:**
  - +1.0: Ratio > 1.2 (PUT premium higher → Bullish sentiment)
  - +0.5: Ratio > 1.0 (PUT premium slightly higher → Mild bullish)
  - -0.5: Ratio < 1.0 (CALL premium slightly higher → Mild bearish)
  - -1.0: Ratio < 0.8 (CALL premium higher → Bearish sentiment)
  - 0: Balanced premiums
- **Interpretation:** Market pricing sentiment

#### **Metric 7: IV_Bias**
- **Calculation:** Average PE IV vs Average CE IV
- **Values:**
  - +1.0: PE IV > CE IV + 3 (PUT IV higher → Bullish fear)
  - +0.5: PE IV > CE IV + 1 (PUT IV slightly higher → Mild bullish fear)
  - -0.5: CE IV > PE IV + 1 (CALL IV slightly higher → Mild bearish fear)
  - -1.0: CE IV > PE IV + 3 (CALL IV higher → Bearish fear)
  - 0: Balanced IV
- **Interpretation:** Implied volatility skew

#### **Metric 8: Delta_Exposure_Bias**
- **Calculation:** Sum(Delta_CE × OI_CE) + Sum(Delta_PE × OI_PE)
- **Values:**
  - +1.0: Net > 1,000,000 (High CALL delta exposure → Bullish pressure)
  - +0.5: Net > 500,000 (Moderate CALL delta exposure → Slightly bullish)
  - -0.5: Net < -500,000 (Moderate PUT delta exposure → Slightly bearish)
  - -1.0: Net < -1,000,000 (High PUT delta exposure → Bearish pressure)
  - 0: Balanced exposure
- **Interpretation:** OI-weighted directional pressure

#### **Metric 9: Gamma_Exposure_Bias**
- **Calculation:** Sum(Gamma_CE × OI_CE) + Sum(Gamma_PE × OI_PE)
- **Values:**
  - +1.0: Net > 500,000 (Positive gamma exposure → Stabilizing → Bullish)
  - +0.5: Net > 100,000 (Mild positive gamma → Slightly stabilizing)
  - -0.5: Net < -100,000 (Mild negative gamma → Slightly explosive)
  - -1.0: Net < -500,000 (Negative gamma exposure → Explosive → Bearish)
  - 0: Balanced gamma exposure
- **Interpretation:** OI-weighted volatility tendency

#### **Metric 10: IV_Skew_Bias**
- **Calculation:** ATM IV vs Nearby (±1 strike) IV comparison
- **Values:**
  - +0.5: ATM PE IV > Nearby PE IV + 2 (ATM PUT IV higher → Bullish skew)
  - -0.5: ATM CE IV > Nearby CE IV + 2 (ATM CALL IV higher → Bearish skew)
  - 0: Flat IV skew
- **Interpretation:** ATM vs OTM IV curve

#### **Metric 11: OI_Change_Bias**
- **Calculation:** OI Change Rate = Total |ΔOI| / Total OI
- **Values:**
  - +0.5: Change rate > 10% AND more PE buildup (Rapid PUT OI buildup → Bullish acceleration)
  - -0.5: Change rate > 10% AND more CE buildup (Rapid CALL OI buildup → Bearish acceleration)
  - 0: Slow OI changes (< 10%)
- **Interpretation:** OI buildup speed and direction

#### **Metric 12: (Implicit in total score)**
- Normalized aggregate score across all 11 metrics

### **ATM Bias Output:**

```python
{
    "instrument": "NIFTY",
    "atm_strike": 24500,
    "zone": "ATM",
    "level": "ATM Cluster",

    # Individual bias scores
    "bias_scores": {
        "OI_Bias": -1.0 to +1.0,
        "ChgOI_Bias": -1.0 to +1.0,
        "Volume_Bias": -1.0 to +1.0,
        "Delta_Bias": -1.0 to +1.0,
        "Gamma_Bias": -1.0 to +1.0,
        "Premium_Bias": -1.0 to +1.0,
        "IV_Bias": -1.0 to +1.0,
        "Delta_Exposure_Bias": -1.0 to +1.0,
        "Gamma_Exposure_Bias": -1.0 to +1.0,
        "IV_Skew_Bias": -0.5 to +0.5,
        "OI_Change_Bias": -0.5 to +0.5
    },

    # Interpretations (text explanations)
    "bias_interpretations": {
        "OI_Bias": "Heavy PUT OI at ATM → Bullish sellers",
        # ... for each metric
    },

    # Emoji indicators
    "bias_emojis": {
        "OI_Bias": "🐂 Bullish" / "🐻 Bearish" / "⚖️ Neutral",
        # ... for each metric
    },

    # Aggregate score
    "total_score": -1.0 to +1.0,  # Normalized average

    # Overall verdict
    "verdict": "🐂 BULLISH" / "🐂 Mild Bullish" / "⚖️ NEUTRAL" / "🐻 Mild Bearish" / "🐻 BEARISH",
    "verdict_color": "#00ff88" / "#00cc66" / "#66b3ff" / "#ff6666" / "#ff4444",
    "verdict_explanation": "ATM zone showing strong bullish bias for sellers",

    # Raw metrics
    "metrics": {
        "ce_oi": 12500000,
        "pe_oi": 18750000,
        "ce_chg": 250000,
        "pe_chg": 500000,
        "ce_vol": 125000,
        "pe_vol": 187500,
        "net_delta": 0.235,
        "net_gamma": -0.045,
        "ce_iv": 18.5,
        "pe_iv": 21.2,
        "delta_exposure": 875000,
        "gamma_exposure": -325000
    }
}
```

**Verdict Thresholds:**
- **🐂 BULLISH:** Normalized score > 0.3
- **🐂 Mild Bullish:** Normalized score > 0.1
- **⚖️ NEUTRAL:** -0.1 ≤ Normalized score ≤ 0.1
- **🐻 Mild Bearish:** Normalized score < -0.1
- **🐻 BEARISH:** Normalized score < -0.3

---

## 2️⃣ MOMENT DETECTOR (4 COMPONENTS)

**Purpose:** Detect if a price move is real or fake

**Weights:**
- Momentum Burst: 40%
- Orderbook Pressure: 20%
- Gamma Cluster: 25%
- OI Acceleration: 15%

### **Component 1: Momentum Burst**
**Function:** `compute_momentum_burst(history)`

**Formula:** `|ΔVolume| × |ΔIV| × |Δ|OI||` (normalized to 0-100)

**Requirements:** At least 2 refresh points in history

**Calculation:**
```python
dvol = (current_total_volume - prev_total_volume) / time_delta
div = (current_total_iv - prev_total_iv) / time_delta
ddoi = (current_abs_doi - prev_abs_doi) / time_delta

burst_raw = abs(dvol) * abs(div) * abs(ddoi)
score = normalized to 0-100
```

**Output:**
```python
{
    "available": True/False,
    "score": 0-100,
    "note": "Momentum burst (energy) is rising" if score > 60 else "No strong energy burst detected"
}
```

**Interpretation:**
- **Score > 60:** Strong energy burst - Real move
- **Score < 60:** No burst - Possibly fake move or consolidation

### **Component 2: Orderbook Pressure**
**Function:** `orderbook_pressure_score(depth, levels=5)`

**Data Source:** Dhan API - NIFTY market depth (5 levels)

**Calculation:**
```python
buy_qty = sum(depth["buy"][:5])  # Top 5 bid levels quantity
sell_qty = sum(depth["sell"][:5])  # Top 5 ask levels quantity

pressure = (buy_qty - sell_qty) / (buy_qty + sell_qty)
# Range: -1.0 to +1.0
```

**Output:**
```python
{
    "available": True/False,
    "pressure": -1.0 to +1.0,
    "buy_qty": 12500.0,  # Total buy quantity (5 levels)
    "sell_qty": 8750.0   # Total sell quantity (5 levels)
}
```

**Interpretation:**
- **Pressure > +0.5:** Heavy buy-side depth → Bullish pressure
- **Pressure > +0.2:** Moderate buy-side → Mild bullish
- **-0.2 to +0.2:** Balanced orderbook
- **Pressure < -0.2:** Moderate sell-side → Mild bearish
- **Pressure < -0.5:** Heavy sell-side depth → Bearish pressure

**Market Depth Structure from Dhan API:**
```python
{
    "buy": [
        [price, quantity],  # Level 1 (best bid)
        [price, quantity],  # Level 2
        [price, quantity],  # Level 3
        [price, quantity],  # Level 4
        [price, quantity]   # Level 5
    ],
    "sell": [
        [price, quantity],  # Level 1 (best ask)
        [price, quantity],  # Level 2
        [price, quantity],  # Level 3
        [price, quantity],  # Level 4
        [price, quantity]   # Level 5
    ],
    "source": "API endpoint URL"
}
```

### **Component 3: Gamma Cluster Concentration**
**Function:** `compute_gamma_cluster(merged_df, atm_strike, window=2)`

**Window:** ATM ±2 strikes (5 strikes total)

**Calculation:**
```python
cluster = sum(|Gamma_CE| + |Gamma_PE|) for ATM ±2 strikes
score = normalized to 0-100
```

**Output:**
```python
{
    "available": True/False,
    "score": 0-100,
    "cluster": 0.12345  # Raw gamma sum
}
```

**Interpretation:**
- **High cluster (score > 70):** Gamma wall at ATM → Price pinning likely
- **Moderate cluster (30-70):** Some gamma concentration
- **Low cluster (< 30):** No gamma wall → Price can move freely

### **Component 4: OI Velocity & Acceleration**
**Function:** `compute_oi_velocity_acceleration(history, atm_strike, window_strikes=3)`

**Requirements:** At least 3 refresh points in history

**Window:** ATM ±3 strikes (7 strikes total)

**Calculation:**
```python
# For each strike in ATM cluster:
velocity_t1 = (OI_t1 - OI_t0) / time_delta_1
velocity_t2 = (OI_t2 - OI_t1) / time_delta_2
acceleration = (velocity_t2 - velocity_t1) / time_delta_2

# Aggregate across all strikes:
median_velocity = median(all_velocities)
median_acceleration = median(all_accelerations)

score = (0.6 × velocity_score) + (0.4 × acceleration_score)
# Normalized to 0-100
```

**Output:**
```python
{
    "available": True/False,
    "score": 0-100,
    "note": "OI speed-up detected in ATM cluster" if score > 60 else "OI changes are slow/steady"
}
```

**Interpretation:**
- **Score > 60:** OI accelerating - Real move with positioning
- **Score < 60:** Slow/steady OI - Possibly false breakout

### **Combined Moment Detector Output:**

```python
moment_metrics = {
    "momentum_burst": {
        "available": True,
        "score": 75,
        "note": "Momentum burst (energy) is rising"
    },
    "orderbook": {
        "available": True,
        "pressure": 0.35,  # +35% buy pressure
        "buy_qty": 15000.0,
        "sell_qty": 10000.0
    },
    "gamma_cluster": {
        "available": True,
        "score": 65,
        "cluster": 0.0875
    },
    "oi_acceleration": {
        "available": True,
        "score": 70,
        "note": "OI speed-up detected in ATM cluster"
    }
}
```

**Composite Moment Score:**
```python
composite = (
    0.40 × momentum_burst_score +
    0.20 × (orderbook_pressure normalized to 0-100) +
    0.25 × gamma_cluster_score +
    0.15 × oi_acceleration_score
)
```

**Overall Interpretation:**
- **Composite > 70:** HIGH CONFIDENCE - Real move confirmed
- **Composite 50-70:** MODERATE - Move has some conviction
- **Composite < 50:** LOW - Possibly fake move or noise

---

## 3️⃣ MARKET DEPTH / ORDERBOOK DATA

**Function:** `get_nifty_orderbook_depth()`

**Data Source:** Dhan API
- Endpoint 1: `/v2/marketfeed/quotes`
- Endpoint 2: `/v2/marketfeed/depth`

**Instrument:** NIFTY Index (IDX_I: 13)

**Refresh Rate:** Every 60 seconds (with auto-refresh)

### **Available Data:**

#### **Buy Side (Bid) - 5 Levels:**
```python
[
    [24503.50, 1250],  # Level 1: Price, Quantity
    [24503.00, 875],   # Level 2
    [24502.50, 1050],  # Level 3
    [24502.00, 725],   # Level 4
    [24501.50, 950]    # Level 5
]
```

#### **Sell Side (Ask) - 5 Levels:**
```python
[
    [24504.00, 900],   # Level 1: Price, Quantity
    [24504.50, 1100],  # Level 2
    [24505.00, 825],   # Level 3
    [24505.50, 1200],  # Level 4
    [24506.00, 775]    # Level 5
]
```

### **Derived Metrics:**

1. **Total Buy Quantity (5 levels):** Sum of all bid quantities
2. **Total Sell Quantity (5 levels):** Sum of all ask quantities
3. **Orderbook Pressure:** (Buy - Sell) / (Buy + Sell)
4. **Bid-Ask Imbalance:** Buy_Qty / Sell_Qty ratio
5. **Spread:** Best Ask - Best Bid
6. **Mid Price:** (Best Bid + Best Ask) / 2

### **Usage in XGBoost:**

Currently **PARTIALLY** integrated:
- ✅ `orderbook_pressure` (1 feature)
- ❌ Buy/Sell quantities separately (NOT integrated)
- ❌ Bid-Ask spread (NOT integrated)
- ❌ Level-wise breakdown (NOT integrated)

**OPPORTUNITY:** Add 6+ more features from market depth data

---

## 4️⃣ EXPIRY SPIKE DETECTOR

**Function:** `detect_expiry_spikes(merged_df, spot, atm_strike, days_to_expiry, expiry_date_str)`

**Purpose:** Detect unusual activity near expiry

**Key Metrics:**

1. **Days to Expiry:**
   - Calculated from current date to next Thursday
   - Critical when ≤ 2 days

2. **Expiry Week Detection:**
   - Flags if current week contains expiry
   - Increased volatility expected

3. **Pin Risk Assessment:**
   - Checks if price is near high OI strikes
   - Max pain calculation

4. **Gamma Spike Detection:**
   - Monitors gamma concentration near expiry
   - Explosive moves possible if gamma flips

**Output Structure:**
```python
{
    "days_to_expiry": 3,
    "is_expiry_week": True,
    "expiry_date": "2024-12-19",
    "pin_risk_detected": False,
    "max_pain_strike": 24500,
    "gamma_spike_risk": "MODERATE"
}
```

---

## 5️⃣ ENHANCED OI/PCR ANALYTICS

### **Available Metrics:**

1. **Put-Call Ratio (PCR):**
   - Total PE OI / Total CE OI
   - ATM PCR (±2 strikes)
   - Full chain PCR

2. **OI Buildup/Unwinding:**
   - CE OI Change (positive/negative)
   - PE OI Change (positive/negative)
   - Net OI Change

3. **Strike-wise OI Concentration:**
   - Max CE OI strike
   - Max PE OI strike
   - OI distribution across strikes

4. **OI Change Patterns:**
   - Bullish: PE buildup + CE unwinding
   - Bearish: CE buildup + PE unwinding
   - Neutral: Both buildup or both unwinding

5. **Volume/OI Ratio:**
   - High ratio = Fresh interest
   - Low ratio = Existing positions

6. **IV Analysis:**
   - Average IV (CE vs PE)
   - IV skew across strikes
   - IV percentile (historical context)

---

## 📊 COMPLETE DATA STRUCTURE

### **Full Output from NiftyOptionScreener.py:**

```python
{
    # ========== BASIC INFO ==========
    "spot_price": 24567.85,
    "atm_strike": 24500,
    "strike_gap": 50,
    "expiry_date": "2024-12-19",
    "days_to_expiry": 3,
    "timestamp": "2024-12-17 14:30:25",

    # ========== ATM BIAS (12 METRICS) ==========
    "atm_bias": {
        "total_score": 0.45,  # Normalized -1 to +1
        "verdict": "🐂 BULLISH",
        "verdict_color": "#00ff88",
        "bias_scores": {
            "OI_Bias": 0.5,
            "ChgOI_Bias": 1.0,
            "Volume_Bias": 0.5,
            "Delta_Bias": -0.5,
            "Gamma_Bias": 0.5,
            "Premium_Bias": 0.5,
            "IV_Bias": 1.0,
            "Delta_Exposure_Bias": 0.5,
            "Gamma_Exposure_Bias": -0.5,
            "IV_Skew_Bias": 0.5,
            "OI_Change_Bias": 0.5
        },
        "metrics": {
            "ce_oi": 12500000,
            "pe_oi": 18750000,
            "ce_chg": 250000,
            "pe_chg": 500000,
            "ce_vol": 125000,
            "pe_vol": 187500,
            "net_delta": 0.235,
            "net_gamma": -0.045,
            "ce_iv": 18.5,
            "pe_iv": 21.2,
            "delta_exposure": 875000,
            "gamma_exposure": -325000
        }
    },

    # ========== MOMENT DETECTOR (4 COMPONENTS) ==========
    "moment_metrics": {
        "momentum_burst": {
            "available": True,
            "score": 75,
            "note": "Momentum burst (energy) is rising"
        },
        "orderbook": {
            "available": True,
            "pressure": 0.35,
            "buy_qty": 15000.0,
            "sell_qty": 10000.0
        },
        "gamma_cluster": {
            "available": True,
            "score": 65,
            "cluster": 0.0875
        },
        "oi_acceleration": {
            "available": True,
            "score": 70,
            "note": "OI speed-up detected in ATM cluster"
        }
    },

    # ========== MARKET DEPTH ==========
    "market_depth": {
        "buy": [
            [24503.50, 1250],
            [24503.00, 875],
            [24502.50, 1050],
            [24502.00, 725],
            [24501.50, 950]
        ],
        "sell": [
            [24504.00, 900],
            [24504.50, 1100],
            [24505.00, 825],
            [24505.50, 1200],
            [24506.00, 775]
        ]
    },

    # ========== EXPIRY SPIKE ==========
    "expiry_spike": {
        "days_to_expiry": 3,
        "is_expiry_week": True,
        "pin_risk_detected": False,
        "max_pain_strike": 24500,
        "gamma_spike_risk": "MODERATE"
    },

    # ========== OI/PCR ANALYTICS ==========
    "oi_pcr_analytics": {
        "total_ce_oi": 45000000,
        "total_pe_oi": 52500000,
        "pcr": 1.167,
        "atm_pcr": 1.50,
        "max_ce_oi_strike": 25000,
        "max_pe_oi_strike": 24000,
        "oi_buildup": "BULLISH"  # PE buildup
    },

    # ========== GREEKS (PER STRIKE) ==========
    "strikes": [
        {
            "strike": 24500,
            "ce_ltp": 125.50,
            "pe_ltp": 87.25,
            "ce_oi": 2500000,
            "pe_oi": 3750000,
            "ce_chg_oi": 50000,
            "pe_chg_oi": 125000,
            "ce_volume": 25000,
            "pe_volume": 37500,
            "ce_iv": 18.5,
            "pe_iv": 21.2,
            "ce_delta": 0.52,
            "pe_delta": -0.48,
            "ce_gamma": 0.015,
            "pe_gamma": 0.015,
            "ce_theta": -12.5,
            "pe_theta": -10.2,
            "ce_vega": 8.5,
            "pe_vega": 8.5
        },
        # ... all other strikes
    ]
}
```

---

## 🎯 INTEGRATION WITH XGBOOST - CURRENT STATUS

### **Currently Integrated (8 features):**

From `src/xgboost_ml_analyzer.py` lines 282-298:

```python
features['momentum_burst'] = option_screener_data.get('momentum_burst', 0)
features['orderbook_pressure'] = option_screener_data.get('orderbook_pressure', 0)
features['gamma_cluster_concentration'] = option_screener_data.get('gamma_cluster', 0)
features['oi_acceleration'] = option_screener_data.get('oi_acceleration', 0)
features['expiry_spike_detected'] = 1 if option_screener_data.get('expiry_spike', False) else 0
features['net_vega_exposure'] = option_screener_data.get('net_vega_exposure', 0)
features['skew_ratio'] = option_screener_data.get('skew_ratio', 0)
features['atm_vol_premium'] = option_screener_data.get('atm_vol_premium', 0)
```

### **NOT YET INTEGRATED (50+ features):**

❌ **ATM Bias Metrics (11 scores):**
- OI_Bias, ChgOI_Bias, Volume_Bias
- Delta_Bias, Gamma_Bias, Premium_Bias
- IV_Bias, Delta_Exposure_Bias, Gamma_Exposure_Bias
- IV_Skew_Bias, OI_Change_Bias

❌ **ATM Bias Raw Metrics (12 values):**
- CE/PE OI, CE/PE Change, CE/PE Volume
- Net Delta, Net Gamma, CE/PE IV
- Delta Exposure, Gamma Exposure

❌ **ATM Bias Aggregate (2 features):**
- Total Score (-1 to +1)
- Verdict encoded (Bullish/Bearish/Neutral)

❌ **Market Depth Details (10 features):**
- Total Buy Quantity (5 levels)
- Total Sell Quantity (5 levels)
- Level 1, 2, 3, 4, 5 buy quantities
- Level 1, 2, 3, 4, 5 sell quantities
- Bid-Ask spread
- Mid price

❌ **Expiry Spike Details (4 features):**
- Days to expiry
- Is expiry week (binary)
- Pin risk detected (binary)
- Gamma spike risk (encoded)

❌ **OI/PCR Details (6 features):**
- Total CE/PE OI
- ATM PCR
- Full chain PCR
- Max pain strike distance
- OI buildup pattern (encoded)

❌ **Greeks Aggregates (8 features):**
- Total Delta (CE + PE)
- Total Gamma (CE + PE)
- Total Theta (CE + PE)
- Total Vega (CE + PE)
- ATM Delta, Gamma, Theta, Vega

---

## 📈 RECOMMENDATION FOR XGBOOST

### **HIGH PRIORITY FEATURES TO ADD (+25 features):**

1. **ATM Bias (13 features):**
   - All 11 individual bias scores
   - Total normalized score
   - Verdict encoded

2. **Market Depth (5 features):**
   - Total buy quantity
   - Total sell quantity
   - Orderbook pressure (already have)
   - Bid-ask spread
   - Buy/sell ratio

3. **Expiry Context (4 features):**
   - Days to expiry
   - Is expiry week
   - Pin risk detected
   - Gamma spike risk encoded

4. **OI/PCR Advanced (3 features):**
   - ATM PCR
   - Full chain PCR
   - OI buildup pattern

### **TOTAL NEW FEATURES: +25**
### **NEW TOTAL: 86 + 25 = 111 features**

---

## 🚀 IMPLEMENTATION PLAN UPDATE

Add to Phase 1.4 in main plan:

```python
# ========== ATM BIAS FEATURES (13) ==========
if 'atm_bias' in option_screener_data:
    atm = option_screener_data['atm_bias']

    # 11 individual bias scores
    for bias_name, score in atm['bias_scores'].items():
        features[f'atm_{bias_name.lower()}'] = score

    # Total score
    features['atm_total_score'] = atm['total_score']

    # Verdict encoded
    verdict_map = {"🐂 BULLISH": 2, "🐂 Mild Bullish": 1, "⚖️ NEUTRAL": 0,
                   "🐻 Mild Bearish": -1, "🐻 BEARISH": -2}
    features['atm_verdict'] = verdict_map.get(atm['verdict'], 0)

# ========== MARKET DEPTH FEATURES (5) ==========
if 'market_depth' in option_screener_data:
    md = option_screener_data['market_depth']

    buy_qty = sum([level[1] for level in md['buy'][:5]])
    sell_qty = sum([level[1] for level in md['sell'][:5]])

    features['depth_buy_qty'] = buy_qty
    features['depth_sell_qty'] = sell_qty
    features['depth_pressure'] = (buy_qty - sell_qty) / (buy_qty + sell_qty) if (buy_qty + sell_qty) > 0 else 0
    features['depth_spread'] = md['sell'][0][0] - md['buy'][0][0]
    features['depth_buy_sell_ratio'] = buy_qty / sell_qty if sell_qty > 0 else 1.0

# ========== EXPIRY CONTEXT FEATURES (4) ==========
if 'expiry_spike' in option_screener_data:
    es = option_screener_data['expiry_spike']

    features['days_to_expiry'] = es['days_to_expiry']
    features['is_expiry_week'] = 1 if es['is_expiry_week'] else 0
    features['pin_risk_detected'] = 1 if es['pin_risk_detected'] else 0

    gamma_risk_map = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "EXTREME": 3}
    features['gamma_spike_risk'] = gamma_risk_map.get(es['gamma_spike_risk'], 0)

# ========== OI/PCR ADVANCED FEATURES (3) ==========
if 'oi_pcr_analytics' in option_screener_data:
    oi = option_screener_data['oi_pcr_analytics']

    features['atm_pcr'] = oi.get('atm_pcr', 1.0)
    features['full_chain_pcr'] = oi.get('pcr', 1.0)

    buildup_map = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0, "MIXED": 0}
    features['oi_buildup_pattern'] = buildup_map.get(oi.get('oi_buildup'), 0)
```

---

## ✅ SUMMARY

**Tab 8 (NIFTY Option Screener v7.0) provides:**

- ✅ **12 ATM Bias Metrics** (comprehensive seller's perspective analysis)
- ✅ **4 Moment Detector Components** (validates if move is real)
- ✅ **Market Depth/Orderbook** (5 levels buy/sell from Dhan API)
- ✅ **Expiry Spike Detection** (gamma risk, pin risk, max pain)
- ✅ **Enhanced OI/PCR Analytics** (buildup/unwinding patterns)

**Current XGBoost Integration:** 8 features
**Available but NOT integrated:** 50+ features
**Recommended to add:** 25 high-priority features

**This will bring total XGBoost features to 111+ and significantly improve signal quality!**
