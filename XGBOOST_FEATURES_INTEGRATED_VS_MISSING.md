# 🎯 XGBOOST FEATURES - INTEGRATED VS MISSING

**Date:** 2025-12-17
**Current Total:** 86 features integrated
**Missing Total:** 60 features available
**Target Total:** 146 features

---

## ✅ CURRENTLY INTEGRATED (86 FEATURES)

### **1. PRICE FEATURES (5 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 1 | `price_current` | DataFrame | Current closing price |
| 2 | `price_change_1` | DataFrame | 1-period price change % |
| 3 | `price_change_5` | DataFrame | 5-period price change % |
| 4 | `price_change_20` | DataFrame | 20-period price change % |
| 5 | `atr` | DataFrame | Average True Range |
| 6 | `atr_pct` | DataFrame | ATR as % of price |

**Subtotal: 5 features** ✅

---

### **2. BIAS ANALYSIS PRO (13 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 7 | `bias_volume_delta` | Bias Analysis Pro | Volume Delta bias score |
| 8 | `bias_hvp` | Bias Analysis Pro | High Volume Profile bias |
| 9 | `bias_vob` | Bias Analysis Pro | Volume Order Blocks bias |
| 10 | `bias_order_blocks` | Bias Analysis Pro | Order Blocks bias |
| 11 | `bias_rsi` | Bias Analysis Pro | RSI bias score |
| 12 | `bias_dmi` | Bias Analysis Pro | DMI bias score |
| 13 | `bias_vidya` | Bias Analysis Pro | VIDYA bias score |
| 14 | `bias_mfi` | Bias Analysis Pro | Money Flow Index bias |
| 15 | `bias_close_vs_vwap` | Bias Analysis Pro | Close vs VWAP bias |
| 16 | `bias_price_vs_vwap` | Bias Analysis Pro | Price vs VWAP bias |
| 17 | `bias_weighted_stocks_daily` | Bias Analysis Pro | Weighted Stocks Daily bias |
| 18 | `bias_weighted_stocks_15m` | Bias Analysis Pro | Weighted Stocks 15min bias |
| 19 | `bias_weighted_stocks_1h` | Bias Analysis Pro | Weighted Stocks 1hour bias |

**Subtotal: 13 features** ✅

---

### **3. VOLATILITY REGIME (9 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 20 | `vix_level` | Volatility Result | India VIX level |
| 21 | `vix_percentile` | Volatility Result | VIX historical percentile |
| 22 | `atr_percentile` | Volatility Result | ATR historical percentile |
| 23 | `iv_rv_ratio` | Volatility Result | Implied Vol / Realized Vol ratio |
| 24 | `regime_strength` | Volatility Result | Volatility regime strength |
| 25 | `compression_score` | Volatility Result | Volatility compression score |
| 26 | `gamma_flip` | Volatility Result | Gamma flip detected (binary) |
| 27 | `expiry_week` | Volatility Result | Is expiry week (binary) |
| 28 | `volatility_regime` | Volatility Result | Regime encoded (1-5) |

**Subtotal: 9 features** ✅

---

### **4. OI TRAP DETECTION (5 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 29 | `trap_detected` | OI Trap Result | Trap detected (binary) |
| 30 | `trap_probability` | OI Trap Result | Trap probability (0-1) |
| 31 | `retail_trap_score` | OI Trap Result | Retail trap score |
| 32 | `oi_manipulation_score` | OI Trap Result | OI manipulation score |
| 33 | `trapped_direction` | OI Trap Result | Trapped direction encoded |

**Subtotal: 5 features** ✅

---

### **5. CVD & DELTA (8 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 34 | `cvd_value` | CVD Result | Cumulative Volume Delta |
| 35 | `delta_imbalance` | CVD Result | Delta imbalance score |
| 36 | `orderflow_strength` | CVD Result | Order flow strength |
| 37 | `delta_divergence` | CVD Result | Delta divergence detected (binary) |
| 38 | `delta_absorption` | CVD Result | Delta absorption detected (binary) |
| 39 | `delta_spike` | CVD Result | Delta spike detected (binary) |
| 40 | `institutional_sweep` | CVD Result | Institutional sweep (binary) |
| 41 | `cvd_bias` | CVD Result | CVD bias encoded (-1/0/+1) |

**Subtotal: 8 features** ✅

---

### **6. INSTITUTIONAL/RETAIL (5 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 42 | `institutional_confidence` | Participant Result | Institutional confidence score |
| 43 | `retail_confidence` | Participant Result | Retail confidence score |
| 44 | `smart_money` | Participant Result | Smart money detected (binary) |
| 45 | `dumb_money` | Participant Result | Dumb money detected (binary) |
| 46 | `dominant_participant` | Participant Result | Dominant participant encoded |

**Subtotal: 5 features** ✅

---

### **7. LIQUIDITY GRAVITY (7 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 47 | `primary_target` | Liquidity Result | Primary liquidity target price |
| 48 | `gravity_strength` | Liquidity Result | Gravity strength score |
| 49 | `num_support_zones` | Liquidity Result | Number of support zones |
| 50 | `num_resistance_zones` | Liquidity Result | Number of resistance zones |
| 51 | `num_hvn_zones` | Liquidity Result | Number of HVN zones |
| 52 | `num_fvg` | Liquidity Result | Number of Fair Value Gaps |
| 53 | `num_gamma_walls` | Liquidity Result | Number of gamma walls |
| 54 | `target_distance_pct` | Liquidity Result | Distance to target % |

**Subtotal: 7 features** ✅

---

### **8. MONEY FLOW PROFILE (8 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 55 | `mfp_poc_price` | Money Flow Signals | Point of Control price |
| 56 | `mfp_bullish_pct` | Money Flow Signals | Bullish volume % |
| 57 | `mfp_bearish_pct` | Money Flow Signals | Bearish volume % |
| 58 | `mfp_distance_from_poc_pct` | Money Flow Signals | Distance from POC % |
| 59 | `mfp_num_hv_levels` | Money Flow Signals | High volume level count |
| 60 | `mfp_num_lv_levels` | Money Flow Signals | Low volume level count |
| 61 | `mfp_sentiment` | Money Flow Signals | Sentiment encoded (-1/0/+1) |
| 62 | `mfp_price_position` | Money Flow Signals | Price position encoded |

**Subtotal: 8 features** ✅

---

### **9. DELTAFLOW PROFILE (10 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 63 | `dfp_overall_delta` | DeltaFlow Signals | Overall delta % |
| 64 | `dfp_bull_pct` | DeltaFlow Signals | Buy volume % |
| 65 | `dfp_bear_pct` | DeltaFlow Signals | Sell volume % |
| 66 | `dfp_poc_price` | DeltaFlow Signals | Point of Control price |
| 67 | `dfp_distance_from_poc_pct` | DeltaFlow Signals | Distance from POC % |
| 68 | `dfp_num_strong_buy` | DeltaFlow Signals | Strong buy levels count |
| 69 | `dfp_num_strong_sell` | DeltaFlow Signals | Strong sell levels count |
| 70 | `dfp_num_absorption` | DeltaFlow Signals | Absorption zones count |
| 71 | `dfp_sentiment` | DeltaFlow Signals | Sentiment encoded (-2 to +2) |
| 72 | `dfp_price_position` | DeltaFlow Signals | Price position encoded |

**Subtotal: 10 features** ✅

---

### **10. ML REGIME (4 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 73 | `trend_strength` | ML Regime Result | Trend strength score |
| 74 | `regime_confidence` | ML Regime Result | Regime confidence |
| 75 | `market_regime` | ML Regime Result | Market regime encoded |
| 76 | `volatility_state` | ML Regime Result | Volatility state encoded |

**Subtotal: 4 features** ✅

---

### **11. OPTION CHAIN BASIC (3 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 77 | `total_ce_oi` | Option Chain | Total Call Open Interest |
| 78 | `total_pe_oi` | Option Chain | Total Put Open Interest |
| 79 | `pcr` | Option Chain | Put-Call Ratio |

**Subtotal: 3 features** ✅

---

### **12. SENTIMENT (1 feature)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 80 | `overall_sentiment` | Sentiment Score | Overall sentiment score |

**Subtotal: 1 feature** ✅

---

### **13. OPTION SCREENER (8 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 81 | `momentum_burst` | Option Screener | Momentum burst score (0-100) |
| 82 | `orderbook_pressure` | Option Screener | Orderbook pressure (-1 to +1) |
| 83 | `gamma_cluster_concentration` | Option Screener | Gamma cluster score (0-100) |
| 84 | `oi_acceleration` | Option Screener | OI acceleration score (0-100) |
| 85 | `expiry_spike_detected` | Option Screener | Expiry spike (binary) |
| 86 | `net_vega_exposure` | Option Screener | Net vega exposure |
| 87 | `skew_ratio` | Option Screener | IV skew ratio |
| 88 | `atm_vol_premium` | Option Screener | ATM volatility premium |

**Subtotal: 8 features** ✅

---

## **TOTAL INTEGRATED: 86 FEATURES** ✅

---
---

## ❌ MISSING FEATURES (60 FEATURES)

### **1. OVERALL MARKET SENTIMENT - TAB 1 (5 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 89 | `stock_performance_bias` | Tab 1 | Stock advances/declines bias (-100 to +100) |
| 90 | `technical_indicators_bias` | Tab 1 | Aggregated 13 indicators bias (-100 to +100) |
| 91 | `atm_verdict_bias` | Tab 1 | ATM ±2 strikes verdict bias (-100 to +100) |
| 92 | `pcr_oi_bias` | Tab 1 | PCR/OI analysis bias (-100 to +100) |
| 93 | `sector_rotation_bias` | Tab 1 | Sector rotation bias (-100 to +100) |

**Subtotal: 5 features** ❌

---

### **2. INDIA VIX - TAB 9 (4 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 94 | `vix_current` | Tab 9 | Current VIX value |
| 95 | `vix_score` | Tab 9 | VIX score (-100 to +100) |
| 96 | `vix_sentiment` | Tab 9 | VIX sentiment encoded |
| 97 | `vix_bias` | Tab 9 | VIX bias encoded (-1/0/+1) |

**Subtotal: 4 features** ❌

---

### **3. SECTOR ROTATION - TAB 9 (5 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 98 | `sector_breadth` | Tab 9 | % of bullish sectors (0-100) |
| 99 | `sector_score` | Tab 9 | Sector rotation score (-100 to +100) |
| 100 | `sector_bullish_count` | Tab 9 | Number of bullish sectors |
| 101 | `sector_bearish_count` | Tab 9 | Number of bearish sectors |
| 102 | `rotation_score` | Tab 9 | Rotation pattern score |

**Subtotal: 5 features** ❌

---

### **4. GLOBAL MARKETS - TAB 9 (3 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 103 | `sp500_change` | Tab 9 | S&P 500 change % |
| 104 | `nasdaq_change` | Tab 9 | Nasdaq change % |
| 105 | `nikkei_change` | Tab 9 | Nikkei 225 change % |

**Subtotal: 3 features** ❌

---

### **5. INTERMARKET - TAB 9 (2 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 106 | `crude_oil_change` | Tab 9 | Crude oil change % |
| 107 | `usdinr_change` | Tab 9 | USD/INR change % |

**Subtotal: 2 features** ❌

---

### **6. GAMMA SQUEEZE - TAB 9 (1 feature)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 108 | `gamma_squeeze_score` | Tab 9 | Gamma squeeze score (-100 to +100) |

**Subtotal: 1 feature** ❌

---

### **7. ATM BIAS (12 METRICS) - TAB 8 (13 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 109 | `atm_oi_bias` | Tab 8 | OI Bias (-1 to +1) |
| 110 | `atm_chgoi_bias` | Tab 8 | Change in OI Bias (-1 to +1) |
| 111 | `atm_volume_bias` | Tab 8 | Volume Bias (-1 to +1) |
| 112 | `atm_delta_bias` | Tab 8 | Delta Bias (-1 to +1) |
| 113 | `atm_gamma_bias` | Tab 8 | Gamma Bias (-1 to +1) |
| 114 | `atm_premium_bias` | Tab 8 | Premium Bias (-1 to +1) |
| 115 | `atm_iv_bias` | Tab 8 | IV Bias (-1 to +1) |
| 116 | `atm_delta_exposure_bias` | Tab 8 | Delta Exposure Bias (-1 to +1) |
| 117 | `atm_gamma_exposure_bias` | Tab 8 | Gamma Exposure Bias (-1 to +1) |
| 118 | `atm_iv_skew_bias` | Tab 8 | IV Skew Bias (-0.5 to +0.5) |
| 119 | `atm_oi_change_bias` | Tab 8 | OI Change Bias (-0.5 to +0.5) |
| 120 | `atm_total_score` | Tab 8 | Total ATM bias score (-1 to +1) |
| 121 | `atm_verdict` | Tab 8 | ATM verdict encoded (-2 to +2) |

**Subtotal: 13 features** ❌

---

### **8. MARKET DEPTH - TAB 8 (5 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 122 | `depth_buy_qty` | Tab 8 | Total buy quantity (5 levels) |
| 123 | `depth_sell_qty` | Tab 8 | Total sell quantity (5 levels) |
| 124 | `depth_pressure` | Tab 8 | Orderbook pressure (-1 to +1) |
| 125 | `depth_spread` | Tab 8 | Bid-Ask spread |
| 126 | `depth_buy_sell_ratio` | Tab 8 | Buy/Sell quantity ratio |

**Subtotal: 5 features** ❌

---

### **9. EXPIRY CONTEXT - TAB 8 (4 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 127 | `days_to_expiry` | Tab 8 | Days until expiry |
| 128 | `is_expiry_week` | Tab 8 | Is expiry week (binary) |
| 129 | `pin_risk_detected` | Tab 8 | Pin risk detected (binary) |
| 130 | `gamma_spike_risk` | Tab 8 | Gamma spike risk encoded (0-3) |

**Subtotal: 4 features** ❌

---

### **10. OI/PCR ADVANCED - TAB 8 (3 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 131 | `atm_pcr` | Tab 8 | ATM Put-Call Ratio |
| 132 | `full_chain_pcr` | Tab 8 | Full chain Put-Call Ratio |
| 133 | `oi_buildup_pattern` | Tab 8 | OI buildup pattern encoded (-1/0/+1) |

**Subtotal: 3 features** ❌

---

### **11. HTF SUPPORT/RESISTANCE - TAB 7 (4 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 134 | `num_htf_support` | Tab 7 | Number of HTF support levels |
| 135 | `num_htf_resistance` | Tab 7 | Number of HTF resistance levels |
| 136 | `htf_nearest_support_distance` | Tab 7 | Distance to nearest HTF support % |
| 137 | `htf_nearest_resistance_distance` | Tab 7 | Distance to nearest HTF resistance % |

**Subtotal: 4 features** ❌

---

### **12. VOLUME FOOTPRINT - TAB 7 (4 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 138 | `vf_poc_price` | Tab 7 | Volume Footprint POC price |
| 139 | `vf_value_area_high` | Tab 7 | Value Area High |
| 140 | `vf_value_area_low` | Tab 7 | Value Area Low |
| 141 | `vf_price_in_value_area` | Tab 7 | Price in value area (binary) |

**Subtotal: 4 features** ❌

---

### **13. LIQUIDITY SENTIMENT PROFILE - TAB 7 (3 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 142 | `lp_sentiment` | Tab 7 | Liquidity sentiment encoded |
| 143 | `lp_num_supply_zones` | Tab 7 | Number of supply zones |
| 144 | `lp_num_demand_zones` | Tab 7 | Number of demand zones |

**Subtotal: 3 features** ❌

---

### **14. FIBONACCI LEVELS - TAB 7 (4 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 145 | `fib_distance_23` | Tab 7 | Distance from 23.6% Fib level |
| 146 | `fib_distance_38` | Tab 7 | Distance from 38.2% Fib level |
| 147 | `fib_distance_50` | Tab 7 | Distance from 50% Fib level |
| 148 | `fib_distance_61` | Tab 7 | Distance from 61.8% Fib level |

**Subtotal: 4 features** ❌

---

### **15. PATTERNS - TAB 7 (2 features)**

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 149 | `num_bullish_patterns` | Tab 7 | Number of bullish patterns detected |
| 150 | `num_bearish_patterns` | Tab 7 | Number of bearish patterns detected |

**Subtotal: 2 features** ❌

---

## **TOTAL MISSING: 60 FEATURES** ❌

---

## 📊 SUMMARY BY CATEGORY

| Category | Integrated | Missing | Total |
|----------|-----------|---------|-------|
| **Price** | 5 ✅ | 0 | 5 |
| **Bias Analysis Pro** | 13 ✅ | 0 | 13 |
| **Volatility Regime** | 9 ✅ | 0 | 9 |
| **OI Trap** | 5 ✅ | 0 | 5 |
| **CVD & Delta** | 8 ✅ | 0 | 8 |
| **Institutional/Retail** | 5 ✅ | 0 | 5 |
| **Liquidity Gravity** | 7 ✅ | 0 | 7 |
| **Money Flow Profile** | 8 ✅ | 0 | 8 |
| **DeltaFlow Profile** | 10 ✅ | 0 | 10 |
| **ML Regime** | 4 ✅ | 0 | 4 |
| **Option Chain Basic** | 3 ✅ | 0 | 3 |
| **Sentiment** | 1 ✅ | 0 | 1 |
| **Option Screener** | 8 ✅ | 0 | 8 |
| **Overall Market Sentiment (Tab 1)** | 0 | 5 ❌ | 5 |
| **India VIX (Tab 9)** | 0 | 4 ❌ | 4 |
| **Sector Rotation (Tab 9)** | 0 | 5 ❌ | 5 |
| **Global Markets (Tab 9)** | 0 | 3 ❌ | 3 |
| **Intermarket (Tab 9)** | 0 | 2 ❌ | 2 |
| **Gamma Squeeze (Tab 9)** | 0 | 1 ❌ | 1 |
| **ATM Bias (Tab 8)** | 0 | 13 ❌ | 13 |
| **Market Depth (Tab 8)** | 0 | 5 ❌ | 5 |
| **Expiry Context (Tab 8)** | 0 | 4 ❌ | 4 |
| **OI/PCR Advanced (Tab 8)** | 0 | 3 ❌ | 3 |
| **HTF Support/Resistance (Tab 7)** | 0 | 4 ❌ | 4 |
| **Volume Footprint (Tab 7)** | 0 | 4 ❌ | 4 |
| **Liquidity Sentiment (Tab 7)** | 0 | 3 ❌ | 3 |
| **Fibonacci Levels (Tab 7)** | 0 | 4 ❌ | 4 |
| **Patterns (Tab 7)** | 0 | 2 ❌ | 2 |
| **TOTAL** | **86 ✅** | **60 ❌** | **146** |

---

## 🎯 PRIORITY BREAKDOWN

### **HIGH PRIORITY TO ADD (+20 features)**

1. **Overall Market Sentiment (5)** - Tab 1
2. **India VIX (4)** - Tab 9
3. **Sector Rotation (5)** - Tab 9
4. **ATM Bias Total Score (1)** - Tab 8
5. **Market Depth (5)** - Tab 8

### **MEDIUM PRIORITY (+25 features)**

6. **ATM Bias (11 individual metrics)** - Tab 8
7. **Global Markets (3)** - Tab 9
8. **Intermarket (2)** - Tab 9
9. **Expiry Context (4)** - Tab 8
10. **OI/PCR Advanced (3)** - Tab 8
11. **Gamma Squeeze (1)** - Tab 9

### **LOW PRIORITY (+15 features)**

12. **HTF Support/Resistance (4)** - Tab 7
13. **Volume Footprint (4)** - Tab 7
14. **Liquidity Sentiment (3)** - Tab 7
15. **Fibonacci Levels (4)** - Tab 7
16. **Patterns (2)** - Tab 7

---

## ✅ IMPLEMENTATION ORDER

### **Phase 1.1: Overall Market Sentiment (+5 features)**
Add features 89-93 from Tab 1

### **Phase 1.2: Enhanced Market Data (+15 features)**
Add features 94-108 from Tab 9

### **Phase 1.3: Advanced Chart Analysis (+15 features)**
Add features 134-150 from Tab 7

### **Phase 1.4: ATM Bias + Market Depth + Expiry (+25 features)**
Add features 109-133 from Tab 8

**Total After Phase 1: 146 FEATURES**

---

## 📝 NOTES

**Why are some features integrated and others not?**

- ✅ **Integrated:** Features that were part of the original AI module implementations (Volatility Regime, OI Trap, CVD, etc.) plus Money Flow and DeltaFlow that we just added
- ❌ **Missing:** Features from tabs that weren't originally connected to XGBoost (Tab 1, 7, 8, 9)

**Benefits of adding missing features:**
- Better macro context (VIX, sectors, global markets)
- Better micro context (ATM bias, market depth, orderbook)
- Better technical context (HTF, Fibonacci, patterns)
- Better overall signal quality and confidence

**Next step:** Add missing features in 4 sub-phases as outlined in main plan
