# 🏗️ System Architecture

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     ULTIMATE TRADING APP                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │      Streamlit Web Interface            │
         │  (User Input & Visualization Layer)     │
         └────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
┌───────────────────────┐         ┌─────────────────────────┐
│   DhanHQ API Layer    │         │   yfinance (Backup)     │
│                       │         │                         │
│ • Intraday Data       │         │ • Historical Data       │
│ • LTP (Live Price)    │         │ • Fallback Source       │
│ • Market Quote        │         │                         │
└───────────────────────┘         └─────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Processing Layer                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Pandas     │  │    NumPy     │  │   DateTime   │     │
│  │ DataFrames   │  │ Calculations │  │  Time Series │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Indicator Calculation                      │
│                                                              │
│  ┌────────────────────────────────────────────────┐         │
│  │  1. Volume Order Blocks (VOB)                  │         │
│  │     • EMA Crossovers                           │         │
│  │     • ATR Filtering                            │         │
│  │     • Volume Accumulation                      │         │
│  │     • Overlap Removal                          │         │
│  └────────────────────────────────────────────────┘         │
│                                                              │
│  ┌────────────────────────────────────────────────┐         │
│  │  2. HTF Support/Resistance                     │         │
│  │     • Time Resampling (10T, 15T)               │         │
│  │     • Pivot High/Low Detection                 │         │
│  │     • Level Validation                         │         │
│  └────────────────────────────────────────────────┘         │
│                                                              │
│  ┌────────────────────────────────────────────────┐         │
│  │  3. Volumatic VIDYA                            │         │
│  │     • Variable Index Dynamic Average           │         │
│  │     • ATR-based Bands                          │         │
│  │     • Trend Detection                          │         │
│  └────────────────────────────────────────────────┘         │
│                                                              │
│  ┌────────────────────────────────────────────────┐         │
│  │  4. Ultimate RSI                               │         │
│  │     • Augmented RSI Calculation                │         │
│  │     • Signal Line Smoothing                    │         │
│  │     • OB/OS Detection                          │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Alert Management                          │
│                                                              │
│  ┌──────────────────────────────────────────────┐           │
│  │  Price Distance Calculations                 │           │
│  │   • VOB Proximity Check                      │           │
│  │   • HTF Level Proximity Check                │           │
│  │   • Alert Threshold (5 points)               │           │
│  └──────────────────────────────────────────────┘           │
│                                                              │
│  ┌──────────────────────────────────────────────┐           │
│  │  Cooling Period Management                   │           │
│  │   • 10-minute timer per alert type           │           │
│  │   • Prevent alert spam                       │           │
│  │   • Independent type tracking                │           │
│  └──────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Telegram Notification                        │
│                                                              │
│  • Rich formatted messages                                  │
│  • Price, Distance, Volume info                             │
│  • Timestamp                                                 │
│  • Alert type indicators                                     │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Chart Visualization                         │
│                                                              │
│  ┌────────────────────────────────────────────┐             │
│  │  Main Chart (Plotly)                       │             │
│  │   • Candlestick patterns                   │             │
│  │   • VOB shaded regions                     │             │
│  │   • HTF horizontal lines                   │             │
│  │   • VIDYA overlay                          │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
│  ┌────────────────────────────────────────────┐             │
│  │  RSI Subplot                               │             │
│  │   • Ultimate RSI line                      │             │
│  │   • Signal line                            │             │
│  │   • OB/OS zones                            │             │
│  └────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Auto-Refresh Loop                          │
│                                                              │
│  • 60-second timer                                          │
│  • Automatic data fetch                                      │
│  • Indicator recalculation                                   │
│  • Chart update                                              │
│  • Alert check                                               │
└─────────────────────────────────────────────────────────────┘
```

## Component Interaction

### 1. Data Sources
```
DhanHQ API (Primary)
├── Intraday Historical Data
│   ├── 1-minute candles
│   ├── OHLCV data
│   └── Last 5 days
├── Live Market Feed
│   ├── LTP (Last Traded Price)
│   ├── Real-time updates
│   └── Market status
└── Rate Limits
    ├── 25 req/second
    ├── 250 req/minute
    └── 7000 req/day
```

### 2. Indicator Pipeline
```
Raw OHLCV Data
    │
    ├─→ Volume Order Blocks
    │   ├─→ EMA(5) & EMA(18)
    │   ├─→ Crossover detection
    │   ├─→ Volume accumulation
    │   └─→ Overlap filtering
    │
    ├─→ HTF Support/Resistance
    │   ├─→ Resample to 10T & 15T
    │   ├─→ Pivot detection (length=5)
    │   └─→ Level validation
    │
    ├─→ VIDYA
    │   ├─→ Variable Index calculation
    │   ├─→ ATR bands
    │   ├─→ Trend detection
    │   └─→ Smoothing (15-period)
    │
    └─→ Ultimate RSI
        ├─→ Augmented RSI formula
        ├─→ Signal line (EMA 14)
        └─→ OB/OS detection
```

### 3. Alert System Flow
```
Price Update
    │
    ├─→ Check VOB Distance
    │   ├─→ Bullish blocks
    │   ├─→ Bearish blocks
    │   └─→ Within 5 points?
    │
    └─→ Check HTF Distance
        ├─→ 10T Support/Resistance
        ├─→ 15T Support/Resistance
        └─→ Within 5 points?
            │
            ├─→ YES → Check Cooling Period
            │          │
            │          ├─→ Active → Skip
            │          └─→ Expired → Send Alert
            │                         │
            │                         └─→ Update Timer
            │
            └─→ NO → Continue monitoring
```

## Key Classes and Responsibilities

### TelegramNotifier
```python
Responsibilities:
├── Send formatted messages
├── Track cooling periods
├── Manage alert timestamps
└── Handle API errors
```

### DhanDataFetcher
```python
Responsibilities:
├── Fetch intraday data
├── Get live prices (LTP)
├── Handle API authentication
└── Process responses
```

### VolumeOrderBlocks
```python
Responsibilities:
├── Calculate EMAs
├── Detect crossovers
├── Calculate ATR
├── Find supply/demand zones
├── Filter overlaps
└── Return block data
```

### HTFSupportResistance
```python
Responsibilities:
├── Resample to higher timeframes
├── Detect pivot highs
├── Detect pivot lows
├── Validate levels
└── Return level data
```

### VolumaticVIDYA
```python
Responsibilities:
├── Calculate VIDYA
├── Calculate ATR bands
├── Detect trend changes
├── Smooth values
└── Return indicator data
```

### UltimateRSI
```python
Responsibilities:
├── Calculate augmented RSI
├── Apply moving averages
├── Generate signal line
└── Return RSI data
```

### AlertManager
```python
Responsibilities:
├── Check price distances
├── Format alert messages
├── Trigger notifications
└── Coordinate with TelegramNotifier
```

## Performance Considerations

### Optimization Strategies
```
1. Data Caching
   ├── Store recent data in session state
   └── Reduce API calls

2. Efficient Calculations
   ├── Vectorized operations (NumPy/Pandas)
   ├── Avoid loops where possible
   └── Lazy evaluation

3. Rate Limit Management
   ├── Respect API limits
   ├── Implement exponential backoff
   └── Queue requests if needed

4. Chart Rendering
   ├── Plotly WebGL for large datasets
   ├── Downsample if >10000 points
   └── Progressive loading
```

## Security Best Practices

```
1. API Credentials
   ├── Never hardcode tokens
   ├── Use environment variables
   ├── Regenerate daily
   └── Rotate on suspicion

2. Telegram Bot
   ├── Keep token private
   ├── Restrict bot permissions
   └── Monitor usage

3. Data Handling
   ├── Validate all inputs
   ├── Sanitize user data
   └── Error handling everywhere
```

## Scalability Path

### Current Architecture
```
Single User → Single Instrument → Real-time
```

### Future Enhancements
```
1. Multi-Instrument Support
   ├── Parallel data fetching
   ├── Tabs or dropdown selector
   └── Watchlist management

2. Historical Analysis
   ├── Backtest indicators
   ├── Performance metrics
   └── Strategy optimization

3. Advanced Alerts
   ├── Complex conditions
   ├── Multi-indicator signals
   └── Custom alert types

4. Data Persistence
   ├── Database integration
   ├── Historical alert log
   └── Performance tracking
```

## Testing Strategy

```
Unit Tests
├── Indicator calculations
├── Alert logic
└── Data processing

Integration Tests
├── API connectivity
├── Telegram delivery
└── Chart rendering

End-to-End Tests
├── Complete workflow
├── Error scenarios
└── Edge cases
```

---

This architecture ensures:
- ✅ Modularity (easy to maintain)
- ✅ Scalability (can add features)
- ✅ Reliability (error handling)
- ✅ Performance (optimized calculations)
- ✅ Security (credential management)
