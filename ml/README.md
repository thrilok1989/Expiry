# Machine Learning Module for Trading System

This module provides AI-powered market regime detection and classification using indicator data.

## 🎯 Features

### 1. Market Regime Detection
- **Rule-Based Detection**: Uses indicator signals (BOS, CHOCH, S/R, etc.) to identify market regimes
- **XGBoost Classification** (optional): ML-based regime prediction
- **Hybrid Approach**: Combines both methods for higher accuracy

### 2. Regime Types
- **STRONG_UPTREND**: Clear bullish trend with strong momentum
- **WEAK_UPTREND**: Bullish bias with weaker momentum
- **RANGING**: Price oscillating between support/resistance
- **WEAK_DOWNTREND**: Bearish bias with weaker momentum
- **STRONG_DOWNTREND**: Clear bearish trend with strong momentum
- **REVERSAL_TO_UPTREND**: Potential trend reversal to upside
- **REVERSAL_TO_DOWNTREND**: Potential trend reversal to downside
- **UNCERTAIN**: No clear regime

### 3. Trading Recommendations
For each regime, the system provides:
- Position bias (LONG_ONLY, SHORT_ONLY, NEUTRAL, etc.)
- Recommended strategy (BUY_PULLBACKS, SELL_RALLIES, MEAN_REVERSION, etc.)
- Position size multiplier
- Stop loss width multiplier
- Allowed trade setups

## 📁 Module Structure

```
ml/
├── __init__.py                    # Module initialization
├── feature_extractor.py           # Converts indicators to ML features
├── market_regime_detector.py      # Rule-based regime detection
├── regime_classifier.py           # XGBoost classifier (optional)
└── README.md                      # This file
```

## 🚀 Usage

### Basic Usage (Rule-Based)

The Market Regime dashboard is automatically available in the **Advanced Chart** tab as the first tab.

```python
from ml.market_regime_detector import MarketRegimeDetector

# Initialize detector
detector = MarketRegimeDetector()

# Prepare indicator data
indicator_data = {
    'bos': bos_events,           # From AdvancedPriceAction
    'choch': choch_events,       # From AdvancedPriceAction
    'htf_sr': htf_levels,        # From HTFSupportResistance
    'order_blocks': ob_data,     # From VolumeOrderBlocks
    'rsi': rsi_signals           # From UltimateRSI
}

# Detect regime
result = detector.detect_regime(df, indicator_data)

print(f"Regime: {result['regime']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Strategy: {result['recommendations']['strategy']}")
```

### Advanced Usage (XGBoost - Optional)

To use XGBoost for regime prediction, first install it:

```bash
pip install xgboost scikit-learn
```

Then train a model:

```python
from ml.feature_extractor import FeatureExtractor
from ml.regime_classifier import RegimeClassifier

# Extract features
extractor = FeatureExtractor()
features = extractor.extract_features(df, indicator_data)

# Prepare training data
classifier = RegimeClassifier()
X, y = classifier.prepare_training_data(df, features)

# Train model
metrics = classifier.train(X, y)
print(f"Training accuracy: {metrics['accuracy']:.1%}")

# Save model
classifier.save_model('models/regime_classifier.pkl')

# Later, load and use
classifier = RegimeClassifier('models/regime_classifier.pkl')
prediction = classifier.predict(features.tail(1))
print(f"Predicted regime: {prediction['regime']}")
```

### Hybrid Approach

```python
from ml.regime_classifier import HybridRegimeDetector

# Initialize hybrid detector (uses both rule-based and ML)
hybrid = HybridRegimeDetector(model_path='models/regime_classifier.pkl')

# Detect regime
result = hybrid.detect_regime(df, indicator_data)

print(f"Regime: {result['regime']}")
print(f"ML Prediction: {result.get('ml_prediction', 'N/A')}")
print(f"Method: {result['method']}")  # 'hybrid' or 'rule_based'
```

## 📊 Feature Extraction

The `FeatureExtractor` converts raw indicator data into 50+ ML features:

### Price Features
- Price changes (1, 5, 15, 30 periods)
- Moving averages (SMA 20, 50)
- ATR and volatility metrics
- Consecutive higher highs/lower lows

### Volume Features
- Volume ratios
- Volume trends

### Order Block Features
- Distance to nearest bullish/bearish blocks
- Number of active blocks
- In-block indicators

### HTF S/R Features
- Distance to nearest support/resistance
- Confluence counts
- Near-level indicators

### BOS/CHOCH Features
- Recent BOS counts (bullish/bearish)
- Trend strength scores
- Last BOS direction
- CHOCH detection

### RSI Features
- RSI value
- Oversold/overbought flags
- Signal direction

### Pattern Features
- Bullish/bearish pattern detection

## 🎓 Training Your Own Model

To train a custom XGBoost model on your data:

```python
# 1. Collect historical data with indicators
# 2. Extract features
features = extractor.extract_features(df, indicator_data)

# 3. Prepare training labels
# Labels are created based on forward-looking returns
X, y = classifier.prepare_training_data(
    df,
    features,
    forward_bars=15,    # Look 15 bars ahead
    threshold=0.01       # 1% threshold for labeling
)

# 4. Train with custom parameters
params = {
    'objective': 'multi:softmax',
    'num_class': 5,
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8
}
metrics = classifier.train(X, y, params=params)

# 5. Save model
classifier.save_model('my_custom_model.pkl')

# 6. Check feature importance
importance_df = classifier.get_feature_importance()
print(importance_df.head(10))
```

## 📈 Integration with Trading Signals

The regime detection can be used to filter trading signals:

```python
# Get regime
regime_result = detector.detect_regime(df, indicator_data)
regime = regime_result['regime']
recommendations = regime_result['recommendations']

# Filter signals based on regime
if regime == 'STRONG_UPTREND':
    # Only take LONG signals
    if signal == 'BUY' and setup in recommendations['allowed_setups']:
        # Take trade with increased position size
        position_size = base_size * recommendations['position_size_multiplier']
        execute_trade('BUY', position_size)

elif regime == 'RANGING':
    # Use mean reversion strategy
    if price_near_support and signal == 'BUY':
        execute_trade('BUY', base_size * 0.8)  # Smaller size

elif regime in ['REVERSAL_TO_UPTREND', 'REVERSAL_TO_DOWNTREND']:
    # Wait for confirmation, reduce size
    if confirmation_count >= 2:
        position_size = base_size * 0.5
        execute_trade(signal, position_size)
```

## 🔧 Customization

### Adding New Features

Edit `feature_extractor.py`:

```python
def _add_custom_features(self, features, df, indicator_data):
    """Add your custom features"""
    # Example: Add your own indicator
    features['my_custom_indicator'] = calculate_my_indicator(df)
    return features
```

### Adding New Regimes

Edit `market_regime_detector.py` and `MarketRegime` enum:

```python
class MarketRegime(Enum):
    # ... existing regimes ...
    MY_CUSTOM_REGIME = "MY_CUSTOM_REGIME"
```

### Customizing Recommendations

Edit `_generate_recommendations()` in `market_regime_detector.py`:

```python
if regime == MarketRegime.MY_CUSTOM_REGIME.value:
    recommendations.update({
        'position_bias': 'CUSTOM',
        'strategy': 'CUSTOM_STRATEGY',
        # ... your custom settings ...
    })
```

## 📊 Dashboard

The Market Regime Dashboard is automatically integrated into the Advanced Chart tab and displays:

1. **Current Regime** with emoji indicators
2. **Confidence Score** (percentage)
3. **Volatility Regime** (HIGH/NORMAL/LOW)
4. **Regime Indicators Table**:
   - Trend Direction
   - Trend Strength
   - Ranging Status
   - Reversal Signals
5. **Trading Recommendations Table**:
   - Position Bias
   - Strategy
   - Position Size Multiplier
   - Stop Loss Multiplier
6. **Recommended Trade Setups** (bullet list)

## 🚨 Important Notes

1. **XGBoost is optional**: The system works with rule-based detection without XGBoost
2. **Training data**: You need sufficient historical data (500+ samples) for ML training
3. **Retraining**: Retrain the model periodically with recent data for best results
4. **Backtesting**: Always backtest regime-based strategies before live trading
5. **Confidence threshold**: Only act on high-confidence regime detections (>70%)

## 📝 Dependencies

### Required (already in project)
- pandas
- numpy

### Optional (for XGBoost)
- xgboost
- scikit-learn

Install optional dependencies:
```bash
pip install xgboost scikit-learn
```

## 🤝 Contributing

To improve the regime detection:
1. Collect more training data
2. Add new features to `feature_extractor.py`
3. Experiment with different XGBoost parameters
4. Share your results and models

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Market Regime Detection Papers](https://papers.ssrn.com)
- Trading strategies should be adapted based on regime for optimal performance
