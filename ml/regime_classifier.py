"""
XGBoost Regime Classifier
ML-based market regime prediction using XGBoost
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle
import os


class RegimeClassifier:
    """
    XGBoost-based market regime classifier

    Note: This implementation provides the infrastructure for XGBoost classification.
    To use XGBoost, install it: pip install xgboost
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize regime classifier

        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.feature_names = None
        self.label_encoder = None
        self.is_trained = False

        # Try to import XGBoost
        try:
            import xgboost as xgb
            self.xgb = xgb
            self.xgb_available = True
        except ImportError:
            self.xgb = None
            self.xgb_available = False
            print("Warning: XGBoost not installed. Install with: pip install xgboost")

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def prepare_training_data(self, df: pd.DataFrame, features_df: pd.DataFrame,
                             forward_bars: int = 15, threshold: float = 0.01) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with labels

        Args:
            df: DataFrame with OHLCV data
            features_df: DataFrame with extracted features
            forward_bars: Number of bars to look forward for labeling
            threshold: Price change threshold for labeling

        Returns:
            Tuple of (features, labels)
        """
        labels = []

        for i in range(len(df) - forward_bars):
            current_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + forward_bars]
            future_return = (future_price / current_price) - 1

            # Multi-class labeling based on forward returns
            if future_return > threshold * 1.5:
                labels.append('STRONG_UPTREND')
            elif future_return > threshold * 0.5:
                labels.append('WEAK_UPTREND')
            elif future_return < -threshold * 1.5:
                labels.append('STRONG_DOWNTREND')
            elif future_return < -threshold * 0.5:
                labels.append('WEAK_DOWNTREND')
            else:
                labels.append('RANGING')

        # Trim features to match labels
        features_trimmed = features_df.iloc[:len(labels)].copy()
        labels_series = pd.Series(labels, index=features_trimmed.index)

        return features_trimmed, labels_series

    def train(self, X: pd.DataFrame, y: pd.Series, params: Optional[Dict] = None) -> Dict:
        """
        Train XGBoost classifier

        Args:
            X: Feature DataFrame
            y: Labels Series
            params: XGBoost parameters

        Returns:
            Training metrics
        """
        if not self.xgb_available:
            return {
                'error': 'XGBoost not available',
                'message': 'Install XGBoost with: pip install xgboost'
            }

        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Handle missing values
        X_clean = X.fillna(X.mean())

        # Default XGBoost parameters
        if params is None:
            params = {
                'objective': 'multi:softmax',
                'num_class': len(self.label_encoder.classes_),
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'mlogloss'
            }

        # Create DMatrix
        dtrain = self.xgb.DMatrix(X_clean, label=y_encoded)

        # Train model
        self.model = self.xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            verbose_eval=False
        )

        self.is_trained = True

        # Calculate training accuracy
        predictions = self.model.predict(dtrain)
        accuracy = np.mean(predictions == y_encoded)

        return {
            'accuracy': accuracy,
            'num_samples': len(X),
            'num_features': len(self.feature_names),
            'classes': list(self.label_encoder.classes_)
        }

    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Predict regime for new data

        Args:
            X: Feature DataFrame

        Returns:
            Dict with prediction, probabilities, and confidence
        """
        if not self.is_trained or self.model is None:
            # Fall back to rule-based prediction
            return {
                'regime': 'UNCERTAIN',
                'confidence': 0.0,
                'probabilities': {},
                'method': 'fallback'
            }

        # Ensure features match training
        X_aligned = X[self.feature_names].copy()
        X_clean = X_aligned.fillna(X_aligned.mean())

        # Create DMatrix
        dtest = self.xgb.DMatrix(X_clean)

        # Predict probabilities
        probs = self.model.predict(dtest, output_margin=False)

        # Get prediction and confidence
        if len(probs.shape) == 1:
            # Single class prediction
            prediction_idx = int(probs[0])
            confidence = 1.0
        else:
            # Multi-class probabilities
            prediction_idx = np.argmax(probs[0])
            confidence = float(probs[0][prediction_idx])

        regime = self.label_encoder.inverse_transform([prediction_idx])[0]

        # Create probability dict
        prob_dict = {}
        if len(probs.shape) > 1:
            for i, class_name in enumerate(self.label_encoder.classes_):
                prob_dict[class_name] = float(probs[0][i])

        return {
            'regime': regime,
            'confidence': confidence,
            'probabilities': prob_dict,
            'method': 'xgboost'
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained or self.model is None:
            return pd.DataFrame()

        importance = self.model.get_score(importance_type='gain')

        # Create DataFrame
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)

        return importance_df

    def save_model(self, path: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, path: str):
        """Load trained model from file"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = True


class HybridRegimeDetector:
    """
    Hybrid regime detector combining rule-based and ML approaches
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize hybrid detector

        Args:
            model_path: Path to saved XGBoost model
        """
        from ml.market_regime_detector import MarketRegimeDetector
        from ml.feature_extractor import FeatureExtractor

        self.rule_based_detector = MarketRegimeDetector()
        self.ml_classifier = RegimeClassifier(model_path)
        self.feature_extractor = FeatureExtractor()

    def detect_regime(self, df: pd.DataFrame, indicator_data: Dict) -> Dict:
        """
        Detect regime using both rule-based and ML methods

        Args:
            df: DataFrame with OHLCV data
            indicator_data: Dict with indicator outputs

        Returns:
            Combined regime detection result
        """
        # Get rule-based detection
        rule_based_result = self.rule_based_detector.detect_regime(df, indicator_data)

        # Try ML prediction if available
        if self.ml_classifier.is_trained:
            features = self.feature_extractor.extract_features(df, indicator_data)
            ml_result = self.ml_classifier.predict(features.tail(1))

            # Combine predictions (weighted average)
            # If both agree, high confidence
            # If disagree, use rule-based with lower confidence
            if ml_result['regime'] == rule_based_result['regime']:
                combined_confidence = (rule_based_result['confidence'] + ml_result['confidence']) / 2
                combined_confidence = min(combined_confidence * 1.2, 1.0)  # Boost when agreement
            else:
                combined_confidence = rule_based_result['confidence'] * 0.7  # Reduce when disagreement

            result = rule_based_result.copy()
            result['confidence'] = combined_confidence
            result['ml_prediction'] = ml_result['regime']
            result['ml_confidence'] = ml_result['confidence']
            result['ml_probabilities'] = ml_result.get('probabilities', {})
            result['method'] = 'hybrid'
        else:
            result = rule_based_result.copy()
            result['method'] = 'rule_based'

        return result
