"""
MASTER AI ORCHESTRATOR
Combines all advanced modules into a unified trading intelligence system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

# Import all modules
from volatility_regime import VolatilityRegimeDetector, VolatilityRegimeResult
from oi_trap_detection import OITrapDetector, OITrapResult
from cvd_delta_imbalance import CVDAnalyzer, CVDResult
from institutional_retail_detector import InstitutionalRetailDetector, MarketParticipantResult
from liquidity_gravity import LiquidityGravityAnalyzer, LiquidityGravityResult
from position_sizing_engine import PositionSizingEngine, PositionSizeResult
from risk_management_ai import RiskManagementAI, RiskManagementResult
from expectancy_model import ExpectancyCalculator, ExpectancyResult
from ml_market_regime import MLMarketRegimeDetector, MarketSummary, generate_market_summary, MLMarketRegimeResult
from xgboost_ml_analyzer import XGBoostMLAnalyzer, MLPredictionResult

logger = logging.getLogger(__name__)


@dataclass
class MasterAnalysisResult:
    """Complete analysis from all modules"""
    # Individual module results
    volatility_regime: VolatilityRegimeResult
    oi_trap: OITrapResult
    cvd: CVDResult
    participant: MarketParticipantResult
    liquidity: LiquidityGravityResult
    position_size: Optional[PositionSizeResult]
    risk_management: Optional[RiskManagementResult]
    expectancy: Optional[ExpectancyResult]
    ml_regime: MLMarketRegimeResult
    market_summary: MarketSummary
    xgboost_ml: MLPredictionResult  # XGBoost ML prediction

    # Master recommendation
    final_verdict: str  # "STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL", "NO TRADE"
    confidence: float  # 0-100
    risk_reward_ratio: float
    expected_win_probability: float
    trade_quality_score: float  # 0-100
    reasoning: List[str]


class MasterAIOrchestrator:
    """
    Master AI Orchestrator

    Combines ALL advanced modules:
    1. Volatility Regime Detection
    2. OI Trap Detection
    3. CVD Delta Imbalance
    4. Institutional vs Retail Detection
    5. Liquidity Gravity Analysis
    6. Dynamic Position Sizing
    7. Risk Management AI
    8. Expectancy Model
    9. ML Market Regime
    10. Market Summary
    """

    def __init__(
        self,
        account_size: float = 100000,
        max_risk_per_trade: float = 2.0
    ):
        """Initialize Master AI Orchestrator"""
        # Initialize all sub-modules
        self.volatility_detector = VolatilityRegimeDetector()
        self.oi_trap_detector = OITrapDetector()
        self.cvd_analyzer = CVDAnalyzer()
        self.participant_detector = InstitutionalRetailDetector()
        self.liquidity_analyzer = LiquidityGravityAnalyzer()
        self.position_sizer = PositionSizingEngine(account_size, max_risk_per_trade)
        self.risk_manager = RiskManagementAI()
        self.expectancy_calc = ExpectancyCalculator()
        self.ml_regime_detector = MLMarketRegimeDetector()
        self.xgboost_analyzer = XGBoostMLAnalyzer()

        # Thresholds
        self.min_trade_quality_score = 60
        self.min_confidence = 65

    def analyze_complete_market(
        self,
        df: pd.DataFrame,
        option_chain: Dict,
        vix_current: float,
        vix_history: pd.Series,
        instrument: str = "NIFTY",
        days_to_expiry: int = 5,
        historical_trades: Optional[pd.DataFrame] = None,
        trade_params: Optional[Dict] = None,
        option_screener_data: Optional[Dict] = None,
        bias_results: Optional[Dict] = None,
        sentiment_score: float = 0.0
    ) -> MasterAnalysisResult:
        """
        Complete market analysis combining ALL modules

        Args:
            df: OHLCV dataframe with indicators
            option_chain: Option chain data
            vix_current: Current India VIX
            vix_history: Historical VIX series
            instrument: Trading instrument
            days_to_expiry: Days until expiry
            historical_trades: Past trade history for expectancy
            trade_params: Dict with entry, stop, target prices
            option_screener_data: Option screener analysis data (momentum, gamma, etc.)
            bias_results: Technical bias analysis data (13 indicators from bias_analysis.py)
            sentiment_score: Overall market sentiment score from AI news analysis

        Returns:
            MasterAnalysisResult with complete analysis and recommendation
        """
        reasoning = []
        current_price = df['close'].iloc[-1]

        # ========== MODULE 1: VOLATILITY REGIME ==========
        print("🔍 Analyzing Volatility Regime...")
        volatility_result = self.volatility_detector.analyze_regime(
            df, vix_current, vix_history, option_chain, days_to_expiry
        )
        reasoning.append(f"Volatility: {volatility_result.regime.value} ({volatility_result.vix_level:.1f} VIX)")

        # ========== MODULE 2: OI TRAP DETECTION ==========
        print("🎯 Detecting OI Traps...")
        oi_trap_result = self.oi_trap_detector.detect_traps(
            option_chain, current_price, df
        )
        if oi_trap_result.trap_detected:
            reasoning.append(f"⚠️ TRAP: {oi_trap_result.trap_type.value} ({oi_trap_result.trap_probability:.0f}%)")

        # ========== MODULE 3: CVD ANALYSIS ==========
        print("📊 Analyzing CVD & Delta Imbalance...")
        cvd_result = self.cvd_analyzer.analyze_cvd(df)
        reasoning.append(f"CVD: {cvd_result.bias} (Imbalance: {cvd_result.delta_imbalance:+.1f}%)")

        # ========== MODULE 4: INSTITUTIONAL VS RETAIL ==========
        print("🏦 Detecting Institutional vs Retail...")
        participant_result = self.participant_detector.detect_participant(
            df, option_chain, current_price
        )
        reasoning.append(f"Participant: {participant_result.dominant_participant.value} ({participant_result.institutional_confidence:.0f}% inst)")

        # ========== MODULE 5: LIQUIDITY GRAVITY ==========
        print("🧲 Analyzing Liquidity Gravity...")
        liquidity_result = self.liquidity_analyzer.analyze_liquidity_gravity(
            df, option_chain
        )
        reasoning.append(f"Target: {liquidity_result.primary_target:.2f} (Gravity: {liquidity_result.gravity_strength:.0f}%)")

        # ========== MODULE 6: ML MARKET REGIME ==========
        print("🤖 ML Market Regime Detection...")
        ml_regime_result = self.ml_regime_detector.detect_regime(
            df, cvd_result, volatility_result, oi_trap_result
        )
        reasoning.append(f"Regime: {ml_regime_result.regime} ({ml_regime_result.confidence:.0f}% conf)")

        # ========== MODULE 7: EXPECTANCY MODEL ==========
        expectancy_result = None
        if historical_trades is not None:
            print("📈 Calculating Expectancy...")
            expectancy_result = self.expectancy_calc.calculate_expectancy(
                trades_df=historical_trades
            )
            reasoning.append(f"Expectancy: ${expectancy_result.expected_value:.2f}/trade (WR: {expectancy_result.win_rate:.0f}%)")

        # ========== MODULE 8: POSITION SIZING ==========
        position_size_result = None
        if trade_params and 'entry' in trade_params and 'stop' in trade_params and 'target' in trade_params:
            print("💰 Calculating Position Size...")
            position_size_result = self.position_sizer.calculate_position_size(
                instrument=instrument,
                entry_price=trade_params['entry'],
                stop_loss=trade_params['stop'],
                target_price=trade_params['target'],
                df=df,
                win_rate=expectancy_result.win_rate if expectancy_result else None,
                avg_win=expectancy_result.avg_win if expectancy_result else None,
                avg_loss=expectancy_result.avg_loss if expectancy_result else None,
                volatility_regime=volatility_result.regime.value,
                trap_risk=oi_trap_result.trap_probability
            )
            reasoning.append(f"Position: {position_size_result.recommended_lots} lots (Risk: {position_size_result.risk_percentage:.2f}%)")

        # ========== MODULE 9: RISK MANAGEMENT ==========
        risk_result = None
        if trade_params and 'entry' in trade_params and 'direction' in trade_params:
            print("🛡️ Risk Management Check...")
            risk_result = self.risk_manager.evaluate_trade_risk(
                df=df,
                entry_price=trade_params['entry'],
                direction=trade_params['direction'],
                support_level=liquidity_result.support_zones[0].price if liquidity_result.support_zones else None,
                resistance_level=liquidity_result.resistance_zones[0].price if liquidity_result.resistance_zones else None,
                trap_probability=oi_trap_result.trap_probability,
                volatility_regime=volatility_result.regime.value,
                time_until_expiry=days_to_expiry
            )
            reasoning.append(f"Risk: {risk_result.risk_level} (Score: {risk_result.risk_score:.0f}/100)")

        # ========== MODULE 10: XGBOOST ML ANALYSIS ==========
        print("🤖 Running XGBoost ML Analysis...")
        xgboost_result = self.xgboost_analyzer.analyze_complete_market(
            df=df,
            bias_results=bias_results,
            option_chain=option_chain,
            volatility_result=volatility_result,
            oi_trap_result=oi_trap_result,
            cvd_result=cvd_result,
            participant_result=participant_result,
            liquidity_result=liquidity_result,
            ml_regime_result=ml_regime_result,
            sentiment_score=sentiment_score,
            option_screener_data=option_screener_data
        )
        reasoning.append(f"XGBoost ML: {xgboost_result.prediction} ({xgboost_result.confidence:.0f}% conf)")

        # ========== MODULE 11: MARKET SUMMARY ==========
        print("📋 Generating Market Summary...")
        market_summary = generate_market_summary(
            ml_regime=ml_regime_result,
            cvd_result=cvd_result,
            volatility_result=volatility_result,
            oi_trap_result=oi_trap_result,
            participant_result=participant_result,
            liquidity_result=liquidity_result,
            risk_result=risk_result,
            current_price=current_price
        )

        # ========== MASTER DECISION ENGINE ==========
        print("🎯 Making Final Decision...")
        final_verdict, confidence, trade_quality = self._make_final_decision(
            volatility_result,
            oi_trap_result,
            cvd_result,
            participant_result,
            liquidity_result,
            ml_regime_result,
            market_summary,
            risk_result,
            expectancy_result
        )

        # Calculate metrics
        rr_ratio = position_size_result.rr_adjustment if position_size_result else 1.5
        win_prob = expectancy_result.probability_of_profit if expectancy_result else market_summary.conviction_score

        reasoning.append(f"VERDICT: {final_verdict} (Confidence: {confidence:.0f}%, Quality: {trade_quality:.0f})")

        return MasterAnalysisResult(
            volatility_regime=volatility_result,
            oi_trap=oi_trap_result,
            cvd=cvd_result,
            participant=participant_result,
            liquidity=liquidity_result,
            position_size=position_size_result,
            risk_management=risk_result,
            expectancy=expectancy_result,
            ml_regime=ml_regime_result,
            market_summary=market_summary,
            xgboost_ml=xgboost_result,
            final_verdict=final_verdict,
            confidence=confidence,
            risk_reward_ratio=rr_ratio,
            expected_win_probability=win_prob,
            trade_quality_score=trade_quality,
            reasoning=reasoning
        )

    def _make_final_decision(
        self,
        vol_result: VolatilityRegimeResult,
        oi_result: OITrapResult,
        cvd_result: CVDResult,
        participant_result: MarketParticipantResult,
        liquidity_result: LiquidityGravityResult,
        ml_result: MLMarketRegimeResult,
        summary: MarketSummary,
        risk_result: Optional[RiskManagementResult],
        expectancy_result: Optional[ExpectancyResult]
    ) -> Tuple[str, float, float]:
        """
        Make final trading decision by combining all signals

        Returns:
            (verdict, confidence, trade_quality_score)
        """
        # Initialize scoring system
        bullish_score = 0
        bearish_score = 0
        confidence_factors = []
        rejection_reasons = []

        # ===== REJECTION CRITERIA (Immediate NO TRADE) =====

        # 1. High OI Trap Risk
        if oi_result.trap_probability > 70:
            rejection_reasons.append(f"OI Trap Risk: {oi_result.trap_probability:.0f}%")

        # 2. Extreme Volatility
        if vol_result.regime.value == "Extreme Volatility":
            rejection_reasons.append("Extreme Volatility Regime")

        # 3. Risk Management Rejection
        if risk_result and not risk_result.entry_approved:
            rejection_reasons.append("Risk Management Rejection")

        # 4. Negative Expectancy
        if expectancy_result and expectancy_result.expected_value < 0:
            rejection_reasons.append(f"Negative Expectancy: ${expectancy_result.expected_value:.2f}")

        # If any rejection criteria met, return NO TRADE
        if rejection_reasons:
            return "NO TRADE", 0, 0

        # ===== BULLISH SIGNALS =====

        # ML Regime
        if ml_result.regime == "Trending Up":
            bullish_score += 25 * (ml_result.confidence / 100)
            confidence_factors.append(ml_result.confidence)
        elif ml_result.regime == "Volatile Breakout" and cvd_result.bias == "Bullish":
            bullish_score += 20
            confidence_factors.append(ml_result.confidence)

        # CVD
        if cvd_result.bias == "Bullish":
            bullish_score += 20 * (cvd_result.confidence)
            confidence_factors.append(cvd_result.confidence * 100)

        # Institutional Behavior
        if participant_result.smart_money_detected:
            if "Accumulation" in participant_result.entry_type.value:
                bullish_score += 25
                confidence_factors.append(participant_result.institutional_confidence)
            elif "Distribution" in participant_result.entry_type.value:
                bearish_score += 25
                confidence_factors.append(participant_result.institutional_confidence)

        # Retail Trapped (contrarian)
        if oi_result.trapped_direction == "CALL_BUYERS":
            bearish_score += 15
        elif oi_result.trapped_direction == "PUT_BUYERS":
            bullish_score += 15

        # Market Summary
        if summary.overall_bias == "Bullish":
            bullish_score += 15 * (summary.bias_confidence / 100)
            confidence_factors.append(summary.bias_confidence)
        elif summary.overall_bias == "Bearish":
            bearish_score += 15 * (summary.bias_confidence / 100)
            confidence_factors.append(summary.bias_confidence)

        # ===== BEARISH SIGNALS =====

        # ML Regime
        if ml_result.regime == "Trending Down":
            bearish_score += 25 * (ml_result.confidence / 100)
            confidence_factors.append(ml_result.confidence)

        # CVD
        if cvd_result.bias == "Bearish":
            bearish_score += 20 * (cvd_result.confidence)
            confidence_factors.append(cvd_result.confidence * 100)

        # ===== CALCULATE FINAL VERDICT =====

        net_score = bullish_score - bearish_score
        total_signal_strength = abs(net_score)

        # Calculate confidence (average of all confidence factors)
        avg_confidence = np.mean(confidence_factors) if confidence_factors else 50

        # Calculate trade quality score
        quality_score = self._calculate_trade_quality(
            total_signal_strength,
            avg_confidence,
            vol_result,
            oi_result,
            expectancy_result,
            ml_result
        )

        # Determine verdict
        if quality_score < self.min_trade_quality_score:
            return "NO TRADE", quality_score, quality_score

        if net_score > 50 and avg_confidence > 70:
            return "STRONG BUY", avg_confidence, quality_score
        elif net_score > 25:
            return "BUY", avg_confidence, quality_score
        elif net_score < -50 and avg_confidence > 70:
            return "STRONG SELL", avg_confidence, quality_score
        elif net_score < -25:
            return "SELL", avg_confidence, quality_score
        else:
            return "HOLD", avg_confidence, quality_score

    def _calculate_trade_quality(
        self,
        signal_strength: float,
        confidence: float,
        vol_result: VolatilityRegimeResult,
        oi_result: OITrapResult,
        expectancy_result: Optional[ExpectancyResult],
        ml_result: MLMarketRegimeResult
    ) -> float:
        """Calculate overall trade quality score (0-100)"""
        quality = 0

        # Signal strength (40 points)
        quality += min(signal_strength / 100 * 40, 40)

        # Confidence (30 points)
        quality += (confidence / 100) * 30

        # Volatility regime (10 points)
        if vol_result.regime.value in ["Normal Volatility", "Low Volatility"]:
            quality += 10
        elif vol_result.regime.value == "High Volatility":
            quality += 5

        # No OI trap (10 points)
        if not oi_result.trap_detected:
            quality += 10
        elif oi_result.trap_probability < 40:
            quality += 5

        # Expectancy (10 points bonus)
        if expectancy_result and expectancy_result.profit_factor > 1.5:
            quality += 10

        return min(quality, 100)


def format_master_report(result: MasterAnalysisResult) -> str:
    """Format complete master analysis report"""
    report = f"""
╔══════════════════════════════════════════════════════════╗
║           MASTER AI ORCHESTRATOR REPORT                  ║
║              COMPLETE MARKET ANALYSIS                    ║
╚══════════════════════════════════════════════════════════╝

🎯 FINAL VERDICT: {result.final_verdict}
💪 CONFIDENCE: {result.confidence:.1f}%
⭐ TRADE QUALITY: {result.trade_quality_score:.1f}/100
📊 WIN PROBABILITY: {result.expected_win_probability:.1f}%
📈 RISK/REWARD: {result.risk_reward_ratio:.2f}:1

═══════════════════════════════════════════════════════════
                    MASTER REASONING
═══════════════════════════════════════════════════════════
"""
    for i, reason in enumerate(result.reasoning, 1):
        report += f"{i}. {reason}\n"

    report += f"""
═══════════════════════════════════════════════════════════
                MODULE SUMMARIES
═══════════════════════════════════════════════════════════

📊 VOLATILITY REGIME:
   {result.volatility_regime.regime.value} | VIX: {result.volatility_regime.vix_level:.1f}
   Trend: {result.volatility_regime.trend.value}
   Strategy: {result.volatility_regime.recommended_strategy}

🎯 OI TRAP DETECTION:
   Status: {'TRAP DETECTED' if result.oi_trap.trap_detected else 'Clear'}
   Probability: {result.oi_trap.trap_probability:.1f}%
   Retail Trap Score: {result.oi_trap.retail_trap_score:.1f}/100
   Smart Money: {result.oi_trap.smart_money_signal}

📈 CVD ANALYSIS:
   Bias: {result.cvd.bias}
   Delta Imbalance: {result.cvd.delta_imbalance:+.1f}%
   Orderflow Strength: {result.cvd.orderflow_strength:.1f}/100
   Institutional Sweep: {'YES' if result.cvd.institutional_sweep else 'No'}

🏦 PARTICIPANT DETECTION:
   Dominant: {result.participant.dominant_participant.value}
   Entry Type: {result.participant.entry_type.value}
   Inst Confidence: {result.participant.institutional_confidence:.1f}%
   {result.participant.recommendation}

🧲 LIQUIDITY GRAVITY:
   Primary Target: {result.liquidity.primary_target:.2f}
   Gravity Strength: {result.liquidity.gravity_strength:.1f}/100
   Support: {result.liquidity.support_zones[0].price:.2f if result.liquidity.support_zones else 'N/A'}
   Resistance: {result.liquidity.resistance_zones[0].price:.2f if result.liquidity.resistance_zones else 'N/A'}
"""

    if result.position_size:
        report += f"""
💰 POSITION SIZING:
   Recommended Lots: {result.position_size.recommended_lots}
   Contracts: {result.position_size.recommended_contracts}
   Risk: {result.position_size.risk_percentage:.2f}%
   Position Value: ₹{result.position_size.position_value:,.0f}
"""

    if result.risk_management:
        report += f"""
🛡️  RISK MANAGEMENT:
   Entry Approved: {'YES' if result.risk_management.entry_approved else 'NO'}
   Risk Score: {result.risk_management.risk_score:.1f}/100
   Stop Loss: {result.risk_management.stop_loss:.2f}
   Take Profit: {result.risk_management.take_profit:.2f}
"""

    if result.expectancy:
        report += f"""
📊 EXPECTANCY:
   Expected Value: ${result.expectancy.expected_value:.2f}/trade
   Win Rate: {result.expectancy.win_rate:.1f}%
   Profit Factor: {result.expectancy.profit_factor:.2f}
   Payoff Ratio: {result.expectancy.payoff_ratio:.2f}:1
"""

    report += f"""
🤖 ML MARKET REGIME:
   Regime: {result.ml_regime.regime}
   Confidence: {result.ml_regime.confidence:.1f}%
   Trend Strength: {result.ml_regime.trend_strength:.1f}/100
   Market Phase: {result.ml_regime.market_phase}
   Strategy: {result.ml_regime.recommended_strategy}

═══════════════════════════════════════════════════════════
{result.market_summary.summary_text}
═══════════════════════════════════════════════════════════
"""

    return report
