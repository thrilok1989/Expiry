"""
Expectancy & Probability Model
Quant-style edge calculation for systematic trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExpectancyResult:
    """Expectancy analysis result"""
    expected_value: float  # $ per trade
    expectancy_ratio: float  # R-multiple
    win_rate: float  # 0-100
    avg_win: float
    avg_loss: float
    payoff_ratio: float  # Avg win / Avg loss
    profit_factor: float  # Gross profit / Gross loss
    expected_edge: float  # % edge per trade
    probability_of_profit: float  # 0-100
    kelly_criterion: float  # Optimal position size fraction
    sharpe_ratio: Optional[float]
    max_drawdown: float  # %
    recovery_factor: float
    recommendation: str
    confidence: float  # 0-1
    signals: List[str]


class ExpectancyCalculator:
    """
    Expectancy & Probability Calculator

    Calculates statistical edge using:
    1. Win rate & Payoff ratio
    2. Expectancy (expected $ per trade)
    3. Probability of profit
    4. Kelly Criterion
    5. Risk metrics (Sharpe, drawdown)
    """

    def __init__(self, min_sample_size: int = 20):
        """
        Initialize Expectancy Calculator

        Args:
            min_sample_size: Minimum trades required for valid statistics
        """
        self.min_sample_size = min_sample_size

    def calculate_expectancy(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        total_wins: Optional[float] = None,
        total_losses: Optional[float] = None,
        returns_series: Optional[pd.Series] = None
    ) -> ExpectancyResult:
        """
        Calculate complete expectancy analysis

        Can work with either:
        1. Historical trades dataframe
        2. Manual statistics (win_rate, avg_win, avg_loss)

        Args:
            trades_df: DataFrame with columns ['pnl', 'win'] where win is True/False
            win_rate: Manual win rate (0-100)
            avg_win: Manual average win ($)
            avg_loss: Manual average loss ($)
            total_wins: Total gross profit
            total_losses: Total gross loss
            returns_series: Time series of returns for Sharpe ratio

        Returns:
            ExpectancyResult with complete analysis
        """
        signals = []

        # Extract statistics from dataframe if provided
        if trades_df is not None and 'pnl' in trades_df.columns:
            win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
            wins = trades_df[trades_df['pnl'] > 0]['pnl']
            losses = trades_df[trades_df['pnl'] < 0]['pnl']

            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 1  # Positive value

            total_wins = wins.sum() if len(wins) > 0 else 0
            total_losses = abs(losses.sum()) if len(losses) > 0 else 1

            sample_size = len(trades_df)
        else:
            sample_size = 0

        # Validation
        if win_rate is None or avg_win is None or avg_loss is None:
            return self._default_result("Insufficient data for expectancy calculation")

        if avg_loss == 0:
            avg_loss = 1  # Prevent division by zero

        # Calculate core metrics
        lose_rate = 100 - win_rate
        payoff_ratio = avg_win / avg_loss
        expected_value = (win_rate / 100 * avg_win) - (lose_rate / 100 * avg_loss)
        expectancy_ratio = (win_rate / 100 * payoff_ratio) - (lose_rate / 100)

        # Expected edge (% return per trade)
        avg_trade_size = (avg_win + avg_loss) / 2
        expected_edge = (expected_value / avg_trade_size * 100) if avg_trade_size > 0 else 0

        # Profit factor
        if total_wins is not None and total_losses is not None:
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            profit_factor = (win_rate / 100 * avg_win) / (lose_rate / 100 * avg_loss) if lose_rate > 0 else 0

        # Probability of profit (same as win rate for now)
        probability_of_profit = win_rate

        # Kelly Criterion
        kelly = self._calculate_kelly_criterion(win_rate, payoff_ratio)

        # Sharpe Ratio (if returns series provided)
        sharpe = None
        if returns_series is not None and len(returns_series) > 10:
            sharpe = self._calculate_sharpe_ratio(returns_series)

        # Max Drawdown
        max_drawdown = 0
        recovery_factor = 0
        if trades_df is not None and 'pnl' in trades_df.columns:
            max_drawdown = self._calculate_max_drawdown(trades_df['pnl'].cumsum())
            total_return = trades_df['pnl'].sum()
            recovery_factor = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

        # Generate signals
        if expected_value > 0:
            signals.append(f"✅ Positive Expectancy: ${expected_value:.2f} per trade")
        else:
            signals.append(f"🚫 Negative Expectancy: ${expected_value:.2f} per trade")

        if win_rate >= 50:
            signals.append(f"✅ Win Rate: {win_rate:.1f}%")
        else:
            signals.append(f"⚠️ Win Rate: {win_rate:.1f}% (Need better payoff ratio)")

        if payoff_ratio >= 2.0:
            signals.append(f"✅ Excellent Payoff: {payoff_ratio:.2f}:1")
        elif payoff_ratio >= 1.5:
            signals.append(f"✅ Good Payoff: {payoff_ratio:.2f}:1")
        else:
            signals.append(f"⚠️ Low Payoff: {payoff_ratio:.2f}:1")

        if profit_factor >= 2.0:
            signals.append(f"✅ Strong Profit Factor: {profit_factor:.2f}")
        elif profit_factor >= 1.5:
            signals.append(f"✅ Decent Profit Factor: {profit_factor:.2f}")
        elif profit_factor >= 1.0:
            signals.append(f"⚠️ Break-even Profit Factor: {profit_factor:.2f}")
        else:
            signals.append(f"🚫 Losing Profit Factor: {profit_factor:.2f}")

        # Calculate confidence
        confidence = self._calculate_confidence(
            sample_size, expected_value, profit_factor, win_rate
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            expected_value, expectancy_ratio, profit_factor, win_rate,
            payoff_ratio, sample_size
        )

        return ExpectancyResult(
            expected_value=expected_value,
            expectancy_ratio=expectancy_ratio,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            payoff_ratio=payoff_ratio,
            profit_factor=profit_factor,
            expected_edge=expected_edge,
            probability_of_profit=probability_of_profit,
            kelly_criterion=kelly,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            recovery_factor=recovery_factor,
            recommendation=recommendation,
            confidence=confidence,
            signals=signals
        )

    def _calculate_kelly_criterion(self, win_rate: float, payoff_ratio: float) -> float:
        """
        Calculate Kelly Criterion optimal position size

        Kelly % = W - [(1 - W) / R]
        where W = win rate (decimal), R = payoff ratio
        """
        w = win_rate / 100
        r = payoff_ratio

        kelly = w - ((1 - w) / r) if r > 0 else 0

        # Conservative Kelly (half or quarter Kelly)
        return max(0, kelly / 2)  # Half Kelly for safety

    def _calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.06
    ) -> float:
        """
        Calculate Sharpe Ratio

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        Annualized for trading days
        """
        if len(returns) < 10:
            return 0.0

        # Annualize assuming 252 trading days
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)

        if std_return == 0:
            return 0.0

        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        if len(cumulative_returns) == 0:
            return 0.0

        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        max_dd = drawdown.min()

        # Convert to percentage
        if running_max.max() > 0:
            max_dd_pct = (max_dd / running_max.max()) * 100
        else:
            max_dd_pct = 0

        return abs(max_dd_pct)

    def _calculate_confidence(
        self,
        sample_size: int,
        expected_value: float,
        profit_factor: float,
        win_rate: float
    ) -> float:
        """Calculate confidence in expectancy (0-1)"""
        confidence = 0.5  # Base

        # Sample size confidence
        if sample_size >= self.min_sample_size * 2:
            confidence += 0.3
        elif sample_size >= self.min_sample_size:
            confidence += 0.15

        # Expectancy confidence
        if expected_value > 0 and profit_factor > 1.5:
            confidence += 0.15

        # Win rate confidence
        if 40 <= win_rate <= 60:
            confidence += 0.05  # Balanced win rate

        return np.clip(confidence, 0.2, 1.0)

    def _generate_recommendation(
        self,
        expected_value: float,
        expectancy_ratio: float,
        profit_factor: float,
        win_rate: float,
        payoff_ratio: float,
        sample_size: int
    ) -> str:
        """Generate trading recommendation based on expectancy"""
        if sample_size < self.min_sample_size:
            return f"⚠️ INSUFFICIENT DATA - Need {self.min_sample_size - sample_size} more trades for valid statistics"

        if expected_value <= 0:
            return "🚫 NEGATIVE EDGE - Do not trade this system"

        if profit_factor < 1.0:
            return "🚫 LOSING SYSTEM - Profit Factor < 1.0"

        # Positive expectancy
        if profit_factor >= 2.0 and expectancy_ratio >= 0.5:
            return f"✅ EXCELLENT EDGE - Expected: ${expected_value:.2f}/trade, PF: {profit_factor:.2f}"

        if profit_factor >= 1.5 and expectancy_ratio >= 0.3:
            return f"✅ GOOD EDGE - Expected: ${expected_value:.2f}/trade, PF: {profit_factor:.2f}"

        if profit_factor >= 1.2:
            return f"⚠️ SMALL EDGE - Expected: ${expected_value:.2f}/trade, trade selectively"

        return f"⚠️ MARGINAL EDGE - Expected: ${expected_value:.2f}/trade, improve setup quality"

    def _default_result(self, message: str) -> ExpectancyResult:
        """Return default result"""
        return ExpectancyResult(
            expected_value=0.0,
            expectancy_ratio=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            payoff_ratio=0.0,
            profit_factor=0.0,
            expected_edge=0.0,
            probability_of_profit=0.0,
            kelly_criterion=0.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
            recovery_factor=0.0,
            recommendation=message,
            confidence=0.0,
            signals=[message]
        )


def format_expectancy_report(result: ExpectancyResult) -> str:
    """Format expectancy analysis as readable report"""
    report = f"""
╔══════════════════════════════════════════════════════════╗
║          EXPECTANCY & EDGE ANALYSIS                      ║
╚══════════════════════════════════════════════════════════╝

💰 EXPECTED VALUE: ${result.expected_value:.2f} per trade
📊 EXPECTANCY RATIO: {result.expectancy_ratio:.2f}R

─────────────────────────────────────────────────────────
PERFORMANCE METRICS:
  • Win Rate: {result.win_rate:.1f}%
  • Avg Win: ${result.avg_win:.2f}
  • Avg Loss: ${result.avg_loss:.2f}
  • Payoff Ratio: {result.payoff_ratio:.2f}:1
  • Profit Factor: {result.profit_factor:.2f}

EDGE METRICS:
  • Expected Edge: {result.expected_edge:+.2f}% per trade
  • Probability of Profit: {result.probability_of_profit:.1f}%
  • Kelly Criterion: {result.kelly_criterion*100:.1f}% of capital

RISK METRICS:
  • Max Drawdown: {result.max_drawdown:.2f}%
  • Recovery Factor: {result.recovery_factor:.2f}
"""
    if result.sharpe_ratio is not None:
        report += f"  • Sharpe Ratio: {result.sharpe_ratio:.2f}\n"

    report += f"\n✅ CONFIDENCE: {result.confidence*100:.1f}%\n"
    report += f"\n─────────────────────────────────────────────────────────\n"
    report += f"💡 RECOMMENDATION:\n{result.recommendation}\n\n"

    report += f"📌 KEY INSIGHTS:\n"
    for signal in result.signals:
        report += f"  • {signal}\n"

    return report


def calculate_trade_quality_score(result: ExpectancyResult) -> float:
    """
    Calculate overall trade quality score (0-100)

    Used to filter trades - only take trades with high quality score
    """
    score = 0

    # Expectancy (40 points)
    if result.expected_value > 0:
        expectancy_score = min((result.expectancy_ratio / 0.5) * 20, 20)  # 0.5R = full score
        score += expectancy_score

        if result.expected_value > 100:
            score += 20
        elif result.expected_value > 50:
            score += 15
        elif result.expected_value > 20:
            score += 10

    # Profit Factor (25 points)
    if result.profit_factor >= 3:
        score += 25
    elif result.profit_factor >= 2:
        score += 20
    elif result.profit_factor >= 1.5:
        score += 15
    elif result.profit_factor >= 1.2:
        score += 10

    # Win Rate (20 points)
    if result.win_rate >= 60:
        score += 20
    elif result.win_rate >= 50:
        score += 15
    elif result.win_rate >= 40:
        score += 10

    # Payoff Ratio (15 points)
    if result.payoff_ratio >= 3:
        score += 15
    elif result.payoff_ratio >= 2:
        score += 12
    elif result.payoff_ratio >= 1.5:
        score += 8

    return min(score, 100)
