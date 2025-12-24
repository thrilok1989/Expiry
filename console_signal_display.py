"""
Console Signal Display for NIFTY/BANKNIFTY Trading Signals
Formats trading signals for terminal/console output without HTML
"""

from typing import Dict, Optional
from datetime import datetime


class ConsoleColors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def format_nifty_signal_console(signal: Dict) -> str:
    """
    Format NIFTY/BANKNIFTY trading signal for console display

    Args:
        signal: Dictionary containing signal data with keys:
            - index: str (e.g., "NIFTY", "BANKNIFTY")
            - direction: str ("CALL" or "PUT")
            - market_sentiment: str ("BULLISH" or "BEARISH")
            - entry_price: float
            - stop_loss: float
            - target: float
            - risk_reward: str (e.g., "1:1.5")
            - vob_level: float
            - distance_from_vob: float
            - timestamp: datetime or str
            - strength: dict (optional) containing:
                - strength_score: float
                - strength_label: str
                - trend: str
                - times_tested: int
                - respect_rate: float

    Returns:
        Formatted string for console display
    """

    # Extract signal data
    index = signal.get('index', 'NIFTY')
    direction = signal.get('direction', 'CALL')
    sentiment = signal.get('market_sentiment', 'NEUTRAL')
    entry_price = signal.get('entry_price', 0.0)
    stop_loss = signal.get('stop_loss', 0.0)
    target = signal.get('target', 0.0)
    risk_reward = signal.get('risk_reward', '1:1.5')
    vob_level = signal.get('vob_level', 0.0)
    distance = signal.get('distance_from_vob', 0.0)
    signal_time = signal.get('timestamp', datetime.now())

    # Format timestamp
    if isinstance(signal_time, datetime):
        time_str = signal_time.strftime('%H:%M:%S')
    else:
        time_str = str(signal_time)

    # Determine colors and labels
    if direction == 'CALL':
        signal_emoji = '🟢'
        direction_label = 'BULLISH'
        sentiment_color = ConsoleColors.OKGREEN
    else:
        signal_emoji = '🔴'
        direction_label = 'BEARISH'
        sentiment_color = ConsoleColors.FAIL

    # Calculate risk:reward ratio
    risk = abs(entry_price - stop_loss)
    reward = abs(target - entry_price)

    # Build the output
    output = []

    # Header
    output.append(f"\n{'='*70}")
    output.append(f"{sentiment_color}{ConsoleColors.BOLD}{signal_emoji} {index} {direction_label} ENTRY SIGNAL{ConsoleColors.ENDC}")
    output.append(f"{ConsoleColors.BOLD}Market Sentiment: {sentiment}{ConsoleColors.ENDC}")
    output.append(f"{'='*70}\n")

    # Entry Levels
    output.append(f"{ConsoleColors.BOLD}Entry Price{ConsoleColors.ENDC}")
    output.append(f"  ₹{entry_price:,.2f}\n")

    output.append(f"{ConsoleColors.FAIL}{ConsoleColors.BOLD}Stop Loss{ConsoleColors.ENDC}")
    output.append(f"  ₹{stop_loss:,.2f}\n")

    output.append(f"{ConsoleColors.OKGREEN}{ConsoleColors.BOLD}Target{ConsoleColors.ENDC}")
    output.append(f"  ₹{target:,.2f}\n")

    output.append(f"{ConsoleColors.BOLD}Risk:Reward{ConsoleColors.ENDC}")
    output.append(f"  {risk_reward}\n")

    # VOB Details
    output.append(f"{ConsoleColors.OKCYAN}{ConsoleColors.BOLD}VOB Level{ConsoleColors.ENDC}")
    output.append(f"  ₹{vob_level:,.2f}\n")

    output.append(f"{ConsoleColors.BOLD}Distance from VOB{ConsoleColors.ENDC}")
    output.append(f"  {distance:.2f} points\n")

    output.append(f"{ConsoleColors.BOLD}Signal Time{ConsoleColors.ENDC}")
    output.append(f"  {time_str}\n")

    # Order Block Strength Analysis (if available)
    strength = signal.get('strength')
    if strength:
        output.append(f"{'-'*70}")
        output.append(f"{ConsoleColors.BOLD}📊 Order Block Strength Analysis{ConsoleColors.ENDC}\n")

        strength_score = strength.get('strength_score', 0.0)
        strength_label = strength.get('strength_label', 'UNKNOWN')
        trend = strength.get('trend', 'STABLE')
        times_tested = strength.get('times_tested', 0)
        respect_rate = strength.get('respect_rate', 0.0)

        # Determine strength color
        if strength_score >= 70:
            strength_color = ConsoleColors.OKGREEN
        elif strength_score >= 50:
            strength_color = ConsoleColors.WARNING
        else:
            strength_color = ConsoleColors.FAIL

        # Trend emoji
        if trend == "STRENGTHENING":
            trend_emoji = "🔺"
        elif trend == "WEAKENING":
            trend_emoji = "🔻"
        else:
            trend_emoji = "➖"

        output.append(f"  {ConsoleColors.BOLD}Strength Score:{ConsoleColors.ENDC}      {strength_color}{strength_score:.1f}/100{ConsoleColors.ENDC}")
        output.append(f"  {ConsoleColors.BOLD}Status:{ConsoleColors.ENDC}              {strength_label.upper()}")
        output.append(f"  {ConsoleColors.BOLD}Trend:{ConsoleColors.ENDC}               {trend_emoji} {trend}")
        output.append(f"  {ConsoleColors.BOLD}Tests / Respect Rate:{ConsoleColors.ENDC} {times_tested} / {respect_rate:.1f}%")

    output.append(f"\n{'='*70}\n")

    return '\n'.join(output)


def print_signal(signal: Dict):
    """
    Print formatted signal to console

    Args:
        signal: Signal dictionary
    """
    print(format_nifty_signal_console(signal))


def format_simple_signal_console(
    index: str,
    direction: str,
    sentiment: str,
    entry_price: float,
    stop_loss: float,
    target: float,
    risk_reward: str,
    vob_level: float,
    distance_from_vob: float,
    signal_time: str,
    strength_score: Optional[float] = None,
    strength_status: Optional[str] = None,
    trend: Optional[str] = None,
    times_tested: Optional[int] = None,
    respect_rate: Optional[float] = None
) -> str:
    """
    Format signal with individual parameters (convenience function)

    Args:
        index: Index name (NIFTY, BANKNIFTY, etc.)
        direction: CALL or PUT
        sentiment: BULLISH or BEARISH
        entry_price: Entry price in rupees
        stop_loss: Stop loss price
        target: Target price
        risk_reward: Risk:Reward ratio (e.g., "1:1.5")
        vob_level: Volume Order Block level
        distance_from_vob: Distance from VOB in points
        signal_time: Signal time (HH:MM:SS format)
        strength_score: Optional strength score (0-100)
        strength_status: Optional status (WEAK, MODERATE, STRONG)
        trend: Optional trend (STRENGTHENING, WEAKENING, STABLE)
        times_tested: Optional number of times tested
        respect_rate: Optional respect rate percentage

    Returns:
        Formatted console string
    """
    signal = {
        'index': index,
        'direction': direction,
        'market_sentiment': sentiment,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'target': target,
        'risk_reward': risk_reward,
        'vob_level': vob_level,
        'distance_from_vob': distance_from_vob,
        'timestamp': signal_time
    }

    if strength_score is not None:
        signal['strength'] = {
            'strength_score': strength_score,
            'strength_label': strength_status or 'MODERATE',
            'trend': trend or 'STABLE',
            'times_tested': times_tested or 0,
            'respect_rate': respect_rate or 0.0
        }

    return format_nifty_signal_console(signal)


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Example 1: NIFTY Bearish Signal (from user's example)
    bearish_signal = {
        'index': 'NIFTY',
        'direction': 'PUT',
        'market_sentiment': 'BEARISH',
        'entry_price': 26140.0,
        'stop_loss': 26157.85,
        'target': 26113.22,
        'risk_reward': '1:1.5',
        'vob_level': 26141.52,
        'distance_from_vob': 1.52,
        'timestamp': '14:55:11',
        'strength': {
            'strength_score': 62.7,
            'strength_label': 'MODERATE',
            'trend': 'WEAKENING',
            'times_tested': 22,
            'respect_rate': 81.8
        }
    }

    print("\n" + "="*70)
    print("EXAMPLE 1: NIFTY Bearish Signal with Strength Analysis")
    print("="*70)
    print_signal(bearish_signal)

    # Example 2: NIFTY Bullish Signal
    bullish_signal = {
        'index': 'NIFTY',
        'direction': 'CALL',
        'market_sentiment': 'BULLISH',
        'entry_price': 26200.0,
        'stop_loss': 26180.0,
        'target': 26230.0,
        'risk_reward': '1:1.5',
        'vob_level': 26195.0,
        'distance_from_vob': 5.0,
        'timestamp': '10:30:45',
        'strength': {
            'strength_score': 78.5,
            'strength_label': 'STRONG',
            'trend': 'STRENGTHENING',
            'times_tested': 15,
            'respect_rate': 93.3
        }
    }

    print("\n" + "="*70)
    print("EXAMPLE 2: NIFTY Bullish Signal with Strength Analysis")
    print("="*70)
    print_signal(bullish_signal)

    # Example 3: Using simple function
    print("\n" + "="*70)
    print("EXAMPLE 3: Using simple function (without strength data)")
    print("="*70)
    simple_output = format_simple_signal_console(
        index="BANKNIFTY",
        direction="PUT",
        sentiment="BEARISH",
        entry_price=48500.0,
        stop_loss=48550.0,
        target=48425.0,
        risk_reward="1:1.5",
        vob_level=48505.0,
        distance_from_vob=5.0,
        signal_time="11:45:30"
    )
    print(simple_output)

    # Example 4: Using simple function with strength data
    print("\n" + "="*70)
    print("EXAMPLE 4: Using simple function (with strength data)")
    print("="*70)
    simple_output_with_strength = format_simple_signal_console(
        index="NIFTY",
        direction="CALL",
        sentiment="BULLISH",
        entry_price=26300.0,
        stop_loss=26275.0,
        target=26337.5,
        risk_reward="1:1.5",
        vob_level=26298.0,
        distance_from_vob=2.0,
        signal_time="13:20:15",
        strength_score=85.2,
        strength_status="STRONG",
        trend="STRENGTHENING",
        times_tested=18,
        respect_rate=94.4
    )
    print(simple_output_with_strength)
