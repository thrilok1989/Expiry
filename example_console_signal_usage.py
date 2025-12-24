"""
Example: How to use the console signal display with your existing VOB signal generator
"""

from console_signal_display import print_signal, format_simple_signal_console
from vob_signal_generator import VOBSignalGenerator
from datetime import datetime


def display_vob_signal_in_console(signal: dict):
    """
    Display a VOB signal in the console using the new console display

    Args:
        signal: Signal dictionary from VOBSignalGenerator
    """
    if signal is None:
        print("\n⚠️  No signal generated")
        return

    # The signal dict from VOBSignalGenerator already has the right structure
    # Just print it directly
    print_signal(signal)


def example_bearish_signal():
    """Example: Display the exact bearish signal from user's request"""

    signal = {
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
    print("USER'S NIFTY BEARISH SIGNAL - CONVERTED FROM HTML TO PYTHON")
    print("="*70)
    display_vob_signal_in_console(signal)


def example_with_vob_generator():
    """Example: Use with VOBSignalGenerator"""

    # Initialize signal generator
    vob_gen = VOBSignalGenerator(proximity_threshold=7.0)

    # Simulate bearish signal data
    spot_price = 26140.0
    market_sentiment = "BEARISH"

    bearish_blocks = [
        {
            'upper': 26141.52,
            'lower': 26130.0,
            'mid': 26135.76,
            'volume': 150000
        }
    ]

    # Check for signal (would normally pass df for strength calculation)
    signal = vob_gen.check_for_signal(
        spot_price=spot_price,
        market_sentiment=market_sentiment,
        bullish_blocks=[],
        bearish_blocks=bearish_blocks,
        index="NIFTY",
        df=None  # Would pass actual dataframe for strength analysis
    )

    if signal:
        print("\n" + "="*70)
        print("SIGNAL FROM VOB GENERATOR - DISPLAYED IN CONSOLE")
        print("="*70)
        display_vob_signal_in_console(signal)


def example_simple_function():
    """Example: Using the simple function for quick signal display"""

    print("\n" + "="*70)
    print("QUICK SIGNAL DISPLAY USING SIMPLE FUNCTION")
    print("="*70)

    output = format_simple_signal_console(
        index="NIFTY",
        direction="PUT",
        sentiment="BEARISH",
        entry_price=26140.0,
        stop_loss=26157.85,
        target=26113.22,
        risk_reward="1:1.5",
        vob_level=26141.52,
        distance_from_vob=1.52,
        signal_time="14:55:11",
        strength_score=62.7,
        strength_status="MODERATE",
        trend="WEAKENING",
        times_tested=22,
        respect_rate=81.8
    )

    print(output)


if __name__ == "__main__":
    # Run all examples
    example_bearish_signal()
    example_with_vob_generator()
    example_simple_function()

    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("="*70 + "\n")
