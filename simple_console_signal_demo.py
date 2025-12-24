"""
Simple Demo: Console Signal Display (No external dependencies)
Shows exactly how to convert your HTML signal to Python console output
"""

from console_signal_display import print_signal


def main():
    """Main demo - shows your exact NIFTY bearish signal"""

    # Your exact signal from the HTML
    nifty_bearish_signal = {
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

    print("\n🎯 NIFTY BEARISH SIGNAL - HTML CONVERTED TO PYTHON CONSOLE\n")
    print_signal(nifty_bearish_signal)

    # Additional example - Bullish signal
    nifty_bullish_signal = {
        'index': 'NIFTY',
        'direction': 'CALL',
        'market_sentiment': 'BULLISH',
        'entry_price': 26200.0,
        'stop_loss': 26175.0,
        'target': 26237.5,
        'risk_reward': '1:1.5',
        'vob_level': 26195.0,
        'distance_from_vob': 5.0,
        'timestamp': '10:30:00',
        'strength': {
            'strength_score': 85.0,
            'strength_label': 'STRONG',
            'trend': 'STRENGTHENING',
            'times_tested': 30,
            'respect_rate': 96.7
        }
    }

    print("\n🎯 NIFTY BULLISH SIGNAL - EXAMPLE\n")
    print_signal(nifty_bullish_signal)


if __name__ == "__main__":
    main()
