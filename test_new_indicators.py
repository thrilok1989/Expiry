"""
Test script for new indicators: Money Flow Profile and DeltaFlow Volume Profile
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the new indicators
from indicators.money_flow_profile import MoneyFlowProfile
from indicators.deltaflow_volume_profile import DeltaFlowVolumeProfile


def generate_sample_data(num_bars=300):
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)

    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=num_bars)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=num_bars)

    # Generate price data with trend
    base_price = 24000
    trend = np.cumsum(np.random.randn(num_bars) * 10)
    prices = base_price + trend

    # Generate OHLCV
    data = []
    for i, price in enumerate(prices):
        volatility = np.random.uniform(5, 20)
        open_price = price + np.random.uniform(-volatility/2, volatility/2)
        high_price = max(open_price, price) + np.random.uniform(0, volatility)
        low_price = min(open_price, price) - np.random.uniform(0, volatility)
        close_price = price + np.random.uniform(-volatility/2, volatility/2)
        volume = np.random.uniform(1000, 10000)

        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    return df


def test_money_flow_profile():
    """Test Money Flow Profile indicator"""
    print("=" * 60)
    print("TESTING MONEY FLOW PROFILE")
    print("=" * 60)

    # Generate sample data
    df = generate_sample_data(300)

    # Initialize indicator with default settings (10 rows as requested)
    mfp = MoneyFlowProfile(
        lookback=200,
        num_rows=10,  # Default 10 rows as requested
        profile_source='Volume',
        show_volume_profile=True,
        show_sentiment_profile=True
    )

    # Calculate profile
    print("\n1. Calculating profile...")
    profile_data = mfp.calculate(df)

    if profile_data['success']:
        print("✅ Profile calculation successful!")
        print(f"   - Period High: {profile_data['period_high']:.2f}")
        print(f"   - Period Low: {profile_data['period_low']:.2f}")
        print(f"   - POC Price: {profile_data['poc_price']:.2f}")
        print(f"   - Total Volume: {profile_data['total_volume']:,.0f}")
        print(f"   - Number of bins: {len(profile_data['bins'])}")
    else:
        print(f"❌ Profile calculation failed: {profile_data.get('error')}")
        return False

    # Get signals
    print("\n2. Getting signals...")
    signals = mfp.get_signals(df)

    if signals['success']:
        print("✅ Signal generation successful!")
        print(f"   - Sentiment: {signals['sentiment']}")
        print(f"   - Bullish Volume: {signals['bullish_volume_pct']:.1f}%")
        print(f"   - Bearish Volume: {signals['bearish_volume_pct']:.1f}%")
        print(f"   - Current Price: {signals['current_price']:.2f}")
        print(f"   - Price Position: {signals['price_position']}")
        print(f"   - High Volume Levels: {len(signals['high_volume_levels'])}")
        print(f"   - Low Volume Levels: {len(signals['low_volume_levels'])}")
    else:
        print(f"❌ Signal generation failed: {signals.get('error')}")
        return False

    # Format report
    print("\n3. Generating report...")
    report = mfp.format_report(signals)
    print(report)

    return True


def test_deltaflow_volume_profile():
    """Test DeltaFlow Volume Profile indicator"""
    print("\n" + "=" * 60)
    print("TESTING DELTAFLOW VOLUME PROFILE")
    print("=" * 60)

    # Generate sample data
    df = generate_sample_data(300)

    # Initialize indicator
    dfp = DeltaFlowVolumeProfile(
        lookback=200,
        bins=30,
        show_delta_heatmap=True,
        show_delta_display=True,
        show_volume_bars=True,
        show_poc=True
    )

    # Calculate profile
    print("\n1. Calculating profile...")
    profile_data = dfp.calculate(df)

    if profile_data['success']:
        print("✅ Profile calculation successful!")
        print(f"   - Period High: {profile_data['period_high']:.2f}")
        print(f"   - Period Low: {profile_data['period_low']:.2f}")
        print(f"   - POC Price: {profile_data['poc_price']:.2f}")
        print(f"   - Total Volume: {profile_data['total_volume']:,.0f}")
        print(f"   - Overall Delta: {profile_data['overall_delta']:+.1f}%")
        print(f"   - Number of bins: {len(profile_data['bins'])}")
    else:
        print(f"❌ Profile calculation failed: {profile_data.get('error')}")
        return False

    # Get signals
    print("\n2. Getting signals...")
    signals = dfp.get_signals(df)

    if signals['success']:
        print("✅ Signal generation successful!")
        print(f"   - Sentiment: {signals['sentiment']}")
        print(f"   - Overall Delta: {signals['overall_delta']:+.1f}%")
        print(f"   - Buy Volume: {signals['overall_bull_pct']:.1f}%")
        print(f"   - Sell Volume: {signals['overall_bear_pct']:.1f}%")
        print(f"   - Current Price: {signals['current_price']:.2f}")
        print(f"   - Price Position: {signals['price_position']}")
        print(f"   - Strong Buy Levels: {len(signals['strong_buy_levels'])}")
        print(f"   - Strong Sell Levels: {len(signals['strong_sell_levels'])}")
        print(f"   - Absorption Zones: {len(signals['absorption_zones'])}")
    else:
        print(f"❌ Signal generation failed: {signals.get('error')}")
        return False

    # Format report
    print("\n3. Generating report...")
    report = dfp.format_report(signals)
    print(report)

    # Get delta levels summary
    print("\n4. Getting delta distribution...")
    summary = dfp.get_delta_levels_summary(df)

    if summary['success']:
        print("✅ Delta distribution successful!")
        print(f"   - Strong Buy Bins: {summary['strong_buy']}")
        print(f"   - Moderate Buy Bins: {summary['moderate_buy']}")
        print(f"   - Neutral Bins: {summary['neutral']}")
        print(f"   - Moderate Sell Bins: {summary['moderate_sell']}")
        print(f"   - Strong Sell Bins: {summary['strong_sell']}")
    else:
        print(f"❌ Delta distribution failed: {summary.get('error')}")
        return False

    return True


def main():
    """Run all tests"""
    print("\n")
    print("╔════════════════════════════════════════════════════════╗")
    print("║     TESTING NEW INDICATORS                             ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()

    # Test Money Flow Profile
    mfp_success = test_money_flow_profile()

    # Test DeltaFlow Volume Profile
    dfp_success = test_deltaflow_volume_profile()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Money Flow Profile: {'✅ PASSED' if mfp_success else '❌ FAILED'}")
    print(f"DeltaFlow Volume Profile: {'✅ PASSED' if dfp_success else '❌ FAILED'}")

    if mfp_success and dfp_success:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
