#!/usr/bin/env python3
"""
Integration Test for Market Hours Scheduler
Tests all updated modules to ensure IST implementation is working correctly
"""

import sys
from datetime import datetime

def test_market_hours_scheduler():
    """Test the centralized market hours scheduler"""
    print("\n" + "="*70)
    print("TEST 1: Market Hours Scheduler Module")
    print("="*70)

    try:
        from market_hours_scheduler import (
            scheduler,
            is_market_open,
            is_within_trading_hours,
            get_market_status,
            should_run_app
        )

        status = get_market_status()
        print(f"✓ Market Hours Scheduler imported successfully")
        print(f"  Current Time (IST): {status['current_time_ist']}")
        print(f"  Is Trading Day: {status['is_trading_day']}")
        print(f"  Current Session: {status['session']}")
        print(f"  Market Open: {status['is_market_open']}")
        print(f"  Within Trading Hours: {status['is_within_trading_hours']}")

        should_run, reason = should_run_app()
        print(f"  Should App Run: {should_run}")
        print(f"  Reason: {reason}")

        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_config():
    """Test config.py has market hours settings"""
    print("\n" + "="*70)
    print("TEST 2: Config Module")
    print("="*70)

    try:
        import config

        assert hasattr(config, 'MARKET_HOURS_ENABLED'), "MARKET_HOURS_ENABLED not found"
        assert hasattr(config, 'MARKET_HOURS'), "MARKET_HOURS not found"
        assert hasattr(config, 'REFRESH_INTERVALS'), "REFRESH_INTERVALS not found"

        print(f"✓ Config imported successfully")
        print(f"  Market Hours Enabled: {config.MARKET_HOURS_ENABLED}")
        print(f"  Market Hours: {config.MARKET_HOURS}")
        print(f"  Refresh Intervals: {config.REFRESH_INTERVALS}")

        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_market_data():
    """Test market_data.py uses scheduler"""
    print("\n" + "="*70)
    print("TEST 3: Market Data Module")
    print("="*70)

    try:
        from market_data import is_market_open, get_market_status

        market_open = is_market_open()
        status = get_market_status()

        print(f"✓ Market Data imported successfully")
        print(f"  is_market_open(): {market_open}")
        print(f"  Market Status: {status}")

        # Verify it's using IST
        if 'time' in status:
            assert 'IST' in status['time'], "Time should include IST timezone"

        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_data_cache_manager():
    """Test data_cache_manager.py has market hours awareness"""
    print("\n" + "="*70)
    print("TEST 4: Data Cache Manager Module")
    print("="*70)

    try:
        from data_cache_manager import DataCacheManager

        cache_manager = DataCacheManager()

        assert hasattr(cache_manager, 'market_hours_enabled'), \
            "market_hours_enabled attribute not found"

        print(f"✓ Data Cache Manager imported successfully")
        print(f"  Market Hours Enabled: {cache_manager.market_hours_enabled}")
        print(f"  Cache TTL Config: {cache_manager.ttl_config}")
        print(f"  Refresh Intervals: {cache_manager.refresh_intervals}")

        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False




def test_timezone_consistency():
    """Verify all modules use IST consistently"""
    print("\n" + "="*70)
    print("TEST 6: Timezone Consistency Check")
    print("="*70)

    try:
        from market_hours_scheduler import get_current_time_ist
        from market_data import IST
        import pytz

        current_time = get_current_time_ist()
        ist_timezone = pytz.timezone('Asia/Kolkata')

        print(f"✓ Timezone consistency check passed")
        print(f"  Scheduler IST: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Market Data IST: {IST.zone}")
        print(f"  PyTZ IST: {ist_timezone.zone}")

        assert IST.zone == 'Asia/Kolkata', "IST timezone mismatch"
        assert current_time.tzinfo is not None, "Timezone info is missing"

        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def main():
    """Run all integration tests"""
    print("\n" + "#"*70)
    print("# MARKET HOURS INTEGRATION TEST SUITE")
    print("# Testing IST implementation across all modules")
    print("#"*70)

    tests = [
        test_market_hours_scheduler,
        test_config,
        test_market_data,
        test_data_cache_manager,
        test_timezone_consistency,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test.__name__} crashed: {e}")
            results.append(False)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")

    if all(results):
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nMarket hours scheduler is fully integrated!")
        print("All modules are using IST (Indian Standard Time) consistently.")
        return 0
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nPlease review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
