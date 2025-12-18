"""
Market Hours Scheduler for NSE Trading
Provides centralized market hours validation and session management using IST.
"""

from datetime import datetime, time
from enum import Enum
from typing import Optional, Tuple
import pytz

# Indian Standard Time timezone
IST = pytz.timezone('Asia/Kolkata')


class MarketSession(Enum):
    """Market session types"""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    POST_MARKET = "post_market"
    CLOSED = "closed"


class MarketHoliday:
    """NSE Market Holidays for 2025"""

    # List of NSE holidays (format: YYYY-MM-DD)
    # Update this list annually from NSE website
    HOLIDAYS_2025 = [
        "2025-01-26",  # Republic Day
        "2025-03-14",  # Holi
        "2025-03-31",  # Id-Ul-Fitr
        "2025-04-10",  # Mahavir Jayanti
        "2025-04-14",  # Dr. Ambedkar Jayanti
        "2025-04-18",  # Good Friday
        "2025-05-01",  # Maharashtra Day
        "2025-06-07",  # Id-Ul-Adha (Bakri Id)
        "2025-07-06",  # Muharram
        "2025-08-15",  # Independence Day
        "2025-08-27",  # Ganesh Chaturthi
        "2025-09-05",  # Milad-Un-Nabi
        "2025-10-02",  # Mahatma Gandhi Jayanti
        "2025-10-21",  # Dussehra
        "2025-10-24",  # Diwali - Laxmi Pujan
        "2025-10-25",  # Diwali - Balipratipada
        "2025-11-05",  # Gurunanak Jayanti
        "2025-12-25",  # Christmas
    ]

    @classmethod
    def is_holiday(cls, date_obj: datetime) -> bool:
        """Check if given date is a market holiday"""
        date_str = date_obj.strftime("%Y-%m-%d")
        return date_str in cls.HOLIDAYS_2025


class MarketHoursScheduler:
    """
    Centralized market hours scheduler for NSE
    All times are in Indian Standard Time (IST)
    """

    # Market hours configuration (all in IST)
    PRE_MARKET_OPEN = time(8, 30)   # 8:30 AM IST
    MARKET_OPEN = time(9, 15)        # 9:15 AM IST
    MARKET_CLOSE = time(15, 30)      # 3:30 PM IST
    POST_MARKET_CLOSE = time(15, 45) # 3:45 PM IST

    def __init__(self):
        """Initialize the market hours scheduler"""
        self.timezone = IST

    def get_current_time_ist(self) -> datetime:
        """Get current time in IST"""
        return datetime.now(self.timezone)

    def is_trading_day(self, date_obj: Optional[datetime] = None) -> bool:
        """
        Check if the given date is a trading day (weekday and not a holiday)

        Args:
            date_obj: datetime object to check (defaults to current IST time)

        Returns:
            True if it's a trading day, False otherwise
        """
        if date_obj is None:
            date_obj = self.get_current_time_ist()

        # Check if it's a weekend (Saturday=5, Sunday=6)
        if date_obj.weekday() >= 5:
            return False

        # Check if it's a market holiday
        if MarketHoliday.is_holiday(date_obj):
            return False

        return True

    def get_market_session(self, dt: Optional[datetime] = None) -> MarketSession:
        """
        Get the current market session

        Args:
            dt: datetime object to check (defaults to current IST time)

        Returns:
            MarketSession enum value
        """
        if dt is None:
            dt = self.get_current_time_ist()

        # Ensure datetime is in IST
        if dt.tzinfo is None:
            dt = self.timezone.localize(dt)
        else:
            dt = dt.astimezone(self.timezone)

        # Check if it's a non-trading day
        if not self.is_trading_day(dt):
            return MarketSession.CLOSED

        current_time = dt.time()

        # Determine session based on time
        if current_time < self.PRE_MARKET_OPEN:
            return MarketSession.CLOSED
        elif current_time < self.MARKET_OPEN:
            return MarketSession.PRE_MARKET
        elif current_time < self.MARKET_CLOSE:
            return MarketSession.REGULAR
        elif current_time < self.POST_MARKET_CLOSE:
            return MarketSession.POST_MARKET
        else:
            return MarketSession.CLOSED

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if market is currently open for regular trading

        Args:
            dt: datetime object to check (defaults to current IST time)

        Returns:
            True if market is open (9:15 AM - 3:30 PM IST on trading days)
        """
        return self.get_market_session(dt) == MarketSession.REGULAR

    def is_within_trading_hours(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if current time is within extended trading hours (pre-market to post-market)

        Args:
            dt: datetime object to check (defaults to current IST time)

        Returns:
            True if within 8:30 AM - 3:45 PM IST on trading days
        """
        session = self.get_market_session(dt)
        return session in [MarketSession.PRE_MARKET, MarketSession.REGULAR, MarketSession.POST_MARKET]

    def get_market_status(self, dt: Optional[datetime] = None) -> dict:
        """
        Get comprehensive market status information

        Args:
            dt: datetime object to check (defaults to current IST time)

        Returns:
            Dictionary with market status details
        """
        if dt is None:
            dt = self.get_current_time_ist()

        session = self.get_market_session(dt)

        return {
            'current_time_ist': dt.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'is_trading_day': self.is_trading_day(dt),
            'is_holiday': MarketHoliday.is_holiday(dt),
            'is_weekend': dt.weekday() >= 5,
            'session': session.value,
            'is_market_open': session == MarketSession.REGULAR,
            'is_within_trading_hours': self.is_within_trading_hours(dt),
            'market_open_time': self.MARKET_OPEN.strftime('%H:%M'),
            'market_close_time': self.MARKET_CLOSE.strftime('%H:%M'),
            'pre_market_open': self.PRE_MARKET_OPEN.strftime('%H:%M'),
            'post_market_close': self.POST_MARKET_CLOSE.strftime('%H:%M'),
        }

    def get_next_market_open(self, dt: Optional[datetime] = None) -> datetime:
        """
        Get the next market opening time

        Args:
            dt: datetime object to check from (defaults to current IST time)

        Returns:
            datetime of next market open
        """
        if dt is None:
            dt = self.get_current_time_ist()

        # Ensure datetime is in IST
        if dt.tzinfo is None:
            dt = self.timezone.localize(dt)
        else:
            dt = dt.astimezone(self.timezone)

        # Start from next day if current time is past market close
        if dt.time() >= self.MARKET_CLOSE:
            from datetime import timedelta
            dt = dt + timedelta(days=1)

        # Find next trading day
        max_days = 10  # Prevent infinite loop
        for _ in range(max_days):
            if self.is_trading_day(dt):
                # Combine date with market open time
                next_open = datetime.combine(dt.date(), self.MARKET_OPEN)
                return self.timezone.localize(next_open)

            from datetime import timedelta
            dt = dt + timedelta(days=1)

        # Fallback: return current time + 1 day
        from datetime import timedelta
        return dt + timedelta(days=1)

    def should_run_app(self, dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Determine if the application should be running

        Args:
            dt: datetime object to check (defaults to current IST time)

        Returns:
            Tuple of (should_run: bool, reason: str)
        """
        if dt is None:
            dt = self.get_current_time_ist()

        if not self.is_trading_day(dt):
            if dt.weekday() >= 5:
                return False, "Market closed: Weekend"
            else:
                return False, "Market closed: Holiday"

        session = self.get_market_session(dt)

        if session == MarketSession.CLOSED:
            return False, f"Market closed: Outside trading hours (8:30 AM - 3:45 PM IST)"
        elif session == MarketSession.PRE_MARKET:
            return True, "Pre-market session (8:30 AM - 9:15 AM IST)"
        elif session == MarketSession.REGULAR:
            return True, "Regular market session (9:15 AM - 3:30 PM IST)"
        elif session == MarketSession.POST_MARKET:
            return True, "Post-market session (3:30 PM - 3:45 PM IST)"

        return False, "Unknown session"

    def get_refresh_interval(self, session: Optional[MarketSession] = None) -> int:
        """
        Get recommended refresh interval in seconds based on market session

        Args:
            session: Market session (defaults to current session)

        Returns:
            Refresh interval in seconds
        """
        if session is None:
            session = self.get_market_session()

        # Recommended refresh intervals
        if session == MarketSession.PRE_MARKET:
            return 30  # 30 seconds during pre-market
        elif session == MarketSession.REGULAR:
            return 10  # 10 seconds during regular trading
        elif session == MarketSession.POST_MARKET:
            return 60  # 60 seconds during post-market
        else:
            return 300  # 5 minutes when market is closed


# Global instance for easy access
scheduler = MarketHoursScheduler()


# Convenience functions for common operations
def is_market_open() -> bool:
    """Check if market is currently open"""
    return scheduler.is_market_open()


def is_within_trading_hours() -> bool:
    """Check if within extended trading hours (8:30 AM - 3:45 PM IST)"""
    return scheduler.is_within_trading_hours()


def get_current_time_ist() -> datetime:
    """Get current time in IST"""
    return scheduler.get_current_time_ist()


def get_market_status() -> dict:
    """Get comprehensive market status"""
    return scheduler.get_market_status()


def should_run_app() -> Tuple[bool, str]:
    """Determine if app should be running"""
    return scheduler.should_run_app()


if __name__ == "__main__":
    # Test the scheduler
    print("=" * 60)
    print("NSE Market Hours Scheduler - Status Report")
    print("=" * 60)

    status = get_market_status()

    print(f"\nCurrent Time (IST): {status['current_time_ist']}")
    print(f"Is Trading Day: {status['is_trading_day']}")
    print(f"Is Holiday: {status['is_holiday']}")
    print(f"Is Weekend: {status['is_weekend']}")
    print(f"\nCurrent Session: {status['session'].upper()}")
    print(f"Market Open (Regular): {status['is_market_open']}")
    print(f"Within Trading Hours: {status['is_within_trading_hours']}")

    print(f"\nMarket Hours (IST):")
    print(f"  Pre-Market: {status['pre_market_open']}")
    print(f"  Regular: {status['market_open_time']} - {status['market_close_time']}")
    print(f"  Post-Market Close: {status['post_market_close']}")

    should_run, reason = should_run_app()
    print(f"\nShould App Run: {should_run}")
    print(f"Reason: {reason}")

    if not should_run:
        next_open = scheduler.get_next_market_open()
        print(f"\nNext Market Open: {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    print("\n" + "=" * 60)
