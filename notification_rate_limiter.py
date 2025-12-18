"""
Notification Rate Limiter
Prevents spam by enforcing cooldown periods between notifications
"""

import json
import os
from datetime import datetime, timedelta
import pytz
from config import IST, get_current_time_ist
from typing import Dict, Optional
import threading

class NotificationRateLimiter:
    """
    Manages notification rate limiting with configurable cooldown periods
    Stores last notification times in memory and persists to JSON file
    """

    def __init__(self, cooldown_minutes: int = 10, storage_file: str = 'notification_timestamps.json'):
        """
        Initialize the rate limiter

        Args:
            cooldown_minutes: Minimum minutes between notifications for same alert type
            storage_file: JSON file to persist notification timestamps
        """
        self.cooldown_minutes = cooldown_minutes
        self.storage_file = storage_file
        self.lock = threading.Lock()

        # In-memory cache of last notification times
        # Format: {alert_type: timestamp_string}
        self.last_notifications: Dict[str, str] = {}

        # Load persisted timestamps
        self._load_from_file()

    def _load_from_file(self):
        """Load notification timestamps from JSON file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    self.last_notifications = json.load(f)
            except Exception as e:
                print(f"Error loading notification timestamps: {e}")
                self.last_notifications = {}
        else:
            self.last_notifications = {}

    def _save_to_file(self):
        """Persist notification timestamps to JSON file"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.last_notifications, f, indent=2)
        except Exception as e:
            print(f"Error saving notification timestamps: {e}")

    def can_send_notification(self, alert_type: str, symbol: str = '', level: float = None) -> bool:
        """
        Check if enough time has passed to send a notification

        Args:
            alert_type: Type of alert (e.g., 'vob_proximity', 'htf_support', 'htf_resistance')
            symbol: Trading symbol (e.g., 'NIFTY', 'SENSEX')
            level: Price level (optional, for more granular tracking)

        Returns:
            True if notification can be sent, False otherwise
        """
        with self.lock:
            # Create unique key for this alert type
            if level is not None:
                key = f"{alert_type}_{symbol}_{level:.2f}"
            else:
                key = f"{alert_type}_{symbol}"

            # Check if we have a previous notification time
            if key not in self.last_notifications:
                return True

            # Parse the last notification time
            try:
                last_time = datetime.fromisoformat(self.last_notifications[key])
                current_time = get_current_time_ist()

                # Check if cooldown period has passed
                time_since_last = current_time - last_time
                cooldown_delta = timedelta(minutes=self.cooldown_minutes)

                return time_since_last >= cooldown_delta

            except Exception as e:
                print(f"Error checking notification cooldown: {e}")
                # If error, allow notification
                return True

    def record_notification(self, alert_type: str, symbol: str = '', level: float = None):
        """
        Record that a notification was sent

        Args:
            alert_type: Type of alert
            symbol: Trading symbol
            level: Price level (optional)
        """
        with self.lock:
            # Create unique key
            if level is not None:
                key = f"{alert_type}_{symbol}_{level:.2f}"
            else:
                key = f"{alert_type}_{symbol}"

            # Store current timestamp
            self.last_notifications[key] = get_current_time_ist().isoformat()

            # Persist to file
            self._save_to_file()

    def get_time_until_next_notification(self, alert_type: str, symbol: str = '', level: float = None) -> Optional[int]:
        """
        Get seconds remaining until next notification can be sent

        Args:
            alert_type: Type of alert
            symbol: Trading symbol
            level: Price level (optional)

        Returns:
            Seconds until next notification allowed, or None if can send now
        """
        with self.lock:
            # Create unique key
            if level is not None:
                key = f"{alert_type}_{symbol}_{level:.2f}"
            else:
                key = f"{alert_type}_{symbol}"

            if key not in self.last_notifications:
                return None

            try:
                last_time = datetime.fromisoformat(self.last_notifications[key])
                current_time = get_current_time_ist()
                cooldown_delta = timedelta(minutes=self.cooldown_minutes)

                time_since_last = current_time - last_time

                if time_since_last >= cooldown_delta:
                    return None
                else:
                    remaining = cooldown_delta - time_since_last
                    return int(remaining.total_seconds())

            except Exception as e:
                print(f"Error calculating time until next notification: {e}")
                return None

    def clear_old_entries(self, days_old: int = 7):
        """
        Clear notification records older than specified days

        Args:
            days_old: Remove entries older than this many days
        """
        with self.lock:
            cutoff_time = get_current_time_ist() - timedelta(days=days_old)

            keys_to_remove = []
            for key, timestamp_str in self.last_notifications.items():
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp < cutoff_time:
                        keys_to_remove.append(key)
                except:
                    # Remove invalid entries
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.last_notifications[key]

            if keys_to_remove:
                self._save_to_file()
                print(f"Cleared {len(keys_to_remove)} old notification entries")

    def reset_all(self):
        """Clear all notification records"""
        with self.lock:
            self.last_notifications = {}
            self._save_to_file()


# Global instance
_rate_limiter = None

def get_rate_limiter(cooldown_minutes: int = 10) -> NotificationRateLimiter:
    """
    Get or create the global rate limiter instance

    Args:
        cooldown_minutes: Cooldown period in minutes

    Returns:
        NotificationRateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = NotificationRateLimiter(cooldown_minutes=cooldown_minutes)
    return _rate_limiter
