"""
Telegram Signal Manager for Market Regime XGBoost Complete Signal System

Manages Telegram alerts with:
- 5 alert types with different priorities
- Cooldown periods to prevent spam
- Rate limiting
- Alert history tracking
- Duplicate detection
"""

import asyncio
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from collections import deque

from telegram_alerts import TelegramBot
from src.enhanced_signal_generator import TradingSignal, format_signal_for_telegram

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuration for alert type"""
    alert_type: str
    priority: int  # 1 (highest) to 5 (lowest)
    cooldown_seconds: int  # Minimum time between alerts of this type
    enabled: bool = True
    max_per_hour: int = 10  # Maximum alerts per hour


@dataclass
class AlertHistory:
    """Track alert history for cooldown and rate limiting"""
    alert_type: str
    timestamp: datetime
    signal: TradingSignal
    telegram_sent: bool
    telegram_error: Optional[str] = None


class TelegramSignalManager:
    """
    Manages Telegram alerts for trading signals with cooldown and rate limiting

    Alert Types and Priorities:
    1. ENTRY (Priority 1) - Immediate trade opportunity
    2. EXIT (Priority 1) - Immediate exit signal
    3. DIRECTION_CHANGE (Priority 2) - Trend reversal alert
    4. BIAS_CHANGE (Priority 3) - Sentiment shift alert
    5. WAIT (Priority 5) - Low priority, informational only

    Cooldown Periods:
    - ENTRY: 300 seconds (5 minutes)
    - EXIT: 180 seconds (3 minutes)
    - DIRECTION_CHANGE: 600 seconds (10 minutes)
    - BIAS_CHANGE: 900 seconds (15 minutes)
    - WAIT: 1800 seconds (30 minutes)
    """

    def __init__(
        self,
        telegram_bot: Optional[TelegramBot] = None,
        enable_telegram: bool = True
    ):
        """
        Initialize Telegram Signal Manager

        Args:
            telegram_bot: TelegramBot instance (if None, will create new one)
            enable_telegram: Enable/disable Telegram sending (for testing)
        """
        self.telegram_bot = telegram_bot
        self.enable_telegram = enable_telegram

        # Alert configurations
        self.alert_configs = {
            "ENTRY": AlertConfig(
                alert_type="ENTRY",
                priority=1,
                cooldown_seconds=300,  # 5 minutes
                max_per_hour=6
            ),
            "EXIT": AlertConfig(
                alert_type="EXIT",
                priority=1,
                cooldown_seconds=180,  # 3 minutes
                max_per_hour=10
            ),
            "DIRECTION_CHANGE": AlertConfig(
                alert_type="DIRECTION_CHANGE",
                priority=2,
                cooldown_seconds=600,  # 10 minutes
                max_per_hour=4
            ),
            "BIAS_CHANGE": AlertConfig(
                alert_type="BIAS_CHANGE",
                priority=3,
                cooldown_seconds=900,  # 15 minutes
                max_per_hour=3
            ),
            "WAIT": AlertConfig(
                alert_type="WAIT",
                priority=5,
                cooldown_seconds=1800,  # 30 minutes
                max_per_hour=2
            )
        }

        # Alert history (store last 100 alerts)
        self.alert_history: deque = deque(maxlen=100)

        # Last alert timestamp by type
        self.last_alert_time: Dict[str, datetime] = {}

        # Hourly alert counters
        self.hourly_counters: Dict[str, List[datetime]] = {
            alert_type: [] for alert_type in self.alert_configs.keys()
        }

        # Statistics
        self.stats = {
            "total_alerts_generated": 0,
            "total_alerts_sent": 0,
            "total_alerts_blocked_cooldown": 0,
            "total_alerts_blocked_rate_limit": 0,
            "total_telegram_errors": 0
        }

    async def send_signal_alert(
        self,
        signal: TradingSignal,
        force: bool = False
    ) -> Dict[str, any]:
        """
        Send trading signal alert via Telegram

        Args:
            signal: TradingSignal to send
            force: Force send even if cooldown/rate limit active

        Returns:
            Dict with status and details
        """
        self.stats["total_alerts_generated"] += 1

        alert_type = signal.signal_type

        # Check if alert type is configured
        if alert_type not in self.alert_configs:
            logger.warning(f"Unknown alert type: {alert_type}")
            return {
                "success": False,
                "reason": f"Unknown alert type: {alert_type}",
                "sent": False
            }

        config = self.alert_configs[alert_type]

        # Check if alert type is enabled
        if not config.enabled:
            logger.info(f"Alert type {alert_type} is disabled")
            return {
                "success": False,
                "reason": f"Alert type {alert_type} is disabled",
                "sent": False
            }

        # Check cooldown (unless forced)
        if not force:
            cooldown_result = self._check_cooldown(alert_type, config)
            if not cooldown_result["allowed"]:
                self.stats["total_alerts_blocked_cooldown"] += 1
                logger.info(f"Alert blocked by cooldown: {alert_type}")
                return {
                    "success": False,
                    "reason": cooldown_result["reason"],
                    "sent": False,
                    "cooldown_remaining": cooldown_result["remaining_seconds"]
                }

            # Check rate limit
            rate_limit_result = self._check_rate_limit(alert_type, config)
            if not rate_limit_result["allowed"]:
                self.stats["total_alerts_blocked_rate_limit"] += 1
                logger.info(f"Alert blocked by rate limit: {alert_type}")
                return {
                    "success": False,
                    "reason": rate_limit_result["reason"],
                    "sent": False
                }

        # Format message
        message = format_signal_for_telegram(signal)

        # Add alert metadata
        message += f"\n⏰ Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

        # Send via Telegram
        telegram_sent = False
        telegram_error = None

        if self.enable_telegram and self.telegram_bot:
            try:
                # Send message
                success = await self._send_telegram_message(message, config.priority)

                if success:
                    telegram_sent = True
                    self.stats["total_alerts_sent"] += 1
                    logger.info(f"✅ Telegram alert sent: {alert_type}")
                else:
                    telegram_error = "Failed to send message"
                    self.stats["total_telegram_errors"] += 1
                    logger.error(f"❌ Failed to send Telegram alert: {alert_type}")

            except Exception as e:
                telegram_error = str(e)
                self.stats["total_telegram_errors"] += 1
                logger.error(f"❌ Telegram error: {e}")
        else:
            logger.info(f"Telegram disabled or bot not configured")

        # Record alert in history
        alert_record = AlertHistory(
            alert_type=alert_type,
            timestamp=datetime.now(),
            signal=signal,
            telegram_sent=telegram_sent,
            telegram_error=telegram_error
        )
        self.alert_history.append(alert_record)

        # Update last alert time
        self.last_alert_time[alert_type] = datetime.now()

        # Update hourly counter
        self.hourly_counters[alert_type].append(datetime.now())

        return {
            "success": True,
            "sent": telegram_sent,
            "alert_type": alert_type,
            "priority": config.priority,
            "timestamp": alert_record.timestamp,
            "telegram_error": telegram_error
        }

    def _check_cooldown(self, alert_type: str, config: AlertConfig) -> Dict:
        """
        Check if cooldown period has elapsed

        Returns:
            Dict with 'allowed' bool and 'reason' string
        """
        if alert_type not in self.last_alert_time:
            return {"allowed": True, "reason": "No previous alert"}

        last_time = self.last_alert_time[alert_type]
        elapsed = (datetime.now() - last_time).total_seconds()

        if elapsed < config.cooldown_seconds:
            remaining = config.cooldown_seconds - elapsed
            return {
                "allowed": False,
                "reason": f"Cooldown active for {alert_type} ({remaining:.0f}s remaining)",
                "remaining_seconds": remaining
            }

        return {"allowed": True, "reason": "Cooldown elapsed"}

    def _check_rate_limit(self, alert_type: str, config: AlertConfig) -> Dict:
        """
        Check if rate limit has been exceeded

        Returns:
            Dict with 'allowed' bool and 'reason' string
        """
        # Clean up old timestamps (older than 1 hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.hourly_counters[alert_type] = [
            ts for ts in self.hourly_counters[alert_type]
            if ts > cutoff_time
        ]

        # Check count
        count = len(self.hourly_counters[alert_type])

        if count >= config.max_per_hour:
            return {
                "allowed": False,
                "reason": f"Rate limit exceeded for {alert_type} ({count}/{config.max_per_hour} per hour)"
            }

        return {"allowed": True, "reason": "Within rate limit"}

    async def _send_telegram_message(self, message: str, priority: int) -> bool:
        """
        Send message via Telegram with retry logic

        Args:
            message: Message text
            priority: Priority level (1-5)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.telegram_bot:
            logger.warning("Telegram bot not configured")
            return False

        # Retry logic based on priority
        max_retries = 3 if priority <= 2 else 1
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # Send message using telegram_alerts module
                result = await asyncio.to_thread(
                    self.telegram_bot.send_message,
                    message
                )

                if result:
                    return True
                else:
                    logger.warning(f"Telegram send failed (attempt {attempt + 1}/{max_retries})")

            except Exception as e:
                logger.error(f"Telegram error on attempt {attempt + 1}: {e}")

            # Wait before retry (except on last attempt)
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

        return False

    def get_cooldown_status(self, alert_type: str) -> Dict:
        """
        Get current cooldown status for an alert type

        Returns:
            Dict with cooldown information
        """
        if alert_type not in self.alert_configs:
            return {"error": f"Unknown alert type: {alert_type}"}

        config = self.alert_configs[alert_type]

        if alert_type not in self.last_alert_time:
            return {
                "alert_type": alert_type,
                "cooldown_active": False,
                "cooldown_seconds": config.cooldown_seconds,
                "time_remaining": 0
            }

        last_time = self.last_alert_time[alert_type]
        elapsed = (datetime.now() - last_time).total_seconds()
        remaining = max(0, config.cooldown_seconds - elapsed)

        return {
            "alert_type": alert_type,
            "cooldown_active": remaining > 0,
            "cooldown_seconds": config.cooldown_seconds,
            "time_remaining": remaining,
            "last_alert": last_time.isoformat()
        }

    def get_rate_limit_status(self, alert_type: str) -> Dict:
        """
        Get current rate limit status for an alert type

        Returns:
            Dict with rate limit information
        """
        if alert_type not in self.alert_configs:
            return {"error": f"Unknown alert type: {alert_type}"}

        config = self.alert_configs[alert_type]

        # Clean up old timestamps
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.hourly_counters[alert_type] = [
            ts for ts in self.hourly_counters[alert_type]
            if ts > cutoff_time
        ]

        count = len(self.hourly_counters[alert_type])

        return {
            "alert_type": alert_type,
            "count_last_hour": count,
            "max_per_hour": config.max_per_hour,
            "rate_limit_active": count >= config.max_per_hour,
            "remaining_allowance": max(0, config.max_per_hour - count)
        }

    def get_statistics(self) -> Dict:
        """Get alert statistics"""
        return {
            **self.stats,
            "cooldown_status": {
                alert_type: self.get_cooldown_status(alert_type)
                for alert_type in self.alert_configs.keys()
            },
            "rate_limit_status": {
                alert_type: self.get_rate_limit_status(alert_type)
                for alert_type in self.alert_configs.keys()
            },
            "total_alerts_in_history": len(self.alert_history)
        }

    def get_recent_alerts(self, limit: int = 10) -> List[AlertHistory]:
        """
        Get recent alerts from history

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of AlertHistory objects
        """
        return list(self.alert_history)[-limit:]

    def clear_cooldowns(self):
        """Clear all cooldowns (for testing or manual override)"""
        self.last_alert_time.clear()
        logger.info("All cooldowns cleared")

    def reset_statistics(self):
        """Reset alert statistics"""
        self.stats = {
            "total_alerts_generated": 0,
            "total_alerts_sent": 0,
            "total_alerts_blocked_cooldown": 0,
            "total_alerts_blocked_rate_limit": 0,
            "total_telegram_errors": 0
        }
        logger.info("Statistics reset")

    def enable_alert_type(self, alert_type: str):
        """Enable a specific alert type"""
        if alert_type in self.alert_configs:
            self.alert_configs[alert_type].enabled = True
            logger.info(f"Alert type {alert_type} enabled")

    def disable_alert_type(self, alert_type: str):
        """Disable a specific alert type"""
        if alert_type in self.alert_configs:
            self.alert_configs[alert_type].enabled = False
            logger.info(f"Alert type {alert_type} disabled")

    def set_cooldown(self, alert_type: str, cooldown_seconds: int):
        """
        Set custom cooldown for an alert type

        Args:
            alert_type: Alert type to configure
            cooldown_seconds: Cooldown period in seconds
        """
        if alert_type in self.alert_configs:
            self.alert_configs[alert_type].cooldown_seconds = cooldown_seconds
            logger.info(f"Cooldown for {alert_type} set to {cooldown_seconds}s")

    def set_rate_limit(self, alert_type: str, max_per_hour: int):
        """
        Set custom rate limit for an alert type

        Args:
            alert_type: Alert type to configure
            max_per_hour: Maximum alerts per hour
        """
        if alert_type in self.alert_configs:
            self.alert_configs[alert_type].max_per_hour = max_per_hour
            logger.info(f"Rate limit for {alert_type} set to {max_per_hour}/hour")


async def send_signal_with_telegram(
    signal: TradingSignal,
    telegram_manager: TelegramSignalManager,
    force: bool = False
) -> Dict:
    """
    Convenience function to send signal alert

    Args:
        signal: TradingSignal to send
        telegram_manager: TelegramSignalManager instance
        force: Force send even if cooldown active

    Returns:
        Dict with send result
    """
    return await telegram_manager.send_signal_alert(signal, force=force)


def format_alert_statistics(stats: Dict) -> str:
    """
    Format alert statistics for display

    Args:
        stats: Statistics dict from get_statistics()

    Returns:
        Formatted string
    """
    report = f"""
📊 TELEGRAM ALERT STATISTICS
{'=' * 50}

Total Alerts Generated: {stats['total_alerts_generated']}
✅ Successfully Sent: {stats['total_alerts_sent']}
⏸️ Blocked (Cooldown): {stats['total_alerts_blocked_cooldown']}
⏸️ Blocked (Rate Limit): {stats['total_alerts_blocked_rate_limit']}
❌ Telegram Errors: {stats['total_telegram_errors']}

{'=' * 50}
COOLDOWN STATUS:
"""

    for alert_type, status in stats['cooldown_status'].items():
        if 'error' not in status:
            active = "🔴 ACTIVE" if status['cooldown_active'] else "🟢 READY"
            remaining = f" ({status['time_remaining']:.0f}s remaining)" if status['cooldown_active'] else ""
            report += f"\n  {alert_type}: {active}{remaining}"

    report += f"\n\n{'=' * 50}\nRATE LIMIT STATUS:\n"

    for alert_type, status in stats['rate_limit_status'].items():
        if 'error' not in status:
            count = status['count_last_hour']
            limit = status['max_per_hour']
            remaining = status['remaining_allowance']
            report += f"\n  {alert_type}: {count}/{limit} (remaining: {remaining})"

    return report
