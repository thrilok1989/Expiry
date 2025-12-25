import requests
import os
import aiohttp
import html
import streamlit as st
from config import get_telegram_credentials, IST, get_current_time_ist
from datetime import datetime
from typing import Dict, Any, Optional

class TelegramBot:
    def __init__(self):
        """Initialize Telegram bot"""
        creds = get_telegram_credentials()
        self.enabled = creds['enabled']
        
        if self.enabled:
            self.bot_token = creds['bot_token']
            self.chat_id = creds['chat_id']
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_message(self, message: str, parse_mode: str = "HTML"):
        """Send Telegram message"""
        if not self.enabled:
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            response = requests.post(url, json=payload, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def send_message_async(self, message: str, parse_mode: str = "HTML"):
        """Send Telegram message asynchronously"""
        if not self.enabled:
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    return response.status == 200
        except:
            return False
    
    def send_signal_ready(self, setup: dict):
        """Send signal ready alert"""
        message = f"""
🎯 <b>SIGNAL READY - 3/3 Received</b>

<b>Index:</b> {setup['index']}
<b>Direction:</b> {setup['direction']}

<b>VOB Support:</b> {setup['vob_support']}
<b>VOB Resistance:</b> {setup['vob_resistance']}

<b>Status:</b> ✅ Ready to Trade
<b>Time (IST):</b> {get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')}

📱 Open app to execute trade
        """
        return self.send_message(message.strip())
    
    def send_vob_touch_alert(self, setup: dict, current_price: float):
        """Send VOB touch alert"""
        vob_level = setup['vob_support'] if setup['direction'] == 'CALL' else setup['vob_resistance']
        vob_type = "Support" if setup['direction'] == 'CALL' else "Resistance"
        
        message = f"""
🔥 <b>VOB TOUCHED - ENTRY SIGNAL!</b>

<b>Index:</b> {setup['index']}
<b>Direction:</b> {setup['direction']}

<b>Current Price:</b> {current_price}
<b>VOB {vob_type}:</b> {vob_level}

<b>Status:</b> 🚀 Ready to Execute
<b>Time (IST):</b> {get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')}

⚡ Execute trade NOW!
        """
        return self.send_message(message.strip())
    
    def send_order_placed(self, setup: dict, order_id: str, strike: int, 
                         sl: float, target: float):
        """Send order placed confirmation"""
        message = f"""
✅ <b>ORDER PLACED SUCCESSFULLY</b>

<b>Order ID:</b> {order_id}

<b>Index:</b> {setup['index']}
<b>Direction:</b> {setup['direction']}
<b>Strike:</b> {strike}

<b>Stop Loss:</b> {sl}
<b>Target:</b> {target}

<b>Time (IST):</b> {get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')}

📊 Monitor position in app
        """
        return self.send_message(message.strip())
    
    def send_order_failed(self, setup: dict, error: str):
        """Send order failure alert"""
        message = f"""
❌ <b>ORDER PLACEMENT FAILED</b>

<b>Index:</b> {setup['index']}
<b>Direction:</b> {setup['direction']}

<b>Error:</b> {error}

<b>Time (IST):</b> {get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')}

⚠️ Check app for details
        """
        return self.send_message(message.strip())
    
    def send_position_exit(self, order_id: str, pnl: float):
        """Send position exit alert"""
        pnl_emoji = "💰" if pnl > 0 else "📉"
        message = f"""
{pnl_emoji} <b>POSITION EXITED</b>

<b>Order ID:</b> {order_id}
<b>P&L:</b> ₹{pnl:,.2f}

<b>Time (IST):</b> {get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')}
        """
        return self.send_message(message.strip())

    def send_vob_entry_signal(self, signal: dict):
        """Send VOB-based entry signal alert"""
        signal_emoji = "🟢" if signal['direction'] == 'CALL' else "🔴"
        direction_label = "BULLISH" if signal['direction'] == 'CALL' else "BEARISH"

        message = f"""
{signal_emoji} <b>VOB ENTRY SIGNAL - {direction_label}</b>

<b>Index:</b> {signal['index']}
<b>Direction:</b> {signal['direction']}
<b>Market Sentiment:</b> {signal['market_sentiment']}

💰 <b>ENTRY LEVELS</b>
<b>Entry Price:</b> {signal['entry_price']}
<b>Stop Loss:</b> {signal['stop_loss']}
<b>Target:</b> {signal['target']}
<b>Risk:Reward:</b> {signal['risk_reward']}

📊 <b>VOB DETAILS</b>
<b>VOB Level:</b> {signal['vob_level']}
<b>Distance from VOB:</b> {signal['distance_from_vob']} points
<b>VOB Volume:</b> {signal['vob_volume']:,.0f}

<b>Time:</b> {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

⚡ <b>Execute trade NOW!</b>
        """
        return self.send_message(message.strip())

    def send_htf_sr_entry_signal(self, signal: dict):
        """Send HTF Support/Resistance entry signal alert"""
        signal_emoji = "🟢" if signal['direction'] == 'CALL' else "🔴"
        direction_label = "BULLISH" if signal['direction'] == 'CALL' else "BEARISH"

        # Format timeframe for display
        timeframe_display = {
            '5T': '5 Min',
            '10T': '10 Min',
            '15T': '15 Min'
        }.get(signal.get('timeframe', ''), signal.get('timeframe', 'N/A'))

        # Determine if it's support or resistance signal
        if signal['direction'] == 'CALL':
            level_type = "Support"
            level_value = signal['support_level']
        else:
            level_type = "Resistance"
            level_value = signal['resistance_level']

        message = f"""
{signal_emoji} <b>HTF S/R ENTRY SIGNAL - {direction_label}</b>

<b>Index:</b> {signal['index']}
<b>Direction:</b> {signal['direction']}
<b>Market Sentiment:</b> {signal['market_sentiment']}

💰 <b>ENTRY LEVELS</b>
<b>Entry Price:</b> {signal['entry_price']}
<b>Stop Loss:</b> {signal['stop_loss']}
<b>Target:</b> {signal['target']}
<b>Risk:Reward:</b> {signal['risk_reward']}

📊 <b>HTF S/R DETAILS</b>
<b>Timeframe:</b> {timeframe_display}
<b>{level_type} Level:</b> {level_value}
<b>Distance from Level:</b> {signal['distance_from_level']} points

<b>Time:</b> {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

⚡ <b>Execute trade NOW!</b>
        """
        return self.send_message(message.strip())

    def send_classic_entry_alert(self, signal_type: str, entry_zone: tuple, stop_loss: float,
                                 targets: dict, current_price: float, source: str,
                                 confirmations: dict) -> bool:
        """
        Send CLASSIC entry alert (simple VOB-based)

        Args:
            signal_type: "LONG" or "SHORT"
            entry_zone: (lower, upper) tuple
            stop_loss: SL price
            targets: {'t1': price, 't2': price}
            current_price: Current market price
            source: Entry source (e.g., "VOB Resistance")
            confirmations: Dict with regime, atm_bias, volume, price_action status
        """
        signal_emoji = "🟢" if signal_type == "LONG" else "🔴"
        direction_label = "LONG" if signal_type == "LONG" else "SHORT"

        # Calculate points
        entry_mid = (entry_zone[0] + entry_zone[1]) / 2
        sl_points = abs(stop_loss - entry_mid)
        t1_points = abs(targets.get('t1', entry_mid) - entry_mid)
        t2_points = abs(targets.get('t2', entry_mid) - entry_mid)

        # Count confirmations
        confirmed_count = sum(1 for v in confirmations.values() if '✅' in str(v))
        total_checks = len(confirmations)

        message = f"""
{signal_emoji} <b>CLASSIC {direction_label} SIGNAL - NIFTY</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Entry:</b> ₹{entry_zone[0]:,.0f} - ₹{entry_zone[1]:,.0f}
<b>Source:</b> {source}

🛑 <b>SL:</b> ₹{stop_loss:,.0f} ({'+' if signal_type == 'SHORT' else '-'}{sl_points:.0f}pts)
🎯 <b>T1:</b> ₹{targets.get('t1', 0):,.0f} ({'+' if signal_type == 'LONG' else '-'}{t1_points:.0f}pts)
🎯 <b>T2:</b> ₹{targets.get('t2', 0):,.0f} ({'+' if signal_type == 'LONG' else '-'}{t2_points:.0f}pts)

✅ <b>Confirmations: {confirmed_count}/{total_checks}</b>
• <b>Regime:</b> {confirmations.get('regime', 'N/A')}
• <b>ATM Bias:</b> {confirmations.get('atm_bias', 'N/A')}
• <b>Volume:</b> {confirmations.get('volume', 'N/A')}
• <b>Price Action:</b> {confirmations.get('price_action', 'N/A')}

⏰ {get_current_time_ist().strftime('%I:%M %p IST')}
📍 <b>Price:</b> ₹{current_price:,.0f}
        """
        return self.send_message(message.strip())

    def send_advanced_entry_alert(self, signal_type: str, pattern_type: str, entry_zone: tuple,
                                  smart_sl: dict, smart_targets: dict, confluence: dict,
                                  current_price: float, pattern_details: dict = None) -> bool:
        """
        Send ADVANCED entry alert (pattern-based with full confluence analysis)

        Args:
            signal_type: "LONG" or "SHORT"
            pattern_type: Pattern name (e.g., "Head & Shoulders Neckline")
            entry_zone: (lower, upper) tuple
            smart_sl: {'price': float, 'reason': str, 'risk_points': float, 'risk_percent': float,
                       'invalidation_triggers': list}
            smart_targets: {'t1': {...}, 't2': {...}, 't3': {...}} with price, confluence, sources
            confluence: {'score': float, 'confirmed': int, 'total': int, 'checks': {...}}
            current_price: Current market price
            pattern_details: Optional pattern metadata
        """
        signal_emoji = "🚀" if signal_type == "LONG" else "🔴"
        direction_label = "LONG" if signal_type == "LONG" else "SHORT"

        # Calculate points
        entry_mid = (entry_zone[0] + entry_zone[1]) / 2
        t1_points = abs(smart_targets['t1']['price'] - entry_mid)
        t2_points = abs(smart_targets['t2']['price'] - entry_mid)
        t3_points = abs(smart_targets['t3']['price'] - entry_mid)

        # Format confluence checks
        checks = confluence.get('checks', {})
        check_lines = []
        for key, value in checks.items():
            if isinstance(value, dict):
                status = value.get('status', '⚠️')
                detail = value.get('detail', '')
                check_lines.append(f"   {status} {key.replace('_', ' ').title()}: {detail}")
            else:
                check_lines.append(f"   {value}")

        checks_text = '\n'.join(check_lines[:8])  # Limit to 8 checks

        # Pattern details if available
        pattern_info = ""
        if pattern_details:
            pattern_info = f"""
📐 <b>Pattern Details:</b>
   Left Shoulder: ₹{pattern_details.get('left_shoulder', 0):,.0f}
   Head: ₹{pattern_details.get('head', 0):,.0f}
   Right Shoulder: ₹{pattern_details.get('right_shoulder', 0):,.0f}
   Neckline: ₹{pattern_details.get('neckline', 0):,.0f}
"""

        message = f"""
🚀 <b>ADVANCED {direction_label} SIGNAL - NIFTY</b>
━━━━━━━━━━━━━━━━━━━━━━

📊 <b>Pattern:</b> {pattern_type.upper()}
<b>Entry:</b> ₹{entry_zone[0]:,.0f} - ₹{entry_zone[1]:,.0f}

🛑 <b>Smart SL:</b> ₹{smart_sl['price']:,.0f} (+{smart_sl['risk_points']:.0f}pts)
   • {smart_sl['reason']}
   • <b>Risk:</b> {smart_sl['risk_percent']:.1f}%

🎯 <b>Smart Targets:</b>
   <b>T1:</b> ₹{smart_targets['t1']['price']:,.0f} ({'+' if signal_type == 'LONG' else '-'}{t1_points:.0f}pts)
      └─ {smart_targets['t1']['confluence']} ({smart_targets['t1']['source_count']} sources)

   <b>T2:</b> ₹{smart_targets['t2']['price']:,.0f} ({'+' if signal_type == 'LONG' else '-'}{t2_points:.0f}pts) ⭐
      └─ {smart_targets['t2']['confluence']} ({smart_targets['t2']['source_count']} sources)

   <b>T3:</b> ₹{smart_targets['t3']['price']:,.0f} ({'+' if signal_type == 'LONG' else '-'}{t3_points:.0f}pts)
      └─ {smart_targets['t3']['confluence']} ({smart_targets['t3']['source_count']} sources)

🔍 <b>Confluence:</b> {confluence['score']:.0f}% ({confluence['confirmed']}/{confluence['total']} confirmations)
{checks_text}
{pattern_info}
⏰ {get_current_time_ist().strftime('%I:%M %p IST')}
📍 <b>Price:</b> ₹{current_price:,.0f}
        """
        return self.send_message(message.strip())

    def send_oi_unwinding_alert(self, position_type: str, entry_strike: int,
                               oi_check_result: dict, current_price: float) -> bool:
        """
        Send OI unwinding exit alert

        Args:
            position_type: "LONG" or "SHORT"
            entry_strike: Entry strike price
            oi_check_result: Result from OIShiftMonitor.check_oi_shift()
            current_price: Current market price
        """
        alert_emoji = "🚨" if oi_check_result['action'] == 'EXIT_ALL' else "⚠️"

        message = f"""
{alert_emoji} <b>OI UNWINDING ALERT - NIFTY</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Position:</b> {position_type} from ₹{entry_strike:,}
<b>Current Price:</b> ₹{current_price:,.0f}

🔴 <b>OI Change:</b>
{oi_check_result['details']}
<b>Change:</b> {oi_check_result['oi_change_pct']:.1f}% {alert_emoji}

⚠️ <b>SUPPORT/RESISTANCE WALL COLLAPSING!</b>

<b>Action:</b> {oi_check_result['action'].replace('_', ' ')}
<b>Reason:</b> {oi_check_result['reason']}

⏰ {get_current_time_ist().strftime('%I:%M %p IST')}

🚨 Your entry level S/R is disappearing!
        """
        return self.send_message(message.strip())

    def send_opposite_oi_buildup_alert(self, position_type: str, barrier_result: dict,
                                      current_price: float) -> bool:
        """
        Send alert when fresh OI builds on opposite side

        Args:
            position_type: "LONG" or "SHORT"
            barrier_result: Result from OIShiftMonitor.check_opposite_side_buildup()
            current_price: Current market price
        """
        alert_emoji = "🔴" if barrier_result['alert_priority'] == 'CRITICAL' else "⚠️"

        # Format barrier details
        barriers_text = ""
        for barrier in barrier_result.get('barriers', [])[:3]:  # Top 3
            barriers_text += f"\n• ₹{barrier['strike']:,} (+{barrier['oi_change']:,} OI, {barrier['distance']}pts away)"

        message = f"""
{alert_emoji} <b>NEW BARRIER FORMING - NIFTY</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Position:</b> {position_type}
<b>Current Price:</b> ₹{current_price:,.0f}

🔴 <b>Fresh OI Detected:</b>{barriers_text}

⚠️ <b>NEW {'RESISTANCE' if position_type == 'LONG' else 'SUPPORT'} FORMING!</b>

<b>Action:</b> {barrier_result['action'].replace('_', ' ')}
<b>Reason:</b> {barrier_result['reason']}

⏰ {get_current_time_ist().strftime('%I:%M %p IST')}

💡 Consider tightening SL or partial exit
        """
        return self.send_message(message.strip())

    def send_volume_spike_alert(self, position_type: str, volume_check: dict,
                               current_price: float) -> bool:
        """
        Send volume spike exit alert

        Args:
            position_type: "LONG" or "SHORT"
            volume_check: Result from VolumeSpikeMonitor.check_volume_spike()
            current_price: Current market price
        """
        alert_emoji = "🚨" if volume_check['severity'] == 'CRITICAL' else "⚠️"

        message = f"""
{alert_emoji} <b>VOLUME SPIKE ALERT - NIFTY</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Position:</b> {position_type}
<b>Current Price:</b> ₹{current_price:,.0f}

🔴 <b>VOLUME SPIKE DETECTED:</b>
<b>Volume Ratio:</b> {volume_check['volume_ratio']:.1f}x average
<b>Buy Volume:</b> {volume_check['buy_volume']:,} ({volume_check['buy_pct']:.0f}%)
<b>Sell Volume:</b> {volume_check['sell_volume']:,} ({volume_check['sell_pct']:.0f}%)
<b>Delta:</b> {volume_check['delta']:,}

⚠️ <b>INSTITUTIONAL MOVE DETECTED!</b>

<b>Action:</b> {volume_check['action'].replace('_', ' ')}
<b>Reason:</b> {volume_check['reason']}

⏰ {get_current_time_ist().strftime('%I:%M %p IST')}

🚨 Market shifting against you!
        """
        return self.send_message(message.strip())

    def send_volume_absorption_alert(self, position_type: str, absorption_result: dict,
                                    current_price: float) -> bool:
        """
        Send volume absorption exit alert

        Args:
            position_type: "LONG" or "SHORT"
            absorption_result: Result from VolumeSpikeMonitor.detect_absorption()
            current_price: Current market price
        """
        sr_type = "RESISTANCE" if position_type == "LONG" else "SUPPORT"

        message = f"""
⚠️ <b>VOLUME ABSORPTION ALERT - NIFTY</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Position:</b> {position_type}
<b>Current Price:</b> ₹{current_price:,.0f}

🔴 <b>Absorption Detected:</b>
<b>Total Volume:</b> {absorption_result['total_volume']:,}
<b>Price Change:</b> Only {absorption_result['price_change_pct']:.2f}%

⚠️ <b>{sr_type} DEFENDING STRONGLY!</b>

<b>Action:</b> {absorption_result['action'].replace('_', ' ')}
<b>Reason:</b> {absorption_result['reason']}

⏰ {get_current_time_ist().strftime('%I:%M %p IST')}

💡 High volume but price not breaking
Sellers/buyers absorbing all pressure
        """
        return self.send_message(message.strip())

    def send_vob_status_summary(self, nifty_data: dict, sensex_data: dict):
        """Send VOB status summary for both NIFTY and SENSEX"""

        def format_vob_block(symbol: str, vob_type: str, block_data: dict):
            """Format a single VOB block display"""
            emoji = "🟢" if vob_type == "Bullish" else "🔴"
            strength = block_data.get('strength_score', 0)
            trend = block_data.get('trend', 'STABLE')
            lower = block_data.get('lower', 0)
            upper = block_data.get('upper', 0)

            # Determine trend emoji
            if trend == "STRENGTHENING":
                trend_emoji = "🔺"
                trend_text = "STRENGTHENING"
            elif trend == "WEAKENING":
                trend_emoji = "🔻"
                trend_text = "WEAKENING"
            else:
                trend_emoji = "➖"
                trend_text = "STABLE"

            return f"""
{emoji} <b>{vob_type} VOB:</b> ₹{lower:.2f} - ₹{upper:.2f}

<b>Strength:</b> {strength:.1f}/100 {trend_emoji} {trend_text}"""

        message_parts = [
            "<b>📊 Volume Order Block Status</b>",
            "",
            "<b>NIFTY VOB</b>"
        ]

        # Add NIFTY VOB data
        if nifty_data.get('bullish'):
            message_parts.append(format_vob_block("NIFTY", "Bullish", nifty_data['bullish']))
        else:
            message_parts.append("🟢 <b>Bullish VOB:</b> No data available")

        if nifty_data.get('bearish'):
            message_parts.append(format_vob_block("NIFTY", "Bearish", nifty_data['bearish']))
        else:
            message_parts.append("🔴 <b>Bearish VOB:</b> No data available")

        message_parts.extend(["", "<b>SENSEX VOB</b>"])

        # Add SENSEX VOB data
        if sensex_data.get('bullish'):
            message_parts.append(format_vob_block("SENSEX", "Bullish", sensex_data['bullish']))
        else:
            message_parts.append("🟢 <b>Bullish VOB:</b> No data available")

        if sensex_data.get('bearish'):
            message_parts.append(format_vob_block("SENSEX", "Bearish", sensex_data['bearish']))
        else:
            message_parts.append("🔴 <b>Bearish VOB:</b> No data available")

        message_parts.extend([
            "",
            f"<i>Updated (IST): {get_current_time_ist().strftime('%I:%M:%S %p %Z')}</i>"
        ])

        message = "\n".join(message_parts)
        return self.send_message(message)

    def send_htf_sr_status_summary(self, nifty_htf: dict, sensex_htf: dict):
        """Send HTF Support/Resistance status summary"""

        def format_htf_levels(symbol: str, htf_data: dict):
            """Format HTF S/R levels for display"""
            lines = [f"<b>{symbol}</b>"]

            for timeframe, levels in htf_data.items():
                if not levels:
                    continue

                # Format timeframe for display
                tf_display = timeframe.replace('T', 'min')

                support = levels.get('support')
                resistance = levels.get('resistance')
                support_strength = levels.get('support_strength', {})
                resistance_strength = levels.get('resistance_strength', {})

                if support:
                    s_score = support_strength.get('strength_score', 0)
                    s_trend = support_strength.get('trend', 'STABLE')
                    s_emoji = "🔺" if s_trend == "STRENGTHENING" else "🔻" if s_trend == "WEAKENING" else "➖"
                    lines.append(f"  🟢 {tf_display} Support: ₹{support:.2f} ({s_score:.1f}/100 {s_emoji})")

                if resistance:
                    r_score = resistance_strength.get('strength_score', 0)
                    r_trend = resistance_strength.get('trend', 'STABLE')
                    r_emoji = "🔺" if r_trend == "STRENGTHENING" else "🔻" if r_trend == "WEAKENING" else "➖"
                    lines.append(f"  🔴 {tf_display} Resistance: ₹{resistance:.2f} ({r_score:.1f}/100 {r_emoji})")

            return "\n".join(lines) if len(lines) > 1 else f"<b>{symbol}</b>\n  No HTF data available"

        message_parts = [
            "<b>📊 HTF Support/Resistance Status</b>",
            "<b>5min, 10min, 15min Timeframes</b>",
            ""
        ]

        # Add NIFTY HTF data
        message_parts.append(format_htf_levels("NIFTY", nifty_htf))
        message_parts.append("")

        # Add SENSEX HTF data
        message_parts.append(format_htf_levels("SENSEX", sensex_htf))

        message_parts.extend([
            "",
            f"<i>Updated (IST): {get_current_time_ist().strftime('%I:%M:%S %p %Z')}</i>"
        ])

        message = "\n".join(message_parts)
        return self.send_message(message)

    async def send_ai_market_alert(self, report: Dict[str, Any], confidence_thresh: float = 0.60) -> bool:
        """
        Send AI market alert to Telegram
        """
        if not report:
            return False
        
        confidence = float(report.get("confidence", 0.0) or 0.0)
        if confidence < confidence_thresh:
            return False
        
        label = report.get("label", "UNKNOWN")
        rec = report.get("recommendation", "HOLD")
        tech = report.get("technical_score", 0.0)
        news = report.get("news_score", 0.0)
        ai_score = report.get("ai_score", 0.0)
        reasons = report.get("ai_reasons", []) or []
        ai_summary = report.get("ai_summary", "") or ""

        # Format top reasons
        top_reasons = "\n".join(f"{i+1}. {html.escape(str(r))}" for i, r in enumerate(reasons[:4]))
        
        # Format technical contributions
        tech_lines = []
        contribs = report.get("technical_contributions", {}) or {}
        for k, v in contribs.items():
            sign = "Bullish" if v > 0 else "Bearish" if v < 0 else "Neutral"
            tech_lines.append(f"{html.escape(k)}: {sign} ({v:.3f})")
        
        tech_block = "\n".join(tech_lines[:6]) or "N/A"
        
        # Determine emoji based on bias
        if "BULLISH" in label.upper():
            bias_emoji = "🟢"
        elif "BEARISH" in label.upper():
            bias_emoji = "🔴"
        else:
            bias_emoji = "⚪"

        # Create the message
        text = f"""
{bias_emoji} <b>🤖 AI MARKET REPORT</b>

<b>Market:</b> {html.escape(str(report.get('market','')))}
<b>Bias:</b> <b>{bias_emoji} {html.escape(label)}</b>
<b>Recommendation:</b> <b>{html.escape(rec)}</b>

📊 <b>SCORES</b>
<b>Confidence:</b> {confidence:.2f}
<b>AI Score:</b> {ai_score:.3f}
<b>Technical Score:</b> {tech:.3f}
<b>News Score:</b> {news:.3f}

⚙️ <b>TECHNICAL SUMMARY</b>
{tech_block}

📰 <b>NEWS SUMMARY</b>
{html.escape(str(report.get('news_summary',''))[:600])}

🧠 <b>AI REASONING</b>
{top_reasons}

📋 <b>SUMMARY</b>
{html.escape(ai_summary[:800])}

⏰ <b>Time (IST):</b> {get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')}
        """
        
        return await self.send_message_async(text.strip())


def send_test_message():
    """Send test message to verify Telegram setup"""
    bot = TelegramBot()
    if bot.enabled:
        message = """
✅ <b>Telegram Connected!</b>

Your trading alerts are now active.

<b>Test Time (IST):</b> """ + get_current_time_ist().strftime('%Y-%m-%d %H:%M:%S %Z')
        return bot.send_message(message.strip())
    return False


# Async wrapper function for backward compatibility
async def send_ai_market_alert_async(report: Dict[str, Any], confidence_thresh: float = 0.60) -> bool:
    """
    Async wrapper function for sending AI market alerts
    """
    bot = TelegramBot()
    if not bot.enabled:
        return False
    return await bot.send_ai_market_alert(report, confidence_thresh)
