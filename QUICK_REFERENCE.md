# 📋 Quick Reference Card

## 🎯 Essential Information at a Glance

### 📞 Important URLs

```
DhanHQ Web:        https://web.dhan.co
API Documentation: https://dhanhq.co/docs
Security ID List:  https://images.dhan.co/api-data/api-scrip-master.csv
Telegram BotFather: https://t.me/BotFather
Chat ID Bot:       https://t.me/userinfobot
```

### 🔑 Common Security IDs

| Symbol | Security ID | Exchange | Type |
|--------|-------------|----------|------|
| NIFTY 50 | 13 | IDX_I | Index |
| BANK NIFTY | 25 | IDX_I | Index |
| SENSEX | 51 | IDX_I | Index |
| Reliance | 1333 | NSE_EQ | Stock |
| TCS | 11536 | NSE_EQ | Stock |
| HDFC Bank | 1333 | NSE_EQ | Stock |
| Infosys | 7229 | NSE_EQ | Stock |
| ITC | 5258 | NSE_EQ | Stock |

### ⚙️ Default Settings

```python
# Indicator Parameters
VOB Length: 5
HTF Pivot Length: 5
HTF Timeframes: 10T, 15T
VIDYA Length: 10
VIDYA Momentum: 20
RSI Length: 14
RSI Smooth: 14

# Alert Settings
Alert Distance: 5.0 points
Cooling Period: 600 seconds (10 minutes)
Auto Refresh: 60 seconds (1 minute)

# Data Settings
Interval: 1 minute
Days Back: 5
```

### 🎨 Indicator Color Codes

| Indicator | Color | Meaning |
|-----------|-------|---------|
| 🟢 Bullish VOB | Green/Teal | Demand Zone |
| 🔴 Bearish VOB | Purple | Supply Zone |
| 🟦 10T HTF Level | Green | Support/Resistance |
| 🟥 15T HTF Level | Red | Support/Resistance |
| 🟠 VIDYA | Orange | Trend Line |
| ⚪ RSI | Silver | Momentum |
| 🟧 RSI Signal | Orange | Signal Line |

### 📊 RSI Interpretation

```
RSI Value | Zone | Action
----------|------|-------
80-100    | 🔴 Overbought | Consider shorts
60-80     | 🟡 Bullish | Hold longs
40-60     | ⚪ Neutral | Wait for setup
20-40     | 🟢 Bearish | Look for reversal
0-20      | 🟢 Oversold | Consider longs
```

### 🔔 Alert Types & Emojis

| Alert Type | Emoji | Description |
|------------|-------|-------------|
| Bullish VOB | 🟢 | Price near demand zone |
| Bearish VOB | 🔴 | Price near supply zone |
| HTF Resistance | 🔵 | Price near resistance |
| HTF Support | 🟡 | Price near support |

### ⚡ Keyboard Shortcuts

```
Ctrl + R / Cmd + R     Reload page
Ctrl + Shift + R       Hard refresh (clear cache)
F5                     Refresh page
Ctrl + -               Zoom out
Ctrl + +               Zoom in
Ctrl + 0               Reset zoom
```

### 📱 Quick Commands

```bash
# Start Application
streamlit run trading_app.py

# Install Dependencies
pip install -r requirements.txt

# Update Packages
pip install --upgrade -r requirements.txt

# Check Python Version
python --version

# Clear Streamlit Cache
streamlit cache clear
```

### 🔧 Troubleshooting Checklist

```
□ Python 3.8+ installed?
□ Dependencies installed?
□ Access Token valid (<24 hours)?
□ Client ID correct?
□ Security ID exists?
□ Market hours (9:15 AM - 3:30 PM IST)?
□ Internet connection stable?
□ Telegram bot started?
□ Browser cache cleared?
□ Auto-refresh enabled?
```

### 💡 Quick Tips

**DO:**
✓ Regenerate token daily
✓ Use stop losses always
✓ Wait for multi-indicator confirmation
✓ Respect cooling period
✓ Paper trade first
✓ Track performance
✓ Follow your plan

**DON'T:**
✗ Trade without stop loss
✗ Ignore alerts repeatedly
✗ Overtrade
✗ Chase prices
✗ Trust single indicator
✗ Trade during low volume
✗ Risk more than 2% per trade

### 📈 Trading Workflow

```
Morning (Pre-Market):
├── 8:00 AM - Regenerate Access Token
├── 8:30 AM - Start Application
├── 8:45 AM - Review Key Levels
└── 9:00 AM - Plan Trades

Market Hours:
├── 9:15 AM - Market Opens
├── Monitor Telegram Alerts
├── Check Multi-Indicator Confluence
├── Execute Planned Trades
└── 3:30 PM - Market Closes

Evening (Post-Market):
├── Review Performance
├── Update Trading Journal
└── Plan for Tomorrow
```

### 🎯 Entry Checklist

Before taking any trade:

```
□ Alert received?
□ Price at key level?
□ RSI confirming?
□ VIDYA trend aligned?
□ Volume adequate?
□ Stop loss defined?
□ Target identified?
□ Position size calculated?
□ Risk < 2% of capital?
□ Multiple confirmations?
```

### 🚨 Emergency Contacts

```
DhanHQ Support:
- Web: https://support.dhanhq.co
- Email: support@dhan.co
- Phone: [Check website]

Technical Issues:
- Streamlit: https://discuss.streamlit.io
- Python: https://stackoverflow.com
```

### 📊 Performance Tracking Template

```
Trade Log:
Date: ___/___/___
Symbol: _________
Entry: ₹________
Exit: ₹_________
P&L: ₹__________
R:R: ___:___
Setup: __________
Notes: ___________

Daily Summary:
Trades: ___
Wins: ___
Losses: ___
Win Rate: ___%
Net P&L: ₹_____
```

### 🔒 Security Reminders

```
⚠️ NEVER share:
- Access Token
- Client ID
- Telegram Bot Token
- API credentials

✓ ALWAYS:
- Use HTTPS
- Keep tokens private
- Regenerate if exposed
- Log out when done
```

### ⏰ Market Timings (IST)

```
NSE/BSE Equity:
Pre-Open:    9:00 AM - 9:15 AM
Trading:     9:15 AM - 3:30 PM
Post-Close:  3:40 PM - 4:00 PM

NSE F&O:
Trading:     9:15 AM - 3:30 PM

MCX:
9:00 AM - 11:30 PM / 11:55 PM
(varies by commodity)
```

### 📏 Risk Management Rules

```
Position Sizing:
- Max risk per trade: 1-2%
- Max open positions: 3-5
- Max daily loss: 3-5%

Stop Loss:
- Always mandatory
- Place immediately
- Never move away from entry
- Only trail in profit

Take Profit:
- Book 50% at 1:1 R:R
- Trail rest with stop
- Don't be greedy
```

### 🎓 Learning Resources Priority

```
Priority 1 (Must Read):
1. QUICKSTART.md
2. README.md
3. EXAMPLES.md

Priority 2 (Important):
4. TROUBLESHOOTING.md
5. PROJECT_SUMMARY.md

Priority 3 (Advanced):
6. ARCHITECTURE.md
7. config_template.py
```

### 🔄 Maintenance Schedule

```
Daily:
- Regenerate Access Token
- Clear browser cache
- Check alert system

Weekly:
- Review trading performance
- Adjust parameters if needed
- Update security IDs

Monthly:
- Update Python packages
- Review documentation
- Optimize strategy
```

### 💰 Capital Allocation Guide

```
Starting Capital: ₹100,000

Conservative:
- Per Trade: ₹1,000 (1%)
- Stop Loss: ₹500 (0.5%)
- Target: ₹1,500 (1.5%)

Moderate:
- Per Trade: ₹2,000 (2%)
- Stop Loss: ₹1,000 (1%)
- Target: ₹3,000 (3%)

Aggressive:
- Per Trade: ₹3,000 (3%)
- Stop Loss: ₹1,500 (1.5%)
- Target: ₹4,500 (4.5%)
```

### 🎯 Success Metrics

```
Monthly Goals:
- Win Rate: >50%
- Profit Factor: >1.5
- Max Drawdown: <10%
- R:R Average: >1:1.5

Track:
- Total Trades
- Winning Trades
- Losing Trades
- Average Win
- Average Loss
- Largest Win
- Largest Loss
- Net P&L
```

---

## 📱 Quick Access Card (Print & Keep)

```
┌─────────────────────────────────────────┐
│   ULTIMATE TRADING APP - QUICK CARD     │
├─────────────────────────────────────────┤
│ Access Token: Expires in 24 hours       │
│ Alert Distance: 5 points default        │
│ Cooling Period: 10 minutes              │
│ Auto Refresh: Every 1 minute            │
├─────────────────────────────────────────┤
│ ALERTS:                                 │
│ 🟢 Bullish VOB    🔴 Bearish VOB        │
│ 🔵 HTF Resistance 🟡 HTF Support        │
├─────────────────────────────────────────┤
│ RSI ZONES:                              │
│ >80 Overbought   <20 Oversold           │
│ 40-60 Neutral                           │
├─────────────────────────────────────────┤
│ RISK RULES:                             │
│ Max Risk/Trade: 2%                      │
│ Always Use Stop Loss                    │
│ Min R:R: 1:1.5                          │
├─────────────────────────────────────────┤
│ EMERGENCY:                              │
│ Dhan Support: support@dhan.co           │
│ Manual Trade: web.dhan.co               │
└─────────────────────────────────────────┘
```

---

**💾 Save this card for quick reference!**

*Print and keep near your trading desk* 📌
