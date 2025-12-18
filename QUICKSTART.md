# 🚀 Quick Start Guide

## Installation (5 minutes)

### 1. Install Python packages
```bash
pip install -r requirements.txt
```

### 2. Get your DhanHQ credentials
- Go to [web.dhan.co](https://web.dhan.co)
- Login → My Profile → Access DhanHQ APIs
- Generate Access Token (valid 24 hours)
- Copy your Client ID

### 3. Setup Telegram (Optional)
- Message [@BotFather](https://t.me/BotFather) on Telegram
- Send `/newbot` and follow instructions
- Get Chat ID from [@userinfobot](https://t.me/userinfobot)

### 4. Run the application
```bash
streamlit run trading_app.py
```

## ⚡ First Time Setup

When the app opens:

1. **Left Sidebar → API Settings**
   - Paste your Access Token
   - Paste your Client ID

2. **Telegram Settings** (Optional)
   - Paste Bot Token
   - Paste Chat ID

3. **Trading Settings**
   - Security ID: `1333` (Reliance) or `13` (NIFTY Index)
   - Symbol: Name to display
   - Exchange: Select NSE_EQ

4. **Enable Auto Refresh**
   - Check the box for automatic updates

5. Click **"🔄 Refresh Now"**

## 📊 What You'll See

### Main Chart
- Candlestick price chart
- Green/Purple shaded areas = Volume Order Blocks
- Horizontal lines = Support/Resistance levels
- Orange line = VIDYA trend indicator

### Bottom Panel
- Ultimate RSI oscillator
- Overbought (>80) and Oversold (<20) zones

### Metrics (Top)
- Current Price with % change
- Volume
- Ultimate RSI value
- Number of Bullish/Bearish VOBs

### Details (Below Chart)
- List of all Volume Order Blocks with distances
- HTF Support/Resistance levels
- Current distances from your price

## 🔔 Telegram Alerts

You'll receive messages when:
- Price is within 5 points of any Volume Order Block
- Price is within 5 points of HTF Support/Resistance
- 10-minute cooling period between same alerts

## 💡 Pro Tips

1. **For Intraday Trading**:
   - Use 1-minute interval
   - Watch VOB zones for entries
   - HTF levels for targets/stops

2. **For Swing Trading**:
   - Use 5-minute or 15-minute interval
   - Focus on HTF levels
   - Look for VIDYA trend confirmation

3. **Best Time to Use**:
   - Market hours: 9:15 AM - 3:30 PM IST
   - Most accurate during high volume periods

4. **Alert Management**:
   - Keep Telegram open for instant notifications
   - Adjust alert distance based on volatility
   - 5 points works well for NIFTY/stocks

## 🔧 Common Issues

**"Failed to fetch data"**
→ Check Access Token validity (regenerate daily)

**"Telegram not working"**
→ Send a message to your bot first

**"No data showing"**
→ Verify Security ID is correct

**"Auto-refresh not working"**
→ Use manual refresh button

## 📱 Mobile Usage

While the app works on mobile browsers, for best experience:
- Use desktop/laptop for full features
- Telegram alerts work perfectly on mobile
- You can monitor alerts while away from computer

## 🎯 Next Steps

1. Test with paper trading first
2. Adjust indicator parameters based on your style
3. Set up multiple instruments to track
4. Combine with your existing strategy

## ⚠️ Remember

- This is a tool, not a trading system
- Always use stop losses
- Risk only what you can afford to lose
- Past performance ≠ Future results

---

**Need Help?**
- Check README.md for detailed documentation
- DhanHQ Docs: https://dhanhq.co/docs
- Re-read this guide

**Ready to trade? 🚀**
