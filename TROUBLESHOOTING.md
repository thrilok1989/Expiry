# 🔧 Troubleshooting Guide

## Quick Diagnostic Checklist

Before diving into specific issues, run through this checklist:

```
□ Python 3.8+ installed
□ All requirements.txt packages installed
□ DhanHQ Access Token is valid (< 24 hours old)
□ Client ID is correct
□ Security ID exists and is correct
□ Internet connection is stable
□ Market hours (9:15 AM - 3:30 PM IST) for live data
□ Telegram bot is active (if using alerts)
```

---

## Common Issues and Solutions

### 1. Installation Issues

#### Error: `pip: command not found`
**Cause**: Python/pip not installed or not in PATH

**Solution**:
```bash
# macOS/Linux
python3 -m pip install -r requirements.txt

# Windows
python -m pip install -r requirements.txt

# If still failing, install pip first:
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

#### Error: `ModuleNotFoundError: No module named 'streamlit'`
**Cause**: Package not installed

**Solution**:
```bash
pip install --upgrade pip
pip install -r requirements.txt

# If specific package failing:
pip install streamlit pandas numpy plotly requests yfinance
```

#### Error: `Permission denied`
**Cause**: Insufficient permissions

**Solution**:
```bash
# Use --user flag
pip install --user -r requirements.txt

# Or use sudo (Linux/Mac)
sudo pip install -r requirements.txt
```

---

### 2. API Connection Issues

#### Error: "Failed to fetch data" / "Error: 401 Unauthorized"
**Cause**: Invalid or expired Access Token

**Solution**:
1. Go to [web.dhan.co](https://web.dhan.co)
2. Navigate to My Profile → Access DhanHQ APIs
3. Generate NEW Access Token
4. Copy and paste into app
5. Token is valid for 24 hours only

**Verification**:
```python
# Test your credentials
import requests

headers = {
    "access-token": "YOUR_TOKEN",
    "client-id": "YOUR_CLIENT_ID"
}

response = requests.get(
    "https://api.dhan.co/v2/profile",
    headers=headers
)

print(response.json())  # Should show your profile
```

#### Error: "Error: 404 Not Found"
**Cause**: Incorrect API endpoint or Security ID

**Solution**:
1. Verify Security ID from: https://images.dhan.co/api-data/api-scrip-master.csv
2. Common Security IDs:
   - NIFTY Index: `13`
   - BANKNIFTY Index: `25`
   - Reliance: `1333`
   - TCS: `11536`

#### Error: "Error: 429 Too Many Requests"
**Cause**: Rate limit exceeded

**Solution**:
```
Rate Limits:
- 25 requests per second
- 250 requests per minute
- 7,000 requests per day

Actions:
1. Slow down auto-refresh rate
2. Wait a few minutes
3. Check for loops making repeated calls
4. Implement request throttling
```

#### Error: "Network timeout"
**Cause**: Slow/unstable internet connection

**Solution**:
```python
# Increase timeout in code
response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=30  # Increase from 10 to 30 seconds
)
```

---

### 3. Data Issues

#### Problem: "No data displaying" / "Empty DataFrame"
**Cause**: Market closed, incorrect Security ID, or no data available

**Solution**:
1. Check if market is open (9:15 AM - 3:30 PM IST, Mon-Fri)
2. Verify Security ID is correct
3. Try different date range
4. Check exchange segment matches security type

**Outside Market Hours**:
```python
# Use daily data instead of intraday
# Or use yfinance for historical data
import yfinance as yf

symbol = "RELIANCE.NS"  # Add .NS for NSE, .BO for BSE
df = yf.download(symbol, period="5d", interval="1m")
```

#### Problem: "Data looks incorrect" / "Strange values"
**Cause**: Data formatting issue or corrupt response

**Solution**:
```python
# Add data validation
df = df[df['close'] > 0]  # Remove zero/negative values
df = df[df['volume'] >= 0]  # Remove negative volumes
df = df.dropna()  # Remove NaN values
df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
```

#### Problem: "Old data showing"
**Cause**: Caching or stale session state

**Solution**:
1. Click "🔄 Refresh Now" button
2. Clear browser cache (Ctrl+Shift+R or Cmd+Shift+R)
3. Restart Streamlit app
4. Check auto-refresh is enabled

---

### 4. Indicator Issues

#### Problem: "No Volume Order Blocks showing"
**Cause**: Not enough data or no EMA crossovers detected

**Debug Steps**:
```python
# Check data length
print(f"Data points: {len(df)}")  # Should be > 100

# Check for crossovers
df['ema1'] = df['close'].ewm(span=5).mean()
df['ema2'] = df['close'].ewm(span=18).mean()
crossovers = ((df['ema1'] > df['ema2']) & (df['ema1'].shift(1) <= df['ema2'].shift(1))).sum()
print(f"Crossovers found: {crossovers}")
```

**Solutions**:
1. Increase days_back parameter (try 10 instead of 5)
2. Try different timeframe (5T instead of 1T)
3. Check if instrument has sufficient volume
4. Reduce sensitivity (increase length1 parameter)

#### Problem: "HTF levels not appearing"
**Cause**: Insufficient data for higher timeframe analysis

**Solution**:
1. Increase data history (days_back=10)
2. Reduce pivot_length parameter (try 3 instead of 5)
3. Check timeframe settings (10T, 15T should work)
4. Verify data has enough bars after resampling

#### Problem: "RSI always at extremes" / "RSI values look wrong"
**Cause**: Calculation error or data issue

**Debug**:
```python
# Verify RSI calculation
rsi = rsi_calculator.calculate(df)
print(f"RSI min: {rsi['arsi'].min()}")  # Should be > 0
print(f"RSI max: {rsi['arsi'].max()}")  # Should be < 100
print(f"RSI mean: {rsi['arsi'].mean()}")  # Should be around 50
```

**Solution**:
1. Check for data gaps or NaN values
2. Ensure sufficient data points (need > 50)
3. Verify close prices are valid
4. Try different RSI length parameter

---

### 5. Chart Display Issues

#### Problem: "Chart not rendering" / "Blank chart"
**Cause**: Plotly issue or no data to plot

**Solution**:
```bash
# Update Plotly
pip install --upgrade plotly

# If still failing, try:
pip uninstall plotly
pip install plotly==5.18.0

# Clear cache
streamlit cache clear
```

#### Problem: "Chart is too slow" / "Browser freezing"
**Cause**: Too many data points

**Solution**:
```python
# Downsample for display
if len(df) > 5000:
    # Keep every Nth row
    n = len(df) // 5000
    df_display = df.iloc[::n]
else:
    df_display = df

# Use in chart creation
fig = create_chart(df_display, ...)
```

#### Problem: "Colors not showing" / "Theme issues"
**Cause**: Browser dark mode or theme conflict

**Solution**:
1. Try different browser
2. Disable browser dark mode extensions
3. Add to code:
```python
fig.update_layout(template='plotly_dark')  # Force dark theme
```

---

### 6. Telegram Alert Issues

#### Problem: "Telegram alerts not sending"
**Cause**: Invalid bot token, chat ID, or bot not started

**Diagnostic**:
```python
# Test Telegram connection
import requests

bot_token = "YOUR_BOT_TOKEN"
chat_id = "YOUR_CHAT_ID"

url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
payload = {
    "chat_id": chat_id,
    "text": "Test message from Trading App"
}

response = requests.post(url, json=payload)
print(response.json())
```

**Common Errors & Solutions**:

**Error: "Unauthorized"**
- Bot token is incorrect
- Regenerate token from @BotFather

**Error: "Bad Request: chat not found"**
- Chat ID is incorrect
- Send a message to bot first
- Get chat ID from @userinfobot

**Error: "Forbidden: bot was blocked by the user"**
- You blocked the bot
- Search bot in Telegram and "Unblock"

#### Problem: "Too many alert messages"
**Cause**: Cooling period not working

**Solution**:
```python
# Verify cooling period is being respected
print(f"Last notification time: {notifier.last_notification}")
print(f"Current time: {time.time()}")
print(f"Cooling period: {notifier.cooling_period}s")

# Increase cooling period if needed
notifier = TelegramNotifier(token, chat_id, cooling_period=1800)  # 30 minutes
```

#### Problem: "No alerts even when price near levels"
**Cause**: Alert distance too small or conditions not met

**Debug**:
```python
# Check distances
for block in vob_data['bullish']:
    distance = abs(current_price - block['mid'])
    print(f"Bullish VOB at {block['mid']}: {distance} points away")
    
# Increase alert distance
alert_distance = 10.0  # Try larger value
```

---

### 7. Auto-Refresh Issues

#### Problem: "Auto-refresh not working"
**Cause**: Browser settings or Streamlit issue

**Solution**:
1. Check "Enable Auto Refresh" is checked
2. Verify session state is working:
```python
print(f"Last refresh: {st.session_state.last_refresh}")
print(f"Current time: {datetime.now()}")
```
3. Use manual refresh button as backup
4. Try different browser

#### Problem: "Refresh too fast/slow"
**Cause**: Timing issue

**Solution**:
```python
# Adjust refresh interval in code
refresh_interval = 120  # 2 minutes instead of 1

if auto_refresh:
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    if time_since_refresh >= refresh_interval:
        st.session_state.last_refresh = datetime.now()
        st.rerun()
```

---

### 8. Performance Issues

#### Problem: "App is slow" / "High CPU usage"
**Cause**: Heavy calculations or large datasets

**Optimizations**:

```python
# 1. Cache data
@st.cache_data(ttl=60)  # Cache for 60 seconds
def fetch_and_process_data(security_id):
    df = dhan.get_intraday_data(security_id)
    return df

# 2. Reduce data size
df = df.iloc[-1000:]  # Keep only last 1000 rows

# 3. Optimize indicator calculations
# Use vectorized operations, avoid Python loops

# 4. Limit chart points
if len(df) > 2000:
    df_chart = df.iloc[::2]  # Every other point
else:
    df_chart = df
```

#### Problem: "Memory usage increasing"
**Cause**: Memory leak in session state

**Solution**:
```python
# Clear old data from session state
if 'old_data' in st.session_state:
    del st.session_state.old_data

# Limit stored data
if 'data_history' in st.session_state:
    if len(st.session_state.data_history) > 100:
        st.session_state.data_history = st.session_state.data_history[-50:]
```

---

### 9. Error Messages Decoder

#### "DH-901: Authentication Failed"
```
Issue: Invalid credentials
Fix: Regenerate Access Token from Dhan web
```

#### "DH-904: Rate Limit Exceeded"
```
Issue: Too many API requests
Fix: Slow down, wait 1 minute, then resume
```

#### "DH-905: Input Exception"
```
Issue: Invalid parameters (Security ID, dates, etc.)
Fix: Verify all inputs match expected format
```

#### "DH-907: Data Error"
```
Issue: No data available for requested parameters
Fix: Check Security ID, date range, market hours
```

#### "ConnectionError: Network unreachable"
```
Issue: Internet connection problem
Fix: Check internet, try mobile hotspot
```

---

## Debug Mode

Add this to enable detailed logging:

```python
import logging

# Add at the top of trading_app.py
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='trading_app.log'
)

logger = logging.getLogger(__name__)

# Use throughout code:
logger.debug(f"Fetching data for {security_id}")
logger.info(f"Found {len(vob_data['bullish'])} bullish VOBs")
logger.warning(f"Alert distance very large: {alert_distance}")
logger.error(f"Failed to fetch data: {e}")
```

Then check `trading_app.log` file for detailed information.

---

## Getting Help

### Self-Help Resources
1. Re-read README.md
2. Check EXAMPLES.md for similar use case
3. Review ARCHITECTURE.md to understand data flow
4. Enable debug logging

### Community Help
1. DhanHQ Documentation: https://dhanhq.co/docs
2. DhanHQ Support: https://support.dhanhq.co
3. Streamlit Forum: https://discuss.streamlit.io

### Reporting Issues
When reporting an issue, include:

```
1. Python version (python --version)
2. Operating System
3. Error message (exact text)
4. Steps to reproduce
5. Expected vs actual behavior
6. trading_app.log file (if debug enabled)
7. Screenshot of error
```

---

## Prevention Checklist

To avoid issues:

```
✓ Keep Access Token fresh (regenerate daily)
✓ Backup your configuration
✓ Test with small position sizes first
✓ Monitor API rate limits
✓ Keep internet connection stable
✓ Update packages regularly
✓ Clear cache periodically
✓ Restart app daily
✓ Monitor Telegram bot status
✓ Verify data before trading
```

---

## Emergency Procedures

### If App Crashes During Trading
```
1. Don't panic
2. Login to Dhan web/mobile
3. Check open positions
4. Close positions manually if needed
5. Investigate crash cause
6. Fix before resuming
```

### If Alerts Stop Working
```
1. Check Telegram bot is active
2. Send test message manually
3. Verify cooling period hasn't blocked alerts
4. Check internet connection
5. Restart app if needed
```

### If Data Looks Wrong
```
1. Stop trading immediately
2. Verify on Dhan web/mobile
3. Cross-check with other sources
4. Don't trust bad data
5. Fix data issue first
6. Resume only when verified
```

---

**Remember**: When in doubt, don't trade. It's better to miss an opportunity than to trade on faulty data or alerts! 🛡️
