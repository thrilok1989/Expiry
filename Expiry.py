import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from pytz import timezone

# === Telegram Config ===
TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU"
TELEGRAM_CHAT_ID = "5704496584"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("⚠️ Telegram message failed.")
    except Exception as e:
        st.error(f"❌ Telegram error: {e}")

def get_nse_option_chain():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }
    
    session = requests.Session()
    session.headers.update(headers)
    
    try:
        # First request to get cookies
        session.get("https://www.nseindia.com", timeout=10)
        # Second request to get option chain data
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        # Verify response contains valid JSON
        if not response.text.strip():
            raise ValueError("Empty response from NSE API")
            
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch NSE data: {str(e)}")
        send_telegram_message(f"❌ NSE Data Fetch Error: {str(e)}")
        return None

def determine_level(row):
    ce_oi = row['openInterest_CE']
    pe_oi = row['openInterest_PE']
    if pe_oi > 1.12 * ce_oi:
        return "Support"
    elif ce_oi > 1.12 * pe_oi:
        return "Resistance"
    else:
        return "Neutral"

def expiry_bias_score(row):
    score = 0
    if row['changeinOpenInterest_CE'] > 0 and row['lastPrice_CE'] > row['previousClose_CE']:
        score += 1
    if row['changeinOpenInterest_PE'] > 0 and row['lastPrice_PE'] > row['previousClose_PE']:
        score -= 1
    if row['changeinOpenInterest_CE'] > 0 and row['lastPrice_CE'] < row['previousClose_CE']:
        score -= 1
    if row['changeinOpenInterest_PE'] > 0 and row['lastPrice_PE'] < row['previousClose_PE']:
        score += 1

    if 'bidQty_CE' in row and 'bidQty_PE' in row:
        if row['bidQty_CE'] > row['bidQty_PE'] * 1.5:
            score += 1
        if row['bidQty_PE'] > row['bidQty_CE'] * 1.5:
            score -= 1

    if row['totalTradedVolume_CE'] > 2 * row['openInterest_CE']:
        score -= 0.5
    if row['totalTradedVolume_PE'] > 2 * row['openInterest_PE']:
        score += 0.5

    if 'underlyingValue' in row:
        if abs(row['lastPrice_CE'] - row['underlyingValue']) < abs(row['lastPrice_PE'] - row['underlyingValue']):
            score += 0.5
        else:
            score -= 0.5
    return score

def analyze_expiry():
    try:
        # Get current time and check market hours
        now = datetime.now(timezone("Asia/Kolkata"))
        current_time = now.time()
        market_open = timezone("Asia/Kolkata").localize(datetime.strptime("09:15", "%H:%M")).time()
        market_close = timezone("Asia/Kolkata").localize(datetime.strptime("15:30", "%H:%M")).time()
        
        if not (market_open <= current_time <= market_close):
            st.warning("❌ Market is closed now. Analysis available only between 9:15 AM to 3:30 PM IST.")
            return

        # Fetch option chain data
        data = get_nse_option_chain()
        if data is None:
            return

        records = data['records']['data']
        expiry = data['records']['expiryDates'][0]
        underlying = data['records']['underlyingValue']
        
        # Find ATM strike (nearest strike to underlying value)
        strike_prices = sorted(list(set(item['strikePrice'] for item in records if 'strikePrice' in item)))
        atm_strike = min(strike_prices, key=lambda x: abs(x - underlying))
        
        # Check if today is expiry day
        expiry_date = timezone("Asia/Kolkata").localize(datetime.strptime(expiry, "%d-%b-%Y"))
        is_expiry_day = now.date() == expiry_date.date()
        
        if not is_expiry_day:
            st.warning("❌ Today is NOT an expiry day. This script only works on expiry days.")
            return

        st.info(f"""
📅 **EXPIRY DAY ANALYSIS - ATM ±5 STRIKES**
- Current ATM Strike: {atm_strike}
- Spot Price: {underlying}
- Analysis Range: {atm_strike - 250} to {atm_strike + 250}
""")

        # Get previous close data
        prev_close_url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
        session = requests.Session()
        prev_close_data = session.get(prev_close_url, timeout=10).json()
        prev_close = prev_close_data['data'][0]['previousClose']

        # Process records for ATM ±5 strikes only
        calls, puts = [], []
        for item in records:
            if 'strikePrice' not in item:
                continue
                
            if abs(item['strikePrice'] - atm_strike) > 250:  # ±5 strikes (50×5)
                continue
                
            if 'CE' in item and item['CE']['expiryDate'] == expiry:
                ce = item['CE']
                ce['previousClose_CE'] = prev_close
                ce['underlyingValue'] = underlying
                calls.append(ce)
                
            if 'PE' in item and item['PE']['expiryDate'] == expiry:
                pe = item['PE']
                pe['previousClose_PE'] = prev_close
                pe['underlyingValue'] = underlying
                puts.append(pe)

        # Create merged dataframe
        df_ce = pd.DataFrame(calls)
        df_pe = pd.DataFrame(puts)
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
        df['Level'] = df.apply(determine_level, axis=1)
        df['ExpiryBiasScore'] = df.apply(expiry_bias_score, axis=1)

        # Display ATM ±5 strikes
        st.markdown("### 📊 ATM ±5 Strikes Option Chain")
        display_cols = ['strikePrice', 'ExpiryBiasScore', 'Level', 
                       'lastPrice_CE', 'changeinOpenInterest_CE', 'openInterest_CE',
                       'lastPrice_PE', 'changeinOpenInterest_PE', 'openInterest_PE']
        st.dataframe(df[display_cols], height=500)

        # Generate signals
        bullish_signals = df[(df['ExpiryBiasScore'] >= 1.5) & (df['Level'] == "Support")]
        bearish_signals = df[(df['ExpiryBiasScore'] <= -1.5) & (df['Level'] == "Resistance")]

        # Display signals
        st.markdown("### 🔔 Trading Signals")
        if not bullish_signals.empty:
            st.success("**Bullish Signals (BUY CALL)**")
            for _, row in bullish_signals.iterrows():
                st.write(f"""
                - Strike: {row['strikePrice']} CE
                - Score: {row['ExpiryBiasScore']:.1f}
                - LTP: ₹{row['lastPrice_CE']}
                - OI Change: {row['changeinOpenInterest_CE']:+,}
                - Reason: Strong bullish bias at support level
                """)
                send_telegram_message(
                    f"📈 Bullish Signal (ATM±5)\n"
                    f"BUY {row['strikePrice']} CE\n"
                    f"Score: {row['ExpiryBiasScore']:.1f}\n"
                    f"LTP: ₹{row['lastPrice_CE']}\n"
                    f"OI Change: {row['changeinOpenInterest_CE']:+,}\n"
                    f"Spot: {underlying}"
                )

        if not bearish_signals.empty:
            st.error("**Bearish Signals (BUY PUT)**")
            for _, row in bearish_signals.iterrows():
                st.write(f"""
                - Strike: {row['strikePrice']} PE
                - Score: {row['ExpiryBiasScore']:.1f}
                - LTP: ₹{row['lastPrice_PE']}
                - OI Change: {row['changeinOpenInterest_PE']:+,}
                - Reason: Strong bearish bias at resistance level
                """)
                send_telegram_message(
                    f"📉 Bearish Signal (ATM±5)\n"
                    f"BUY {row['strikePrice']} PE\n"
                    f"Score: {row['ExpiryBiasScore']:.1f}\n"
                    f"LTP: ₹{row['lastPrice_PE']}\n"
                    f"OI Change: {row['changeinOpenInterest_PE']:+,}\n"
                    f"Spot: {underlying}"
                )

        if bullish_signals.empty and bearish_signals.empty:
            st.warning("No strong signals detected in ATM±5 strikes")

    except Exception as e:
        st.error(f"❌ Analysis Error: {str(e)}")
        send_telegram_message(f"❌ Analysis Error: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(page_title="Nifty Expiry ATM±5 Analyzer", layout="wide")
    st.title("NIFTY Expiry Day Analysis - ATM ±5 Strikes")
    analyze_expiry()