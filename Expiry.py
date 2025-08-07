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
    # OI + Price Based Bias Logic
    if row['changeinOpenInterest_CE'] > 0 and row['lastPrice_CE'] > row['previousClose_CE']:
        score += 1  # New CE longs → Bullish
    if row['changeinOpenInterest_PE'] > 0 and row['lastPrice_PE'] > row['previousClose_PE']:
        score -= 1  # New PE longs → Bearish
    if row['changeinOpenInterest_CE'] > 0 and row['lastPrice_CE'] < row['previousClose_CE']:
        score -= 1  # CE writing → Bearish
    if row['changeinOpenInterest_PE'] > 0 and row['lastPrice_PE'] < row['previousClose_PE']:
        score += 1  # PE writing → Bullish

    # Bid Volume Dominance
    if 'bidQty_CE' in row and 'bidQty_PE' in row:
        if row['bidQty_CE'] > row['bidQty_PE'] * 1.5:
            score += 1  # CE Bid dominance → Bullish
        if row['bidQty_PE'] > row['bidQty_CE'] * 1.5:
            score -= 1  # PE Bid dominance → Bearish

    # Volume Churn vs OI
    if row['totalTradedVolume_CE'] > 2 * row['openInterest_CE']:
        score -= 0.5  # CE churn → Possibly noise
    if row['totalTradedVolume_PE'] > 2 * row['openInterest_PE']:
        score += 0.5  # PE churn → Possibly noise

    # Bid-Ask Pressure
    if 'underlyingValue' in row:
        if abs(row['lastPrice_CE'] - row['underlyingValue']) < abs(row['lastPrice_PE'] - row['underlyingValue']):
            score += 0.5  # CE closer to spot → Bullish
        else:
            score -= 0.5  # PE closer to spot → Bearish
    return score

def expiry_entry_signal(df, atm_strike, score_threshold=1.5):
    entries = []
    # Filter only ATM ±5 strikes
    df = df[df['strikePrice'].between(atm_strike - 5*50, atm_strike + 5*50)]
    
    for _, row in df.iterrows():
        strike = row['strikePrice']
        score = expiry_bias_score(row)
        level = row['Level']

        if score >= score_threshold and level == "Support":
            entries.append({
                'type': 'BUY CALL',
                'strike': strike,
                'score': score,
                'ltp': row['lastPrice_CE'],
                'reason': f'Bullish score (ATM±5) at support: {strike}'
            })

        if score <= -score_threshold and level == "Resistance":
            entries.append({
                'type': 'BUY PUT',
                'strike': strike,
                'score': score,
                'ltp': row['lastPrice_PE'],
                'reason': f'Bearish score (ATM±5) at resistance: {strike}'
            })
    return entries

def analyze_expiry():
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        session.headers.update(headers)
        session.get("https://www.nseindia.com", timeout=5)
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        response = session.get(url, timeout=10)
        data = response.json()

        records = data['records']['data']
        expiry = data['records']['expiryDates'][0]
        underlying = data['records']['underlyingValue']
        atm_strike = min(data['records']['strikePrices'], key=lambda x: abs(x - underlying))
        
        # Check if today is expiry day
        today = datetime.now(timezone("Asia/Kolkata"))
        expiry_date = timezone("Asia/Kolkata").localize(datetime.strptime(expiry, "%d-%b-%Y"))
        is_expiry_day = today.date() == expiry_date.date()
        
        if not is_expiry_day:
            st.warning("❌ Today is NOT an expiry day. This script only works on expiry days.")
            return

        st.info(f"""
📅 **EXPIRY DAY DETECTED**
- Analyzing ATM ±5 strikes only
- Current ATM: {atm_strike}
- Spot Price: {underlying}
""")
        send_telegram_message(f"⚠️ Expiry Day Detected. Analyzing ATM {atm_strike}±5 strikes")

        # Get previous close data
        prev_close_url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
        prev_close_data = session.get(prev_close_url, timeout=10).json()
        prev_close = prev_close_data['data'][0]['previousClose']

        # Process records with expiry day logic
        calls, puts = [], []
        for item in records:
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

        df_ce = pd.DataFrame(calls)
        df_pe = pd.DataFrame(puts)
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

        # Get support/resistance levels
        df['Level'] = df.apply(determine_level, axis=1)
        
        # Generate expiry day signals for ATM ±5 strikes only
        expiry_signals = expiry_entry_signal(df, atm_strike)

        # Display ATM ±5 strikes data
        st.markdown(f"### 🎯 ATM {atm_strike} ±5 Strikes Analysis")
        atm_df = df[df['strikePrice'].between(atm_strike - 5*50, atm_strike + 5*50)]
        atm_df['ExpiryBiasScore'] = atm_df.apply(expiry_bias_score, axis=1)
        st.dataframe(atm_df[['strikePrice', 'ExpiryBiasScore', 'Level', 
                           'lastPrice_CE', 'lastPrice_PE',
                           'changeinOpenInterest_CE', 'changeinOpenInterest_PE']])
        
        # Display signals
        st.markdown("### 🔔 Trading Signals (ATM±5)")
        if expiry_signals:
            for signal in expiry_signals:
                st.success(f"""
                **{signal['type']}** at {signal['strike']} 
                - Score: {signal['score']:.1f} 
                - LTP: ₹{signal['ltp']}
                - Reason: {signal['reason']}
                """)
                
                # Send Telegram alert
                send_telegram_message(
                    f"📅 EXPIRY DAY SIGNAL (ATM±5)\n"
                    f"Type: {signal['type']}\n"
                    f"Strike: {signal['strike']}\n"
                    f"Score: {signal['score']:.1f}\n"
                    f"LTP: ₹{signal['ltp']}\n"
                    f"Reason: {signal['reason']}\n"
                    f"Spot: {underlying}"
                )
        else:
            st.warning("No strong signals detected in ATM±5 strikes")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        send_telegram_message(f"❌ Expiry Day Analysis Error: {str(e)}")

# === Main Function Call ===
if __name__ == "__main__":
    st.set_page_config(page_title="Nifty Expiry ATM±5 Analyzer", layout="wide")
    analyze_expiry()