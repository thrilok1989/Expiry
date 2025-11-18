"""
NIFTY/SENSEX Trading App - Single File Version
Upload ONLY this file to GitHub and deploy
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="NIFTY Trading", page_icon="📈", layout="wide")

# Title
st.title("📈 NIFTY/SENSEX Trading App")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Get secrets
    try:
        access_token = st.secrets["dhan"]["access_token"]
        client_id = st.secrets["dhan"]["client_id"]
        st.success("✅ Connected")
    except:
        st.error("❌ Add secrets!")
        st.stop()
    
    # Quick select
    index = st.selectbox("Select Index", ["NIFTY 50", "SENSEX", "BANK NIFTY"])
    
    if index == "NIFTY 50":
        security_id = "13"
    elif index == "SENSEX":
        security_id = "51"
    else:
        security_id = "25"
    
    if st.button("🔄 Refresh"):
        st.rerun()

# Fetch data
@st.cache_data(ttl=60)
def get_data(sec_id, token, client):
    try:
        headers = {
            "access-token": token,
            "client-id": client,
            "Content-Type": "application/json"
        }
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=5)
        
        payload = {
            "securityId": sec_id,
            "exchangeSegment": "IDX_I",
            "instrument": "INDEX",
            "interval": "1",
            "fromDate": from_date.strftime("%Y-%m-%d 09:15:00"),
            "toDate": to_date.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        response = requests.post(
            "https://api.dhan.co/v2/charts/intraday",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('timestamp'):
                df = pd.DataFrame({
                    'time': pd.to_datetime(data['timestamp'], unit='s'),
                    'open': data['open'],
                    'high': data['high'],
                    'low': data['low'],
                    'close': data['close'],
                    'volume': data['volume']
                })
                return df
        return None
    except:
        return None

# Get LTP
def get_ltp(sec_id, token, client):
    try:
        headers = {
            "access-token": token,
            "client-id": client,
            "Content-Type": "application/json"
        }
        
        payload = {"IDX_I": [int(sec_id)]}
        
        response = requests.post(
            "https://api.dhan.co/v2/marketfeed/ltp",
            headers=headers,
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['data']['IDX_I'][sec_id]['last_price']
    except:
        pass
    return None

# Main
with st.spinner("Loading data..."):
    df = get_data(security_id, access_token, client_id)
    ltp = get_ltp(security_id, access_token, client_id)

if df is not None and len(df) > 0:
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    if ltp:
        change = ltp - df['close'].iloc[-2]
        change_pct = (change / df['close'].iloc[-2]) * 100
        col1.metric("Current Price", f"₹{ltp:.2f}", f"{change_pct:.2f}%")
    else:
        col1.metric("Last Close", f"₹{df['close'].iloc[-1]:.2f}")
    
    col2.metric("Volume", f"{df['volume'].iloc[-1]:,.0f}")
    col3.metric("Data Points", len(df))
    
    # Simple chart
    st.subheader(f"{index} - Price Chart")
    st.line_chart(df.set_index('time')['close'])
    
    # Data table
    with st.expander("📊 View Data"):
        st.dataframe(df.tail(20))
    
else:
    st.error("❌ No data available. Check:")
    st.write("1. Access token is valid (regenerate at web.dhan.co)")
    st.write("2. Market is open (9:15 AM - 3:30 PM IST)")
    st.write("3. Security ID is correct")

# Footer
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
