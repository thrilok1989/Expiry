#!/usr/bin/env python3
"""
Complete Streamlit App for DhanHQ Options SuperTrend Analysis with Live WebSocket Data
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
import numpy as np
import threading
from collections import deque
import websocket
import struct
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebSocket Handler Class
class DhanWebSocketClient:
    def __init__(self, access_token, client_id):
        self.access_token = access_token
        self.client_id = client_id
        self.websocket = None
        self.is_connected = False
        self.subscribed_instruments = []
        self.price_callbacks = []
        self.connection_callbacks = []
        self.message_count = 0
        
    def add_price_callback(self, callback):
        """Add callback function for price updates"""
        self.price_callbacks.append(callback)
    
    def add_connection_callback(self, callback):
        """Add callback function for connection status updates"""
        self.connection_callbacks.append(callback)
    
    def connect(self):
        """Establish WebSocket connection"""
        try:
            ws_url = f"wss://api-feed.dhan.co?version=2&token={self.access_token}&clientId={self.client_id}&authType=2"
            
            # Create WebSocket connection
            self.websocket = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start connection in separate thread
            self.ws_thread = threading.Thread(target=self.websocket.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            logger.info("WebSocket connection initiated...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close WebSocket connection"""
        if self.websocket and self.is_connected:
            disconnect_msg = {"RequestCode": 12}
            self.websocket.send(json.dumps(disconnect_msg))
            self.websocket.close()
            self.is_connected = False
            logger.info("WebSocket disconnected")
    
    def subscribe_instrument(self, exchange_segment, security_id):
        """Subscribe to instrument for live data"""
        if not self.is_connected:
            logger.error("WebSocket not connected")
            return False
        
        try:
            subscribe_message = {
                "RequestCode": 15,  # Subscribe to quote data
                "InstrumentCount": 1,
                "InstrumentList": [
                    {
                        "ExchangeSegment": exchange_segment,
                        "SecurityId": str(security_id)
                    }
                ]
            }
            
            self.websocket.send(json.dumps(subscribe_message))
            self.subscribed_instruments.append({
                "exchange": exchange_segment,
                "security_id": security_id
            })
            
            logger.info(f"Subscribed to {exchange_segment}:{security_id}")
            return True
            
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            return False
    
    def _on_open(self, ws):
        """WebSocket connection opened"""
        self.is_connected = True
        logger.info("WebSocket connected successfully")
        
        # Notify connection callbacks
        for callback in self.connection_callbacks:
            try:
                callback("connected", "WebSocket connected")
            except Exception as e:
                logger.error(f"Connection callback error: {e}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed"""
        self.is_connected = False
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        # Notify connection callbacks
        for callback in self.connection_callbacks:
            try:
                callback("disconnected", f"Connection closed: {close_msg}")
            except Exception as e:
                logger.error(f"Connection callback error: {e}")
    
    def _on_error(self, ws, error):
        """WebSocket error occurred"""
        logger.error(f"WebSocket error: {error}")
        
        # Notify connection callbacks
        for callback in self.connection_callbacks:
            try:
                callback("error", str(error))
            except Exception as e:
                logger.error(f"Connection callback error: {e}")
    
    def _on_message(self, ws, message):
        """Process incoming WebSocket message"""
        try:
            if isinstance(message, bytes):
                self._process_binary_message(message)
            else:
                self._process_text_message(message)
            
            self.message_count += 1
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    def _process_text_message(self, message):
        """Process text message (JSON)"""
        try:
            data = json.loads(message)
            logger.info(f"Text message received: {data}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
    
    def _process_binary_message(self, message):
        """Process binary message according to DhanHQ protocol"""
        try:
            if len(message) < 8:
                logger.warning("Message too short for header")
                return
            
            # Parse response header (8 bytes)
            feed_response_code = struct.unpack('B', message[0:1])[0]
            message_length = struct.unpack('<H', message[1:3])[0]  # Little endian
            exchange_segment = struct.unpack('B', message[3:4])[0]
            security_id = struct.unpack('<I', message[4:8])[0]  # Little endian
            
            logger.debug(f"Header - Code: {feed_response_code}, Length: {message_length}, "
                        f"Segment: {exchange_segment}, SecurityID: {security_id}")
            
            # Process different packet types
            if feed_response_code == 2:  # Ticker packet
                self._process_ticker_packet(message, security_id)
            elif feed_response_code == 4:  # Quote packet
                self._process_quote_packet(message, security_id)
            elif feed_response_code == 6:  # Previous close
                self._process_prev_close_packet(message, security_id)
            elif feed_response_code == 50:  # Disconnect packet
                self._process_disconnect_packet(message)
            else:
                logger.debug(f"Unknown packet type: {feed_response_code}")
                
        except struct.error as e:
            logger.error(f"Binary parsing error: {e}")
        except Exception as e:
            logger.error(f"Binary message processing error: {e}")
    
    def _process_ticker_packet(self, message, security_id):
        """Process ticker packet (LTP + LTT)"""
        try:
            if len(message) >= 16:
                ltp = struct.unpack('<f', message[8:12])[0]  # Little endian float32
                ltt = struct.unpack('<I', message[12:16])[0]  # Little endian int32
                
                price_data = {
                    'type': 'ticker',
                    'security_id': security_id,
                    'ltp': ltp,
                    'ltt': ltt,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Ticker - SecurityID: {security_id}, LTP: {ltp}, LTT: {ltt}")
                
                # Notify price callbacks
                self._notify_price_callbacks(price_data)
                
        except struct.error as e:
            logger.error(f"Ticker packet parsing error: {e}")
    
    def _process_quote_packet(self, message, security_id):
        """Process quote packet (detailed price data)"""
        try:
            if len(message) >= 50:  # Minimum size for quote packet
                ltp = struct.unpack('<f', message[8:12])[0]
                ltq = struct.unpack('<H', message[12:14])[0]
                ltt = struct.unpack('<I', message[14:18])[0]
                atp = struct.unpack('<f', message[18:22])[0]
                volume = struct.unpack('<I', message[22:26])[0]
                total_sell_qty = struct.unpack('<I', message[26:30])[0]
                total_buy_qty = struct.unpack('<I', message[30:34])[0]
                
                price_data = {
                    'type': 'quote',
                    'security_id': security_id,
                    'ltp': ltp,
                    'ltq': ltq,
                    'ltt': ltt,
                    'atp': atp,
                    'volume': volume,
                    'total_sell_qty': total_sell_qty,
                    'total_buy_qty': total_buy_qty,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Quote - SecurityID: {security_id}, LTP: {ltp}, Volume: {volume}")
                
                # Notify price callbacks
                self._notify_price_callbacks(price_data)
                
        except struct.error as e:
            logger.error(f"Quote packet parsing error: {e}")
    
    def _process_prev_close_packet(self, message, security_id):
        """Process previous close packet"""
        try:
            if len(message) >= 16:
                prev_close = struct.unpack('<f', message[8:12])[0]
                prev_oi = struct.unpack('<I', message[12:16])[0]
                
                price_data = {
                    'type': 'prev_close',
                    'security_id': security_id,
                    'prev_close': prev_close,
                    'prev_oi': prev_oi,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Prev Close - SecurityID: {security_id}, Close: {prev_close}, OI: {prev_oi}")
                
                # Notify price callbacks
                self._notify_price_callbacks(price_data)
                
        except struct.error as e:
            logger.error(f"Previous close packet parsing error: {e}")
    
    def _process_disconnect_packet(self, message):
        """Process disconnect packet"""
        try:
            if len(message) >= 10:
                disconnect_code = struct.unpack('<H', message[8:10])[0]
                logger.warning(f"Received disconnect packet with code: {disconnect_code}")
                
                # Notify connection callbacks
                for callback in self.connection_callbacks:
                    try:
                        callback("server_disconnect", f"Server disconnect code: {disconnect_code}")
                    except Exception as e:
                        logger.error(f"Connection callback error: {e}")
                        
        except struct.error as e:
            logger.error(f"Disconnect packet parsing error: {e}")
    
    def _notify_price_callbacks(self, price_data):
        """Notify all registered price callbacks"""
        for callback in self.price_callbacks:
            try:
                callback(price_data)
            except Exception as e:
                logger.error(f"Price callback error: {e}")
    
    def get_connection_status(self):
        """Get current connection status"""
        return {
            'is_connected': self.is_connected,
            'subscribed_instruments': len(self.subscribed_instruments),
            'message_count': self.message_count,
            'instruments': self.subscribed_instruments
        }

# Page configuration
st.set_page_config(
    page_title="NIFTY ATM SuperTrend Live",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
    }
    .error-text {
        color: #dc3545;
        font-weight: bold;
    }
    .warning-text {
        color: #ffc107;
        font-weight: bold;
    }
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #28a745;
        border-radius: 50%;
        margin-right: 5px;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    default_states = {
        'strikes_data': [],
        'price_data': deque(maxlen=200),
        'ws_connected': False,
        'selected_strike': None,
        'ws_client': None,
        'live_price': None,
        'last_update': None,
        'supertrend_signal': None,
        'message_count': 0,
        'connection_status': 'Disconnected',
        'st_period': 10,
        'st_multiplier': 3.0
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Helper functions
@st.cache_data(ttl=300)
def get_expiry_list(access_token, client_id):
    """Get available expiry dates"""
    headers = {
        'Content-Type': 'application/json',
        'access-token': access_token,
        'client-id': client_id
    }
    
    payload = {
        'UnderlyingScrip': 13,
        'UnderlyingSeg': 'IDX_I'
    }
    
    try:
        response = requests.post(
            'https://api.dhan.co/v2/optionchain/expirylist',
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

@st.cache_data(ttl=60)
def get_option_chain(access_token, client_id, expiry):
    """Get option chain data"""
    headers = {
        'Content-Type': 'application/json',
        'access-token': access_token,
        'client-id': client_id
    }
    
    payload = {
        'UnderlyingScrip': 13,
        'UnderlyingSeg': 'IDX_I',
        'Expiry': expiry
    }
    
    try:
        response = requests.post(
            'https://api.dhan.co/v2/optionchain',
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def get_instrument_master():
    """Get instrument master list - placeholder function"""
    st.info("Instrument master lookup not yet implemented. Please enter Security ID manually.")
    return None

def process_option_chain(option_chain_data, expiry_date):
    """Process option chain data into strikes list"""
    if not option_chain_data or 'data' not in option_chain_data:
        return [], 0
    
    nifty_ltp = option_chain_data['data'].get('last_price', 0)
    option_chain = option_chain_data['data'].get('oc', {})
    
    strikes = []
    for strike_str, strike_data in option_chain.items():
        if 'ce' in strike_data:
            strike_price = float(strike_str)
            ce_data = strike_data['ce']
            
            # Try to extract Security ID from various possible fields
            security_id = None
            possible_id_fields = ['security_id', 'instrument_token', 'token', 'scrip_code', 'instrument_key']
            
            for field in possible_id_fields:
                if field in ce_data and ce_data[field]:
                    security_id = str(ce_data[field])
                    break
            
            # If no Security ID found, create a placeholder that indicates manual entry needed
            if not security_id:
                security_id = f"MANUAL_REQUIRED_{int(strike_price)}_CE"
            
            strike_info = {
                'strike': strike_price,
                'ltp': ce_data.get('last_price', 0),
                'volume': ce_data.get('volume', 0),
                'oi': ce_data.get('oi', 0),
                'distance_from_spot': abs(strike_price - nifty_ltp),
                'security_id': security_id,
                'needs_manual_id': security_id.startswith('MANUAL_REQUIRED'),
                'expiry': expiry_date
            }
            strikes.append(strike_info)
    
    strikes.sort(key=lambda x: x['distance_from_spot'])
    return strikes, nifty_ltp

def calculate_supertrend(prices, period=10, multiplier=3.0):
    """Calculate SuperTrend indicator"""
    if len(prices) < period + 1:
        return [], [], None
    
    price_list = list(prices)
    
    # Generate high/low from prices
    high = [p * 1.002 for p in price_list]
    low = [p * 0.998 for p in price_list]
    
    # Calculate HL2 and ATR
    hl2 = [(h + l) / 2 for h, l in zip(high, low)]
    
    # Simple ATR calculation
    atr_values = []
    for i in range(period, len(price_list)):
        tr_values = []
        for j in range(i - period, i):
            if j > 0:
                tr = max(
                    high[j] - low[j],
                    abs(high[j] - price_list[j-1]),
                    abs(low[j] - price_list[j-1])
                )
                tr_values.append(tr)
        
        if tr_values:
            atr_values.append(sum(tr_values) / len(tr_values))
        else:
            atr_values.append(0)
    
    if not atr_values:
        return [], [], None
    
    # Calculate SuperTrend
    supertrend_up = []
    supertrend_down = []
    trend_direction = []
    
    for i in range(len(price_list)):
        if i < len(atr_values):
            atr = atr_values[i]
        else:
            atr = atr_values[-1] if atr_values else 0
        
        upper_band = hl2[i] + multiplier * atr
        lower_band = hl2[i] - multiplier * atr
        
        if i == 0:
            trend_direction.append(1)
        else:
            if trend_direction[-1] == 1 and price_list[i] <= lower_band:
                trend_direction.append(-1)
            elif trend_direction[-1] == -1 and price_list[i] >= upper_band:
                trend_direction.append(1)
            else:
                trend_direction.append(trend_direction[-1])
        
        if trend_direction[-1] == 1:
            supertrend_up.append(lower_band)
            supertrend_down.append(None)
        else:
            supertrend_up.append(None)
            supertrend_down.append(upper_band)
    
    current_signal = "BULLISH" if trend_direction[-1] == 1 else "BEARISH"
    return supertrend_up, supertrend_down, current_signal

# WebSocket callback functions
def on_price_update(price_data):
    """Handle price updates from WebSocket"""
    if price_data.get('type') in ['ticker', 'quote'] and 'ltp' in price_data:
        st.session_state.price_data.append(price_data['ltp'])
        st.session_state.live_price = price_data['ltp']
        st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        st.session_state.message_count += 1
        
        if len(st.session_state.price_data) > 10:
            period = st.session_state.get('st_period', 10)
            multiplier = st.session_state.get('st_multiplier', 3.0)
            _, _, signal = calculate_supertrend(st.session_state.price_data, period, multiplier)
            st.session_state.supertrend_signal = signal

def on_connection_change(status, message):
    """Handle connection status changes"""
    st.session_state.connection_status = f"{status}: {message}"
    if status == "connected":
        st.session_state.ws_connected = True
    else:
        st.session_state.ws_connected = False

# Title
st.markdown('<h1 class="main-header">NIFTY ATM Call SuperTrend - Live Analysis</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Credentials - Use Streamlit Secrets
st.sidebar.subheader("DhanHQ Credentials")

try:
    access_token = st.secrets["DHAN_ACCESS_TOKEN"]
    client_id = st.secrets["DHAN_CLIENT_ID"]
    st.sidebar.success("✅ Using credentials from Streamlit secrets")
    st.sidebar.info(f"Client ID: {client_id[:8]}...")
except KeyError:
    st.sidebar.warning("⚠️ Streamlit secrets not configured. Using manual input.")
    access_token = st.sidebar.text_input("Access Token", type="password", help="Your DhanHQ access token")
    client_id = st.sidebar.text_input("Client ID", help="Your DhanHQ client ID")

# SuperTrend Parameters
st.sidebar.subheader("SuperTrend Settings")
st_period = st.sidebar.selectbox("Period", [10, 14, 21], index=0)
st_multiplier = st.sidebar.selectbox("Multiplier", [2.0, 3.0, 4.0], index=1)

st.session_state.st_period = st_period
st.session_state.st_multiplier = st_multiplier

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Strike Selection & Connection")
    
    if st.button("🔄 Fetch Available Strikes", type="primary"):
        if not access_token or not client_id:
            st.error("Please enter both Access Token and Client ID")
        else:
            with st.spinner("Fetching option chain data and instrument master..."):
                get_expiry_list.clear()
                get_option_chain.clear()
                
                expiry_data = get_expiry_list(access_token, client_id)
                
                if expiry_data and 'data' in expiry_data:
                    current_expiry = expiry_data['data'][0]
                    st.success(f"Using expiry: {current_expiry}")
                    
                    option_chain_data = get_option_chain(access_token, client_id, current_expiry)
                    
                    if option_chain_data:
                        # Load instrument master for Security ID lookup
                        with st.spinner("Loading instrument master for Security ID lookup..."):
                            instrument_df = get_instrument_master()
                            if instrument_df is not None:
                                st.success(f"Loaded {len(instrument_df)} instruments from master file")
                            else:
                                st.warning("Could not load instrument master. Security IDs will need manual entry.")
                        
                        strikes, nifty_ltp = process_option_chain(option_chain_data, current_expiry)
                        st.session_state.strikes_data = strikes
                        
                        # Count how many strikes have automatic Security IDs
                        auto_count = sum(1 for s in strikes if not s.get('needs_manual_id', True))
                        manual_count = len(strikes) - auto_count
                        
                        st.success(f"Found {len(strikes)} strikes. NIFTY LTP: {nifty_ltp:.2f}")
                        if auto_count > 0:
                            st.info(f"Auto-found Security IDs: {auto_count}, Manual required: {manual_count}")
                        else:
                            st.warning("No automatic Security IDs found. Please verify instrument master loading.")
                else:
                    st.error("Failed to fetch expiry list")
    
    if st.session_state.strikes_data:
        st.subheader("Available Strikes")
        
        selected_idx = st.selectbox(
            "Select Strike",
            range(len(st.session_state.strikes_data)),
            format_func=lambda x: f"{st.session_state.strikes_data[x]['strike']:.0f} CE - LTP: {st.session_state.strikes_data[x]['ltp']:.2f}",
            index=0
        )
        
        if selected_idx is not None:
            st.session_state.selected_strike = st.session_state.strikes_data[selected_idx]
            
            with st.container():
                st.markdown("**Selected Strike:**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Strike", f"{st.session_state.selected_strike['strike']:.0f}")
                    st.metric("LTP", f"₹{st.session_state.selected_strike['ltp']:.2f}")
                with col_b:
                    st.metric("Volume", f"{st.session_state.selected_strike['volume']:,}")
                    st.metric("OI", f"{st.session_state.selected_strike['oi']:,}")
    
    if st.session_state.selected_strike:
        st.subheader("WebSocket Configuration")
        
        # Show Security ID status
        if st.session_state.selected_strike.get('needs_manual_id', False):
            st.warning("⚠️ Security ID not found in instrument master. Please enter manually.")
        else:
            st.success("✅ Security ID found automatically from instrument master")
        
        # Get the Security ID - either auto-found or empty for manual entry
        default_security_id = st.session_state.selected_strike.get('security_id', '')
        if default_security_id.startswith('MANUAL_REQUIRED'):
            default_security_id = ''
        
        security_id = st.text_input(
            "Security ID",
            value=default_security_id,
            help="Security ID from DhanHQ instrument master file",
            placeholder="Will be auto-filled if found in instrument master"
        )
        
        # Show strike and expiry info
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.caption(f"Strike: {st.session_state.selected_strike['strike']:.0f}")
        with col_info2:
            if 'expiry' in st.session_state.selected_strike:
                st.caption(f"Expiry: {st.session_state.selected_strike['expiry']}")
        
        # Show helper information only if manual entry is needed
        if st.session_state.selected_strike.get('needs_manual_id', False):
            with st.expander("How to find Security ID manually"):
                st.markdown("""
                **If automatic lookup failed, try these methods:**
                
                1. **DhanHQ Trading Platform:**
                   - Go to Options → NIFTY → Find your strike
                   - Look for Security ID or Instrument Token in option details
                
                2. **Download Detailed Instrument Master:**
                   ```
                   https://images.dhan.co/api-data/api-scrip-master-detailed.csv
                   ```
                   - Search for: NIFTY + your strike + CE + current expiry
                
                3. **API Endpoint for specific segment:**
                   ```
                   https://api.dhan.co/v2/instrument/NSE_FNO
                   ```
                """)
        
        col_ws1, col_ws2 = st.columns(2)
        
        with col_ws1:
            if st.button("🔗 Connect Live", disabled=st.session_state.ws_connected):
                if not access_token or not client_id or not security_id:
                    st.error("Please provide all credentials and Security ID")
                else:
                    st.session_state.ws_client = DhanWebSocketClient(access_token, client_id)
                    st.session_state.ws_client.add_price_callback(on_price_update)
                    st.session_state.ws_client.add_connection_callback(on_connection_change)
                    
                    if st.session_state.ws_client.connect():
                        time.sleep(1)
                        st.session_state.ws_client.subscribe_instrument("NSE_FNO", security_id)
                        st.success("WebSocket connected and subscribed!")
                    else:
                        st.error("Failed to connect WebSocket")
        
        with col_ws2:
            if st.button("❌ Disconnect", disabled=not st.session_state.ws_connected):
                if st.session_state.ws_client:
                    st.session_state.ws_client.disconnect()
                    st.session_state.ws_client = None
                    st.session_state.ws_connected = False
                    st.info("WebSocket disconnected")
    
    st.subheader("Connection Status")
    if st.session_state.ws_connected:
        st.markdown('<span class="live-indicator"></span>**LIVE**', unsafe_allow_html=True)
        st.success(f"Connected - Messages: {st.session_state.message_count}")
    else:
        st.error("Not Connected")
    
    if st.session_state.last_update:
        st.info(f"Last Update: {st.session_state.last_update}")

with col2:
    st.subheader("Live SuperTrend Chart")
    
    if st.session_state.price_data and len(st.session_state.price_data) > 1:
        prices = list(st.session_state.price_data)
        timestamps = [f"{i:03d}" for i in range(len(prices))]
        
        st_up, st_down, current_signal = calculate_supertrend(prices, st_period, st_multiplier)
        
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=[f"Live SuperTrend - {st.session_state.selected_strike['strike']:.0f} CE" if st.session_state.selected_strike else "Live SuperTrend"]
        )
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=prices,
            mode='lines+markers',
            name='Option Price',
            line=dict(color='blue', width=2),
            marker=dict(size=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=st_up,
            mode='lines',
            name='SuperTrend Up',
            line=dict(color='green', width=3),
            connectgaps=False
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=st_down,
            mode='lines',
            name='SuperTrend Down',
            line=dict(color='red', width=3),
            connectgaps=False
        ))
        
        fig.update_layout(
            height=500,
            xaxis_title="Time Points",
            yaxis_title="Price (₹)",
            hovermode='x unified',
            showlegend=True,
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            if st.session_state.live_price:
                st.metric("Current LTP", f"₹{st.session_state.live_price:.2f}")
        
        with col_m2:
            if current_signal:
                color = "normal" if current_signal == "BULLISH" else "inverse"
                st.metric("Signal", current_signal, delta=None, delta_color=color)
        
        with col_m3:
            st.metric("Data Points", len(prices))
        
        with col_m4:
            st.metric("Messages", st.session_state.message_count)
        
        if current_signal:
            if current_signal == "BULLISH":
                st.success("📈 Current Signal: BULLISH")
            else:
                st.error("📉 Current Signal: BEARISH")
    
    elif st.session_state.selected_strike and not st.session_state.ws_connected:
        st.info("Connect to WebSocket to see live SuperTrend analysis")
        
        if st.button("Show Sample Chart"):
            sample_prices = np.random.normal(100, 5, 50).cumsum() + st.session_state.selected_strike['ltp']
            sample_timestamps = [f"{i:03d}" for i in range(len(sample_prices))]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sample_timestamps,
                y=sample_prices,
                mode='lines',
                name='Sample Price Movement',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                height=400,
                title="Sample SuperTrend Chart",
                xaxis_title="Time Points",
                yaxis_title="Price (₹)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Select a strike and connect to WebSocket to view live data")

st.markdown("---")

if st.session_state.ws_connected:
    time.sleep(2)
    st.rerun()

with st.expander("📋 Instructions & Setup"):
    st.markdown("""
    ### How to Use:
    1. **Enter Credentials**: Add your DhanHQ Access Token and Client ID in the sidebar
    2. **Fetch Strikes**: Click "Fetch Available Strikes" to load current option data
    3. **Select Strike**: Choose your preferred strike from the dropdown
    4. **Enter Security ID**: Input the actual Security ID from DhanHQ instrument master
    5. **Connect Live**: Click "Connect Live" to start receiving real-time data
    6. **Monitor**: Watch the SuperTrend chart update in real-time
    
    ### Files for GitHub Repository:
    - streamlit_app.py (Main application)
    - websocket_handler.py (WebSocket client)
    - requirements.txt (Dependencies)
    """)

if st.sidebar.checkbox("Show Debug Info"):
    st.subheader("Debug Information")
    
    debug_info = {
        "Connection Status": st.session_state.connection_status,
        "WebSocket Connected": st.session_state.ws_connected,
        "Selected Strike": st.session_state.selected_strike,
        "Price Data Length": len(st.session_state.price_data),
        "Message Count": st.session_state.message_count,
        "Last Update": st.session_state.last_update,
        "Live Price": st.session_state.live_price,
        "SuperTrend Signal": st.session_state.supertrend_signal
    }
    
    st.json(debug_info)
