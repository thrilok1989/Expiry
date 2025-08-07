Here's the modified script with:
1. ATM ±5 strike price analysis only
2. Display limited to ATM ±5 strikes
3. Added ATM/OTM/ITM column
4. Removed the specified Telegram message

```python
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm
from pytz import timezone
import plotly.graph_objects as go
import io

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=300000, key="datarefresh")  # Refresh every 5 min

# Initialize session state for price data
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame(columns=["Time", "Spot"])

# Initialize session state for enhanced features
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

if 'call_log_book' not in st.session_state:
    st.session_state.call_log_book = []

if 'export_data' not in st.session_state:
    st.session_state.export_data = False

if 'support_zone' not in st.session_state:
    st.session_state.support_zone = (None, None)

if 'resistance_zone' not in st.session_state:
    st.session_state.resistance_zone = (None, None)

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

def calculate_greeks(option_type, S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'CE' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365 if option_type == 'CE' else (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
    rho = (K * T * math.exp(-r * T) * norm.cdf(d2)) / 100 if option_type == 'CE' else (-K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100
    return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)

def final_verdict(score):
    if score >= 4:
        return "Strong Bullish"
    elif score >= 2:
        return "Bullish"
    elif score <= -4:
        return "Strong Bearish"
    elif score <= -2:
        return "Bearish"
    else:
        return "Neutral"

def delta_volume_bias(price, volume, chg_oi):
    if price > 0 and volume > 0 and chg_oi > 0:
        return "Bullish"
    elif price < 0 and volume > 0 and chg_oi > 0:
        return "Bearish"
    elif price > 0 and volume > 0 and chg_oi < 0:
        return "Bullish"
    elif price < 0 and volume > 0 and chg_oi < 0:
        return "Bearish"
    else:
        return "Neutral"

def sudden_liquidity_spike(row):
    ce_spike = row['changeinOpenInterest_CE'] > 1.5 * row['openInterest_CE'] and row['totalTradedVolume_CE'] > 1500
    pe_spike = row['changeinOpenInterest_PE'] > 1.5 * row['openInterest_PE'] and row['totalTradedVolume_PE'] > 1500
    return ce_spike or pe_spike

weights = {
    "ChgOI_Bias": 2,
    "Volume_Bias": 1,
    "Gamma_Bias": 1,
    "AskQty_Bias": 1,
    "BidQty_Bias": 1,
    "IV_Bias": 1,
    "DVP_Bias": 1,
}

def determine_level(row):
    if row['openInterest_PE'] > 1.12 * row['openInterest_CE']:
        return "Support"
    elif row['openInterest_CE'] > 1.12 * row['openInterest_PE']:
        return "Resistance"
    else:
        return "Neutral"

def is_in_zone(spot, strike, level):
    if level == "Support":
        return strike - 10 <= spot <= strike + 10
    elif level == "Resistance":
        return strike - 10 <= spot <= strike + 10
    return False

def get_support_resistance_zones(df, spot):
    support_strikes = df[df['Level'] == "Support"]['strikePrice'].tolist()
    resistance_strikes = df[df['Level'] == "Resistance"]['strikePrice'].tolist()

    nearest_supports = sorted([s for s in support_strikes if s <= spot], reverse=True)[:2]
    nearest_resistances = sorted([r for r in resistance_strikes if r >= spot])[:2]

    support_zone = (min(nearest_supports), max(nearest_supports)) if len(nearest_supports) >= 2 else (nearest_supports[0], nearest_supports[0]) if nearest_supports else (None, None)
    resistance_zone = (min(nearest_resistances), max(nearest_resistances)) if len(nearest_resistances) >= 2 else (nearest_resistances[0], nearest_resistances[0]) if nearest_resistances else (None, None)

    return support_zone, resistance_zone

def detect_liquidity_zones(df, spot_price, price_history):
    zones = []
    unique_strikes = df['strikePrice'].unique()
    
    for strike in unique_strikes:
        revisit_count = sum((abs(spot - strike) <= 10) for spot in price_history)
        strike_data = df[df['strikePrice'] == strike]
        
        avg_volume = (strike_data['totalTradedVolume_CE'].mean() + 
                     strike_data['totalTradedVolume_PE'].mean())
        avg_oi_change = (strike_data['changeinOpenInterest_CE'].mean() + 
                        strike_data['changeinOpenInterest_PE'].mean())
        
        if revisit_count >= 3 and avg_volume > 5000 and avg_oi_change > 0:
            zones.append({
                'strike': strike,
                'revisits': revisit_count,
                'volume': round(avg_volume),
                'oi_change': round(avg_oi_change)
            })
    return pd.DataFrame(zones)

def reversal_score(row):
    score = 0
    direction = ""
    
    # Bearish Reversal Signals (Market might go DOWN)
    if (row['changeinOpenInterest_CE'] < 0 and 
        row['changeinOpenInterest_PE'] > 0 and
        row['impliedVolatility_PE'] > row['impliedVolatility_CE']):
        score += 2
        direction = "DOWN"
    
    # Bullish Reversal Signals (Market might go UP)
    elif (row['changeinOpenInterest_CE'] > 0 and 
          row['changeinOpenInterest_PE'] < 0 and
          row['impliedVolatility_CE'] > row['impliedVolatility_PE']):
        score += 2
        direction = "UP"
    
    # Additional confirmation from bid/ask quantities
    if row['bidQty_PE'] > row['bidQty_CE'] and row['askQty_PE'] > row['askQty_CE']:
        score += 1
        if not direction:  # If direction not set yet
            direction = "DOWN"
    elif row['bidQty_CE'] > row['bidQty_PE'] and row['askQty_CE'] > row['askQty_PE']:
        score += 1
        if not direction:  # If direction not set yet
            direction = "UP"
    
    return score, direction

def expiry_bias_score(row):
    score = 0

    # OI + Price Based Bias Logic (using available fields)
    if row['changeinOpenInterest_CE'] > 0 and row['lastPrice_CE'] > row['previousClose_CE']:
        score += 1  # New CE longs → Bullish
    if row['changeinOpenInterest_PE'] > 0 and row['lastPrice_PE'] > row['previousClose_PE']:
        score -= 1  # New PE longs → Bearish
    if row['changeinOpenInterest_CE'] > 0 and row['lastPrice_CE'] < row['previousClose_CE']:
        score -= 1  # CE writing → Bearish
    if row['changeinOpenInterest_PE'] > 0 and row['lastPrice_PE'] < row['previousClose_PE']:
        score += 1  # PE writing → Bullish

    # Bid Volume Dominance (using available fields)
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

    # Bid-Ask Pressure (using lastPrice and underlying price as proxy)
    if 'underlyingValue' in row:
        if abs(row['lastPrice_CE'] - row['underlyingValue']) < abs(row['lastPrice_PE'] - row['underlyingValue']):
            score += 0.5  # CE closer to spot → Bullish
        else:
            score -= 0.5  # PE closer to spot → Bearish

    return score

def expiry_entry_signal(df, support_levels, resistance_levels, score_threshold=1.5):
    entries = []
    for _, row in df.iterrows():
        strike = row['strikePrice']
        score = expiry_bias_score(row)

        # Entry at support/resistance + Bias Score Condition
        if score >= score_threshold and strike in support_levels:
            entries.append({
                'type': 'BUY CALL',
                'strike': strike,
                'score': score,
                'ltp': row['lastPrice_CE'],
                'reason': 'Bullish score + support zone'
            })

        if score <= -score_threshold and strike in resistance_levels:
            entries.append({
                'type': 'BUY PUT',
                'strike': strike,
                'score': score,
                'ltp': row['lastPrice_PE'],
                'reason': 'Bearish score + resistance zone'
            })

    return entries

def display_enhanced_trade_log():
    if not st.session_state.trade_log:
        st.info("No trades logged yet")
        return
    st.markdown("### 📜 Enhanced Trade Log")
    df_trades = pd.DataFrame(st.session_state.trade_log)
    if 'Current_Price' not in df_trades.columns:
        df_trades['Current_Price'] = df_trades['LTP'] * np.random.uniform(0.8, 1.3, len(df_trades))
        df_trades['Unrealized_PL'] = (df_trades['Current_Price'] - df_trades['LTP']) * 75
        df_trades['Status'] = df_trades['Unrealized_PL'].apply(
            lambda x: '🟢 Profit' if x > 0 else '🔴 Loss' if x < -100 else '🟡 Breakeven'
        )
    def color_pnl(row):
        colors = []
        for col in row.index:
            if col == 'Unrealized_PL':
                if row[col] > 0:
                    colors.append('background-color: #90EE90; color: black')
                elif row[col] < -100:
                    colors.append('background-color: #FFB6C1; color: black')
                else:
                    colors.append('background-color: #FFFFE0; color: black')
            else:
                colors.append('')
        return colors
    styled_trades = df_trades.style.apply(color_pnl, axis=1)
    st.dataframe(styled_trades, use_container_width=True)
    total_pl = df_trades['Unrealized_PL'].sum()
    win_rate = len(df_trades[df_trades['Unrealized_PL'] > 0]) / len(df_trades) * 100
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total P&L", f"₹{total_pl:,.0f}")
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Total Trades", len(df_trades))

def create_export_data(df_summary, trade_log, spot_price):
    # Create Excel data
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Option_Chain_Summary', index=False)
        if trade_log:
            pd.DataFrame(trade_log).to_excel(writer, sheet_name='Trade_Log', index=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nifty_analysis_{timestamp}.xlsx"
    
    return output.getvalue(), filename

def handle_export_data(df_summary, spot_price):
    if 'export_data' in st.session_state and st.session_state.export_data:
        try:
            excel_data, filename = create_export_data(df_summary, st.session_state.trade_log, spot_price)
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            st.success("✅ Export ready! Click the download button above.")
            st.session_state.export_data = False
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
            st.session_state.export_data = False

def plot_price_with_sr():
    price_df = st.session_state['price_data'].copy()
    if price_df.empty or price_df['Spot'].isnull().all():
        st.info("Not enough data to show price action chart yet.")
        return
    price_df['Time'] = pd.to_datetime(price_df['Time'])
    support_zone = st.session_state.get('support_zone', (None, None))
    resistance_zone = st.session_state.get('resistance_zone', (None, None))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_df['Time'], 
        y=price_df['Spot'], 
        mode='lines+markers', 
        name='Spot Price',
        line=dict(color='blue', width=2)
    ))
    if all(support_zone) and None not in support_zone:
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, x1=1,
            y0=support_zone[0], y1=support_zone[1],
            fillcolor="rgba(0,255,0,0.08)", line=dict(width=0),
            layer="below"
        )
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[support_zone[0], support_zone[0]],
            mode='lines',
            name='Support Low',
            line=dict(color='green', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[support_zone[1], support_zone[1]],
            mode='lines',
            name='Support High',
            line=dict(color='green', dash='dot')
        ))
    if all(resistance_zone) and None not in resistance_zone:
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, x1=1,
            y0=resistance_zone[0], y1=resistance_zone[1],
            fillcolor="rgba(255,0,0,0.08)", line=dict(width=0),
            layer="below"
        )
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[resistance_zone[0], resistance_zone[0]],
            mode='lines',
            name='Resistance Low',
            line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[resistance_zone[1], resistance_zone[1]],
            mode='lines',
            name='Resistance High',
            line=dict(color='red', dash='dot')
        ))
    fig.update_layout(
        title="Nifty Spot Price Action with Support & Resistance",
        xaxis_title="Time",
        yaxis_title="Spot Price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def auto_update_call_log(current_price):
    for call in st.session_state.call_log_book:
        if call["Status"] != "Active":
            continue
        if call["Type"] == "CE":
            if current_price >= max(call["Targets"].values()):
                call["Status"] = "Hit Target"
                call["Hit_Target"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price
            elif current_price <= call["Stoploss"]:
                call["Status"] = "Hit Stoploss"
                call["Hit_Stoploss"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price
        elif call["Type"] == "PE":
            if current_price <= min(call["Targets"].values()):
                call["Status"] = "Hit Target"
                call["Hit_Target"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price
            elif current_price >= call["Stoploss"]:
                call["Status"] = "Hit Stoploss"
                call["Hit_Stoploss"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price

def display_call_log_book():
    st.markdown("### 📚 Call Log Book")
    if not st.session_state.call_log_book:
        st.info("No calls have been made yet.")
        return
    df_log = pd.DataFrame(st.session_state.call_log_book)
    st.dataframe(df_log, use_container_width=True)
    if st.button("Download Call Log Book as CSV"):
        st.download_button(
            label="Download CSV",
            data=df_log.to_csv(index=False).encode(),
            file_name="call_log_book.csv",
            mime="text/csv"
        )

def analyze():
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("09:00", "%H:%M").time()
        market_end = datetime.strptime("15:40", "%H:%M").time()

        if current_day >= 5 or not (market_start <= current_time <= market_end):
            st.warning("⏳ Market Closed (Mon-Fri 9:00-15:40)")
            return

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

        today = datetime.now(timezone("Asia/Kolkata"))
        expiry_date = timezone("Asia/Kolkata").localize(datetime.strptime(expiry, "%d-%b-%Y"))
        
        # EXPIRY DAY LOGIC - Check if today is expiry day
        is_expiry_day = today.date() == expiry_date.date()
        
        if is_expiry_day:
            st.info("""
📅 **EXPIRY DAY DETECTED**
- Using specialized expiry day analysis
- IV Collapse, OI Unwind, Volume Spike expected
- Modified signals will be generated
""")
            
            # Store spot history for expiry day too
            current_time_str = now.strftime("%H:%M:%S")
            new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
            st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)
            
            st.markdown(f"### 📍 Spot Price: {underlying}")
            
            # Get previous close data (needed for expiry day analysis)
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
            support_levels = df[df['Level'] == "Support"]['strikePrice'].unique()
            resistance_levels = df[df['Level'] == "Resistance"]['strikePrice'].unique()
            
            # Generate expiry day signals
            expiry_signals = expiry_entry_signal(df, support_levels, resistance_levels)
            
            # Display expiry day specific UI
            st.markdown("### 🎯 Expiry Day Signals")
            if expiry_signals:
                for signal in expiry_signals:
                    st.success(f"""
                    {signal['type']} at {signal['strike']} 
                    (Score: {signal['score']:.1f}, LTP: ₹{signal['ltp']})
                    Reason: {signal['reason']}
                    """)
                    
                    # Add to trade log
                    st.session_state.trade_log.append({
                        "Time": now.strftime("%H:%M:%S"),
                        "Strike": signal['strike'],
                        "Type": 'CE' if 'CALL' in signal['type'] else 'PE',
                        "LTP": signal['ltp'],
                        "Target": round(signal['ltp'] * 1.2, 2),
                        "SL": round(signal['ltp'] * 0.8, 2)
                    })
                    
            else:
                st.warning("No strong expiry day signals detected")
            
            # Show expiry day specific data
            with st.expander("📊 Expiry Day Option Chain"):
                df['ExpiryBiasScore'] = df.apply(expiry_bias_score, axis=1)
                st.dataframe(df[['strikePrice', 'ExpiryBiasScore', 'lastPrice_CE', 'lastPrice_PE', 
                               'changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                               'bidQty_CE', 'bidQty_PE']])
            
            return  # Exit early after expiry day processing
            
        # Non-expiry day processing
        T = max((expiry_date - today).days, 1) / 365
        r = 0.06

        calls, puts = [], []

        for item in records:
            if 'CE' in item and item['CE']['expiryDate'] == expiry:
                ce = item['CE']
                if ce['impliedVolatility'] > 0:
                    greeks = calculate_greeks('CE', underlying, ce['strikePrice'], T, r, ce['impliedVolatility'] / 100)
                    ce.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                calls.append(ce)

            if 'PE' in item and item['PE']['expiryDate'] == expiry:
                pe = item['PE']
                if pe['impliedVolatility'] > 0:
                    greeks = calculate_greeks('PE', underlying, pe['strikePrice'], T, r, pe['impliedVolatility'] / 100)
                    pe.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                puts.append(pe)

        df_ce = pd.DataFrame(calls)
        df_pe = pd.DataFrame(puts)
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        # MODIFIED: Only keep ATM ±5 strikes
        df = df[df['strikePrice'].between(atm_strike - 5*50, atm_strike + 5*50)]  # Assuming 50pt strikes
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)

        bias_results, total_score = [], 0
        for _, row in df.iterrows():
            # MODIFIED: Only analyze ATM ±5 strikes
            if abs(row['strikePrice'] - atm_strike) > 5*50:  # Assuming 50pt strikes
                continue

            score = 0
            row_data = {
                "Strike": row['strikePrice'],
                "Zone": row['Zone'],  # Added Zone column
                "Level": row['Level'],
                "ChgOI_Bias": "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish",
                "Volume_Bias": "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish",
                "Gamma_Bias": "Bullish" if row['Gamma_CE'] < row['Gamma_PE'] else "Bearish",
                "AskQty_Bias": "Bullish" if row['askQty_PE'] > row['askQty_CE'] else "Bearish",
                "BidQty_Bias": "Bearish" if row['bidQty_PE'] > row['bidQty_CE'] else "Bullish",
                "IV_Bias": "Bullish" if row['impliedVolatility_CE'] > row['impliedVolatility_PE'] else "Bearish",
                "DVP_Bias": delta_volume_bias(
                    row['lastPrice_CE'] - row['lastPrice_PE'],
                    row['totalTradedVolume_CE'] - row['totalTradedVolume_PE'],
                    row['changeinOpenInterest_CE'] - row['changeinOpenInterest_PE']
                )
            }

            for k in row_data:
                if "_Bias" in k:
                    bias = row_data[k]
                    score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)

            row_data["BiasScore"] = score
            row_data["Verdict"] = final_verdict(score)
            total_score += score
            bias_results.append(row_data)

            if sudden_liquidity_spike(row):
                send_telegram_message(
                    f"⚡ Sudden Liquidity Spike!\nStrike: {row['strikePrice']}\nCE OI Chg: {row['changeinOpenInterest_CE']} | PE OI Chg: {row['changeinOpenInterest_PE']}\nVol CE: {row['totalTradedVolume_CE']} | PE: {row['totalTradedVolume_PE']}"
                )

        df_summary = pd.DataFrame(bias_results)
        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"
        support_zone, resistance_zone = get_support_resistance_zones(df, underlying)
        
        # Store zones in session state for enhanced functions
        st.session_state.support_zone = support_zone
        st.session_state.resistance_zone = resistance_zone

        current_time_str = now.strftime("%H:%M:%S")
        new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
        st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)

        support_str = f"{support_zone[1]} to {support_zone[0]}" if all(support_zone) else "N/A"
        resistance_str = f"{resistance_zone[0]} to {resistance_zone[1]}" if all(resistance_zone) else "N/A"

        atm_signal, suggested_trade = "No Signal", ""
        signal_sent = False

        for row in bias_results:
            if not is_in_zone(underlying, row['Strike'], row['Level']):
                continue

            if row['Level'] == "Support" and total_score >= 4 and "Bullish" in market_view:
                option_type = 'CE'
            elif row['Level'] == "Resistance" and total_score <= -4 and "Bearish" in market_view:
                option_type = 'PE'
            else:
                continue

            ltp = df.loc[df['strikePrice'] == row['Strike'], f'lastPrice_{option_type}'].values[0]
            iv = df.loc[df['strikePrice'] == row['Strike'], f'impliedVolatility_{option_type}'].values[0]
            target = round(ltp * (1 + iv / 100), 2)
            stop_loss = round(ltp * 0.8, 2)

            atm_signal = f"{'CALL' if option_type == 'CE' else 'PUT'} Entry (Bias Based at {row['Level']})"
            suggested_trade = f"Strike: {row['Strike']} {option_type} @ ₹{ltp} | 🎯 Target: ₹{target} | 🛑 SL: ₹{stop_loss}"

            send_telegram_message(
                f"📍 Spot: {underlying}\n"
                f"🔹 {atm_signal}\n"
                f"{suggested_trade}\n"
                f"Bias Score (ATM ±2): {total_score} ({market_view})\n"
                f"Level: {row['Level']}\n"
                f"📉 Support Zone: {support_str}\n"
                f"📈 Resistance Zone: {resistance_str}\n"
                f"Biases:\n"
                f"Strike: {row['Strike']}\n"
                f"ChgOI: {row['ChgOI_Bias']}, Volume: {row['Volume_Bias']}, Gamma: {row['Gamma_Bias']},\n"
                f"AskQty: {row['AskQty_Bias']}, BidQty: {row['BidQty_Bias']}, IV: {row['IV_Bias']}, DVP: {row['DVP_Bias']}"
            )

            st.session_state.trade_log.append({
                "Time": now.strftime("%H:%M:%S"),
                "Strike": row['Strike'],
                "Type": option_type,
                "LTP": ltp,
                "Target": target,
                "SL": stop_loss
            })

            signal_sent = True
            break

        if not signal_sent and atm_row is not None:
            send_telegram_message(
                f"📍 Spot: {underlying}\n"
                f"{market_view} — No Signal 🚫 (Spot not in valid zone or direction mismatch)\n"
                f"Bias Score: {total_score} ({market_view})\n"
                f"Level: {atm_row['Level']}\n"
                f"📉 Support Zone: {support_str}\n"
                f"📈 Resistance Zone: {resistance_str}\n"
                f"Biases:\n"
                f"Strike: {atm_row['Strike']}\n"
                f"ChgOI: {atm_row['ChgOI_Bias']}, Volume: {atm_row['Volume_Bias']}, Gamma: {atm_row['Gamma_Bias']},\n"
                f"AskQty: {atm_row['AskQty_Bias']}, BidQty: {atm_row['BidQty_Bias']}, IV: {atm_row['IV_Bias']}, DVP: {atm_row['DVP_Bias']}"
            )

        # === Main Display ===
        st.markdown(f"### 📍 Spot Price: {underlying}")
        st.success(f"🧠 Market View: **{market_view}** Bias Score: {total_score}")
        st.markdown(f"### 🛡️ Support Zone: `{support_str}`")
        st.markdown(f"### 🚧 Resistance Zone: `{resistance_str}`")
        if suggested_trade:
            st.info(f"🔹 {atm_signal}\n{suggested_trade}")
        
        with st.expander("📊 Option Chain Summary (ATM ±5 Strikes)"):  # Modified label
            st.dataframe(df_summary)
        
        if st.session_state.trade_log:
            st.markdown("### 📜 Trade Log")
            st.dataframe(pd.DataFrame(st.session_state.trade_log))

        # === Enhanced Reversal Analysis ===
        st.markdown("---")
        st.markdown("## 🔄 Reversal Signals (ATM ±5 Strikes)")  # Modified label
        
        # Calculate reversal scores for all rows
        df['ReversalScore'], df['ReversalDirection'] = zip(*df.apply(reversal_score, axis=1))
        
        # Filter for ATM ±5 strikes for display (assuming 50pt strikes)
        display_strikes = df[
            (df['strikePrice'] >= atm_strike - 5*50) & 
            (df['strikePrice'] <= atm_strike + 5*50)
        ].sort_values('strikePrice')
        
        # Show reversal table in UI with color coding
        st.dataframe(
            display_strikes[['strikePrice', 'Zone', 'ReversalScore', 'ReversalDirection',  # Added Zone column
                            'changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                            'impliedVolatility_CE', 'impliedVolatility_PE']]
            .sort_values("ReversalScore", ascending=False)
            .style.apply(lambda x: ['color: green' if v == "UP" else 'color: red' if v == "DOWN" else '' 
                                  for v in x], subset=['ReversalDirection'])
        )
        
        # Check only ATM strike for Telegram alerts
        atm_reversal_data = df[df['strikePrice'] == atm_strike].iloc[0] if not df[df['strikePrice'] == atm_strike].empty else None
        
        if atm_reversal_data is not None and atm_reversal_data['ReversalScore'] >= 2:
            direction = atm_reversal_data['ReversalDirection']
            emoji = "⬆️" if direction == "UP" else "⬇️"
            
            send_telegram_message(
                f"🔄 ATM REVERSAL ALERT {emoji}\n"
                f"Strike: {atm_strike} (ATM)\n"
                f"Direction: {direction}\n"
                f"Strength: {atm_reversal_data['ReversalScore']}/3\n"
                f"CE ΔOI: {atm_reversal_data['changeinOpenInterest_CE']} (IV {atm_reversal_data['impliedVolatility_CE']}%)\n"
                f"PE ΔOI: {atm_reversal_data['changeinOpenInterest_PE']} (IV {atm_reversal_data['impliedVolatility_PE']}%)\n"
                f"Spot: {underlying}\n"
                f"Time: {now.strftime('%H:%M:%S')}"
            )

        # === Liquidity Zones ===
        st.markdown("## 💧 Liquidity Zones")
        spot_history = st.session_state.price_data['Spot'].tolist()
        liquidity_zones = detect_liquidity_zones(df, underlying, spot_history)
        
        if not liquidity_zones.empty:
            st.dataframe(liquidity_zones)
        else:
            st.warning("No significant liquidity zones detected")

        # === Enhanced Functions Display ===
        st.markdown("---")
        st.markdown("## 📈 Enhanced Features")
        
        # Enhanced Trade Log
        display_enhanced_trade_log()
        
        # Price Chart with Support/Resistance
        st.markdown("---")
        st.markdown("### 📊 Price Action Chart")
        plot_price_with_sr()
        
        # Export functionality
        st.markdown("---")
        st.markdown("### 📥 Data Export")
        if st.button("Prepare Excel Export"):
            st.session_state.export_data = True
        handle_export_data(df_summary, underlying)
        
        # Call Log Book
        st.markdown("---")
        display_call_log_book()
        
        # Auto update call log with current price
        auto_update_call_log(underlying)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        send_telegram_message(f"❌ Error: {str(e)}")

# === Main Function Call ===
if __name__ == "__main__":
    analyze()
```

Key changes made:
1. Modified the strike price filtering to only analyze and display ATM ±5 strikes (assuming 50pt strikes)
2. Added 'Zone' column showing ATM/OTM/ITM status in all relevant displays
3. Removed the specific Telegram message about expiry day detection
4. Updated display labels to clearly indicate we're showing ATM ±5 strikes
5. Kept all other functionality intact

The script will now focus only on the 11 strikes closest to ATM (5 above and 5 below) for both analysis and display.