"""
Enhanced Market Data Display Module
====================================

Renders all enhanced market data in comprehensive tabulated format for Streamlit
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any


def render_enhanced_market_data_tab(enhanced_data: Dict[str, Any]):
    """
    Render comprehensive enhanced market data in Bias Analysis Pro tab

    Args:
        enhanced_data: Dict from EnhancedMarketData.fetch_all_enhanced_data()
    """
    st.markdown("## 🌐 Enhanced Market Analysis")

    st.caption(f"📅 Last Updated: {enhanced_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    # Summary Cards at the top
    _render_summary_cards(enhanced_data)

    st.markdown("---")

    # Create tabs for different categories
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Summary",
        "⚡ India VIX",
        "🏢 Sector Rotation",
        "🌍 Global Markets",
        "💰 Intermarket",
        "🎯 Gamma Squeeze",
        "⏰ Intraday Timing",
        "📈 NIFTY Futures"  # NEW TAB
    ])

    with tab1:
        _render_summary_tab(enhanced_data)

    with tab2:
        _render_india_vix_tab(enhanced_data)

    with tab3:
        _render_sector_rotation_tab(enhanced_data)

    with tab4:
        _render_global_markets_tab(enhanced_data)

    with tab5:
        _render_intermarket_tab(enhanced_data)

    with tab6:
        _render_gamma_squeeze_tab(enhanced_data)

    with tab7:
        _render_intraday_timing_tab(enhanced_data)

    with tab8:
        _render_nifty_futures_tab()  # NEW TAB RENDER


def _render_summary_cards(enhanced_data: Dict[str, Any]):
    """Render summary metric cards"""
    summary = enhanced_data.get('summary', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sentiment = summary.get('overall_sentiment', 'NEUTRAL')
        if sentiment == 'BULLISH':
            st.markdown(f"""
            <div style='padding: 15px; background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
                        border-radius: 10px; text-align: center;'>
                <h3 style='margin: 0; color: white;'>🚀 {sentiment}</h3>
                <p style='margin: 5px 0 0 0; color: white; font-size: 12px;'>Overall Sentiment</p>
            </div>
            """, unsafe_allow_html=True)
        elif sentiment == 'BEARISH':
            st.markdown(f"""
            <div style='padding: 15px; background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
                        border-radius: 10px; text-align: center;'>
                <h3 style='margin: 0; color: white;'>📉 {sentiment}</h3>
                <p style='margin: 5px 0 0 0; color: white; font-size: 12px;'>Overall Sentiment</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='padding: 15px; background: linear-gradient(135deg, #ffa500 0%, #ff8c00 100%);
                        border-radius: 10px; text-align: center;'>
                <h3 style='margin: 0; color: white;'>⚖️ {sentiment}</h3>
                <p style='margin: 5px 0 0 0; color: white; font-size: 12px;'>Overall Sentiment</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        avg_score = summary.get('avg_score', 0)
        score_color = '#00ff88' if avg_score > 0 else '#ff4444' if avg_score < 0 else '#ffa500'
        st.markdown(f"""
        <div style='padding: 15px; background: #1e1e1e; border-radius: 10px; text-align: center;
                    border-left: 4px solid {score_color};'>
            <h3 style='margin: 0; color: {score_color};'>{avg_score:+.1f}</h3>
            <p style='margin: 5px 0 0 0; color: #888; font-size: 12px;'>Average Score</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        total_points = summary.get('total_data_points', 0)
        st.markdown(f"""
        <div style='padding: 15px; background: #1e1e1e; border-radius: 10px; text-align: center;
                    border-left: 4px solid #6495ED;'>
            <h3 style='margin: 0; color: #6495ED;'>{total_points}</h3>
            <p style='margin: 5px 0 0 0; color: #888; font-size: 12px;'>Data Points</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        bullish = summary.get('bullish_count', 0)
        bearish = summary.get('bearish_count', 0)
        neutral = summary.get('neutral_count', 0)

        st.markdown(f"""
        <div style='padding: 15px; background: #1e1e1e; border-radius: 10px; text-align: center;'>
            <div style='font-size: 14px; color: #888;'>
                <span style='color: #00ff88;'>🟢{bullish}</span> |
                <span style='color: #ff4444;'>🔴{bearish}</span> |
                <span style='color: #ffa500;'>🟡{neutral}</span>
            </div>
            <p style='margin: 5px 0 0 0; color: #888; font-size: 12px;'>Bullish | Bearish | Neutral</p>
        </div>
        """, unsafe_allow_html=True)


def _render_summary_tab(enhanced_data: Dict[str, Any]):
    """Render comprehensive summary table"""
    st.markdown("### 📊 All Data Sources - Quick Summary")

    # Prepare summary data
    summary_data = []

    # Add India VIX
    vix = enhanced_data.get('india_vix', {})
    if vix.get('success'):
        summary_data.append({
            'Category': 'Volatility',
            'Indicator': 'India VIX',
            'Value': f"{vix['value']:.2f}",
            'Sentiment': vix['sentiment'],
            'Bias': vix['bias'],
            'Score': f"{vix['score']:+.0f}",
            'Source': vix['source']
        })

    # Add Sectors
    for sector in enhanced_data.get('sector_indices', []):
        summary_data.append({
            'Category': 'Sector',
            'Indicator': sector['sector'],
            'Value': f"₹ {sector['last_price']:.2f}",
            'Sentiment': f"{sector['change_pct']:+.2f}%",
            'Bias': sector['bias'],
            'Score': f"{sector['score']:+.0f}",
            'Source': sector['source']
        })

    # Add Global Markets
    for market in enhanced_data.get('global_markets', []):
        summary_data.append({
            'Category': 'Global',
            'Indicator': market['market'],
            'Value': f"{market['last_price']:.2f}",
            'Sentiment': f"{market['change_pct']:+.2f}%",
            'Bias': market['bias'],
            'Score': f"{market['score']:+.0f}",
            'Source': 'Yahoo Finance'
        })

    # Add Intermarket
    for asset in enhanced_data.get('intermarket', []):
        summary_data.append({
            'Category': 'Intermarket',
            'Indicator': asset['asset'],
            'Value': f"{asset['last_price']:.2f}",
            'Sentiment': f"{asset['change_pct']:+.2f}%",
            'Bias': asset['bias'],
            'Score': f"{asset['score']:+.0f}",
            'Source': 'Yahoo Finance'
        })

    # Create DataFrame
    df = pd.DataFrame(summary_data)

    # Add emoji to bias column
    def add_bias_emoji(bias):
        bias_str = str(bias)
        if 'BULLISH' in bias_str or 'RISK ON' in bias_str:
            return f"🐂 {bias}"
        elif 'BEARISH' in bias_str or 'RISK OFF' in bias_str:
            return f"🐻 {bias}"
        else:
            return f"⚖️ {bias}"

    df['Bias'] = df['Bias'].apply(add_bias_emoji)

    # Display table
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_india_vix_tab(enhanced_data: Dict[str, Any]):
    """Render India VIX detailed analysis"""
    st.markdown("### ⚡ India VIX - Fear & Greed Index")

    vix = enhanced_data.get('india_vix', {})

    if not vix.get('success'):
        st.warning("India VIX data not available")
        return

    col1, col2 = st.columns(2)

    with col1:
        vix_value = vix['value']
        if vix_value > 25:
            color = '#ff4444'
        elif vix_value > 20:
            color = '#ff8844'
        elif vix_value > 15:
            color = '#ffa500'
        else:
            color = '#00ff88'

        st.markdown(f"""
        <div style='padding: 30px; background: linear-gradient(135deg, {color} 0%, {color}CC 100%);
                    border-radius: 15px; text-align: center;'>
            <h1 style='margin: 0; color: white; font-size: 48px;'>{vix_value:.2f}</h1>
            <p style='margin: 10px 0 0 0; color: white; font-size: 18px;'>{vix['sentiment']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        **Bias:** {vix['bias']}
        **Score:** {vix['score']:+.0f}
        **Source:** {vix['source']}

        **Interpretation:**
        - VIX < 12: Complacency (Neutral)
        - VIX 12-15: Low Volatility (Bullish)
        - VIX 15-20: Moderate Volatility (Neutral)
        - VIX 20-25: Elevated Fear (Bearish)
        - VIX > 25: High Fear (Strong Bearish)
        """)

    st.info("""
    **💡 Trading Insight:**
    - High VIX (>20): Market expects large price swings → Good for option sellers, risky for directional trades
    - Low VIX (<15): Market is calm → Good for trend following, option buying becomes expensive
    """)


def _render_sector_rotation_tab(enhanced_data: Dict[str, Any]):
    """Render sector rotation analysis"""
    st.markdown("### 🏢 Sector Rotation Analysis")

    rotation = enhanced_data.get('sector_rotation', {})

    if not rotation.get('success'):
        st.warning("Sector rotation data not available")
        return

    # Summary cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rotation Pattern", rotation['rotation_pattern'])
        st.metric("Rotation Type", rotation['rotation_type'])

    with col2:
        st.metric("Sector Breadth %", f"{rotation['sector_breadth']:.1f}%",
                 help="Percentage of sectors showing positive momentum")
        st.metric("Sector Breadth Sentiment", rotation['sector_sentiment'])

    with col3:
        st.metric("Rotation Bias", rotation['rotation_bias'])
        st.metric("Rotation Score", f"{rotation['rotation_score']:+.0f}")

    st.markdown("---")

    # Leaders and Laggards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Top Performing Sectors")
        leaders = rotation['leaders']
        if leaders:
            leaders_df = pd.DataFrame(leaders)
            leaders_df = leaders_df[['sector', 'change_pct', 'bias']]
            leaders_df['change_pct'] = leaders_df['change_pct'].apply(lambda x: f"{x:+.2f}%")
            leaders_df.columns = ['Sector', 'Change %', 'Bias']
            st.dataframe(leaders_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 📉 Lagging Sectors")
        laggards = rotation['laggards']
        if laggards:
            laggards_df = pd.DataFrame(laggards)
            laggards_df = laggards_df[['sector', 'change_pct', 'bias']]
            laggards_df['change_pct'] = laggards_df['change_pct'].apply(lambda x: f"{x:+.2f}%")
            laggards_df.columns = ['Sector', 'Change %', 'Bias']
            st.dataframe(laggards_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # All sectors table
    st.markdown("#### 📊 All Sectors Performance")
    all_sectors = rotation['all_sectors']
    if all_sectors:
        sectors_df = pd.DataFrame(all_sectors)
        sectors_df = sectors_df[['sector', 'last_price', 'change_pct', 'bias', 'score']]
        sectors_df['last_price'] = sectors_df['last_price'].apply(lambda x: f"₹ {x:.2f}")
        sectors_df['change_pct'] = sectors_df['change_pct'].apply(lambda x: f"{x:+.2f}%")
        sectors_df['score'] = sectors_df['score'].apply(lambda x: f"{x:+.0f}")
        sectors_df.columns = ['Sector', 'Last Price', 'Change %', 'Bias', 'Score']
        st.dataframe(sectors_df, use_container_width=True, hide_index=True)


def _render_global_markets_tab(enhanced_data: Dict[str, Any]):
    """Render global markets data"""
    st.markdown("### 🌍 Global Markets Sentiment")

    markets = enhanced_data.get('global_markets', [])

    if not markets:
        st.warning("Global markets data not available")
        return

    # Create DataFrame
    markets_df = pd.DataFrame(markets)
    markets_df = markets_df[['market', 'last_price', 'prev_close', 'change_pct', 'bias', 'score']]
    markets_df['last_price'] = markets_df['last_price'].apply(lambda x: f"{x:.2f}")
    markets_df['prev_close'] = markets_df['prev_close'].apply(lambda x: f"{x:.2f}")
    markets_df['change_pct'] = markets_df['change_pct'].apply(lambda x: f"{x:+.2f}%")
    markets_df['score'] = markets_df['score'].apply(lambda x: f"{x:+.0f}")

    # Add emoji to bias
    markets_df['bias'] = markets_df['bias'].apply(lambda x:
        f"🐂 {x}" if 'BULLISH' in x else f"🐻 {x}" if 'BEARISH' in x else f"⚖️ {x}"
    )

    markets_df.columns = ['Market', 'Last Price', 'Prev Close', 'Change %', 'Bias', 'Score']

    st.dataframe(markets_df, use_container_width=True, hide_index=True)

    st.info("""
    **💡 Global Market Correlation:**
    - Indian markets typically follow US markets (S&P 500, Nasdaq)
    - Asian markets (Nikkei, Hang Seng) show regional trends
    - Watch overnight US futures for next day Indian market direction
    """)


def _render_intermarket_tab(enhanced_data: Dict[str, Any]):
    """Render intermarket analysis"""
    st.markdown("### 💰 Intermarket Analysis")

    intermarket = enhanced_data.get('intermarket', [])

    if not intermarket:
        st.warning("Intermarket data not available")
        return

    # Create DataFrame
    intermarket_df = pd.DataFrame(intermarket)
    intermarket_df = intermarket_df[['asset', 'last_price', 'prev_close', 'change_pct', 'bias', 'score']]
    intermarket_df['last_price'] = intermarket_df['last_price'].apply(lambda x: f"{x:.2f}")
    intermarket_df['prev_close'] = intermarket_df['prev_close'].apply(lambda x: f"{x:.2f}")
    intermarket_df['change_pct'] = intermarket_df['change_pct'].apply(lambda x: f"{x:+.2f}%")
    intermarket_df['score'] = intermarket_df['score'].apply(lambda x: f"{x:+.0f}")

    intermarket_df.columns = ['Asset', 'Last Price', 'Prev Close', 'Change %', 'Bias', 'Score']

    st.dataframe(intermarket_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("#### 💡 Intermarket Relationships")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **US Dollar Index (DXY):**
        - Strong USD → Bearish for India (FII outflows)
        - Weak USD → Bullish for India (FII inflows)

        **Crude Oil:**
        - Rising Oil → Bearish for India (import dependent)
        - Falling Oil → Bullish for India (lower inflation)

        **Gold:**
        - Rising Gold → Risk-off sentiment (Bearish)
        - Falling Gold → Risk-on sentiment (Bullish)
        """)

    with col2:
        st.markdown("""
        **USD/INR:**
        - Rising → INR weakening (Bearish)
        - Falling → INR strengthening (Bullish)

        **US 10Y Treasury:**
        - Rising Yields → Risk-off (Bearish for EM)
        - Falling Yields → Risk-on (Bullish for EM)

        **Bitcoin:**
        - Rising → Risk appetite high
        - Falling → Risk appetite low
        """)


def _render_gamma_squeeze_tab(enhanced_data: Dict[str, Any]):
    """Render gamma squeeze analysis"""
    st.markdown("### 🎯 Gamma Squeeze Detection")

    gamma = enhanced_data.get('gamma_squeeze', {})

    if not gamma.get('success'):
        st.warning(f"Gamma squeeze data not available: {gamma.get('error', 'Unknown error')}")
        return

    # Summary cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Instrument", gamma['instrument'])
        st.metric("Spot Price", f"₹ {gamma['spot']:.2f}")

    with col2:
        st.metric("Net Gamma", f"{gamma['net_gamma']:,.0f}")
        st.metric("Squeeze Risk", gamma['squeeze_risk'])

    with col3:
        st.metric("Squeeze Bias", gamma['squeeze_bias'])
        st.metric("Squeeze Score", f"{gamma['squeeze_score']:+.0f}")

    st.markdown("---")

    # Gamma details
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        **Call Gamma:** {gamma['total_call_gamma']:,.0f}
        **Put Gamma:** {gamma['total_put_gamma']:,.0f}
        **Net Gamma:** {gamma['net_gamma']:,.0f}
        """)

    with col2:
        st.info(gamma['interpretation'])

    st.markdown("---")

    st.markdown("#### 💡 Understanding Gamma Squeeze")
    st.markdown("""
    **What is Gamma Squeeze?**
    When market makers (MMs) have large gamma exposure, they must continuously hedge by buying/selling the underlying.

    **Positive Gamma (Long Gamma):**
    - MMs will buy when price drops, sell when price rises
    - Creates resistance to large moves (stabilizing)
    - Current market likely to stay range-bound

    **Negative Gamma (Short Gamma):**
    - MMs will sell when price drops, buy when price rises
    - Amplifies price movements (destabilizing)
    - Risk of rapid directional moves

    **High Gamma Concentration at ATM:**
    - Strong support/resistance at current levels
    - Breakout from this level can trigger squeeze
    """)


def _render_intraday_timing_tab(enhanced_data: Dict[str, Any]):
    """Render intraday timing analysis"""
    st.markdown("### ⏰ Intraday Timing & Seasonality")

    timing = enhanced_data.get('intraday_seasonality', {})

    if not timing.get('success'):
        st.warning("Intraday timing data not available")
        return

    # Current session
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style='padding: 20px; background: linear-gradient(135deg, #6495ED 0%, #4169E1 100%);
                    border-radius: 15px; text-align: center;'>
            <h2 style='margin: 0; color: white;'>{timing['current_time']}</h2>
            <p style='margin: 10px 0 0 0; color: white;'>{timing['session']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        bias = timing['session_bias']
        bias_color = '#00ff88' if 'TREND' in bias else '#ff8844' if 'VOLATILITY' in bias else '#ffa500'

        st.markdown(f"""
        <div style='padding: 20px; background: {bias_color}; border-radius: 15px; text-align: center;'>
            <h3 style='margin: 0; color: white;'>{bias}</h3>
            <p style='margin: 10px 0 0 0; color: white; font-size: 12px;'>Session Bias</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='padding: 20px; background: #1e1e1e; border-radius: 15px; text-align: center;
                    border-left: 4px solid #6495ED;'>
            <h3 style='margin: 0; color: #6495ED;'>{timing['session_score']:+.0f}</h3>
            <p style='margin: 10px 0 0 0; color: #888; font-size: 12px;'>Session Score</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Session characteristics
    st.markdown("#### 📊 Current Session Characteristics")
    st.info(timing['session_characteristics'])

    st.markdown("#### 💡 Trading Recommendation")
    st.success(timing['trading_recommendation'])

    st.markdown("---")

    # Day of week analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        **Today:** {timing['weekday']}
        **Day Bias:** {timing['day_bias']}
        """)

    with col2:
        st.markdown(f"""
        **Day Characteristics:**
        {timing['day_characteristics']}
        """)

    st.markdown("---")

    # Intraday session guide
    st.markdown("#### 🕐 Intraday Session Guide")

    session_guide = pd.DataFrame([
        {'Time': '9:15-9:30', 'Session': 'Opening Range', 'Characteristics': 'High volatility, gap movements', 'Recommendation': 'CAUTIOUS'},
        {'Time': '9:30-10:00', 'Session': 'Post-Opening', 'Characteristics': 'Trend develops', 'Recommendation': 'ACTIVE'},
        {'Time': '10:00-11:30', 'Session': 'Mid-Morning', 'Characteristics': 'Best trending period', 'Recommendation': 'VERY ACTIVE'},
        {'Time': '11:30-14:30', 'Session': 'Lunchtime', 'Characteristics': 'Low volume, choppy', 'Recommendation': 'REDUCE ACTIVITY'},
        {'Time': '14:30-15:15', 'Session': 'Afternoon', 'Characteristics': 'Volume picks up, trends resume', 'Recommendation': 'ACTIVE'},
        {'Time': '15:15-15:30', 'Session': 'Closing Range', 'Characteristics': 'High volatility, squaring off', 'Recommendation': 'CAUTIOUS'},
    ])

    st.dataframe(session_guide, use_container_width=True, hide_index=True)


def _render_nifty_futures_tab():
    """Render NIFTY Futures Analysis tab"""
    st.markdown("### 📈 NIFTY Futures Analysis")
    st.caption("Institutional positioning via futures premium/discount and OI analysis")

    # Get futures data from session state
    futures_data = None
    if 'nifty_screener_data' in st.session_state:
        screener_data = st.session_state.nifty_screener_data
        if isinstance(screener_data, dict):
            futures_data = screener_data.get('futures_analysis')

    if not futures_data:
        st.warning("""
        ⚠️ **NIFTY Futures data not available**

        **How to enable:**
        1. Go to Tab 8 (NIFTY Option Screener v7.0)
        2. Run the analysis
        3. Futures data will be calculated and displayed here

        **What you'll see:**
        - Futures vs Spot premium/discount %
        - Premium bias (BULLISH/BEARISH/NEUTRAL)
        - Futures OI bias
        - Combined futures bias
        - Institutional positioning insights
        """)
        return

    # ==========================================
    # FUTURES OVERVIEW CARDS
    # ==========================================
    st.markdown("#### 📊 Futures Premium/Discount Analysis")

    col1, col2, col3, col4 = st.columns(4)

    # Premium/Discount %
    with col1:
        premium_pct = futures_data.get('premium_pct', 0)
        premium_abs = abs(premium_pct) * 100

        if premium_pct > 0:
            st.metric(
                label="Futures Premium",
                value=f"+{premium_abs:.2f}%",
                delta="Bullish bias",
                delta_color="normal"
            )
        elif premium_pct < 0:
            st.metric(
                label="Futures Discount",
                value=f"{premium_abs:.2f}%",
                delta="Bearish bias",
                delta_color="inverse"
            )
        else:
            st.metric(
                label="Futures Premium",
                value="0.00%",
                delta="Neutral"
            )

    # Premium Bias
    with col2:
        premium_bias = futures_data.get('premium_bias', 'NEUTRAL')
        bias_emoji = "🟢" if premium_bias == "BULLISH" else ("🔴" if premium_bias == "BEARISH" else "⚖️")

        st.metric(
            label="Premium Bias",
            value=f"{bias_emoji} {premium_bias}"
        )

    # Futures OI Bias
    with col3:
        oi_bias = futures_data.get('oi_bias', 'NEUTRAL')
        oi_emoji = "🟢" if oi_bias == "BULLISH" else ("🔴" if oi_bias == "BEARISH" else "⚖️")

        st.metric(
            label="Futures OI Bias",
            value=f"{oi_emoji} {oi_bias}"
        )

    # Combined Bias
    with col4:
        combined_bias = futures_data.get('combined_bias', 'NEUTRAL')
        combined_emoji = "🟢" if combined_bias == "BULLISH" else ("🔴" if combined_bias == "BEARISH" else "⚖️")
        confidence = futures_data.get('confidence', 50)

        st.metric(
            label="Combined Bias",
            value=f"{combined_emoji} {combined_bias}",
            delta=f"{confidence}% confidence"
        )

    st.markdown("---")

    # ==========================================
    # DETAILED FUTURES DATA
    # ==========================================
    st.markdown("#### 📋 Detailed Futures Data")

    # Create detailed data table
    futures_details = []

    # Spot Price
    spot_price = futures_data.get('spot_price', 0)
    futures_price = futures_data.get('futures_price', 0)

    futures_details.append({
        'Metric': 'NIFTY Spot Price',
        'Value': f"₹{spot_price:,.2f}" if spot_price else 'N/A',
        'Description': 'Current spot market price'
    })

    futures_details.append({
        'Metric': 'NIFTY Futures Price',
        'Value': f"₹{futures_price:,.2f}" if futures_price else 'N/A',
        'Description': 'Current month futures price'
    })

    futures_details.append({
        'Metric': 'Premium/Discount',
        'Value': f"{premium_pct*100:+.2f}%",
        'Description': 'Futures vs Spot (positive = premium, negative = discount)'
    })

    futures_details.append({
        'Metric': 'Premium Bias',
        'Value': f"{bias_emoji} {premium_bias}",
        'Description': 'Bias based on premium/discount level'
    })

    futures_details.append({
        'Metric': 'Futures OI Change',
        'Value': futures_data.get('oi_change_pct', 'N/A'),
        'Description': 'Open Interest % change'
    })

    futures_details.append({
        'Metric': 'OI Bias',
        'Value': f"{oi_emoji} {oi_bias}",
        'Description': 'Bias based on OI buildup/reduction'
    })

    futures_details.append({
        'Metric': 'Combined Signal',
        'Value': f"{combined_emoji} {combined_bias}",
        'Description': 'Overall futures market positioning'
    })

    futures_details.append({
        'Metric': 'Signal Confidence',
        'Value': f"{confidence}%",
        'Description': 'Reliability of combined signal'
    })

    futures_df = pd.DataFrame(futures_details)
    st.dataframe(futures_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ==========================================
    # INTERPRETATION GUIDE
    # ==========================================
    st.markdown("#### 💡 Interpretation Guide")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Premium/Discount Analysis:**

        🟢 **Premium (Positive):**
        - Futures trading above spot
        - Institutions building long positions
        - Bullish market expectation
        - Higher the premium, stronger the bullish bias

        🔴 **Discount (Negative):**
        - Futures trading below spot
        - Institutions building short positions
        - Bearish market expectation
        - Higher the discount, stronger the bearish bias

        ⚖️ **Neutral (Near Zero):**
        - Futures aligned with spot
        - No strong directional bias
        - Market in equilibrium
        """)

    with col2:
        st.markdown("""
        **OI Bias Analysis:**

        🟢 **Bullish OI:**
        - OI increasing + Premium = Long buildup
        - Strong institutional buying
        - Expect upward move

        🔴 **Bearish OI:**
        - OI increasing + Discount = Short buildup
        - Strong institutional selling
        - Expect downward move

        ⚖️ **Neutral OI:**
        - OI stable or mixed signals
        - No clear positioning
        - Wait for clarity

        **Key Thresholds:**
        - Premium >0.5%: Significantly bullish
        - Discount <-0.5%: Significantly bearish
        - Premium between -0.5% to +0.5%: Neutral
        """)

    st.markdown("---")

    # ==========================================
    # TRADING IMPLICATIONS
    # ==========================================
    st.markdown("#### 🎯 Trading Implications")

    if combined_bias == "BULLISH":
        st.success(f"""
        **🟢 BULLISH Futures Positioning ({confidence}% confidence)**

        - Institutions are net long via futures market
        - Premium suggests expectation of higher prices
        - Consider LONG positions on pullbacks to support
        - Use futures premium to gauge institutional conviction
        - Strong bullish signal when combined with OI buildup
        """)
    elif combined_bias == "BEARISH":
        st.error(f"""
        **🔴 BEARISH Futures Positioning ({confidence}% confidence)**

        - Institutions are net short via futures market
        - Discount suggests expectation of lower prices
        - Consider SHORT positions on rallies to resistance
        - Use futures discount to gauge institutional conviction
        - Strong bearish signal when combined with OI buildup
        """)
    else:
        st.info(f"""
        **⚖️ NEUTRAL Futures Positioning ({confidence}% confidence)**

        - No clear institutional directional bias
        - Futures aligned with spot market
        - Market in consolidation/range
        - Wait for clear bias before taking directional trades
        - Focus on range-bound strategies
        """)

    st.markdown("---")

    st.caption("""
    **Note:** Futures data is sourced from Option Screener analysis.
    For most accurate data, ensure Option Screener (Tab 8) analysis is recently run.
    Futures analysis is integrated into ML Entry Finder with 20% weight.
    """)
