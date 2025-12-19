"""
Signal Tracker - Records all trading signals to Supabase and tracks performance

This module handles:
1. Saving entry signals to Supabase with all details
2. Auto-updating signals when targets/SL are hit
3. Performance tracking and analytics
4. Display tabulation of all signals with outcomes
"""

import streamlit as st
from supabase import create_client, Client
from datetime import datetime, timedelta
import pytz
import pandas as pd
from typing import Dict, Optional, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# SUPABASE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

def get_supabase_client() -> Optional[Client]:
    """Get Supabase client from credentials"""
    try:
        SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
        SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "")

        if SUPABASE_URL and SUPABASE_ANON_KEY:
            return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        else:
            logger.warning("Supabase credentials not found")
            return None
    except Exception as e:
        logger.error(f"Error creating Supabase client: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# SIGNAL RECORDING
# ═══════════════════════════════════════════════════════════════════

def save_entry_signal(
    signal_type: str,  # LONG or SHORT
    entry_price: float,
    stop_loss: float,
    target1: float,
    target2: float,
    support_level: float,
    resistance_level: float,
    entry_reason: str,
    current_price: float,
    confidence: float = 0.0,
    source: str = "VOB",  # VOB, HTF, Confluence, etc.
    additional_data: Optional[Dict] = None
) -> Optional[str]:
    """
    Save entry signal to Supabase

    Returns:
        signal_id if successful, None otherwise
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            logger.warning("Supabase not available - signal not saved")
            return None

        # Get IST timestamp
        ist = pytz.timezone('Asia/Kolkata')
        entry_time = datetime.now(ist)

        # Prepare signal data
        signal_data = {
            'signal_type': signal_type,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target1': target1,
            'target2': target2,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'entry_reason': entry_reason,
            'current_price_at_entry': current_price,
            'confidence': confidence,
            'source': source,
            'status': 'ACTIVE',
            'entry_time': entry_time.isoformat(),
            'created_at': entry_time.isoformat(),
            'updated_at': entry_time.isoformat()
        }

        # Add additional data if provided
        if additional_data:
            signal_data.update(additional_data)

        # Insert to Supabase
        result = supabase.table('trading_signals').insert(signal_data).execute()

        if result.data and len(result.data) > 0:
            signal_id = result.data[0].get('id')
            logger.info(f"Signal saved successfully: {signal_id}")
            return str(signal_id)
        else:
            logger.error("No data returned from Supabase insert")
            return None

    except Exception as e:
        logger.error(f"Error saving signal to Supabase: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# SIGNAL MONITORING & AUTO-UPDATE
# ═══════════════════════════════════════════════════════════════════

def update_active_signals(current_price: float, market_data: Optional[Dict] = None) -> Dict:
    """
    Check all active signals and update if targets/SL hit
    Uses ML analysis to determine WHY exit occurred

    Args:
        current_price: Current market price
        market_data: Optional dict with ml_regime, money_flow_signals, deltaflow_signals, etc.

    Returns:
        Dict with update statistics
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return {'updated': 0, 'errors': 0}

        # Get all active signals
        result = supabase.table('trading_signals')\
            .select('*')\
            .eq('status', 'ACTIVE')\
            .execute()

        if not result.data:
            return {'updated': 0, 'errors': 0}

        updated_count = 0
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)

        for signal in result.data:
            signal_id = signal['id']
            signal_type = signal['signal_type']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            target1 = signal['target1']
            target2 = signal['target2']

            new_status = None
            exit_price = None

            if signal_type == 'LONG':
                # Check if targets hit (for LONG)
                if current_price >= target2:
                    new_status = 'HIT_TARGET2'
                    exit_price = target2
                elif current_price >= target1:
                    new_status = 'HIT_TARGET1'
                    exit_price = target1
                elif current_price <= stop_loss:
                    new_status = 'HIT_SL'
                    exit_price = stop_loss

            elif signal_type == 'SHORT':
                # Check if targets hit (for SHORT)
                if current_price <= target2:
                    new_status = 'HIT_TARGET2'
                    exit_price = target2
                elif current_price <= target1:
                    new_status = 'HIT_TARGET1'
                    exit_price = target1
                elif current_price >= stop_loss:
                    new_status = 'HIT_SL'
                    exit_price = stop_loss

            # Update signal if status changed
            if new_status:
                profit_loss = exit_price - entry_price if signal_type == 'LONG' else entry_price - exit_price
                profit_loss_pct = (profit_loss / entry_price) * 100

                # Analyze WHY exit occurred using ML
                exit_reason = analyze_exit_reason(
                    signal_data=signal,
                    exit_price=exit_price,
                    exit_status=new_status,
                    current_market_data=market_data
                )

                update_data = {
                    'status': new_status,
                    'exit_price': exit_price,
                    'exit_time': current_time.isoformat(),
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'exit_reason': exit_reason,
                    'updated_at': current_time.isoformat()
                }

                supabase.table('trading_signals')\
                    .update(update_data)\
                    .eq('id', signal_id)\
                    .execute()

                updated_count += 1
                logger.info(f"Signal {signal_id} updated to {new_status}: {exit_reason}")

        return {'updated': updated_count, 'errors': 0}

    except Exception as e:
        logger.error(f"Error updating active signals: {e}")
        return {'updated': 0, 'errors': 1}


# ═══════════════════════════════════════════════════════════════════
# ML-BASED EXIT REASON ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_exit_reason(
    signal_data: Dict,
    exit_price: float,
    exit_status: str,
    current_market_data: Optional[Dict] = None
) -> str:
    """
    Use ML and market analysis to determine WHY a target or stop loss was hit

    Args:
        signal_data: Original signal data from database
        exit_price: Price at which exit occurred
        exit_status: HIT_TARGET1/HIT_TARGET2/HIT_SL
        current_market_data: Current market conditions (regime, volume, flow, etc.)

    Returns:
        Detailed exit reason string
    """
    try:
        reasons = []

        # Get signal details
        signal_type = signal_data.get('signal_type', 'LONG')
        entry_price = signal_data.get('entry_price', 0)
        support_level = signal_data.get('support_level', 0)
        resistance_level = signal_data.get('resistance_level', 0)

        # ========== 1. MARKET REGIME ANALYSIS ==========
        if current_market_data:
            ml_regime = current_market_data.get('ml_regime')
            if ml_regime:
                regime = ml_regime.get('regime', 'Unknown')
                trend_strength = ml_regime.get('trend_strength', 0)

                if exit_status in ['HIT_TARGET1', 'HIT_TARGET2']:
                    # Target hit - WHY?
                    if signal_type == 'LONG':
                        if 'Trending Up' in regime:
                            reasons.append(f"✅ Market Regime: {regime} (Strength: {trend_strength:.1f}%) - Strong uptrend continuation")
                        elif 'Breakout' in regime:
                            reasons.append(f"✅ Volatile Breakout detected - Price surged through resistance")
                        elif 'Range Bound' in regime:
                            reasons.append(f"⚠️ Range Bound market - Target hit at range boundary, potential reversal zone")
                        else:
                            reasons.append(f"✅ Regime: {regime} - Favorable conditions for LONG")
                    else:  # SHORT
                        if 'Trending Down' in regime:
                            reasons.append(f"✅ Market Regime: {regime} (Strength: {trend_strength:.1f}%) - Strong downtrend continuation")
                        elif 'Range Bound' in regime:
                            reasons.append(f"⚠️ Range Bound market - Target hit at range boundary")
                        else:
                            reasons.append(f"✅ Regime: {regime} - Favorable conditions for SHORT")

                else:  # HIT_SL
                    # Stop loss hit - WHY?
                    if signal_type == 'LONG':
                        if 'Trending Down' in regime:
                            reasons.append(f"❌ Market Regime: {regime} - Trend reversed against position")
                        elif 'Volatile Breakout' in regime:
                            reasons.append(f"❌ Volatile Breakout - Sharp reversal broke support")
                        elif 'Range Bound' in regime:
                            reasons.append(f"⚠️ Range Bound - Failed to break resistance, reversed to support")
                        else:
                            reasons.append(f"❌ Regime: {regime} - Unfavorable shift")
                    else:  # SHORT
                        if 'Trending Up' in regime:
                            reasons.append(f"❌ Market Regime: {regime} - Trend reversed against position")
                        elif 'Range Bound' in regime:
                            reasons.append(f"⚠️ Range Bound - Failed to break support, bounced to resistance")
                        else:
                            reasons.append(f"❌ Regime: {regime} - Unfavorable shift")

        # ========== 2. VOLUME ANALYSIS ==========
        if current_market_data:
            money_flow = current_market_data.get('money_flow_signals', {})
            if money_flow and money_flow.get('success'):
                total_volume = money_flow.get('total_volume', 0)
                bullish_pct = money_flow.get('bullish_volume_pct', 0)
                bearish_pct = money_flow.get('bearish_volume_pct', 0)

                if exit_status in ['HIT_TARGET1', 'HIT_TARGET2']:
                    if signal_type == 'LONG' and bullish_pct > 60:
                        reasons.append(f"📊 Volume Support: {bullish_pct:.1f}% bullish volume - Strong institutional buying")
                    elif signal_type == 'SHORT' and bearish_pct > 60:
                        reasons.append(f"📊 Volume Support: {bearish_pct:.1f}% bearish volume - Strong institutional selling")
                    else:
                        reasons.append(f"📊 Volume Profile: {bullish_pct:.1f}% bullish, {bearish_pct:.1f}% bearish")
                else:  # HIT_SL
                    if signal_type == 'LONG' and bearish_pct > 60:
                        reasons.append(f"❌ Volume Reversal: {bearish_pct:.1f}% bearish volume - Institutional selling pressure")
                    elif signal_type == 'SHORT' and bullish_pct > 60:
                        reasons.append(f"❌ Volume Reversal: {bullish_pct:.1f}% bullish volume - Institutional buying pressure")

        # ========== 3. DELTA FLOW ANALYSIS ==========
        if current_market_data:
            deltaflow = current_market_data.get('deltaflow_signals', {})
            if deltaflow and deltaflow.get('success'):
                overall_delta = deltaflow.get('overall_delta', 0)
                sentiment = deltaflow.get('sentiment', 'NEUTRAL')

                if exit_status in ['HIT_TARGET1', 'HIT_TARGET2']:
                    if signal_type == 'LONG' and overall_delta > 0:
                        reasons.append(f"🔄 Delta Flow: +{overall_delta:,.0f} ({sentiment}) - Positive orderflow momentum")
                    elif signal_type == 'SHORT' and overall_delta < 0:
                        reasons.append(f"🔄 Delta Flow: {overall_delta:,.0f} ({sentiment}) - Negative orderflow momentum")
                else:  # HIT_SL
                    if signal_type == 'LONG' and overall_delta < 0:
                        reasons.append(f"❌ Delta Flow Reversal: {overall_delta:,.0f} ({sentiment}) - Orderflow turned negative")
                    elif signal_type == 'SHORT' and overall_delta > 0:
                        reasons.append(f"❌ Delta Flow Reversal: +{overall_delta:,.0f} ({sentiment}) - Orderflow turned positive")

        # ========== 4. SUPPORT/RESISTANCE ANALYSIS ==========
        if exit_status in ['HIT_TARGET1', 'HIT_TARGET2']:
            # Target hit - did it break through resistance/support?
            if signal_type == 'LONG' and resistance_level > 0:
                if exit_price >= resistance_level:
                    reasons.append(f"🚀 Resistance Break: Price broke ₹{resistance_level:.0f} resistance level")
                else:
                    reasons.append(f"✅ Target reached at ₹{exit_price:.2f} (below resistance ₹{resistance_level:.0f})")
            elif signal_type == 'SHORT' and support_level > 0:
                if exit_price <= support_level:
                    reasons.append(f"📉 Support Break: Price broke ₹{support_level:.0f} support level")
                else:
                    reasons.append(f"✅ Target reached at ₹{exit_price:.2f} (above support ₹{support_level:.0f})")
        else:  # HIT_SL
            # Stop loss hit - support/resistance held
            if signal_type == 'LONG' and support_level > 0:
                if exit_price <= support_level:
                    reasons.append(f"❌ Support Failed: Price broke support at ₹{support_level:.0f}")
                else:
                    reasons.append(f"❌ Stop loss hit at ₹{exit_price:.2f} (support at ₹{support_level:.0f} failed to hold)")
            elif signal_type == 'SHORT' and resistance_level > 0:
                if exit_price >= resistance_level:
                    reasons.append(f"❌ Resistance Held: Price bounced from resistance at ₹{resistance_level:.0f}")
                else:
                    reasons.append(f"❌ Stop loss hit at ₹{exit_price:.2f} (resistance at ₹{resistance_level:.0f} held)")

        # ========== 5. OPTION CHAIN ANALYSIS ==========
        if current_market_data:
            atm_bias = current_market_data.get('atm_bias_data', {})
            if atm_bias:
                verdict = atm_bias.get('verdict', 'NEUTRAL')

                if exit_status in ['HIT_TARGET1', 'HIT_TARGET2']:
                    if (signal_type == 'LONG' and 'BULLISH' in verdict) or (signal_type == 'SHORT' and 'BEARISH' in verdict):
                        reasons.append(f"✅ ATM Options: {verdict} - Option flow confirms direction")
                else:  # HIT_SL
                    if (signal_type == 'LONG' and 'BEARISH' in verdict) or (signal_type == 'SHORT' and 'BULLISH' in verdict):
                        reasons.append(f"❌ ATM Options: {verdict} - Option flow against position")

        # ========== 6. VOLATILITY CONTEXT ==========
        if current_market_data:
            volatility = current_market_data.get('volatility_result')
            if volatility:
                vix_level = volatility.get('vix_level', 0)
                regime = volatility.get('regime', 'Normal')

                if vix_level > 20:
                    reasons.append(f"⚠️ High Volatility: VIX at {vix_level:.1f} - {regime} - Sharp price movements")
                elif vix_level < 12:
                    reasons.append(f"📊 Low Volatility: VIX at {vix_level:.1f} - Calm market conditions")

        # ========== 7. PRICE MOVEMENT ANALYSIS ==========
        price_move = abs(exit_price - entry_price)
        price_move_pct = (price_move / entry_price) * 100

        if exit_status in ['HIT_TARGET1', 'HIT_TARGET2']:
            reasons.append(f"💰 Price Movement: ₹{price_move:.2f} ({price_move_pct:.2f}%) from entry ₹{entry_price:.2f} to exit ₹{exit_price:.2f}")
        else:
            reasons.append(f"⛔ Price Movement: ₹{price_move:.2f} ({price_move_pct:.2f}%) adverse move from entry ₹{entry_price:.2f} to exit ₹{exit_price:.2f}")

        # Combine all reasons
        if reasons:
            exit_reason = " | ".join(reasons)
        else:
            exit_reason = f"Exit at ₹{exit_price:.2f} - No detailed market data available for analysis"

        return exit_reason

    except Exception as e:
        logger.error(f"Error analyzing exit reason: {e}")
        return f"Exit at ₹{exit_price:.2f} - Analysis error: {str(e)}"


# ═══════════════════════════════════════════════════════════════════
# SIGNAL RETRIEVAL & ANALYTICS
# ═══════════════════════════════════════════════════════════════════

def get_all_signals(limit: int = 100, status_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve signals from Supabase

    Args:
        limit: Maximum number of signals to retrieve
        status_filter: Filter by status (ACTIVE, HIT_TARGET1, HIT_TARGET2, HIT_SL, CLOSED)

    Returns:
        DataFrame with all signals
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return pd.DataFrame()

        # Build query
        query = supabase.table('trading_signals')\
            .select('*')\
            .order('entry_time', desc=True)\
            .limit(limit)

        if status_filter:
            query = query.eq('status', status_filter)

        result = query.execute()

        if result.data:
            return pd.DataFrame(result.data)
        else:
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error retrieving signals: {e}")
        return pd.DataFrame()


def get_performance_metrics() -> Dict:
    """
    Calculate performance metrics from all closed signals

    Returns:
        Dict with performance stats
    """
    try:
        df = get_all_signals(limit=1000)

        if df.empty:
            return {
                'total_signals': 0,
                'active_signals': 0,
                'closed_signals': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_profit_loss': 0
            }

        # Filter closed signals (not ACTIVE)
        closed_df = df[df['status'] != 'ACTIVE'].copy()

        if closed_df.empty:
            return {
                'total_signals': len(df),
                'active_signals': len(df),
                'closed_signals': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_profit_loss': 0
            }

        # Calculate metrics
        total_signals = len(df)
        active_signals = len(df[df['status'] == 'ACTIVE'])
        closed_signals = len(closed_df)

        # Win rate (targets hit vs SL hit)
        wins = len(closed_df[closed_df['status'].isin(['HIT_TARGET1', 'HIT_TARGET2'])])
        losses = len(closed_df[closed_df['status'] == 'HIT_SL'])
        win_rate = (wins / closed_signals * 100) if closed_signals > 0 else 0

        # Profit/Loss
        total_profit_loss = closed_df['profit_loss'].sum() if 'profit_loss' in closed_df.columns else 0
        avg_profit = closed_df['profit_loss'].mean() if 'profit_loss' in closed_df.columns else 0

        # Best and worst trades
        best_trade = closed_df['profit_loss'].max() if 'profit_loss' in closed_df.columns else 0
        worst_trade = closed_df['profit_loss'].min() if 'profit_loss' in closed_df.columns else 0

        return {
            'total_signals': total_signals,
            'active_signals': active_signals,
            'closed_signals': closed_signals,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_profit_loss': total_profit_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade
        }

    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return {
            'total_signals': 0,
            'active_signals': 0,
            'closed_signals': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'total_profit_loss': 0
        }


# ═══════════════════════════════════════════════════════════════════
# DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def display_signal_history_tab():
    """Display complete signal history with tabulation and filters"""

    st.markdown("## 📊 Signal History & Performance")
    st.caption("**All recorded entry signals tracked automatically with Supabase**")

    # Performance metrics at top
    st.markdown("### 📈 Performance Overview")

    metrics = get_performance_metrics()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Signals", metrics['total_signals'])
    with col2:
        st.metric("Active", metrics['active_signals'])
    with col3:
        st.metric("Closed", metrics['closed_signals'])
    with col4:
        win_rate_color = "🟢" if metrics['win_rate'] >= 60 else "🟡" if metrics['win_rate'] >= 50 else "🔴"
        st.metric("Win Rate", f"{win_rate_color} {metrics['win_rate']:.1f}%")
    with col5:
        pnl_color = "🟢" if metrics['total_profit_loss'] > 0 else "🔴"
        st.metric("Total P/L", f"{pnl_color} ₹{metrics['total_profit_loss']:.2f}")

    st.markdown("---")

    # Filters
    st.markdown("### 🔍 Filter Signals")

    col_filter1, col_filter2, col_filter3 = st.columns(3)

    with col_filter1:
        status_filter = st.selectbox(
            "Status",
            options=['ALL', 'ACTIVE', 'HIT_TARGET1', 'HIT_TARGET2', 'HIT_SL', 'CLOSED'],
            index=0
        )

    with col_filter2:
        limit = st.selectbox("Show", options=[50, 100, 200, 500], index=1)

    with col_filter3:
        auto_update = st.checkbox("Auto-update active signals", value=True)

    # Auto-update active signals if enabled
    if auto_update and 'bias_analysis_results' in st.session_state:
        if isinstance(st.session_state.bias_analysis_results, dict):
            df_price = st.session_state.bias_analysis_results.get('df')
            if df_price is not None and len(df_price) > 0:
                current_price_update = df_price['close'].iloc[-1]
                update_result = update_active_signals(current_price_update)
                if update_result['updated'] > 0:
                    st.success(f"✅ Updated {update_result['updated']} signals")

    # Get signals
    df_signals = get_all_signals(
        limit=limit,
        status_filter=None if status_filter == 'ALL' else status_filter
    )

    if df_signals.empty:
        st.info("📭 No signals recorded yet. Signals will be saved automatically when ENTER NOW is triggered.")
        return

    # Display signals table
    st.markdown("### 📋 Signal Records")

    # Format datetime columns
    if 'entry_time' in df_signals.columns:
        df_signals['entry_time'] = pd.to_datetime(df_signals['entry_time']).dt.strftime('%Y-%m-%d %H:%M:%S IST')

    if 'exit_time' in df_signals.columns:
        df_signals['exit_time'] = pd.to_datetime(df_signals['exit_time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S IST')

    # Select and rename columns for display
    display_columns = [
        'id', 'entry_time', 'signal_type', 'entry_price', 'stop_loss',
        'target1', 'target2', 'support_level', 'resistance_level',
        'entry_reason', 'source', 'status', 'exit_price', 'exit_time',
        'profit_loss', 'profit_loss_pct', 'exit_reason'
    ]

    # Filter to only existing columns
    available_columns = [col for col in display_columns if col in df_signals.columns]
    df_display = df_signals[available_columns].copy()

    # Rename for better display
    column_renames = {
        'id': 'ID',
        'entry_time': 'Entry Time',
        'signal_type': 'Type',
        'entry_price': 'Entry',
        'stop_loss': 'Stop Loss',
        'target1': 'Target 1',
        'target2': 'Target 2',
        'support_level': 'Support',
        'resistance_level': 'Resistance',
        'entry_reason': 'Entry Reason',
        'source': 'Source',
        'status': 'Status',
        'exit_price': 'Exit',
        'exit_time': 'Exit Time',
        'profit_loss': 'P/L',
        'profit_loss_pct': 'P/L %',
        'exit_reason': 'Exit Analysis (ML)'
    }

    df_display = df_display.rename(columns={k: v for k, v in column_renames.items() if k in df_display.columns})

    # Style the dataframe
    def highlight_status(row):
        if 'Status' in row:
            status = row['Status']
            if status == 'HIT_TARGET1' or status == 'HIT_TARGET2':
                return ['background-color: #1a3d1a'] * len(row)
            elif status == 'HIT_SL':
                return ['background-color: #3d1a1a'] * len(row)
            elif status == 'ACTIVE':
                return ['background-color: #1a2e3d'] * len(row)
        return [''] * len(row)

    st.dataframe(
        df_display.style.apply(highlight_status, axis=1),
        use_container_width=True,
        height=600
    )

    # Download button
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Download Signal History (CSV)",
        data=csv,
        file_name=f"signal_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


# ═══════════════════════════════════════════════════════════════════
# SQL SCHEMA FOR SUPABASE TABLE
# ═══════════════════════════════════════════════════════════════════
"""
-- ================================================================
-- TRADING SIGNALS TABLE WITH ML EXIT ANALYSIS
-- ================================================================
-- Run this SQL in your Supabase SQL Editor to create the table
-- ================================================================

CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Signal Details
    signal_type TEXT NOT NULL,  -- LONG or SHORT
    entry_price NUMERIC NOT NULL,
    stop_loss NUMERIC NOT NULL,
    target1 NUMERIC NOT NULL,
    target2 NUMERIC NOT NULL,

    -- Support/Resistance Levels
    support_level NUMERIC,
    resistance_level NUMERIC,

    -- Entry Context
    entry_reason TEXT,  -- Why entry was taken (confluence, VOB, etc.)
    current_price_at_entry NUMERIC,
    confidence NUMERIC DEFAULT 0,
    source TEXT DEFAULT 'VOB',  -- VOB, HTF, Confluence, Fibonacci, etc.

    -- Status Tracking
    status TEXT DEFAULT 'ACTIVE',  -- ACTIVE, HIT_TARGET1, HIT_TARGET2, HIT_SL, CLOSED
    entry_time TIMESTAMPTZ NOT NULL,

    -- Exit Details
    exit_price NUMERIC,
    exit_time TIMESTAMPTZ,
    exit_reason TEXT,  -- ML-based analysis of WHY target/SL was hit

    -- Performance Metrics
    profit_loss NUMERIC,  -- Absolute profit/loss in points
    profit_loss_pct NUMERIC,  -- Percentage profit/loss

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_trading_signals_status ON trading_signals(status);
CREATE INDEX IF NOT EXISTS idx_trading_signals_entry_time ON trading_signals(entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_signal_type ON trading_signals(signal_type);

-- ================================================================
-- OPTIONAL: Enable Row Level Security (RLS)
-- ================================================================
-- Uncomment if you want to enable RLS for security

-- ALTER TABLE trading_signals ENABLE ROW LEVEL SECURITY;

-- CREATE POLICY "Enable read access for authenticated users" ON trading_signals
--     FOR SELECT USING (auth.role() = 'authenticated');

-- CREATE POLICY "Enable insert access for authenticated users" ON trading_signals
--     FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- CREATE POLICY "Enable update access for authenticated users" ON trading_signals
--     FOR UPDATE USING (auth.role() = 'authenticated');

-- ================================================================
"""
