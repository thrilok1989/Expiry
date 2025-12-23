"""
Advanced Chart Analysis Module
Integrates all 4 TradingView indicators with Plotly charts
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pytz

# Indian Standard Time (IST)
IST = pytz.timezone('Asia/Kolkata')

from indicators.volume_order_blocks import VolumeOrderBlocks
from indicators.htf_support_resistance import HTFSupportResistance
from indicators.htf_volume_footprint import HTFVolumeFootprint
from indicators.ultimate_rsi import UltimateRSI
from indicators.om_indicator import OMIndicator
from indicators.liquidity_sentiment_profile import LiquiditySentimentProfile
from indicators.advanced_price_action import AdvancedPriceAction
from indicators.money_flow_profile import MoneyFlowProfile
from indicators.deltaflow_volume_profile import DeltaFlowVolumeProfile
from dhan_data_fetcher import DhanDataFetcher
from config import get_dhan_credentials


class AdvancedChartAnalysis:
    """
    Advanced Chart Analysis with multiple technical indicators
    """

    def __init__(self):
        """Initialize Advanced Chart Analysis"""
        # Indicators will be created with custom parameters when needed
        self.htf_sr_indicator = HTFSupportResistance()

        # Initialize Dhan data fetcher for Indian indices
        try:
            self.dhan_fetcher = DhanDataFetcher()
            self.use_dhan = True
        except Exception as e:
            print(f"Warning: Could not initialize Dhan API: {e}")
            self.dhan_fetcher = None
            self.use_dhan = False

    def fetch_intraday_data(self, symbol, period='1d', interval='1m'):
        """
        Fetch intraday data using Dhan API for Indian indices or yfinance for others

        Args:
            symbol: Stock symbol (e.g., '^NSEI' for NIFTY, '^BSESN' for SENSEX, '^DJI' for DOW)
            period: Period to fetch ('1d', '5d', '1mo')
            interval: Interval ('1m', '5m', '15m', '1h')

        Returns:
            DataFrame: OHLCV data with volume
        """
        # Map Yahoo Finance symbols to Dhan instruments
        symbol_map = {
            '^NSEI': 'NIFTY',
            '^NSEBANK': 'BANKNIFTY',
            '^BSESN': 'SENSEX'
        }

        # Check if this is an Indian index
        if symbol in symbol_map and self.use_dhan and self.dhan_fetcher:
            return self._fetch_from_dhan(symbol_map[symbol], period, interval)
        else:
            return self._fetch_from_yfinance(symbol, period, interval)

    def _fetch_from_dhan(self, instrument, period, interval):
        """Fetch data from Dhan API for Indian indices"""
        try:
            # Map yfinance interval to Dhan interval
            interval_map = {
                '1m': '1',
                '5m': '5',
                '15m': '15',
                '1h': '60',
                '60m': '60'
            }
            dhan_interval = interval_map.get(interval, '1')

            # Calculate date range based on period - Use IST timezone
            to_date = datetime.now(IST)
            if period == '1d':
                from_date = to_date - timedelta(days=1)
            elif period == '5d':
                from_date = to_date - timedelta(days=5)
            elif period == '1mo':
                from_date = to_date - timedelta(days=30)
            else:
                from_date = to_date - timedelta(days=1)

            # Format dates for Dhan API
            from_date_str = from_date.strftime('%Y-%m-%d')
            to_date_str = to_date.strftime('%Y-%m-%d')

            # Fetch data from Dhan
            result = self.dhan_fetcher.fetch_intraday_data(
                instrument=instrument,
                interval=dhan_interval,
                from_date=from_date_str,
                to_date=to_date_str
            )

            if result.get('success') and result.get('data') is not None:
                df = result['data']

                # Ensure index is datetime
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)

                # Ensure required columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    return df
                else:
                    print(f"Warning: Dhan data missing required columns. Falling back to yfinance.")
                    return None
            else:
                print(f"Warning: Dhan API failed. Falling back to yfinance.")
                return None

        except Exception as e:
            print(f"Error fetching from Dhan: {e}. Falling back to yfinance.")
            return None

    def _fetch_from_yfinance(self, symbol, period, interval):
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if len(df) == 0:
                return None

            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]

            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return None

            return df

        except Exception as e:
            print(f"Error fetching data from yfinance: {e}")
            return None

    def create_advanced_chart(self, df, symbol, show_vob=True, show_htf_sr=True,
                             show_footprint=True, show_rsi=True, show_om=False,
                             show_volume=True, show_liquidity_profile=False,
                             show_money_flow_profile=False, show_deltaflow_profile=False,
                             show_bos=False, show_choch=False, show_fibonacci=False,
                             show_patterns=False,
                             vob_params=None, htf_params=None, footprint_params=None,
                             rsi_params=None, om_params=None, liquidity_params=None,
                             money_flow_params=None, deltaflow_params=None,
                             price_action_params=None):
        """
        Create advanced chart with all indicators

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name for title
            show_vob: Show Volume Order Blocks
            show_htf_sr: Show HTF Support/Resistance
            show_footprint: Show HTF Volume Footprint
            show_rsi: Show Ultimate RSI
            show_om: Show OM Indicator (comprehensive order flow)
            show_volume: Show Volume bars
            show_liquidity_profile: Show Liquidity Sentiment Profile
            show_money_flow_profile: Show Money Flow Profile (volume/money flow weighted)
            show_deltaflow_profile: Show DeltaFlow Volume Profile (delta per price level)
            vob_params: Parameters for Volume Order Blocks indicator
            htf_params: Parameters for HTF Support/Resistance indicator
            footprint_params: Parameters for HTF Volume Footprint indicator
            rsi_params: Parameters for Ultimate RSI indicator
            om_params: Parameters for OM Indicator
            liquidity_params: Parameters for Liquidity Sentiment Profile indicator
            money_flow_params: Parameters for Money Flow Profile indicator
            deltaflow_params: Parameters for DeltaFlow Volume Profile indicator

        Returns:
            plotly Figure object
        """
        try:
            # Ensure dataframe has datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    raise ValueError(f"Unable to convert dataframe index to datetime: {e}")

            # Convert timestamps to IST timezone for proper display
            if df.index.tz is None:
                # If timestamps are naive (no timezone), assume UTC and convert to IST
                df.index = df.index.tz_localize('UTC').tz_convert(IST)
            elif df.index.tz != IST:
                # If timestamps have a different timezone, convert to IST
                df.index = df.index.tz_convert(IST)

            # Create indicators with custom parameters
            vob_indicator = None
            if show_vob:
                if vob_params:
                    vob_indicator = VolumeOrderBlocks(
                        sensitivity=vob_params.get('sensitivity', 5),
                        mid_line=vob_params.get('mid_line', True),
                        trend_shadow=vob_params.get('trend_shadow', True)
                    )
                else:
                    vob_indicator = VolumeOrderBlocks()

            ultimate_rsi = None
            if show_rsi:
                if rsi_params:
                    ultimate_rsi = UltimateRSI(**rsi_params)
                else:
                    ultimate_rsi = UltimateRSI()

            om_indicator = None
            if show_om:
                if om_params:
                    om_indicator = OMIndicator(**om_params)
                else:
                    om_indicator = OMIndicator()

            htf_footprint = None
            if show_footprint:
                if footprint_params:
                    htf_footprint = HTFVolumeFootprint(**footprint_params)
                else:
                    htf_footprint = HTFVolumeFootprint(bins=10, timeframe='D', dynamic_poc=True)

            lsp_indicator = None
            if show_liquidity_profile:
                if liquidity_params:
                    lsp_indicator = LiquiditySentimentProfile(**liquidity_params)
                else:
                    lsp_indicator = LiquiditySentimentProfile()

            # NEW: Money Flow Profile indicator
            money_flow_indicator = None
            if show_money_flow_profile:
                if money_flow_params:
                    money_flow_indicator = MoneyFlowProfile(**money_flow_params)
                else:
                    # Default: 10 rows as requested by user
                    money_flow_indicator = MoneyFlowProfile(num_rows=10, lookback=200)

            # NEW: DeltaFlow Volume Profile indicator
            deltaflow_indicator = None
            if show_deltaflow_profile:
                if deltaflow_params:
                    deltaflow_indicator = DeltaFlowVolumeProfile(**deltaflow_params)
                else:
                    deltaflow_indicator = DeltaFlowVolumeProfile(bins=30, lookback=200)

            price_action_indicator = None
            if show_bos or show_choch or show_fibonacci or show_patterns:
                if price_action_params:
                    price_action_indicator = AdvancedPriceAction(**price_action_params)
                else:
                    price_action_indicator = AdvancedPriceAction()

            # Calculate all indicators
            vob_data = vob_indicator.calculate(df) if vob_indicator else None
            rsi_data = ultimate_rsi.get_signals(df) if ultimate_rsi else None
            om_data = om_indicator.calculate(df) if om_indicator else None
            lsp_data = lsp_indicator.calculate(df) if lsp_indicator else None
            money_flow_data = money_flow_indicator.calculate(df) if money_flow_indicator else None
            deltaflow_data = deltaflow_indicator.calculate(df) if deltaflow_indicator else None
            price_action_data = price_action_indicator.analyze(df) if price_action_indicator else None
        except Exception as e:
            raise Exception(f"Error calculating indicators: {str(e)}")

        # HTF Support/Resistance configuration
        htf_levels = []
        if show_htf_sr:
            if htf_params and htf_params.get('levels_config'):
                levels_config = htf_params['levels_config']
            else:
                # Default configuration
                levels_config = [
                    {'timeframe': '3T', 'length': 4, 'style': 'Solid', 'color': '#26a69a'},   # 3 min - Teal
                    {'timeframe': '5T', 'length': 5, 'style': 'Solid', 'color': '#2196f3'},   # 5 min - Blue
                    {'timeframe': '10T', 'length': 5, 'style': 'Solid', 'color': '#9c27b0'},  # 10 min - Purple
                    {'timeframe': '15T', 'length': 5, 'style': 'Solid', 'color': '#ff9800'}   # 15 min - Orange
                ]
            htf_levels = self.htf_sr_indicator.calculate_multi_timeframe(df, levels_config)

            # CRITICAL FIX: Store HTF S/R levels in session state for signal generation
            # This makes HTF levels available to signal_display_simple.py and overall_market_sentiment.py
            try:
                import streamlit as st
                # Store raw HTF levels
                st.session_state.htf_sr_levels = htf_levels

                # Also extract nearest support/resistance for current price
                current_price = df['Close'].iloc[-1]

                # Collect all support and resistance levels from all timeframes
                all_supports = []
                all_resistances = []

                for level in htf_levels:
                    if level.get('pivot_low') and level['pivot_low'] is not None:
                        if level['pivot_low'] < current_price:
                            all_supports.append({
                                'price': level['pivot_low'],
                                'timeframe': level.get('timeframe', ''),
                                'type': 'Support'
                            })
                    if level.get('pivot_high') and level['pivot_high'] is not None:
                        if level['pivot_high'] > current_price:
                            all_resistances.append({
                                'price': level['pivot_high'],
                                'timeframe': level.get('timeframe', ''),
                                'type': 'Resistance'
                            })

                # Find nearest levels
                if all_supports:
                    nearest_support = max(all_supports, key=lambda x: x['price'])
                    st.session_state.htf_nearest_support = nearest_support

                if all_resistances:
                    nearest_resistance = min(all_resistances, key=lambda x: x['price'])
                    st.session_state.htf_nearest_resistance = nearest_resistance

            except Exception as e:
                # Silent fail if not in Streamlit context
                pass

        # HTF Volume Footprint
        footprint_data = None
        if show_footprint and htf_footprint:
            footprint_data = htf_footprint.calculate(df)

        # Create subplots based on what indicators are enabled
        if show_rsi and show_volume:
            # Price + Volume + RSI
            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.6, 0.2, 0.2],
                vertical_spacing=0.03,
                shared_xaxes=True,
                subplot_titles=(f'{symbol} - 1 Minute Chart', 'Volume', 'Ultimate RSI')
            )
            price_row = 1
            volume_row = 2
            rsi_row = 3
        elif show_rsi:
            # Price + RSI (no volume)
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.05,
                shared_xaxes=True,
                subplot_titles=(f'{symbol} - 1 Minute Chart', 'Ultimate RSI')
            )
            price_row = 1
            volume_row = None
            rsi_row = 2
        elif show_volume:
            # Price + Volume (no RSI)
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.05,
                shared_xaxes=True,
                subplot_titles=(f'{symbol} - 1 Minute Chart', 'Volume')
            )
            price_row = 1
            volume_row = 2
            rsi_row = None
        else:
            # Price only
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=(f'{symbol} - 1 Minute Chart',)
            )
            price_row = 1
            volume_row = None
            rsi_row = None

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=price_row, col=1
        )

        # Add Volume Order Blocks
        if show_vob and vob_data:
            self._add_volume_order_blocks(fig, df, vob_data, row=price_row, col=1)

        # Add HTF Support/Resistance
        if show_htf_sr and htf_levels:
            self._add_htf_support_resistance(fig, df, htf_levels, row=price_row, col=1)

        # Add HTF Volume Footprint
        if show_footprint and footprint_data and footprint_data['current_footprint']:
            self._add_volume_footprint(fig, df, footprint_data, row=price_row, col=1)

        # Add Volume bars
        if show_volume and volume_row is not None:
            self._add_volume_bars(fig, df, row=volume_row, col=1)

        # Add Ultimate RSI
        if show_rsi and rsi_data and rsi_row is not None:
            self._add_ultimate_rsi(fig, df, rsi_data, row=rsi_row, col=1)

        # Add OM Indicator
        if show_om and om_data:
            self._add_om_indicator(fig, df, om_data, row=price_row, col=1)

        # Add Liquidity Sentiment Profile
        if show_liquidity_profile and lsp_data and lsp_data.get('success'):
            fig = lsp_indicator.add_to_chart(fig, df, lsp_data)

        # NEW: Add Money Flow Profile
        if show_money_flow_profile and money_flow_data and money_flow_data.get('success'):
            self._add_money_flow_profile(fig, df, money_flow_data, money_flow_indicator, row=price_row, col=1)

        # NEW: Add DeltaFlow Volume Profile
        if show_deltaflow_profile and deltaflow_data and deltaflow_data.get('success'):
            self._add_deltaflow_profile(fig, df, deltaflow_data, deltaflow_indicator, row=price_row, col=1)

        # Add Advanced Price Action Features
        if price_action_data and price_action_data.get('success'):
            if show_bos and price_action_data.get('bos_events'):
                self._add_bos(fig, df, price_action_data['bos_events'], row=price_row, col=1)

            if show_choch and price_action_data.get('choch_events'):
                self._add_choch(fig, df, price_action_data['choch_events'], row=price_row, col=1)

            if show_fibonacci and price_action_data.get('fibonacci', {}).get('success'):
                self._add_fibonacci(fig, df, price_action_data['fibonacci'], row=price_row, col=1)

            if show_patterns and price_action_data.get('patterns'):
                self._add_patterns(fig, df, price_action_data['patterns'], row=price_row, col=1)

        # Update layout
        # Calculate height based on number of subplots
        if show_rsi and show_volume:
            chart_height = 900
        elif show_rsi or show_volume:
            chart_height = 800
        else:
            chart_height = 600

        fig.update_layout(
            title=f'{symbol} Advanced Chart Analysis',
            xaxis_title='Time (IST)',
            yaxis_title='Price',
            template='plotly_dark',
            height=chart_height,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            xaxis=dict(
                tickformat='%H:%M:%S',  # Format time as HH:MM:SS
                hoverformat='%Y-%m-%d %H:%M:%S %Z'  # Show full datetime with timezone on hover
            )
        )

        return fig

    def _add_volume_order_blocks(self, fig, df, vob_data, row, col):
        """Add Volume Order Blocks to chart"""
        # Add EMA lines
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vob_data['ema1'],
                mode='lines',
                name='EMA Fast',
                line=dict(color='#00bcd4', width=1),
                opacity=0.7
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vob_data['ema2'],
                mode='lines',
                name='EMA Slow',
                line=dict(color='#ff9800', width=1),
                opacity=0.7
            ),
            row=row, col=col
        )

        # Add bullish blocks
        for block in vob_data['bullish_blocks']:
            if block['active']:
                idx = block['index']
                if idx < len(df):
                    x_start = df.index[idx]
                    x_end = df.index[-1]

                    # Upper line
                    fig.add_trace(
                        go.Scatter(
                            x=[x_start, x_end],
                            y=[block['upper'], block['upper']],
                            mode='lines',
                            name=f"Bullish OB",
                            line=dict(color='#26ba9f', width=2),
                            showlegend=False
                        ),
                        row=row, col=col
                    )

                    # Lower line
                    fig.add_trace(
                        go.Scatter(
                            x=[x_start, x_end],
                            y=[block['lower'], block['lower']],
                            mode='lines',
                            line=dict(color='#26ba9f', width=2),
                            fill='tonexty',
                            fillcolor='rgba(38, 186, 159, 0.1)',
                            showlegend=False
                        ),
                        row=row, col=col
                    )

        # Add bearish blocks
        for block in vob_data['bearish_blocks']:
            if block['active']:
                idx = block['index']
                if idx < len(df):
                    x_start = df.index[idx]
                    x_end = df.index[-1]

                    # Upper line
                    fig.add_trace(
                        go.Scatter(
                            x=[x_start, x_end],
                            y=[block['upper'], block['upper']],
                            mode='lines',
                            name=f"Bearish OB",
                            line=dict(color='#6626ba', width=2),
                            showlegend=False
                        ),
                        row=row, col=col
                    )

                    # Lower line
                    fig.add_trace(
                        go.Scatter(
                            x=[x_start, x_end],
                            y=[block['lower'], block['lower']],
                            mode='lines',
                            line=dict(color='#6626ba', width=2),
                            fill='tonexty',
                            fillcolor='rgba(102, 38, 186, 0.1)',
                            showlegend=False
                        ),
                        row=row, col=col
                    )

    def _add_volume_bars(self, fig, df, row, col):
        """Add Volume bars to chart (TradingView style)"""
        # Determine bar colors based on candle direction
        colors = ['#26a69a' if close >= open else '#ef5350'
                  for close, open in zip(df['close'], df['open'])]

        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker=dict(
                    color=colors,
                    line=dict(width=0)
                ),
                showlegend=False
            ),
            row=row, col=col
        )

        # Update volume y-axis
        fig.update_yaxes(title_text="Volume", row=row, col=col)

    def _add_htf_support_resistance(self, fig, df, htf_levels, row, col):
        """Add HTF Support/Resistance levels to chart"""
        x_start = df.index[0]
        x_end = df.index[-1]

        # Map timeframe codes to display names
        timeframe_display = {
            '3T': '3 min',
            '5T': '5 min',
            '10T': '10 min',
            '15T': '15 min',
            '4H': '4H',
            '12H': '12H',
            'D': 'Daily',
            'W': 'Weekly'
        }

        for level in htf_levels:
            tf_display = timeframe_display.get(level['timeframe'], level['timeframe'])

            # Add pivot high
            if level['pivot_high'] is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[x_start, x_end],
                        y=[level['pivot_high'], level['pivot_high']],
                        mode='lines',
                        name=f"{tf_display} Resistance",
                        line=dict(color=level['color'], width=2, dash='dash'),
                        showlegend=True
                    ),
                    row=row, col=col
                )

            # Add pivot low
            if level['pivot_low'] is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[x_start, x_end],
                        y=[level['pivot_low'], level['pivot_low']],
                        mode='lines',
                        name=f"{tf_display} Support",
                        line=dict(color=level['color'], width=2, dash='dash'),
                        showlegend=True
                    ),
                    row=row, col=col
                )

    def _add_volume_footprint(self, fig, df, footprint_data, row, col):
        """Add HTF Volume Footprint to chart with volume bins and dynamic POC"""
        current = footprint_data['current_footprint']
        if not current:
            return

        dynamic_poc = footprint_data.get('dynamic_poc', False)
        historical_pocs = footprint_data.get('historical_pocs', [])

        # Get current period bounds
        period_start = current['period_start']
        current_time = df.index[-1]

        # Add Volume Profile Bins (horizontal rectangles showing volume distribution)
        if 'bins' in current and len(current['bins']) > 0:
            max_volume = max([bin_data['volume'] for bin_data in current['bins']])

            # Calculate the width for bins (proportional to timeframe)
            time_range = (current_time - period_start).total_seconds()
            # Make bins visible but not too wide - scale to ~10% of period
            bin_width_seconds = time_range * 0.1

            for bin_data in current['bins']:
                if bin_data['volume'] > 0:  # Only show bins with volume
                    # Scale the bin width based on volume
                    volume_ratio = bin_data['volume'] / max_volume if max_volume > 0 else 0
                    bin_x_end = period_start + pd.Timedelta(seconds=bin_width_seconds * volume_ratio)

                    # Use different color for POC bin
                    if bin_data['is_poc']:
                        bin_color = 'rgba(41, 138, 218, 0.4)'  # Blue for POC
                        border_color = '#298ada'
                        border_width = 2
                    else:
                        # Gradient color based on volume
                        intensity = int(120 + (volume_ratio * 135))  # Range from 120 to 255
                        bin_color = f'rgba({intensity}, {intensity}, {intensity}, 0.2)'
                        border_color = f'rgba({intensity}, {intensity}, {intensity}, 0.5)'
                        border_width = 1

                    # Add rectangle for this bin
                    # For subplots, we need to specify xref and yref correctly
                    if row == 1 and col == 1:
                        xref, yref = 'x', 'y'
                    else:
                        xref, yref = f'x{row}', f'y{row}'

                    fig.add_shape(
                        type="rect",
                        x0=period_start,
                        x1=bin_x_end,
                        y0=bin_data['lower'],
                        y1=bin_data['upper'],
                        fillcolor=bin_color,
                        line=dict(color=border_color, width=border_width),
                        layer='below',
                        xref=xref,
                        yref=yref
                    )

        # Handle POC display based on dynamic_poc setting
        if dynamic_poc:
            # Dynamic POC: Show POC line extending from period start to current time (real-time update)
            fig.add_trace(
                go.Scatter(
                    x=[period_start, current_time],
                    y=[current['poc'], current['poc']],
                    mode='lines',
                    name='Dynamic POC',
                    line=dict(color='#298ada', width=3, dash='solid'),
                    showlegend=True
                ),
                row=row, col=col
            )
        else:
            # Static POC: Show historical POC lines for all completed periods
            for hist_poc in historical_pocs:
                fig.add_trace(
                    go.Scatter(
                        x=[hist_poc['period_start'], hist_poc['period_end']],
                        y=[hist_poc['poc_price'], hist_poc['poc_price']],
                        mode='lines',
                        name='Historical POC',
                        line=dict(color='#298ada', width=2, dash='dash'),
                        showlegend=False
                    ),
                    row=row, col=col
                )

            # Also show current period POC
            fig.add_trace(
                go.Scatter(
                    x=[period_start, current_time],
                    y=[current['poc'], current['poc']],
                    mode='lines',
                    name='POC (Point of Control)',
                    line=dict(color='#298ada', width=3, dash='solid'),
                    showlegend=True
                ),
                row=row, col=col
            )

        # Add Value Area (same for both dynamic and static)
        fig.add_trace(
            go.Scatter(
                x=[period_start, current_time],
                y=[current['value_area_high'], current['value_area_high']],
                mode='lines',
                name='Value Area High',
                line=dict(color='#64b5f6', width=1, dash='dot'),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=[period_start, current_time],
                y=[current['value_area_low'], current['value_area_low']],
                mode='lines',
                name='Value Area Low',
                line=dict(color='#64b5f6', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(100, 181, 246, 0.1)',
                showlegend=False
            ),
            row=row, col=col
        )

    def _add_ultimate_rsi(self, fig, df, rsi_data, row, col):
        """Add Ultimate RSI indicator to chart"""
        # Add RSI line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rsi_data['ultimate_rsi'],
                mode='lines',
                name='Ultimate RSI',
                line=dict(color='#00bcd4', width=2)
            ),
            row=row, col=col
        )

        # Add signal line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rsi_data['signal'],
                mode='lines',
                name='Signal',
                line=dict(color='#ff5d00', width=2)
            ),
            row=row, col=col
        )

        # Add overbought/oversold lines
        fig.add_hline(y=80, line_dash="dash", line_color="red",
                     annotation_text="Overbought", row=row, col=col)
        fig.add_hline(y=20, line_dash="dash", line_color="green",
                     annotation_text="Oversold", row=row, col=col)
        fig.add_hline(y=50, line_dash="dot", line_color="gray",
                     annotation_text="Midline", row=row, col=col)

        # Update RSI yaxis
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=col)

    def _add_om_indicator(self, fig, df, om_data, row, col):
        """Add OM (Order Flow & Momentum) Indicator to chart"""
        x_start = df.index[0]
        x_end = df.index[-1]

        # Add VWAP
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=om_data['vwap'],
                mode='lines',
                name='VWAP',
                line=dict(color='rgba(0, 47, 255, 0.5)', width=1),
                showlegend=True
            ),
            row=row, col=col
        )

        # Add VOB EMAs
        vob = om_data['vob_data']
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vob['ema1'],
                mode='lines',
                name='VOB EMA Fast',
                line=dict(color='#26ba9f', width=1, dash='dot'),
                opacity=0.5,
                showlegend=False
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vob['ema2'],
                mode='lines',
                name='VOB EMA Slow',
                line=dict(color='#ba2646', width=1, dash='dot'),
                opacity=0.5,
                showlegend=False
            ),
            row=row, col=col
        )

        # Add VOB Bullish Blocks
        for block in vob['bullish_blocks']:
            idx = block['index']
            if idx < len(df):
                x_start_block = block['start_time']
                fig.add_trace(
                    go.Scatter(
                        x=[x_start_block, x_end],
                        y=[block['upper'], block['upper']],
                        mode='lines',
                        name='Bullish VOB',
                        line=dict(color='#26ba9f', width=2),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Scatter(
                        x=[x_start_block, x_end],
                        y=[block['lower'], block['lower']],
                        mode='lines',
                        line=dict(color='#26ba9f', width=2),
                        fill='tonexty',
                        fillcolor='rgba(38, 186, 159, 0.15)',
                        showlegend=False
                    ),
                    row=row, col=col
                )

        # Add VOB Bearish Blocks
        for block in vob['bearish_blocks']:
            idx = block['index']
            if idx < len(df):
                x_start_block = block['start_time']
                fig.add_trace(
                    go.Scatter(
                        x=[x_start_block, x_end],
                        y=[block['upper'], block['upper']],
                        mode='lines',
                        name='Bearish VOB',
                        line=dict(color='#ba2646', width=2),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Scatter(
                        x=[x_start_block, x_end],
                        y=[block['lower'], block['lower']],
                        mode='lines',
                        line=dict(color='#ba2646', width=2),
                        fill='tonexty',
                        fillcolor='rgba(186, 38, 70, 0.15)',
                        showlegend=False
                    ),
                    row=row, col=col
                )

        # Add HVP (High Volume Pivots)
        hvp = om_data['hvp_data']
        for pivot in hvp['pivot_highs']:
            fig.add_trace(
                go.Scatter(
                    x=[pivot['time']],
                    y=[pivot['price']],
                    mode='markers+text',
                    name='HVP Resistance',
                    marker=dict(symbol='circle', size=8, color='rgba(230, 14, 147, 0.6)'),
                    text='🔴',
                    textposition='top center',
                    showlegend=False
                ),
                row=row, col=col
            )

        for pivot in hvp['pivot_lows']:
            fig.add_trace(
                go.Scatter(
                    x=[pivot['time']],
                    y=[pivot['price']],
                    mode='markers+text',
                    name='HVP Support',
                    marker=dict(symbol='circle', size=8, color='rgba(34, 212, 204, 0.6)'),
                    text='🟢',
                    textposition='bottom center',
                    showlegend=False
                ),
                row=row, col=col
            )

        # Add VIDYA
        vidya = om_data['vidya_data']
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vidya['smoothed'],
                mode='lines',
                name='VIDYA',
                line=dict(
                    color='#ffa726',  # Orange color for VIDYA line
                    width=2
                ),
                showlegend=True
            ),
            row=row, col=col
        )

        # Add VIDYA trend crossover markers
        for i in range(len(df)):
            if vidya['trend_cross_up'][i]:
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i]],
                        y=[vidya['smoothed'][i]],
                        mode='markers+text',
                        marker=dict(symbol='triangle-up', size=12, color='#17dfad'),
                        text='▲',
                        textposition='bottom center',
                        name='VIDYA Trend Up',
                        showlegend=False
                    ),
                    row=row, col=col
                )
            elif vidya['trend_cross_down'][i]:
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i]],
                        y=[vidya['smoothed'][i]],
                        mode='markers+text',
                        marker=dict(symbol='triangle-down', size=12, color='#dd326b'),
                        text='▼',
                        textposition='top center',
                        name='VIDYA Trend Down',
                        showlegend=False
                    ),
                    row=row, col=col
                )

        # Add Delta Buy/Sell Spikes
        delta = om_data['delta_data']
        buy_spike_indices = [i for i, spike in enumerate(delta['buy_spike']) if spike]
        sell_spike_indices = [i for i, spike in enumerate(delta['sell_spike']) if spike]

        if buy_spike_indices:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i] for i in buy_spike_indices],
                    y=[df['low'].iloc[i] for i in buy_spike_indices],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=6, color='green'),
                    name='Delta Buy Spike',
                    showlegend=False
                ),
                row=row, col=col
            )

        if sell_spike_indices:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i] for i in sell_spike_indices],
                    y=[df['high'].iloc[i] for i in sell_spike_indices],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=6, color='red'),
                    name='Delta Sell Spike',
                    showlegend=False
                ),
                row=row, col=col
            )

        # Add LTP Trap signals
        ltp = om_data['ltp_trap']
        ltp_buy_indices = [i for i, trap in enumerate(ltp['ltp_trap_buy']) if trap]
        ltp_sell_indices = [i for i, trap in enumerate(ltp['ltp_trap_sell']) if trap]

        if ltp_buy_indices:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i] for i in ltp_buy_indices],
                    y=[df['low'].iloc[i] for i in ltp_buy_indices],
                    mode='markers+text',
                    marker=dict(symbol='square', size=10, color='rgba(0, 137, 123, 0.7)'),
                    text='LTP↑',
                    textposition='bottom center',
                    name='LTP Trap Buy',
                    showlegend=False
                ),
                row=row, col=col
            )

        if ltp_sell_indices:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i] for i in ltp_sell_indices],
                    y=[df['high'].iloc[i] for i in ltp_sell_indices],
                    mode='markers+text',
                    marker=dict(symbol='square', size=10, color='rgba(136, 14, 79, 0.7)'),
                    text='LTP↓',
                    textposition='top center',
                    name='LTP Trap Sell',
                    showlegend=False
                ),
                row=row, col=col
            )

    # =========================================================================
    # ADVANCED PRICE ACTION VISUALIZATION METHODS
    # =========================================================================

    def _add_bos(self, fig, df, bos_events, row, col):
        """Add Break of Structure (BOS) markers to chart"""
        for bos in bos_events:
            if bos['type'] == 'BULLISH':
                # Bullish BOS - Green arrow pointing up
                fig.add_trace(
                    go.Scatter(
                        x=[bos['time']],
                        y=[bos['price']],
                        mode='markers+text',
                        name='Bullish BOS',
                        marker=dict(
                            symbol='triangle-up',
                            size=15,
                            color='#00ff00',
                            line=dict(color='white', width=2)
                        ),
                        text='BOS↑',
                        textposition='bottom center',
                        textfont=dict(color='#00ff00', size=12, family='Arial Black'),
                        showlegend=False
                    ),
                    row=row, col=col
                )

                # Draw line from structure level to break
                fig.add_trace(
                    go.Scatter(
                        x=[bos['structure_time'], bos['time']],
                        y=[bos['structure_level'], bos['structure_level']],
                        mode='lines',
                        line=dict(color='#00ff00', width=1, dash='dot'),
                        showlegend=False
                    ),
                    row=row, col=col
                )

            else:  # BEARISH BOS
                # Bearish BOS - Red arrow pointing down
                fig.add_trace(
                    go.Scatter(
                        x=[bos['time']],
                        y=[bos['price']],
                        mode='markers+text',
                        name='Bearish BOS',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='#ff0000',
                            line=dict(color='white', width=2)
                        ),
                        text='BOS↓',
                        textposition='top center',
                        textfont=dict(color='#ff0000', size=12, family='Arial Black'),
                        showlegend=False
                    ),
                    row=row, col=col
                )

                # Draw line from structure level to break
                fig.add_trace(
                    go.Scatter(
                        x=[bos['structure_time'], bos['time']],
                        y=[bos['structure_level'], bos['structure_level']],
                        mode='lines',
                        line=dict(color='#ff0000', width=1, dash='dot'),
                        showlegend=False
                    ),
                    row=row, col=col
                )

    def _add_choch(self, fig, df, choch_events, row, col):
        """Add Change of Character (CHOCH) markers to chart"""
        for choch in choch_events:
            if choch['type'] == 'BULLISH':
                # Bullish CHOCH - Cyan diamond
                fig.add_trace(
                    go.Scatter(
                        x=[choch['time']],
                        y=[choch['price']],
                        mode='markers+text',
                        name='Bullish CHOCH',
                        marker=dict(
                            symbol='diamond',
                            size=12,
                            color='#00ffff',
                            line=dict(color='white', width=1)
                        ),
                        text='CHOCH',
                        textposition='bottom center',
                        textfont=dict(color='#00ffff', size=10),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            else:  # BEARISH CHOCH
                # Bearish CHOCH - Orange diamond
                fig.add_trace(
                    go.Scatter(
                        x=[choch['time']],
                        y=[choch['price']],
                        mode='markers+text',
                        name='Bearish CHOCH',
                        marker=dict(
                            symbol='diamond',
                            size=12,
                            color='#ff8800',
                            line=dict(color='white', width=1)
                        ),
                        text='CHOCH',
                        textposition='top center',
                        textfont=dict(color='#ff8800', size=10),
                        showlegend=False
                    ),
                    row=row, col=col
                )

    def _add_fibonacci(self, fig, df, fib_data, row, col):
        """Add Fibonacci retracement and extension levels to chart"""
        x_start = fib_data['swing_low']['time']
        x_end = df.index[-1]

        # Add swing high and low markers
        fig.add_trace(
            go.Scatter(
                x=[fib_data['swing_high']['time']],
                y=[fib_data['swing_high']['price']],
                mode='markers',
                name='Swing High',
                marker=dict(symbol='star', size=12, color='yellow'),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=[fib_data['swing_low']['time']],
                y=[fib_data['swing_low']['price']],
                mode='markers',
                name='Swing Low',
                marker=dict(symbol='star', size=12, color='blue'),
                showlegend=False
            ),
            row=row, col=col
        )

        # Fibonacci colors
        fib_colors = {
            '0.0': 'rgba(150, 150, 150, 0.3)',
            '0.236': 'rgba(100, 181, 246, 0.3)',
            '0.382': 'rgba(66, 165, 245, 0.3)',
            '0.5': 'rgba(255, 235, 59, 0.4)',      # Yellow for 50%
            '0.618': 'rgba(255, 167, 38, 0.4)',    # Golden ratio
            '0.786': 'rgba(239, 83, 80, 0.3)',
            '1.0': 'rgba(150, 150, 150, 0.3)',
            '1.272': 'rgba(156, 39, 176, 0.3)',
            '1.414': 'rgba(103, 58, 183, 0.3)',
            '1.618': 'rgba(63, 81, 181, 0.4)',     # Golden extension
            '2.0': 'rgba(33, 150, 243, 0.3)',
            '2.618': 'rgba(3, 169, 244, 0.3)'
        }

        # Add retracement levels
        for label, price in fib_data['retracement_levels'].items():
            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_end],
                    y=[price, price],
                    mode='lines',
                    name=f'Fib {label}',
                    line=dict(
                        color=fib_colors.get(label, 'rgba(150, 150, 150, 0.3)'),
                        width=1,
                        dash='dash' if label not in ['0.5', '0.618'] else 'solid'
                    ),
                    hovertemplate=f'Fib {label}: {price:.2f}<extra></extra>',
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add label annotation
            fig.add_annotation(
                x=x_end,
                y=price,
                text=f'{label}',
                showarrow=False,
                xanchor='left',
                font=dict(size=9, color='white'),
                bgcolor=fib_colors.get(label, 'rgba(150, 150, 150, 0.5)'),
                borderpad=2,
                row=row, col=col
            )

        # Add extension levels (if calculated)
        for label, price in fib_data.get('extension_levels', {}).items():
            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_end],
                    y=[price, price],
                    mode='lines',
                    name=f'Fib Ext {label}',
                    line=dict(
                        color=fib_colors.get(label, 'rgba(150, 150, 150, 0.3)'),
                        width=1,
                        dash='dot'
                    ),
                    hovertemplate=f'Fib Ext {label}: {price:.2f}<extra></extra>',
                    showlegend=False
                ),
                row=row, col=col
            )

    def _add_patterns(self, fig, df, patterns, row, col):
        """Add geometrical pattern overlays to chart"""
        # Head and Shoulders
        for pattern in patterns.get('head_and_shoulders', []):
            # Draw shoulders and head
            fig.add_trace(
                go.Scatter(
                    x=[pattern['left_shoulder']['time'], pattern['head']['time'], pattern['right_shoulder']['time']],
                    y=[pattern['left_shoulder']['price'], pattern['head']['price'], pattern['right_shoulder']['price']],
                    mode='lines+markers',
                    name='H&S Pattern',
                    line=dict(color='red', width=2),
                    marker=dict(size=10, color='red'),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Draw neckline
            x_start = pattern['left_shoulder']['time']
            x_end = df.index[-1]
            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_end],
                    y=[pattern['neckline_price'], pattern['neckline_price']],
                    mode='lines',
                    name='H&S Neckline',
                    line=dict(color='yellow', width=2, dash='dash'),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add text annotation
            fig.add_annotation(
                x=pattern['head']['time'],
                y=pattern['head']['price'],
                text='H&S',
                showarrow=True,
                arrowhead=2,
                arrowcolor='red',
                font=dict(color='white', size=12),
                bgcolor='red',
                row=row, col=col
            )

        # Inverse Head and Shoulders
        for pattern in patterns.get('inverse_head_and_shoulders', []):
            # Draw shoulders and head
            fig.add_trace(
                go.Scatter(
                    x=[pattern['left_shoulder']['time'], pattern['head']['time'], pattern['right_shoulder']['time']],
                    y=[pattern['left_shoulder']['price'], pattern['head']['price'], pattern['right_shoulder']['price']],
                    mode='lines+markers',
                    name='Inv H&S Pattern',
                    line=dict(color='green', width=2),
                    marker=dict(size=10, color='green'),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Draw neckline
            x_start = pattern['left_shoulder']['time']
            x_end = df.index[-1]
            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_end],
                    y=[pattern['neckline_price'], pattern['neckline_price']],
                    mode='lines',
                    name='Inv H&S Neckline',
                    line=dict(color='lime', width=2, dash='dash'),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add text annotation
            fig.add_annotation(
                x=pattern['head']['time'],
                y=pattern['head']['price'],
                text='INV H&S',
                showarrow=True,
                arrowhead=2,
                arrowcolor='green',
                font=dict(color='white', size=12),
                bgcolor='green',
                row=row, col=col
            )

        # Triangles
        for pattern in patterns.get('triangles', []):
            upper_tl = pattern['upper_trendline']
            lower_tl = pattern['lower_trendline']

            # Draw upper trendline
            fig.add_trace(
                go.Scatter(
                    x=[pt['time'] for pt in upper_tl],
                    y=[pt['price'] for pt in upper_tl],
                    mode='lines',
                    name=f'{pattern["type"]} Upper',
                    line=dict(color='orange', width=2, dash='dash'),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Draw lower trendline
            fig.add_trace(
                go.Scatter(
                    x=[pt['time'] for pt in lower_tl],
                    y=[pt['price'] for pt in lower_tl],
                    mode='lines',
                    name=f'{pattern["type"]} Lower',
                    line=dict(color='orange', width=2, dash='dash'),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add pattern label
            mid_x = upper_tl[len(upper_tl)//2]['time']
            mid_y = (upper_tl[len(upper_tl)//2]['price'] + lower_tl[len(lower_tl)//2]['price']) / 2
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=pattern['type'].replace('_', ' '),
                showarrow=False,
                font=dict(color='white', size=10),
                bgcolor='orange',
                row=row, col=col
            )

        # Flags and Pennants
        for pattern in patterns.get('flags_pennants', []):
            consolidation_highs = pattern['consolidation_highs']
            consolidation_lows = pattern['consolidation_lows']

            if consolidation_highs and consolidation_lows:
                # Draw consolidation box
                fig.add_trace(
                    go.Scatter(
                        x=[pt['time'] for pt in consolidation_highs],
                        y=[pt['price'] for pt in consolidation_highs],
                        mode='lines',
                        name=f'{pattern["type"]} Upper',
                        line=dict(color='purple', width=2, dash='dash'),
                        showlegend=False
                    ),
                    row=row, col=col
                )

                fig.add_trace(
                    go.Scatter(
                        x=[pt['time'] for pt in consolidation_lows],
                        y=[pt['price'] for pt in consolidation_lows],
                        mode='lines',
                        name=f'{pattern["type"]} Lower',
                        line=dict(color='purple', width=2, dash='dash'),
                        showlegend=False
                    ),
                    row=row, col=col
                )

                # Add pattern label
                mid_x = consolidation_highs[0]['time']
                mid_y = (consolidation_highs[0]['price'] + consolidation_lows[0]['price']) / 2
                fig.add_annotation(
                    x=mid_x,
                    y=mid_y,
                    text=pattern['type'].replace('_', ' '),
                    showarrow=True,
                    arrowhead=2,
                    font=dict(color='white', size=10),
                    bgcolor='purple',
                    row=row, col=col
                )

    def _add_money_flow_profile(self, fig, df, money_flow_data, indicator, row, col):
        """Add Money Flow Profile to chart"""
        if not money_flow_data.get('success'):
            return

        bins = money_flow_data['bins']
        poc_price = money_flow_data['poc_price']
        period_high = money_flow_data['period_high']
        period_low = money_flow_data['period_low']

        # Get chart time range
        x_start = df.index[-min(indicator.lookback, len(df))]
        x_end = df.index[-1]

        # Add POC line if enabled
        if indicator.show_poc == 'Last(Line)' or indicator.show_poc == 'Last(Zone)':
            fig.add_shape(
                type="line",
                x0=x_start,
                x1=x_end,
                y0=poc_price,
                y1=poc_price,
                line=dict(color=indicator.poc_color, width=2),
                name="Money Flow POC",
                row=row, col=col
            )

        # Add POC zone if enabled
        if indicator.show_poc == 'Last(Zone)':
            poc_bin = next((b for b in bins if b['is_poc']), None)
            if poc_bin:
                fig.add_shape(
                    type="rect",
                    x0=x_start,
                    x1=x_end,
                    y0=poc_bin['lower'],
                    y1=poc_bin['upper'],
                    fillcolor='rgba(255, 235, 59, 0.27)',
                    line=dict(width=0),
                    layer="below",
                    row=row, col=col
                )

        # Add consolidation zones
        if indicator.show_consolidation:
            for zone in money_flow_data.get('consolidation_zones', []):
                fig.add_shape(
                    type="rect",
                    x0=x_start,
                    x1=x_end,
                    y0=zone['lower'],
                    y1=zone['upper'],
                    fillcolor=indicator.consolidation_color,
                    line=dict(width=0),
                    layer="below",
                    row=row, col=col
                )

        # Add annotation with summary
        if bins:
            summary_text = (
                f"Money Flow Profile<br>"
                f"POC: {poc_price:.2f}<br>"
                f"Range: {period_high:.2f} - {period_low:.2f}<br>"
                f"Bullish: {money_flow_data.get('total_bullish', 0) / money_flow_data.get('total_volume', 1) * 100:.1f}%"
            )
            fig.add_annotation(
                x=x_end,
                y=period_high,
                text=summary_text,
                showarrow=False,
                font=dict(size=9, color='rgba(255, 235, 59, 0.8)'),
                bgcolor='rgba(0, 0, 0, 0.6)',
                bordercolor='rgba(255, 235, 59, 0.5)',
                borderwidth=1,
                xanchor='right',
                yanchor='top',
                row=row, col=col
            )

    def _add_deltaflow_profile(self, fig, df, deltaflow_data, indicator, row, col):
        """Add DeltaFlow Volume Profile to chart"""
        if not deltaflow_data.get('success'):
            return

        bins = deltaflow_data['bins']
        poc_price = deltaflow_data['poc_price']
        period_high = deltaflow_data['period_high']
        period_low = deltaflow_data['period_low']
        overall_delta = deltaflow_data['overall_delta']

        # Get chart time range
        x_start = df.index[-min(indicator.lookback, len(df))]
        x_end = df.index[-1]

        # Add POC line if enabled
        if indicator.show_poc:
            fig.add_shape(
                type="line",
                x0=x_start,
                x1=x_end,
                y0=poc_price,
                y1=poc_price,
                line=dict(color=indicator.poc_color, width=2, dash='dot'),
                name="DeltaFlow POC",
                row=row, col=col
            )

        # Add horizontal lines for strong delta levels
        for bin_data in bins:
            if abs(bin_data['delta_pct']) > 30:  # Strong delta levels
                # Bullish delta levels (green)
                if bin_data['delta_pct'] > 30:
                    fig.add_shape(
                        type="line",
                        x0=x_start,
                        x1=x_end,
                        y0=bin_data['mid'],
                        y1=bin_data['mid'],
                        line=dict(color='rgba(0, 150, 136, 0.3)', width=1, dash='dash'),
                        row=row, col=col
                    )
                # Bearish delta levels (red)
                elif bin_data['delta_pct'] < -30:
                    fig.add_shape(
                        type="line",
                        x0=x_start,
                        x1=x_end,
                        y0=bin_data['mid'],
                        y1=bin_data['mid'],
                        line=dict(color='rgba(230, 150, 30, 0.3)', width=1, dash='dash'),
                        row=row, col=col
                    )

        # Add annotation with summary
        delta_sentiment = "BULLISH" if overall_delta > 10 else "BEARISH" if overall_delta < -10 else "NEUTRAL"
        summary_text = (
            f"DeltaFlow Profile<br>"
            f"Sentiment: {delta_sentiment}<br>"
            f"Delta: {overall_delta:+.1f}%<br>"
            f"POC: {poc_price:.2f}"
        )
        fig.add_annotation(
            x=x_end,
            y=period_low,
            text=summary_text,
            showarrow=False,
            font=dict(size=9, color='rgba(0, 183, 255, 0.8)'),
            bgcolor='rgba(0, 0, 0, 0.6)',
            bordercolor='rgba(0, 183, 255, 0.5)',
            borderwidth=1,
            xanchor='right',
            yanchor='bottom',
            row=row, col=col
        )
