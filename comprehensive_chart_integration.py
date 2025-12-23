"""
Comprehensive Chart Integration Module
Integrates data from ALL tabs into Advanced Chart Analysis

This module enriches the chart with:
- OI Walls (Max PUT/CALL OI strikes)
- GEX Walls (Gamma exposure levels)
- Bias Analysis indicators
- Enhanced Market Data (VIX, sector rotation)
- ML predictions (XGBoost)
- Money Flow & DeltaFlow signals
"""

import pandas as pd
import streamlit as st
from typing import Dict, Optional, Any, List
import logging

logger = logging.getLogger(__name__)


class ComprehensiveChartIntegrator:
    """
    Integrates data from all tabs into chart analysis
    Provides enriched chart with institutional levels and ML insights
    """

    def __init__(self):
        self.logger = logger

    def gather_all_tab_data(self) -> Dict[str, Any]:
        """
        Gather data from ALL tabs in the app

        Returns:
            Dict with all available data sources
        """
        comprehensive_data = {}

        # ==========================================
        # TAB 1: Overall Market Sentiment
        # ==========================================
        if 'enhanced_market_data' in st.session_state:
            comprehensive_data['enhanced_market'] = st.session_state.enhanced_market_data

        if 'money_flow_signals' in st.session_state:
            comprehensive_data['money_flow'] = st.session_state.money_flow_signals

        if 'deltaflow_signals' in st.session_state:
            comprehensive_data['deltaflow'] = st.session_state.deltaflow_signals

        # ==========================================
        # TAB 5: Bias Analysis Pro
        # ==========================================
        if 'bias_analysis_results' in st.session_state:
            comprehensive_data['bias_analysis'] = st.session_state.bias_analysis_results

        # ==========================================
        # TAB 6: Option Chain Analysis
        # ==========================================
        if 'overall_option_data' in st.session_state:
            comprehensive_data['option_chain'] = st.session_state.overall_option_data

        # ==========================================
        # TAB 7: Chart Analysis (HTF S/R)
        # ==========================================
        if 'htf_sr_levels' in st.session_state:
            comprehensive_data['htf_sr'] = st.session_state.htf_sr_levels

        if 'htf_nearest_support' in st.session_state:
            comprehensive_data['htf_nearest_support'] = st.session_state.htf_nearest_support

        if 'htf_nearest_resistance' in st.session_state:
            comprehensive_data['htf_nearest_resistance'] = st.session_state.htf_nearest_resistance

        # ==========================================
        # TAB 8: NIFTY Option Screener (OI/GEX)
        # ==========================================
        if 'nifty_screener_data' in st.session_state:
            screener_data = st.session_state.nifty_screener_data

            # OI Walls (Max PUT/CALL OI strikes)
            if 'oi_pcr_metrics' in screener_data:
                comprehensive_data['oi_walls'] = {
                    'max_pe_strike': screener_data['oi_pcr_metrics'].get('max_pe_strike'),
                    'max_ce_strike': screener_data['oi_pcr_metrics'].get('max_ce_strike'),
                    'pcr': screener_data['oi_pcr_metrics'].get('pcr', 1.0)
                }

            # GEX Walls (Gamma Exposure)
            if 'gamma_exposure' in screener_data:
                comprehensive_data['gex'] = screener_data['gamma_exposure']

            # Market Depth
            if 'market_depth' in screener_data:
                comprehensive_data['market_depth'] = screener_data['market_depth']

            # VOB Signals
            if 'vob_signals' in screener_data:
                comprehensive_data['vob_signals'] = screener_data['vob_signals']

            # ATM Bias
            if 'atm_bias' in screener_data:
                comprehensive_data['atm_bias'] = screener_data['atm_bias']

            # NIFTY Futures Analysis
            if 'futures_analysis' in screener_data:
                comprehensive_data['futures_analysis'] = screener_data['futures_analysis']

        # ==========================================
        # TAB 9: Enhanced Market Data (VIX, etc.)
        # ==========================================
        if 'enhanced_market_data' in st.session_state:
            enhanced = st.session_state.enhanced_market_data

            # Real VIX (not hardcoded 15.0)
            if 'vix' in enhanced:
                vix_data = enhanced['vix']
                if isinstance(vix_data, dict):
                    comprehensive_data['vix'] = vix_data.get('current', vix_data.get('value', 15.0))
                else:
                    comprehensive_data['vix'] = vix_data
            elif 'india_vix' in enhanced:
                comprehensive_data['vix'] = enhanced['india_vix']
            else:
                comprehensive_data['vix'] = 15.0  # Fallback

            # Sector Rotation
            if 'sector_rotation' in enhanced:
                comprehensive_data['sector_rotation'] = enhanced['sector_rotation']

            # Global Markets
            if 'global_markets' in enhanced:
                comprehensive_data['global_markets'] = enhanced['global_markets']

            # Intermarket
            if 'intermarket' in enhanced:
                comprehensive_data['intermarket'] = enhanced['intermarket']

        # ==========================================
        # TAB 10: Master AI Analysis (XGBoost ML)
        # ==========================================
        if 'ml_prediction_result' in st.session_state:
            comprehensive_data['ml_prediction'] = st.session_state.ml_prediction_result

        # ==========================================
        # Current Price & ATM Strike
        # ==========================================
        if 'nifty_spot_price' in st.session_state:
            comprehensive_data['current_price'] = st.session_state.nifty_spot_price

        if 'nifty_atm_strike' in st.session_state:
            comprehensive_data['atm_strike'] = st.session_state.nifty_atm_strike

        return comprehensive_data

    def extract_institutional_levels(self, data: Dict) -> Dict[str, List[Dict]]:
        """
        Extract institutional support/resistance levels from all sources

        Returns:
            Dict with 'support' and 'resistance' lists
        """
        levels = {
            'support': [],
            'resistance': []
        }

        current_price = data.get('current_price', 0)
        if current_price == 0:
            return levels

        # 1. OI Walls (Max PUT/CALL OI)
        if 'oi_walls' in data:
            oi = data['oi_walls']
            if oi.get('max_pe_strike') and oi['max_pe_strike'] < current_price:
                levels['support'].append({
                    'price': oi['max_pe_strike'],
                    'type': 'OI Wall',
                    'source': 'Max PUT OI',
                    'strength': 'HIGH',
                    'color': '#FF6B6B'
                })
            if oi.get('max_ce_strike') and oi['max_ce_strike'] > current_price:
                levels['resistance'].append({
                    'price': oi['max_ce_strike'],
                    'type': 'OI Wall',
                    'source': 'Max CALL OI',
                    'strength': 'HIGH',
                    'color': '#4ECDC4'
                })

        # 2. GEX Walls (Gamma Exposure)
        if 'gex' in data and 'gamma_walls' in data['gex']:
            for wall in data['gex']['gamma_walls']:
                if isinstance(wall, dict):
                    wall_price = wall.get('strike', 0)
                    if wall_price < current_price:
                        levels['support'].append({
                            'price': wall_price,
                            'type': 'GEX Wall',
                            'source': 'Gamma Support',
                            'strength': 'HIGH',
                            'color': '#FFB347'
                        })
                    elif wall_price > current_price:
                        levels['resistance'].append({
                            'price': wall_price,
                            'type': 'GEX Wall',
                            'source': 'Gamma Resistance',
                            'strength': 'HIGH',
                            'color': '#87CEEB'
                        })

        # 3. HTF S/R Levels
        if 'htf_nearest_support' in data:
            htf_sup = data['htf_nearest_support']
            if isinstance(htf_sup, dict):
                levels['support'].append({
                    'price': htf_sup.get('price', 0),
                    'type': 'HTF Support',
                    'source': f"{htf_sup.get('timeframe', '')}",
                    'strength': 'MEDIUM',
                    'color': '#98D8C8'
                })

        if 'htf_nearest_resistance' in data:
            htf_res = data['htf_nearest_resistance']
            if isinstance(htf_res, dict):
                levels['resistance'].append({
                    'price': htf_res.get('price', 0),
                    'type': 'HTF Resistance',
                    'source': f"{htf_res.get('timeframe', '')}",
                    'strength': 'MEDIUM',
                    'color': '#F7DC6F'
                })

        # 4. VOB Levels
        if 'vob_signals' in data:
            for vob in data['vob_signals']:
                if isinstance(vob, dict):
                    vob_price = vob.get('price', 0)
                    vob_strength = vob.get('strength', 'Medium')
                    if vob_price < current_price:
                        levels['support'].append({
                            'price': vob_price,
                            'type': 'VOB Support',
                            'source': f"VOB ({vob_strength})",
                            'strength': 'MEDIUM' if vob_strength == 'Major' else 'LOW',
                            'color': '#BB8FCE'
                        })
                    elif vob_price > current_price:
                        levels['resistance'].append({
                            'price': vob_price,
                            'type': 'VOB Resistance',
                            'source': f"VOB ({vob_strength})",
                            'strength': 'MEDIUM' if vob_strength == 'Major' else 'LOW',
                            'color': '#85C1E2'
                        })

        # Sort by price
        levels['support'] = sorted(levels['support'], key=lambda x: x['price'], reverse=True)
        levels['resistance'] = sorted(levels['resistance'], key=lambda x: x['price'])

        return levels

    def get_market_sentiment_summary(self, data: Dict) -> Dict:
        """
        Summarize market sentiment from all sources

        Returns:
            Dict with sentiment analysis
        """
        sentiment = {
            'overall': 'NEUTRAL',
            'score': 0,
            'confidence': 50,
            'sources': []
        }

        # Bias Analysis
        if 'bias_analysis' in data:
            bias = data['bias_analysis']
            if bias.get('success'):
                sentiment['sources'].append({
                    'name': 'Bias Analysis',
                    'verdict': bias.get('overall_bias', 'NEUTRAL'),
                    'score': bias.get('overall_score', 0),
                    'confidence': bias.get('overall_confidence', 50)
                })

        # ATM Bias
        if 'atm_bias' in data:
            atm = data['atm_bias']
            sentiment['sources'].append({
                'name': 'ATM Bias',
                'verdict': atm.get('verdict', 'NEUTRAL'),
                'score': atm.get('total_score', 0)
            })

        # ML Prediction
        if 'ml_prediction' in data:
            ml = data['ml_prediction']
            if hasattr(ml, 'prediction'):
                sentiment['sources'].append({
                    'name': 'XGBoost ML',
                    'verdict': ml.prediction,
                    'confidence': ml.confidence if hasattr(ml, 'confidence') else 50
                })

        # Money Flow
        if 'money_flow' in data:
            mf = data['money_flow']
            if 'signal' in mf:
                sentiment['sources'].append({
                    'name': 'Money Flow',
                    'verdict': mf['signal']
                })

        # Calculate overall sentiment
        if sentiment['sources']:
            bullish_count = sum(1 for s in sentiment['sources'] if 'BULL' in str(s.get('verdict', '')).upper())
            bearish_count = sum(1 for s in sentiment['sources'] if 'BEAR' in str(s.get('verdict', '')).upper())

            if bullish_count > bearish_count:
                sentiment['overall'] = 'BULLISH'
                sentiment['score'] = (bullish_count / len(sentiment['sources'])) * 100
            elif bearish_count > bullish_count:
                sentiment['overall'] = 'BEARISH'
                sentiment['score'] = -(bearish_count / len(sentiment['sources'])) * 100
            else:
                sentiment['overall'] = 'NEUTRAL'
                sentiment['score'] = 0

        return sentiment

    def create_comprehensive_chart_params(self) -> Dict:
        """
        Create comprehensive parameters for chart with ALL tab data

        Returns:
            Dict with enriched chart parameters
        """
        # Gather all data
        all_data = self.gather_all_tab_data()

        # Extract institutional levels
        institutional_levels = self.extract_institutional_levels(all_data)

        # Get market sentiment
        market_sentiment = self.get_market_sentiment_summary(all_data)

        # Compile comprehensive params
        params = {
            'institutional_levels': institutional_levels,
            'market_sentiment': market_sentiment,
            'vix': all_data.get('vix', 15.0),
            'pcr': all_data.get('oi_walls', {}).get('pcr', 1.0),
            'current_price': all_data.get('current_price', 0),
            'atm_strike': all_data.get('atm_strike', 0),
            'raw_data': all_data  # Include all raw data for advanced uses
        }

        return params


# ==========================================
# Helper Functions
# ==========================================

def add_institutional_levels_to_chart(fig, institutional_levels, row=1, col=1):
    """
    Add institutional support/resistance levels to Plotly chart

    Args:
        fig: Plotly figure
        institutional_levels: Dict with 'support' and 'resistance' lists
        row: Subplot row
        col: Subplot column
    """
    import plotly.graph_objects as go

    # Add support levels
    for level in institutional_levels['support']:
        fig.add_hline(
            y=level['price'],
            line_dash="dash",
            line_color=level['color'],
            line_width=2,
            annotation_text=f"{level['type']}: {level['price']}",
            annotation_position="right",
            row=row,
            col=col
        )

    # Add resistance levels
    for level in institutional_levels['resistance']:
        fig.add_hline(
            y=level['price'],
            line_dash="dash",
            line_color=level['color'],
            line_width=2,
            annotation_text=f"{level['type']}: {level['price']}",
            annotation_position="right",
            row=row,
            col=col
        )

    return fig


def display_comprehensive_chart_info(params: Dict):
    """
    Display comprehensive chart information in Streamlit sidebar

    Args:
        params: Comprehensive chart parameters
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Institutional Levels")

    # Support levels
    st.sidebar.markdown("**🟢 Support:**")
    for level in params['institutional_levels']['support'][:3]:  # Top 3
        st.sidebar.markdown(
            f"- **{level['type']}**: ₹{level['price']:,.0f} "
            f"({level['source']}) - {level['strength']}"
        )

    # Resistance levels
    st.sidebar.markdown("**🔴 Resistance:**")
    for level in params['institutional_levels']['resistance'][:3]:  # Top 3
        st.sidebar.markdown(
            f"- **{level['type']}**: ₹{level['price']:,.0f} "
            f"({level['source']}) - {level['strength']}"
        )

    # Market Sentiment
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Market Sentiment")
    sentiment = params['market_sentiment']
    sentiment_emoji = "🟢" if sentiment['overall'] == "BULLISH" else ("🔴" if sentiment['overall'] == "BEARISH" else "⚖️")
    st.sidebar.markdown(f"**{sentiment_emoji} {sentiment['overall']}** (Score: {sentiment['score']:.1f})")

    # Market Data
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Market Data")
    st.sidebar.markdown(f"**VIX:** {params['vix']:.2f}")
    st.sidebar.markdown(f"**PCR:** {params['pcr']:.2f}")
    st.sidebar.markdown(f"**Current:** ₹{params['current_price']:,.2f}")
