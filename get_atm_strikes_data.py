"""
Utility script to retrieve and display ATM ±2 Strikes data
Shows all 12 bias metrics for each strike in the ATM window
"""

import streamlit as st
import pandas as pd
from datetime import datetime


def get_atm_strikes_data():
    """
    Retrieve ATM ±2 Strikes data from session state
    Returns a formatted dictionary with all metrics
    """

    # Check if option screener data exists
    if 'nifty_option_screener_data' not in st.session_state:
        return {
            'status': 'NOT_LOADED',
            'message': 'Option chain data not yet loaded. Please load data first.',
            'data': None
        }

    option_data = st.session_state.nifty_option_screener_data
    strike_analyses = option_data.get('strike_analyses', [])
    atm_bias = option_data.get('atm_bias', {})

    if not strike_analyses:
        return {
            'status': 'NO_DATA',
            'message': 'No strike analysis data available',
            'data': None
        }

    # Get ATM strike
    atm_strike = atm_bias.get('atm_strike', 0)

    # Find ATM strike analysis
    atm_analysis = next((a for a in strike_analyses if a["strike_price"] == atm_strike), None)

    # Prepare formatted data
    formatted_data = {
        'status': 'SUCCESS',
        'last_updated': option_data.get('last_updated'),
        'atm_strike': atm_strike,
        'strikes': []
    }

    # Format each strike's data
    for strike_data in strike_analyses:
        strike_info = {
            'strike_price': strike_data['strike_price'],
            'is_atm': strike_data['strike_price'] == atm_strike,
            'total_bias_score': strike_data['total_bias'],
            'verdict': strike_data['verdict'],
            'verdict_color': strike_data['verdict_color'],
            'metrics': {}
        }

        # Add all 12 bias metrics
        metric_names = {
            'OI': 'Open Interest Bias',
            'ChgOI': 'Change in OI Bias',
            'Volume': 'Volume Bias',
            'Delta': 'Delta Bias',
            'Gamma': 'Gamma Bias',
            'Premium': 'Premium Bias',
            'IV': 'Implied Volatility Bias',
            'DeltaExp': 'Delta Exposure',
            'GammaExp': 'Gamma Exposure',
            'IVSkew': 'IV Skew Bias',
            'OIChgRate': 'OI Change Rate',
            'PCR': 'Put-Call Ratio at Strike'
        }

        for metric_key, metric_name in metric_names.items():
            strike_info['metrics'][metric_key] = {
                'name': metric_name,
                'score': strike_data['bias_scores'].get(metric_key, 0),
                'emoji': strike_data['bias_emojis'].get(metric_key, '⚖️'),
                'interpretation': strike_data['bias_interpretations'].get(metric_key, 'N/A')
            }

        formatted_data['strikes'].append(strike_info)

    # Calculate ATM verdict summary
    if atm_analysis:
        bullish_metrics = sum(1 for score in atm_analysis["bias_scores"].values() if score > 0)
        bearish_metrics = sum(1 for score in atm_analysis["bias_scores"].values() if score < 0)
        total_metrics = len(atm_analysis["bias_scores"])

        formatted_data['atm_summary'] = {
            'verdict': atm_analysis['verdict'],
            'total_bias_score': atm_analysis['total_bias'],
            'bullish_metrics': bullish_metrics,
            'bearish_metrics': bearish_metrics,
            'neutral_metrics': total_metrics - bullish_metrics - bearish_metrics,
            'total_metrics': total_metrics,
            'bullish_percentage': (bullish_metrics / total_metrics * 100) if total_metrics > 0 else 0,
            'bearish_percentage': (bearish_metrics / total_metrics * 100) if total_metrics > 0 else 0
        }

    return formatted_data


def display_atm_strikes_data_detailed():
    """
    Display ATM ±2 Strikes data in a detailed, readable format
    """
    data = get_atm_strikes_data()

    if data['status'] != 'SUCCESS':
        st.warning(f"⚠️ {data['message']}")
        return

    st.markdown("## 📊 ATM ±2 Strikes - Complete Data Dump")
    st.markdown("---")

    # Display last updated
    if data['last_updated']:
        st.info(f"📅 Last Updated: {data['last_updated'].strftime('%Y-%m-%d %H:%M:%S IST')}")

    # Display ATM summary
    if 'atm_summary' in data:
        summary = data['atm_summary']
        st.markdown("### 🎯 ATM STRIKE VERDICT")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("ATM Strike", data['atm_strike'])
            st.metric("Total Bias Score", f"{summary['total_bias_score']:+.2f}")
            st.markdown(f"**Verdict:** {summary['verdict']}")

        with col2:
            st.metric("🐂 Bullish Metrics", f"{summary['bullish_metrics']}/{summary['total_metrics']}")
            st.metric("Bullish %", f"{summary['bullish_percentage']:.1f}%")

        with col3:
            st.metric("🐻 Bearish Metrics", f"{summary['bearish_metrics']}/{summary['total_metrics']}")
            st.metric("Bearish %", f"{summary['bearish_percentage']:.1f}%")

    st.markdown("---")

    # Display each strike's data
    for strike in data['strikes']:
        strike_emoji = "🎯" if strike['is_atm'] else "📍"
        st.markdown(f"### {strike_emoji} Strike: {strike['strike_price']} {'(ATM)' if strike['is_atm'] else ''}")
        st.markdown(f"**Verdict:** {strike['verdict']} | **Total Score:** {strike['total_bias_score']:+.2f}")

        # Create DataFrame for metrics
        metrics_data = []
        for metric_key, metric_info in strike['metrics'].items():
            metrics_data.append({
                'Metric': metric_info['name'],
                'Emoji': metric_info['emoji'],
                'Score': f"{metric_info['score']:+.1f}",
                'Interpretation': metric_info['interpretation']
            })

        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("---")


def export_atm_strikes_to_json():
    """
    Export ATM ±2 Strikes data to JSON format
    Returns: JSON string
    """
    import json

    data = get_atm_strikes_data()

    if data['status'] != 'SUCCESS':
        return json.dumps({'error': data['message']}, indent=2)

    # Convert datetime to string for JSON serialization
    if data['last_updated']:
        data['last_updated'] = data['last_updated'].strftime('%Y-%m-%d %H:%M:%S IST')

    return json.dumps(data, indent=2)


def get_atm_strike_metrics_summary():
    """
    Get a concise summary of ATM strike metrics
    Returns: Dictionary with metric counts and interpretations
    """
    data = get_atm_strikes_data()

    if data['status'] != 'SUCCESS':
        return None

    if 'atm_summary' not in data:
        return None

    # Find ATM strike data
    atm_strike_data = next((s for s in data['strikes'] if s['is_atm']), None)

    if not atm_strike_data:
        return None

    # Get metric interpretations
    metrics_breakdown = {
        'bullish': [],
        'bearish': [],
        'neutral': []
    }

    for metric_key, metric_info in atm_strike_data['metrics'].items():
        score = metric_info['score']
        metric_detail = {
            'name': metric_info['name'],
            'score': score,
            'interpretation': metric_info['interpretation']
        }

        if score > 0:
            metrics_breakdown['bullish'].append(metric_detail)
        elif score < 0:
            metrics_breakdown['bearish'].append(metric_detail)
        else:
            metrics_breakdown['neutral'].append(metric_detail)

    return {
        'atm_strike': data['atm_strike'],
        'verdict': data['atm_summary']['verdict'],
        'total_score': data['atm_summary']['total_bias_score'],
        'bullish_count': data['atm_summary']['bullish_metrics'],
        'bearish_count': data['atm_summary']['bearish_metrics'],
        'neutral_count': data['atm_summary']['neutral_metrics'],
        'metrics_breakdown': metrics_breakdown
    }


# Example usage in Streamlit app
if __name__ == "__main__":
    st.set_page_config(page_title="ATM Strikes Data Viewer", layout="wide")

    st.title("📊 ATM ±2 Strikes Data Viewer")

    tab1, tab2, tab3 = st.tabs(["📊 Detailed View", "📋 Summary", "💾 Export JSON"])

    with tab1:
        display_atm_strikes_data_detailed()

    with tab2:
        summary = get_atm_strike_metrics_summary()
        if summary:
            st.markdown("### 🎯 ATM Strike Metrics Summary")
            st.json(summary)
        else:
            st.warning("⚠️ No ATM strike data available")

    with tab3:
        st.markdown("### 💾 Export Data as JSON")
        json_data = export_atm_strikes_to_json()
        st.code(json_data, language='json')
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"atm_strikes_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
