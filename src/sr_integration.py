"""
Integration module for S/R Strength Tracker with Comprehensive S/R Analysis

Connects the real-time S/R analysis with historical tracking and ML predictions
"""

import streamlit as st
from typing import Dict, Optional, List, Tuple
import pandas as pd
from datetime import datetime

from src.sr_strength_tracker import SRStrengthTracker, SRStrengthTrend, SRTransition
from src.comprehensive_sr_analysis import analyze_sr_strength_comprehensive


def initialize_sr_tracker():
    """Initialize S/R tracker in session state if not exists"""
    if 'sr_strength_tracker' not in st.session_state:
        st.session_state.sr_strength_tracker = SRStrengthTracker(
            max_history_hours=24,
            transition_threshold=0.6
        )
    return st.session_state.sr_strength_tracker


def update_sr_tracker_from_analysis(
    features: Dict,
    support_price: float,
    resistance_price: float,
    current_price: float
):
    """
    Update S/R tracker with current analysis results

    Args:
        features: Feature dictionary from XGBoost/analysis
        support_price: Current support level
        resistance_price: Current resistance level
        current_price: Current market price
    """
    tracker = initialize_sr_tracker()

    # Get comprehensive analysis
    analysis = analyze_sr_strength_comprehensive(features)

    # Extract volume (if available)
    volume = features.get('volume', 0)

    # Add support observation
    if support_price > 0:
        tracker.add_observation(
            price_level=support_price,
            level_type='support',
            strength=analysis['support_strength'],
            status=analysis['support_status'],
            factors=analysis['reasons'],
            current_price=current_price,
            volume=volume
        )

    # Add resistance observation
    if resistance_price > 0:
        tracker.add_observation(
            price_level=resistance_price,
            level_type='resistance',
            strength=analysis['resistance_strength'],
            status=analysis['resistance_status'],
            factors=analysis['reasons'],
            current_price=current_price,
            volume=volume
        )


def get_sr_trend_analysis(
    support_price: float,
    resistance_price: float
) -> Dict:
    """
    Get comprehensive trend analysis for current S/R levels

    Returns:
        Dict with 'support_trend', 'resistance_trend', and 'all_trends'
    """
    tracker = initialize_sr_tracker()

    result = {
        'support_trend': None,
        'resistance_trend': None,
        'all_trends': [],
        'has_data': False
    }

    # Get trend for support
    if support_price > 0:
        support_trend = tracker.analyze_trend(support_price, 'support')
        result['support_trend'] = support_trend
        if support_trend:
            result['has_data'] = True

    # Get trend for resistance
    if resistance_price > 0:
        resistance_trend = tracker.analyze_trend(resistance_price, 'resistance')
        result['resistance_trend'] = resistance_trend
        if resistance_trend:
            result['has_data'] = True

    # Get all trends (for display in collapsible section)
    all_trends = tracker.get_all_levels_analysis()
    result['all_trends'] = all_trends

    return result


def get_sr_transitions(hours: int = 6) -> List[SRTransition]:
    """Get recent S/R transitions"""
    tracker = initialize_sr_tracker()
    return tracker.get_recent_transitions(hours)


def get_sr_summary_stats() -> Dict:
    """Get summary statistics about tracked S/R levels"""
    tracker = initialize_sr_tracker()
    return tracker.get_summary_stats()


def format_trend_for_display(trend: Optional[SRStrengthTrend]) -> Dict:
    """
    Format SRStrengthTrend object for display in UI

    Returns:
        Dict with formatted data
    """
    if not trend:
        return None

    return {
        'level_type': trend.level_type,
        'price_level': trend.price_level,
        'current_strength': trend.current_strength,
        'trend': trend.trend,
        'trend_confidence': trend.trend_confidence,
        'strength_change_rate': trend.strength_change_rate,
        'prediction_1h': trend.prediction_1h,
        'prediction_4h': trend.prediction_4h,
        'reasons': trend.reasons
    }


def format_transition_for_display(transition: SRTransition) -> Dict:
    """
    Format SRTransition object for display in UI

    Returns:
        Dict with formatted data
    """
    return {
        'transition_type': transition.transition_type,
        'price_level': transition.price_level,
        'start_time': transition.start_time,
        'end_time': transition.end_time,
        'confidence': transition.confidence,
        'strength_before': transition.strength_before,
        'strength_after': transition.strength_after,
        'reason': transition.reason
    }


def get_sr_data_for_signal_display(
    features: Dict,
    support_price: float,
    resistance_price: float,
    current_price: float
) -> Tuple[Dict, List[Dict]]:
    """
    Get complete S/R data for signal display

    Returns:
        Tuple of (trend_analysis_dict, transitions_list)
    """
    # Update tracker with current data
    update_sr_tracker_from_analysis(features, support_price, resistance_price, current_price)

    # Get trend analysis
    trend_analysis = get_sr_trend_analysis(support_price, resistance_price)

    # Format trends for display
    formatted_trends = {
        'trends': [
            format_trend_for_display(t) for t in trend_analysis['all_trends']
            if t is not None
        ],
        'support_trend': format_trend_for_display(trend_analysis['support_trend']),
        'resistance_trend': format_trend_for_display(trend_analysis['resistance_trend']),
        'has_data': trend_analysis['has_data']
    }

    # Get transitions
    transitions = get_sr_transitions(hours=6)
    formatted_transitions = [
        format_transition_for_display(t) for t in transitions
    ]

    return formatted_trends, formatted_transitions


def display_sr_trend_summary(support_price: float, resistance_price: float):
    """
    Display a compact S/R trend summary (for main view)

    Shows only the most important trend information
    """
    trend_analysis = get_sr_trend_analysis(support_price, resistance_price)

    if not trend_analysis['has_data']:
        st.info("🔄 S/R trend tracking initializing... (needs historical data)")
        return

    col1, col2 = st.columns(2)

    # Support trend
    with col1:
        support_trend = trend_analysis['support_trend']
        if support_trend:
            trend_emoji = {
                'STRENGTHENING': '📈🟢',
                'WEAKENING': '📉🔴',
                'STABLE': '➡️🟡',
                'TRANSITIONING': '🔄🟣'
            }.get(support_trend.trend, '⚪')

            st.markdown(f"**Support @ ₹{support_price:,.0f}**")
            st.markdown(f"{trend_emoji} {support_trend.trend}")
            st.markdown(f"Strength: {support_trend.current_strength:.0f}% → {support_trend.prediction_1h:.0f}% (1h)")

    # Resistance trend
    with col2:
        resistance_trend = trend_analysis['resistance_trend']
        if resistance_trend:
            trend_emoji = {
                'STRENGTHENING': '📈🟢',
                'WEAKENING': '📉🔴',
                'STABLE': '➡️🟡',
                'TRANSITIONING': '🔄🟣'
            }.get(resistance_trend.trend, '⚪')

            st.markdown(f"**Resistance @ ₹{resistance_price:,.0f}**")
            st.markdown(f"{trend_emoji} {resistance_trend.trend}")
            st.markdown(f"Strength: {resistance_trend.current_strength:.0f}% → {resistance_trend.prediction_1h:.0f}% (1h)")

    # Show recent transitions
    transitions = get_sr_transitions(hours=2)
    if transitions:
        st.warning(f"⚠️ {len(transitions)} S/R transition(s) detected in last 2 hours!")


def get_sr_strength_indicator(
    level_type: str,
    price: float
) -> Tuple[str, str, float]:
    """
    Get a simple strength indicator for a S/R level

    Returns:
        Tuple of (emoji, trend_text, confidence)
    """
    tracker = initialize_sr_tracker()
    trend = tracker.analyze_trend(price, level_type)

    if not trend:
        return '⚪', 'NO DATA', 0.0

    emoji_map = {
        'STRENGTHENING': '📈🟢',
        'WEAKENING': '📉🔴',
        'STABLE': '➡️🟡',
        'TRANSITIONING': '🔄🟣'
    }

    emoji = emoji_map.get(trend.trend, '⚪')
    trend_text = f"{trend.trend} ({trend.current_strength:.0f}%)"

    return emoji, trend_text, trend.trend_confidence


# Streamlit display helper
def display_sr_tracker_stats():
    """Display S/R tracker statistics (useful for debugging/monitoring)"""
    stats = get_sr_summary_stats()

    st.markdown("### 📊 S/R Tracker Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Levels Tracked", stats['total_levels'])

    with col2:
        st.metric("Observations", f"{stats['total_observations']} ({stats['observations_last_hour']}/hr)")

    with col3:
        st.metric("Transitions Detected", stats['total_transitions'])

    col4, col5 = st.columns(2)

    with col4:
        st.metric("Strengthening", stats['strengthening_levels'])

    with col5:
        st.metric("Weakening", stats['weakening_levels'])


# Example usage
if __name__ == "__main__":
    # This would be called from the main signal display
    features = {
        'price_change_1': 0.3,
        'price_change_5': 0.5,
        'htf_nearest_support_distance_pct': 0.2,
        'vob_major_support_distance_pct': 0.3,
        'volume_concentration': 0.7,
        'volume_buy_sell_ratio': 1.8,
    }

    support = 26100
    resistance = 26300
    current = 26150

    # Update and get data
    trend_data, transitions = get_sr_data_for_signal_display(features, support, resistance, current)

    print("Trend Analysis:", trend_data)
    print("Transitions:", transitions)
