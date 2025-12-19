"""
Overall Market Sentiment Analysis

Aggregates bias data from all sources to provide a comprehensive market sentiment view

Data Sources (from Tab 0 - Overall Market Sentiment Dashboard):
1. Stock Performance (Market Breadth) - Weight: 2.0
2. Technical Indicators (Bias Analysis Pro - 13 indicators) - Weight: 3.0
3. ATM Strike Verdict (ATM ±2 Strikes - 12 Bias Metrics) - Weight: 3.5
4. PCR/OI Analysis (Put-Call Ratio Sentiment) - Weight: 2.5
5. Sector Rotation Analysis (Sector Rotation Bias) - Weight: 3.0

Note: All option chain data comes from NiftyOptionScreener.py displayed on Tab 0
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import time
import asyncio
import os
from market_hours_scheduler import is_within_trading_hours, scheduler
from ai_analysis_adapter import run_ai_analysis, shutdown_ai_engine
from perplexity_market_insights import render_market_insights_widget

# Import option screener display function
try:
    from NiftyOptionScreener import display_overall_market_sentiment_summary
except ImportError:
    display_overall_market_sentiment_summary = None


def calculate_stock_performance_sentiment(stock_data):
    """
    Calculate sentiment from individual stock performance
    Returns: dict with sentiment, score, and details
    """
    if not stock_data:
        return None

    bullish_stocks = 0
    bearish_stocks = 0
    neutral_stocks = 0

    total_weighted_change = 0
    total_weight = 0

    for stock in stock_data:
        change_pct = stock.get('change_pct', 0)
        weight = stock.get('weight', 1)

        total_weighted_change += change_pct * weight
        total_weight += weight

        if change_pct > 0.5:
            bullish_stocks += 1
        elif change_pct < -0.5:
            bearish_stocks += 1
        else:
            neutral_stocks += 1

    # Calculate weighted average change
    avg_change = total_weighted_change / total_weight if total_weight > 0 else 0

    # Calculate market breadth
    total_stocks = len(stock_data)
    breadth_pct = (bullish_stocks / total_stocks * 100) if total_stocks > 0 else 50

    # Determine bias
    if avg_change > 1:
        bias = "BULLISH"
    elif avg_change < -1:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    # Calculate score based on weighted change and breadth
    score = avg_change * 5 + (breadth_pct - 50)

    # Calculate confidence
    confidence = min(100, abs(score))

    return {
        'bias': bias,
        'score': score,
        'breadth_pct': breadth_pct,
        'avg_change': avg_change,
        'bullish_stocks': bullish_stocks,
        'bearish_stocks': bearish_stocks,
        'neutral_stocks': neutral_stocks,
        'confidence': confidence,
        'stock_details': stock_data
    }


def calculate_technical_indicators_sentiment(bias_results):
    """
    Calculate sentiment from technical indicators (Bias Analysis Pro)
    Returns: dict with sentiment, score, and details
    """
    if not bias_results:
        return None

    bullish_indicators = 0
    bearish_indicators = 0
    neutral_indicators = 0

    total_weighted_score = 0
    total_weight = 0

    for indicator in bias_results:
        bias = indicator.get('bias', 'NEUTRAL')
        score = indicator.get('score', 0)
        weight = indicator.get('weight', 1)

        total_weighted_score += score * weight
        total_weight += weight

        if 'BULLISH' in bias.upper():
            bullish_indicators += 1
        elif 'BEARISH' in bias.upper():
            bearish_indicators += 1
        else:
            neutral_indicators += 1

    # Calculate overall score
    overall_score = total_weighted_score / total_weight if total_weight > 0 else 0

    # Determine bias
    if overall_score > 20:
        bias = "BULLISH"
    elif overall_score < -20:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    # Calculate confidence based on score magnitude and indicator agreement
    score_magnitude = min(100, abs(overall_score))

    total_indicators = len(bias_results)
    if bias == "BULLISH":
        agreement = bullish_indicators / total_indicators if total_indicators > 0 else 0
    elif bias == "BEARISH":
        agreement = bearish_indicators / total_indicators if total_indicators > 0 else 0
    else:
        agreement = neutral_indicators / total_indicators if total_indicators > 0 else 0

    confidence = score_magnitude * agreement

    return {
        'bias': bias,
        'score': overall_score,
        'bullish_count': bullish_indicators,
        'bearish_count': bearish_indicators,
        'neutral_count': neutral_indicators,
        'total_count': total_indicators,
        'confidence': confidence,
        'indicator_details': bias_results
    }


def calculate_atm_strike_verdict_sentiment():
    """
    Calculate sentiment from ATM Strike Verdict (from Option Screener Tab 0)
    Uses the strike_analyses data (ATM strike verdict) displayed on Overall Market Sentiment dashboard
    Returns: dict with sentiment, score, and details
    """
    if 'nifty_option_screener_data' not in st.session_state:
        return None

    option_data = st.session_state.nifty_option_screener_data
    strike_analyses = option_data.get('strike_analyses')
    atm_bias = option_data.get('atm_bias')

    if not strike_analyses or not atm_bias:
        return None

    # FIX: Use the ATM strike's verdict from strike_analyses (this is what's displayed in Tab 0)
    # Find the ATM strike analysis
    atm_strike = atm_bias.get('atm_strike', 0)
    atm_analysis = next((a for a in strike_analyses if a.get("strike_price") == atm_strike), None)

    if not atm_analysis:
        return None

    # Extract the verdict and score from the ATM strike analysis (this matches the display)
    verdict = atm_analysis.get('verdict', 'Neutral')
    strike = atm_strike
    bias_score = atm_analysis.get('total_bias', 0)

    # Determine bias from verdict
    verdict_upper = str(verdict).upper()
    if 'STRONG BULLISH' in verdict_upper:
        bias = "BULLISH"
        score = 75
    elif 'BULLISH' in verdict_upper:
        bias = "BULLISH"
        score = 40
    elif 'STRONG BEARISH' in verdict_upper:
        bias = "BEARISH"
        score = -75
    elif 'BEARISH' in verdict_upper:
        bias = "BEARISH"
        score = -40
    else:
        bias = "NEUTRAL"
        score = bias_score if abs(bias_score) < 30 else 0

    confidence = min(100, abs(score))

    return {
        'bias': bias,
        'score': score,
        'confidence': confidence,
        'strike': strike,
        'verdict': verdict,
        'details': f"Strike: {strike} | Verdict: {verdict}"
    }


def calculate_pcr_sentiment_from_screener():
    """
    Calculate sentiment from PCR/OI analysis (from Option Screener Tab 0)
    Uses oi_pcr_metrics displayed on Overall Market Sentiment dashboard
    Returns: dict with sentiment, score, and details
    """
    if 'nifty_option_screener_data' not in st.session_state:
        return None

    option_data = st.session_state.nifty_option_screener_data
    oi_pcr_metrics = option_data.get('oi_pcr_metrics')

    if not oi_pcr_metrics:
        return None

    # Extract PCR sentiment - FIX: Use correct keys from analyze_oi_pcr_metrics() structure
    pcr_sentiment = oi_pcr_metrics.get('pcr_sentiment', 'NEUTRAL')  # FIX: Changed from 'sentiment' to 'pcr_sentiment'
    pcr_description = oi_pcr_metrics.get('pcr_interpretation', '')  # FIX: Changed from 'description' to 'pcr_interpretation'
    pcr_value = oi_pcr_metrics.get('pcr_total', 1.0)  # FIX: Changed from 'pcr_value' to 'pcr_total'

    # Determine bias and score based on PCR sentiment
    sentiment_upper = str(pcr_sentiment).upper()

    if 'BULLISH' in sentiment_upper:
        bias = "BULLISH"
        # Score based on how far PCR is from neutral (1.0)
        # PCR > 1.2 is bullish
        score = min(50, (pcr_value - 1.0) * 100) if pcr_value > 1.0 else 25
    elif 'BEARISH' in sentiment_upper:
        bias = "BEARISH"
        # PCR < 0.8 is bearish
        score = -min(50, (1.0 - pcr_value) * 100) if pcr_value < 1.0 else -25
    else:
        bias = "NEUTRAL"
        score = 0

    confidence = min(100, abs(score))

    return {
        'bias': bias,
        'score': score,
        'confidence': confidence,
        'pcr_value': pcr_value,
        'description': pcr_description,
        'details': f"PCR: {pcr_value:.2f} | {pcr_description}"
    }


def calculate_sector_rotation_sentiment():
    """
    Calculate sentiment from Sector Rotation Analysis (from Option Screener Tab 0)
    Uses sector_rotation_data displayed on Overall Market Sentiment dashboard
    Returns: dict with sentiment, score, and details
    """
    if 'nifty_option_screener_data' not in st.session_state:
        return None

    option_data = st.session_state.nifty_option_screener_data
    sector_rotation = option_data.get('sector_rotation_data')

    if not sector_rotation:
        return None

    # Extract sector rotation bias - FIX: Use correct keys from enhanced_market_data structure
    rotation_bias = sector_rotation.get('rotation_bias', 'NEUTRAL')
    rotation_score = sector_rotation.get('rotation_score', 0)
    rotation_description = sector_rotation.get('rotation_pattern', sector_rotation.get('sector_sentiment', ''))

    # Determine bias from rotation bias
    bias_upper = str(rotation_bias).upper()

    if 'STRONG BULLISH' in bias_upper:
        bias = "BULLISH"
        score = 75
    elif 'BULLISH' in bias_upper:
        bias = "BULLISH"
        score = 40
    elif 'STRONG BEARISH' in bias_upper:
        bias = "BEARISH"
        score = -75
    elif 'BEARISH' in bias_upper:
        bias = "BEARISH"
        score = -40
    else:
        bias = "NEUTRAL"
        score = rotation_score if abs(rotation_score) < 30 else 0

    confidence = min(100, abs(score))

    return {
        'bias': bias,
        'score': score,
        'confidence': confidence,
        'rotation_bias': rotation_bias,
        'description': rotation_description,
        'details': f"{rotation_bias} | {rotation_description}"
    }


def calculate_option_chain_atm_sentiment(NSE_INSTRUMENTS):
    """
    Calculate sentiment from Option Chain ATM Zone Analysis
    Returns: dict with sentiment, score, and details
    """
    # Check if ATM zone bias data exists in session state
    instruments = ['NIFTY', 'SENSEX', 'FINNIFTY', 'MIDCPNIFTY']

    bullish_instruments = 0
    bearish_instruments = 0
    neutral_instruments = 0
    total_score = 0
    instruments_analyzed = 0

    atm_details = []

    for instrument in instruments:
        atm_key = f'{instrument}_atm_zone_bias'
        if atm_key not in st.session_state:
            continue

        df_atm = st.session_state[atm_key]

        # Get ATM zone data (Zone == "ATM")
        atm_row = df_atm[df_atm["Zone"] == "ATM"]
        if atm_row.empty:
            continue

        atm_row = atm_row.iloc[0]
        verdict = str(atm_row.get('Verdict', 'Neutral')).upper()

        # Calculate score based on verdict
        if 'STRONG BULLISH' in verdict:
            score = 75
            bullish_instruments += 1
        elif 'BULLISH' in verdict:
            score = 40
            bullish_instruments += 1
        elif 'STRONG BEARISH' in verdict:
            score = -75
            bearish_instruments += 1
        elif 'BEARISH' in verdict:
            score = -40
            bearish_instruments += 1
        else:
            score = 0
            neutral_instruments += 1

        total_score += score
        instruments_analyzed += 1

        # Collect detailed ATM zone information for this instrument with ALL bias metrics
        # Note: OI_Change_Bias is same as ChgOI_Bias (included for compatibility)
        atm_detail = {
            'Instrument': instrument,
            'Strike': atm_row.get('Strike', 'N/A'),
            'Zone': atm_row.get('Zone', 'ATM'),
            'Level': atm_row.get('Level', 'N/A'),
            'OI_Bias': atm_row.get('OI_Bias', 'N/A'),
            'ChgOI_Bias': atm_row.get('ChgOI_Bias', 'N/A'),
            'Volume_Bias': atm_row.get('Volume_Bias', 'N/A'),
            'Delta_Bias': atm_row.get('Delta_Bias', 'N/A'),
            'Gamma_Bias': atm_row.get('Gamma_Bias', 'N/A'),
            'Premium_Bias': atm_row.get('Premium_Bias', 'N/A'),
            'AskQty_Bias': atm_row.get('AskQty_Bias', 'N/A'),
            'BidQty_Bias': atm_row.get('BidQty_Bias', 'N/A'),
            'IV_Bias': atm_row.get('IV_Bias', 'N/A'),
            'DVP_Bias': atm_row.get('DVP_Bias', 'N/A'),
            'Delta_Exposure_Bias': atm_row.get('Delta_Exposure_Bias', 'N/A'),
            'Gamma_Exposure_Bias': atm_row.get('Gamma_Exposure_Bias', 'N/A'),
            'IV_Skew_Bias': atm_row.get('IV_Skew_Bias', 'N/A'),
            'OI_Change_Bias': atm_row.get('ChgOI_Bias', atm_row.get('OI_Change_Bias', 'N/A')),  # Alias for ChgOI_Bias
            'BiasScore': atm_row.get('BiasScore', 0),
            'Verdict': atm_row.get('Verdict', 'Neutral'),
            'Score': f"{score:+.0f}"
        }
        atm_details.append(atm_detail)

    # Calculate overall score and bias
    if instruments_analyzed == 0:
        return None

    overall_score = total_score / instruments_analyzed

    # Determine overall bias
    if overall_score > 30:
        bias = "BULLISH"
    elif overall_score < -30:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    # Calculate confidence
    confidence = min(100, abs(overall_score))

    return {
        'bias': bias,
        'score': overall_score,
        'bullish_instruments': bullish_instruments,
        'bearish_instruments': bearish_instruments,
        'neutral_instruments': neutral_instruments,
        'total_instruments': instruments_analyzed,
        'confidence': confidence,
        'atm_details': atm_details
    }


def calculate_nifty_advanced_metrics_sentiment():
    """
    Calculate sentiment from NIFTY advanced metrics:
    - Synthetic Future Bias
    - ATM Buildup Pattern
    - ATM Vega Bias
    - Distance from Max Pain
    - Call Resistance / Put Support positioning
    - Total Vega Bias

    Returns: dict with sentiment, score, and details
    """
    if 'NIFTY_comprehensive_metrics' not in st.session_state:
        return None

    metrics = st.session_state['NIFTY_comprehensive_metrics']

    # Initialize score
    score = 0
    details = []

    # 1. Synthetic Future Bias (Display Only - Not used in scoring)
    synthetic_bias = metrics.get('Synthetic Future Bias', 'Neutral')
    synthetic_diff = metrics.get('synthetic_diff', 0)
    if 'BULLISH' in str(synthetic_bias).upper():
        # score += 20  # Removed from scoring
        details.append(f"Synthetic Future: Bullish (+{synthetic_diff:.2f})")
    elif 'BEARISH' in str(synthetic_bias).upper():
        # score -= 20  # Removed from scoring
        details.append(f"Synthetic Future: Bearish ({synthetic_diff:.2f})")
    else:
        details.append(f"Synthetic Future: Neutral")

    # 2. ATM Buildup Pattern (Weight: 2.5)
    atm_buildup = metrics.get('ATM Buildup Pattern', 'Neutral')
    if 'SHORT BUILDUP' in str(atm_buildup).upper() or 'PUT WRITING' in str(atm_buildup).upper() or 'SHORT COVERING' in str(atm_buildup).upper():
        score += 25
        details.append(f"ATM Buildup: Bullish ({atm_buildup})")
    elif 'LONG BUILDUP' in str(atm_buildup).upper() or 'CALL WRITING' in str(atm_buildup).upper() or 'LONG UNWINDING' in str(atm_buildup).upper():
        score -= 25
        details.append(f"ATM Buildup: Bearish ({atm_buildup})")
    else:
        details.append(f"ATM Buildup: Neutral")

    # 3. ATM Vega Bias (Display Only - Not used in scoring)
    atm_vega_bias = metrics.get('ATM Vega Bias', 'Neutral')
    if 'BULLISH' in str(atm_vega_bias).upper():
        # score += 15  # Removed from scoring
        details.append(f"ATM Vega: Bullish (High Put Vega)")
    elif 'BEARISH' in str(atm_vega_bias).upper():
        # score -= 15  # Removed from scoring
        details.append(f"ATM Vega: Bearish (High Call Vega)")
    else:
        details.append(f"ATM Vega: Neutral")

    # 4. Distance from Max Pain (Display Only - Not used in scoring)
    distance_from_max_pain = metrics.get('distance_from_max_pain_value', 0)
    if distance_from_max_pain > 50:
        # score -= 20  # Removed from scoring - Above max pain suggests downward pull
        details.append(f"Max Pain Distance: Bearish (+{distance_from_max_pain:.2f}, above max pain)")
    elif distance_from_max_pain < -50:
        # score += 20  # Removed from scoring - Below max pain suggests upward pull
        details.append(f"Max Pain Distance: Bullish ({distance_from_max_pain:.2f}, below max pain)")
    else:
        details.append(f"Max Pain Distance: Neutral ({distance_from_max_pain:+.2f})")

    # 5. Total Vega Bias (Weight: 1.5)
    total_vega_bias = metrics.get('Total Vega Bias', 'Neutral')
    if 'BULLISH' in str(total_vega_bias).upper():
        score += 15
        details.append(f"Total Vega: Bullish (Put Heavy)")
    elif 'BEARISH' in str(total_vega_bias).upper():
        score -= 15
        details.append(f"Total Vega: Bearish (Call Heavy)")
    else:
        details.append(f"Total Vega: Neutral")

    # 6. Call Resistance / Put Support Distance (Weight: 1.0)
    call_resistance_distance = metrics.get('call_resistance_distance', 0)
    put_support_distance = metrics.get('put_support_distance', 0)

    # If close to call resistance, bearish; if close to put support, bullish
    if call_resistance_distance < 50 and call_resistance_distance > 0:
        score -= 10
        details.append(f"Near Call Resistance: Bearish ({call_resistance_distance:.2f} pts)")
    elif put_support_distance < 50 and put_support_distance > 0:
        score += 10
        details.append(f"Near Put Support: Bullish ({put_support_distance:.2f} pts)")

    # Determine overall bias
    if score > 30:
        bias = "BULLISH"
    elif score < -30:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    # Calculate confidence based on score magnitude
    confidence = min(100, abs(score))

    return {
        'bias': bias,
        'score': score,
        'confidence': confidence,
        'details': details,
        'metrics': metrics
    }


def calculate_ai_analysis_sentiment(ai_report):
    """
    Calculate sentiment from AI Market Analysis
    Returns: dict with sentiment, score, and details
    """
    if not ai_report:
        return None
    
    # Extract sentiment from AI report
    ai_sentiment = ai_report.get('sentiment', 'NEUTRAL').upper()
    ai_score = ai_report.get('score', 0)
    ai_confidence = ai_report.get('confidence', 0)
    
    # Extract reasoning and key metrics
    reasoning = ai_report.get('reasoning', [])
    final_verdict = ai_report.get('final_verdict', '')
    
    # Convert AI sentiment to our format
    if 'BULLISH' in ai_sentiment:
        bias = "BULLISH"
        icon = "🤖🐂"
    elif 'BEARISH' in ai_sentiment:
        bias = "BEARISH"
        icon = "🤖🐻"
    else:
        bias = "NEUTRAL"
        icon = "🤖⚖️"
    
    # Prepare details
    details = []
    if 'news_sentiment' in ai_report:
        details.append(f"News Analysis: {ai_report['news_sentiment'].get('overall', 'Neutral')}")
    if 'global_markets' in ai_report:
        details.append(f"Global Markets: {ai_report['global_markets'].get('overall_sentiment', 'Neutral')}")
    
    return {
        'bias': bias,
        'score': ai_score,
        'confidence': ai_confidence,
        'icon': icon,
        'reasoning': reasoning,
        'final_verdict': final_verdict,
        'details': details,
        'full_report': ai_report
    }


def calculate_overall_sentiment():
    """
    Calculate overall market sentiment by combining all data sources
    """
    # Initialize sentiment sources
    sentiment_sources = {}

    # 1. Stock Performance Sentiment
    if 'bias_analysis_results' in st.session_state and st.session_state.bias_analysis_results:
        analysis = st.session_state.bias_analysis_results
        if analysis.get('success'):
            stock_data = analysis.get('stock_data', [])
            stock_sentiment = calculate_stock_performance_sentiment(stock_data)
            if stock_sentiment:
                sentiment_sources['Stock Performance'] = stock_sentiment

    # 2. Technical Indicators Sentiment
    if 'bias_analysis_results' in st.session_state and st.session_state.bias_analysis_results:
        analysis = st.session_state.bias_analysis_results
        if analysis.get('success'):
            bias_results = analysis.get('bias_results', [])
            tech_sentiment = calculate_technical_indicators_sentiment(bias_results)
            if tech_sentiment:
                sentiment_sources['Technical Indicators'] = tech_sentiment

    # 3. ATM Strike Verdict (from Tab 0 - Option Screener)
    atm_verdict_sentiment = calculate_atm_strike_verdict_sentiment()
    if atm_verdict_sentiment:
        sentiment_sources['ATM Strike Verdict'] = atm_verdict_sentiment

    # 4. PCR Sentiment (from Tab 0 - Option Screener)
    pcr_screener_sentiment = calculate_pcr_sentiment_from_screener()
    if pcr_screener_sentiment:
        sentiment_sources['PCR/OI Analysis'] = pcr_screener_sentiment

    # 5. Sector Rotation (from Tab 0 - Option Screener)
    sector_rotation_sentiment = calculate_sector_rotation_sentiment()
    if sector_rotation_sentiment:
        sentiment_sources['Sector Rotation'] = sector_rotation_sentiment

    # If no data available
    if not sentiment_sources:
        return {
            'data_available': False,
            'overall_sentiment': 'NEUTRAL',
            'overall_score': 0,
            'confidence': 0,
            'sources': {},
            'bullish_sources': 0,
            'bearish_sources': 0,
            'neutral_sources': 0,
            'source_count': 0
        }

    # Calculate weighted overall sentiment
    source_weights = {
        'Stock Performance': 2.0,
        'Technical Indicators': 3.0,
        'ATM Strike Verdict': 3.5,      # ATM ±2 strikes analysis
        'PCR/OI Analysis': 2.5,          # PCR sentiment from Tab 0
        'Sector Rotation': 3.0           # Sector rotation bias from Tab 0
    }

    total_weighted_score = 0
    total_weight = 0

    bullish_sources = 0
    bearish_sources = 0
    neutral_sources = 0

    for source_name, source_data in sentiment_sources.items():
        score = source_data.get('score', 0)
        weight = source_weights.get(source_name, 1.0)

        total_weighted_score += score * weight
        total_weight += weight

        # Count source bias
        bias = source_data.get('bias', 'NEUTRAL')
        if bias == 'BULLISH':
            bullish_sources += 1
        elif bias == 'BEARISH':
            bearish_sources += 1
        else:
            neutral_sources += 1

    # Calculate overall score
    overall_score = total_weighted_score / total_weight if total_weight > 0 else 0

    # Determine overall sentiment
    if overall_score > 25:
        overall_sentiment = "BULLISH"
    elif overall_score < -25:
        overall_sentiment = "BEARISH"
    else:
        overall_sentiment = "NEUTRAL"

    # Calculate confidence
    score_magnitude = min(100, abs(overall_score))

    # Factor in source agreement
    total_sources = len(sentiment_sources)
    if overall_sentiment == 'BULLISH':
        source_agreement = bullish_sources / total_sources if total_sources > 0 else 0
    elif overall_sentiment == 'BEARISH':
        source_agreement = bearish_sources / total_sources if total_sources > 0 else 0
    else:
        source_agreement = neutral_sources / total_sources if total_sources > 0 else 0

    final_confidence = score_magnitude * source_agreement

    return {
        'overall_sentiment': overall_sentiment,
        'overall_score': overall_score,
        'confidence': final_confidence,
        'sources': sentiment_sources,
        'data_available': True,
        'bullish_sources': bullish_sources,
        'bearish_sources': bearish_sources,
        'neutral_sources': neutral_sources,
        'source_count': len(sentiment_sources)
    }


def _run_bias_analysis():
    """
    Helper function to run bias analysis
    Returns: (success, errors)
    """
    errors = []
    try:
        # Initialize bias_analyzer if not exists
        if 'bias_analyzer' not in st.session_state:
            from bias_analysis import BiasAnalysisPro
            st.session_state.bias_analyzer = BiasAnalysisPro()

        symbol = "^NSEI"  # NIFTY 50
        results = st.session_state.bias_analyzer.analyze_all_bias_indicators(symbol)
        st.session_state.bias_analysis_results = results
        if results.get('success'):
            return True, []
        else:
            errors.append(f"Bias Analysis Pro: {results.get('error', 'Unknown error')}")
            return False, errors
    except Exception as e:
        errors.append(f"Bias Analysis Pro: {str(e)}")
        return False, errors


async def _run_ai_analysis():
    """
    Helper function to run AI market analysis
    Returns: (success, errors)
    """
    errors = []
    try:
        # Check if AI engine should run (environment variable control)
        ai_run_only_directional = os.environ.get('AI_RUN_ONLY_DIRECTIONAL', 'true').lower() == 'true'
        
        if ai_run_only_directional:
            # Get current sentiment to decide if we should run AI analysis
            current_sentiment = calculate_overall_sentiment()
            overall_sentiment = current_sentiment.get('overall_sentiment', 'NEUTRAL')
            
            # Only run AI if sentiment is strongly directional
            if overall_sentiment not in ['BULLISH', 'BEARISH']:
                return True, ["AI Analysis: Skipped (non-directional market)"]
        
        # Prepare module biases from existing analysis
        module_biases = {}
        sources = current_sentiment.get('sources', {})
        for source_name, source_data in sources.items():
            if 'score' in source_data:
                module_biases[source_name] = source_data['score']
        
        # Prepare market metadata
        market_meta = {
            'timestamp': datetime.now().isoformat(),
            'market_session': scheduler.get_market_session(),
            'trading_hours': is_within_trading_hours()
        }
        
        # Get API keys from environment or session state
        news_api_key = os.environ.get('NEWS_API_KEY') or st.session_state.get('news_api_key')
        perplexity_api_key = os.environ.get('PERPLEXITY_API_KEY') or st.session_state.get('perplexity_api_key')

        # Run AI analysis
        with st.spinner("🤖 Running AI Market Analysis..."):
            ai_report = await run_ai_analysis(
                overall_market="Indian Stock Market (NIFTY 50)",
                module_biases=module_biases,
                market_meta=market_meta,
                news_api_key=news_api_key,
                perplexity_api_key=perplexity_api_key,
                save_report=True,
                telegram_send=False  # Disable telegram for auto-refresh
            )
            
            st.session_state.ai_market_analysis = ai_report
            st.session_state.ai_last_run = time.time()
            
        return True, []
        
    except Exception as e:
        errors.append(f"AI Market Analysis: {str(e)}")
        return False, errors


async def run_all_analyses(NSE_INSTRUMENTS, show_progress=True):
    """
    Runs all analyses and stores results in session state:
    1. Bias Analysis Pro (includes stock data and technical indicators)
    2. AI Market Analysis (enhanced with news, global data, and reasoning)

    Args:
        NSE_INSTRUMENTS: Instrument configuration
        show_progress: Whether to show progress bars (default True, set False for silent auto-refresh)
    """
    success = True
    errors = []

    try:
        # 1. Run Bias Analysis Pro
        spinner_text = "🎯 Running Bias Analysis Pro..."
        if show_progress:
            with st.spinner(spinner_text):
                success_bias, errors_bias = _run_bias_analysis()
                success = success and success_bias
                errors.extend(errors_bias)
        else:
            success_bias, errors_bias = _run_bias_analysis()
            success = success and success_bias
            errors.extend(errors_bias)

        # 2. Run AI Market Analysis (only once per hour to save API quota)
        current_time = time.time()
        ai_last_run = st.session_state.get('ai_last_run', 0)
        
        # Only run AI analysis if it hasn't been run in the last hour
        if current_time - ai_last_run > 3600:  # 1 hour cooldown
            ai_success, ai_errors = await _run_ai_analysis()
            success = success and ai_success
            errors.extend(ai_errors)
        else:
            if show_progress:
                st.info("🤖 AI analysis recently run. Using cached results.")

    except Exception as e:
        errors.append(f"Overall error: {str(e)}")
        success = False

    return success, errors


def render_overall_market_sentiment(NSE_INSTRUMENTS=None):
    """
    Renders the Overall Market Sentiment tab - SIMPLIFIED DASHBOARD VIEW
    Shows only essential summary metrics at a glance
    Auto-refreshes every 60 seconds
    """
    st.markdown("## 🌟 Overall Market Sentiment Dashboard")
    st.caption("Quick summary view - For detailed analysis, see dedicated tabs below")

    # Show refresh interval based on market session
    market_session = scheduler.get_market_session()
    refresh_interval = scheduler.get_refresh_interval(market_session)

    if is_within_trading_hours():
        st.caption(f"🔄 Auto-refreshing every {refresh_interval} seconds during trading hours")
    else:
        st.caption("⏸️ Auto-refresh paused (market closed). Using cached data.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════
    # 🎯 CURRENT POSITION & ACTION - MOST IMPORTANT INFO AT TOP!
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("## 🎯 CURRENT POSITION & ACTION")

    # Get current price
    current_price = 0.0
    if 'bias_analysis_results' in st.session_state and isinstance(st.session_state.bias_analysis_results, dict):
        df = st.session_state.bias_analysis_results.get('df')
        if df is not None and len(df) > 0:
            current_price = df['close'].iloc[-1]

    # Get VOB data for nearest levels
    nearest_support = None
    nearest_resistance = None

    if 'vob_data_nifty' in st.session_state and st.session_state.vob_data_nifty:
        vob_data = st.session_state.vob_data_nifty

        # Find nearest bullish VOB below current price (support)
        bullish_blocks = vob_data.get('bullish_blocks', [])
        min_dist_sup = float('inf')
        for block in bullish_blocks:
            if isinstance(block, dict):
                block_mid = (block.get('upper', 0) + block.get('lower', 0)) / 2
                if block_mid < current_price:
                    dist = current_price - block_mid
                    if dist < min_dist_sup:
                        min_dist_sup = dist
                        nearest_support = {
                            'price': block_mid,
                            'type': 'VOB Support',
                            'lower': block.get('lower', block_mid),
                            'upper': block.get('upper', block_mid)
                        }

        # Find nearest bearish VOB above current price (resistance)
        bearish_blocks = vob_data.get('bearish_blocks', [])
        min_dist_res = float('inf')
        for block in bearish_blocks:
            if isinstance(block, dict):
                block_mid = (block.get('upper', 0) + block.get('lower', 0)) / 2
                if block_mid > current_price:
                    dist = block_mid - current_price
                    if dist < min_dist_res:
                        min_dist_res = dist
                        nearest_resistance = {
                            'price': block_mid,
                            'type': 'VOB Resistance',
                            'lower': block.get('lower', block_mid),
                            'upper': block.get('upper', block_mid)
                        }

    # Display action based on position
    if current_price > 0 and nearest_support and nearest_resistance:
        dist_to_sup = current_price - nearest_support['price']
        dist_to_res = nearest_resistance['price'] - current_price

        if dist_to_sup <= 5:
            st.success(f"""
### 🟢 AT SUPPORT - LONG SETUP ACTIVE

**Entry NOW:** ₹{nearest_support['lower']:,.0f} - ₹{nearest_support['upper']:,.0f} ({nearest_support['type']})
**Stop Loss:** ₹{nearest_support['lower'] - 20:,.0f} (below support zone)
**Target 1:** ₹{current_price + 30:,.0f} (+30 pts, Quick scalp)
**Target 2:** ₹{nearest_resistance['price']:,.0f} (Next resistance, +{dist_to_res + dist_to_sup:.0f} pts)

**✅ Entry Confirmation Required:**
1. Price bounces FROM support zone (don't chase if already moved up)
2. Volume increases on bounce candle
3. Regime supports LONG (check Market Regime in AI Trading Signal)
4. ATM Bias BULLISH (check below)
            """)

            # Auto-save signal to Supabase
            try:
                from signal_tracker import save_entry_signal
                import logging
                logger = logging.getLogger(__name__)

                entry_price_avg = (nearest_support['lower'] + nearest_support['upper']) / 2
                stop_loss_price = nearest_support['lower'] - 20
                target1_price = current_price + 30
                target2_price = nearest_resistance['price']

                # Build entry reason
                entry_reason = f"LONG at {nearest_support['type']} ₹{nearest_support['price']:,.0f} | "
                entry_reason += f"Dashboard Entry Zone Alert | "
                entry_reason += f"Distance to support: {dist_to_sup:.0f} pts"

                signal_id = save_entry_signal(
                    signal_type="LONG",
                    entry_price=entry_price_avg,
                    stop_loss=stop_loss_price,
                    target1=target1_price,
                    target2=target2_price,
                    support_level=nearest_support['price'],
                    resistance_level=nearest_resistance['price'],
                    entry_reason=entry_reason,
                    current_price=current_price,
                    source=nearest_support['type']
                )

                if signal_id:
                    st.caption(f"✅ Signal #{signal_id} saved to trading history")

            except Exception as e:
                logger.warning(f"Could not auto-save signal: {e}")

        elif dist_to_res <= 5:
            st.error(f"""
### 🔴 AT RESISTANCE - SHORT SETUP ACTIVE

**Entry NOW:** ₹{nearest_resistance['lower']:,.0f} - ₹{nearest_resistance['upper']:,.0f} ({nearest_resistance['type']})
**Stop Loss:** ₹{nearest_resistance['upper'] + 20:,.0f} (above resistance zone)
**Target 1:** ₹{current_price - 30:,.0f} (-30 pts, Quick scalp)
**Target 2:** ₹{nearest_support['price']:,.0f} (Next support, -{dist_to_res + dist_to_sup:.0f} pts)

**✅ Entry Confirmation Required:**
1. Price rejects FROM resistance zone (don't chase if already moved down)
2. Volume increases on rejection candle
3. Regime supports SHORT (check Market Regime in AI Trading Signal)
4. ATM Bias BEARISH (check below)
            """)

            # Auto-save signal to Supabase
            try:
                from signal_tracker import save_entry_signal
                import logging
                logger = logging.getLogger(__name__)

                entry_price_avg = (nearest_resistance['lower'] + nearest_resistance['upper']) / 2
                stop_loss_price = nearest_resistance['upper'] + 20
                target1_price = current_price - 30
                target2_price = nearest_support['price']

                # Build entry reason
                entry_reason = f"SHORT at {nearest_resistance['type']} ₹{nearest_resistance['price']:,.0f} | "
                entry_reason += f"Dashboard Entry Zone Alert | "
                entry_reason += f"Distance to resistance: {dist_to_res:.0f} pts"

                signal_id = save_entry_signal(
                    signal_type="SHORT",
                    entry_price=entry_price_avg,
                    stop_loss=stop_loss_price,
                    target1=target1_price,
                    target2=target2_price,
                    support_level=nearest_support['price'],
                    resistance_level=nearest_resistance['price'],
                    entry_reason=entry_reason,
                    current_price=current_price,
                    source=nearest_resistance['type']
                )

                if signal_id:
                    st.caption(f"✅ Signal #{signal_id} saved to trading history")

            except Exception as e:
                logger.warning(f"Could not auto-save signal: {e}")
        else:
            st.info(f"""
### ⚠️ MID-ZONE - WAIT FOR ENTRY ZONES

**Current Price:** ₹{current_price:,.2f}
**Nearest Support:** ₹{nearest_support['price']:,.0f} (-{dist_to_sup:.0f} pts) - {nearest_support['type']}
**Nearest Resistance:** ₹{nearest_resistance['price']:,.0f} (+{dist_to_res:.0f} pts) - {nearest_resistance['type']}

**🚫 DO NOT TRADE HERE:**
- Poor risk/reward ratio in the middle
- Wait for price to reach entry zones (±5 pts of levels)
- Set alerts at ₹{nearest_support['price']:,.0f} (LONG) and ₹{nearest_resistance['price']:,.0f} (SHORT)

**Missing a trade is 100x better than a bad entry!**
            """)
    elif current_price > 0:
        st.warning(f"""
### ⏳ LOADING ENTRY ZONES...

**Current Price:** ₹{current_price:,.2f}

Waiting for Volume Order Block data to identify precise entry zones.
Check **Advanced Chart Analysis** tab to load VOB data.
        """)
    else:
        st.warning("""
### ⏳ WAITING FOR MARKET DATA...

Loading current price and entry zones. Please wait...
        """)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED MARKET ANALYSIS SUMMARY CARD
    # ═══════════════════════════════════════════════════════════════════
    # This will be populated after we calculate overall sentiment

    # Initialize auto-refresh timestamp
    if 'sentiment_last_refresh' not in st.session_state:
        st.session_state.sentiment_last_refresh = 0

    # Initialize auto-run flag
    if 'sentiment_auto_run_done' not in st.session_state:
        st.session_state.sentiment_auto_run_done = False

    # Auto-run analyses on first load
    if not st.session_state.sentiment_auto_run_done and NSE_INSTRUMENTS is not None:
        with st.spinner("🔄 Running initial analyses..."):
            success, errors = asyncio.run(run_all_analyses(NSE_INSTRUMENTS))
            st.session_state.sentiment_auto_run_done = True
            st.session_state.sentiment_last_refresh = time.time()
            if not success:
                for error in errors:
                    st.warning(f"⚠️ {error}")

    # Auto-refresh based on market session (skip when market is closed for performance)
    current_time = time.time()
    time_since_refresh = current_time - st.session_state.sentiment_last_refresh

    # Get recommended refresh interval based on market session
    market_session = scheduler.get_market_session()
    refresh_interval = scheduler.get_refresh_interval(market_session)

    # Only auto-refresh during trading hours to conserve resources
    if time_since_refresh >= refresh_interval and NSE_INSTRUMENTS is not None and is_within_trading_hours():
        # Use show_progress=False for faster background refresh
        success, errors = asyncio.run(run_all_analyses(NSE_INSTRUMENTS, show_progress=False))
        st.session_state.sentiment_last_refresh = time.time()
        if success:
            st.rerun()

    # Calculate overall sentiment
    result = calculate_overall_sentiment()

    if not result['data_available']:
        st.warning("⚠️ No data available. Running analyses...")

        # Automatically run analyses
        if NSE_INSTRUMENTS is not None:
            with st.spinner("🔄 Running all analyses..."):
                success, errors = asyncio.run(run_all_analyses(NSE_INSTRUMENTS))

                if success:
                    st.session_state.sentiment_last_refresh = time.time()
                    st.success("✅ Analyses completed! Refreshing...")
                    st.rerun()
                else:
                    st.error("❌ Some analyses failed:")
                    for error in errors:
                        st.error(f"  - {error}")

        return

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED MARKET ANALYSIS SUMMARY
    # ═══════════════════════════════════════════════════════════════════

    # Get last updated time
    last_update_time = datetime.fromtimestamp(st.session_state.sentiment_last_refresh)
    last_updated_str = last_update_time.strftime('%Y-%m-%d %H:%M:%S')

    # Get sentiment data
    sentiment = result['overall_sentiment']
    score = result['overall_score']
    data_points = result['source_count']
    bullish_count = result['bullish_sources']
    bearish_count = result['bearish_sources']
    neutral_count = result['neutral_sources']

    # Determine sentiment icon and color
    if sentiment == 'BULLISH':
        sentiment_icon = '📈'
        sentiment_color = '#00ff88'
    elif sentiment == 'BEARISH':
        sentiment_icon = '📉'
        sentiment_color = '#ff4444'
    else:
        sentiment_icon = '⚖️'
        sentiment_color = '#ffa500'

    # Enhanced Summary Card
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
                padding: 25px; border-radius: 15px; margin-bottom: 20px;
                border: 1px solid #3d3d3d;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
            <h3 style='margin: 0; color: #ffffff; font-size: 24px;'>📊 Enhanced Market Analysis</h3>
            <span style='color: #888; font-size: 14px;'>📅 Last Updated: {last_updated_str}</span>
        </div>
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;'>
            <div style='background: {sentiment_color}; padding: 20px; border-radius: 10px; text-align: center;'>
                <div style='font-size: 32px; margin-bottom: 5px;'>{sentiment_icon}</div>
                <div style='font-size: 20px; font-weight: bold; color: white; margin-bottom: 5px;'>{sentiment}</div>
                <div style='font-size: 12px; color: rgba(255,255,255,0.8);'>Overall Sentiment</div>
            </div>
            <div style='background: #2d2d2d; padding: 20px; border-radius: 10px; text-align: center;
                        border-left: 4px solid {sentiment_color};'>
                <div style='font-size: 32px; color: {sentiment_color}; font-weight: bold; margin-bottom: 5px;'>{score:+.1f}</div>
                <div style='font-size: 12px; color: #888;'>Average Score</div>
            </div>
            <div style='background: #2d2d2d; padding: 20px; border-radius: 10px; text-align: center;
                        border-left: 4px solid #6495ED;'>
                <div style='font-size: 32px; color: #6495ED; font-weight: bold; margin-bottom: 5px;'>{data_points}</div>
                <div style='font-size: 12px; color: #888;'>Data Points</div>
            </div>
            <div style='background: #2d2d2d; padding: 20px; border-radius: 10px; text-align: center;'>
                <div style='font-size: 16px; color: #ffffff; font-weight: bold; margin-bottom: 5px;'>
                    🟢{bullish_count} | 🔴{bearish_count} | 🟡{neutral_count}
                </div>
                <div style='font-size: 12px; color: #888;'>Bullish | Bearish | Neutral</div>
            </div>
        </div>
        <div style='background: #252525; padding: 15px; border-radius: 10px;'>
            <div style='color: #888; font-size: 14px; margin-bottom: 10px; font-weight: bold;'>📊 Summary</div>
            <div style='display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px;'>
                <div style='text-align: center; padding: 10px; background: #1e1e1e; border-radius: 8px;'>
                    <div style='font-size: 24px; margin-bottom: 5px;'>⚡</div>
                    <div style='font-size: 11px; color: #888;'>India VIX</div>
                </div>
                <div style='text-align: center; padding: 10px; background: #1e1e1e; border-radius: 8px;'>
                    <div style='font-size: 24px; margin-bottom: 5px;'>🏢</div>
                    <div style='font-size: 11px; color: #888;'>Sector Rotation</div>
                </div>
                <div style='text-align: center; padding: 10px; background: #1e1e1e; border-radius: 8px;'>
                    <div style='font-size: 24px; margin-bottom: 5px;'>🌍</div>
                    <div style='font-size: 11px; color: #888;'>Global Markets</div>
                </div>
                <div style='text-align: center; padding: 10px; background: #1e1e1e; border-radius: 8px;'>
                    <div style='font-size: 24px; margin-bottom: 5px;'>💰</div>
                    <div style='font-size: 11px; color: #888;'>Intermarket</div>
                </div>
                <div style='text-align: center; padding: 10px; background: #1e1e1e; border-radius: 8px;'>
                    <div style='font-size: 24px; margin-bottom: 5px;'>🎯</div>
                    <div style='font-size: 11px; color: #888;'>Gamma Squeeze</div>
                </div>
                <div style='text-align: center; padding: 10px; background: #1e1e1e; border-radius: 8px;'>
                    <div style='font-size: 24px; margin-bottom: 5px;'>⏰</div>
                    <div style='font-size: 11px; color: #888;'>Intraday Timing</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════
    # HEADER METRICS
    # ═══════════════════════════════════════════════════════════════════

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sentiment = result['overall_sentiment']
        if sentiment == 'BULLISH':
            st.markdown(f"""
            <div style='padding: 20px; background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
                        border-radius: 10px; text-align: center;'>
                <h2 style='margin: 0; color: white;'>🚀 {sentiment}</h2>
                <p style='margin: 5px 0 0 0; color: white; font-size: 14px;'>Overall Sentiment</p>
            </div>
            """, unsafe_allow_html=True)
        elif sentiment == 'BEARISH':
            st.markdown(f"""
            <div style='padding: 20px; background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
                        border-radius: 10px; text-align: center;'>
                <h2 style='margin: 0; color: white;'>📉 {sentiment}</h2>
                <p style='margin: 5px 0 0 0; color: white; font-size: 14px;'>Overall Sentiment</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='padding: 20px; background: linear-gradient(135deg, #ffa500 0%, #ff8c00 100%);
                        border-radius: 10px; text-align: center;'>
                <h2 style='margin: 0; color: white;'>⚖️ {sentiment}</h2>
                <p style='margin: 5px 0 0 0; color: white; font-size: 14px;'>Overall Sentiment</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        score = result['overall_score']
        score_color = '#00ff88' if score > 0 else '#ff4444' if score < 0 else '#ffa500'
        st.markdown(f"""
        <div style='padding: 20px; background: #1e1e1e; border-radius: 10px; text-align: center;
                    border-left: 4px solid {score_color};'>
            <h2 style='margin: 0; color: {score_color};'>{score:+.1f}</h2>
            <p style='margin: 5px 0 0 0; color: #888; font-size: 14px;'>Overall Score</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        confidence = result['confidence']
        conf_color = '#00ff88' if confidence > 70 else '#ffa500' if confidence > 40 else '#ff4444'
        st.markdown(f"""
        <div style='padding: 20px; background: #1e1e1e; border-radius: 10px; text-align: center;
                    border-left: 4px solid {conf_color};'>
            <h2 style='margin: 0; color: {conf_color};'>{confidence:.1f}%</h2>
            <p style='margin: 5px 0 0 0; color: #888; font-size: 14px;'>Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        source_count = result['source_count']
        st.markdown(f"""
        <div style='padding: 20px; background: #1e1e1e; border-radius: 10px; text-align: center;
                    border-left: 4px solid #6495ED;'>
            <h2 style='margin: 0; color: #6495ED;'>{source_count}</h2>
            <p style='margin: 5px 0 0 0; color: #888; font-size: 14px;'>Active Sources</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════
    # AI TRADING SIGNAL - PHASE 4 INTEGRATION
    # ═══════════════════════════════════════════════════════════════════

    st.markdown("## 🎯 AI Trading Signal")

    try:
        from signal_display_integration import (
            generate_trading_signal,
            display_signal_card,
            display_signal_history,
            display_telegram_stats,
            display_final_assessment
        )

        # Get all required data for signal generation
        if 'bias_analysis_results' in st.session_state:
            bias_analysis = st.session_state.bias_analysis_results
            df = bias_analysis.get('df')
            bias_results = bias_analysis.get('bias_results')
        else:
            df = None
            bias_results = None

        # Get other data from session state
        option_chain = st.session_state.get('overall_option_data', {}).get('NIFTY', {})
        volatility_result = st.session_state.get('volatility_regime_result')
        oi_trap_result = st.session_state.get('oi_trap_result')
        cvd_result = st.session_state.get('cvd_result')
        participant_result = st.session_state.get('participant_result')
        liquidity_result = st.session_state.get('liquidity_result')
        ml_regime_result = st.session_state.get('ml_regime_result')
        money_flow_signals = st.session_state.get('money_flow_signals')
        deltaflow_signals = st.session_state.get('deltaflow_signals')
        enhanced_market_data = st.session_state.get('enhanced_market_data')
        nifty_screener_data = st.session_state.get('nifty_option_screener_data')

        # Get current price and ATM strike
        current_price = 0.0
        atm_strike = None
        if df is not None and len(df) > 0:
            current_price = df['close'].iloc[-1]
            atm_strike = round(current_price / 50) * 50  # Round to nearest 50

        # Fallback: Try to get current price from nifty_screener_data or enhanced_market_data
        if current_price == 0.0:
            if nifty_screener_data and 'current_price' in nifty_screener_data:
                current_price = nifty_screener_data['current_price']
            elif nifty_screener_data and 'spot_price' in nifty_screener_data:
                current_price = nifty_screener_data['spot_price']
            elif enhanced_market_data and 'nifty_spot' in enhanced_market_data:
                current_price = enhanced_market_data['nifty_spot']

        if current_price > 0 and atm_strike is None:
            atm_strike = round(current_price / 50) * 50

        # Calculate sentiment score
        sentiment_score = result.get('overall_score', 0.0)

        # Display FINAL ASSESSMENT first - Pass ALL analysis data
        if nifty_screener_data or enhanced_market_data:
            display_final_assessment(
                nifty_screener_data=nifty_screener_data,
                enhanced_market_data=enhanced_market_data,
                ml_regime_result=ml_regime_result,
                liquidity_result=liquidity_result,
                current_price=current_price if current_price > 0 else 24500,
                atm_strike=atm_strike if atm_strike else 24500,
                option_chain=option_chain,  # Pass option chain for real premiums
                money_flow_signals=money_flow_signals,  # Money Flow Profile (Tab 7)
                deltaflow_signals=deltaflow_signals,  # DeltaFlow Profile (Tab 7)
                cvd_result=cvd_result,  # CVD Analysis (Tab 4)
                volatility_result=volatility_result,  # Volatility Regime (Tab 2)
                oi_trap_result=oi_trap_result,  # OI Trap (Tab 3)
                participant_result=participant_result  # Participant Analysis (Tab 5)
            )

        # Generate signal if we have data
        if df is not None and len(df) > 10:
            with st.spinner("🤖 Generating AI trading signal..."):
                signal = generate_trading_signal(
                    df=df,
                    bias_results=bias_results,
                    option_chain=option_chain,
                    volatility_result=volatility_result,
                    oi_trap_result=oi_trap_result,
                    cvd_result=cvd_result,
                    participant_result=participant_result,
                    liquidity_result=liquidity_result,
                    ml_regime_result=ml_regime_result,
                    sentiment_score=sentiment_score,
                    option_screener_data=None,  # Will be extracted from nifty_screener_data
                    money_flow_signals=money_flow_signals,
                    deltaflow_signals=deltaflow_signals,
                    overall_sentiment_data=result,
                    enhanced_market_data=enhanced_market_data,
                    nifty_screener_data=nifty_screener_data,
                    current_price=current_price,
                    atm_strike=atm_strike
                )

                if signal:
                    # Display signal card
                    display_signal_card(signal)

                    # Auto-create Active Signal entry if ENTRY signal
                    if signal.signal_type == "ENTRY" and 'signal_manager' in st.session_state:
                        from signal_display_integration import create_active_signal_from_trading_signal

                        # Check if already auto-created
                        if not hasattr(signal, '_auto_created'):
                            signal_id = create_active_signal_from_trading_signal(
                                signal,
                                st.session_state.signal_manager
                            )
                            if signal_id:
                                st.success(f"✅ Auto-created Active Signal! View in Active Signals tab (ID: {signal_id[:20]}...)")
                                signal._auto_created = True  # Mark to prevent duplicate creation

                    # Show signal history in expander
                    with st.expander("📊 Signal History & Statistics"):
                        col1, col2 = st.columns(2)
                        with col1:
                            display_signal_history()
                        with col2:
                            display_telegram_stats()
                else:
                    st.warning("⚠️ Unable to generate signal. Insufficient data.")
        else:
            st.info("💡 Signal generation requires bias analysis data. Run 'Re-run All Analyses' to enable signal generation.")

    except Exception as e:
        st.error(f"❌ Signal generation error: {e}")
        st.caption("Signal system requires all indicator data to be loaded. Please ensure all analyses have completed successfully.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════
    # BIAS METRICS SUMMARY - NEW SECTION
    # ═══════════════════════════════════════════════════════════════════

    # ─────────────────────────────────────────────────────────────────
    # AUTO-LOAD REQUIRED DATA FOR BIAS METRICS
    # ─────────────────────────────────────────────────────────────────

    # Auto-load Enhanced Market Data if not present (for Sector Rotation)
    if 'enhanced_market_data' not in st.session_state:
        try:
            with st.spinner("🔄 Auto-loading Enhanced Market Data for complete analysis..."):
                from enhanced_market_data import get_enhanced_market_data
                enhanced_data = get_enhanced_market_data()
                st.session_state.enhanced_market_data = enhanced_data
        except Exception as e:
            # Silently handle error - will show info message later
            pass

    # ═══════════════════════════════════════════════════════════════════
    # SOURCE AGREEMENT VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════

    st.markdown("### 📊 Source Agreement")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🟢 Bullish Sources", result['bullish_sources'])
    with col2:
        st.metric("🔴 Bearish Sources", result['bearish_sources'])
    with col3:
        st.metric("🟡 Neutral Sources", result['neutral_sources'])

    # Progress bar for source distribution
    total = result['source_count']
    if total > 0:
        bullish_pct = (result['bullish_sources'] / total) * 100
        bearish_pct = (result['bearish_sources'] / total) * 100
        neutral_pct = (result['neutral_sources'] / total) * 100

        st.markdown(f"""
        <div style='background: #1e1e1e; border-radius: 10px; padding: 10px; margin: 10px 0;'>
            <div style='display: flex; height: 30px; border-radius: 5px; overflow: hidden;'>
                <div style='width: {bullish_pct}%; background: #00ff88; display: flex; align-items: center;
                            justify-content: center; color: black; font-weight: bold;'>
                    {bullish_pct:.0f}%
                </div>
                <div style='width: {bearish_pct}%; background: #ff4444; display: flex; align-items: center;
                            justify-content: center; color: white; font-weight: bold;'>
                    {bearish_pct:.0f}%
                </div>
                <div style='width: {neutral_pct}%; background: #ffa500; display: flex; align-items: center;
                            justify-content: center; color: white; font-weight: bold;'>
                    {neutral_pct:.0f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════
    # SIMPLIFIED INTERPRETATION & ACTION PLAN
    # ═══════════════════════════════════════════════════════════════════

    st.markdown("### 💡 What Should I Do?")

    sentiment = result['overall_sentiment']
    confidence = result['confidence']
    score = result['overall_score']

    # Generate simple interpretation
    if sentiment == 'BULLISH':
        if confidence > 70:
            st.success("✅ **Strong Bullish Signal**: Consider bullish strategies (long positions, call options, bull spreads)")
        else:
            st.info("📈 **Moderate Bullish**: Bullish bias with caution. Consider smaller position sizes.")
    elif sentiment == 'BEARISH':
        if confidence > 70:
            st.error("⚠️ **Strong Bearish Signal**: Consider bearish strategies (short positions, put options, bear spreads)")
        else:
            st.warning("📉 **Moderate Bearish**: Bearish bias with caution. Monitor key resistance levels.")
    else:
        st.info("⚖️ **Neutral/Range-bound**: Stay on the sidelines or use neutral strategies (iron condors, straddles)")

    # Risk Warning
    st.caption("""
    ⚠️ **Risk Warning**: This sentiment analysis is based on technical indicators and historical data.
    Past performance does not guarantee future results. Always use proper risk management.
    """)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════
    # QUICK LINKS TO DETAILED TABS
    # ═══════════════════════════════════════════════════════════════════

    st.markdown("### 🔗 Detailed Analysis Available In:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **🎯 NIFTY Option Screener v7.0** (Tab 8)
        - ATM Bias Analyzer
        - OI/PCR Analytics
        - Seller's Perspective
        - Moment Detector
        - Expiry Spike Analysis
        """)

    with col2:
        st.info("""
        **🎲 Bias Analysis Pro** (Tab 5)
        - 13 Technical Indicators
        - Stock Performance Details
        - Weighted Scoring
        - Historical Analysis
        """)

    with col3:
        st.info("""
        **🌐 Enhanced Market Data** (Tab 9)
        - India VIX
        - Sector Rotation
        - Global Markets
        - Intermarket Data
        """)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════
    # DETAILED ANALYSIS BY SOURCE (COLLAPSED BY DEFAULT)
    # ═══════════════════════════════════════════════════════════════════

    with st.expander("📊 **Show Detailed Source Analysis** (Click to expand)", expanded=False):
        st.markdown("### 📈 Detailed Analysis by Source")

        sources = result['sources']

        # ─────────────────────────────────────────────────────────────────
        # 1. STOCK PERFORMANCE TABLE
        # ─────────────────────────────────────────────────────────────────
        if 'Stock Performance' in sources:
            source_data = sources['Stock Performance']
            st.markdown("#### 📊 Stock Performance (Market Breadth)")
            bias = source_data.get('bias', 'NEUTRAL')
            score = source_data.get('score', 0)
            confidence = source_data.get('confidence', 0)

            # Color based on bias
            if bias == 'BULLISH':
                bg_color = '#00ff88'
                text_color = 'black'
                icon = '🐂'
            elif bias == 'BEARISH':
                bg_color = '#ff4444'
                text_color = 'white'
                icon = '🐻'
            else:
                bg_color = '#ffa500'
                text_color = 'white'
                icon = '⚖️'

            # Display source card
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"""
                <div style='background: {bg_color}; padding: 15px; border-radius: 10px;'>
                    <h3 style='margin: 0; color: {text_color};'>{icon} {bias}</h3>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.metric("Score", f"{score:+.1f}")

            with col3:
                st.metric("Confidence", f"{confidence:.1f}%")

            st.markdown(f"""
            **Market Breadth:** {source_data.get('breadth_pct', 0):.1f}%
            **Avg Weighted Change:** {source_data.get('avg_change', 0):+.2f}%
            **Bullish Stocks:** {source_data.get('bullish_stocks', 0)} | **Bearish:** {source_data.get('bearish_stocks', 0)} | **Neutral:** {source_data.get('neutral_stocks', 0)}
            """)

            # Stock Performance Table
            stock_details = source_data.get('stock_details', [])
            if stock_details:
                # Create DataFrame
                stock_df = pd.DataFrame(stock_details)
                stock_df['symbol'] = stock_df['symbol'].str.replace('.NS', '')
                stock_df['change_pct'] = stock_df['change_pct'].apply(lambda x: f"{x:.2f}%")
                stock_df['weight'] = stock_df['weight'].apply(lambda x: f"{x:.2f}%")

                # Add bias column
                def get_stock_bias(row):
                    change = float(row['change_pct'].replace('%', ''))
                    if change > 0.5:
                        return "🐂 BULLISH"
                    elif change < -0.5:
                        return "🐻 BEARISH"
                    else:
                        return "⚖️ NEUTRAL"

                stock_df['bias'] = stock_df.apply(get_stock_bias, axis=1)

                # Rename columns
                stock_df = stock_df.rename(columns={
                    'symbol': 'Symbol',
                    'change_pct': 'Change %',
                    'weight': 'Weight',
                    'bias': 'Bias'
                })

                st.dataframe(stock_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────
    # 2. TECHNICAL INDICATORS TABLE
    # ─────────────────────────────────────────────────────────────────
    if 'Technical Indicators' in sources:
        source_data = sources['Technical Indicators']
        with st.expander("**📊 Technical Indicators (Bias Analysis Pro)**", expanded=True):
            bias = source_data.get('bias', 'NEUTRAL')
            score = source_data.get('score', 0)
            confidence = source_data.get('confidence', 0)

            # Color based on bias
            if bias == 'BULLISH':
                bg_color = '#00ff88'
                text_color = 'black'
                icon = '🐂'
            elif bias == 'BEARISH':
                bg_color = '#ff4444'
                text_color = 'white'
                icon = '🐻'
            else:
                bg_color = '#ffa500'
                text_color = 'white'
                icon = '⚖️'

            # Display source card
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"""
                <div style='background: {bg_color}; padding: 15px; border-radius: 10px;'>
                    <h3 style='margin: 0; color: {text_color};'>{icon} {bias}</h3>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.metric("Score", f"{score:+.1f}")

            with col3:
                st.metric("Confidence", f"{confidence:.1f}%")

            st.markdown(f"""
            **Bullish Indicators:** {source_data.get('bullish_count', 0)} | **Bearish:** {source_data.get('bearish_count', 0)} | **Neutral:** {source_data.get('neutral_count', 0)}
            **Total Analyzed:** {source_data.get('total_count', 0)}
            """)

            # Technical Indicators Table
            indicator_details = source_data.get('indicator_details', [])
            if indicator_details:
                # Create DataFrame
                tech_df = pd.DataFrame(indicator_details)

                # Add emoji to bias
                def get_bias_emoji(bias):
                    bias_upper = str(bias).upper()
                    if 'BULLISH' in bias_upper or 'STRONG BUY' in bias_upper or 'STABLE' in bias_upper:
                        return f"🐂 {bias}"
                    elif 'BEARISH' in bias_upper or 'WEAK' in bias_upper or 'HIGH RISK' in bias_upper:
                        return f"🐻 {bias}"
                    else:
                        return f"⚖️ {bias}"

                tech_df['bias'] = tech_df['bias'].apply(get_bias_emoji)
                tech_df['score'] = tech_df['score'].apply(lambda x: f"{x:.2f}")
                tech_df['weight'] = tech_df['weight'].apply(lambda x: f"{x:.1f}")

                # Rename columns
                tech_df = tech_df.rename(columns={
                    'indicator': 'Indicator',
                    'value': 'Value',
                    'bias': 'Bias',
                    'score': 'Score',
                    'weight': 'Weight'
                })

                st.dataframe(tech_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────
    # 3. NIFTY OPTION SCREENER v7.0 - OVERALL MARKET SENTIMENT SUMMARY
    # ─────────────────────────────────────────────────────────────────
    st.markdown("---")

    # Check if we have option screener data in session state
    if 'nifty_option_screener_data' not in st.session_state:
        # Auto-load option screener data
        st.markdown("### 🎯 NIFTY Option Screener v7.0 - Market Sentiment Summary")
        with st.spinner("🔄 Auto-loading option chain data..."):
            try:
                from NiftyOptionScreener import load_option_screener_data_silently
                success = load_option_screener_data_silently()
                if success:
                    st.success("✅ Option chain data loaded successfully!")
                    st.rerun()
                else:
                    st.warning("""
                    ⚠️ **Unable to load option chain data automatically**

                    Please navigate to the **"🎯 NIFTY Option Screener v7.0"** tab to load the data manually.
                    """)
                    # Show detailed error if available
                    if 'nifty_option_screener_error' in st.session_state:
                        error_info = st.session_state.nifty_option_screener_error
                        st.error(f"🔍 Detailed error: {error_info.get('error', 'Unknown error')}")
                        st.caption(f"Error occurred at: {error_info.get('timestamp', 'Unknown time').strftime('%Y-%m-%d %H:%M:%S IST') if hasattr(error_info.get('timestamp'), 'strftime') else error_info.get('timestamp')}")
            except Exception as e:
                st.error(f"❌ Error loading option chain data: {e}")
                # Check if there's a detailed error in session state
                if 'nifty_option_screener_error' in st.session_state:
                    error_info = st.session_state.nifty_option_screener_error
                    st.error(f"🔍 Detailed error: {error_info.get('error', 'Unknown error')}")
                    st.caption(f"Error occurred at: {error_info.get('timestamp', 'Unknown time').strftime('%Y-%m-%d %H:%M:%S IST') if hasattr(error_info.get('timestamp'), 'strftime') else error_info.get('timestamp')}")
                st.warning("""
                ⚠️ **Option Chain Data Not Yet Loaded**

                Please navigate to the **"🎯 NIFTY Option Screener v7.0"** tab to load the data.
                """)

    if 'nifty_option_screener_data' in st.session_state and display_overall_market_sentiment_summary is not None:
        option_data = st.session_state.nifty_option_screener_data

        # Display the comprehensive market sentiment summary
        display_overall_market_sentiment_summary(
            overall_bias=option_data.get('overall_bias'),
            atm_bias=option_data.get('atm_bias'),
            seller_max_pain=option_data.get('seller_max_pain'),
            total_gex_net=option_data.get('total_gex_net'),
            expiry_spike_data=option_data.get('expiry_spike_data'),
            oi_pcr_metrics=option_data.get('oi_pcr_metrics'),
            strike_analyses=option_data.get('strike_analyses'),
            sector_rotation_data=option_data.get('sector_rotation_data')
        )

        # Show last updated timestamp
        last_updated = option_data.get('last_updated')
        if last_updated:
            st.caption(f"📅 Option chain data last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S IST')}")

        st.info("💡 For more detailed analysis, visit the **NIFTY Option Screener v7.0** tab")

    # ─────────────────────────────────────────────────────────────────
    # 5. PERPLEXITY AI MARKET INSIGHTS (REAL-TIME WEB RESEARCH)
    # ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("**🤖 Perplexity AI Market Insights (Real-Time Research)**", expanded=False):
        st.markdown("""
        ### 🚀 AI-Powered Market Insights with Web Research

        Get real-time market insights powered by Perplexity AI's advanced web research capabilities.

        **Features:**
        - Real-time market analysis with web citations
        - Breaking news impact analysis
        - Custom market questions answered
        - Bias setup probability analysis
        """)

        # Get current bias data to provide context
        bias_context = None
        if 'bias_analysis_results' in st.session_state and st.session_state.bias_analysis_results:
            analysis = st.session_state.bias_analysis_results
            if analysis.get('success'):
                bias_context = {
                    'overall_bias': result.get('overall_sentiment', 'NEUTRAL'),
                    'overall_confidence': result.get('confidence', 0),
                    'bullish_count': result.get('bullish_sources', 0),
                    'bearish_count': result.get('bearish_sources', 0),
                    'current_price': analysis.get('current_price', 0)
                }

        # Render the insights widget
        render_market_insights_widget(bias_context)

    # ─────────────────────────────────────────────────────────────────
    # 6. AI MARKET ANALYSIS (NEW SECTION)
    # ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    if '🤖 AI Market Analysis' in sources:
        source_data = sources['🤖 AI Market Analysis']
        with st.expander("**🤖 AI Market Analysis (Enhanced with News & Global Data)**", expanded=True):
            bias = source_data.get('bias', 'NEUTRAL')
            score = source_data.get('score', 0)
            confidence = source_data.get('confidence', 0)
            icon = source_data.get('icon', '🤖')
            
            # Color based on bias
            if bias == 'BULLISH':
                bg_color = '#00ff88'
                text_color = 'black'
            elif bias == 'BEARISH':
                bg_color = '#ff4444'
                text_color = 'white'
            else:
                bg_color = '#ffa500'
                text_color = 'white'

            # Display source card
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"""
                <div style='background: {bg_color}; padding: 15px; border-radius: 10px;'>
                    <h3 style='margin: 0; color: {text_color};'>{icon} AI: {bias}</h3>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.metric("Score", f"{score:+.1f}")

            with col3:
                st.metric("Confidence", f"{confidence:.1f}%")

            # Display AI Reasoning
            reasoning = source_data.get('reasoning', [])
            if reasoning:
                st.markdown("#### 🤔 AI Reasoning")
                for i, reason in enumerate(reasoning, 1):
                    st.markdown(f"{i}. {reason}")

            # Display Final Verdict
            final_verdict = source_data.get('final_verdict', '')
            if final_verdict:
                st.markdown("#### 🎯 Final Verdict")
                st.info(final_verdict)

            # Display Detailed Metrics
            full_report = source_data.get('full_report', {})
            
            if 'news_sentiment' in full_report:
                st.markdown("#### 📰 News Sentiment Analysis")
                news_data = full_report['news_sentiment']
                news_score = news_data.get('score', 0)
                news_articles = news_data.get('articles', [])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall News Sentiment", f"{news_score:+.2f}")
                with col2:
                    st.metric("Articles Analyzed", len(news_articles))
                with col3:
                    st.metric("Top Keywords", ", ".join(news_data.get('top_keywords', [])[:3]))
                
                if news_articles:
                    with st.expander("📋 Top News Articles"):
                        for article in news_articles[:5]:
                            st.markdown(f"**{article.get('title', 'No title')}**")
                            st.markdown(f"*Source: {article.get('source', 'Unknown')}*")
                            st.markdown(f"Sentiment: {article.get('sentiment', 'Neutral')}")
                            st.markdown("---")

            if 'global_markets' in full_report:
                st.markdown("#### 🌍 Global Market Analysis")
                global_data = full_report['global_markets']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("US Markets", f"{global_data.get('us_markets', {}).get('sentiment', 'N/A')}")
                with col2:
                    st.metric("Asian Markets", f"{global_data.get('asian_markets', {}).get('sentiment', 'N/A')}")
                with col3:
                    st.metric("European Markets", f"{global_data.get('european_markets', {}).get('sentiment', 'N/A')}")
                with col4:
                    st.metric("Commodities", f"{global_data.get('commodities', {}).get('sentiment', 'N/A')}")

            # Display Technical Analysis Integration
            if 'technical_integration' in full_report:
                st.markdown("#### 🔧 Technical Analysis Integration")
                tech_data = full_report['technical_integration']
                st.markdown(f"**Patterns Identified:** {', '.join(tech_data.get('patterns', []))}")
                st.markdown(f"**Key Levels:** {tech_data.get('key_levels', 'N/A')}")

    # ─────────────────────────────────────────────────────────────────
    # 4. OPTION CHAIN ANALYSIS TABLE
    # ─────────────────────────────────────────────────────────────────
    if 'Option Chain Analysis' in sources:
        source_data = sources['Option Chain Analysis']
        with st.expander("**📊 Option Chain ATM Zone Analysis**", expanded=True):
            bias = source_data.get('bias', 'NEUTRAL')
            score = source_data.get('score', 0)
            confidence = source_data.get('confidence', 0)

            # Color based on bias
            if bias == 'BULLISH':
                bg_color = '#00ff88'
                text_color = 'black'
                icon = '🐂'
            elif bias == 'BEARISH':
                bg_color = '#ff4444'
                text_color = 'white'
                icon = '🐻'
            else:
                bg_color = '#ffa500'
                text_color = 'white'
                icon = '⚖️'

            # Display source card
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"""
                <div style='background: {bg_color}; padding: 15px; border-radius: 10px;'>
                    <h3 style='margin: 0; color: {text_color};'>{icon} {bias}</h3>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.metric("Score", f"{score:+.1f}")

            with col3:
                st.metric("Confidence", f"{confidence:.1f}%")

            st.markdown(f"""
            **Bullish Instruments:** {source_data.get('bullish_instruments', 0)} | **Bearish:** {source_data.get('bearish_instruments', 0)} | **Neutral:** {source_data.get('neutral_instruments', 0)}
            **Total Analyzed:** {source_data.get('total_instruments', 0)}
            """)

            # Display ATM Details Summary Table
            atm_details = source_data.get('atm_details', [])
            if atm_details:
                st.markdown("#### 📊 ATM Zone Summary - All Bias Metrics")

                # Create DataFrame from atm_details
                atm_df = pd.DataFrame(atm_details)

                # Add emoji indicators for all bias columns
                bias_columns = [
                    'OI_Bias', 'ChgOI_Bias', 'Volume_Bias', 'Delta_Bias', 'Gamma_Bias',
                    'Premium_Bias', 'AskQty_Bias', 'BidQty_Bias', 'IV_Bias', 'DVP_Bias',
                    'Delta_Exposure_Bias', 'Gamma_Exposure_Bias', 'IV_Skew_Bias',
                    'OI_Change_Bias', 'Verdict'
                ]

                for col in bias_columns:
                    if col in atm_df.columns:
                        atm_df[col] = atm_df[col].apply(lambda x:
                            f"🐂 {x}" if 'BULLISH' in str(x).upper() else
                            f"🐻 {x}" if 'BEARISH' in str(x).upper() else
                            f"⚖️ {x}" if 'NEUTRAL' in str(x).upper() else
                            str(x)
                        )

                st.dataframe(atm_df, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("#### 📋 Detailed ATM Zone Bias Tables")

            # Display detailed ATM Zone tables for each instrument
            instruments = ['NIFTY', 'SENSEX', 'FINNIFTY', 'MIDCPNIFTY']

            atm_data_available = False
            for instrument in instruments:
                if f'{instrument}_atm_zone_bias' in st.session_state:
                    atm_data_available = True
                    df_atm = st.session_state[f'{instrument}_atm_zone_bias']

                    st.markdown(f"##### {instrument} ATM Zone Bias")

                    # Add emoji indicators for bias columns
                    df_display = df_atm.copy()
                    bias_columns = [col for col in df_display.columns if '_Bias' in col or col == 'Verdict']

                    for col in bias_columns:
                        df_display[col] = df_display[col].apply(lambda x:
                            f"🐂 {x}" if 'BULLISH' in str(x).upper() else
                            f"🐻 {x}" if 'BEARISH' in str(x).upper() else
                            f"⚖️ {x}" if 'NEUTRAL' in str(x).upper() else
                            str(x)
                        )

                    st.dataframe(df_display, use_container_width=True, hide_index=True)

            if not atm_data_available:
                st.info("ℹ️ ATM Zone analysis data will be displayed here when available. Please run bias analysis from individual instrument tabs (NIFTY, BANKNIFTY, SENSEX, etc.) first.")

        # ═══════════════════════════════════════════════════════════════════
        # COMPREHENSIVE OPTION CHAIN METRICS
        # ═══════════════════════════════════════════════════════════════════
        if 'Option Chain Analysis' in result.get('sources', {}):
            st.markdown("---")
            st.markdown("### 🌐 Comprehensive Option Chain Analysis")
            st.caption("Advanced metrics: Max Pain, Synthetic Future, Vega Bias, Buildup Patterns, and more")

            # Check if comprehensive metrics are available in session state
            comprehensive_metrics = []
            instruments_to_check = ['NIFTY', 'BANKNIFTY', 'SENSEX', 'FINNIFTY', 'MIDCPNIFTY']

            for instrument in instruments_to_check:
                # Check if comprehensive metrics are stored
                metrics_key = f'{instrument}_comprehensive_metrics'
                if metrics_key in st.session_state:
                    metrics = st.session_state[metrics_key]
                    comprehensive_metrics.append(metrics)

            if comprehensive_metrics:
                st.markdown("#### 📊 Comprehensive Metrics Summary")
                comp_df = pd.DataFrame(comprehensive_metrics)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                # Expandable section with detailed explanations
                with st.expander("📖 Understanding Comprehensive Metrics"):
                    st.markdown("""
                    ### Advanced Option Chain Metrics Explained

                    **ATM-Specific Metrics:**

                    1. **Synthetic Future Bias**
                       - Calculated as: Strike + CE Premium - PE Premium
                       - Compares synthetic future price vs spot price
                       - Bullish if synthetic > spot, Bearish if synthetic < spot
                       - Indicates market expectations embedded in options pricing

                    2. **ATM Buildup Pattern**
                       - Analyzes OI changes at ATM strike
                       - Long Buildup: Rising OI + Rising Prices (Bearish)
                       - Short Buildup: Rising OI + Falling Prices (Bullish)
                       - Call Writing: CE OI rising, PE OI falling (Bearish)
                       - Put Writing: PE OI rising, CE OI falling (Bullish)

                    3. **ATM Vega Bias**
                       - Measures volatility exposure at ATM
                       - Higher Put Vega → Bullish (expecting upside volatility)
                       - Higher Call Vega → Bearish (expecting downside volatility)

                    4. **Distance from Max Pain**
                       - Shows how far current price is from Max Pain strike
                       - Price tends to gravitate toward Max Pain near expiry
                       - Positive distance: Above Max Pain (potential downward pull)
                       - Negative distance: Below Max Pain (potential upward pull)

                    **Overall Market Metrics:**

                    5. **Max Pain Strike**
                       - Strike where option writers lose least money
                       - Calculated by summing all option pain values
                       - Market tends to drift toward this level

                    6. **Call Resistance (OI)**
                       - Strike with highest Call OI above spot
                       - Major resistance level from option positioning

                    7. **Put Support (OI)**
                       - Strike with highest Put OI below spot
                       - Major support level from option positioning

                    8. **Total Vega Bias**
                       - Aggregate vega exposure across all strikes
                       - Indicates overall market volatility expectations

                    9. **Unusual Activity Alerts**
                       - Strikes with abnormally high volume/OI ratio
                       - May indicate smart money positioning

                    10. **Overall Buildup Pattern**
                        - Combined analysis of ITM, ATM, and OTM activity
                        - Identifies protective strategies and directional bets
                    """)
            else:
                st.info("ℹ️ Comprehensive option chain metrics will be displayed here. Visit individual instrument tabs in the Option Chain Analysis section to generate these metrics.")

    # Last Updated and Next Refresh
    st.markdown("---")

    # Calculate time until next refresh
    time_since_refresh = time.time() - st.session_state.sentiment_last_refresh
    time_until_refresh = max(0, refresh_interval - time_since_refresh)

    col1, col2 = st.columns(2)

    with col1:
        last_update_time = datetime.fromtimestamp(st.session_state.sentiment_last_refresh)
        st.caption(f"📅 Last updated: {last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")

    with col2:
        if time_until_refresh > 0:
            st.caption(f"⏱️ Next refresh in: {int(time_until_refresh)} seconds")
        else:
            st.caption(f"⏱️ Refreshing now...")

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.session_state.sentiment_last_refresh = 0  # Force immediate refresh
            st.rerun()

    with col2:
        if NSE_INSTRUMENTS is not None:
            if st.button("🎯 Re-run All Analyses", type="primary", use_container_width=True, key="rerun_bias_button"):
                success, errors = asyncio.run(run_all_analyses(NSE_INSTRUMENTS))
                st.session_state.sentiment_last_refresh = time.time()

                if success:
                    st.balloons()
                    st.success("🎉 All analyses completed successfully! Refreshing results...")
                    st.rerun()
                else:
                    st.error("❌ Some analyses failed:")
                    for error in errors:
                        st.error(f"  - {error}")

    # Add AI-specific control
    st.markdown("---")
    st.markdown("### 🤖 AI Analysis Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧠 Run AI Analysis Only", use_container_width=True):
            with st.spinner("🤖 Running AI Market Analysis..."):
                try:
                    ai_success, ai_errors = asyncio.run(_run_ai_analysis())
                    if ai_success:
                        st.success("✅ AI analysis completed!")
                        st.rerun()
                    else:
                        for error in ai_errors:
                            st.error(f"  - {error}")
                except Exception as e:
                    st.error(f"❌ AI analysis failed: {str(e)}")
    
    with col2:
        # API Key Configuration
        with st.expander("🔧 Configure API Keys"):
            news_api_key = st.text_input("News API Key", type="password",
                                         value=st.session_state.get('news_api_key', ''))
            perplexity_api_key = st.text_input("Perplexity API Key", type="password",
                                        value=st.session_state.get('perplexity_api_key', ''))

            if st.button("💾 Save API Keys"):
                st.session_state.news_api_key = news_api_key
                st.session_state.perplexity_api_key = perplexity_api_key
                st.success("✅ API keys saved to session state")

    # Auto-refresh handled by the refresh logic at the top of this function
    # No need for additional sleep/rerun here as it causes duplicate rendering
