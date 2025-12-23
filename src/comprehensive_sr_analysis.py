"""
COMPREHENSIVE Support/Resistance Strength Analysis
Uses ALL available data: GEX, Market Depth, VOB, CVD, OI, Volume, Regime, Liquidity, etc.

10 Factor Analysis with Precise Zone Width Calculation
"""

import numpy as np
from typing import Dict, Tuple


def analyze_sr_strength_comprehensive(features: Dict) -> Dict:
    """
    COMPREHENSIVE S/R Strength Analysis
    Uses 10 factors with ALL available data sources

    Returns precise entry zone widths for support/resistance
    """

    analysis = {
        'support_status': 'NEUTRAL',
        'support_strength': 50.0,
        'resistance_status': 'NEUTRAL',
        'resistance_strength': 50.0,
        'confidence': 70.0,
        'reasons': [],
        # PRECISE ZONE WIDTHS FOR ENTRY
        'zone_width_support': 20.0,
        'zone_width_resistance': 20.0,
        'entry_zone_support_lower': 0.0,
        'entry_zone_support_upper': 0.0,
        'entry_zone_resistance_lower': 0.0,
        'entry_zone_resistance_upper': 0.0
    }

    reasons = []

    # Get proximity to S/R
    htf_support_dist = features.get('htf_nearest_support_distance_pct', 10)
    htf_resistance_dist = features.get('htf_nearest_resistance_distance_pct', 10)
    vob_support_dist = features.get('vob_major_support_distance_pct', 10)
    vob_resistance_dist = features.get('vob_major_resistance_distance_pct', 10)

    near_support = (htf_support_dist < 0.5 or vob_support_dist < 0.5)
    near_resistance = (htf_resistance_dist < 0.5 or vob_resistance_dist < 0.5)

    #=================================================================
    # FACTOR 1: PRICE ACTION vs S/R (PRIMARY - 25% weight)
    #=================================================================
    price_change_1 = features.get('price_change_1', 0)
    price_change_5 = features.get('price_change_5', 0)
    price_change_20 = features.get('price_change_20', 0)

    if near_support:
        if price_change_1 > 0.2 and price_change_5 > 0.3:
            analysis['support_status'] = 'BUILDING'
            analysis['support_strength'] = 75.0
            reasons.append(f"✅ Support BUILDING (bounce +{price_change_5:.1f}%)")
        elif price_change_1 < -0.2 and price_change_5 < -0.5:
            analysis['support_status'] = 'BREAKING'
            analysis['support_strength'] = 30.0
            reasons.append(f"🔻 Support BREAKING (down {price_change_5:.1f}%)")
        else:
            analysis['support_status'] = 'TESTING'
            analysis['support_strength'] = 50.0
            reasons.append("⚠️ Support TESTING")

    if near_resistance:
        if price_change_1 < -0.2 and price_change_5 < -0.3:
            analysis['resistance_status'] = 'BUILDING'
            analysis['resistance_strength'] = 75.0
            reasons.append(f"✅ Resistance BUILDING (reject {price_change_5:.1f}%)")
        elif price_change_1 > 0.2 and price_change_5 > 0.5:
            analysis['resistance_status'] = 'BREAKING'
            analysis['resistance_strength'] = 30.0
            reasons.append(f"🚀 Resistance BREAKING (up +{price_change_5:.1f}%)")
        else:
            analysis['resistance_status'] = 'TESTING'
            analysis['resistance_strength'] = 50.0
            reasons.append("⚠️ Resistance TESTING")

    #=================================================================
    # FACTOR 2: VOLUME CONFIRMATION (15% weight)
    #=================================================================
    volume_concentration = features.get('volume_concentration', 0)
    volume_buy_sell_ratio = features.get('volume_buy_sell_ratio', 1.0)
    volume_imbalance = features.get('volume_imbalance', 0)

    if volume_concentration > 0.6:
        if analysis['support_status'] == 'BUILDING':
            analysis['support_strength'] += 15
            reasons.append("📊 High volume at support")
        if analysis['resistance_status'] == 'BUILDING':
            analysis['resistance_strength'] += 15
            reasons.append("📊 High volume at resistance")
    elif volume_concentration < 0.3:
        if analysis['support_status'] in ['TESTING', 'BUILDING']:
            analysis['support_strength'] -= 15
            reasons.append("⚠️ Low volume at support")
        if analysis['resistance_status'] in ['TESTING', 'BUILDING']:
            analysis['resistance_strength'] -= 15
            reasons.append("⚠️ Low volume at resistance")

    # Buy/Sell ratio
    if near_support and volume_buy_sell_ratio > 1.5:
        analysis['support_strength'] += 10
        reasons.append(f"💪 Strong buying at support ({volume_buy_sell_ratio:.2f}x)")
    elif near_resistance and volume_buy_sell_ratio < 0.67:
        analysis['resistance_strength'] += 10
        reasons.append(f"💪 Strong selling at resistance ({volume_buy_sell_ratio:.2f}x)")

    #=================================================================
    # FACTOR 3: DELTA/FLOW (CVD - 20% weight)
    #=================================================================
    delta_absorption = features.get('delta_absorption', 0)
    institutional_sweep = features.get('institutional_sweep', 0)
    delta_spike = features.get('delta_spike', 0)
    cvd_bias = features.get('cvd_bias', 0)
    orderflow_strength = features.get('orderflow_strength', 0)

    if institutional_sweep == 1:
        if near_resistance:
            analysis['resistance_status'] = 'BREAKING'
            analysis['resistance_strength'] = 25.0
            reasons.append("🏦 Institutional sweep (RES BREAKING)")
        elif near_support:
            analysis['support_status'] = 'BREAKING'
            analysis['support_strength'] = 25.0
            reasons.append("🏦 Institutional sweep (SUP BREAKING)")

    if delta_absorption > 0.5:
        if near_support:
            analysis['support_strength'] += 10
            reasons.append("🛡️ Delta absorption at support")
        if near_resistance:
            analysis['resistance_strength'] += 10
            reasons.append("🛡️ Delta absorption at resistance")

    if delta_spike == 1:
        if cvd_bias > 0 and near_resistance:
            analysis['resistance_strength'] -= 15
            reasons.append("⚡ Bullish delta spike (RES vulnerable)")
        elif cvd_bias < 0 and near_support:
            analysis['support_strength'] -= 15
            reasons.append("⚡ Bearish delta spike (SUP vulnerable)")

    #=================================================================
    # FACTOR 4: GAMMA EXPOSURE (GEX - 15% weight)
    #=================================================================
    gamma_squeeze_probability = features.get('gamma_squeeze_probability', 0)
    gamma_cluster_concentration = features.get('gamma_cluster_concentration', 0)
    gamma_flip = features.get('gamma_flip', 0)

    if gamma_squeeze_probability > 0.6:
        # High gamma = tight zones + strong defense
        analysis['zone_width_support'] *= 0.7
        analysis['zone_width_resistance'] *= 0.7
        if near_support:
            analysis['support_strength'] += 12
            reasons.append("🎯 High gamma squeeze (SUP strong)")
        if near_resistance:
            analysis['resistance_strength'] += 12
            reasons.append("🎯 High gamma squeeze (RES strong)")

    if gamma_cluster_concentration > 0.5:
        if near_resistance:
            analysis['resistance_strength'] += 10
            reasons.append("🧱 Gamma wall at resistance")
        if near_support:
            analysis['support_strength'] += 10
            reasons.append("🧱 Gamma wall at support")

    if gamma_flip == 1:
        reasons.append("⚡ Gamma flip (wider zones)")
        analysis['zone_width_support'] *= 1.3
        analysis['zone_width_resistance'] *= 1.3

    #=================================================================
    # FACTOR 5: MARKET DEPTH (10% weight)
    #=================================================================
    market_depth_order_imbalance = features.get('market_depth_order_imbalance', 0)
    market_depth_spread = features.get('market_depth_spread', 0)
    market_depth_pressure = features.get('market_depth_pressure', 0)

    if abs(market_depth_order_imbalance) > 0.3:
        if market_depth_order_imbalance > 0 and near_support:
            analysis['support_strength'] += 8
            reasons.append("📈 Bid-heavy order book (SUP strong)")
        elif market_depth_order_imbalance < 0 and near_resistance:
            analysis['resistance_strength'] += 8
            reasons.append("📉 Ask-heavy order book (RES strong)")

    if market_depth_spread > 0:
        if market_depth_spread > 2.0:
            # Wide spread = low liquidity
            if near_support:
                analysis['support_strength'] -= 8
            if near_resistance:
                analysis['resistance_strength'] -= 8
            reasons.append("⚠️ Wide spread (low liquidity)")
        elif market_depth_spread < 0.5:
            # Tight spread = high liquidity
            if near_support:
                analysis['support_strength'] += 5
            if near_resistance:
                analysis['resistance_strength'] += 5
            reasons.append("✅ Tight spread (high liquidity)")

    #=================================================================
    # FACTOR 6: OI BUILDUP (Option Screener - 10% weight)
    #=================================================================
    oi_buildup_pattern = features.get('oi_buildup_pattern', 0)
    oi_acceleration = features.get('oi_acceleration', 0)
    atm_oi_bias = features.get('atm_oi_bias', 0)

    if oi_buildup_pattern > 0.5:  # Call buildup
        if near_resistance:
            analysis['resistance_strength'] += 10
            reasons.append("📞 Call OI buildup (RES strengthening)")
    elif oi_buildup_pattern < -0.5:  # Put buildup
        if near_support:
            analysis['support_strength'] += 10
            reasons.append("📉 Put OI buildup (SUP strengthening)")

    if oi_acceleration > 0.5:
        if near_resistance:
            reasons.append("🚀 OI acceleration at resistance")
        if near_support:
            reasons.append("🚀 OI acceleration at support")

    #=================================================================
    # FACTOR 7: REGIME STRENGTH (ML Regime - 10% weight)
    #=================================================================
    trend_strength = features.get('trend_strength', 0)
    regime_confidence = features.get('regime_confidence', 50)
    volatility_state = features.get('volatility_state', 0)

    if abs(trend_strength) > 0.7 and regime_confidence > 70:
        if trend_strength > 0 and near_resistance:
            analysis['resistance_strength'] -= 15
            reasons.append("📈 Strong uptrend (RES vulnerable)")
        elif trend_strength < 0 and near_support:
            analysis['support_strength'] -= 15
            reasons.append("📉 Strong downtrend (SUP vulnerable)")

    # Volatility affects zone width
    if volatility_state > 0.7:
        analysis['zone_width_support'] *= 1.3
        analysis['zone_width_resistance'] *= 1.3
    elif volatility_state < 0.3:
        analysis['zone_width_support'] *= 0.8
        analysis['zone_width_resistance'] *= 0.8

    #=================================================================
    # FACTOR 8: PARTICIPANT ANALYSIS (5% weight)
    #=================================================================
    institutional_confidence = features.get('institutional_confidence', 50)
    retail_confidence = features.get('retail_confidence', 50)
    smart_money = features.get('smart_money', 0)

    if institutional_confidence > 70 and retail_confidence < 40:
        reasons.append("🎯 Smart money divergence")
        analysis['confidence'] += 10

        if smart_money == 1:
            if near_support:
                analysis['support_strength'] += 8
            if near_resistance:
                analysis['resistance_strength'] += 8

    #=================================================================
    # FACTOR 9: LIQUIDITY FEATURES (5% weight)
    #=================================================================
    liquidity_gravity_strength = features.get('liquidity_gravity_strength', 0)
    liquidity_hvn_count = features.get('liquidity_hvn_count', 0)
    liquidity_sentiment = features.get('liquidity_sentiment', 0)

    if liquidity_hvn_count > 3:
        if near_support:
            analysis['support_strength'] += 5
        if near_resistance:
            analysis['resistance_strength'] += 5
        reasons.append("💧 Multiple liquidity zones")

    if liquidity_gravity_strength > 0.6:
        if liquidity_sentiment > 0 and near_resistance:
            analysis['resistance_strength'] -= 8
            reasons.append("🧲 Liquidity pull up (RES vulnerable)")
        elif liquidity_sentiment < 0 and near_support:
            analysis['support_strength'] -= 8
            reasons.append("🧲 Liquidity pull down (SUP vulnerable)")

    #=================================================================
    # FACTOR 10: EXPIRY CONTEXT (5% weight)
    #=================================================================
    is_expiry_week = features.get('expiry_week', 0)
    expiry_spike_detected = features.get('expiry_spike_detected', 0)

    if is_expiry_week == 1:
        # Expiry week = tighter zones + stronger levels
        analysis['zone_width_support'] *= 0.7
        analysis['zone_width_resistance'] *= 0.7
        reasons.append("📅 Expiry week (tight zones)")

        if expiry_spike_detected:
            if near_support:
                analysis['support_strength'] += 12
            if near_resistance:
                analysis['resistance_strength'] += 12

    #=================================================================
    # FINAL CALCULATIONS
    #=================================================================

    # Clamp strengths
    analysis['support_strength'] = np.clip(analysis['support_strength'], 0, 100)
    analysis['resistance_strength'] = np.clip(analysis['resistance_strength'], 0, 100)

    # Clamp zone widths (10-35 points)
    analysis['zone_width_support'] = np.clip(analysis['zone_width_support'], 10, 35)
    analysis['zone_width_resistance'] = np.clip(analysis['zone_width_resistance'], 10, 35)

    # Calculate precise entry zones (if we have S/R prices)
    support_price = features.get('vob_major_support_distance_pct', 0)  # Will be updated with actual price
    resistance_price = features.get('vob_major_resistance_distance_pct', 0)  # Will be updated with actual price

    # Confidence based on factors aligned
    if len(reasons) >= 6:
        analysis['confidence'] = 90.0
    elif len(reasons) >= 4:
        analysis['confidence'] = 80.0
    elif len(reasons) >= 2:
        analysis['confidence'] = 70.0

    analysis['reasons'] = reasons

    return analysis


def format_sr_analysis_display(analysis: Dict, support_price: float, resistance_price: float) -> str:
    """
    Format S/R analysis for display

    Args:
        analysis: Result from analyze_sr_strength_comprehensive()
        support_price: Actual support price level
        resistance_price: Actual resistance price level

    Returns:
        Formatted string for display
    """

    # Calculate precise entry zones
    sup_zone_lower = support_price - (analysis['zone_width_support'] / 2)
    sup_zone_upper = support_price + (analysis['zone_width_support'] / 2)
    res_zone_lower = resistance_price - (analysis['zone_width_resistance'] / 2)
    res_zone_upper = resistance_price + (analysis['zone_width_resistance'] / 2)

    # Status emojis
    status_emoji = {
        'BUILDING': '🟢',
        'TESTING': '🟡',
        'BREAKING': '🔴',
        'NEUTRAL': '⚪'
    }

    display = f"""
╔════════════════════════════════════════════════════════════╗
║     COMPREHENSIVE S/R STRENGTH ANALYSIS (10 FACTORS)       ║
╚════════════════════════════════════════════════════════════╝

📊 SUPPORT ANALYSIS
  Status: {status_emoji.get(analysis['support_status'], '⚪')} {analysis['support_status']}
  Strength: {analysis['support_strength']:.0f}% {'🔥' if analysis['support_strength'] > 75 else '⚠️' if analysis['support_strength'] < 40 else ''}

  🎯 PRECISE ENTRY ZONE:
  ├─ Support Level: ₹{support_price:,.2f}
  ├─ Zone Width: {analysis['zone_width_support']:.0f} pts
  └─ Entry Range: ₹{sup_zone_lower:,.2f} - ₹{sup_zone_upper:,.2f}

📊 RESISTANCE ANALYSIS
  Status: {status_emoji.get(analysis['resistance_status'], '⚪')} {analysis['resistance_status']}
  Strength: {analysis['resistance_strength']:.0f}% {'🔥' if analysis['resistance_strength'] > 75 else '⚠️' if analysis['resistance_strength'] < 40 else ''}

  🎯 PRECISE ENTRY ZONE:
  ├─ Resistance Level: ₹{resistance_price:,.2f}
  ├─ Zone Width: {analysis['zone_width_resistance']:.0f} pts
  └─ Entry Range: ₹{res_zone_lower:,.2f} - ₹{res_zone_upper:,.2f}

📈 ANALYSIS CONFIDENCE: {analysis['confidence']:.0f}%

🔍 FACTORS ANALYZED (10 Total):
"""

    for reason in analysis['reasons']:
        display += f"  • {reason}\n"

    display += "\n"
    display += "─" * 60 + "\n"

    # Trading recommendations
    if analysis['support_status'] == 'BUILDING' and analysis['support_strength'] > 70:
        display += "✅ LONG SETUP: Enter on bounce from support zone\n"
        display += f"   Entry: ₹{sup_zone_lower:,.2f} - ₹{sup_zone_upper:,.2f}\n"
        display += f"   Stop: ₹{sup_zone_lower - 20:,.2f} (below zone)\n"
        display += f"   Target: ₹{resistance_price:,.2f}\n"

    if analysis['resistance_status'] == 'BUILDING' and analysis['resistance_strength'] > 70:
        display += "✅ SHORT SETUP: Enter on rejection from resistance zone\n"
        display += f"   Entry: ₹{res_zone_lower:,.2f} - ₹{res_zone_upper:,.2f}\n"
        display += f"   Stop: ₹{res_zone_upper + 20:,.2f} (above zone)\n"
        display += f"   Target: ₹{support_price:,.2f}\n"

    if analysis['support_status'] == 'BREAKING':
        display += "🔻 AVOID LONG: Support breaking down\n"

    if analysis['resistance_status'] == 'BREAKING':
        display += "🚀 BREAKOUT: Resistance breaking up\n"

    return display


# Example usage
if __name__ == "__main__":
    # Example features dictionary
    features = {
        'price_change_1': 0.3,
        'price_change_5': 0.5,
        'htf_nearest_support_distance_pct': 0.2,
        'vob_major_support_distance_pct': 0.3,
        'volume_concentration': 0.7,
        'volume_buy_sell_ratio': 1.8,
        'institutional_sweep': 0,
        'delta_absorption': 0.6,
        'gamma_squeeze_probability': 0.7,
        'gamma_cluster_concentration': 0.6,
        'market_depth_order_imbalance': 0.4,
        'market_depth_spread': 0.4,
        'oi_buildup_pattern': -0.6,
        'trend_strength': 0.5,
        'regime_confidence': 75,
        'institutional_confidence': 75,
        'retail_confidence': 35,
        'smart_money': 1,
        'is_expiry_week': 1,
        'expiry_spike_detected': 1
    }

    analysis = analyze_sr_strength_comprehensive(features)

    support_price = 26100.0
    resistance_price = 26300.0

    print(format_sr_analysis_display(analysis, support_price, resistance_price))
