from config import STRIKE_INTERVALS, SENSEX_NIFTY_RATIO

def calculate_strike(index: str, nifty_price: float, direction: str, expiry_date: str):
    """Calculate option strike - Always uses ATM"""

    interval = STRIKE_INTERVALS[index]

    # Calculate spot price for selected index
    if index == "SENSEX":
        spot_price = nifty_price * SENSEX_NIFTY_RATIO
    else:
        spot_price = nifty_price

    # Calculate ATM
    atm_strike = round(spot_price / interval) * interval

    # Always use ATM strike
    strike = atm_strike
    strike_type = "ATM"

    return {
        'strike': strike,
        'strike_type': strike_type,
        'atm_strike': atm_strike,
        'spot_price': spot_price,
        'option_type': 'CE' if direction == 'CALL' else 'PE'
    }

def calculate_levels(index: str, direction: str, vob_support: float,
                     vob_resistance: float, sl_offset: int = 8):
    """
    Calculate entry, SL, target levels based on VOB/HTF levels

    Args:
        index: "NIFTY" or "SENSEX"
        direction: "CALL" or "PUT"
        vob_support: Support level (VOB or HTF)
        vob_resistance: Resistance level (VOB or HTF)
        sl_offset: Stop loss offset in points (default 8)

    Returns:
        dict with entry, target, SL levels and risk-reward ratio

    Logic:
        For CALL trades:
            - Entry: Support level
            - Target: Resistance level
            - Stop Loss: Support - 8 points

        For PUT trades:
            - Entry: Resistance level
            - Target: Support level
            - Stop Loss: Resistance + 8 points
    """

    # Scale for SENSEX (if needed, levels are already in index units)
    if index == "SENSEX":
        # Note: If vob_support and vob_resistance are already in SENSEX scale, no need to multiply
        # If they're in NIFTY scale, multiply by ratio
        # Assuming they're already scaled appropriately from signal generators
        pass

    if direction == "CALL":
        entry = vob_support
        target = vob_resistance
        sl = vob_support - sl_offset  # Stop loss 8 points below support
    else:  # PUT
        entry = vob_resistance
        target = vob_support
        sl = vob_resistance + sl_offset  # Stop loss 8 points above resistance

    return {
        'entry_level': round(entry, 2),
        'target_level': round(target, 2),
        'sl_level': round(sl, 2),
        'risk_points': abs(entry - sl),
        'reward_points': abs(target - entry),
        'rr_ratio': round(abs(target - entry) / abs(entry - sl), 2) if abs(entry - sl) > 0 else 0
    }
