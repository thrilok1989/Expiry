from dhan_api import DhanAPI
from telegram_alerts import TelegramBot
from strike_calculator import calculate_strike, calculate_levels
from config import LOT_SIZES, DEMO_MODE
import streamlit as st

class TradeExecutor:
    def __init__(self):
        """Initialize trade executor"""
        self.dhan = DhanAPI() if not DEMO_MODE else None
        self.telegram = TelegramBot()
    
    def execute_trade(self, setup: dict, nifty_price: float, expiry_date: str):
        """
        Execute trade based on signal setup
        
        Args:
            setup: Signal setup dict
            nifty_price: Current NIFTY price
            expiry_date: Current expiry date
        
        Returns:
            dict with execution result
        """
        
        index = setup['index']
        direction = setup['direction']
        
        # Calculate strike
        strike_info = calculate_strike(index, nifty_price, direction, expiry_date)
        
        # Calculate levels
        levels = calculate_levels(
            index, 
            direction,
            setup['vob_support'],
            setup['vob_resistance']
        )
        
        # Calculate quantity
        lot_size = LOT_SIZES[index]
        quantity = lot_size  # 1 lot
        
        # Prepare order details
        order_details = {
            'index': index,
            'strike': strike_info['strike'],
            'option_type': strike_info['option_type'],
            'direction': 'BUY',  # Always BUY for long positions
            'quantity': quantity,
            'sl_price': levels['sl_level'],
            'target_price': levels['target_level'],
            'strike_type': strike_info['strike_type'],
            'entry_level': levels['entry_level'],
            'rr_ratio': levels['rr_ratio']
        }
        
        # Demo mode
        if DEMO_MODE:
            return {
                'success': True,
                'order_id': f"DEMO_{index}_{direction}_{int(strike_info['strike'])}",
                'message': 'DEMO MODE - Order simulated',
                'order_details': order_details
            }
        
        # Place order via DhanHQ
        try:
            result = self.dhan.place_super_order(
                index=order_details['index'],
                strike=order_details['strike'],
                option_type=order_details['option_type'],
                direction=order_details['direction'],
                quantity=order_details['quantity'],
                sl_price=order_details['sl_price'],
                target_price=order_details['target_price']
            )
            
            if result['success']:
                # Send Telegram alert
                self.telegram.send_order_placed(
                    setup,
                    result['order_id'],
                    order_details['strike'],
                    order_details['sl_price'],
                    order_details['target_price']
                )
                
                return {
                    'success': True,
                    'order_id': result['order_id'],
                    'message': 'Order placed successfully',
                    'order_details': order_details
                }
            else:
                # Send failure alert
                self.telegram.send_order_failed(setup, result.get('error', 'Unknown error'))
                
                return {
                    'success': False,
                    'error': result.get('error'),
                    'message': result.get('message'),
                    'order_details': order_details
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Trade execution failed',
                'order_details': order_details
            }
