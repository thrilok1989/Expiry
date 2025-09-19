#!/usr/bin/env python3
"""
WebSocket Handler for DhanHQ Live Market Data
Handles binary data parsing and live price streaming
"""

import websocket
import json
import struct
import threading
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DhanWebSocketClient:
    def __init__(self, access_token, client_id):
        self.access_token = access_token
        self.client_id = client_id
        self.websocket = None
        self.is_connected = False
        self.subscribed_instruments = []
        self.price_callbacks = []
        self.connection_callbacks = []
        self.message_count = 0
        
    def add_price_callback(self, callback):
        """Add callback function for price updates"""
        self.price_callbacks.append(callback)
    
    def add_connection_callback(self, callback):
        """Add callback function for connection status updates"""
        self.connection_callbacks.append(callback)
    
    def connect(self):
        """Establish WebSocket connection"""
        try:
            ws_url = f"wss://api-feed.dhan.co?version=2&token={self.access_token}&clientId={self.client_id}&authType=2"
            
            # Create WebSocket connection
            self.websocket = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start connection in separate thread
            self.ws_thread = threading.Thread(target=self.websocket.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            logger.info("WebSocket connection initiated...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close WebSocket connection"""
        if self.websocket and self.is_connected:
            disconnect_msg = {"RequestCode": 12}
            self.websocket.send(json.dumps(disconnect_msg))
            self.websocket.close()
            self.is_connected = False
            logger.info("WebSocket disconnected")
    
    def subscribe_instrument(self, exchange_segment, security_id):
        """Subscribe to instrument for live data"""
        if not self.is_connected:
            logger.error("WebSocket not connected")
            return False
        
        try:
            subscribe_message = {
                "RequestCode": 15,  # Subscribe to quote data
                "InstrumentCount": 1,
                "InstrumentList": [
                    {
                        "ExchangeSegment": exchange_segment,
                        "SecurityId": str(security_id)
                    }
                ]
            }
            
            self.websocket.send(json.dumps(subscribe_message))
            self.subscribed_instruments.append({
                "exchange": exchange_segment,
                "security_id": security_id
            })
            
            logger.info(f"Subscribed to {exchange_segment}:{security_id}")
            return True
            
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            return False
    
    def _on_open(self, ws):
        """WebSocket connection opened"""
        self.is_connected = True
        logger.info("WebSocket connected successfully")
        
        # Notify connection callbacks
        for callback in self.connection_callbacks:
            try:
                callback("connected", "WebSocket connected")
            except Exception as e:
                logger.error(f"Connection callback error: {e}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed"""
        self.is_connected = False
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        # Notify connection callbacks
        for callback in self.connection_callbacks:
            try:
                callback("disconnected", f"Connection closed: {close_msg}")
            except Exception as e:
                logger.error(f"Connection callback error: {e}")
    
    def _on_error(self, ws, error):
        """WebSocket error occurred"""
        logger.error(f"WebSocket error: {error}")
        
        # Notify connection callbacks
        for callback in self.connection_callbacks:
            try:
                callback("error", str(error))
            except Exception as e:
                logger.error(f"Connection callback error: {e}")
    
    def _on_message(self, ws, message):
        """Process incoming WebSocket message"""
        try:
            if isinstance(message, bytes):
                self._process_binary_message(message)
            else:
                self._process_text_message(message)
            
            self.message_count += 1
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    def _process_text_message(self, message):
        """Process text message (JSON)"""
        try:
            data = json.loads(message)
            logger.info(f"Text message received: {data}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
    
    def _process_binary_message(self, message):
        """Process binary message according to DhanHQ protocol"""
        try:
            if len(message) < 8:
                logger.warning("Message too short for header")
                return
            
            # Parse response header (8 bytes)
            feed_response_code = struct.unpack('B', message[0:1])[0]
            message_length = struct.unpack('<H', message[1:3])[0]  # Little endian
            exchange_segment = struct.unpack('B', message[3:4])[0]
            security_id = struct.unpack('<I', message[4:8])[0]  # Little endian
            
            logger.debug(f"Header - Code: {feed_response_code}, Length: {message_length}, "
                        f"Segment: {exchange_segment}, SecurityID: {security_id}")
            
            # Process different packet types
            if feed_response_code == 2:  # Ticker packet
                self._process_ticker_packet(message, security_id)
            elif feed_response_code == 4:  # Quote packet
                self._process_quote_packet(message, security_id)
            elif feed_response_code == 6:  # Previous close
                self._process_prev_close_packet(message, security_id)
            elif feed_response_code == 50:  # Disconnect packet
                self._process_disconnect_packet(message)
            else:
                logger.debug(f"Unknown packet type: {feed_response_code}")
                
        except struct.error as e:
            logger.error(f"Binary parsing error: {e}")
        except Exception as e:
            logger.error(f"Binary message processing error: {e}")
    
    def _process_ticker_packet(self, message, security_id):
        """Process ticker packet (LTP + LTT)"""
        try:
            if len(message) >= 16:
                ltp = struct.unpack('<f', message[8:12])[0]  # Little endian float32
                ltt = struct.unpack('<I', message[12:16])[0]  # Little endian int32
                
                price_data = {
                    'type': 'ticker',
                    'security_id': security_id,
                    'ltp': ltp,
                    'ltt': ltt,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Ticker - SecurityID: {security_id}, LTP: {ltp}, LTT: {ltt}")
                
                # Notify price callbacks
                self._notify_price_callbacks(price_data)
                
        except struct.error as e:
            logger.error(f"Ticker packet parsing error: {e}")
    
    def _process_quote_packet(self, message, security_id):
        """Process quote packet (detailed price data)"""
        try:
            if len(message) >= 50:  # Minimum size for quote packet
                ltp = struct.unpack('<f', message[8:12])[0]
                ltq = struct.unpack('<H', message[12:14])[0]
                ltt = struct.unpack('<I', message[14:18])[0]
                atp = struct.unpack('<f', message[18:22])[0]
                volume = struct.unpack('<I', message[22:26])[0]
                total_sell_qty = struct.unpack('<I', message[26:30])[0]
                total_buy_qty = struct.unpack('<I', message[30:34])[0]
                
                price_data = {
                    'type': 'quote',
                    'security_id': security_id,
                    'ltp': ltp,
                    'ltq': ltq,
                    'ltt': ltt,
                    'atp': atp,
                    'volume': volume,
                    'total_sell_qty': total_sell_qty,
                    'total_buy_qty': total_buy_qty,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Quote - SecurityID: {security_id}, LTP: {ltp}, Volume: {volume}")
                
                # Notify price callbacks
                self._notify_price_callbacks(price_data)
                
        except struct.error as e:
            logger.error(f"Quote packet parsing error: {e}")
    
    def _process_prev_close_packet(self, message, security_id):
        """Process previous close packet"""
        try:
            if len(message) >= 16:
                prev_close = struct.unpack('<f', message[8:12])[0]
                prev_oi = struct.unpack('<I', message[12:16])[0]
                
                price_data = {
                    'type': 'prev_close',
                    'security_id': security_id,
                    'prev_close': prev_close,
                    'prev_oi': prev_oi,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Prev Close - SecurityID: {security_id}, Close: {prev_close}, OI: {prev_oi}")
                
                # Notify price callbacks
                self._notify_price_callbacks(price_data)
                
        except struct.error as e:
            logger.error(f"Previous close packet parsing error: {e}")
    
    def _process_disconnect_packet(self, message):
        """Process disconnect packet"""
        try:
            if len(message) >= 10:
                disconnect_code = struct.unpack('<H', message[8:10])[0]
                logger.warning(f"Received disconnect packet with code: {disconnect_code}")
                
                # Notify connection callbacks
                for callback in self.connection_callbacks:
                    try:
                        callback("server_disconnect", f"Server disconnect code: {disconnect_code}")
                    except Exception as e:
                        logger.error(f"Connection callback error: {e}")
                        
        except struct.error as e:
            logger.error(f"Disconnect packet parsing error: {e}")
    
    def _notify_price_callbacks(self, price_data):
        """Notify all registered price callbacks"""
        for callback in self.price_callbacks:
            try:
                callback(price_data)
            except Exception as e:
                logger.error(f"Price callback error: {e}")
    
    def get_connection_status(self):
        """Get current connection status"""
        return {
            'is_connected': self.is_connected,
            'subscribed_instruments': len(self.subscribed_instruments),
            'message_count': self.message_count,
            'instruments': self.subscribed_instruments
        }

# Example usage and testing
if __name__ == "__main__":
    # Example callback functions
    def on_price_update(price_data):
        print(f"Price Update: {price_data}")
    
    def on_connection_change(status, message):
        print(f"Connection Status: {status} - {message}")
    
    # Test the WebSocket client
    # Replace with your actual credentials
    ACCESS_TOKEN = "your_access_token_here"
    CLIENT_ID = "your_client_id_here"
    SECURITY_ID = "your_security_id_here"  # ATM Call option
    
    # Create client
    client = DhanWebSocketClient(ACCESS_TOKEN, CLIENT_ID)
    
    # Add callbacks
    client.add_price_callback(on_price_update)
    client.add_connection_callback(on_connection_change)
    
    # Connect
    if client.connect():
        # Wait for connection
        time.sleep(2)
        
        # Subscribe to instrument
        client.subscribe_instrument("NSE_FNO", SECURITY_ID)
        
        # Keep running
        try:
            while True:
                time.sleep(1)
                status = client.get_connection_status()
                print(f"Status: Connected={status['is_connected']}, Messages={status['message_count']}")
        except KeyboardInterrupt:
            print("Stopping...")
            client.disconnect()
    else:
        print("Failed to connect")