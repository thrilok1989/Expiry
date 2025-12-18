import json
import os
from datetime import datetime
import pytz
from config import IST, get_current_time_ist
from typing import Dict, Optional

SIGNALS_FILE = "trading_signals.json"

class SignalManager:
    def __init__(self, filepath=SIGNALS_FILE):
        self.filepath = filepath
        self.signals = self._load_signals()
    
    def _load_signals(self) -> Dict:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_signals(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.signals, f, indent=4)
    
    def create_setup(self, index: str, direction: str, 
                     vob_support: float, vob_resistance: float) -> str:
        """Create new signal setup"""
        signal_id = f"{index}_{direction}_{int(vob_support)}_{int(vob_resistance)}_{get_current_time_ist().strftime('%Y%m%d_%H%M%S')}"
        
        self.signals[signal_id] = {
            'index': index,
            'direction': direction,
            'vob_support': vob_support,
            'vob_resistance': vob_resistance,
            'signals': [],
            'signal_count': 0,
            'status': 'pending',
            'created_at': get_current_time_ist().isoformat(),
            'executed_at': None,
            'order_id': None
        }
        
        self._save_signals()
        return signal_id
    
    def add_signal(self, signal_id: str) -> bool:
        """Add signal to setup"""
        if signal_id not in self.signals:
            return False
        
        self.signals[signal_id]['signals'].append({
            'timestamp': get_current_time_ist().isoformat()
        })
        self.signals[signal_id]['signal_count'] += 1
        
        if self.signals[signal_id]['signal_count'] >= 3:
            self.signals[signal_id]['status'] = 'ready'
        
        self._save_signals()
        return True
    
    def remove_signal(self, signal_id: str) -> bool:
        """Remove last signal"""
        if signal_id not in self.signals:
            return False
        
        if len(self.signals[signal_id]['signals']) > 0:
            self.signals[signal_id]['signals'].pop()
            self.signals[signal_id]['signal_count'] -= 1
            
            if self.signals[signal_id]['signal_count'] < 3:
                self.signals[signal_id]['status'] = 'pending'
            
            self._save_signals()
            return True
        return False
    
    def mark_executed(self, signal_id: str, order_id: str):
        """Mark as executed"""
        if signal_id in self.signals:
            self.signals[signal_id]['status'] = 'executed'
            self.signals[signal_id]['executed_at'] = get_current_time_ist().isoformat()
            self.signals[signal_id]['order_id'] = order_id
            self._save_signals()
    
    def delete_setup(self, signal_id: str):
        """Delete signal setup"""
        if signal_id in self.signals:
            del self.signals[signal_id]
            self._save_signals()
    
    def get_active_setups(self) -> Dict:
        """Get active setups"""
        return {
            sid: data for sid, data in self.signals.items() 
            if data['status'] in ['pending', 'ready']
        }
    
    def get_setup(self, signal_id: str) -> Optional[Dict]:
        """Get specific setup"""
        return self.signals.get(signal_id)
