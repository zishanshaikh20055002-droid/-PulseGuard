import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class HardwareAgnosticBuffer:
    def __init__(self, window_size=30, num_features=14):
        self.window_size = window_size
        self.num_features = num_features
        self.raw_streams = defaultdict(list)
        self.last_known_good = defaultdict(lambda: np.zeros(num_features))

    def process_payload(self, machine_id, step, features):
        try:
            parsed_features = [float(x) for x in features]
        except (TypeError, ValueError):
            parsed_features = []

        if not parsed_features or len(parsed_features) != self.num_features:
            clean_features = self.last_known_good[machine_id].copy()
        else:
            clean_features = np.clip(parsed_features, a_min=-1e6, a_max=1e6).copy()
            self.last_known_good[machine_id] = clean_features.copy()

        self.raw_streams[machine_id].append({
            'step': int(step), 
            'features': clean_features.copy()
        })
        
        if len(self.raw_streams[machine_id]) > self.window_size:
            self.raw_streams[machine_id].pop(0)

    def get_valid_window(self, machine_id):
        stream = self.raw_streams[machine_id]
        
        if len(stream) < self.window_size:
            return None
            
        ordered_packets = sorted(stream, key=lambda x: x['step'])
        
        window = np.array([item['features'] for item in ordered_packets], dtype=np.float32)
        window = np.nan_to_num(window, nan=0.0)
        
        return window.reshape(1, self.window_size, self.num_features)