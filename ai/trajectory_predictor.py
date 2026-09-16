import os
import pickle
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn

class TrajectoryGRU(nn.Module):
    """Identical architecture to the trained PyTorch model."""
    def __init__(self, input_size=8, hidden_size=64, num_layers=1, output_size=2):
        super(TrajectoryGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class AIPredictor:
    def __init__(self, base_dir="."):
        self.history = deque(maxlen=30)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.status = "WAITING FOR HISTORY"
        
        models_dir = os.path.join(base_dir, "models")
        model_path = os.path.join(models_dir, "vehicle_trajectory_gru.pth")
        feat_scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
        target_scaler_path = os.path.join(models_dir, "target_scaler.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model weights: {model_path}")
            
        with open(feat_scaler_path, "rb") as f:
            self.feature_scaler = pickle.load(f)
            
        with open(target_scaler_path, "rb") as f:
            self.target_scaler = pickle.load(f)
            
        self.model = TrajectoryGRU(input_size=8, hidden_size=64).to(self.device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.model.eval()

    def add_state(self, x, y, speed, vel_x, vel_y, heading, lane, accel):
        """Append a new state to the 30-timestep history buffer."""
        state = [x, y, speed, vel_x, vel_y, heading, lane, accel]
        self.history.append(state)
        
        if len(self.history) == 30:
            self.status = "ACTIVE"
            
    def ready(self):
        return len(self.history) == 30
        
    def predict(self):
        """Run GRU inference on the buffered history."""
        if not self.ready():
            return None, None, 0.0
            
        start_time = time.perf_counter()
        
        # Convert history to numpy array
        hist_arr = np.array(self.history, dtype=np.float32)
        
        # Scale inputs using the fitted standard scaler
        hist_flat = hist_arr.reshape(-1, 8)
        hist_scaled = self.feature_scaler.transform(hist_flat)
        hist_scaled = hist_scaled.reshape(1, 30, 8)
        
        input_tensor = torch.tensor(hist_scaled, dtype=torch.float32).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            pred_scaled = self.model(input_tensor).cpu().numpy()
            
        # Inverse scale targets to real-world meters
        pred_raw = self.target_scaler.inverse_transform(pred_scaled)[0]
        pred_x, pred_y = float(pred_raw[0]), float(pred_raw[1])
        
        inference_ms = (time.perf_counter() - start_time) * 1000.0
        return pred_x, pred_y, inference_ms
