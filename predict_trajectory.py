import os
import sys
import pickle
import numpy as np
import torch

from train_trajectory_model import TrajectoryGRU

def main():
    print("=========================================")
    print("Vehicle C Trajectory Inference Test")
    print("=========================================")
    
    models_dir = "models"
    model_path = os.path.join(models_dir, "vehicle_trajectory_gru.pth")
    feature_scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
    target_scaler_path = os.path.join(models_dir, "target_scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(feature_scaler_path):
        print("Error: Model or scaler files not found. Please run train_trajectory_model.py first.")
        sys.exit(1)
        
    print("Loading scalers...")
    with open(feature_scaler_path, "rb") as f:
        feature_scaler = pickle.load(f)
    with open(target_scaler_path, "rb") as f:
        target_scaler = pickle.load(f)
        
    print("Loading GRU model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 8 input features, 64 hidden units
    model = TrajectoryGRU(input_size=8, hidden_size=64).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    
    print("\nCreating a mock 30-timestep history (1 second)...")
    # In reality, this comes directly from the active simulation via V2X.
    # We will simulate a straight line approach for testing the pipeline.
    
    # 30 steps, 8 features
    mock_history = np.zeros((30, 8), dtype=np.float32)
    # Start at x=200, y=4, speed=12, moving exactly left
    for t in range(30):
        mock_history[t, 0] = 200.0 - (12.0 * t * 0.033)  # x
        mock_history[t, 1] = 4.0                         # y
        mock_history[t, 2] = 12.0                        # speed
        mock_history[t, 3] = -12.0                       # velocity_x
        mock_history[t, 4] = 0.0                         # velocity_y
        mock_history[t, 5] = 3.14159                     # heading
        mock_history[t, 6] = 1.0                         # lane
        mock_history[t, 7] = 0.0                         # accel
        
    current_x = mock_history[-1, 0]
    current_y = mock_history[-1, 1]
    
    print("Applying preprocessing...")
    # Reshape to (1, 30, 8), then flatten to scale, then reshape back
    mock_flat = mock_history.reshape(-1, 8)
    mock_scaled = feature_scaler.transform(mock_flat)
    mock_scaled = mock_scaled.reshape(1, 30, 8)
    
    print("Running AI inference...")
    input_tensor = torch.tensor(mock_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy()
        
    print("Inverse scaling output...")
    pred_raw = target_scaler.inverse_transform(pred_scaled)[0]
    
    future_x, future_y = pred_raw[0], pred_raw[1]
    
    print("\n-----------------------------------------")
    print("Vehicle C trajectory prediction")
    print("-----------------------------------------")
    print("Current position:")
    print(f"X = {current_x:.3f} m")
    print(f"Y = {current_y:.3f} m")
    print("\nPredicted position after ~0.49 seconds:")
    print(f"X = {future_x:.3f} m")
    print(f"Y = {future_y:.3f} m")
    
    # Simple sanity check for the user
    # CV baseline would predict current_x - (12 * 0.495)
    cv_x = current_x - 12.0 * 0.495
    cv_y = current_y
    print("\n(Baseline Physics Check)")
    print(f"CV Predicted X: {cv_x:.3f} m")
    print(f"CV Predicted Y: {cv_y:.3f} m")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()
