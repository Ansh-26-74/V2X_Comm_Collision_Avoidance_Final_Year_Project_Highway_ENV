import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class TrajectoryGRU(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=1, output_size=2):
        super(TrajectoryGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq, feature)
        out, _ = self.gru(x)
        # We want the output at the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def evaluate_metrics(y_true, y_pred, name="Model"):
    # y_true and y_pred are (N, 2) in raw meters
    err_x = np.abs(y_true[:, 0] - y_pred[:, 0])
    err_y = np.abs(y_true[:, 1] - y_pred[:, 1])
    
    mae_x = np.mean(err_x)
    mae_y = np.mean(err_y)
    
    rmse_x = np.sqrt(np.mean(err_x**2))
    rmse_y = np.sqrt(np.mean(err_y**2))
    
    euclidean = np.sqrt(err_x**2 + err_y**2)
    mean_euc = np.mean(euclidean)
    median_euc = np.median(euclidean)
    mae_euc = mean_euc  # Same as mean euclidean error in this context
    rmse_euc = np.sqrt(np.mean(euclidean**2))
    
    return {
        "mae_x": mae_x, "mae_y": mae_y,
        "rmse_x": rmse_x, "rmse_y": rmse_y,
        "mean_euc": mean_euc, "median_euc": median_euc,
        "rmse_euc": rmse_euc
    }

def print_metrics(metrics, name):
    print(f"\n--- {name} Performance ---")
    print(f"X-position MAE: {metrics['mae_x']:.4f} m")
    print(f"Y-position MAE: {metrics['mae_y']:.4f} m")
    print(f"X-position RMSE: {metrics['rmse_x']:.4f} m")
    print(f"Y-position RMSE: {metrics['rmse_y']:.4f} m")
    print(f"Euclidean Mean (MAE): {metrics['mean_euc']:.4f} m")
    print(f"Euclidean Median: {metrics['median_euc']:.4f} m")
    print(f"Euclidean RMSE: {metrics['rmse_euc']:.4f} m")

def get_constant_velocity_baseline(X_raw, horizon_dt=0.495):
    # X_raw shape: (N, 30, 8)
    # Features: x(0), y(1), speed(2), vel_x(3), vel_y(4), heading(5), lane(6), accel(7)
    
    # Last historical timestep
    last_step = X_raw[:, -1, :]
    
    current_x = last_step[:, 0]
    current_y = last_step[:, 1]
    vel_x = last_step[:, 3]
    vel_y = last_step[:, 4]
    
    pred_x = current_x + vel_x * horizon_dt
    pred_y = current_y + vel_y * horizon_dt
    
    return np.column_stack((pred_x, pred_y))

def main():
    parser = argparse.ArgumentParser(description="Train GRU Trajectory Predictor")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    data_dir = os.path.join("data", "processed")
    models_dir = "models"
    results_dir = os.path.join("data", "results")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading datasets...")
    train_data = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=True)
    val_data = np.load(os.path.join(data_dir, "validation.npz"), allow_pickle=True)
    test_data = np.load(os.path.join(data_dir, "test.npz"), allow_pickle=True)
    
    X_train_raw, y_train_raw = train_data['X'], train_data['y']
    X_val_raw, y_val_raw = val_data['X'], val_data['y']
    X_test_raw, y_test_raw = test_data['X'], test_data['y']
    
    test_scenarios = test_data['scenarios']
    
    print("\n--- Step 1: Baseline Evaluation ---")
    cv_preds = get_constant_velocity_baseline(X_test_raw, horizon_dt=0.495)
    baseline_metrics = evaluate_metrics(y_test_raw, cv_preds, "Constant Velocity")
    print_metrics(baseline_metrics, "Constant Velocity Baseline")
    
    print("\n--- Step 2: Data Normalization ---")
    # Flatten X for scaling, then reshape back
    N_train, T, F = X_train_raw.shape
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Fit strictly on train
    X_train_flat = X_train_raw.reshape(-1, F)
    feature_scaler.fit(X_train_flat)
    target_scaler.fit(y_train_raw)
    
    def scale_X(X_raw):
        N = X_raw.shape[0]
        return feature_scaler.transform(X_raw.reshape(-1, F)).reshape(N, T, F)
        
    X_train = scale_X(X_train_raw)
    y_train = target_scaler.transform(y_train_raw)
    
    X_val = scale_X(X_val_raw)
    y_val = target_scaler.transform(y_val_raw)
    
    X_test = scale_X(X_test_raw)
    y_test = target_scaler.transform(y_test_raw)
    
    with open(os.path.join(models_dir, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(os.path.join(models_dir, "target_scaler.pkl"), "wb") as f:
        pickle.dump(target_scaler, f)
    print("Scalers fitted and saved.")
    
    print("\n--- Step 3 & 4: Model Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = TrajectoryGRU(input_size=F, hidden_size=args.hidden_size).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: GRU (Hidden: {args.hidden_size}) -> Dense(2)")
    print(f"Trainable parameters: {num_params}")
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    model_path = os.path.join(models_dir, "vehicle_trajectory_gru.pth")
    
    epochs_run = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
            
        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item() * batch_X.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        epochs_run += 1
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch+1:03d} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f} (Best)")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1:03d} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")
            
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    # Plot Training Curve
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Huber Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(results_dir, "training_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()
    
    print("\n--- Step 6: Test Evaluation ---")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    test_tensor = torch.tensor(X_test).to(device)
    with torch.no_grad():
        gru_preds_scaled = model(test_tensor).cpu().numpy()
        
    gru_preds_raw = target_scaler.inverse_transform(gru_preds_scaled)
    gru_metrics = evaluate_metrics(y_test_raw, gru_preds_raw, "GRU AI")
    print_metrics(gru_metrics, "GRU AI Model")
    
    baseline_err = baseline_metrics['mean_euc']
    gru_err = gru_metrics['mean_euc']
    improvement = ((baseline_err - gru_err) / baseline_err) * 100
    
    print("\n--- Comparison ---")
    print(f"Baseline Euclidean MAE: {baseline_err:.4f} m")
    print(f"GRU Euclidean MAE:      {gru_err:.4f} m")
    print(f"Improvement:            {improvement:.2f}%")
    
    print("\n--- Step 7: Scenario-wise Evaluation ---")
    print("Scenario                         MAE (m)")
    print("-" * 40)
    unique_scenarios = np.unique(test_scenarios)
    for sc in unique_scenarios:
        idx = (test_scenarios == sc)
        sc_y_true = y_test_raw[idx]
        sc_y_pred = gru_preds_raw[idx]
        sc_metrics = evaluate_metrics(sc_y_true, sc_y_pred)
        print(f"{sc:<30} {sc_metrics['mean_euc']:.4f}")
        
    print("\n--- Step 8: Visualization ---")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Pick 6 random examples to plot
    np.random.seed(42)
    sample_indices = np.random.choice(len(y_test_raw), size=6, replace=False)
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        
        hist_x = X_test_raw[idx, :, 0]
        hist_y = X_test_raw[idx, :, 1]
        
        true_x, true_y = y_test_raw[idx]
        pred_x, pred_y = gru_preds_raw[idx]
        
        ax.plot(hist_x, hist_y, 'k.-', label='History (30 steps)')
        ax.plot(true_x, true_y, 'go', markersize=10, label='Actual Future')
        ax.plot(pred_x, pred_y, 'rx', markersize=10, label='GRU Predicted Future')
        
        scenario = test_scenarios[idx]
        ax.set_title(f"Scenario: {scenario}")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "prediction_examples.png")
    plt.savefig(plot_path)
    plt.close()
    
    print("\n=========================================")
    print("FINAL OUTPUT SUMMARY")
    print("=========================================")
    print(f"1. Model architecture: GRU (Hidden={args.hidden_size}) -> Dense(2)")
    print(f"2. Trainable parameters: {num_params}")
    print(f"3. Training epochs completed: {epochs_run}")
    print(f"4. Best validation loss: {best_val_loss:.6f}")
    print(f"5. Test MAE (Euclidean): {gru_err:.4f} m")
    print(f"6. Test RMSE (Euclidean): {gru_metrics['rmse_euc']:.4f} m")
    print(f"7. GRU Euclidean position error: {gru_err:.4f} m")
    print(f"8. Constant-velocity baseline error: {baseline_err:.4f} m")
    print(f"9. Percentage improvement over baseline: {improvement:.2f}%")
    print(f"10. Model file: {model_path}")
    print(f"11. Scaler files: {models_dir}/feature_scaler.pkl, target_scaler.pkl")
    print(f"12. Prediction visualization: {plot_path}")
    print(f"13. Training curve: {loss_plot_path}")
    print(f"14. Exact command used: python train_trajectory_model.py --epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr} --hidden_size {args.hidden_size}")
    print("=========================================")

if __name__ == "__main__":
    main()
