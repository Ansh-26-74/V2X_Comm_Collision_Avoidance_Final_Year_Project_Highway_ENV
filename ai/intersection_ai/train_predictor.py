"""Phase 13D Step 1: Synthetic Dataset Generator and Training Script for Intersection Trajectory AI.

Generates ~10,000 diverse synthetic trajectory samples across multiple urban kinematic regimes:
- Constant speed
- Acceleration
- Gradual braking
- Sudden emergency braking (C-01 hazard profile)
- Stopping and queue hold
- Stopping and restarting
- Lateral lane drift perturbations

Extracts 5-step V2V history features (0.5s at 10 Hz, 40 features total),
and trains IntersectionTrajectoryMLP to predict 6 future waypoints:
+0.25s, +0.50s, +0.75s, +1.00s, +1.25s, +1.50s (12 outputs total).
"""

import math
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from ai.intersection_ai.model import IntersectionTrajectoryMLP

# Discrete forecast offsets (in seconds) from latest observed telemetry
FORECAST_HORIZONS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]


def simulate_kinematic_trajectory(
    duration: float = 4.0,
    dt_sim: float = 0.025,  # 40 Hz internal physics simulation
    regime: str = "constant",
    rng: random.Random | None = None,
) -> list[dict]:
    """Simulate a single vehicle approach trajectory along the South approach (traveling North, -Y)."""
    if rng is None:
        rng = random.Random()

    total_steps = int(duration / dt_sim)

    # Initial coordinates on South approach (lane centered around x=592 or x=628)
    lane_center_x = rng.choice([592.0, 628.0])
    x = lane_center_x + rng.uniform(-4.0, 4.0)
    y = rng.uniform(620.0, 850.0)

    # Initial speed (urban range 40 - 105 px/s)
    speed = rng.uniform(45.0, 105.0)
    heading = -math.pi / 2.0  # Northbound (vy = -1.0)

    # Regime parameter setup
    target_speed = speed
    accel = 0.0
    hazard_code = 0.0
    restart_time = 0.0

    if regime == "constant":
        accel = 0.0
        target_speed = speed
        hazard_code = 0.0

    elif regime == "acceleration":
        speed = rng.uniform(15.0, 40.0)
        target_speed = rng.uniform(85.0, 110.0)
        accel = rng.uniform(25.0, 65.0)
        hazard_code = 0.0

    elif regime == "gradual_brake":
        target_speed = rng.uniform(15.0, 35.0)
        accel = rng.uniform(-35.0, -15.0)
        hazard_code = 0.5

    elif regime == "sudden_brake":
        # Sudden heavy deceleration like C-01 hazard
        target_speed = rng.uniform(5.0, 18.0)
        accel = rng.uniform(-120.0, -60.0)
        hazard_code = 1.0

    elif regime == "stopping":
        target_speed = 0.0
        accel = rng.uniform(-90.0, -40.0)
        hazard_code = 1.0

    elif regime == "stop_and_restart":
        target_speed = 0.0
        accel = rng.uniform(-80.0, -45.0)
        hazard_code = 1.0
        restart_time = rng.uniform(1.5, 2.2)

    elif regime == "stopped":
        speed = 0.0
        accel = 0.0
        target_speed = 0.0
        hazard_code = 1.0

    # Mild lateral drift
    lat_drift = rng.uniform(-5.0, 5.0) if regime != "stopped" else 0.0

    trajectory = []
    sim_time = 0.0

    for step in range(total_steps):
        # Stop and restart logic
        if regime == "stop_and_restart" and sim_time >= restart_time:
            target_speed = rng.uniform(70.0, 95.0)
            accel = rng.uniform(30.0, 60.0)
            hazard_code = 0.0

        # Update speed
        if accel < 0.0 and speed > target_speed:
            speed = max(target_speed, speed + accel * dt_sim)
        elif accel > 0.0 and speed < target_speed:
            speed = min(target_speed, speed + accel * dt_sim)
        elif speed == 0.0 and target_speed == 0.0:
            speed = 0.0

        vx = 0.0
        vy = -1.0

        # Observation noise added to telemetry
        noise_x = rng.gauss(0.0, 0.4)
        noise_y = rng.gauss(0.0, 0.5)
        noise_s = rng.gauss(0.0, 0.3)
        noise_h = rng.gauss(0.0, 0.01)

        trajectory.append({
            "time": sim_time,
            "x": x + noise_x,
            "y": y + noise_y,
            "true_x": x,
            "true_y": y,
            "vx": vx,
            "vy": vy,
            "speed": max(0.0, speed + noise_s),
            "true_speed": speed,
            "heading": heading + noise_h,
            "hazard_code": hazard_code,
            "accel": accel,
        })

        # Advance physical state
        y += vy * speed * dt_sim
        x += (lat_drift / total_steps)
        sim_time += dt_sim

    return trajectory


def build_dataset(target_samples: int = 10000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate diverse trajectory slices with 5-step history (10 Hz) and 6 future waypoints."""
    rng = random.Random(seed)
    regimes = [
        "constant",
        "acceleration",
        "gradual_brake",
        "sudden_brake",
        "stopping",
        "stop_and_restart",
        "stopped",
    ]
    regime_weights = [0.22, 0.16, 0.18, 0.22, 0.08, 0.08, 0.06]

    dt_sim = 0.025  # 40 Hz physics
    dt_v2v = 0.100  # 10 Hz V2V broadcast
    v2v_step_interval = int(round(dt_v2v / dt_sim))  # 4 steps per V2V packet

    # 5 history packets: indices 0, 4, 8, 12, 16 (time 0.0 to 0.4s)
    # The latest observed packet is at index 16.
    history_indices = [i * v2v_step_interval for i in range(5)]
    latest_history_idx = history_indices[-1]  # index 16 (t = 0.40s)

    # 6 future waypoints at +0.25, +0.50, +0.75, +1.00, +1.25, +1.50s from latest packet
    future_step_offsets = [int(round(h / dt_sim)) for h in FORECAST_HORIZONS]
    future_indices = [latest_history_idx + offset for offset in future_step_offsets]
    max_required_steps = max(future_indices) + 1

    X_list = []
    Y_list = []

    trajectories_needed = int(math.ceil(target_samples / 4.0)) + 500

    for _ in range(trajectories_needed):
        if len(X_list) >= target_samples:
            break

        reg = rng.choices(regimes, weights=regime_weights)[0]
        traj = simulate_kinematic_trajectory(duration=3.6, dt_sim=dt_sim, regime=reg, rng=rng)

        if len(traj) < max_required_steps:
            continue

        # Extract multiple rolling slices from one trajectory
        step_stride = v2v_step_interval * 2  # stride every 0.2s
        for start_idx in range(0, len(traj) - max_required_steps, step_stride):
            if len(X_list) >= target_samples:
                break

            hist_pts = [traj[start_idx + h_idx] for h_idx in history_indices]
            curr_pt = hist_pts[-1]
            curr_x = curr_pt["x"]
            curr_y = curr_pt["y"]

            # 40-dimensional feature vector
            feats = []
            for h in hist_pts:
                dx = h["x"] - curr_x
                dy = h["y"] - curr_y
                feats.extend([
                    dx,
                    dy,
                    h["vx"],
                    h["vy"],
                    h["speed"] / 100.0,       # Normalized scalar speed
                    h["heading"] / math.pi,   # Normalized heading
                    h["accel"] / 100.0,       # Normalized acceleration
                    h["hazard_code"],
                ])

            # 12-dimensional target vector (future waypoints relative to curr_x, curr_y)
            targets = []
            for f_idx in future_indices:
                fut_pt = traj[start_idx + f_idx]
                targets.append(fut_pt["true_x"] - curr_x)
                targets.append(fut_pt["true_y"] - curr_y)

            X_list.append(feats)
            Y_list.append(targets)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y


def train_pipeline(
    output_dir: str = "models/intersection",
    epochs: int = 40,
    seed: int = 42,
    target_samples: int = 10000,
) -> dict:
    """Train IntersectionTrajectoryMLP, save artifacts, and return training summary."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"[AI TRAINER] Generating synthetic dataset (~{target_samples} samples, seed={seed})...")
    X, Y = build_dataset(target_samples=target_samples, seed=seed)
    total_samples = len(X)
    print(f"[AI TRAINER] Generated {total_samples} samples. Input shape: {X.shape}, Target shape: {Y.shape}")

    # Standardize inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 80 / 20 Train / Validation Split
    n_val = int(total_samples * 0.20)
    indices = np.random.permutation(total_samples)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32)
    Y_val = torch.tensor(Y[val_idx], dtype=torch.float32)

    print(f"[AI TRAINER] Train set: {len(X_train)} samples | Validation set: {len(X_val)} samples")

    model = IntersectionTrajectoryMLP(input_dim=40, hidden_dim=64, output_dim=12)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    criterion = nn.MSELoss()

    batch_size = 64
    n_batches = int(math.ceil(len(X_train) / batch_size))

    best_val_loss = float("inf")
    best_state = None
    final_train_loss = 0.0

    print(f"[AI TRAINER] Commencing training for {epochs} epochs (batch_size={batch_size})...")
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_train_loss = 0.0

        for b in range(n_batches):
            b_idx = perm[b * batch_size : (b + 1) * batch_size]
            bx, by = X_train[b_idx], Y_train[b_idx]

            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / n_batches
        final_train_loss = avg_train_loss

        # Evaluate validation loss
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, Y_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train MSE: {avg_train_loss:.4f} | Val MSE: {val_loss:.4f}")

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save exclusively to models/intersection/ (highway models untouched)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "trajectory_model.pth")
    scaler_path = os.path.join(output_dir, "feature_scaler.pkl")

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[AI TRAINER] Model successfully saved to: {model_path}")
    print(f"[AI TRAINER] Scaler successfully saved to: {scaler_path}")

    return {
        "dataset_size": total_samples,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "final_train_loss": final_train_loss,
        "best_val_loss": best_val_loss,
        "model_path": model_path,
        "scaler_path": scaler_path,
    }


def verify_saved_model(output_dir: str = "models/intersection"):
    """Load the trained model and scaler, run sample inferences, and verify waypoint dimensions."""
    print("\n[VERIFICATION] Verifying saved model and feature scaler...")

    model_path = os.path.join(output_dir, "trajectory_model.pth")
    scaler_path = os.path.join(output_dir, "feature_scaler.pkl")

    assert os.path.exists(model_path), f"Missing model checkpoint: {model_path}"
    assert os.path.exists(scaler_path), f"Missing feature scaler: {scaler_path}"

    with open(scaler_path, "rb") as f:
        scaler: StandardScaler = pickle.load(f)

    model = IntersectionTrajectoryMLP(input_dim=40, hidden_dim=64, output_dim=12)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()

    print("[VERIFICATION] Model checkpoint and scaler loaded successfully.")

    # Test Sample 1: Constant cruising speed (80 px/s, South moving North: dy ~ -8 px per 0.1s)
    feats_cruising = []
    for step in range(5):
        dt_back = (4 - step) * 0.1
        feats_cruising.extend([
            0.0,              # dx from current
            80.0 * dt_back,   # dy from current (behind current y)
            0.0,              # vx
            -1.0,             # vy
            0.80,             # normalized speed (80 px/s)
            -0.5,             # normalized heading (-pi/2 / pi)
            0.0,              # accel
            0.0,              # hazard code
        ])

    x_cruising = scaler.transform([feats_cruising])
    with torch.no_grad():
        pred_cruising = model(torch.tensor(x_cruising, dtype=torch.float32)).numpy()[0]

    waypoints_cruising = [(pred_cruising[2*i], pred_cruising[2*i + 1]) for i in range(6)]
    print(f"\nSample 1: Cruising @ 80 px/s (Northbound)")
    for i, (dx, dy) in enumerate(waypoints_cruising):
        t_horiz = FORECAST_HORIZONS[i]
        print(f"  Waypoint #{i+1} (+{t_horiz:.2f}s): dx = {dx:+.2f} px, dy = {dy:+.2f} px (expected dy ~ {-80*t_horiz:.1f} px)")

    assert len(waypoints_cruising) == 6, "Expected exactly 6 waypoints"
    # Vehicle is moving North, dy should be negative and decreasing over time
    assert waypoints_cruising[0][1] < 0.0, "dy must be negative (Northbound)"
    assert waypoints_cruising[-1][1] < waypoints_cruising[0][1], "Trajectory must progress Northward over horizon"

    # Test Sample 2: Sudden Deceleration / Emergency Braking (speed drops from 80 to 20 px/s)
    feats_braking = []
    speeds = [80.0, 65.0, 50.0, 35.0, 20.0]
    for step, spd in enumerate(speeds):
        dt_back = (4 - step) * 0.1
        feats_braking.extend([
            0.0,
            spd * dt_back,
            0.0,
            -1.0,
            spd / 100.0,
            -0.5,
            -0.80,   # Heavy deceleration
            1.0,    # Hazard flag
        ])

    x_braking = scaler.transform([feats_braking])
    with torch.no_grad():
        pred_braking = model(torch.tensor(x_braking, dtype=torch.float32)).numpy()[0]

    waypoints_braking = [(pred_braking[2*i], pred_braking[2*i + 1]) for i in range(6)]
    print(f"\nSample 2: Sudden Braking (80 -> 20 px/s, Hazard active)")
    for i, (dx, dy) in enumerate(waypoints_braking):
        t_horiz = FORECAST_HORIZONS[i]
        print(f"  Waypoint #{i+1} (+{t_horiz:.2f}s): dx = {dx:+.2f} px, dy = {dy:+.2f} px")

    # The braking vehicle covers significantly less distance in 1.5s than the cruising vehicle
    print(f"\nComparison at 1.50s Horizon:")
    print(f"  Cruising 1.5s displacement: {waypoints_cruising[-1][1]:.2f} px")
    print(f"  Braking 1.5s displacement:  {waypoints_braking[-1][1]:.2f} px (travels shorter distance due to deceleration)")
    assert abs(waypoints_braking[-1][1]) < abs(waypoints_cruising[-1][1]), "Braking vehicle must cover less distance than cruising"

    print("\n[VERIFICATION] All inference and waypoint structure checks PASSED.")


if __name__ == "__main__":
    summary = train_pipeline(output_dir="models/intersection", epochs=40, seed=42, target_samples=10000)
    verify_saved_model(output_dir="models/intersection")
