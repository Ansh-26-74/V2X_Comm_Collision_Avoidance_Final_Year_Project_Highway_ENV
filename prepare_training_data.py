import argparse
import os
import sys
import csv
from collections import Counter
import pandas as pd
import numpy as np

def split_episodes(df, seed, train_frac=0.8, val_frac=0.1):
    rng = np.random.RandomState(seed)
    
    scenario_to_episodes = {}
    for ep, group in df.groupby('episode'):
        scenario = group['scenario'].iloc[0]
        if scenario not in scenario_to_episodes:
            scenario_to_episodes[scenario] = []
        scenario_to_episodes[scenario].append(ep)
        
    train_eps, val_eps, test_eps = [], [], []
    
    for scenario, eps in sorted(scenario_to_episodes.items()):
        eps_array = np.array(eps)
        rng.shuffle(eps_array)
        
        n_total = len(eps_array)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)
        
        train_eps.extend(eps_array[:n_train])
        val_eps.extend(eps_array[n_train:n_train+n_val])
        test_eps.extend(eps_array[n_train+n_val:])
        
    return set(train_eps), set(val_eps), set(test_eps)

def extract_sequences(df, episodes_set, history_len, horizon, features):
    X, y, ep_labels, scenario_labels = [], [], [], []
    
    # Sort just to be safe
    df_sorted = df.sort_values(['episode', 'time'])
    
    for ep in episodes_set:
        ep_data = df_sorted[df_sorted['episode'] == ep]
        n_rows = len(ep_data)
        
        if n_rows < history_len + horizon:
            continue
            
        feat_vals = ep_data[features].values
        target_vals = ep_data[['x', 'y']].values
        scenario = ep_data['scenario'].iloc[0]
        
        for i in range(n_rows - history_len - horizon + 1):
            X_seq = feat_vals[i : i + history_len]
            y_val = target_vals[i + history_len + horizon - 1]
            
            X.append(X_seq)
            y.append(y_val)
            ep_labels.append(ep)
            scenario_labels.append(scenario)
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(ep_labels), np.array(scenario_labels)

def save_flattened_csv(X, y, ep_labels, scenario_labels, features, output_path):
    history_len = X.shape[1]
    
    cols = ['episode', 'scenario']
    for t_step in range(history_len - 1, -1, -1):
        suffix = f"_t-{t_step}" if t_step > 0 else "_t"
        for f in features:
            cols.append(f"{f}{suffix}")
    cols.extend(['future_x', 'future_y'])
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        
        for i in range(len(X)):
            row = [ep_labels[i], scenario_labels[i]]
            row.extend(X[i].flatten().tolist())
            row.extend(y[i].tolist())
            writer.writerow(row)

def print_scenario_distribution(split_name, ep_labels, scenario_labels):
    unique_eps = {}
    for ep, sc in zip(ep_labels, scenario_labels):
        if ep not in unique_eps:
            unique_eps[ep] = sc
            
    counts = Counter(unique_eps.values())
    print(f"\n{split_name.upper()}:")
    for scenario, count in sorted(counts.items()):
        print(f"{scenario}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Prepare Training Data for Trajectory Prediction")
    parser.add_argument('--history', type=int, default=30, help='Number of historical timesteps to use as input')
    parser.add_argument('--horizon', type=int, default=15, help='Number of timesteps into the future to predict')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--input', type=str, default=os.path.join("data", "raw", "highway_trajectories.csv"), help='Input CSV path')
    parser.add_argument('--output', type=str, default=os.path.join("data", "processed"), help='Output directory')
    args = parser.parse_args()

    print(f"Loading raw dataset from: {args.input} ...")
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        sys.exit(1)
        
    df = pd.read_csv(args.input)
    
    # Calculate timestep
    # We take the difference between the second and first timestamp of episode 1
    ep1_times = df[df['episode'] == df['episode'].iloc[0]]['time'].values
    if len(ep1_times) >= 2:
        dt = ep1_times[1] - ep1_times[0]
    else:
        dt = 0.033
        
    actual_history_duration = args.history * dt
    actual_horizon_duration = args.horizon * dt

    print(f"Detected timestep: {dt:.3f} s")
    print(f"History length: {args.history} timesteps (~{actual_history_duration:.2f} s)")
    print(f"Prediction horizon: {args.horizon} timesteps (~{actual_horizon_duration:.2f} s)\n")
    
    train_eps, val_eps, test_eps = split_episodes(df, args.seed)
    
    # Overlap check
    train_val_overlap = train_eps.intersection(val_eps)
    train_test_overlap = train_eps.intersection(test_eps)
    val_test_overlap = val_eps.intersection(test_eps)
    
    print("Episode overlap check:")
    print(f"Train INTERSECT Validation: {len(train_val_overlap)}")
    print(f"Train INTERSECT Test: {len(train_test_overlap)}")
    print(f"Validation INTERSECT Test: {len(val_test_overlap)}\n")
    
    if len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0:
        print("ERROR: Episode overlap detected. Stopping.")
        sys.exit(1)

    features = ['x', 'y', 'speed', 'velocity_x', 'velocity_y', 'heading', 'lane', 'acceleration']
    
    print("Extracting sliding window sequences...")
    X_train, y_train, ep_train, sc_train = extract_sequences(df, train_eps, args.history, args.horizon, features)
    X_val, y_val, ep_val, sc_val = extract_sequences(df, val_eps, args.history, args.horizon, features)
    X_test, y_test, ep_test, sc_test = extract_sequences(df, test_eps, args.history, args.horizon, features)

    # Validation Checks
    def validate_data(X, y, name):
        if np.isnan(X).any() or np.isnan(y).any():
            print(f"ERROR: NaN values found in {name} dataset.")
            sys.exit(1)
        if np.isinf(X).any() or np.isinf(y).any():
            print(f"ERROR: Infinite values found in {name} dataset.")
            sys.exit(1)
            
    validate_data(X_train, y_train, "Train")
    validate_data(X_val, y_val, "Validation")
    validate_data(X_test, y_test, "Test")

    os.makedirs(args.output, exist_ok=True)
    
    print("Saving processed datasets to NPZ...")
    np.savez(os.path.join(args.output, "train.npz"), X=X_train, y=y_train, episodes=ep_train, scenarios=sc_train)
    np.savez(os.path.join(args.output, "validation.npz"), X=X_val, y=y_val, episodes=ep_val, scenarios=sc_val)
    np.savez(os.path.join(args.output, "test.npz"), X=X_test, y=y_test, episodes=ep_test, scenarios=sc_test)
    
    print("Saving flattened CSV formats...")
    save_flattened_csv(X_train, y_train, ep_train, sc_train, features, os.path.join(args.output, "train.csv"))
    save_flattened_csv(X_val, y_val, ep_val, sc_val, features, os.path.join(args.output, "validation.csv"))
    save_flattened_csv(X_test, y_test, ep_test, sc_test, features, os.path.join(args.output, "test.csv"))
    
    # Temporal validation sample
    if len(X_train) > 0:
        sample_idx = len(X_train) // 2
        sample_ep = ep_train[sample_idx]
        sample_scen = sc_train[sample_idx]
        ep_data_sample = df[df['episode'] == sample_ep].sort_values('time')
        # We need to find the exact times. Since we slid a window, we can infer from the row indices
        # But this is just for printing a sample check.
        # Actually, let's just find the first match of X_train[sample_idx][0]['x'] in ep_data_sample
        x_start = X_train[sample_idx][0][0]
        # Allow small float diff
        match_start = ep_data_sample.iloc[(np.abs(ep_data_sample['x'] - x_start)).argmin()]
        t_start = match_start['time']
        t_end = t_start + (args.history - 1) * dt
        t_target = t_start + (args.history + args.horizon - 1) * dt
        
        print("\nTemporal Validation Sample 1:")
        print(f"Episode: {sample_ep}")
        print(f"Scenario: {sample_scen}")
        print(f"Input time range: {t_start:.2f} -> {t_end:.2f} seconds")
        print(f"Target time: {t_target:.2f} seconds")

    total_episodes = len(df['episode'].unique())
    missing_values_raw = df.isnull().sum().sum()

    print("\n========================================")
    print("TRAINING DATA PREPARATION COMPLETE")
    print("========================================")
    print(f"Raw dataset: {args.input}")
    print(f"Total episodes: {total_episodes}")
    print(f"Train episodes: {len(train_eps)}")
    print(f"Validation episodes: {len(val_eps)}")
    print(f"Test episodes: {len(test_eps)}")
    
    print(f"\nHistory length: {args.history} timesteps")
    print(f"History duration: approximately {actual_history_duration:.2f} second")
    print(f"Prediction horizon: {args.horizon} timesteps")
    print(f"Prediction duration: approximately {actual_horizon_duration:.2f} seconds")
    
    print(f"\nInput features: {len(features)}")
    print(f"Target features: 2")
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    print(f"\nTrain X shape: {X_train.shape}")
    print(f"Train y shape: {y_train.shape}")
    print(f"Validation X shape: {X_val.shape}")
    print(f"Validation y shape: {y_val.shape}")
    print(f"Test X shape: {X_test.shape}")
    print(f"Test y shape: {y_test.shape}")
    
    print(f"\nEpisode overlap: {len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)}")
    print(f"Missing values: {missing_values_raw}")
    print("========================================\n")
    
    print("Episode distribution by scenario for each split:")
    print_scenario_distribution("TRAIN", ep_train, sc_train)
    print_scenario_distribution("VALIDATION", ep_val, sc_val)
    print_scenario_distribution("TEST", ep_test, sc_test)
    
if __name__ == "__main__":
    main()
