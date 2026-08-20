import argparse
import logging
import math
import os
import csv
import sys
import numpy as np
import gymnasium as gym

# Import necessary configuration from the main script
try:
    from main import (
        LANES, A_LANE, A_POSITION, A_SPEED,
        B_LANE, B_POSITION, B_SPEED,
        C_LANE, C_POSITION, C_SPEED,
        A_TARGET_LANE, C_TARGET_LANE,
        C_CHANGE_DISTANCE, A_CHANGE_DISTANCE,
        steering_to_lane, BlockingTruck
    )
except ImportError:
    print("Error: Could not import configuration from main.py. Ensure you run this script from the project root.")
    sys.exit(1)

from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

SCENARIOS = [
    "constant_speed",
    "acceleration",
    "deceleration",
    "speed_variation",
    "lane_change_early",
    "lane_change_late",
    "lane_change_with_acceleration",
    "lane_change_with_deceleration"
]

def get_scenario_config(scenario_name, np_random):
    """Returns initial_speed, initial_pos_offset, target_speed_func, lane_change_distance."""
    init_pos_offset = np_random.uniform(-15.0, 15.0)
    
    if scenario_name == "constant_speed":
        init_speed = np_random.uniform(10.0, 14.0)
        target_speed_func = lambda t: init_speed
        change_distance = C_CHANGE_DISTANCE
        
    elif scenario_name == "acceleration":
        init_speed = np_random.uniform(8.0, 10.0)
        target_speed_val = np_random.uniform(12.0, 14.0)
        target_speed_func = lambda t: min(target_speed_val, init_speed + 1.2 * t)
        change_distance = C_CHANGE_DISTANCE
        
    elif scenario_name == "deceleration":
        init_speed = np_random.uniform(13.0, 15.0)
        target_speed_val = np_random.uniform(8.0, 11.0)
        target_speed_func = lambda t: max(target_speed_val, init_speed - 1.2 * t)
        change_distance = C_CHANGE_DISTANCE
        
    elif scenario_name == "speed_variation":
        init_speed = np_random.uniform(10.0, 12.0)
        amplitude = np_random.uniform(1.0, 2.0)
        freq = np_random.uniform(0.5, 1.5)
        target_speed_func = lambda t: init_speed + amplitude * math.sin(freq * t)
        change_distance = C_CHANGE_DISTANCE
        
    elif scenario_name == "lane_change_early":
        init_speed = np_random.uniform(10.0, 14.0)
        target_speed_func = lambda t: init_speed
        change_distance = np_random.uniform(60.0, 80.0)
        
    elif scenario_name == "lane_change_late":
        init_speed = np_random.uniform(10.0, 14.0)
        target_speed_func = lambda t: init_speed
        change_distance = np_random.uniform(15.0, 25.0)
        
    elif scenario_name == "lane_change_with_acceleration":
        init_speed = np_random.uniform(8.0, 10.0)
        target_speed_val = np_random.uniform(12.0, 14.0)
        target_speed_func = lambda t: min(target_speed_val, init_speed + 1.2 * t)
        change_distance = np_random.uniform(35.0, 50.0)
        
    elif scenario_name == "lane_change_with_deceleration":
        init_speed = np_random.uniform(13.0, 15.0)
        target_speed_val = np_random.uniform(8.0, 11.0)
        target_speed_func = lambda t: max(target_speed_val, init_speed - 1.2 * t)
        change_distance = np_random.uniform(35.0, 50.0)

    else:
        # Fallback
        init_speed = 12.0
        target_speed_func = lambda t: 12.0
        change_distance = 45.0

    return init_speed, init_pos_offset, target_speed_func, change_distance


def create_randomized_scene(env, init_speed, pos_offset, np_random):
    """Replicates the main scene but injects specific speed and position offsets."""
    network = RoadNetwork.straight_road_network(lanes=LANES, speed_limit=40)
    road = Road(network=network, np_random=np_random)

    a_lane = road.network.get_lane(("0", "1", A_LANE))
    c_lane = road.network.get_lane(("0", "1", C_LANE))
    
    a_speed_var = np_random.uniform(-1.0, 1.0)
    
    a = Vehicle(road, a_lane.position(A_POSITION, 0),
                heading=a_lane.heading_at(A_POSITION), speed=A_SPEED + a_speed_var)
    b = BlockingTruck.make_on_lane(road, ("0", "1", B_LANE), B_POSITION, speed=B_SPEED)
    c = Vehicle(road, c_lane.position(C_POSITION + pos_offset, 0), heading=math.pi, speed=init_speed)

    road.vehicles = [a, b, c]
    env.unwrapped.road = road
    env.unwrapped.vehicle = a
    env.unwrapped.controlled_vehicles = [a]
    return a, b, c


def save_visualization(dataset_rows):
    """Optionally generate a simple preview of the trajectories."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not found. Skipping visualization.")
        return
        
    output_dir = os.path.join("data", "raw")
    plot_file = os.path.join(output_dir, "trajectory_preview.png")
    
    plt.figure(figsize=(10, 6))
    
    # Plot first few episodes of each scenario type to avoid clutter
    plotted_per_scenario = {s: 0 for s in SCENARIOS}
    episodes = set(row['episode'] for row in dataset_rows)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(SCENARIOS)))
    scenario_colors = {s: c for s, c in zip(SCENARIOS, colors)}
    
    for ep in list(episodes)[:100]:  # Look through first 100 episodes
        ep_data = [r for r in dataset_rows if r['episode'] == ep]
        if not ep_data:
            continue
        scenario = ep_data[0]['scenario']
        if plotted_per_scenario[scenario] < 2:  # Plot 2 lines per scenario
            xs = [r['x'] for r in ep_data]
            ys = [r['y'] for r in ep_data]
            label = scenario if plotted_per_scenario[scenario] == 0 else None
            plt.plot(xs, ys, color=scenario_colors[scenario], label=label, alpha=0.7)
            plotted_per_scenario[scenario] += 1
            
    plt.title("Sample Vehicle C Trajectories by Scenario")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Saved visualization to: {plot_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate Vehicle C Trajectory Dataset")
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to generate')
    parser.add_argument('--scenario', type=str, default="all", help='Specific scenario to generate (or "all")')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    print("Generating trajectory dataset...\n")

    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "highway_trajectories.csv")

    env = gym.make(
        "highway-v0",
        config={
            "lanes_count": LANES,
            "vehicles_count": 0,
            "controlled_vehicles": 1,
            "duration": 15,
            "simulation_frequency": 30,
            "policy_frequency": 30,
            "real_time_rendering": False,
            "action": {"type": "ContinuousAction", "longitudinal": True,
                       "lateral": True, "dynamical": False, "clip": True},
        },
    )

    columns = [
        "episode", "time", "vehicle_id", "x", "y", 
        "speed", "velocity_x", "velocity_y", 
        "heading", "lane", "acceleration", "scenario"
    ]
    
    dataset_rows = []
    successful = 0
    failed = 0
    scenario_counts = {s: 0 for s in SCENARIOS}
    episode_lengths = []

    dt = 1 / 30

    for ep in range(1, args.episodes + 1):
        try:
            if args.scenario != "all":
                active_scenario = args.scenario
            else:
                active_scenario = SCENARIOS[(ep - 1) % len(SCENARIOS)]

            ep_seed = args.seed + ep if args.seed is not None else None
            env.reset(seed=ep_seed)
            np_random = env.unwrapped.np_random
            
            init_speed, pos_offset, target_speed_func, change_distance = get_scenario_config(active_scenario, np_random)
            
            a, b, c = create_randomized_scene(env, init_speed, pos_offset, np_random)
            
            c_target_y = env.unwrapped.road.network.get_lane(("0", "1", C_TARGET_LANE)).position(0, 0)[1]
            a_target_y = env.unwrapped.road.network.get_lane(("0", "1", A_TARGET_LANE)).position(0, 0)[1]
            
            c_is_changing_lane = False
            a_is_changing_lane = False
            
            time_s = 0.0
            prev_speed = None
            ep_rows = 0
            
            terminated = truncated = False
            while not (terminated or truncated):
                # Extract actual state
                x = c.position[0]
                y = c.position[1]
                speed = c.speed
                heading = c.heading
                vel_x = speed * math.cos(heading)
                vel_y = speed * math.sin(heading)
                lane = int(c.lane_index[2]) if c.lane_index else C_LANE
                
                # Acceleration is physically derived, not leaked from target
                if prev_speed is not None:
                    accel = (speed - prev_speed) / dt
                else:
                    accel = 0.0
                prev_speed = speed
                
                dataset_rows.append({
                    "episode": ep,
                    "time": round(time_s, 3),
                    "vehicle_id": "C",
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "speed": round(speed, 3),
                    "velocity_x": round(vel_x, 3),
                    "velocity_y": round(vel_y, 3),
                    "heading": round(heading, 3),
                    "lane": lane,
                    "acceleration": round(accel, 3),
                    "scenario": active_scenario
                })
                ep_rows += 1
                
                # Perform simulation step with AI behavior injected
                target_speed = target_speed_func(time_s)
                c_accel_cmd = np.clip(1.5 * (target_speed - speed), -3.0, 3.0)
                
                c_to_truck = float(c.position[0] - b.position[0])
                if 0 < c_to_truck <= change_distance:
                    c_is_changing_lane = True
                
                a_to_truck = float(b.position[0] - a.position[0])
                if 0 < a_to_truck <= A_CHANGE_DISTANCE:
                    a_is_changing_lane = True
                
                c_steering = steering_to_lane(c, c_target_y, math.pi) if c_is_changing_lane else 0.0
                a_steering = steering_to_lane(a, a_target_y, 0.0) if a_is_changing_lane else 0.0
                
                c.act({"acceleration": c_accel_cmd, "steering": c_steering * math.pi / 4})
                
                # Advance simulation
                action = np.array([0.0, a_steering], dtype=np.float32)
                _, _, terminated, truncated, _ = env.step(action)
                
                time_s += dt
                
                # End episode early if collision occurs
                if getattr(c, 'crashed', False) or getattr(a, 'crashed', False):
                    break

            successful += 1
            scenario_counts[active_scenario] = scenario_counts.get(active_scenario, 0) + 1
            episode_lengths.append(ep_rows)
            
            # Reduce console spam for large datasets
            if ep % max(1, args.episodes // 20) == 0 or ep == args.episodes:
                print(f"Episode {ep}/{args.episodes} [OK]")
            
        except Exception as e:
            failed += 1
            print(f"Episode {ep}/{args.episodes} [FAILED] (Error: {e})")
            continue

    env.close()

    # Write CSV
    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(dataset_rows)

    # Print Validation & Statistics
    total_rows = len(dataset_rows)
    avg_rows = total_rows / max(successful, 1)
    min_length = min(episode_lengths) if episode_lengths else 0
    max_length = max(episode_lengths) if episode_lengths else 0
    missing_values = sum(1 for row in dataset_rows for val in row.values() if val is None)
    unique_ids = list(set(row["vehicle_id"] for row in dataset_rows))
    
    print("\nDataset generation complete.\n")
    print("Episodes:")
    print(f"{args.episodes}")
    print("\nSuccessful:")
    print(f"{successful}")
    print("\nFailed:")
    print(f"{failed}")
    print("\nTotal rows:")
    print(f"~{total_rows}")
    print("\nAverage rows/episode:")
    print(f"~{int(avg_rows)}")
    print(f"Min length: {min_length}, Max length: {max_length}")
    print("\nMissing values:")
    print(f"{missing_values}")
    print(f"Unique vehicle IDs: {unique_ids}")
    
    print("\nScenario distribution:")
    for sc, count in scenario_counts.items():
        if count > 0:
            print(f"{sc}: {count}")
            
    print(f"\nSaved to:\n{output_file}")
    
    save_visualization(dataset_rows)


if __name__ == "__main__":
    main()
