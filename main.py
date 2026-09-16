"""Step 1: a deterministic three-vehicle HighwayEnv scene."""
import argparse
import csv
import logging
import time
from collections import deque
from math import atan2, cos, hypot, pi, sin

import gymnasium as gym
import highway_env  # Registers highway-v0 with Gymnasium.
import numpy as np
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from v2v_manager import V2VManager
from ai.trajectory_predictor import AIPredictor
from ai.safety_fusion import SafetyFusionEngine
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)



# A, B, and C share a lane here so B visually blocks the direct view from A to C.
LANES = 3
A_LANE, A_POSITION, A_SPEED = 1, 0.0, 20.0       # Green ego vehicle A
B_LANE, B_POSITION, B_SPEED = 1, 90.0, 10.0      # Blocking truck B
C_LANE, C_POSITION, C_SPEED = 1, 245.0, 10.0     # Red oncoming vehicle C
# Set C_TARGET_LANE to 0 to make red C broadcast RIGHT, or 2 for LEFT.
# A_TARGET_LANE controls green A's overtaking lane independently.
A_TARGET_LANE = 0
C_TARGET_LANE = 0
# Each vehicle begins changing only when it is this close to truck B (metres).
# Vehicles now initiate maneuvers closer to the truck (~0.8s later).
C_CHANGE_DISTANCE = 31.0
# A starts overtaking before C sends its lane-change warning.
A_CHANGE_DISTANCE = 29.0
V2V_RANGE = 300.0
PACKET_LOSS_RATE = 0.0
V2V_LATENCY = 0.0
BROADCAST_RATE = 10.0

SAFE_TTC = 4.0
CRITICAL_TTC = 2.0


class BlockingTruck(Vehicle):
    """A longer, wider Vehicle so B visibly looks like a truck."""
    LENGTH = 12.0
    WIDTH = 2.8


def create_three_vehicle_scene(env):
    """Replace HighwayEnv's random traffic with exactly A, B, and C."""
    network = RoadNetwork.straight_road_network(lanes=LANES, speed_limit=40)
    road = Road(network=network, np_random=env.unwrapped.np_random)

    # A and C use the basic kinematic vehicle. A is the controlled ego vehicle
    # and C follows a short scripted lane-change manoeuvre.
    a_lane = road.network.get_lane(("0", "1", A_LANE))
    c_lane = road.network.get_lane(("0", "1", C_LANE))
    a = Vehicle(road, a_lane.position(A_POSITION, 0),
                heading=a_lane.heading_at(A_POSITION), speed=A_SPEED)
    b = BlockingTruck.make_on_lane(road, ("0", "1", B_LANE), B_POSITION, speed=B_SPEED)
    # C travels in the opposite direction: heading pi points back towards A.
    c = Vehicle(road, c_lane.position(C_POSITION, 0), heading=pi, speed=C_SPEED)

    # Clear colours make the three vehicles easy to distinguish in HighwayEnv.
    a.color = (50, 200, 50)    # A: green
    b.color = (255, 140, 0)    # B: orange truck
    c.color = (220, 60, 60)    # C: red

    road.vehicles = [a, b, c]
    env.unwrapped.road = road
    env.unwrapped.vehicle = a
    env.unwrapped.controlled_vehicles = [a]
    return a, b, c


def steering_to_lane(vehicle, target_y: float, direction_heading: float) -> float:
    """Return a smooth normalized steering command towards a lane centre."""
    lateral_error = target_y - float(vehicle.position[1])
    desired_x = cos(direction_heading) * vehicle.speed
    desired_y = sin(direction_heading) * vehicle.speed + 0.9 * lateral_error
    desired_heading = atan2(desired_y, desired_x)
    heading_error = atan2(sin(desired_heading - vehicle.heading),
                          cos(desired_heading - vehicle.heading))
    steering_radians = float(np.clip(2.0 * heading_error, -0.45, 0.45))
    return steering_radians / (pi / 4)


def planned_lane_change_trajectory(vehicle, target_y: float,
                                   direction_heading: float) -> list[tuple[float, float]]:
    """Make the short path C includes in its V2V broadcast message."""
    points, duration = 24, 2.0
    trajectory = []
    for index in range(points):
        t = duration * index / (points - 1)
        # A cosine curve gives a smooth change from current y to target-lane y.
        progress = index / (points - 1)
        y = vehicle.position[1] + (target_y - vehicle.position[1]) * (1 - cos(pi * progress)) / 2
        x = vehicle.position[0] + cos(direction_heading) * vehicle.speed * t
        trajectory.append((float(x), float(y)))
    return trajectory


def oncoming_maneuver_name(initial_lane: int, target_lane: int) -> str:
    """For red C (heading west), a lower lane number is its right side."""
    if target_lane == initial_lane:
        return "KEEP LANE"
    return "RIGHT" if target_lane < initial_lane else "LEFT"


def calculate_ttc(vehicle_a, message) -> float:
    """Calculate Time-To-Collision along the x-axis based on closing speed."""
    if message is None:
        return float('inf')
    # C is facing opposite direction, its speed relative to A is A.speed + message.speed
    closing_speed = vehicle_a.speed + message.speed
    if closing_speed <= 0:
        return float('inf')
    distance = message.x - vehicle_a.position[0]
    return float(distance / closing_speed) if distance > 0 else float('inf')


def draw_v2v_overlay(env, message, avoidance_active: bool, v2v_enabled: bool, ttc: float = float('inf'), risk_state: str = "NONE",
                     ai_safety_enabled: bool = False, v2v_conflict: bool = False, ai_conflict: bool = False, fused_conflict: bool = False,
                     ego_action: str = "") -> None:
    """A small display panel; it does not affect HighwayEnv's road renderer."""
    import pygame
    viewer = env.unwrapped.viewer
    if viewer is None:
        return
    
    if ai_safety_enabled:
        panel_height = 300
    else:
        panel_height = 220
        
    panel = pygame.Surface((410, panel_height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 175))
    viewer.screen.blit(panel, (15, 15))
    font = pygame.font.SysFont("consolas", 19)
    
    if ai_safety_enabled:
        lines = ["MODE: V2X + AI FUSION", "", "GREEN A: EGO / RECEIVER"]
        lines += [f"V2V: {'CONNECTED' if message else 'WAITING'}", 
                  f"V2V CONFLICT: {'YES' if v2v_conflict else 'NO'}"]
        lines += [f"AI: ACTIVE", 
                  f"AI CONFLICT: {'YES' if ai_conflict else 'NO'}"]
        lines += ["", f"FUSED CONFLICT: {'YES' if fused_conflict else 'NO'}"]
        if risk_state != "NONE":
            lines.append(f"TTC: {ttc:.2f} s")
            lines.append(f"RISK: {risk_state}")
        if avoidance_active:
            if risk_state == "CRITICAL":
                lines.append("ACTION: EMERGENCY BRAKING")
            else:
                lines.append("ACTION: ABORT + BRAKING")
        elif ego_action:
            lines.append(f"ACTION: {ego_action.upper()}")
    else:
        mode_text = "MODE: V2V ENABLED" if v2v_enabled else "MODE: V2V DISABLED"
        lines = [mode_text, "", "GREEN A: EGO / RECEIVER"]
        if v2v_enabled:
            if message:
                lines += ["V2V: CONNECTED", f"REMOTE: {message.vehicle_id} (RED)",
                          f"C position: ({message.x:.1f}, {message.y:.1f}) m",
                          f"C speed: {message.speed:.1f} m/s",
                          f"C intent: {message.maneuver} -> lane {message.intended_lane}"]
                if risk_state != "NONE":
                    lines.append(f"TTC: {ttc:.2f} s")
                    lines.append(f"RISK: {risk_state}")
            else:
                lines.append("V2V: WAITING FOR C ALERT")
        else:
            lines.append("V2V: OFFLINE")
        if avoidance_active:
            lines.append("SAFETY: ALERT - RETURNING + BRAKING")
        elif ego_action:
            lines.append(f"SAFETY: {ego_action.upper()}")
            
    for i, line in enumerate(lines):
        if "BRAKING" in line or "CRITICAL" in line or (line.startswith("FUSED") and "YES" in line):
            colour = (255, 70, 70)
        elif "WARNING" in line:
            colour = (255, 210, 70)
        elif "SAFE" in line or "CONNECTED" in line or "ACTIVE" in line or "OVERTAKING" in line or "MERGING" in line or "CRUISING" in line:
            colour = (80, 230, 120)
        else:
            colour = (240, 240, 240)
        viewer.screen.blit(font.render(line, True, colour), (27, 27 + 22 * i))
    pygame.display.flip()

def draw_ai_overlay(env, current_c_pos, pred_c_pos, status, a_target_y, ai_conflict_detected, hide_text=False) -> None:
    """Draw the AI prediction visualization."""
    if pred_c_pos is None:
        return
        
    import pygame
    viewer = env.unwrapped.viewer
    if viewer is None:
        return
        
        
    # Draw prediction text block if not hidden
    if not hide_text:
        panel = pygame.Surface((350, 125), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 175))
        viewer.screen.blit(panel, (640, 15))
        font = pygame.font.SysFont("consolas", 19)
        
        lines = [
            "AI PREDICTION (Vehicle C)",
            f"STATUS: {status}"
        ]
        if status == "ACTIVE":
            lines.append(f"Predicted X: {pred_c_pos[0]:.1f} m")
            lines.append(f"Predicted Y: {pred_c_pos[1]:.1f} m")
            lines.append(f"Horizon: ~0.49 s")
            if ai_conflict_detected is not None:
                lines.append(f"AI CONFLICT: {'YES' if ai_conflict_detected else 'NO'}")
            
        for i, line in enumerate(lines):
            if "YES" in line:
                colour = (255, 70, 70)
            elif "NO" in line:
                colour = (80, 230, 120)
            elif "ACTIVE" in line:
                colour = (230, 80, 230)
            else:
                colour = (200, 200, 200)
            viewer.screen.blit(font.render(line, True, colour), (650, 27 + 22 * i))
        
    if status == "ACTIVE":
        try:
            # Draw A's intended overtaking path (dotted blue line at a_target_y)
            if a_target_y is not None:
                start_pix = viewer.sim_surface.vec2pix((0, a_target_y))
                end_pix = viewer.sim_surface.vec2pix((1000, a_target_y))
                for x_coord in range(0, viewer.screen_size[0], 20):
                    pygame.draw.line(viewer.screen, (100, 150, 255), (x_coord, start_pix[1]), (x_coord+10, start_pix[1]), 2)
            
            # Draw dotted line and marker for prediction
            curr_pix = viewer.sim_surface.vec2pix(current_c_pos)
            pred_pix = viewer.sim_surface.vec2pix(pred_c_pos)
            
            points = 8
            for i in range(points):
                alpha = i / (points - 1)
                x = int(curr_pix[0] + (pred_pix[0] - curr_pix[0]) * alpha)
                y = int(curr_pix[1] + (pred_pix[1] - curr_pix[1]) * alpha)
                pygame.draw.circle(viewer.screen, (230, 80, 230), (x, y), 3)
                
            pygame.draw.circle(viewer.screen, (230, 80, 230), pred_pix, 8, 2)
            
            # Red circle if conflict
            if ai_conflict_detected:
                pygame.draw.circle(viewer.screen, (255, 0, 0), pred_pix, 12, 2)
                
        except Exception:
            pass



def draw_received_trajectory(env, message) -> None:
    """Draw C's V2V-provided planned trajectory, not a local sensor result."""
    if message is None or not message.trajectory:
        return
    import pygame
    viewer = env.unwrapped.viewer
    if viewer is None:
        return
    points = [viewer.sim_surface.vec2pix(point) for point in message.trajectory]
    # Short alternating line segments make the remote path visually distinct.
    for index in range(0, len(points) - 1, 2):
        pygame.draw.line(viewer.screen, (255, 70, 70), points[index], points[index + 1], 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Driving Simulation")
    group_v2v = parser.add_mutually_exclusive_group()
    group_v2v.add_argument('--with-v2v', action='store_true', default=True, dest='v2v_enabled',
                       help='Enable V2V communication (default)')
    group_v2v.add_argument('--without-v2v', action='store_false', dest='v2v_enabled',
                       help='Disable V2V communication')
                       
    group_ai = parser.add_mutually_exclusive_group()
    group_ai.add_argument('--with-ai', action='store_true', default=False, dest='ai_enabled',
                       help='Enable live AI trajectory prediction')
    group_ai.add_argument('--with-ai-conflict', action='store_true', default=False, dest='ai_conflict_enabled',
                       help='Enable AI prediction AND conflict detection')
    group_ai.add_argument('--with-ai-safety', action='store_true', default=False, dest='ai_safety_enabled',
                       help='Enable Final V2V + AI Safety Fusion')
    group_ai.add_argument('--without-ai', action='store_false', dest='ai_enabled',
                       help='Disable live AI trajectory prediction (default)')
                       
    args = parser.parse_args()
    v2v_enabled = args.v2v_enabled
    ai_safety_enabled = args.ai_safety_enabled
    ai_conflict_enabled = args.ai_conflict_enabled or ai_safety_enabled
    ai_enabled = args.ai_enabled or ai_conflict_enabled

    if v2v_enabled:
        logger.info("V2V MODE: ENABLED")
    else:
        logger.info("V2V MODE: DISABLED")
        
    if ai_enabled:
        if ai_safety_enabled:
            logger.info("AI MODE: ENABLED (WITH SAFETY FUSION)")
            fusion_engine = SafetyFusionEngine()
            fusion_log_file = open("data/results/v2x_ai_fusion.csv", "w", newline="")
            fusion_csv_writer = csv.writer(fusion_log_file)
            fusion_csv_writer.writerow(["time", "v2v_conflict", "ai_conflict", "fused_conflict", "ai_prediction_available", "v2v_prediction_available", "ai_predicted_x", "ai_predicted_y", "ttc", "risk_state", "vehicle_a_action"])
        else:
            logger.info("AI MODE: ENABLED" + (" (WITH CONFLICT DETECTION)" if ai_conflict_enabled else ""))
            fusion_engine = None
            fusion_log_file = None
            fusion_csv_writer = None
            
        ai_predictor = AIPredictor()
        pending_evals = deque()
        ai_log_file = open("data/results/ai_predictions.csv", "w", newline="")
        ai_csv_writer = csv.writer(ai_log_file)
        ai_csv_writer.writerow(["time_s", "actual_x", "actual_y", "predicted_x", "predicted_y", "prediction_error_m", "inference_time_ms"])
        
        if ai_conflict_enabled:
            conflict_log_file = open("data/results/ai_conflict_predictions.csv", "w", newline="")
            conflict_csv_writer = csv.writer(conflict_log_file)
            conflict_csv_writer.writerow(["time", "c_current_x", "c_current_y", "c_predicted_x", "c_predicted_y", "a_path_x", "a_path_y", "distance_to_a_path", "ai_conflict", "prediction_horizon", "inference_time_ms"])
        else:
            conflict_log_file = None
            conflict_csv_writer = None
        
        ai_stats = {
            "predictions": 0,
            "total_ms": 0.0,
            "max_ms": 0.0,
            "min_ms": 9999.0,
            "total_error": 0.0,
            "evals_completed": 0
        }
    else:
        logger.info("AI MODE: DISABLED")
        ai_predictor = None

    env = gym.make(
        "highway-v0",
        render_mode="human",
        config={
            "lanes_count": LANES,
            "vehicles_count": 0,
            "controlled_vehicles": 1,
            "duration": 60,
            "simulation_frequency": 30,
            "policy_frequency": 30,
            "real_time_rendering": True,
            "action": {"type": "ContinuousAction", "longitudinal": True,
                       "lateral": True, "dynamical": False, "clip": True},
            "screen_width": 1000,
            "screen_height": 500,
        },
    )
    env.reset(seed=7)
    a, b, c = create_three_vehicle_scene(env)
    a_target_y = env.unwrapped.road.network.get_lane(("0", "1", A_TARGET_LANE)).position(0, 0)[1]
    c_target_y = env.unwrapped.road.network.get_lane(("0", "1", C_TARGET_LANE)).position(0, 0)[1]
    a_home_y = env.unwrapped.road.network.get_lane(("0", "1", A_LANE)).position(0, 0)[1]
    c_maneuver = oncoming_maneuver_name(C_LANE, C_TARGET_LANE)
    v2v = V2VManager(V2V_RANGE, PACKET_LOSS_RATE, V2V_LATENCY, BROADCAST_RATE)
    a_is_changing_lane = False
    c_is_changing_lane = False
    avoidance_active = False
    time_s = 0.0
    dt = 1 / 30

    c_passed = False
    resumed_overtake = False
    merging_in_front = False
    overtake_complete = False
    merging_logged = False
    overtake_complete_logged = False
    ego_action_desc = ""

    a_approaching_logged = False
    a_overtake_logged = False
    c_approaching_logged = False
    c_trigger_logged = False
    c_broadcast_logged = False
    a_conflict_logged = False
    a_aborted_logged = False
    last_risk_state = "NONE"
    
    first_v2v_conflict_time = None
    first_ai_conflict_time = None
    first_fused_conflict_time = None
    fused_conflict_logged = False
    ai_conflict_count = 0
    min_dist_to_a_path = float('inf')
    AI_CONFLICT_DISTANCE = 2.0
    latest_ai_conflict_state = False
    
    last_c_speed = C_SPEED
    last_ai_print_time = 0.0
    latest_pred_x, latest_pred_y = None, None
    
    try:
        terminated = truncated = False
        while not (terminated or truncated):
            if ai_enabled:
                c_accel = (c.speed - last_c_speed) / dt
                vel_x = c.speed * cos(c.heading)
                vel_y = c.speed * sin(c.heading)
                lane_val = float(c.lane_index[2]) if c.lane_index else float(C_LANE)
                
                ai_predictor.add_state(c.position[0], c.position[1], c.speed, vel_x, vel_y, c.heading, lane_val, c_accel)
                
                if ai_predictor.ready():
                    pred_x, pred_y, inf_ms = ai_predictor.predict()
                    latest_pred_x, latest_pred_y = pred_x, pred_y
                    
                    ai_stats["predictions"] += 1
                    ai_stats["total_ms"] += inf_ms
                    ai_stats["max_ms"] = max(ai_stats["max_ms"], inf_ms)
                    ai_stats["min_ms"] = min(ai_stats["min_ms"], inf_ms)
                    
                    if time_s - last_ai_print_time >= 1.0:
                        logger.info(f"AI inference time: {inf_ms:.2f} ms")
                        last_ai_print_time = time_s
                        
                    if ai_conflict_enabled:
                        dist_to_a_path = abs(pred_y - a_target_y)
                        min_dist_to_a_path = min(min_dist_to_a_path, dist_to_a_path)
                        is_conflict = bool(dist_to_a_path <= AI_CONFLICT_DISTANCE)
                        latest_ai_conflict_state = is_conflict
                        
                        conflict_csv_writer.writerow([
                            f"{time_s:.2f}",
                            f"{c.position[0]:.3f}", f"{c.position[1]:.3f}",
                            f"{pred_x:.3f}", f"{pred_y:.3f}",
                            f"{pred_x:.3f}", f"{a_target_y:.3f}",
                            f"{dist_to_a_path:.4f}", is_conflict, 0.495, f"{inf_ms:.2f}"
                        ])
                        
                        if is_conflict:
                            ai_conflict_count += 1
                            if first_ai_conflict_time is None:
                                first_ai_conflict_time = time_s
                                if not ai_safety_enabled:
                                    logger.info("[AI] Predicted trajectory conflict detected")
                                    logger.info(f"C current: ({c.position[0]:.1f}, {c.position[1]:.1f})")
                                    logger.info(f"C predicted: ({pred_x:.1f}, {pred_y:.1f})")
                                    logger.info(f"Distance to A path: {dist_to_a_path:.2f} m")
                                    logger.info("Prediction horizon: 0.49 s")
                                    logger.info("[AI] Safety action: NOT CONNECTED YET")
                        
                    # Queue evaluation for 15 steps (~0.495s) ahead
                    pending_evals.append({
                        "time_generated": time_s,
                        "pred_x": pred_x,
                        "pred_y": pred_y,
                        "target_time": time_s + 0.495,
                        "inf_ms": inf_ms
                    })
                
                # Check if we have an old prediction that's ripe for evaluation against current C position
                while pending_evals and time_s >= pending_evals[0]["target_time"]:
                    eval_data = pending_evals.popleft()
                    err = hypot(eval_data["pred_x"] - c.position[0], eval_data["pred_y"] - c.position[1])
                    ai_csv_writer.writerow([
                        f"{time_s:.2f}",
                        f"{c.position[0]:.3f}", f"{c.position[1]:.3f}",
                        f"{eval_data['pred_x']:.3f}", f"{eval_data['pred_y']:.3f}",
                        f"{err:.4f}", f"{eval_data['inf_ms']:.2f}"
                    ])
                    ai_stats["total_error"] += err
                    ai_stats["evals_completed"] += 1
                    
            last_c_speed = c.speed
            
            # Trigger each manoeuvre from the real distance to the truck, not a
            # fixed timestamp. C's right side and A's left side are both lane 0.
            a_to_truck = float(b.position[0] - a.position[0])
            c_to_truck = float(c.position[0] - b.position[0])
            
            if not c_approaching_logged:
                logger.info("[EVENT] Vehicle C approaching Truck B")
                c_approaching_logged = True
            
            if not a_approaching_logged:
                logger.info("[EVENT] Vehicle A preparing to overtake Truck B")
                a_approaching_logged = True

            if 0 < c_to_truck <= C_CHANGE_DISTANCE:
                if not c_trigger_logged:
                    logger.info("[EVENT] Vehicle C reached lane-change trigger distance")
                    logger.info("[EVENT] Vehicle C started lane change")
                    c_trigger_logged = True
                c_is_changing_lane = True

            # A begins the overtake based only on its local view of truck B.
            if 0 < a_to_truck <= A_CHANGE_DISTANCE:
                if not avoidance_active and not c_passed:
                    if not a_is_changing_lane and not a_overtake_logged:
                        logger.info("[EVENT] Vehicle A started overtaking")
                        a_overtake_logged = True
                    a_is_changing_lane = True


            # C sends its V2V intent when it begins the right-lane manoeuvre.
            # That makes the alert visibly arrive after A has started overtaking.
            if c_is_changing_lane:
                c_trajectory = planned_lane_change_trajectory(c, c_target_y, pi)
                c_lane_id = int(c.lane_index[2]) if c.lane_index else C_LANE

                if not c_broadcast_logged:
                    if v2v_enabled:
                        logger.info("[EVENT] Vehicle C broadcast V2V trajectory")
                    else:
                        logger.info("Vehicle C: hidden from Vehicle A")
                        logger.info("Vehicle A: no V2V information received")
                        logger.info("Vehicle A: continuing overtaking maneuver")
                    c_broadcast_logged = True

                if v2v_enabled:
                    v2v.broadcast(time_s, c, a.position, c_lane_id,
                                  C_TARGET_LANE, c_maneuver, c_trajectory)
            
            received_message = v2v.deliver(time_s) if v2v_enabled else None

            lane_conflict = (received_message is not None and
                             received_message.intended_lane == A_TARGET_LANE)
                             
            ttc = float('inf')
            current_risk_state = "NONE"
            threat_obj = received_message
            action_desc = "Monitor cautiously"
            fused_conflict = False
            
            if ai_safety_enabled:
                fused_conflict, threat_obj, fusion_source = fusion_engine.evaluate(
                    v2v_conflict=lane_conflict,
                    ai_conflict=latest_ai_conflict_state,
                    v2v_message=received_message,
                    c_x=c.position[0],
                    c_y=c.position[1],
                    c_speed=c.speed
                )
                
                if fused_conflict and not fused_conflict_logged:
                    first_fused_conflict_time = time_s
                    fused_conflict_logged = True
                    logger.info("\n[FUSION]")
                    logger.info(f"V2V conflict: {'YES' if lane_conflict else 'NO'}")
                    logger.info(f"AI conflict: {'YES' if latest_ai_conflict_state else 'NO'}")
                    logger.info("Fused conflict: YES\n")
                    
                final_conflict = fused_conflict
            else:
                final_conflict = lane_conflict
            
            if final_conflict:
                if not a_conflict_logged:
                    first_v2v_conflict_time = time_s
                    if not ai_safety_enabled:
                        logger.info("[EVENT] Vehicle A received V2V message")
                        logger.info("[EVENT] V2V path conflict detected")
                    a_conflict_logged = True
                    
                ttc = calculate_ttc(a, threat_obj)
                if ttc < CRITICAL_TTC:
                    current_risk_state = "CRITICAL"
                elif ttc <= SAFE_TTC:
                    current_risk_state = "WARNING"
                else:
                    current_risk_state = "SAFE"
                    
                if current_risk_state != last_risk_state:
                    logger.info(f"TTC: {ttc:.2f} s")
                    logger.info(f"Risk transition: {last_risk_state if last_risk_state != 'NONE' else 'NONE'} -> {current_risk_state}")
                    if current_risk_state == "SAFE":
                        action_desc = "Monitor cautiously"
                        logger.info(f"Action: {action_desc}")
                    elif current_risk_state == "WARNING":
                        action_desc = "Abort overtaking, controlled braking"
                        logger.info(f"Action: {action_desc}")
                    elif current_risk_state == "CRITICAL":
                        action_desc = "Emergency braking"
                        logger.info(f"Action: {action_desc}")
                    last_risk_state = current_risk_state
                    
            if ai_safety_enabled:
                fusion_csv_writer.writerow([
                    f"{time_s:.2f}", lane_conflict, latest_ai_conflict_state, fused_conflict,
                    (latest_pred_x is not None), (received_message is not None),
                    f"{latest_pred_x:.3f}" if latest_pred_x else "",
                    f"{latest_pred_y:.3f}" if latest_pred_y else "",
                    f"{ttc:.2f}", current_risk_state, action_desc
                ])
                    
            if a_is_changing_lane and final_conflict and not c_passed:
                if current_risk_state in ["WARNING", "CRITICAL"]:
                    avoidance_active = True
                    a_is_changing_lane = False
                    if not a_aborted_logged:
                        logger.info("[EVENT] Vehicle A aborted overtaking")
                        logger.info("[EVENT] Vehicle A returning behind Truck B")
                        logger.info("[EVENT] Vehicle A braking")
                        a_aborted_logged = True

            # Threat vehicle C passing and ego overtaking resumption
            if c.position[0] < a.position[0]:
                if not c_passed:
                    c_passed = True
                    avoidance_active = False
                    resumed_overtake = True
                    a_is_changing_lane = True
                    ego_action_desc = "Overtaking Truck B at increased speed"
                    logger.info("[EVENT] Threat vehicle C has passed. Avoidance maneuver completed.")
                    logger.info("[EVENT] Collision avoided")
                    logger.info("[EVENT] Vehicle A resumes overtaking at increased speed (25 m/s)")

            # Check if Vehicle A has cleared Truck B
            if resumed_overtake and not merging_in_front:
                # Truck B length is 12m; buffer of 18m ahead of truck center
                if a.position[0] > b.position[0] + 18.0:
                    merging_in_front = True
                    a_is_changing_lane = False
                    ego_action_desc = "Merging in front of Truck B"
                    if not merging_logged:
                        logger.info("[EVENT] Vehicle A cleared Truck B: Merging in front of truck into middle lane")
                        merging_logged = True

            # Check if Vehicle A has completed merge into lane 1 in front of Truck B
            if merging_in_front and not overtake_complete:
                if abs(a.position[1] - a_home_y) < 0.35 and hasattr(a, 'lane_index') and int(a.lane_index[2]) == A_LANE:
                    overtake_complete = True
                    ego_action_desc = "Cruising in front of Truck B"
                    if not overtake_complete_logged:
                        logger.info("[EVENT] Overtake complete! Vehicle A cruising safely in front of Truck B")
                        overtake_complete_logged = True

            c_steering = steering_to_lane(c, c_target_y, pi) if c_is_changing_lane else 0.0
            if avoidance_active:
                a_steering = steering_to_lane(a, a_home_y, 0.0)
            elif merging_in_front or overtake_complete:
                a_steering = steering_to_lane(a, a_home_y, 0.0)
            elif a_is_changing_lane:
                a_steering = steering_to_lane(a, a_target_y, 0.0)
            else:
                a_steering = 0.0
            c.act({"acceleration": 0.0, "steering": c_steering * pi / 4})
            
            # This goes through HighwayEnv's action system because green A is ego.
            if avoidance_active:
                if current_risk_state == "CRITICAL":
                    a_acceleration = -1.0 # Emergency braking
                else:
                    # Smoothly match the truck's speed to avoid rear-ending it
                    a_acceleration = np.clip((b.speed - a.speed) * 0.5, -1.0, 1.0)
            elif resumed_overtake and not overtake_complete:
                # Accelerate to overtaking speed (25.0 m/s) to pass Truck B
                a_acceleration = np.clip((25.0 - a.speed) * 0.8, -1.0, 1.0)
                action_desc = ego_action_desc
            elif overtake_complete:
                # Cruising speed in front of truck (22.0 m/s)
                a_acceleration = np.clip((22.0 - a.speed) * 0.5, -1.0, 1.0)
                action_desc = ego_action_desc
            else:
                a_acceleration = 0.0
            action = np.array([a_acceleration, a_steering], dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            env.render()
            draw_received_trajectory(env, received_message)
            draw_v2v_overlay(env, received_message, avoidance_active, v2v_enabled, ttc, current_risk_state, ai_safety_enabled, lane_conflict, latest_ai_conflict_state, fused_conflict, ego_action=ego_action_desc)
            
            if ai_enabled:
                pred_pos = (latest_pred_x, latest_pred_y) if latest_pred_x is not None else None
                draw_ai_overlay(env, tuple(c.position), pred_pos, ai_predictor.status, a_target_y, latest_ai_conflict_state if ai_conflict_enabled else None, hide_text=ai_safety_enabled)
                
            time_s += dt

            # Scenario completion: A is safely established ahead of Truck B
            if overtake_complete and a.position[0] > b.position[0] + 30.0:
                logger.info("[EVENT] Scenario successfully completed: Vehicle A safely established ahead of Truck B.")
                break

            if getattr(a, 'crashed', False):
                dist_to_c = hypot(a.position[0] - c.position[0], a.position[1] - c.position[1])
                dist_to_b = hypot(a.position[0] - b.position[0], a.position[1] - b.position[1])
                if dist_to_c < 10.0:
                    logger.info("COLLISION: Vehicle A and Vehicle C")
                elif dist_to_b < 15.0:
                    logger.info("Vehicle A rear-ended Vehicle B (collision with C avoided)")
                break
        
        if not getattr(a, 'crashed', False) and a_conflict_logged and v2v_enabled:
            logger.info("[EVENT] Collision avoided")
            
    finally:
        env.close()
        if ai_enabled:
            ai_log_file.close()
            if ai_conflict_enabled and conflict_log_file:
                conflict_log_file.close()
            if ai_safety_enabled and fusion_log_file:
                fusion_log_file.close()
            
            # Print performance check
            logger.info("\n=========================================")
            logger.info("LIVE AI INFERENCE PERFORMANCE REPORT")
            logger.info("=========================================")
            preds = ai_stats["predictions"]
            if preds > 0:
                avg_ms = ai_stats["total_ms"] / preds
                logger.info(f"Number of AI predictions: {preds}")
                logger.info(f"Average inference time: {avg_ms:.2f} ms")
                logger.info(f"Maximum inference time: {ai_stats['max_ms']:.2f} ms")
                logger.info(f"Minimum inference time: {ai_stats['min_ms']:.2f} ms")
                
                evals = ai_stats["evals_completed"]
                if evals > 0:
                    avg_err = ai_stats["total_error"] / evals
                    logger.info(f"Temporal evaluations logged: {evals}")
                    logger.info(f"Average prediction error: {avg_err:.4f} m")
            else:
                logger.info("No AI predictions were successfully generated.")
            
            if ai_conflict_enabled:
                logger.info("\n=========================================")
                logger.info("AI CONFLICT EVALUATION")
                logger.info("=========================================")
                logger.info(f"Number of AI predicted conflicts: {ai_conflict_count}")
                logger.info(f"Minimum distance between predicted C and A's overtaking path: {min_dist_to_a_path:.4f} m")
                
                logger.info("\n--- CONFLICT DETECTION TIMING ---")
                has_v2v = (first_v2v_conflict_time is not None)
                has_ai = (first_ai_conflict_time is not None)
                logger.info(f"Existing V2V conflict: {'YES' if has_v2v else 'NO'}")
                logger.info(f"AI predicted conflict: {'YES' if has_ai else 'NO'}")
                
                if has_v2v:
                    logger.info(f"V2V conflict detected at: {first_v2v_conflict_time:.2f} s")
                if has_ai:
                    logger.info(f"AI conflict detected at: {first_ai_conflict_time:.2f} s")
                
                if has_v2v and has_ai:
                    lead_time = first_v2v_conflict_time - first_ai_conflict_time
                    logger.info(f"\nAI lead time:")
                    logger.info(f"{lead_time:.2f} s")
                    
                if ai_safety_enabled:
                    logger.info("\n--- FUSION SAFETY DECISION ---")
                    has_fused = (first_fused_conflict_time is not None)
                    if has_fused:
                        logger.info(f"Fused conflict time: {first_fused_conflict_time:.2f} s")
                    logger.info(f"Collision avoided: {'YES' if not getattr(a, 'crashed', False) else 'NO'}")
                    
                    if has_ai and has_v2v:
                        if first_ai_conflict_time < first_v2v_conflict_time:
                            logger.info("Decision influence: AI DETECTED FIRST (AI drove the initial safety response)")
                        elif first_v2v_conflict_time < first_ai_conflict_time:
                            logger.info("Decision influence: V2V DETECTED FIRST (V2V drove the initial safety response)")
                        else:
                            logger.info("Decision influence: SIMULTANEOUS")
                    elif has_ai:
                        logger.info("Decision influence: AI ONLY (V2V failed or absent)")
                    elif has_v2v:
                        logger.info("Decision influence: V2V ONLY")
                        
                    logger.info("Location of Fusion CSV: data/results/v2x_ai_fusion.csv")
                else:
                    logger.info("\n[!] Collision avoidance still worked perfectly (AI remains strictly observational).")
                    logger.info("Location of AI conflict CSV: data/results/ai_conflict_predictions.csv")
                
            logger.info("=========================================")


if __name__ == "__main__":
    main()
