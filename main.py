"""Step 1: a deterministic three-vehicle HighwayEnv scene."""
import argparse
import logging
from math import atan2, cos, hypot, pi, sin

import gymnasium as gym
import highway_env  # Registers highway-v0 with Gymnasium.
import numpy as np
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from v2v_manager import V2VManager

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
C_CHANGE_DISTANCE = 45.0
# A starts overtaking before C sends its lane-change warning.
A_CHANGE_DISTANCE = 37.0
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


def draw_v2v_overlay(env, message, avoidance_active: bool, v2v_enabled: bool, ttc: float = float('inf'), risk_state: str = "NONE") -> None:
    """A small display panel; it does not affect HighwayEnv's road renderer."""
    import pygame
    viewer = env.unwrapped.viewer
    if viewer is None:
        return
    panel = pygame.Surface((410, 200), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 175))
    viewer.screen.blit(panel, (15, 15))
    font = pygame.font.SysFont("consolas", 19)
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
    for i, line in enumerate(lines):
        if "BRAKING" in line or "CRITICAL" in line:
            colour = (255, 70, 70)
        elif "WARNING" in line:
            colour = (255, 210, 70)
        elif "SAFE" in line or "CONNECTED" in line:
            colour = (80, 230, 120)
        else:
            colour = (240, 240, 240)
        viewer.screen.blit(font.render(line, True, colour), (27, 27 + 22 * i))
    pygame.display.flip()


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
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--with-v2v', action='store_true', default=True, dest='v2v_enabled',
                       help='Enable V2V communication (default)')
    group.add_argument('--without-v2v', action='store_false', dest='v2v_enabled',
                       help='Disable V2V communication')
    args = parser.parse_args()
    v2v_enabled = args.v2v_enabled

    if v2v_enabled:
        logger.info("V2V MODE: ENABLED")
    else:
        logger.info("V2V MODE: DISABLED")

    env = gym.make(
        "highway-v0",
        render_mode="human",
        config={
            "lanes_count": LANES,
            "vehicles_count": 0,
            "controlled_vehicles": 1,
            "duration": 10,
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

    a_approaching_logged = False
    a_overtake_logged = False
    c_approaching_logged = False
    c_trigger_logged = False
    c_broadcast_logged = False
    a_conflict_logged = False
    a_aborted_logged = False
    last_risk_state = "NONE"
    try:
        terminated = truncated = False
        while not (terminated or truncated):
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
                if not avoidance_active:
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
            if lane_conflict:
                if not a_conflict_logged:
                    logger.info("[EVENT] Vehicle A received V2V message")
                    logger.info("[EVENT] V2V path conflict detected")
                    a_conflict_logged = True
                    
                ttc = calculate_ttc(a, received_message)
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
                        logger.info("Action: Monitor cautiously")
                    elif current_risk_state == "WARNING":
                        logger.info("Action: Abort overtaking, controlled braking")
                    elif current_risk_state == "CRITICAL":
                        logger.info("Action: Emergency braking")
                    last_risk_state = current_risk_state

            if a_is_changing_lane and lane_conflict:
                if current_risk_state in ["WARNING", "CRITICAL"]:
                    avoidance_active = True
                    a_is_changing_lane = False
                    if not a_aborted_logged:
                        logger.info("[EVENT] Vehicle A aborted overtaking")
                        logger.info("[EVENT] Vehicle A returning behind Truck B")
                        logger.info("[EVENT] Vehicle A braking")
                        a_aborted_logged = True

            c_steering = steering_to_lane(c, c_target_y, pi) if c_is_changing_lane else 0.0
            if avoidance_active:
                a_steering = steering_to_lane(a, a_home_y, 0.0)
            else:
                a_steering = steering_to_lane(a, a_target_y, 0.0) if a_is_changing_lane else 0.0
            c.act({"acceleration": 0.0, "steering": c_steering * pi / 4})
            
            # This goes through HighwayEnv's action system because green A is ego.
            if avoidance_active:
                if current_risk_state == "CRITICAL":
                    a_acceleration = -1.0 # Emergency braking
                else:
                    # Smoothly match the truck's speed to avoid rear-ending it
                    a_acceleration = np.clip((b.speed - a.speed) * 0.5, -1.0, 1.0)
            else:
                a_acceleration = 0.0
            action = np.array([a_acceleration, a_steering], dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            env.render()
            draw_received_trajectory(env, received_message)
            draw_v2v_overlay(env, received_message, avoidance_active, v2v_enabled, ttc, current_risk_state)
            time_s += dt

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


if __name__ == "__main__":
    main()
