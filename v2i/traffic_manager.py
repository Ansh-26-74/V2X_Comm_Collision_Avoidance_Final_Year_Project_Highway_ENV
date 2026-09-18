"""Phase 2: Civilian Traffic Simulation & Kinematic Car-Following System.

Implements:
  - CivilianVehicle: continuous kinematic vehicle model with approach-based heading,
    smooth acceleration/deceleration, signal compliance, and intersection crossing.
  - TrafficManager: deterministic traffic generation, safe spacing, following distance
    maintenance (IDM-style car following), and lifecycle management (spawn/despawn).
  - Strict signal compliance:
      RED    -> smooth deceleration, complete stop behind stop line
      YELLOW -> prepare to stop if safe, or clear if already committed
      GREEN  -> accelerate to cruise speed and cross intersection
      IN_BOX -> always clear intersection safely

This module has NO dependency on any V2V code.
Ambulance / emergency vehicles are strictly excluded in this phase.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
import logging
import math
import random
from typing import Any
import pygame

from v2i.smart_signal import SignalState, SmartTrafficSignal

logger = logging.getLogger(__name__)


class VehicleState(str, Enum):
    APPROACHING = "APPROACHING"
    DECELERATING = "DECELERATING"
    STOPPED = "STOPPED"
    ACCELERATING = "ACCELERATING"
    OVERTAKING = "OVERTAKING"
    CROSSING = "CROSSING"
    CLEARED = "CLEARED"
    # Phase 7 Emergency States
    WAITING_AT_RED = "WAITING_AT_RED"
    AUTHORIZED_TO_PROCEED = "AUTHORIZED_TO_PROCEED"
    CROSSING_INTERSECTION = "CROSSING_INTERSECTION"
    EXITING_INTERSECTION = "EXITING_INTERSECTION"
    # Phase 13B V2V Evasive States
    V2V_EVASIVE_MANEUVER = "V2V_EVASIVE_MANEUVER"
    V2V_EMERGENCY_BRAKING = "V2V_EMERGENCY_BRAKING"


class AmbulanceState(str, Enum):
    APPROACHING = "APPROACHING"
    DECELERATING = "DECELERATING"
    STOPPED = "STOPPED"
    WAITING_AT_RED = "WAITING_AT_RED"
    AUTHORIZED_TO_PROCEED = "AUTHORIZED_TO_PROCEED"
    ACCELERATING = "ACCELERATING"
    OVERTAKING = "OVERTAKING"
    CROSSING_INTERSECTION = "CROSSING_INTERSECTION"
    CROSSING = "CROSSING"
    EXITING_INTERSECTION = "EXITING_INTERSECTION"
    CLEARED = "CLEARED"
    # Phase 13B V2V Evasive States
    V2V_EVASIVE_MANEUVER = "V2V_EVASIVE_MANEUVER"
    V2V_EMERGENCY_BRAKING = "V2V_EMERGENCY_BRAKING"


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle Physical & Kinematic Parameters
# ─────────────────────────────────────────────────────────────────────────────
VEHICLE_LENGTH = 36.0        # pixels (~3.6 metres)
VEHICLE_WIDTH  = 18.0        # pixels (~1.8 metres)
AMBULANCE_LENGTH = 44.0      # pixels (~4.4 metres, longer wheelbase)
AMBULANCE_WIDTH  = 20.0      # pixels (~2.0 metres, wider emergency chassis)
DEFAULT_CRUISE_SPEED = 95.0  # pixels/second (~34 km/h)
MAX_ACCEL = 70.0             # pixels/s^2 (smooth acceleration)
MAX_BRAKE = 130.0            # pixels/s^2 (comfortable deceleration)
EMERGENCY_BRAKE = 220.0      # pixels/s^2 (max safety brake)
MIN_FOLLOW_GAP = 22.0        # pixels buffer between stopped vehicles
STOP_LINE_BUFFER = 10.0      # pixels to stop before the physical stop line

# Phase 7 Bug Fix: Overtaking & Multi-Lane Parameters
SOUTH_PRIMARY_LANE_X = 592.0   # Primary South->North travel lane
SOUTH_OVERTAKE_LANE_X = 628.0  # Adjacent same-direction overtaking lane (East)
LATERAL_SPEED = 45.0           # pixels/second for smooth lateral lane changing
OBSTACLE_LOOKAHEAD = 95.0      # pixels lookahead to detect blocking vehicle


CIVILIAN_PALETTE = [
    ((45, 110, 210), "Sapphire Blue"),
    ((175, 45, 55),  "Crimson Red"),
    ((90, 100, 115), "Slate Gray"),
    ((200, 205, 215), "Silver Pearl"),
    ((35, 140, 95),  "Emerald Green"),
    ((215, 150, 40), "Amber Gold"),
    ((125, 75, 175), "Deep Purple"),
]


class CivilianVehicle:
    """Represents an individual civilian automobile."""

    def __init__(
        self,
        vehicle_id: str,
        approach: str,
        x: float,
        y: float,
        speed: float = DEFAULT_CRUISE_SPEED,
        target_speed: float = DEFAULT_CRUISE_SPEED,
        color: tuple = (45, 110, 210),
    ):
        self.vehicle_id = vehicle_id
        self.approach = approach.upper()  # "NORTH", "SOUTH", "EAST", "WEST"
        self.x = float(x)
        self.y = float(y)
        self.speed = float(speed)
        self.target_speed = float(target_speed)
        self.color = color

        self.length = VEHICLE_LENGTH
        self.width = VEHICLE_WIDTH
        self.state = VehicleState.APPROACHING

        self.braking = False
        self.past_stop_line = False
        self.in_intersection = False
        self.cleared = False

        # Phase 13A: V2V Inter-vehicle status
        self.hazard_status: str = "NORMAL"

        # Set heading angle (radians) and unit movement vector
        if self.approach == "NORTH":
            self.heading = math.pi / 2.0   # Facing South (+Y)
            self.vx, self.vy = 0.0, 1.0
        elif self.approach == "SOUTH":
            self.heading = -math.pi / 2.0  # Facing North (-Y)
            self.vx, self.vy = 0.0, -1.0
        elif self.approach == "WEST":
            self.heading = 0.0             # Facing East (+X)
            self.vx, self.vy = 1.0, 0.0
        elif self.approach == "EAST":
            self.heading = math.pi         # Facing West (-X)
            self.vx, self.vy = -1.0, 0.0

    def get_v2v_telemetry(self, now: float = 0.0):
        """Phase 13A: Expose Cooperative Awareness Message (CAM) telemetry."""
        from v2i.v2v_inter_manager import IntersectionV2VMessage
        return IntersectionV2VMessage(
            sender_id=self.vehicle_id,
            receiver_id="BROADCAST",
            vehicle_type="EMERGENCY" if getattr(self, "is_emergency", False) else "CIVILIAN",
            x=float(self.x),
            y=float(self.y),
            vx=float(self.vx),
            vy=float(self.vy),
            speed=float(self.speed),
            heading=float(self.heading),
            hazard_status=self.hazard_status,
            timestamp=float(now),
        )

    def set_hazard(self, hazard_status: str = "DECELERATING", target_speed: float | None = None) -> None:
        """Phase 13A: Set hazard state and optional target deceleration speed."""
        self.hazard_status = hazard_status
        if target_speed is not None:
            self.target_speed = float(target_speed)

    @property
    def front_pos(self) -> tuple[float, float]:
        """Coordinates of vehicle's front bumper."""
        return (
            self.x + self.vx * (self.length / 2.0),
            self.y + self.vy * (self.length / 2.0),
        )

    @property
    def rear_pos(self) -> tuple[float, float]:
        """Coordinates of vehicle's rear bumper."""
        return (
            self.x - self.vx * (self.length / 2.0),
            self.y - self.vy * (self.length / 2.0),
        )

    def get_distance_to_point(self, pt: float) -> float:
        """Distance along travel direction from front bumper to point."""
        if self.approach == "NORTH":
            return pt - (self.y + self.length / 2.0)
        elif self.approach == "SOUTH":
            return (self.y - self.length / 2.0) - pt
        elif self.approach == "WEST":
            return pt - (self.x + self.length / 2.0)
        elif self.approach == "EAST":
            return (self.x - self.length / 2.0) - pt
        return 9999.0

    @property
    def bounding_box(self) -> pygame.Rect:
        """Axis-aligned bounding box (pygame.Rect) for deterministic collision checks."""
        if self.approach in ("NORTH", "SOUTH"):
            w = self.width
            h = self.length
        else:
            w = self.length
            h = self.width
        return pygame.Rect(
            int(self.x - w / 2.0),
            int(self.y - h / 2.0),
            int(w),
            int(h),
        )

    def collides_with(self, other: "CivilianVehicle") -> bool:
        """Deterministic AABB overlap check with another vehicle."""
        return self.bounding_box.colliderect(other.bounding_box)

    def update_control(
        self,
        dt: float,
        signal_state: SignalState,
        stop_line_coord: float,
        intersection_enter: float,
        intersection_exit: float,
        lead_vehicle: "CivilianVehicle | None",
        intersection_blocked: bool = False,
    ) -> None:
        """Update vehicle acceleration, speed, position, and state."""
        # Check spatial landmarks along movement axis
        dist_to_stop_line = self.get_distance_to_point(stop_line_coord)
        dist_to_enter = self.get_distance_to_point(intersection_enter)
        dist_to_exit = self.get_distance_to_point(intersection_exit)

        # 1. Evaluate Traffic Signal restriction (only if before stop line and not inside intersection)
        signal_stop_needed = False
        target_stop_dist = dist_to_stop_line - STOP_LINE_BUFFER

        if not self.past_stop_line and not self.in_intersection:
            if signal_state == SignalState.RED:
                signal_stop_needed = True
            elif signal_state == SignalState.YELLOW:
                # Dilemma zone evaluation: continue stopping if already braking or if safely ahead of stop line
                stopping_dist_needed = (self.speed * self.speed) / (2.0 * MAX_BRAKE)
                if self.braking or (dist_to_stop_line > max(20.0, stopping_dist_needed * 0.8)):
                    signal_stop_needed = True  # Safe to stop
                else:
                    signal_stop_needed = False # Clear intersection
            elif signal_state == SignalState.GREEN and intersection_blocked:
                # Perpendicular cross-traffic is still clearing the junction box -> yield behind stop line
                signal_stop_needed = True

        # Flag progression past stop line (only legal if NOT required to stop)
        if dist_to_stop_line <= 0.0 and not self.past_stop_line and not signal_stop_needed:
            self.past_stop_line = True

        if dist_to_enter <= 0.0 and not self.in_intersection and not self.cleared:
            self.in_intersection = True

        if dist_to_exit <= 0.0 and not self.cleared:
            self.cleared = True
            self.in_intersection = False

        # 2. Car-Following / Obstacle Spacing restriction
        lead_gap = None
        if lead_vehicle is not None:
            lead_rear = lead_vehicle.rear_pos
            if self.approach in ("NORTH", "SOUTH"):
                lead_gap = abs(lead_rear[1] - self.front_pos[1])
            else:
                lead_gap = abs(lead_rear[0] - self.front_pos[0])

        # 3. Compute acceleration based on earliest stopping constraint
        effective_stop_dist = 9999.0

        if signal_stop_needed:
            effective_stop_dist = target_stop_dist

        if lead_gap is not None:
            lead_stop_dist = lead_gap - MIN_FOLLOW_GAP
            if lead_stop_dist < effective_stop_dist:
                effective_stop_dist = lead_stop_dist

        # Apply smooth proportional deceleration profile
        desired_accel = 0.0
        self.braking = False
        BRAKE_LOOKAHEAD = 160.0

        if effective_stop_dist <= 1.5:
            # At or slightly past stopping target -> brake to complete stop
            target_v = 0.0
            desired_accel = -MAX_BRAKE
            self.braking = True
            if self.speed < 4.0 or effective_stop_dist <= 0.0:
                self.speed = 0.0
                desired_accel = 0.0
        elif effective_stop_dist < BRAKE_LOOKAHEAD:
            # Smooth proportional deceleration curve: v_target = v_cruise * sqrt(d / d_lookahead)
            target_v = min(self.target_speed, self.target_speed * math.sqrt(effective_stop_dist / BRAKE_LOOKAHEAD))
            err = target_v - self.speed
            if err < 0:
                desired_accel = max(-MAX_BRAKE, err * 4.0)
                self.braking = True
            else:
                desired_accel = min(MAX_ACCEL, err * 2.0)
        else:
            # Free flow: accelerate toward cruise speed
            speed_err = self.target_speed - self.speed
            desired_accel = math.copysign(min(abs(speed_err) * 1.5, MAX_ACCEL), speed_err)

        # 4. Integrate kinematics
        self.speed = max(0.0, self.speed + desired_accel * dt)
        if (signal_stop_needed or (lead_gap is not None and lead_gap <= MIN_FOLLOW_GAP + 2.0)) and effective_stop_dist <= 2.0 and self.speed < 2.5:
            self.speed = 0.0

        self.x += self.vx * self.speed * dt
        self.y += self.vy * self.speed * dt

        # 5. Classify visual state
        if self.cleared:
            self.state = VehicleState.CLEARED
        elif self.in_intersection:
            self.state = VehicleState.CROSSING
        elif self.speed == 0.0:
            self.state = VehicleState.STOPPED
        elif self.braking:
            self.state = VehicleState.DECELERATING
        elif desired_accel > 5.0:
            self.state = VehicleState.ACCELERATING
        else:
            self.state = VehicleState.APPROACHING


class AmbulanceVehicle(CivilianVehicle):
    """Represents emergency vehicle AMB-01 traveling South -> North."""

    def __init__(
        self,
        vehicle_id: str = "AMB-01",
        approach: str = "SOUTH",
        x: float = 0.0,
        y: float = 0.0,
        speed: float = DEFAULT_CRUISE_SPEED,
        target_speed: float = DEFAULT_CRUISE_SPEED,
    ):
        super().__init__(
            vehicle_id=vehicle_id,
            approach=approach,
            x=x,
            y=y,
            speed=speed,
            target_speed=target_speed,
            color=(250, 250, 252),
        )
        self.length = AMBULANCE_LENGTH
        self.width = AMBULANCE_WIDTH
        self.is_emergency = True
        self.strobe_timer = 0.0
        self.strobe_left_on = True

        # V2I Communication attributes (Phase 4)
        self.tx_interval: float = 1.0  # Anti-spam: maximum 1 update per second
        self._last_tx_time: float = -float("inf")
        self.has_transmitted_initial: bool = False
        self.v2i_tx_active: bool = False
        self.latest_request_payload: dict | None = None

        # Phase 7: Emergency Crossing State attributes
        self.is_authorized: bool = False
        self.cleared_event_fired: bool = False
        self.ambulance_state: AmbulanceState = AmbulanceState.APPROACHING
        self.emergency_cruise_speed: float = 110.0
        self.max_emergency_accel: float = 85.0

        # Phase 7 Bug Fix: Safe Overtaking & Lane Change attributes
        self.target_lane_x: float = x if x > 0 else SOUTH_PRIMARY_LANE_X
        self.is_overtaking: bool = False
        self.overtaking_complete: bool = False
        self.obstacle_detected: bool = False

        # Phase 13A: Intersection V2V Communication & Hazard Detection
        self.v2v_enabled: bool = True
        self.v2v_ttc: float = float("inf")
        self.v2v_risk_state: str = "NONE"
        self.latest_v2v_threat: Any | None = None
        self.v2v_detected_vehicle_id: str | None = None
        self.dist_to_conflict_zone: float = 9999.0
        self.lateral_maneuver_allowed: bool = True

        # Phase 13B: V2V Evasive Maneuver & Braking Fallback attributes
        self.is_v2v_evading: bool = False
        self.v2v_evasion_complete: bool = False
        self.v2v_emergency_braking: bool = False
        self.v2v_maneuver_attempted: bool = False

        # Phase 13D Step 3: Live AI Trajectory Prediction attributes
        self.latest_ai_prediction: Any | None = None
        self.ai_risk_state: str = "INSUFFICIENT_DATA"
        self.ai_prediction_available: bool = False

        # Phase 13D Step 4: Deterministic Safety Fusion attribute
        self.latest_safety_fusion: Any | None = None

    def evaluate_overtaking(
        self,
        vehicles: list[CivilianVehicle],
        stop_line_coord: float,
        intersection_enter: float,
        target_x: float = SOUTH_OVERTAKE_LANE_X,
    ) -> bool:
        """Evaluate whether adjacent same-direction lane is safe for overtaking.
        
        Requirements:
        1. Ambulance must be on South approach before entering the intersection conflict zone.
        2. Front clearance: Target lane ahead must have sufficient distance.
        3. Rear clearance: Target lane behind must have sufficient distance and safe closure rate.
        4. No vehicle occupies or moves into the lateral transition zone.
        """
        # Section 9 constraint: Never initiate an overtaking maneuver inside the intersection conflict zone
        if self.front_pos[1] <= intersection_enter + 15.0:
            return False

        # Check all civilian vehicles for conflicts in target lane or transition space
        for v in vehicles:
            if v is self:
                continue

            # 1. Target lane occupancy check
            if abs(v.x - target_x) < 22.0:
                if v.y < self.y:
                    # Vehicle ahead in target lane: check front clearance
                    gap_ahead = (self.front_pos[1] - v.rear_pos[1])
                    if gap_ahead < 65.0:
                        return False
                else:
                    # Vehicle behind in target lane: check rear clearance
                    gap_behind = (v.front_pos[1] - self.rear_pos[1])
                    if gap_behind < 50.0:
                        return False
                    # Check closure speed if vehicle behind is approaching fast
                    if v.speed > self.speed and gap_behind < 75.0:
                        return False

            # 2. Check if a vehicle is between current lane and target lane
            low_x = min(self.x, target_x) + 12.0
            high_x = max(self.x, target_x) - 12.0
            if low_x <= v.x <= high_x:
                v_dist_y = abs(v.y - self.y)
                if v_dist_y < (self.length + v.length) / 2.0 + 10.0:
                    return False

            # 3. Current lane check: cannot initiate lateral change if already penetrated into lead vehicle
            if abs(v.x - self.x) < 12.0 and v.y < self.y:
                if self.front_pos[1] < v.rear_pos[1]:
                    return False

        return True

    def evaluate_v2v_hazards(
        self,
        v2v_messages: list,
        now: float = 0.0,
    ) -> tuple[float, str, Any]:
        """Phase 13A: Evaluate V2V Cooperative Awareness Messages from lead vehicles in corridor.

        Filters:
          1. Vehicle is not self
          2. Movement direction is aligned (dot product >= 0.7)
          3. Vehicle is ahead along travel axis (longitudinal_dist > 0)
          4. Lateral offset is within movement corridor (|dx| <= 45.0px)

        TTC calculation:
          closing_speed = ambulance_speed - lead_speed
          If closing_speed <= 1.0 px/s: TTC = inf (SAFE)
          bumper_gap = longitudinal_dist - (self.length + lead_len)/2.0
          TTC = bumper_gap / closing_speed

        Risk:
          CRITICAL: TTC < 2.0s
          WARNING:  2.0s <= TTC <= 4.0s
          SAFE:     TTC > 4.0s
        """
        if not self.v2v_enabled:
            return self.v2v_ttc, self.v2v_risk_state, self.latest_v2v_threat

        if not v2v_messages:
            # Wireless CAM validity hold: preserve active threat between 10Hz packets until timeout (0.4s)
            if self.latest_v2v_threat is not None:
                msg_time = getattr(self.latest_v2v_threat, "timestamp", 0.0)
                if now - msg_time > 0.4:
                    self.v2v_ttc = float("inf")
                    self.v2v_risk_state = "NONE"
                    self.latest_v2v_threat = None
                    self.v2v_detected_vehicle_id = None
            else:
                self.v2v_ttc = float("inf")
                self.v2v_risk_state = "NONE"
            return self.v2v_ttc, self.v2v_risk_state, self.latest_v2v_threat

        min_ttc = float("inf")
        highest_threat = None
        vehicle_detected = False

        for msg in v2v_messages:
            if msg.sender_id == self.vehicle_id:
                continue

            # Direction alignment check
            dot = self.vx * msg.vx + self.vy * msg.vy
            if dot < 0.7:
                continue

            # Longitudinal distance along travel direction (South approach: traveling North, -Y)
            dx = msg.x - self.x
            dy = msg.y - self.y
            longitudinal_dist = dx * self.vx + dy * self.vy
            if longitudinal_dist <= 0.0:
                # Behind or level with ambulance
                continue

            # Lateral corridor check (must be in same or adjacent lane)
            lateral_dist = abs(dx * (-self.vy) + dy * self.vx)
            corridor_threshold = 22.0 if (self.v2v_evasion_complete or abs(self.x - SOUTH_OVERTAKE_LANE_X) < 1.0) else 45.0
            if lateral_dist > corridor_threshold:
                continue

            vehicle_detected = True
            lead_len = 36.0  # standard civilian length
            bumper_gap = longitudinal_dist - (self.length + lead_len) / 2.0
            closing_speed = self.speed - msg.speed

            if closing_speed <= 1.0:
                ttc = float("inf")
            else:
                ttc = max(0.0, bumper_gap) / closing_speed

            if ttc < min_ttc:
                min_ttc = ttc
                highest_threat = msg

        if not vehicle_detected:
            self.v2v_ttc = float("inf")
            self.v2v_risk_state = "NONE"
            self.latest_v2v_threat = None
            self.v2v_detected_vehicle_id = None
            return float("inf"), "NONE", None

        self.v2v_ttc = min_ttc
        self.latest_v2v_threat = highest_threat
        self.v2v_detected_vehicle_id = highest_threat.sender_id if highest_threat else None

        if min_ttc < 2.0:
            self.v2v_risk_state = "CRITICAL"
        elif min_ttc <= 4.0:
            self.v2v_risk_state = "WARNING"
        else:
            self.v2v_risk_state = "SAFE"

        return self.v2v_ttc, self.v2v_risk_state, self.latest_v2v_threat

    def calculate_eta(self, stop_line_coord: float) -> float:
        """Calculate dynamic Estimated Time of Arrival to the South stop line."""
        # South moves North (-Y), front bumper is at y - length / 2
        dist_to_stop = max(0.0, self.front_pos[1] - stop_line_coord)
        if dist_to_stop <= 12.0:
            return 0.0
        if self.speed > 5.0:
            return max(0.0, dist_to_stop / self.speed)
        return max(0.0, dist_to_stop / max(self.target_speed, 10.0))

    def generate_emergency_request(self, stop_line_coord: float, sim_time: float) -> dict:
        """Generate structured emergency request message with dynamic kinematics and ETA."""
        eta_val = self.calculate_eta(stop_line_coord)
        payload = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": self.vehicle_id,
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "position": (round(self.x, 1), round(self.y, 1)),
            "velocity": (round(self.vx * self.speed, 1), round(self.vy * self.speed, 1)),
            "speed": round(self.speed, 1),
            "heading": round(self.heading, 3),
            "approach": self.approach,
            "eta": round(eta_val, 1),
            "priority": "HIGH",
            "timestamp": round(sim_time, 2),
        }
        self.latest_request_payload = payload
        return payload

    def step_v2i_communication(
        self,
        now: float,
        rsu_pos: tuple[float, float],
        v2i_channel,
        stop_line_coord: float,
    ) -> tuple[bool, str]:
        """Check range and transmit periodic emergency requests without spamming.

        Returns (transmitted, status_str).
        """
        dx = rsu_pos[0] - self.x
        dy = rsu_pos[1] - self.y
        dist = math.hypot(dx, dy)

        if dist <= v2i_channel.v2i_range:
            # Within range! Check transmission throttle
            if (not self.has_transmitted_initial) or (now - self._last_tx_time >= self.tx_interval):
                msg = self.generate_emergency_request(stop_line_coord, now)
                success, status = v2i_channel.send_to_rsu(now, (self.x, self.y), rsu_pos, msg)
                self._last_tx_time = now
                self.has_transmitted_initial = True
                self.v2i_tx_active = success
                return success, status
            return False, "THROTTLED"
        else:
            self.v2i_tx_active = False
            v2i_channel.comm_state = "OUT_OF_RANGE"
            return False, "OUT_OF_RANGE"

    def update_control(
        self,
        dt: float,
        signal_state: SignalState,
        stop_line_coord: float,
        intersection_enter: float,
        intersection_exit: float,
        lead_vehicle: "CivilianVehicle | None",
        intersection_blocked: bool = False,
        all_vehicles: "list[CivilianVehicle] | None" = None,
    ) -> None:
        """Phase 7: Update emergency vehicle movement, authorization, crossing, and clearance."""
        # Toggle lightbar strobe at ~4.5 Hz
        self.strobe_timer += dt
        if self.strobe_timer >= 0.11:
            self.strobe_timer = 0.0
            self.strobe_left_on = not self.strobe_left_on

        # 1. Authorization trigger on South GREEN
        if signal_state == SignalState.GREEN:
            if not self.is_authorized:
                self.is_authorized = True
                self.target_speed = self.emergency_cruise_speed
                logger.info(f"[{self.vehicle_id}] AUTHORIZED TO PROCEED on South GREEN")

        # 2. Check spatial landmarks along movement axis (traveling North, -Y)
        dist_to_stop_line = self.get_distance_to_point(stop_line_coord)
        dist_to_enter = self.get_distance_to_point(intersection_enter)

        # Phase 13A: Update conflict zone proximity and maneuver constraint
        self.dist_to_conflict_zone = max(0.0, dist_to_enter)
        self.lateral_maneuver_allowed = (dist_to_enter > 15.0 and not self.in_intersection and not self.cleared)

        # Signal stop requirement:
        # Ambulance only stops for RED if NOT authorized, before stop line, and not yet in intersection
        signal_stop_needed = False
        target_stop_dist = dist_to_stop_line - STOP_LINE_BUFFER

        if not self.past_stop_line and not self.in_intersection and not self.cleared:
            if not self.is_authorized and signal_state in (SignalState.RED, SignalState.YELLOW):
                signal_stop_needed = True

        # Stop line progression
        if dist_to_stop_line <= 0.0 and not self.past_stop_line and not signal_stop_needed:
            self.past_stop_line = True

        # Intersection entry progression
        if dist_to_enter <= 0.0 and not self.in_intersection and not self.cleared:
            self.in_intersection = True
            logger.info(f"[{self.vehicle_id}] ENTERED INTERSECTION (Crossing protected phase)")

        # Exit detection: full vehicle body clearance
        # Northern exit boundary is intersection_exit.
        # Front bumper: y - length / 2; Rear bumper: y + length / 2.
        front_y = self.front_pos[1]
        rear_y = self.rear_pos[1]

        is_exiting = False
        if front_y <= intersection_exit and rear_y > intersection_exit and not self.cleared:
            is_exiting = True

        if rear_y <= intersection_exit and not self.cleared:
            self.cleared = True
            self.in_intersection = False
            if not self.cleared_event_fired:
                self.cleared_event_fired = True
                logger.info(f"[{self.vehicle_id}] AMB-01 CLEARED INTERSECTION")

        # 3. Detect blocking lead vehicle in current travel corridor
        vehicles_list: list[CivilianVehicle] = []
        if all_vehicles is not None:
            vehicles_list = [v for v in all_vehicles if v is not self]
        elif lead_vehicle is not None:
            vehicles_list = [lead_vehicle]

        lead_blocking = None
        min_lead_gap = 9999.0

        for v in vehicles_list:
            if v.y < self.y:
                lateral_overlap = (abs(v.x - self.x) < (self.width + v.width) / 2.0 + 2.0)
                if lateral_overlap:
                    gap = (self.front_pos[1] - v.rear_pos[1])
                    if 0.0 <= gap < min_lead_gap:
                        min_lead_gap = gap
                        lead_blocking = v

        if lead_blocking is None and lead_vehicle is not None:
            if lead_vehicle.y < self.y and abs(lead_vehicle.x - self.x) < (self.width + lead_vehicle.width) / 2.0 + 2.0:
                gap = (self.front_pos[1] - lead_vehicle.rear_pos[1])
                if 0.0 <= gap < min_lead_gap:
                    min_lead_gap = gap
                    lead_blocking = lead_vehicle

        self.obstacle_detected = False
        if lead_blocking is not None and min_lead_gap < OBSTACLE_LOOKAHEAD:
            if lead_blocking.speed < self.target_speed:
                self.obstacle_detected = True

        # 4. Phase 13B: V2V Evasive Maneuver vs. Phase 7 Overtaking decision
        # Check if a V2V CRITICAL hazard warrants an evasive lane maneuver:
        v2v_critical_threat = (
            self.v2v_risk_state == "CRITICAL"
            and self.latest_v2v_threat is not None
            and self.latest_v2v_threat.y < self.y
            and not self.v2v_evasion_complete
            and not self.is_v2v_evading
            and not self.in_intersection
            and not self.cleared
            and not self.past_stop_line
            and self.lateral_maneuver_allowed
        )

        if v2v_critical_threat:
            target_x = SOUTH_OVERTAKE_LANE_X if abs(self.x - SOUTH_PRIMARY_LANE_X) < 15.0 else SOUTH_PRIMARY_LANE_X
            if self.evaluate_overtaking(vehicles_list, stop_line_coord, intersection_enter, target_x):
                # Adjacent lane is SAFE -> initiate smooth lateral evasion
                self.is_v2v_evading = True
                self.is_overtaking = True
                self.target_lane_x = target_x
                self.v2v_emergency_braking = False
                self.v2v_maneuver_attempted = True
                self.ambulance_state = AmbulanceState.V2V_EVASIVE_MANEUVER
                self.state = VehicleState.V2V_EVASIVE_MANEUVER
                logger.info(f"[{self.vehicle_id}] V2V CRITICAL TTC ({self.v2v_ttc:.2f}s) -> INITIATING V2V EVASIVE MANEUVER to x={target_x}")
            else:
                # Adjacent lane is BLOCKED -> Fallback to controlled emergency braking
                self.is_v2v_evading = False
                self.is_overtaking = False
                self.v2v_emergency_braking = True
                self.v2v_maneuver_attempted = True
                self.ambulance_state = AmbulanceState.V2V_EMERGENCY_BRAKING
                self.state = VehicleState.V2V_EMERGENCY_BRAKING
                logger.warning(f"[{self.vehicle_id}] V2V CRITICAL TTC ({self.v2v_ttc:.2f}s) -> TARGET LANE BLOCKED -> INITIATING EMERGENCY BRAKING")
        elif (
            self.is_authorized
            and self.obstacle_detected
            and not self.is_overtaking
            and not self.overtaking_complete
            and not self.in_intersection
            and not self.past_stop_line
        ):
            target_x = SOUTH_OVERTAKE_LANE_X if abs(self.x - SOUTH_PRIMARY_LANE_X) < 15.0 else SOUTH_PRIMARY_LANE_X
            if self.evaluate_overtaking(vehicles_list, stop_line_coord, intersection_enter, target_x):
                self.is_overtaking = True
                self.target_lane_x = target_x
                logger.info(f"[{self.vehicle_id}] INITIATING OVERTAKING maneuver -> target lane x={self.target_lane_x}")

        # V2V Recovery: if hazard cleared while emergency braking, release emergency brake
        if self.v2v_emergency_braking and self.v2v_risk_state in ("SAFE", "NONE"):
            self.v2v_emergency_braking = False

        # 5. Smooth lateral lane change kinematics
        if self.is_overtaking or self.is_v2v_evading:
            dx = self.target_lane_x - self.x
            step = math.copysign(min(abs(dx), LATERAL_SPEED * dt), dx)
            prospective_rect = pygame.Rect(
                int(self.x + step - self.width / 2.0),
                int(self.y - self.length / 2.0),
                int(self.width),
                int(self.length),
            )
            collides_lateral = any(
                prospective_rect.colliderect(v.bounding_box)
                for v in vehicles_list
            )
            if not collides_lateral:
                self.x += step
                if abs(self.target_lane_x - self.x) < 0.5:
                    self.x = self.target_lane_x
                    if self.is_v2v_evading:
                        self.is_v2v_evading = False
                        self.v2v_evasion_complete = True
                        logger.info(f"[{self.vehicle_id}] V2V EVASIVE lane change complete at x={self.x}")
                    self.is_overtaking = False
                    self.overtaking_complete = True
                    logger.info(f"[{self.vehicle_id}] OVERTAKING lane change complete at x={self.x}")

        # 6. Spacing constraint & stopping distance calculation (Section 5 & 13)
        effective_stop_dist = 9999.0
        if signal_stop_needed:
            effective_stop_dist = target_stop_dist

        lead_in_lane = None
        lead_in_lane_gap = 9999.0
        for v in vehicles_list:
            if v.y < self.y:
                if abs(v.x - self.x) < (self.width + v.width) / 2.0 + 2.0:
                    gap = (self.front_pos[1] - v.rear_pos[1])
                    if 0.0 <= gap < lead_in_lane_gap:
                        lead_in_lane_gap = gap
                        lead_in_lane = v

        if lead_in_lane is not None:
            veh_stop_dist = lead_in_lane_gap - MIN_FOLLOW_GAP
            if veh_stop_dist < effective_stop_dist:
                effective_stop_dist = veh_stop_dist

        # Phase 13B: If V2V emergency braking is active, enforce strict stop behind threat
        if self.v2v_emergency_braking:
            threat_gap = 9999.0
            if self.latest_v2v_threat is not None and self.latest_v2v_threat.y < self.y:
                threat_rear_y = self.latest_v2v_threat.y + 18.0  # civilian half length
                threat_gap = max(0.0, self.front_pos[1] - threat_rear_y)
            elif lead_in_lane is not None:
                threat_gap = (self.front_pos[1] - lead_in_lane.rear_pos[1])

            v2v_stop_dist = max(0.0, threat_gap - MIN_FOLLOW_GAP)
            if v2v_stop_dist < effective_stop_dist:
                effective_stop_dist = v2v_stop_dist

        # 7. Acceleration and smooth deterministic deceleration
        desired_accel = 0.0
        self.braking = False
        BRAKE_LOOKAHEAD = 160.0

        if self.v2v_emergency_braking:
            # High-priority controlled emergency braking fallback
            if effective_stop_dist <= 2.0:
                target_v = 0.0
                desired_accel = -EMERGENCY_BRAKE
                self.braking = True
                if self.speed < 4.0 or effective_stop_dist <= 0.0:
                    self.speed = 0.0
                    desired_accel = 0.0
            elif effective_stop_dist < BRAKE_LOOKAHEAD:
                target_v = min(self.target_speed, self.target_speed * math.sqrt(max(0.0, effective_stop_dist) / BRAKE_LOOKAHEAD))
                err = target_v - self.speed
                if err < 0:
                    desired_accel = max(-EMERGENCY_BRAKE, err * 6.0)
                    self.braking = True
                else:
                    desired_accel = min(self.max_emergency_accel, err * 2.0)
            else:
                desired_accel = -MAX_BRAKE
                self.braking = True
        elif effective_stop_dist < 9990.0:
            if effective_stop_dist <= 1.5:
                target_v = 0.0
                desired_accel = -MAX_BRAKE
                self.braking = True
                if self.speed < 4.0 or effective_stop_dist <= 0.0:
                    self.speed = 0.0
                    desired_accel = 0.0
            elif effective_stop_dist < BRAKE_LOOKAHEAD:
                target_v = min(self.target_speed, self.target_speed * math.sqrt(max(0.0, effective_stop_dist) / BRAKE_LOOKAHEAD))
                err = target_v - self.speed
                if err < 0:
                    desired_accel = max(-MAX_BRAKE, err * 4.0)
                    self.braking = True
                else:
                    desired_accel = min(self.max_emergency_accel, err * 2.0)
            else:
                speed_err = self.target_speed - self.speed
                desired_accel = math.copysign(min(abs(speed_err) * 1.5, self.max_emergency_accel), speed_err)
        else:
            # Free flow / protected acceleration
            speed_err = self.target_speed - self.speed
            desired_accel = min(self.max_emergency_accel, max(-MAX_BRAKE, speed_err * 2.0))

        # 8. Integrate forward kinematics
        self.speed = max(0.0, self.speed + desired_accel * dt)
        if effective_stop_dist <= 2.0 and self.speed < 2.5:
            self.speed = 0.0

        # Prospective forward movement collision check (HARD COLLISION INVARIANT)
        prospective_y = self.y + self.vy * self.speed * dt
        prospective_rect = pygame.Rect(
            int(self.x - self.width / 2.0),
            int(prospective_y - self.length / 2.0),
            int(self.width),
            int(self.length),
        )
        collision_detected = any(
            prospective_rect.colliderect(v.bounding_box)
            for v in vehicles_list
        )
        if collision_detected:
            self.speed = 0.0
            self.braking = True
        else:
            self.y = prospective_y

        # 9. Classify visual and logical state
        if self.cleared:
            self.state = VehicleState.CLEARED
            self.ambulance_state = AmbulanceState.CLEARED
        elif is_exiting:
            self.state = VehicleState.EXITING_INTERSECTION
            self.ambulance_state = AmbulanceState.EXITING_INTERSECTION
        elif self.in_intersection:
            self.state = VehicleState.CROSSING_INTERSECTION
            self.ambulance_state = AmbulanceState.CROSSING_INTERSECTION
        elif self.is_v2v_evading:
            self.state = VehicleState.V2V_EVASIVE_MANEUVER
            self.ambulance_state = AmbulanceState.V2V_EVASIVE_MANEUVER
        elif self.v2v_emergency_braking:
            self.state = VehicleState.V2V_EMERGENCY_BRAKING
            self.ambulance_state = AmbulanceState.V2V_EMERGENCY_BRAKING
        elif self.is_overtaking:
            self.state = VehicleState.OVERTAKING
            self.ambulance_state = AmbulanceState.OVERTAKING
        elif self.is_authorized:
            if self.speed == 0.0:
                self.state = VehicleState.AUTHORIZED_TO_PROCEED
                self.ambulance_state = AmbulanceState.AUTHORIZED_TO_PROCEED
            else:
                self.state = VehicleState.ACCELERATING
                self.ambulance_state = AmbulanceState.ACCELERATING
        elif self.speed == 0.0:
            self.state = VehicleState.STOPPED
            self.ambulance_state = AmbulanceState.WAITING_AT_RED
        elif self.braking:
            self.state = VehicleState.DECELERATING
            self.ambulance_state = AmbulanceState.DECELERATING
        elif desired_accel > 5.0:
            self.state = VehicleState.ACCELERATING
            self.ambulance_state = AmbulanceState.ACCELERATING
        else:
            self.state = VehicleState.APPROACHING
            self.ambulance_state = AmbulanceState.APPROACHING


class TrafficManager:
    """Spawns, coordinates, and tracks civilian vehicles and emergency vehicle AMB-01."""

    def __init__(
        self,
        center_x: int = 560,
        center_y: int = 415,
        road_width: int = 130,
        stop_line_dist: int = 75,
        max_vehicles: int = 7,
        spawn_interval: float = 3.6,
        ambulance_spawn_time: float = 8.5,
        enable_v2v_hazard: bool = True,
        ai_predictor: Any | None = None,
        safety_fusion: Any | None = None,
    ):
        self.cx = center_x
        self.cy = center_y
        self.road_w = road_width
        self.half_road = road_width // 2
        self.stop_line_dist = stop_line_dist

        self.max_vehicles = max_vehicles
        self.spawn_interval = spawn_interval
        self._spawn_timer = 0.5  # Spawn first vehicle quickly
        self._vehicle_counter = 1

        self.vehicles: list[CivilianVehicle] = []

        # Emergency vehicle AMB-01 (Phase 3)
        self.ambulance: AmbulanceVehicle | None = None
        self.ambulance_spawn_time = ambulance_spawn_time
        self._ambulance_spawned = False
        self.ambulance_cleared_events: list[str] = []

        # Phase 13A: V2V Inter-vehicle hazard simulation attributes
        self.enable_v2v_hazard: bool = enable_v2v_hazard
        self._v2v_hazard_injected: bool = False

        # Phase 13D: Live AI Trajectory Predictor (Step 3)
        self.ai_predictor = ai_predictor
        self.ai_loading: bool = (ai_predictor is None)
        self.ai_error: str | None = None
        self.latest_ai_prediction: Any | None = None
        self._pending_v2v_telemetry: dict[str, deque] = {}
        self._last_ai_feed_time: float = -float("inf")

        # Phase 13D: Deterministic Safety Fusion Engine (Step 4)
        if safety_fusion is None:
            try:
                from ai.intersection_ai.safety_fusion import IntersectionSafetyFusion
                self.safety_fusion = IntersectionSafetyFusion()
            except Exception:
                self.safety_fusion = None
        else:
            self.safety_fusion = safety_fusion

        # Lane center coordinates
        self.lane_coords = {
            "NORTH": {"x": self.cx - self.half_road // 2, "stop": self.cy - self.stop_line_dist, "enter": self.cy - self.half_road, "exit": self.cy + self.half_road},
            "SOUTH": {"x": self.cx + self.half_road // 2, "stop": self.cy + self.stop_line_dist, "enter": self.cy + self.half_road, "exit": self.cy - self.half_road, "overtake_x": SOUTH_OVERTAKE_LANE_X},
            "WEST":  {"y": self.cy + self.half_road // 2, "stop": self.cx - self.stop_line_dist, "enter": self.cx - self.half_road, "exit": self.cx + self.half_road},
            "EAST":  {"y": self.cy - self.half_road // 2, "stop": self.cx + self.stop_line_dist, "enter": self.cx + self.half_road, "exit": self.cx - self.half_road},
        }

        # Deterministic rotation order for approaches
        self.approach_queue = ["NORTH", "EAST", "SOUTH", "WEST"]
        self._approach_idx = 0

    def set_ai_predictor(self, predictor: Any) -> None:
        """Safely attach asynchronously loaded AI predictor and flush pending telemetry."""
        self.ai_predictor = predictor
        self.ai_loading = False
        if predictor is not None:
            # Flush retained telemetry samples in chronological order
            for sender_id, msgs in list(self._pending_v2v_telemetry.items()):
                for msg in msgs:
                    self.ai_predictor.add_telemetry(msg)
            self._pending_v2v_telemetry.clear()

            # Generate prediction immediately if C-01 or threat is available
            threat_id = "C-01"
            if self.ambulance is not None and self.ambulance.latest_v2v_threat:
                threat_id = self.ambulance.latest_v2v_threat.sender_id
            try:
                amb_pos = (self.ambulance.x, self.ambulance.y) if self.ambulance else None
                amb_spd = self.ambulance.speed if self.ambulance else 95.0
                pred = self.ai_predictor.predict(
                    sender_id=threat_id,
                    ambulance_pos=amb_pos,
                    ambulance_speed=amb_spd,
                )
                self.latest_ai_prediction = pred
                if self.ambulance is not None and pred.prediction_available:
                    self.ambulance.latest_ai_prediction = pred
                    self.ambulance.ai_prediction_available = True
                    self.ambulance.ai_risk_state = pred.ai_risk_state
            except Exception as e:
                logger.warning(f"[TRAFFIC_MANAGER] Initial AI prediction after attach handled safely: {e}")

    def _spawn_ambulance(self) -> None:
        """Spawn emergency vehicle AMB-01 on South approach lane."""
        lc = self.lane_coords["SOUTH"]
        self.ambulance = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=lc["x"],
            y=800.0,
            speed=95.0,
            target_speed=95.0,
        )
        self._ambulance_spawned = True

    def is_intersection_blocked_for(self, approach: str) -> bool:
        """Check if any perpendicular vehicle is currently inside the intersection box."""
        conflicting = ("EAST", "WEST") if approach in ("NORTH", "SOUTH") else ("NORTH", "SOUTH")
        return any(v.in_intersection for v in self.vehicles if v.approach in conflicting)

    def update(
        self,
        dt: float,
        signal_controller: SmartTrafficSignal,
        sim_time: float = 0.0,
        v2i_channel: Any = None,
        rsu_pos: tuple[float, float] | None = None,
        v2v_inter_manager: Any = None,
    ) -> None:
        """Update all active vehicles, ambulance, and periodically spawn new traffic."""
        # 1. Deterministic spawn check for AMB-01
        if self.ambulance is None and not self._ambulance_spawned and sim_time >= self.ambulance_spawn_time:
            self._spawn_ambulance()

        # 2. Update civilian vehicles
        for vehicle in self.vehicles:
            app = vehicle.approach
            sig_state = signal_controller.get_signal(app)
            lc = self.lane_coords[app]

            # Find lead vehicle ahead on the same approach
            lead_veh = self._find_lead_vehicle(vehicle)
            blocked = self.is_intersection_blocked_for(app)

            vehicle.update_control(
                dt=dt,
                signal_state=sig_state,
                stop_line_coord=lc["stop"],
                intersection_enter=lc["enter"],
                intersection_exit=lc["exit"],
                lead_vehicle=lead_veh,
                intersection_blocked=blocked,
            )

        # Ingestion of civilian telemetry (e.g. C-01) for intersection-level AI trajectory tracking
        for v in self.vehicles:
            if v.vehicle_id == "C-01":
                v_msg = v.get_v2v_telemetry(sim_time)
                if self.ai_predictor is not None:
                    hist = self.ai_predictor.history_buffers.get("C-01")
                    if not hist or abs(hist[-1].timestamp - v_msg.timestamp) > 0.05:
                        self.ai_predictor.add_telemetry(v_msg)
                        amb_pos = (self.ambulance.x, self.ambulance.y) if self.ambulance else None
                        amb_spd = self.ambulance.speed if self.ambulance else 95.0
                        pred = self.ai_predictor.predict(
                            sender_id="C-01",
                            ambulance_pos=amb_pos,
                            ambulance_speed=amb_spd,
                        )
                        self.latest_ai_prediction = pred
                        if self.ambulance is not None and not self.ambulance.ai_prediction_available:
                            self.ambulance.latest_ai_prediction = pred
                            self.ambulance.ai_prediction_available = pred.prediction_available
                            self.ambulance.ai_risk_state = pred.ai_risk_state
                            self._last_ai_feed_time = sim_time
                else:
                    if "C-01" not in self._pending_v2v_telemetry:
                        self._pending_v2v_telemetry["C-01"] = deque(maxlen=20)
                    buf = self._pending_v2v_telemetry["C-01"]
                    if not buf or abs(buf[-1].timestamp - v_msg.timestamp) > 0.05:
                        buf.append(v_msg)

        # 3. Update ambulance (Phase 3 & 4: obeys traffic signals, no V2I preemption)
        if self.ambulance is not None:
            sig_state = signal_controller.get_signal("SOUTH")
            lc = self.lane_coords["SOUTH"]
            lead_veh = self._find_lead_vehicle(self.ambulance)
            blocked = self.is_intersection_blocked_for("SOUTH")

            self.ambulance.update_control(
                dt=dt,
                signal_state=sig_state,
                stop_line_coord=lc["stop"],
                intersection_enter=lc["enter"],
                intersection_exit=lc["exit"],
                lead_vehicle=lead_veh,
                intersection_blocked=blocked,
                all_vehicles=self.vehicles,
            )

            # Phase 7 & 8: Record single cleared event & notify signal controller
            if self.ambulance.cleared and self.ambulance.cleared_event_fired:
                event_str = "AMB-01 CLEARED INTERSECTION"
                if event_str not in self.ambulance_cleared_events:
                    self.ambulance_cleared_events.append(event_str)
                    if signal_controller is not None and hasattr(signal_controller, "notify_ambulance_cleared"):
                        signal_controller.notify_ambulance_cleared(self.ambulance.vehicle_id)

            # Phase 4 V2I communication step (within range & throttled rate)
            if v2i_channel is not None and rsu_pos is not None:
                self.ambulance.step_v2i_communication(
                    now=sim_time,
                    rsu_pos=rsu_pos,
                    v2i_channel=v2i_channel,
                    stop_line_coord=lc["stop"],
                )

            # Phase 13A: V2V communication step (AMB-01 <-> Civilian vehicles)
            if v2v_inter_manager is not None:
                amb_pos = (self.ambulance.x, self.ambulance.y)
                for v in self.vehicles:
                    msg = v.get_v2v_telemetry(sim_time)
                    v2v_inter_manager.broadcast(sim_time, msg, receiver_pos=amb_pos)

                delivered_v2v = v2v_inter_manager.deliver(sim_time, receiver_id="AMB-01")
                self.ambulance.evaluate_v2v_hazards(delivered_v2v, now=sim_time)

                # Phase 13D Step 3: Live V2V + AI Trajectory Prediction Integration
                if self.ai_predictor is not None:
                    try:
                        if delivered_v2v:
                            for v2v_msg in delivered_v2v:
                                self.ai_predictor.add_telemetry(v2v_msg)

                                target_id = v2v_msg.sender_id
                                threat_id = self.ambulance.latest_v2v_threat.sender_id if self.ambulance.latest_v2v_threat else None

                                if target_id == threat_id or threat_id is None or target_id == "C-01" or not self.ambulance.ai_prediction_available:
                                    pred = self.ai_predictor.predict(
                                        sender_id=target_id,
                                        ambulance_pos=(self.ambulance.x, self.ambulance.y),
                                        ambulance_speed=self.ambulance.speed,
                                    )
                                    self.latest_ai_prediction = pred
                                    if pred.prediction_available:
                                        self.ambulance.latest_ai_prediction = pred
                                        self.ambulance.ai_prediction_available = True
                                        self.ambulance.ai_risk_state = pred.ai_risk_state
                                        self._last_ai_feed_time = sim_time
                                    elif self.ambulance.latest_ai_prediction is None:
                                        self.ambulance.latest_ai_prediction = pred
                                        self.ambulance.ai_prediction_available = False
                                        self.ambulance.ai_risk_state = pred.ai_risk_state
                        else:
                            # 0.4s telemetry hold across 10 Hz packet boundaries
                            if self.ambulance.latest_ai_prediction is not None:
                                if sim_time - self._last_ai_feed_time > 0.4:
                                    self.ambulance.latest_ai_prediction = None
                                    self.ambulance.ai_prediction_available = False
                                    self.ambulance.ai_risk_state = "INSUFFICIENT_DATA"
                    except Exception as e:
                        logger.warning(f"[TRAFFIC_MANAGER] AI prediction exception handled safely: {e}")
                else:
                    # Retain recent valid V2V telemetry while AI is still loading asynchronously
                    if delivered_v2v:
                        for v2v_msg in delivered_v2v:
                            sender = v2v_msg.sender_id
                            if sender not in self._pending_v2v_telemetry:
                                self._pending_v2v_telemetry[sender] = deque(maxlen=20)
                            self._pending_v2v_telemetry[sender].append(v2v_msg)

                # Phase 13D Step 4: Deterministic Safety Fusion Evaluation
                if self.safety_fusion is not None:
                    try:
                        gap = 9999.0
                        if self.ambulance.latest_v2v_threat is not None:
                            lead_threat = self.ambulance.latest_v2v_threat
                            gap = (self.ambulance.y - self.ambulance.length / 2.0) - (lead_threat.y + 36.0 / 2.0)

                        target_x = SOUTH_OVERTAKE_LANE_X if abs(self.ambulance.x - SOUTH_PRIMARY_LANE_X) < 15.0 else SOUTH_PRIMARY_LANE_X
                        target_safe = self.ambulance.evaluate_overtaking(
                            vehicles=self.vehicles,
                            stop_line_coord=lc["stop"],
                            intersection_enter=lc["enter"],
                            target_x=target_x,
                        )

                        fusion_res = self.safety_fusion.evaluate(
                            ai_prediction=self.ambulance.latest_ai_prediction,
                            ttc=self.ambulance.v2v_ttc,
                            ttc_risk=self.ambulance.v2v_risk_state,
                            deterministic_clearance=max(0.0, gap),
                            lateral_maneuver_allowed=self.ambulance.lateral_maneuver_allowed,
                            target_lane_safe=target_safe,
                            sim_time=sim_time,
                            in_intersection=self.ambulance.in_intersection,
                        )
                        self.ambulance.latest_safety_fusion = fusion_res
                    except Exception as e:
                        logger.warning(f"[TRAFFIC_MANAGER] Safety fusion exception handled safely: {e}")

                # Deterministic demonstration hazard injection before conflict zone
                if self.enable_v2v_hazard and not self._v2v_hazard_injected:
                    if (
                        self.ambulance.is_authorized
                        and self.ambulance.speed > 30.0
                        and self.ambulance.dist_to_conflict_zone > 25.0
                    ):
                        for v in self.vehicles:
                            if v.approach == "SOUTH" and v.y < self.ambulance.y and (self.ambulance.y - v.y) < 200.0:
                                v.set_hazard("DECELERATING", target_speed=15.0)
                                self._v2v_hazard_injected = True
                                logger.info(f"[{v.vehicle_id}] V2V HAZARD TRIGGERED: Unexpected deceleration to 15.0 px/s")
                                break

            if self._is_off_screen(self.ambulance):
                self.ambulance = None

        # 4. Despawn vehicles that have traveled off-screen
        self.vehicles = [v for v in self.vehicles if not self._is_off_screen(v)]

        # 5. Traffic generator timer
        self._spawn_timer -= dt
        if self._spawn_timer <= 0.0:
            self._spawn_timer = self.spawn_interval
            if len(self.vehicles) < self.max_vehicles:
                self._try_spawn_vehicle()

    def _find_lead_vehicle(self, veh: CivilianVehicle) -> CivilianVehicle | None:
        """Find the vehicle directly ahead in the same approach lane."""
        same_lane = [v for v in self.vehicles if v.approach == veh.approach and v is not veh]
        if self.ambulance is not None and veh is not self.ambulance and self.ambulance.approach == veh.approach:
            same_lane.append(self.ambulance)

        lead = None
        min_dist = 9999.0

        for other in same_lane:
            # Check if other is in the same lateral lane corridor
            if veh.approach in ("NORTH", "SOUTH") and abs(other.x - veh.x) > 22.0:
                continue
            if veh.approach in ("EAST", "WEST") and abs(other.y - veh.y) > 22.0:
                continue

            # Check if other is ahead of veh
            if veh.approach == "NORTH" and other.y > veh.y:
                dist = other.y - veh.y
            elif veh.approach == "SOUTH" and other.y < veh.y:
                dist = veh.y - other.y
            elif veh.approach == "WEST" and other.x > veh.x:
                dist = other.x - veh.x
            elif veh.approach == "EAST" and other.x < veh.x:
                dist = veh.x - other.x
            else:
                dist = -1.0

            if 0 < dist < min_dist:
                min_dist = dist
                lead = other

        return lead

    def _try_spawn_vehicle(self) -> None:
        """Attempt to spawn a vehicle on the next approach in queue if entrance is clear."""
        # Try approaches in rotation
        for _ in range(len(self.approach_queue)):
            approach = self.approach_queue[self._approach_idx]
            self._approach_idx = (self._approach_idx + 1) % len(self.approach_queue)

            # Determine spawn position
            lc = self.lane_coords[approach]
            if approach == "NORTH":
                spawn_x, spawn_y = lc["x"], -40.0
            elif approach == "SOUTH":
                spawn_x, spawn_y = lc["x"], 790.0
            elif approach == "WEST":
                spawn_x, spawn_y = -40.0, lc["y"]
            elif approach == "EAST":
                spawn_x, spawn_y = 1240.0, lc["y"]

            # Ensure entrance is clear (no vehicle within 95px of spawn point)
            clear = True
            for v in self.vehicles:
                if v.approach == approach:
                    dist = math.hypot(v.x - spawn_x, v.y - spawn_y)
                    if dist < 95.0:
                        clear = False
                        break

            if clear and approach == "SOUTH" and self.ambulance is not None:
                dist_amb = math.hypot(self.ambulance.x - spawn_x, self.ambulance.y - spawn_y)
                if dist_amb < 95.0:
                    clear = False

            if clear:
                color, _ = random.choice(CIVILIAN_PALETTE)
                # Ensure primary lead vehicle on SOUTH corridor is deterministically named C-01
                if approach == "SOUTH" and not any(v.vehicle_id == "C-01" for v in self.vehicles):
                    vid = "C-01"
                else:
                    vid = f"C{self._vehicle_counter:02d}"
                    self._vehicle_counter += 1
                veh = CivilianVehicle(
                    vehicle_id=vid,
                    approach=approach,
                    x=spawn_x,
                    y=spawn_y,
                    speed=random.uniform(90.0, 105.0),
                    target_speed=random.uniform(90.0, 105.0),
                    color=color,
                )
                self.vehicles.append(veh)
                return

    def _is_off_screen(self, v: CivilianVehicle) -> bool:
        """Check if vehicle has driven completely out of visible boundaries."""
        if v.approach == "NORTH" and v.y > 840:
            return True
        if v.approach == "SOUTH" and v.y < -80:
            return True
        if v.approach == "WEST" and v.x > 1290:
            return True
        if v.approach == "EAST" and v.x < -80:
            return True
        return False

    def reset(self) -> None:
        """Clear all vehicles and reset ambulance."""
        self.vehicles.clear()
        self.ambulance = None
        self._ambulance_spawned = False
        self.ambulance_cleared_events.clear()
        self._spawn_timer = 0.5
        self._vehicle_counter = 1
        self._v2v_hazard_injected = False
        self._pending_v2v_telemetry.clear()
        self.latest_ai_prediction = None
        if self.ai_predictor is not None:
            self.ai_predictor.reset()
        if self.safety_fusion is not None:
            self.safety_fusion.reset()
        self._last_ai_feed_time = -float("inf")
