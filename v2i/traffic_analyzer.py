"""Phase 5: RSU Traffic & Conflict Analysis System.

Provides intelligent, continuous traffic state and conflict zone analysis
for the Roadside Unit (RSU) following the reception of a valid V2I
EMERGENCY_REQUEST from AMB-01.

Key capabilities:
  - Emergency request validation (schema, non-empty fields, staleness detection).
  - Trajectory & intended path deduction (e.g. SOUTH -> NORTH).
  - Conflict movement mapping (e.g. conflicting approaches: EAST <-> WEST).
  - Geometric conflict zone tracking and civilian vehicle state classification:
      SAFE, APPROACHING_CONFLICT, IN_CONFLICT_ZONE, CLEARED.
  - Evaluation of preemption readiness (safe_for_future_preemption) based on
    actual real-time vehicle positions.
  - Zero-signal-control constraint: Phase 5 is an ANALYSIS ONLY engine.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Geometry & Threshold Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MAX_STALE_TIME = 5.0  # seconds — request is rejected if older than this
CONFLICT_ZONE_MARGIN = 8.0     # pixels expansion around intersection box
APPROACH_DISTANCE_THRESHOLD = 140.0  # pixels — distance from stop line considered approaching conflict


class VehicleConflictStatus(str, Enum):
    """Classification of a civilian vehicle's relationship to the emergency corridor."""
    SAFE = "SAFE"
    APPROACHING_CONFLICT = "APPROACHING_CONFLICT"
    IN_CONFLICT_ZONE = "IN_CONFLICT_ZONE"
    CLEARED = "CLEARED"


@dataclass
class EmergencyAnalysis:
    """Structured result of RSU traffic and conflict zone analysis."""
    emergency_vehicle_id: str
    approach: str
    intended_direction: str
    conflicting_approaches: list[str]
    vehicles_in_conflict: list[str] = field(default_factory=list)
    vehicles_approaching_conflict: list[str] = field(default_factory=list)
    vehicles_cleared: list[str] = field(default_factory=list)
    intersection_occupied: bool = False
    conflict_detected: bool = False
    safe_for_future_preemption: bool = False
    timestamp: float = 0.0
    status_text: str = "INITIALIZING"

    def summary_dict(self) -> dict:
        """Return a serializable dictionary representation."""
        return {
            "emergency_vehicle_id": self.emergency_vehicle_id,
            "approach": self.approach,
            "intended_direction": self.intended_direction,
            "conflicting_approaches": self.conflicting_approaches,
            "in_conflict_count": len(self.vehicles_in_conflict),
            "approaching_count": len(self.vehicles_approaching_conflict),
            "intersection_occupied": self.intersection_occupied,
            "conflict_detected": self.conflict_detected,
            "safe_for_future_preemption": self.safe_for_future_preemption,
            "status": self.status_text,
            "timestamp": round(self.timestamp, 2),
        }


class RSUTrafficAnalyzer:
    """Continuously evaluates intersection safety and traffic conflicts for RSU-01."""

    OPPOSITE_APPROACH = {
        "SOUTH": "NORTH",
        "NORTH": "SOUTH",
        "WEST":  "EAST",
        "EAST":  "WEST",
    }

    CONFLICTING_MAP = {
        "SOUTH": ["EAST", "WEST"],
        "NORTH": ["EAST", "WEST"],
        "EAST":  ["NORTH", "SOUTH"],
        "WEST":  ["NORTH", "SOUTH"],
    }

    def __init__(
        self,
        center_x: float = 560.0,
        center_y: float = 415.0,
        road_width: float = 130.0,
        max_stale_time: float = DEFAULT_MAX_STALE_TIME,
    ):
        self.cx = float(center_x)
        self.cy = float(center_y)
        self.road_w = float(road_width)
        self.half_road = self.road_w / 2.0
        self.max_stale_time = float(max_stale_time)

        # Conflict zone bounding box (central intersection box + margin)
        self.cz_x_min = self.cx - self.half_road - CONFLICT_ZONE_MARGIN
        self.cz_x_max = self.cx + self.half_road + CONFLICT_ZONE_MARGIN
        self.cz_y_min = self.cy - self.half_road - CONFLICT_ZONE_MARGIN
        self.cz_y_max = self.cy + self.half_road + CONFLICT_ZONE_MARGIN

        # State tracking for clean edge-triggered transition logging
        self._last_state_signature: str = ""
        self._last_transition_event: str | None = None
        self._analysis_active: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Emergency Request Validation
    # ─────────────────────────────────────────────────────────────────────────

    def validate_emergency_request(
        self,
        request: dict | None,
        current_time: float,
    ) -> tuple[bool, str]:
        """Validate that incoming emergency request conforms to required schema and freshness.

        Returns (is_valid, reason_str).
        """
        if not isinstance(request, dict):
            return False, "REJECTED_NOT_A_DICT"

        if request.get("type") != "EMERGENCY_REQUEST":
            return False, "REJECTED_INVALID_TYPE"

        vid = request.get("vehicleId")
        if not vid or not isinstance(vid, str):
            return False, "REJECTED_MISSING_VEHICLE_ID"

        vtype = request.get("vehicleType")
        if vtype not in ("EMERGENCY", "AMBULANCE"):
            return False, "REJECTED_INVALID_VEHICLE_TYPE"

        if request.get("emergency") is not True:
            return False, "REJECTED_NOT_FLAGGED_EMERGENCY"

        approach = request.get("approach")
        if approach not in ("NORTH", "SOUTH", "EAST", "WEST"):
            return False, "REJECTED_INVALID_APPROACH"

        pos = request.get("position")
        if not pos or not (isinstance(pos, (tuple, list)) and len(pos) == 2):
            return False, "REJECTED_INVALID_POSITION"

        ts = request.get("timestamp")
        if ts is None or not isinstance(ts, (int, float)):
            return False, "REJECTED_MISSING_TIMESTAMP"

        # Staleness check: reject if message timestamp is too far behind current simulation time
        age = current_time - float(ts)
        if age > self.max_stale_time:
            return False, f"REJECTED_STALE_REQUEST (age={age:.1f}s > {self.max_stale_time}s)"

        return True, "VALID_EMERGENCY_REQUEST"

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Movement & Conflicting Approach Derivation
    # ─────────────────────────────────────────────────────────────────────────

    def determine_path_and_conflicts(self, approach: str) -> tuple[str, list[str]]:
        """Determine intended destination direction and list of conflicting approaches."""
        app = approach.upper()
        intended = self.OPPOSITE_APPROACH.get(app, "UNKNOWN")
        conflicts = list(self.CONFLICTING_MAP.get(app, []))
        return intended, conflicts

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Geometric Conflict Zone Verification
    # ─────────────────────────────────────────────────────────────────────────

    def is_in_conflict_zone(self, x: float, y: float) -> bool:
        """Check whether physical (x, y) coordinates fall within the intersection conflict zone."""
        return (self.cz_x_min <= x <= self.cz_x_max) and (self.cz_y_min <= y <= self.cz_y_max)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Civilian Vehicle Classification
    # ─────────────────────────────────────────────────────────────────────────

    def classify_vehicle(
        self,
        vehicle,
        conflicting_approaches: list[str],
    ) -> VehicleConflictStatus:
        """Classify a vehicle relative to the ambulance's intersection corridor."""
        # Non-conflicting approaches (e.g. North/South when ambulance is South)
        if vehicle.approach not in conflicting_approaches:
            # Check if physically blocking intersection
            if getattr(vehicle, "in_intersection", False) or self.is_in_conflict_zone(vehicle.x, vehicle.y):
                return VehicleConflictStatus.IN_CONFLICT_ZONE
            return VehicleConflictStatus.SAFE

        # Conflicting approach (e.g. EAST or WEST)
        # 1. Check if physically inside conflict zone
        if getattr(vehicle, "in_intersection", False) or self.is_in_conflict_zone(vehicle.x, vehicle.y):
            return VehicleConflictStatus.IN_CONFLICT_ZONE

        # 2. Check if already cleared through
        if vehicle.approach == "EAST":
            # East travels West (-X). Entrance is cx + half_road (~625), Exit is cx - half_road (~495)
            if vehicle.x < self.cx - self.half_road:
                return VehicleConflictStatus.CLEARED
            # Approaching if east of entrance and moving towards it
            dist_to_box = vehicle.x - (self.cx + self.half_road)
            if 0.0 <= dist_to_box <= APPROACH_DISTANCE_THRESHOLD:
                return VehicleConflictStatus.APPROACHING_CONFLICT

        elif vehicle.approach == "WEST":
            # West travels East (+X). Entrance is cx - half_road (~495), Exit is cx + half_road (~625)
            if vehicle.x > self.cx + self.half_road:
                return VehicleConflictStatus.CLEARED
            # Approaching if west of entrance and moving towards it
            dist_to_box = (self.cx - self.half_road) - vehicle.x
            if 0.0 <= dist_to_box <= APPROACH_DISTANCE_THRESHOLD:
                return VehicleConflictStatus.APPROACHING_CONFLICT

        elif vehicle.approach == "NORTH":
            if vehicle.y > self.cy + self.half_road:
                return VehicleConflictStatus.CLEARED
            dist_to_box = (self.cy - self.half_road) - vehicle.y
            if 0.0 <= dist_to_box <= APPROACH_DISTANCE_THRESHOLD:
                return VehicleConflictStatus.APPROACHING_CONFLICT

        elif vehicle.approach == "SOUTH":
            if vehicle.y < self.cy - self.half_road:
                return VehicleConflictStatus.CLEARED
            dist_to_box = vehicle.y - (self.cy + self.half_road)
            if 0.0 <= dist_to_box <= APPROACH_DISTANCE_THRESHOLD:
                return VehicleConflictStatus.APPROACHING_CONFLICT

        return VehicleConflictStatus.SAFE

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Continuous Traffic & Conflict Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_traffic(
        self,
        request: dict | None,
        civilian_vehicles: list,
        ambulance=None,
        current_time: float = 0.0,
    ) -> EmergencyAnalysis | None:
        """Continuously analyze intersection state and produce an EmergencyAnalysis report."""
        valid, reason = self.validate_emergency_request(request, current_time)
        if not valid:
            self._analysis_active = False
            return None

        # Request is valid
        self._analysis_active = True
        approach = request["approach"]
        intended_dir, conflicting_apps = self.determine_path_and_conflicts(approach)

        in_conflict: list[str] = []
        approaching_conflict: list[str] = []
        cleared_vehicles: list[str] = []

        # Analyze each civilian vehicle
        for v in civilian_vehicles:
            classification = self.classify_vehicle(v, conflicting_apps)
            # Store classification on vehicle instance for visual highlight rendering
            v.conflict_status = classification.value

            if classification == VehicleConflictStatus.IN_CONFLICT_ZONE:
                in_conflict.append(v.vehicle_id)
            elif classification == VehicleConflictStatus.APPROACHING_CONFLICT:
                approaching_conflict.append(v.vehicle_id)
            elif classification == VehicleConflictStatus.CLEARED:
                cleared_vehicles.append(v.vehicle_id)

        # Intersection is occupied if any vehicle is in the central conflict zone
        intersection_occupied = (len(in_conflict) > 0)
        conflict_detected = (len(in_conflict) > 0 or len(approaching_conflict) > 0)

        # Preemption readiness for Phase 6:
        # Safe for future preemption when the central conflict zone is completely empty
        safe_for_preemption = (len(in_conflict) == 0)

        if intersection_occupied:
            status_text = f"CONFLICT DETECTED ({len(in_conflict)} VEHICLE IN ZONE)"
        elif len(approaching_conflict) > 0:
            status_text = f"APPROACHING CONFLICT ({len(approaching_conflict)} VEHICLES APPROACHING)"
        else:
            status_text = "INTERSECTION CLEAR (READY FOR PREEMPTION)"

        analysis = EmergencyAnalysis(
            emergency_vehicle_id=request.get("vehicleId", "AMB-01"),
            approach=approach,
            intended_direction=intended_dir,
            conflicting_approaches=conflicting_apps,
            vehicles_in_conflict=in_conflict,
            vehicles_approaching_conflict=approaching_conflict,
            vehicles_cleared=cleared_vehicles,
            intersection_occupied=intersection_occupied,
            conflict_detected=conflict_detected,
            safe_for_future_preemption=safe_for_preemption,
            timestamp=current_time,
            status_text=status_text,
        )

        # Edge-triggered transition detection for logging
        self._check_and_record_transitions(analysis)

        return analysis

    def _check_and_record_transitions(self, analysis: EmergencyAnalysis) -> None:
        """Detect and log state transitions without flooding console."""
        sig = f"{analysis.conflict_detected}-{analysis.intersection_occupied}-{len(analysis.vehicles_in_conflict)}"
        if sig != self._last_state_signature:
            self._last_state_signature = sig

            if not self._last_transition_event:
                event = f"RSU ANALYSIS STARTED — {analysis.emergency_vehicle_id} ({analysis.approach} -> {analysis.intended_direction})"
            elif analysis.intersection_occupied:
                event = f"RSU CONFLICT DETECTED: {len(analysis.vehicles_in_conflict)} vehicle(s) in conflict zone ({analysis.vehicles_in_conflict})"
            elif analysis.conflict_detected:
                event = f"RSU CONFLICT CLEARED FROM ZONE: {len(analysis.vehicles_approaching_conflict)} approaching"
            else:
                event = "RSU INTERSECTION CLEAR — READY FOR FUTURE PREEMPTION"

            self._last_transition_event = event
            logger.info(f"[RSU-ANALYSIS] {event}")

    def pop_last_transition_event(self) -> str | None:
        """Return and clear the latest transition event string, if any."""
        ev = self._last_transition_event
        self._last_transition_event = None
        return ev

    def is_safe_for_emergency_green(
        self,
        civilian_vehicles: list,
        ambulance=None,
        conflicting_approaches: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Safety Gate before granting EMERGENCY_SOUTH_GREEN.

        Verifies:
          1. Central conflict zone is completely empty (zero vehicles inside).
          2. No conflicting vehicle (EAST/WEST) is breaching or within critical entry distance at speed.
          3. Conflicting vehicles are either safely stopped behind stop line or fully cleared.

        Returns (is_safe, reason_string).
        """
        conflicts = conflicting_approaches if conflicting_approaches is not None else ["EAST", "WEST"]

        for v in civilian_vehicles:
            # 1. Any vehicle physically inside the conflict zone blocks preemption immediately
            if getattr(v, "in_intersection", False) or self.is_in_conflict_zone(v.x, v.y):
                return False, f"CONFLICT_ZONE_OCCUPIED ({v.vehicle_id})"

            # 2. Check vehicles on conflicting approaches (EAST / WEST)
            if v.approach in conflicts:
                if v.approach == "EAST":
                    # East travels West (-X). Stop line is cx + stop_line_dist (~635)
                    # If vehicle is between stop line and exit, it has crossed stop line but not exited
                    if (self.cx - self.half_road) <= v.x <= (self.cx + self.half_road + 15.0):
                        return False, f"CONFLICT_ZONE_NOT_CLEARED ({v.vehicle_id})"
                    # If vehicle is approaching close to stop line at significant speed
                    dist_to_stop = v.x - (self.cx + self.half_road)
                    if 0.0 < dist_to_stop < 50.0 and v.speed > 15.0 and not v.braking:
                        return False, f"VEHICLE_APPROACHING_AT_SPEED ({v.vehicle_id})"

                elif v.approach == "WEST":
                    # West travels East (+X). Stop line is cx - stop_line_dist (~485)
                    if (self.cx - self.half_road - 15.0) <= v.x <= (self.cx + self.half_road):
                        return False, f"CONFLICT_ZONE_NOT_CLEARED ({v.vehicle_id})"
                    dist_to_stop = (self.cx - self.half_road) - v.x
                    if 0.0 < dist_to_stop < 50.0 and v.speed > 15.0 and not v.braking:
                        return False, f"VEHICLE_APPROACHING_AT_SPEED ({v.vehicle_id})"

        return True, "CLEARANCE_SAFE"

    def reset(self) -> None:
        """Reset analyzer state."""
        self._last_state_signature = ""
        self._last_transition_event = None
        self._analysis_active = False
