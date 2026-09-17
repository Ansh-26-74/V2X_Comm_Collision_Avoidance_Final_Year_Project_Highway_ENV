"""Phase 1: SmartTrafficSignal — Coordinated 4-Way Traffic Signal & RSU Controller.

Owns:
  - 4-Way Coordinated Signal State Machine (North, South, East, West)
  - Strict conflict-free phase transitions:
      NS_GREEN_EW_RED -> NS_YELLOW_EW_RED -> NS_RED_EW_GREEN -> NS_RED_EW_YELLOW -> Repeat
  - Safety Invariant: North/South and East/West are NEVER simultaneously GREEN.
  - RSU metadata and broadcast message generation.

This module has NO dependency on any V2V code.
"""

from enum import Enum
import logging

from v2i.traffic_analyzer import RSUTrafficAnalyzer, EmergencyAnalysis

logger = logging.getLogger(__name__)


class SignalState(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class CyclePhase(str, Enum):
    NS_GREEN_EW_RED = "NS_GREEN_EW_RED"
    NS_YELLOW_EW_RED = "NS_YELLOW_EW_RED"
    NS_RED_EW_GREEN = "NS_RED_EW_GREEN"
    NS_RED_EW_YELLOW = "NS_RED_EW_YELLOW"
    # Phase 6: Emergency Preemption States
    PREEMPTION_YELLOW = "PREEMPTION_YELLOW"
    PREEMPTION_ALL_RED = "PREEMPTION_ALL_RED"
    EMERGENCY_SOUTH_GREEN = "EMERGENCY_SOUTH_GREEN"
    # Phase 8: Emergency Termination & Recovery States
    EMERGENCY_TERMINATING = "EMERGENCY_TERMINATING"
    RECOVERY_ALL_RED = "RECOVERY_ALL_RED"


# ─────────────────────────────────────────────────────────────────────────────
# Configurable Default Timings (seconds)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PHASE_TIMINGS = {
    CyclePhase.NS_GREEN_EW_RED: 8.0,
    CyclePhase.NS_YELLOW_EW_RED: 3.0,
    CyclePhase.NS_RED_EW_GREEN: 8.0,
    CyclePhase.NS_RED_EW_YELLOW: 3.0,
    # Phase 6 Timings
    CyclePhase.PREEMPTION_YELLOW: 3.0,
    CyclePhase.PREEMPTION_ALL_RED: 1.0,
    CyclePhase.EMERGENCY_SOUTH_GREEN: 8.0,
    # Phase 8 Timings
    CyclePhase.EMERGENCY_TERMINATING: 3.0,
    CyclePhase.RECOVERY_ALL_RED: 1.0,
}


class SmartTrafficSignal:
    """RSU + 4-Way Coordinated Smart Traffic Signal Controller.

    Controls traffic signals on all four approaches of a PLUS (+) intersection:
    - NORTH approach
    - SOUTH approach
    - EAST approach
    - WEST approach

    Guarantees that conflicting directions (North/South vs. East/West)
    are NEVER GREEN at the same time.
    """

    def __init__(
        self,
        intersection_id: str = "INT-01",
        intersection_x: float = 560.0,
        intersection_y: float = 415.0,
        phase_timings: dict | None = None,
    ):
        self.intersection_id = intersection_id
        self.intersection_x = intersection_x
        self.intersection_y = intersection_y

        # Phase timings
        self.phase_timings = dict(DEFAULT_PHASE_TIMINGS)
        if phase_timings:
            self.phase_timings.update(phase_timings)

        # State machine initialization
        self._current_phase = CyclePhase.NS_GREEN_EW_RED
        self._time_in_phase = 0.0
        self.cycle_count = 0

        # RSU state
        self.rsu_id = f"RSU-{intersection_id}"
        self.broadcasting = True
        self.preemption_active = False
        self.latest_emergency_request: dict | None = None
        self.emergency_request_received: bool = False

        # Phase 6: Emergency Preemption State
        self.preemption_requested: bool = False
        self.preemption_clearing: bool = False
        self.preemption_approach: str = "SOUTH"
        self._last_civilian_vehicles: list = []
        self._last_ambulance = None

        # Phase 8: Emergency Termination & Recovery State
        self.emergency_terminating: bool = False
        self.emergency_completed: bool = False

        # Phase 5: RSU Traffic & Conflict Analyzer
        self.traffic_analyzer = RSUTrafficAnalyzer(
            center_x=self.intersection_x,
            center_y=self.intersection_y,
        )
        self.latest_analysis: EmergencyAnalysis | None = None

        # Invariant check
        self._verify_safety_invariants()

    # ─────────────────────────────────────────────────────────────────────────
    # Public Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def current_phase(self) -> CyclePhase:
        return self._current_phase

    @current_phase.setter
    def current_phase(self, phase: CyclePhase) -> None:
        self._current_phase = phase
        self._time_in_phase = 0.0
        self._verify_safety_invariants()

    @property
    def time_in_phase(self) -> float:
        return self._time_in_phase

    @property
    def phase_duration(self) -> float:
        return self.phase_timings[self._current_phase]

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.phase_duration - self._time_in_phase)

    @property
    def rsu_priority_status(self) -> str:
        """Phase 8: Return the RSU emergency priority lifecycle status.

        Returns:
            - 'ACTIVE' while preemption is active or EMERGENCY_SOUTH_GREEN is running.
            - 'TERMINATING' during EMERGENCY_TERMINATING or RECOVERY_ALL_RED.
            - 'INACTIVE' during normal coordinated operation.
        """
        if self.preemption_active or self._current_phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
            return "ACTIVE"
        elif (
            self.emergency_terminating
            or self._current_phase in (CyclePhase.EMERGENCY_TERMINATING, CyclePhase.RECOVERY_ALL_RED)
        ):
            return "TERMINATING"
        else:
            return "INACTIVE"

    # ─────────────────────────────────────────────────────────────────────────
    # Signal Head Accessors
    # ─────────────────────────────────────────────────────────────────────────

    def get_signal(self, approach: str) -> SignalState:
        """Return the current signal state for a given approach.

        Parameters
        ----------
        approach : str
            One of "NORTH", "SOUTH", "EAST", "WEST".
        """
        app = approach.upper()

        if self._current_phase == CyclePhase.NS_GREEN_EW_RED:
            if app in ("NORTH", "SOUTH"):
                return SignalState.GREEN
            return SignalState.RED

        elif self._current_phase == CyclePhase.NS_YELLOW_EW_RED:
            if app in ("NORTH", "SOUTH"):
                return SignalState.YELLOW
            return SignalState.RED

        elif self._current_phase == CyclePhase.NS_RED_EW_GREEN:
            if app in ("EAST", "WEST"):
                return SignalState.GREEN
            return SignalState.RED

        elif self._current_phase == CyclePhase.NS_RED_EW_YELLOW:
            if app in ("EAST", "WEST"):
                return SignalState.YELLOW
            return SignalState.RED

        elif self._current_phase == CyclePhase.PREEMPTION_YELLOW:
            if app in ("EAST", "WEST"):
                return SignalState.YELLOW
            return SignalState.RED

        elif self._current_phase == CyclePhase.PREEMPTION_ALL_RED:
            return SignalState.RED

        elif self._current_phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
            if app in ("NORTH", "SOUTH"):
                return SignalState.GREEN
            return SignalState.RED

        # Phase 8: Emergency Priority Termination & Recovery
        elif self._current_phase == CyclePhase.EMERGENCY_TERMINATING:
            if app in ("NORTH", "SOUTH"):
                return SignalState.YELLOW
            return SignalState.RED

        elif self._current_phase == CyclePhase.RECOVERY_ALL_RED:
            return SignalState.RED

        # Failsafe default: all RED
        return SignalState.RED

    def get_all_signals(self) -> dict[str, str]:
        """Return signal states for all 4 approaches as a dictionary."""
        return {
            "NORTH": self.get_signal("NORTH").value,
            "SOUTH": self.get_signal("SOUTH").value,
            "EAST": self.get_signal("EAST").value,
            "WEST": self.get_signal("WEST").value,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Preemption Control (Phase 6)
    # ─────────────────────────────────────────────────────────────────────────

    def request_preemption(self, approach: str = "SOUTH") -> bool:
        """Request emergency signal preemption for the specified approach.

        Initiates a safe transition sequence through yellow and all-red clearance
        intervals before granting green. Never instantly transitions to green.
        """
        self.preemption_requested = True
        self.preemption_approach = approach.upper()

        # If already in emergency green, keep it active
        if self._current_phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
            return True

        # If already in yellow or all-red clearance, let it proceed
        if self._current_phase in (CyclePhase.PREEMPTION_YELLOW, CyclePhase.PREEMPTION_ALL_RED):
            return True

        # If currently in NS_GREEN_EW_RED, South already has green!
        # Transition immediately to EMERGENCY_SOUTH_GREEN to hold green without artificial red
        if self._current_phase == CyclePhase.NS_GREEN_EW_RED:
            logger.info(f"[{self.rsu_id}] Preemption requested while NS GREEN: Extending green for emergency vehicle")
            self._current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
            self._time_in_phase = 0.0
            self.preemption_active = True
            self.preemption_clearing = False
            return True

        # If currently in NS_RED_EW_GREEN, conflicting EW traffic must first clear via YELLOW
        if self._current_phase == CyclePhase.NS_RED_EW_GREEN:
            logger.info(f"[{self.rsu_id}] Preemption requested while EW GREEN: Initiating PREEMPTION_YELLOW clearance")
            self._current_phase = CyclePhase.PREEMPTION_YELLOW
            self._time_in_phase = 0.0
            self.preemption_clearing = True
            self.preemption_active = False
            return True

        # If currently in NS_RED_EW_YELLOW, conflicting EW traffic is already in yellow
        if self._current_phase == CyclePhase.NS_RED_EW_YELLOW:
            logger.info(f"[{self.rsu_id}] Preemption requested while EW YELLOW: Holding in PREEMPTION_YELLOW clearance")
            self._current_phase = CyclePhase.PREEMPTION_YELLOW
            self.preemption_clearing = True
            self.preemption_active = False
            return True

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Emergency Termination & Normal Recovery (Phase 8)
    # ─────────────────────────────────────────────────────────────────────────

    def notify_ambulance_cleared(self, vehicle_id: str = "AMB-01") -> bool:
        """Safely terminate emergency priority after AMB-01 has completely cleared.

        Authoritative trigger: Phase 7 'AMB-01 CLEARED INTERSECTION' event.
        Controlled recovery sequence:
            EMERGENCY_SOUTH_GREEN -> EMERGENCY_TERMINATING (South Yellow) -> RECOVERY_ALL_RED -> Normal Cycle (NS_RED_EW_GREEN)

        Idempotence: If called multiple times, duplicate calls are safely ignored without
        restarting timers or corrupting signal state.
        """
        # Defensive check: only trigger if in EMERGENCY_SOUTH_GREEN and not already terminating
        if self._current_phase != CyclePhase.EMERGENCY_SOUTH_GREEN or self.emergency_terminating:
            logger.debug(
                f"[{self.rsu_id}] Clearance notification ignored: current_phase={self._current_phase.value}, "
                f"terminating={self.emergency_terminating}"
            )
            return False

        logger.info(
            f"[{self.rsu_id}] Emergency clearance event received for {vehicle_id}. "
            f"Initiating controlled termination: EMERGENCY_SOUTH_GREEN -> EMERGENCY_TERMINATING (YELLOW)"
        )
        self.emergency_terminating = True
        self.emergency_completed = True
        self.preemption_active = False
        self.preemption_requested = False
        self.preemption_clearing = True

        # Retire / consume the emergency request buffer
        if self.latest_emergency_request is not None:
            self.latest_emergency_request["consumed"] = True
            self.latest_emergency_request["active"] = False

        # Transition immediately to controlled yellow termination
        self._current_phase = CyclePhase.EMERGENCY_TERMINATING
        self._time_in_phase = 0.0
        self._verify_safety_invariants()
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # State Machine Update
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """Advance the signal controller state machine by dt seconds.

        Parameters
        ----------
        dt : float
            Simulation time-step in seconds.
        """
        self._time_in_phase += dt
        duration = self.phase_timings[self._current_phase]

        if self._current_phase == CyclePhase.PREEMPTION_ALL_RED:
            # Must satisfy both: duration >= 1.0s clearance AND safety gate confirms clear
            if self._time_in_phase >= duration:
                # Query safety gate
                safe, reason = self.traffic_analyzer.is_safe_for_emergency_green(
                    self._last_civilian_vehicles,
                    self._last_ambulance,
                )
                if safe:
                    logger.info(f"[{self.rsu_id}] Preemption Safety Gate CLEARED: Granting EMERGENCY_SOUTH_GREEN")
                    self._current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
                    self._time_in_phase = 0.0
                    self.preemption_active = True
                    self.preemption_clearing = False
                else:
                    # Hold in all-red clearance until safe
                    self._time_in_phase = duration
                    logger.debug(f"[{self.rsu_id}] Preemption held in ALL_RED: {reason}")
        elif self._time_in_phase >= duration:
            self._time_in_phase = 0.0
            self._transition_to_next_phase()

        self._verify_safety_invariants()

    def _transition_to_next_phase(self) -> None:
        """Execute explicit coordinated phase sequence."""
        if self._current_phase == CyclePhase.NS_GREEN_EW_RED:
            self._current_phase = CyclePhase.NS_YELLOW_EW_RED
            logger.info(f"[{self.rsu_id}] Phase transition: NS GREEN -> NS YELLOW (E/W RED)")

        elif self._current_phase == CyclePhase.NS_YELLOW_EW_RED:
            if self.preemption_requested:
                self._current_phase = CyclePhase.PREEMPTION_ALL_RED
                self.preemption_clearing = True
                logger.info(f"[{self.rsu_id}] Preemption transition: NS YELLOW -> PREEMPTION_ALL_RED")
            else:
                self._current_phase = CyclePhase.NS_RED_EW_GREEN
                logger.info(f"[{self.rsu_id}] Phase transition: NS RED, EW -> GREEN")

        elif self._current_phase == CyclePhase.NS_RED_EW_GREEN:
            self._current_phase = CyclePhase.NS_RED_EW_YELLOW
            logger.info(f"[{self.rsu_id}] Phase transition: EW GREEN -> EW YELLOW (N/S RED)")

        elif self._current_phase == CyclePhase.NS_RED_EW_YELLOW:
            if self.preemption_requested:
                self._current_phase = CyclePhase.PREEMPTION_ALL_RED
                self.preemption_clearing = True
                logger.info(f"[{self.rsu_id}] Preemption transition: EW YELLOW -> PREEMPTION_ALL_RED")
            else:
                self._current_phase = CyclePhase.NS_GREEN_EW_RED
                self.cycle_count += 1
                logger.info(f"[{self.rsu_id}] Phase transition: EW RED, NS -> GREEN (Cycle #{self.cycle_count} completed)")

        elif self._current_phase == CyclePhase.PREEMPTION_YELLOW:
            self._current_phase = CyclePhase.PREEMPTION_ALL_RED
            self.preemption_clearing = True
            logger.info(f"[{self.rsu_id}] Preemption transition: PREEMPTION_YELLOW -> PREEMPTION_ALL_RED (All approaches RED)")

        elif self._current_phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
            self._current_phase = CyclePhase.NS_YELLOW_EW_RED
            self.preemption_active = False
            self.preemption_requested = False
            self.preemption_clearing = False
            logger.info(f"[{self.rsu_id}] Preemption complete: EMERGENCY_SOUTH_GREEN -> NS_YELLOW (Recovering normal cycle)")

        # Phase 8: Termination & Recovery transitions
        elif self._current_phase == CyclePhase.EMERGENCY_TERMINATING:
            self._current_phase = CyclePhase.RECOVERY_ALL_RED
            self.preemption_clearing = True
            logger.info(f"[{self.rsu_id}] Termination transition: EMERGENCY_TERMINATING -> RECOVERY_ALL_RED (All approaches RED)")

        elif self._current_phase == CyclePhase.RECOVERY_ALL_RED:
            self._current_phase = CyclePhase.NS_RED_EW_GREEN
            self.emergency_terminating = False
            self.preemption_clearing = False
            self.preemption_active = False
            self.preemption_requested = False
            logger.info(f"[{self.rsu_id}] Normal cycle recovered: RECOVERY_ALL_RED -> NS_RED_EW_GREEN (Normal coordinated cycle resumed)")

    def reset(self) -> None:
        """Reset the controller to the initial normal cycle phase."""
        self._current_phase = CyclePhase.NS_GREEN_EW_RED
        self._time_in_phase = 0.0
        self.cycle_count = 0
        self.preemption_active = False
        self.preemption_requested = False
        self.preemption_clearing = False
        self.preemption_approach = "SOUTH"
        self.emergency_terminating = False
        self.emergency_completed = False
        self._last_civilian_vehicles = []
        self._last_ambulance = None
        self.latest_emergency_request = None
        self.emergency_request_received = False
        self.traffic_analyzer.reset()
        self.latest_analysis = None
        self._verify_safety_invariants()

    # ─────────────────────────────────────────────────────────────────────────
    # Safety Invariant Verification
    # ─────────────────────────────────────────────────────────────────────────

    def _verify_safety_invariants(self) -> None:
        """Strictly assert that conflicting green signals can NEVER co-exist."""
        ns_green = (
            self.get_signal("NORTH") == SignalState.GREEN
            or self.get_signal("SOUTH") == SignalState.GREEN
        )
        ew_green = (
            self.get_signal("EAST") == SignalState.GREEN
            or self.get_signal("WEST") == SignalState.GREEN
        )
        if ns_green and ew_green:
            raise RuntimeError(
                "CRITICAL SAFETY VIOLATION: North/South and East/West signals are GREEN simultaneously!"
            )
        if self._current_phase in (CyclePhase.PREEMPTION_ALL_RED, CyclePhase.RECOVERY_ALL_RED):
            for app in ("NORTH", "SOUTH", "EAST", "WEST"):
                if self.get_signal(app) != SignalState.RED:
                    raise RuntimeError(
                        f"CRITICAL SAFETY VIOLATION: In {self._current_phase.value}, approach {app} is not RED!"
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # V2I Message Generation, Reception & Continuous Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def generate_message(self) -> dict:
        """Create a V2I infrastructure broadcast message dict."""
        return {
            "type": "V2I_SIGNAL_STATUS",
            "rsuId": self.rsu_id,
            "intersectionId": self.intersection_id,
            "phase": self._current_phase.value,
            "timeRemaining": round(self.time_remaining, 1),
            "signals": self.get_all_signals(),
            "preemptionActive": self.preemption_active,
            "preemptionRequested": self.preemption_requested,
            "preemptionClearing": self.preemption_clearing,
            "rsuPriorityStatus": self.rsu_priority_status,
        }

    def receive_emergency_request(
        self,
        request_msg: dict,
        current_time: float = 0.0,
        auto_preempt: bool = False,
    ) -> bool:
        """Process, validate, and buffer incoming emergency request from AMB-01.

        Parameters
        ----------
        request_msg : dict
            Emergency request message dictionary.
        current_time : float
            Current simulation timestamp.
        auto_preempt : bool
            If True, automatically triggers safe signal preemption upon valid reception.
        """
        # Phase 8: If emergency priority has completed for this vehicle, ignore redundant request
        if self.emergency_completed and request_msg.get("vehicleId") == "AMB-01":
            logger.debug(
                f"[{self.rsu_id}] EMERGENCY_REQUEST from {request_msg.get('vehicleId')} ignored: "
                f"emergency priority already completed/retired"
            )
            return False
        # Validate request via Phase 5 analyzer
        msg_time = current_time if current_time > 0.0 else float(request_msg.get("timestamp", 0.0))
        is_valid, reason = self.traffic_analyzer.validate_emergency_request(request_msg, msg_time)

        if not is_valid:
            logger.warning(f"[{self.rsu_id}] EMERGENCY_REQUEST rejected: {reason}")
            return False

        self.latest_emergency_request = dict(request_msg)
        self.emergency_request_received = True
        logger.info(
            f"[{self.rsu_id}] EMERGENCY_REQUEST accepted from {request_msg.get('vehicleId')} "
            f"on {request_msg.get('approach')} approach | ETA: {request_msg.get('eta')}s | "
            f"Priority: {request_msg.get('priority')}"
        )

        if auto_preempt:
            self.request_preemption(approach=request_msg.get("approach", "SOUTH"))

        return True

    def update_traffic_analysis(
        self,
        civilian_vehicles: list,
        ambulance=None,
        current_time: float = 0.0,
    ) -> EmergencyAnalysis | None:
        """Continuously perform intelligent intersection conflict analysis."""
        self._last_civilian_vehicles = list(civilian_vehicles)
        self._last_ambulance = ambulance

        if not self.latest_emergency_request:
            self.latest_analysis = None
            return None

        analysis = self.traffic_analyzer.analyze_traffic(
            request=self.latest_emergency_request,
            civilian_vehicles=civilian_vehicles,
            ambulance=ambulance,
            current_time=current_time,
        )
        self.latest_analysis = analysis
        return analysis
