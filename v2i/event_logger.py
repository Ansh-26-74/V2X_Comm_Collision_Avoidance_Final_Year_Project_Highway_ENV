"""Phase 9: V2I Event Logger & Observability System.

Provides:
  - Structured event abstraction (SimulationEvent) with timestamp, category, and message
  - Automatic event deduplication (only real state transitions generate timeline entries)
  - Bounded chronological event retention (FIFO queue of recent events)
  - Observer system tracking signal transitions, V2I communication, and ambulance progression
  - Reset & pause safety without phantom events
"""

from dataclasses import dataclass
from enum import Enum

from v2i.smart_signal import CyclePhase


class EventCategory(str, Enum):
    """Categorization for visualization color-coding and filtering."""
    COMMUNICATION = "COMMUNICATION"
    TRAFFIC = "TRAFFIC"
    SIGNAL = "SIGNAL"
    EMERGENCY = "EMERGENCY"
    SAFETY = "SAFETY"
    SYSTEM = "SYSTEM"


@dataclass
class SimulationEvent:
    """Represents a single discrete system transition in the V2I Smart Intersection."""
    timestamp: float
    category: EventCategory
    event_type: str
    message: str

    def formatted_str(self) -> str:
        """Return formatted string for timeline display."""
        return f"{self.timestamp:05.1f}s [{self.category.value[:6]}] {self.message}"


class EventLogger:
    """Bounded, deduplicated event timeline manager for live demonstration observability."""

    def __init__(self, max_events: int = 14):
        self.max_events = max_events
        self.events: list[SimulationEvent] = []
        self._last_event_signature: str | None = None

        # Observer tracking states
        self._last_phase: CyclePhase | None = None
        self._last_comm_state: str | None = None
        self._last_amb_overtaking: bool = False
        self._last_amb_in_intersection: bool = False
        self._last_amb_cleared: bool = False
        self._last_conflict_occupied: bool | None = None
        self._delivered_packet_count: int = 0
        self._dropped_packet_count: int = 0

    def add_event(
        self,
        timestamp: float,
        category: EventCategory,
        event_type: str,
        message: str,
    ) -> bool:
        """Add a transition event with strict deduplication against the immediate predecessor.

        Returns True if added, False if duplicate.
        """
        sig = f"{category.value}:{event_type}:{message}"
        if sig == self._last_event_signature:
            return False

        self._last_event_signature = sig
        evt = SimulationEvent(
            timestamp=round(float(timestamp), 2),
            category=category,
            event_type=event_type,
            message=message,
        )
        self.events.append(evt)

        # Enforce bounded retention
        if len(self.events) > self.max_events:
            self.events.pop(0)

        return True

    # Convenience alias
    log_event = add_event

    def get_events(self) -> list[SimulationEvent]:
        """Return chronological list of retained events."""
        return list(self.events)

    def clear(self) -> None:
        """Clear all events and reset observer tracking state."""
        self.events.clear()
        self._last_event_signature = None
        self._last_phase = None
        self._last_comm_state = None
        self._last_amb_overtaking = False
        self._last_amb_in_intersection = False
        self._last_amb_cleared = False
        self._last_conflict_occupied = None
        self._delivered_packet_count = 0
        self._dropped_packet_count = 0

    def observe_system(
        self,
        sim_time: float,
        signal_controller,
        traffic_manager=None,
        v2i_channel=None,
    ) -> list[SimulationEvent]:
        """Inspect active simulation components and emit events on verified state transitions.

        Returns list of newly added events this step.
        """
        new_events: list[SimulationEvent] = []

        # 1. Signal Phase Transitions
        if signal_controller is not None:
            curr_phase = signal_controller.current_phase
            if curr_phase != self._last_phase:
                self._last_phase = curr_phase
                msg, cat = self._format_phase_event(curr_phase)
                if self.add_event(sim_time, cat, curr_phase.value, msg):
                    new_events.append(self.events[-1])

        # 2. V2I Wireless Communication Events
        if v2i_channel is not None:
            # Range transitions
            curr_comm = v2i_channel.comm_state
            if curr_comm != self._last_comm_state:
                if curr_comm in ("IN_RANGE", "TRANSMITTED") and self._last_comm_state == "OUT_OF_RANGE":
                    if self.add_event(sim_time, EventCategory.COMMUNICATION, "AMB_IN_RANGE", "AMB-01 entered RSU DSRC coverage range (400px)"):
                        new_events.append(self.events[-1])
                elif curr_comm == "OUT_OF_RANGE" and self._last_comm_state is not None:
                    if self.add_event(sim_time, EventCategory.COMMUNICATION, "AMB_OUT_OF_RANGE", "AMB-01 exited RSU communication range"):
                        new_events.append(self.events[-1])
                self._last_comm_state = curr_comm

            # Delivered packets
            if v2i_channel.total_delivered > self._delivered_packet_count:
                diff = v2i_channel.total_delivered - self._delivered_packet_count
                self._delivered_packet_count = v2i_channel.total_delivered
                msg = f"EMERGENCY_REQUEST delivered to RSU-01 (latency: {int(v2i_channel.latency*1000)}ms)"
                if self.add_event(sim_time, EventCategory.COMMUNICATION, "PACKET_DELIVERED", msg):
                    new_events.append(self.events[-1])

            # Dropped packets
            if v2i_channel.total_dropped > self._dropped_packet_count:
                self._dropped_packet_count = v2i_channel.total_dropped
                if self.add_event(sim_time, EventCategory.COMMUNICATION, "PACKET_DROPPED", "V2I Packet Dropped (simulated channel loss)"):
                    new_events.append(self.events[-1])

        # 3. RSU Traffic Analysis Transitions
        if signal_controller is not None and getattr(signal_controller, "latest_analysis", None) is not None:
            analysis = signal_controller.latest_analysis
            occ = analysis.intersection_occupied
            if occ != self._last_conflict_occupied:
                self._last_conflict_occupied = occ
                if occ:
                    cnt = len(analysis.vehicles_in_conflict)
                    msg = f"RSU CONFLICT: {cnt} vehicle(s) in central junction conflict zone"
                    if self.add_event(sim_time, EventCategory.SAFETY, "CONFLICT_DETECTED", msg):
                        new_events.append(self.events[-1])
                else:
                    msg = "RSU ANALYSIS: Conflict zone CLEAR — Safe for preemption"
                    if self.add_event(sim_time, EventCategory.SAFETY, "CONFLICT_CLEARED", msg):
                        new_events.append(self.events[-1])

        # 4. Ambulance Navigation Milestones
        if traffic_manager is not None and traffic_manager.ambulance is not None:
            amb = traffic_manager.ambulance

            # Overtaking
            is_ot = getattr(amb, "is_overtaking", False)
            if is_ot and not self._last_amb_overtaking:
                self._last_amb_overtaking = True
                msg = f"AMB-01 OVERTAKING: Obstacle avoided -> Changing to Lane 2 (x={amb.target_lane_x:.0f})"
                if self.add_event(sim_time, EventCategory.TRAFFIC, "AMB_OVERTAKING", msg):
                    new_events.append(self.events[-1])
            elif not is_ot and self._last_amb_overtaking:
                self._last_amb_overtaking = False
                if getattr(amb, "overtaking_complete", False):
                    msg = "AMB-01 Overtaking complete — Established in safe passing lane"
                    if self.add_event(sim_time, EventCategory.TRAFFIC, "AMB_OVERTAKE_COMPLETE", msg):
                        new_events.append(self.events[-1])

            # Intersection Entry
            in_int = getattr(amb, "in_intersection", False)
            if in_int and not self._last_amb_in_intersection:
                self._last_amb_in_intersection = True
                msg = "AMB-01 ENTERED INTERSECTION (Crossing on protected priority green)"
                if self.add_event(sim_time, EventCategory.TRAFFIC, "AMB_ENTERED_INTERSECTION", msg):
                    new_events.append(self.events[-1])

            # Full Clearance
            cleared = getattr(amb, "cleared", False)
            if cleared and not self._last_amb_cleared:
                self._last_amb_cleared = True
                msg = "AMB-01 CLEARED INTERSECTION (Full body rear bumper exit)"
                if self.add_event(sim_time, EventCategory.EMERGENCY, "AMB_CLEARED_INTERSECTION", msg):
                    new_events.append(self.events[-1])

        return new_events

    def _format_phase_event(self, phase: CyclePhase) -> tuple[str, EventCategory]:
        """Map CyclePhase to clear narrative message and category."""
        if phase == CyclePhase.PREEMPTION_YELLOW:
            return "East/West YELLOW clearance initiated for preemption", EventCategory.SIGNAL
        elif phase == CyclePhase.PREEMPTION_ALL_RED:
            return "ALL-RED Clearance (Safety Gate Evaluation)", EventCategory.SAFETY
        elif phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
            return "SOUTH GREEN — EMERGENCY PRIORITY GRANTED", EventCategory.EMERGENCY
        elif phase == CyclePhase.EMERGENCY_TERMINATING:
            return "EMERGENCY PRIORITY TERMINATING (South Yellow)", EventCategory.SIGNAL
        elif phase == CyclePhase.RECOVERY_ALL_RED:
            return "POST-EMERGENCY ALL-RED CLEARANCE (1.0s buffer)", EventCategory.SAFETY
        elif phase == CyclePhase.NS_RED_EW_GREEN:
            return "NORMAL OPERATION RESTORED (East/West GREEN)", EventCategory.SIGNAL
        elif phase == CyclePhase.NS_RED_EW_YELLOW:
            return "East/West YELLOW (Normal Cycle)", EventCategory.SIGNAL
        elif phase == CyclePhase.NS_GREEN_EW_RED:
            return "North/South GREEN (Normal Cycle)", EventCategory.SIGNAL
        elif phase == CyclePhase.NS_YELLOW_EW_RED:
            return "North/South YELLOW (Normal Cycle)", EventCategory.SIGNAL
        return f"Signal transition: {phase.value}", EventCategory.SIGNAL
