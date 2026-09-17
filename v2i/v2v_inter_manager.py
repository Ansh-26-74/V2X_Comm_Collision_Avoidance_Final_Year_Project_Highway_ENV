"""Phase 13A: Intersection-Specific V2V Communication Manager.

Provides peer-to-peer V2V communication between AMB-01 and nearby civilian vehicles (e.g. C-01)
in the urban smart intersection environment.

Operates independently of the highway V2V scenario (v2v_manager.py).
"""

from dataclasses import dataclass
import math
import random


DEFAULT_V2V_INTER_RANGE = 200.0   # pixels (~70 metres)
DEFAULT_V2V_INTER_RATE  = 10.0    # Hz (10 packets / second, 100ms period)
DEFAULT_V2V_INTER_LATENCY = 0.020 # seconds (20ms direct DSRC delay)


@dataclass
class IntersectionV2VMessage:
    """Represents a standard V2V Cooperative Awareness Message (CAM) in the intersection."""
    sender_id: str          # e.g., "C-01" or "AMB-01"
    receiver_id: str        # e.g., "AMB-01" or "BROADCAST"
    vehicle_type: str       # "CIVILIAN" or "EMERGENCY"
    x: float                # World position X (pixels)
    y: float                # World position Y (pixels)
    vx: float               # Unit directional vector X
    vy: float               # Unit directional vector Y
    speed: float            # Current scalar speed (pixels/s)
    heading: float          # Heading angle in radians
    hazard_status: str      # "NORMAL", "DECELERATING", "STOPPED", "HAZARD"
    timestamp: float        # Simulation timestamp (seconds)


class IntersectionV2VManager:
    """Simulates inter-vehicle V2V communication for urban smart intersections."""

    def __init__(
        self,
        v2v_range: float = DEFAULT_V2V_INTER_RANGE,
        broadcast_rate: float = DEFAULT_V2V_INTER_RATE,
        latency: float = DEFAULT_V2V_INTER_LATENCY,
        packet_loss_rate: float = 0.0,
        random_seed: int = 42,
    ):
        self.v2v_range = float(v2v_range)
        self.broadcast_period = 1.0 / broadcast_rate if broadcast_rate > 0 else 0.1
        self.latency = float(latency)
        self.packet_loss_rate = float(packet_loss_rate)

        self._random = random.Random(random_seed)
        self._last_broadcast_times: dict[str, float] = {}
        self._pending: list[tuple[float, IntersectionV2VMessage]] = []
        self._latest_messages: dict[str, IntersectionV2VMessage] = {}

        # Telemetry & Diagnostics
        self.total_transmitted: int = 0
        self.total_delivered: int = 0
        self.total_dropped: int = 0
        self.last_status: str = "IDLE"

    def broadcast(
        self,
        now: float,
        msg: IntersectionV2VMessage,
        receiver_pos: tuple[float, float],
    ) -> tuple[bool, str]:
        """Broadcast a V2V message from sender toward a receiver position with range & loss checks.

        Returns (success, status_str).
        """
        # 1. Range verification
        dist = math.hypot(msg.x - receiver_pos[0], msg.y - receiver_pos[1])
        if dist > self.v2v_range:
            self.last_status = "OUT_OF_RANGE"
            return False, "OUT_OF_RANGE"

        # 2. Anti-spam / rate-limiting per sender
        last_tx = self._last_broadcast_times.get(msg.sender_id, -float("inf"))
        if now - last_tx < self.broadcast_period:
            self.last_status = "RATE_LIMITED"
            return False, "RATE_LIMITED"

        self._last_broadcast_times[msg.sender_id] = now
        self.total_transmitted += 1

        # 3. Packet loss simulation
        if self.packet_loss_rate > 0.0 and self._random.random() < self.packet_loss_rate:
            self.total_dropped += 1
            self.last_status = "DROPPED"
            return False, "DROPPED"

        # 4. Enqueue for latency delivery
        deliver_time = now + self.latency
        self._pending.append((deliver_time, msg))
        self.last_status = "TRANSMITTED"
        return True, "TRANSMITTED"

    def deliver(self, now: float, receiver_id: str = "AMB-01") -> list[IntersectionV2VMessage]:
        """Deliver all pending packets that have reached their latency delivery time."""
        delivered: list[IntersectionV2VMessage] = []
        remaining: list[tuple[float, IntersectionV2VMessage]] = []

        for del_time, msg in self._pending:
            if del_time <= now:
                if msg.receiver_id in ("BROADCAST", receiver_id):
                    delivered.append(msg)
                    self._latest_messages[receiver_id] = msg
                    self.total_delivered += 1
            else:
                remaining.append((del_time, msg))

        self._pending = remaining
        return delivered

    def get_latest_message(self, receiver_id: str = "AMB-01") -> IntersectionV2VMessage | None:
        """Return the newest delivered message for the given receiver."""
        return self._latest_messages.get(receiver_id)

    def reset(self) -> None:
        """Reset internal queues and statistics."""
        self._pending.clear()
        self._latest_messages.clear()
        self._last_broadcast_times.clear()
        self.total_transmitted = 0
        self.total_delivered = 0
        self.total_dropped = 0
        self.last_status = "IDLE"
