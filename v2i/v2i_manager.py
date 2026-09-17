"""Phase 4: V2IManager — Simulated Bidirectional V2I & I2V Wireless Channel.

Supports:
  - V2I (Vehicle -> Infrastructure): Ambulance emergency request delivery to RSU.
  - I2V (Infrastructure -> Vehicle): RSU signal status & hazard broadcast to vehicles.
  - Physical distance-based range limitation (default: 400 pixels).
  - Configurable simulated wireless propagation latency (e.g. 0.20s / 200ms).
  - Configurable packet loss simulation (e.g. 0.0 -> no loss, 0.1 -> 10% drop).
  - Rate limiting & anti-spam controls.
  - Packet animation metadata for visual communication rendering.

Completely separate from v2v_manager.py.
"""

from dataclasses import dataclass
import math
import random

V2I_BROADCAST_RATE = 5.0     # Hz — RSU broadcasts 5 times per second
V2I_RANGE          = 400.0   # pixels — wireless radio range (~140 metres equivalent)


@dataclass
class V2IPacketAnimation:
    """Represents an active in-flight wireless packet for visual rendering."""
    start_time: float
    deliver_time: float
    start_pos: tuple[float, float]
    end_pos: tuple[float, float]
    msg_type: str
    direction: str = "V2I"  # "V2I" (Veh -> RSU) or "I2V" (RSU -> Veh)
    delivered: bool = False
    dropped: bool = False
    progress: float = 0.0

    def calc_progress(self, now: float) -> float:
        """Return animation progress ratio between 0.0 (tx) and 1.0 (rx)."""
        duration = max(0.001, self.deliver_time - self.start_time)
        return min(1.0, max(0.0, (now - self.start_time) / duration))

    def current_pos(self, now: float) -> tuple[float, float]:
        """Interpolated world-space position of packet."""
        p = self.calc_progress(now)
        cx = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * p
        cy = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * p
        return (cx, cy)


class V2IManager:
    """Simulates bidirectional V2I (Vehicle->RSU) and I2V (RSU->Vehicle) communication."""

    def __init__(
        self,
        v2i_range: float = V2I_RANGE,
        broadcast_rate: float = V2I_BROADCAST_RATE,
        latency: float = 0.20,
        packet_loss_rate: float = 0.0,
    ):
        self.v2i_range = float(v2i_range)
        self.broadcast_period = 1.0 / broadcast_rate
        self.latency = float(latency)
        self.packet_loss_rate = float(packet_loss_rate)

        # Communication states: OUT_OF_RANGE, IN_RANGE, TRANSMITTED, DELIVERED, DROPPED
        self.comm_state: str = "OUT_OF_RANGE"
        self.last_status_text: str = "STANDBY"

        # I2V broadcast state (RSU -> Vehicles)
        self._last_broadcast = -float("inf")
        self._pending_i2v: list = []       # [(deliver_time, message_dict)]
        self._latest_i2v_message = None

        # V2I transmission state (Vehicles -> RSU)
        self._pending_rsu: list = []       # [(deliver_time, message_dict)]
        self._latest_rsu_message = None

        # Visualization packet queue
        self.active_animations: list[V2IPacketAnimation] = []

        # Diagnostics & Metrics
        self.total_transmitted: int = 0
        self.total_delivered: int = 0
        self.total_dropped: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # V2I: Vehicle -> RSU
    # ─────────────────────────────────────────────────────────────────────────

    def send_to_rsu(
        self,
        now: float,
        vehicle_pos: tuple[float, float],
        rsu_pos: tuple[float, float],
        message: dict,
    ) -> tuple[bool, str]:
        """Transmit a packet from a vehicle to the RSU with range, loss, and latency checks.

        Returns (success, status_str).
        """
        # 1. Physical wireless range verification
        dx = rsu_pos[0] - vehicle_pos[0]
        dy = rsu_pos[1] - vehicle_pos[1]
        dist = math.hypot(dx, dy)

        if dist > self.v2i_range:
            self.comm_state = "OUT_OF_RANGE"
            self.last_status_text = f"OUT OF RANGE ({dist:.0f}px > {self.v2i_range:.0f}px)"
            return False, "OUT_OF_RANGE"

        self.comm_state = "IN_RANGE"

        # 2. Simulated packet loss verification
        if self.packet_loss_rate > 0.0 and random.random() < self.packet_loss_rate:
            self.comm_state = "DROPPED"
            self.last_status_text = f"PACKET DROPPED (Loss {self.packet_loss_rate*100:.0f}%)"
            self.total_dropped += 1
            # Add dropped animation that vanishes halfway
            anim = V2IPacketAnimation(
                start_time=now,
                deliver_time=now + max(0.1, self.latency),
                start_pos=vehicle_pos,
                end_pos=rsu_pos,
                msg_type=message.get("type", "EMERGENCY_REQUEST"),
                direction="V2I",
                dropped=True,
            )
            self.active_animations.append(anim)
            return False, "DROPPED"

        # 3. Queue packet for delivery after latency
        deliver_time = now + self.latency
        self._pending_rsu.append((deliver_time, dict(message)))
        self.comm_state = "TRANSMITTED"
        self.last_status_text = f"TRANSMITTED ({message.get('type')})"
        self.total_transmitted += 1

        # Add active animation packet
        anim = V2IPacketAnimation(
            start_time=now,
            deliver_time=deliver_time,
            start_pos=vehicle_pos,
            end_pos=rsu_pos,
            msg_type=message.get("type", "EMERGENCY_REQUEST"),
            direction="V2I",
            delivered=False,
            dropped=False,
        )
        self.active_animations.append(anim)
        return True, "TRANSMITTED"

    def deliver_to_rsu(self, now: float = 0.0, sim_time: float | None = None) -> list[dict]:
        """Deliver all matured packets arriving at the RSU (latency elapsed)."""
        current_time = sim_time if sim_time is not None else now
        due = [pkt for pkt in self._pending_rsu if pkt[0] <= current_time]
        self._pending_rsu = [pkt for pkt in self._pending_rsu if pkt[0] > current_time]

        delivered_messages = []
        for pkt in due:
            msg = pkt[1]
            delivered_messages.append(msg)
            self._latest_rsu_message = msg
            self.total_delivered += 1
            self.comm_state = "DELIVERED"
            self.last_status_text = f"DELIVERED TO RSU ({msg.get('type')})"

        # Mark corresponding animations as delivered
        for anim in self.active_animations:
            if anim.direction == "V2I" and not anim.dropped and now >= anim.deliver_time:
                anim.delivered = True

        return delivered_messages

    # ─────────────────────────────────────────────────────────────────────────
    # I2V: RSU -> Vehicle (Backward-compatible Infrastructure Broadcast)
    # ─────────────────────────────────────────────────────────────────────────

    def broadcast(
        self,
        now: float,
        rsu_position: tuple[float, float],
        ego_position: tuple[float, float],
        message: dict,
    ) -> bool:
        """Queue an I2V infrastructure broadcast message if in range and period elapsed."""
        if now - self._last_broadcast < self.broadcast_period:
            return False
        self._last_broadcast = now

        dx = rsu_position[0] - ego_position[0]
        dy = rsu_position[1] - ego_position[1]
        dist = math.hypot(dx, dy)
        if dist > self.v2i_range:
            return False

        if self.packet_loss_rate > 0.0 and random.random() < self.packet_loss_rate:
            return False

        deliver_time = now + self.latency
        self._pending_i2v.append((deliver_time, dict(message)))
        return True

    def deliver(self, now: float) -> dict | None:
        """Deliver due I2V broadcast packets to vehicles."""
        due = [pkt for pkt in self._pending_i2v if pkt[0] <= now]
        self._pending_i2v = [pkt for pkt in self._pending_i2v if pkt[0] > now]
        if due:
            self._latest_i2v_message = due[-1][1]
        return self._latest_i2v_message

    # ─────────────────────────────────────────────────────────────────────────
    # Animation Maintenance & Reset
    # ─────────────────────────────────────────────────────────────────────────

    def update_animations(self, now: float) -> None:
        """Update animation progress and prune finished animations."""
        for anim in self.active_animations:
            anim.progress = anim.calc_progress(now)
        self.active_animations = [
            anim for anim in self.active_animations
            if now <= anim.deliver_time + 0.45
        ]

    @property
    def packets_sent(self) -> int:
        return self.total_transmitted

    @property
    def packets_delivered(self) -> int:
        return self.total_delivered

    @property
    def packets_dropped(self) -> int:
        return self.total_dropped

    @property
    def pending_packets(self) -> list:
        return self._pending_rsu

    @property
    def packet_animations(self) -> list[V2IPacketAnimation]:
        return self.active_animations

    def reset(self) -> None:
        """Clear all pending messages, animations, and counters."""
        self._last_broadcast = -float("inf")
        self._pending_i2v.clear()
        self._pending_rsu.clear()
        self._latest_i2v_message = None
        self._latest_rsu_message = None
        self.active_animations.clear()
        self.comm_state = "OUT_OF_RANGE"
        self.last_status_text = "STANDBY"
        self.total_transmitted = 0
        self.total_delivered = 0
        self.total_dropped = 0
