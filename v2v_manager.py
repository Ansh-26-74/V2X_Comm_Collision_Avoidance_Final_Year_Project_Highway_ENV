"""Small simulated V2V channel used by the HighwayEnv demonstration."""
from dataclasses import dataclass
from math import hypot
import random


@dataclass
class V2VMessage:
    vehicle_id: str
    x: float
    y: float
    speed: float
    lane_id: int
    timestamp: float
    intended_lane: int
    maneuver: str
    trajectory: list[tuple[float, float]]


class V2VManager:
    def __init__(self, v2v_range=300.0, packet_loss_rate=0.0,
                 latency=0.0, broadcast_rate=10.0):
        self.v2v_range = v2v_range
        self.packet_loss_rate = packet_loss_rate
        self.latency = latency
        self.broadcast_period = 1.0 / broadcast_rate
        self.last_broadcast = -float("inf")
        self.pending = []
        self.latest_message = None
        self.random = random.Random(7)

    def broadcast(self, now, sender, receiver_position, lane_id,
                  intended_lane, maneuver, trajectory):
        """C broadcasts; packets are queued only if A is in range."""
        if now - self.last_broadcast < self.broadcast_period:
            return
        self.last_broadcast = now
        x, y = float(sender.position[0]), float(sender.position[1])
        rx, ry = receiver_position
        if hypot(x - rx, y - ry) > self.v2v_range:
            return
        if self.random.random() < self.packet_loss_rate:
            return
        message = V2VMessage("C", x, y, float(sender.speed), lane_id, now,
                             intended_lane, maneuver, trajectory)
        self.pending.append((now + self.latency, message))

    def deliver(self, now):
        """Deliver every due packet and keep the newest one at vehicle A."""
        due = [packet for packet in self.pending if packet[0] <= now]
        self.pending = [packet for packet in self.pending if packet[0] > now]
        if due:
            self.latest_message = due[-1][1]
        return self.latest_message
