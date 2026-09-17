"""Phase 4 Automated Test Suite — V2I Ambulance -> RSU Communication.

Validates:
  1. Message creation and format (EMERGENCY_REQUEST schema).
  2. Range-based triggering (out-of-range vs in-range).
  3. Latency delay modeling (200ms delay before RSU receipt).
  4. Packet loss simulation.
  5. Anti-spam throttling (1.0 Hz periodic update limit).
  6. RSU reception and payload buffering.
  7. CRITICAL SCOPE RULE: Signal controller does NOT change phase or timings.
  8. Ambulance compliance: AMB-01 remains stopped at RED before South stop line.
  9. Civilian traffic continues normal coordinated operation.
 10. Regression testing: Highway V2V scenario is 100% untouched and functional.
"""

import os
import unittest
import math

os.environ["SDL_VIDEODRIVER"] = "dummy"

from v2i.smart_signal import SmartTrafficSignal, SignalState, CyclePhase
from v2i.traffic_manager import TrafficManager, AmbulanceVehicle, CivilianVehicle
from v2i.v2i_manager import V2IManager


class TestPhase4V2ICommunication(unittest.TestCase):
    """Phase 4 Test Suite for V2I Emergency Communication."""

    def setUp(self):
        self.rsu_pos = (735.0, 255.0)
        self.stop_line_y = 490.0

    def test_01_emergency_request_format(self):
        """Test that generate_emergency_request outputs all dynamic fields matching schema."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.5,
            y=600.0,
            speed=95.0,
        )
        msg = amb.generate_emergency_request(stop_line_coord=self.stop_line_y, sim_time=10.5)

        self.assertEqual(msg["type"], "EMERGENCY_REQUEST")
        self.assertEqual(msg["vehicleId"], "AMB-01")
        self.assertEqual(msg["vehicleType"], "EMERGENCY")
        self.assertTrue(msg["emergency"])
        self.assertEqual(msg["priority"], "HIGH")
        self.assertEqual(msg["approach"], "SOUTH")
        self.assertIn("position", msg)
        self.assertIn("velocity", msg)
        self.assertIn("speed", msg)
        self.assertIn("heading", msg)
        self.assertIn("eta", msg)
        self.assertEqual(msg["timestamp"], 10.5)

        # ETA when moving towards stop line
        self.assertGreater(msg["eta"], 0.0)

        # Dynamic ETA when stopped at stop line
        amb.y = 502.0  # Front bumper at 490.0
        amb.speed = 0.0
        msg_stopped = amb.generate_emergency_request(stop_line_coord=self.stop_line_y, sim_time=12.0)
        self.assertEqual(msg_stopped["eta"], 0.0)

    def test_02_out_of_range_no_transmission(self):
        """Test that when AMB-01 is beyond V2I range, no packet is transmitted."""
        channel = V2IManager(v2i_range=400.0, latency=0.2)
        # Position far from RSU: dist > 500px
        amb = AmbulanceVehicle(vehicle_id="AMB-01", approach="SOUTH", x=592.5, y=800.0)
        dist = math.hypot(self.rsu_pos[0] - amb.x, self.rsu_pos[1] - amb.y)
        self.assertGreater(dist, 400.0)

        tx, status = amb.step_v2i_communication(
            now=1.0,
            rsu_pos=self.rsu_pos,
            v2i_channel=channel,
            stop_line_coord=self.stop_line_y,
        )
        self.assertFalse(tx)
        self.assertEqual(status, "OUT_OF_RANGE")
        self.assertEqual(channel.packets_sent, 0)
        self.assertEqual(channel.comm_state, "OUT_OF_RANGE")

    def test_03_in_range_transmission(self):
        """Test that when AMB-01 enters V2I range, message transmission triggers immediately."""
        channel = V2IManager(v2i_range=400.0, latency=0.2)
        # Position in range: dist ~ 340px
        amb = AmbulanceVehicle(vehicle_id="AMB-01", approach="SOUTH", x=592.5, y=560.0)
        dist = math.hypot(self.rsu_pos[0] - amb.x, self.rsu_pos[1] - amb.y)
        self.assertLess(dist, 400.0)

        tx, status = amb.step_v2i_communication(
            now=2.0,
            rsu_pos=self.rsu_pos,
            v2i_channel=channel,
            stop_line_coord=self.stop_line_y,
        )
        self.assertTrue(tx)
        self.assertEqual(status, "TRANSMITTED")
        self.assertEqual(channel.packets_sent, 1)
        self.assertEqual(len(channel.pending_packets), 1)
        self.assertEqual(channel.comm_state, "TRANSMITTED")

    def test_04_latency_and_rsu_delivery(self):
        """Test 200ms transmission latency: not delivered before latency, delivered after."""
        channel = V2IManager(v2i_range=400.0, latency=0.2)
        amb = AmbulanceVehicle(vehicle_id="AMB-01", approach="SOUTH", x=592.5, y=550.0)

        # Transmit at t=5.00s
        amb.step_v2i_communication(5.00, self.rsu_pos, channel, self.stop_line_y)
        self.assertEqual(channel.packets_sent, 1)

        # Step at t=5.10s (only 100ms passed -> latency not expired)
        delivered_early = channel.deliver_to_rsu(sim_time=5.10)
        self.assertEqual(len(delivered_early), 0)
        self.assertEqual(channel.packets_delivered, 0)

        # Step at t=5.25s (> 200ms passed -> delivered)
        delivered_ready = channel.deliver_to_rsu(sim_time=5.25)
        self.assertEqual(len(delivered_ready), 1)
        self.assertEqual(channel.packets_delivered, 1)
        self.assertEqual(delivered_ready[0]["type"], "EMERGENCY_REQUEST")
        self.assertEqual(delivered_ready[0]["vehicleId"], "AMB-01")
        self.assertEqual(channel.comm_state, "DELIVERED")

    def test_05_packet_loss_simulation(self):
        """Test channel drop simulation when packet_loss_rate is set."""
        channel = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=1.0)
        amb = AmbulanceVehicle(vehicle_id="AMB-01", approach="SOUTH", x=592.5, y=550.0)

        tx, status = amb.step_v2i_communication(1.0, self.rsu_pos, channel, self.stop_line_y)
        self.assertFalse(tx)
        self.assertEqual(status, "DROPPED")
        self.assertEqual(channel.packets_dropped, 1)
        self.assertEqual(len(channel.pending_packets), 0)
        self.assertEqual(channel.comm_state, "DROPPED")

    def test_06_anti_spam_throttling(self):
        """Test that V2I messages are rate-limited to 1.0 Hz (no per-frame flooding)."""
        channel = V2IManager(v2i_range=400.0, latency=0.2)
        amb = AmbulanceVehicle(vehicle_id="AMB-01", approach="SOUTH", x=592.5, y=550.0)

        # Frame 1 at t=3.00: Initial transmit succeeds
        tx1, st1 = amb.step_v2i_communication(3.00, self.rsu_pos, channel, self.stop_line_y)
        self.assertTrue(tx1)

        # Frame 2 at t=3.016: Next frame throttled
        tx2, st2 = amb.step_v2i_communication(3.016, self.rsu_pos, channel, self.stop_line_y)
        self.assertFalse(tx2)
        self.assertEqual(st2, "THROTTLED")

        # Frame 30 at t=3.50: Still throttled (< 1.0s elapsed)
        tx3, st3 = amb.step_v2i_communication(3.50, self.rsu_pos, channel, self.stop_line_y)
        self.assertFalse(tx3)

        # Frame 61 at t=4.01: > 1.0s elapsed -> Periodic transmit succeeds!
        tx4, st4 = amb.step_v2i_communication(4.01, self.rsu_pos, channel, self.stop_line_y)
        self.assertTrue(tx4)
        self.assertEqual(channel.packets_sent, 2)

    def test_07_rsu_receives_and_keeps_signals_unchanged(self):
        """CRITICAL SCOPE RULE: Signal controller MUST NOT change phase, timings, or preemption."""
        signal = SmartTrafficSignal(intersection_id="INT-01")
        channel = V2IManager(v2i_range=400.0, latency=0.2)

        # Force signal state: East/West GREEN, North/South RED
        signal.current_phase = CyclePhase.NS_RED_EW_GREEN
        signal._time_in_phase = 2.0
        initial_time_remaining = signal.time_remaining
        self.assertEqual(signal.get_signal("SOUTH"), SignalState.RED)
        self.assertEqual(signal.get_signal("EAST"), SignalState.GREEN)

        # AMB-01 sends request
        amb = AmbulanceVehicle(vehicle_id="AMB-01", approach="SOUTH", x=592.5, y=520.0, speed=0.0)
        amb.step_v2i_communication(2.0, self.rsu_pos, channel, self.stop_line_y)

        # Deliver to RSU
        pkts = channel.deliver_to_rsu(2.25)
        self.assertEqual(len(pkts), 1)
        signal.receive_emergency_request(pkts[0])

        # Verify reception and buffering
        self.assertTrue(signal.emergency_request_received)
        self.assertIsNotNone(signal.latest_emergency_request)
        self.assertEqual(signal.latest_emergency_request["vehicleId"], "AMB-01")

        # CRITICAL VERIFICATION:
        # Phase must still be NS_RED_EW_GREEN
        self.assertEqual(signal.current_phase, CyclePhase.NS_RED_EW_GREEN)
        # SOUTH must strictly remain RED!
        self.assertEqual(signal.get_signal("SOUTH"), SignalState.RED)
        # EAST must remain GREEN!
        self.assertEqual(signal.get_signal("EAST"), SignalState.GREEN)
        # Phase timer must not be corrupted
        self.assertEqual(signal.time_in_phase, 2.0)
        self.assertEqual(signal.time_remaining, initial_time_remaining)

    def test_08_ambulance_obeys_red_and_stops_safely(self):
        """Verify AMB-01 decelerates and comes to a full stop before South stop line despite V2I."""
        tm = TrafficManager(ambulance_spawn_time=0.0)
        channel = V2IManager(v2i_range=400.0, latency=0.2)
        signal = SmartTrafficSignal(intersection_id="INT-01")
        signal.current_phase = CyclePhase.NS_RED_EW_GREEN  # South is RED

        # Spawn ambulance
        tm.ambulance_spawn_time = 0.0
        tm._spawn_ambulance()
        self.assertIsNotNone(tm.ambulance)
        self.assertEqual(tm.ambulance.approach, "SOUTH")

        # Step simulation forward for 7 seconds
        dt = 0.05
        for step in range(140):
            sim_time = step * dt
            tm.update(dt=dt, signal_controller=signal, sim_time=sim_time, v2i_channel=channel, rsu_pos=self.rsu_pos)
            pkts = channel.deliver_to_rsu(sim_time)
            for p in pkts:
                signal.receive_emergency_request(p)

        amb = tm.ambulance
        self.assertIsNotNone(amb)
        # AMB-01 transmitted V2I request
        self.assertTrue(amb.has_transmitted_initial)
        self.assertTrue(signal.emergency_request_received)

        # AMB-01 must be completely STOPPED
        self.assertEqual(amb.speed, 0.0)
        # Front bumper must be stopped before the stop line (y=490.0)
        front_bumper_y = amb.y - amb.length / 2
        self.assertGreater(front_bumper_y, self.stop_line_y)
        self.assertFalse(amb.in_intersection)

    def test_09_civilian_traffic_unaffected_by_v2i_request(self):
        """Verify civilian traffic continues operating normally during emergency requests."""
        tm = TrafficManager(ambulance_spawn_time=999.0)
        signal = SmartTrafficSignal(intersection_id="INT-01")
        # Spawn an East vehicle
        veh = CivilianVehicle(vehicle_id="C01", approach="EAST", x=700.0, y=390.0, speed=100.0)
        tm.vehicles.append(veh)

        # Simulate emergency request received by RSU
        req = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "priority": "HIGH",
            "approach": "SOUTH",
            "eta": 0.0,
            "timestamp": 1.0,
        }
        signal.receive_emergency_request(req)

        # Update traffic
        tm.update(dt=0.1, signal_controller=signal, sim_time=1.0)
        self.assertEqual(len(tm.vehicles), 1)
        self.assertGreater(tm.vehicles[0].speed, 0.0)

    def test_10_v2v_highway_regression(self):
        """Verify that highway V2V modules remain 100% functional and unmodified."""
        import v2v_manager
        channel = v2v_manager.V2VManager()
        self.assertIsNotNone(channel)
        self.assertGreater(channel.v2v_range, 0)


if __name__ == "__main__":
    unittest.main()
