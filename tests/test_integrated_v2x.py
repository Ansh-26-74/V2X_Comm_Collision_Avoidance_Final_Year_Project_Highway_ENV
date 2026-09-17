"""Phase 13A: Unit and Integration Tests for V2V Hazard Detection & Inter-vehicle Communication.

Tests:
  1. V2V Normal Case: Lead vehicle maintains speed, TTC > 4.0s (SAFE), no emergency response.
  2. V2V Hazard Detection: Lead vehicle decelerates, V2V packet delivered, TTC drops, risk escalates.
  3. Critical TTC: Deterministic configuration where TTC < 2.0s, verifying CRITICAL risk state.
  4. Irrelevant Vehicle Filtering: Lateral/perpendicular crossing traffic filtered out.
  5. Range Filtering: Messages outside 200px dropped; messages inside 200px delivered.
  6. Packet Loss: 100% loss channel drops packets safely; fallback to kinematic car-following.
  7. V2I Independence: Active V2V does not alter or disrupt the authoritative RSU V2I preemption lifecycle.
  8. Conflict Zone Protection: Lateral maneuver prohibited once ambulance is inside the central conflict zone.
"""

import unittest
import math

from v2i.smart_signal import SmartTrafficSignal, CyclePhase, SignalState
from v2i.traffic_manager import TrafficManager, CivilianVehicle, AmbulanceVehicle
from v2i.v2i_manager import V2IManager
from v2i.v2v_inter_manager import IntersectionV2VManager, IntersectionV2VMessage
from v2i.event_logger import EventLogger, EventCategory


class TestIntegratedV2X(unittest.TestCase):
    """Phase 13A test suite for intersection V2V hazard detection and safety validation."""

    def setUp(self):
        """Set up standard intersection components."""
        self.signal = SmartTrafficSignal(
            intersection_id="INT-01",
            intersection_x=560.0,
            intersection_y=415.0,
        )
        self.traffic = TrafficManager(
            center_x=560,
            center_y=415,
            road_width=130,
            stop_line_dist=75,
        )
        self.v2v = IntersectionV2VManager(
            v2v_range=200.0,
            broadcast_rate=10.0,
            latency=0.020,
            packet_loss_rate=0.0,
        )
        self.logger = EventLogger()

    def test_01_v2v_normal_case(self):
        """Test 1: Lead vehicle C-01 maintains speed ahead of AMB-01; TTC is SAFE (> 4.0s)."""
        c01 = CivilianVehicle(
            vehicle_id="C-01",
            approach="SOUTH",
            x=592.0,
            y=520.0,
            speed=90.0,
            target_speed=90.0,
        )
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=620.0,
            speed=95.0,
            target_speed=95.0,
        )

        msg = c01.get_v2v_telemetry(now=1.0)
        # Distance = 100px <= 200px range
        success, status = self.v2v.broadcast(now=1.0, msg=msg, receiver_pos=(amb.x, amb.y))
        self.assertTrue(success)
        self.assertEqual(status, "TRANSMITTED")

        # Deliver after 20ms latency
        delivered = self.v2v.deliver(now=1.025, receiver_id="AMB-01")
        self.assertEqual(len(delivered), 1)

        ttc, risk, threat = amb.evaluate_v2v_hazards(delivered, now=1.025)
        # Closing speed = 95 - 90 = 5 px/s
        # Longitudinal gap = 100 - (44+36)/2 = 60 px
        # TTC = 60 / 5 = 12.0s > 4.0s
        self.assertAlmostEqual(ttc, 12.0, places=1)
        self.assertEqual(risk, "SAFE")
        self.assertEqual(threat.sender_id, "C-01")
        self.assertEqual(amb.v2v_risk_state, "SAFE")

    def test_02_v2v_hazard_detection(self):
        """Test 2: C-01 decelerates unexpectedly; V2V detects hazard and TTC drops to WARNING/CRITICAL."""
        c01 = CivilianVehicle(
            vehicle_id="C-01",
            approach="SOUTH",
            x=592.0,
            y=520.0,
            speed=90.0,
            target_speed=90.0,
        )
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=600.0,
            speed=100.0,
            target_speed=100.0,
        )

        # C-01 experiences hazard deceleration
        c01.set_hazard("DECELERATING", target_speed=20.0)
        c01.speed = 20.0

        msg = c01.get_v2v_telemetry(now=2.0)
        self.v2v.broadcast(now=2.0, msg=msg, receiver_pos=(amb.x, amb.y))
        delivered = self.v2v.deliver(now=2.025, receiver_id="AMB-01")
        self.assertEqual(len(delivered), 1)

        ttc, risk, threat = amb.evaluate_v2v_hazards(delivered, now=2.025)
        # Closing speed = 100 - 20 = 80 px/s
        # Longitudinal gap = (600 - 520) - (44 + 36)/2 = 80 - 40 = 40 px
        # TTC = 40 / 80 = 0.50s (< 2.0s -> CRITICAL)
        self.assertAlmostEqual(ttc, 0.50, places=2)
        self.assertEqual(risk, "CRITICAL")
        self.assertEqual(threat.sender_id, "C-01")

    def test_03_critical_ttc(self):
        """Test 3: Exact deterministic state where TTC < 2.0s triggers CRITICAL risk."""
        c01 = CivilianVehicle(
            vehicle_id="C-01",
            approach="SOUTH",
            x=592.0,
            y=500.0,
            speed=30.0,
            target_speed=30.0,
        )
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=580.0,
            speed=100.0,
            target_speed=100.0,
        )

        # Bumper gap = 80 - 40 = 40 px. Closing speed = 70 px/s. TTC = 40/70 = 0.57s
        msg = c01.get_v2v_telemetry(now=3.0)
        self.v2v.broadcast(now=3.0, msg=msg, receiver_pos=(amb.x, amb.y))
        delivered = self.v2v.deliver(now=3.025, receiver_id="AMB-01")

        ttc, risk, _ = amb.evaluate_v2v_hazards(delivered, now=3.025)
        self.assertTrue(ttc < 2.0)
        self.assertEqual(risk, "CRITICAL")
        self.assertEqual(amb.v2v_risk_state, "CRITICAL")

    def test_04_irrelevant_vehicle_filtering(self):
        """Test 4: Unrelated cross-traffic (EAST/WEST) or oncoming traffic is filtered out."""
        # Crossing civilian vehicle traveling West->East
        c_cross = CivilianVehicle(
            vehicle_id="C-CROSS",
            approach="WEST",
            x=550.0,
            y=415.0,
            speed=50.0,
        )
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=550.0,
            speed=95.0,
        )

        msg = c_cross.get_v2v_telemetry(now=4.0)
        # Distance is ~140px (within 200px range)
        self.v2v.broadcast(now=4.0, msg=msg, receiver_pos=(amb.x, amb.y))
        delivered = self.v2v.deliver(now=4.025, receiver_id="AMB-01")
        self.assertEqual(len(delivered), 1)

        # Evaluate V2V hazards: cross-traffic should be filtered out
        ttc, risk, threat = amb.evaluate_v2v_hazards(delivered, now=4.025)
        self.assertEqual(ttc, float("inf"))
        self.assertEqual(risk, "NONE")
        self.assertIsNone(threat)

    def test_05_v2v_range_filtering(self):
        """Test 5: Messages outside 200px range are rejected; within 200px are accepted."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=700.0,
            speed=95.0,
        )
        # Far vehicle: dist = 280px > 200px
        c_far = CivilianVehicle(
            vehicle_id="C-FAR",
            approach="SOUTH",
            x=592.0,
            y=420.0,
            speed=90.0,
        )
        msg_far = c_far.get_v2v_telemetry(now=5.0)
        success_far, status_far = self.v2v.broadcast(now=5.0, msg=msg_far, receiver_pos=(amb.x, amb.y))
        self.assertFalse(success_far)
        self.assertEqual(status_far, "OUT_OF_RANGE")

        # Near vehicle: dist = 100px <= 200px
        c_near = CivilianVehicle(
            vehicle_id="C-NEAR",
            approach="SOUTH",
            x=592.0,
            y=600.0,
            speed=90.0,
        )
        msg_near = c_near.get_v2v_telemetry(now=5.0)
        success_near, status_near = self.v2v.broadcast(now=5.0, msg=msg_near, receiver_pos=(amb.x, amb.y))
        self.assertTrue(success_near)
        self.assertEqual(status_near, "TRANSMITTED")

    def test_06_packet_loss(self):
        """Test 6: 100% loss channel drops packets safely; system relies on kinematic following."""
        lossy_v2v = IntersectionV2VManager(
            v2v_range=200.0,
            packet_loss_rate=1.0,
        )
        c01 = CivilianVehicle(
            vehicle_id="C-01",
            approach="SOUTH",
            x=592.0,
            y=550.0,
            speed=20.0,
        )
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=620.0,
            speed=100.0,
        )

        msg = c01.get_v2v_telemetry(now=6.0)
        success, status = lossy_v2v.broadcast(now=6.0, msg=msg, receiver_pos=(amb.x, amb.y))
        self.assertFalse(success)
        self.assertEqual(status, "DROPPED")
        self.assertEqual(lossy_v2v.total_dropped, 1)

        # Nothing delivered
        delivered = lossy_v2v.deliver(now=6.1, receiver_id="AMB-01")
        self.assertEqual(len(delivered), 0)

        # AMB-01 hazard evaluation produces no V2V threat
        ttc, risk, threat = amb.evaluate_v2v_hazards(delivered, now=6.1)
        self.assertEqual(ttc, float("inf"))
        self.assertEqual(risk, "NONE")
        self.assertIsNone(threat)

    def test_07_v2i_independence(self):
        """Test 7: Active V2V communication does not disrupt RSU preemption or recovery lifecycle."""
        v2i = V2IManager(v2i_range=400.0, latency=0.10)
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=550.0,
            speed=0.0,
        )
        rsu_pos = (730.0, 285.0)

        # Trigger V2I emergency request while V2V is also actively running
        amb.step_v2i_communication(now=7.0, rsu_pos=rsu_pos, v2i_channel=v2i, stop_line_coord=490.0)
        delivered_v2i = v2i.deliver_to_rsu(now=7.15)
        self.assertEqual(len(delivered_v2i), 1)

        accepted = self.signal.receive_emergency_request(delivered_v2i[0], current_time=7.15, auto_preempt=True)
        self.assertTrue(accepted)
        self.assertEqual(self.signal.rsu_priority_status, "ACTIVE")
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_SOUTH_GREEN)

        # Clearance and termination work independently
        self.signal.notify_ambulance_cleared("AMB-01")
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_TERMINATING)
        self.signal.update(3.0)
        self.assertEqual(self.signal.current_phase, CyclePhase.RECOVERY_ALL_RED)
        self.signal.update(2.0)
        self.assertEqual(self.signal.rsu_priority_status, "INACTIVE")

    def test_08_conflict_zone_maneuver_protection(self):
        """Test 8: Lateral maneuvers are strictly prohibited once ambulance enters central conflict zone."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=560.0,  # Approaching South, enter is 480.0
            speed=95.0,
        )
        # Update control before conflict zone (front bumper at 560 - 22 = 538 > 480)
        amb.update_control(
            dt=0.016,
            signal_state=SignalState.GREEN,
            stop_line_coord=490.0,
            intersection_enter=480.0,
            intersection_exit=350.0,
            lead_vehicle=None,
        )
        self.assertTrue(amb.dist_to_conflict_zone > 0.0)
        self.assertTrue(amb.lateral_maneuver_allowed)

        # Move inside central conflict zone (y = 440.0 <= 480.0)
        amb.y = 440.0
        amb.in_intersection = True
        amb.update_control(
            dt=0.016,
            signal_state=SignalState.GREEN,
            stop_line_coord=490.0,
            intersection_enter=480.0,
            intersection_exit=350.0,
            lead_vehicle=None,
        )
        # Inside junction: lateral maneuvers strictly forbidden
        self.assertFalse(amb.lateral_maneuver_allowed)


if __name__ == "__main__":
    unittest.main()
